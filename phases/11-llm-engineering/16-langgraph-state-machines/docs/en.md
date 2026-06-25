# LangGraph — 智能体（agent）的状态机（state machine）

> 手写的 ReAct 循环（ReAct loop）就是一个 `while True`。用 LangGraph 写的 ReAct 循环是一张可以执行检查点（checkpoint）、中断（interrupt）、分支（branch）和时间旅行（time-travel）的图。智能体本身没变，包裹它的执行框架（harness）变了。

**类型：** 构建
**语言：** Python
**先修要求：** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**时长：** 约 75 分钟

## 问题所在

你交付了一个函数调用智能体（function-calling agent）。它在前三轮对话里表现正常，然后出问题：模型（model）调用了一个返回 500 的工具（tool），用户在任务中途改变主意，或者智能体在没有人工批准的情况下决定退款。`while True:` 循环没有任何钩子。你无法暂停它，无法回退它，也无法分叉到“如果模型选择了另一个工具会怎样”。一旦你把这东西从演示环境搬到生产环境，智能体就成了一个要么成功要么失败的黑盒。

一旦看清这一点，下一步显而易见。智能体本身就是一个状态机（state machine）——系统提示（system prompt）加上消息历史（message history）加上待处理的工具调用（pending tool calls）再加上下一步动作。把状态机显式化：用节点（nodes）表示“模型思考”“工具运行”“人工审批”，用边（edges）表示它们之间的条件转移。一旦图被显式化，执行框架（harness）就能免费获得四项能力：检查点（checkpointing，在步骤之间保存状态）、中断（interrupts，暂停等待人工）、流式传输（streaming，流式输出 token 和中间事件）以及时间旅行（time-travel，回退到之前状态并尝试不同分支）。

LangGraph 就是提供这种抽象的库。它不像 LangChain 意义上的智能体框架（“给你一个 AgentExecutor，祝你好运”）。它是一个图运行时（graph runtime），把状态、持久化和中断都作为一等公民（first-class）。智能体循环是你画出来的，而不是手写出来的。

## 核心概念

![LangGraph StateGraph：节点、边和检查点保存器](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 包含三样东西。

1. **状态（State）。** 一个类型化的字典（TypedDict 或 Pydantic model），在整张图中流动。每个节点接收完整状态并返回部分更新，LangGraph 通过每个字段的*归约器（reducer）*来合并这些更新——列表累积用 `operator.add`，默认则是覆盖。
2. **节点（Nodes）。** Python 函数 `state -> partial_state`。每个节点都是一个离散步骤：“调用模型”“运行工具”“总结”。
3. **边（Edges）。** 节点之间的转移。静态边（static edges）指向固定位置。条件边（conditional edges）接收一个路由函数（router function）`state -> next_node_name`，让图能够根据模型输出分支。

然后编译（compile）这张图。编译会绑定拓扑结构、附加检查点保存器（checkpointer，可选但对生产环境至关重要），并返回一个可运行对象（runnable）。你用初始状态（initial state）和 `thread_id` 调用它。执行的每一步都会持久化一个以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四项超能力

**检查点（Checkpointing）。** 每次节点转移都会把新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。用相同的 `thread_id` 再次调用图即可恢复。图会从暂停处继续执行。

**中断（Interrupts）。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行会在该节点运行前停止。状态会被持久化。你的 API 可以向用户返回“等待审批”。稍后向同一个 `thread_id` 发送 `Command(resume=...)` 即可恢复执行。

**流式传输（Streaming）。** `graph.stream(state, mode="updates")` 会实时产生状态增量。`mode="messages"` 会在模型节点内部流式输出 LLM token。`mode="values"` 会产出完整快照。你可以按需选择要展示给 UI 的内容。

**时间旅行（Time-travel）。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。把任意历史 `checkpoint_id` 传给 `graph.invoke`，你就能从该点分叉。这对调试（“如果模型选择了工具 B 会怎样？”）以及重放生产轨迹的回归测试非常有用。

### 归约器（reducer）才是关键

每个状态字段都有一个归约器。大部分默认值就能满足需求——新值覆盖旧值。但消息列表需要 `operator.add` 或 `add_messages`，这样新消息才会追加而不是替换。并行边（parallel edges）的更新也通过归约器合并。如果两个节点都更新了 `messages`，而你忘了加上 `Annotated[list, add_messages]`，第二个节点会静默获胜，你会丢失半个回合。归约器是库中唯一微妙的概念；把它搞对，其余部分就能自然组合。

### 用四个节点实现 ReAct 图

一个生产级的 ReAct 智能体只需要四个节点和两条边：

1. `agent` —— 用当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools` —— 执行上一条助手消息中的 tool_calls，并把工具结果追加为 tool messages。
3. 一条从 `agent` 出发的条件边：如果最后一条消息包含 tool_calls，则路由到 `tools`，否则路由到 `END`。
4. 一条从 `tools` 回到 `agent` 的静态边。

就这些。你能在约 40 行代码内获得完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → ……），并附带检查点、中断和流式传输能力。

### StateGraph 与 Send（扇出）

`Send(node_name, state)` 允许一个节点派发并行的子图。例如：智能体决定同时查询三个检索器。每个 `Send` 都会生成目标节点的一次并行执行；它们的输出通过状态归约器合并。这就是 LangGraph 表达编排器-工作者模式（orchestrator-workers pattern）的方式，无需使用线程原语。

### 子图（Subgraphs）

一张已编译的图可以成为另一张图的节点。外层图只看到一个节点；内层图有自己的状态和检查点。团队正是用这种方式构建主管-工作者智能体（supervisor-worker agents）：主管图把用户意图路由到各个领域的工作者子图。

## 动手实现

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让消息列表累积而不是被覆盖的归约器。忘记它是 LangGraph 最常见的 bug。

### 步骤 2：用 thread 运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以把这些实时推送到 UI，让用户看到“智能体正在思考……正在调用 search_web……已获得结果……正在回答”。

### 步骤 3：添加人机协同（human-in-the-loop）中断

标记一个节点，使其在运行前暂停执行。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每次工具调用前暂停
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] 已被设置。检查建议的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写入拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和会话线程在中断期间都会持久化。除了执行期间，没有任何东西只存在于内存中。

### 步骤 4：用时间旅行调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从之前的检查点分叉
target = history[3].config  # 回退三步
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从该点向前重放
```

把 `None` 作为输入会重放给定检查点；传入一个值则会先把它作为该检查点状态的更新追加，然后再恢复执行。这样你就可以复现一次失败的智能体运行，而无需重跑整段对话。

### 步骤 5：为生产环境更换检查点保存器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已内置。`MemorySaver` 只用于测试。任何需要跨重启持久化的场景都应该使用真正的存储。

## 技能

> 你把智能体构建成图，而不是 `while True` 循环。

在使用 LangGraph 之前，先花 60 秒做设计：

1. **命名节点。** 每一个离散的决策或产生副作用的动作都是一个节点。“智能体思考”“工具运行”“审核员批准”“响应流式输出”。如果你列不出来，说明这个任务还不是智能体形状。
2. **声明状态。** 用最小的 TypedDict，为每个列表字段配置归约器。不要把所有东西都塞进 `messages`；把任务相关的字段（一个工作中的 `plan`、`budget` 计数器、`retrieved_docs` 列表）提升到顶层。
3. **画出边。** 除非下一步依赖模型输出，否则使用静态边。每条条件边都需要一个带命名分支的路由函数。
4. **一开始就选好检查点保存器。** 测试用 `MemorySaver`，其他情况用 Postgres/Redis/SQLite。不要在没有检查点保存器的情况下交付——没有它就没有恢复、没有中断、没有时间旅行。
5. **在工具运行前决定中断，而不是之后。** 审批放在进入产生副作用节点的边上，这样可以在造成伤害前取消；验证放在模型输出后的边上，这样可以低成本地拒绝错误调用。
6. **默认启用流式传输。** UI 用 `mode="updates"`，模型节点内部 token 级流式用 `mode="messages"`，评估时用 `mode="values"` 获取完整快照。

拒绝交付没有检查点保存器的 LangGraph 智能体。拒绝在副作用发生之后才中断。拒绝让 `messages` 字段不使用 `add_messages` 作为归约器。

## 练习

1. **简单。** 用上面的四节点 ReAct 图实现一个带计算器工具和网页搜索工具的智能体。验证 `list(app.get_state_history(config))` 在两轮对话后至少返回四个检查点。
2. **中等。** 添加一个 `planner` 节点，在 `agent` 之前运行，并向状态写入结构化 `plan: list[str]`。让 `agent` 标记计划步骤已完成。如果 `plan` 在检查点恢复后丢失（归约器错误），则测试失败。
3. **困难。** 构建一张主管图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图都有自己的状态和检查点保存器。在外层图上添加 `interrupt_before=["writer"]`，让人工可以批准研究简报。确认从之前的检查点进行时间旅行时，只重新运行被分叉的分支。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| StateGraph（状态图） | "LangGraph 的图" | 编译前用于添加节点和边的构建器对象。 |
| Reducer（归约器） | "字段如何合并" | 一个函数 `(old, new) -> merged`，在节点返回该字段的更新时应用；默认覆盖，`add_messages` 追加。 |
| Thread（会话线程） | "一个对话 ID" | `thread_id` 字符串，用于限定一次会话的所有检查点。 |
| Checkpoint（检查点） | "一个暂停状态" | 节点转移后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| Interrupt（中断） | "暂停等待人工" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| Time-travel（时间旅行） | "从之前的步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send（发送） | "并行子图分发" | 一个节点可以返回的构造器，用于生成目标节点的 N 次并行执行。 |
| Subgraph（子图） | "作为节点的已编译图" | 一张已编译的 StateGraph 作为另一张图的节点使用；保留自己的状态作用域。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) —— StateGraph、归约器、检查点保存器和中断的权威参考。
- [LangGraph 概念：状态、归约器、检查点保存器](https://langchain-ai.github.io/langgraph/concepts/low_level/) —— 本节课使用的心智模型，来自官方源头。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) —— Postgres/SQLite/Redis 存储、检查点命名空间和 thread ID 的细节。
- [LangGraph 人机协同](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) —— `interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及编辑状态模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个 LangGraph 智能体都在实现的模式；阅读它以理解推理轨迹的底层依据。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 不同图结构（链、路由器、编排器-工作者、评估器-优化器）的适用场景。
- Phase 11 · 09 (Function Calling) —— 每个 LangGraph 智能体节点都会复用的工具调用原语。
- Phase 11 · 14 (Model Context Protocol) —— 通过 MCP 适配器接入 LangGraph `ToolNode` 的外部工具发现机制。
- Phase 11 · 17 (Agent framework tradeoffs) —— 何时选择 LangGraph 而非 CrewAI、AutoGen 或 Agno。
