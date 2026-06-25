# 智能体框架权衡 —— LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都展示同一个 demo（研究智能体生成一份报告），却隐藏同一个 bug（状态模式与编排层互相打架）。选择抽象与你的问题形状相匹配的框架；其他一切不过是你要写两遍的胶水代码。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)  
**Time:** ~45 分钟

## 问题所在

你有一个任务需要多次调用大语言模型（LLM）。可能是研究工作流（规划、搜索、总结、引用），可能是代码审查流水线（解析 diff、评审、打补丁、验证），也可能是一个多轮助手，它能订机票、写邮件、报销费用。于是你选了一个框架。

三天后，你发现框架的抽象开始漏水。CrewAI 给你角色，但当“研究员”需要把一份结构化计划交给“写手”时，它会和你对着干。AutoGen 给你智能体之间的聊天，却没有一等状态，因此你的检查点只能是对话日志的 pickle 文件。LangGraph 给你状态图，却迫使你提前命名每一次转换，而那时你还不清楚智能体会做什么。Agno 给你一个单智能体原语，当你想扇出到三个并发 worker 时，它会尖叫。

解决方案不是“选最好的框架”，而是把框架的核心抽象与你问题的形状匹配起来。本节课会画出这张地图。

## 核心概念

![智能体框架矩阵：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

2026 年的格局由四个框架主导。它们的核心抽象并不相同。

| Framework | 核心抽象 | 最适合 | 最不适合 |
|-----------|----------|--------|----------|
| **LangGraph** | `StateGraph` —— 类型化状态、节点、条件边、检查点器（checkpointer）。 | 具有显式状态和人机协作中断的工作流；需要时光倒流调试的生产级智能体。 | 拓扑未知的松散角色驱动式头脑风暴。 |
| **CrewAI** | `Crew` —— 角色（目标、背景故事）、任务、流程（顺序或层级）。 | 角色扮演或人格驱动、计划短且线性/层级的工作流。 | 超出 crew 轮次历史的有状态需求；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 —— 两个或多个智能体轮流发言，直到满足退出条件。 | 多智能体*对话*（师生、提议者-批评者、执行者-审核者），思考从聊天中涌现。 | 已知 DAG 的确定性工作流；需要跨重启持久化状态的场景。 |
| **Agno** | `Agent` —— 单个 LLM + 工具 + 记忆，可组合成团队。 | 快速构建单智能体和轻量级团队；强大的多模态和内置存储驱动。 | 需要自定义 reducer 的深度显式分支图。 |

### “抽象”到底指什么

框架的核心抽象，就是你在白板上画架构图时画的那东西。

- **LangGraph** → 你画一张图。节点是步骤，边是转换，每一点的状态对象都是类型化的。心智模型是状态机。
- **CrewAI** → 你画组织架构图。每个角色都有岗位说明，由经理分配任务。心智模型是一个小型专家团队。
- **AutoGen** → 你画 Slack 私信。两个智能体互相发消息；需要 moderator 时第三人加入。心智模型是聊天。
- **Agno** → 你画一个挂着工具的方框。把方框并排放就是团队。心智模型是“开箱即用的智能体”。

### 状态问题

生产环境中，状态是大多数框架选型崩溃的地方。

- **LangGraph.** 类型化状态（`TypedDict` 或 Pydantic 模型）、按字段 reducer、一等检查点器（SQLite/Postgres/Redis）。恢复、中断、时光倒流都是原生能力。*（参见 Phase 11 · 16。）*
- **CrewAI.** 状态以字符串形式通过 `context` 字段在任务间流动，或通过 `output_pydantic` 结构化。默认没有持久的 per-crew 存储；如果 crew 必须跨重启存活，得自己加装。
- **AutoGen.** 状态是聊天记录以及任何用户自定义的 `context`。对话记录可以持久化；任意工作流状态除非你自己写适配器，否则不会。
- **Agno.** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB），通过 `storage=` 挂到 `Agent` 上 —— 对话会话和用户记忆会自动持久化。它不是完整的图检查点器，而是会话存储。

### 分支问题

每个非平凡智能体都会分支。由谁决定分支至关重要。

- **LangGraph** —— 由你决定，通过条件边。路由是一个带命名分支的 Python 函数。分支在编译后的图中是一等公民；检查点器会记录走了哪条分支。
- **CrewAI** —— 在层级模式下由经理决定；在顺序模式下由你在构建时决定。路由隐含在任务列表里；除了经理的 prompt 之外，没有一等“if”。
- **AutoGen** —— 由智能体通过聊天决定。分支来自“下一个谁发言”。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认由 LLM 驱动。
- **Agno** —— 由智能体通过接下来调用哪个工具决定。团队有 coordinator/router/collaborator 模式；除此之外的分支由开发者负责。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任意 OpenTelemetry 导出器。每个节点转换都是一个 trace span；检查点器本身也是可回放 trace。LangSmith 是一方方案；Langfuse/Phoenix 也有适配器。
- **CrewAI** —— 自 2025 年末起原生支持 OpenTelemetry；集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen** —— 通过 `autogen-core` 提供 OpenTelemetry 集成；AgentOps 和 Opik 有连接器。Trace 粒度是 per-agent-message，而不是 per-node。
- **Agno** —— 内置 `monitoring=True` 开关，加 OpenTelemetry 导出器；与 Langfuse 深度集成以做会话 trace。

### 成本与延迟

四个框架都会带来每次调用的开销（框架逻辑、校验、序列化）。大致开销从低到高：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架额外做了多少 LLM 路由。CrewAI 的层级经理要花 token 决定下一个谁执行；AutoGen 的 `GroupChatManager` 同样如此。LangGraph 只在你写 `llm.invoke` 的地方花 token。Agno 的单智能体路径很薄。

当每次运行的成本很重要时，优先选择显式路由（LangGraph 边、AutoGen `speaker_selection_method`），而不是 LLM 选择路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一等 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具、MCP 工具都能接入。Crew 之间通过 `allow_delegation=True` 委托。
- **AutoGen** → `FunctionTool` 包装任意 Python 可调用对象；MCP 适配器可用。智能体间模式与 AG2 生态紧耦合。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可在智能体和团队间共享。

## 技能目标

> 你能用一句话解释，为什么某个框架适合某个智能体问题。

构建前检查清单：

1. **画出形状。** 这是图（类型化状态、命名转换）？角色扮演（专家交接）？聊天（智能体聊到结束）？还是带工具的单智能体？
2. **决定由谁分支。** 开发者决定分支 → LangGraph。经理智能体决定 → CrewAI 层级。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 是否需要从检查点恢复？时光倒流？运行中途人机中断？如果需要，LangGraph 是默认选择；Agno 会话可覆盖对话范围的状态。
4. **检查成本预算。** LLM 选择路由每次轮次都要额外花 token。如果智能体每天运行数千次，优先选择显式路由。
5. **评估框架开销。** 每个框架都是额外依赖。如果任务只是两次 LLM 调用加一个工具，写 30 行纯 Python 即可；没有框架比没有框架更便宜。

在你能画出图、组织架构图、聊天或智能体方框之前，不要伸手去拿框架。不要选一个为了你要做的事而被迫与其状态模型打架的框架。

## 决策矩阵

| 问题形状 | 首选框架 | 原因 |
|----------|----------|------|
| 带类型化状态、人工审批、长时间运行的工作流 DAG | LangGraph | 一等状态、检查点器、中断、时光倒流。 |
| 有明确角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | 按角色分配任务在 CrewAI 中表达便宜；分支复杂时升级到 LangGraph。 |
| 提议者-批评者或师生对话 | AutoGen | 双智能体聊天是它的原生形状。 |
| 带工具、会话、记忆的单智能体 | Agno | 最薄配置，内置存储和记忆。 |
| 数千个并行扇出且带 reducer | LangGraph + `Send` | 唯一拥有一等并行分发原语的框架。 |
| 快速原型，不想绑定框架 | 纯 Python + 提供商 SDK | 没有框架就是最快的框架。 |

## 练习

1. **简单。** 用同一个任务 —— “调研 Anthropic 总部，写一份 200 字简报并引用来源” —— 分别在 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）中实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 在 AutoGen（researcher ↔ writer 聊天，editor 通过 `GroupChat` 加入）和 Agno（一个带 `search_tools` 和 `write_tools` 的单智能体，加会话存储）中实现同一任务。从 (a) 每次运行成本、(b) 崩溃后恢复能力、(c) 在 write 步骤前注入人工审批的能力，对四种实现排序。
3. **困难。** 写一个决策树脚本 `pick_framework.py`，接收一段简短问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回推荐和一句话理由。在你自己设计的六个案例上验证它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Orchestration（编排） | “智能体如何协调” | 决定下一个运行哪个节点/角色/智能体的层。 |
| Durable state（持久状态） | “重启后恢复” | 绑定到检查点或会话存储、能跨进程死亡存活的状态。 |
| LLM-selected routing（LLM 选择路由） | “让模型决定” | 规划器 LLM 每轮选择下一步；灵活，但每次都花 token。 |
| Explicit routing（显式路由） | “开发者决定” | Python 函数或静态边选择下一步；便宜且可审计。 |
| Crew | “一个 CrewAI 团队” | 角色 + 任务 + 流程（顺序或层级）绑定成的单一可运行对象。 |
| GroupChat | “AutoGen 的多智能体聊天” | 由 N 个智能体参与、带发言选择器的受控对话。 |
| Team (Agno) | “多智能体 Agno” | 在一组智能体上的 route / coordinate / collaborate 模式。 |
| StateGraph | “LangGraph 的图” | 类型化状态、节点、条件边、检查点器原语。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) —— StateGraph、checkpointers、interrupts、time-travel。
- [CrewAI documentation](https://docs.crewai.com/) —— Crews、Flows、Agents、Tasks、Processes。
- [AutoGen documentation](https://microsoft.github.io/autogen/) —— ConversableAgent、GroupChat、teams、tools。
- [Agno documentation](https://docs.agno.com/) —— Agent、Team、Workflow、storage、memory。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 与框架无关的模式库（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个框架都在包装的基础原语。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) —— AutoGen 设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) —— CrewAI 式人格栈所基于的角色扮演基础。
- Phase 11 · 16 (LangGraph) —— 本节课用作基准的框架。
- Phase 11 · 19 (Reflexion) —— 一个映射到 LangGraph 很自然、映射到 CrewAI 很别扭的模式。
- Phase 11 · 22 (Production observability) —— 如何为所选框架做 instrumentation。
