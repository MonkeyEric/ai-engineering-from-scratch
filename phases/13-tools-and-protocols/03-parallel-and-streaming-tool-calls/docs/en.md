# 并行工具调用与流式工具调用

> 三次相互独立的天气查询如果串行执行，就要走三轮往返。并行运行则总时间坍缩到最慢那一次调用。如今所有前沿模型提供商都能在一个回合内发出多个工具调用。收益真实存在，但底层 plumbing 却很微妙。本节课同时讲解两个部分：并行扇出（parallel fan-out）与流式参数重组，并重点剖析 id 关联陷阱。

**Type:** 动手实践
**Languages:** Python（标准库、线程池 + 流式封装）
**Prerequisites:** Phase 13 · 02（function calling 深入探讨）
**Time:** ~75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 存在的意义以及何时应关闭它。
- 在并行扇出时，将流式参数块正确关联到对应的工具调用 id。
- 将部分 `arguments` 字符串重组为完整 JSON，而不是过早解析。
- 运行一个三城市天气基准，演示串行与并行之间的延迟差异。

## 问题背景

如果没有并行调用，智能体在回答“班加罗尔、东京和苏黎世的天气如何”时，会执行以下流程：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，每一次还要加上执行器（executor）延迟。墙上时钟时间大约是理想情况的 4 倍。

启用并行调用后：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次 LLM 往返。执行器时间取三者最大值，而非总和。在 OpenAI、Anthropic 和 Gemini 上的生产基准测试显示，扇出工作负载的墙上时钟时间可减少 60% 到 70%。

代价是关联复杂度。当三次调用以不同顺序完成时，结果必须携带匹配的 `tool_call_id`，模型才能对齐它们。如果结果以流式到达，你还必须将部分参数片段组装成完整 JSON 后才能执行。Gemini 3 引入唯一 id，部分原因就是为了解决现实问题：两次并行调用同一个工具时难以区分。

## 核心概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 可强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 启用并行（Claude 3.5 及以上默认开启）。设为 `true` 则串行。
- **Gemini。** 始终支持并行；设置 `tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

当工具之间存在顺序依赖（如先 `create_file` 再 `write_file`）、一次调用的输出是下一次调用的输入、或速率限制器无法承受扇出时，应关闭并行。

### Id 关联

模型发出的每一次调用都有一个 `id`。宿主返回的每一次结果也必须包含相同的 id。否则结果就会产生歧义。

- **OpenAI。** 每条 tool 角色的消息都带 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块都带 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 都带 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，导致同名并行调用会出错）。

### 并发执行调用

宿主将每次调用的执行器放在独立线程、协程或远程 worker 上运行。最简单的封装使用线程池；生产环境则使用 asyncio 配合 `asyncio.gather` 或结构化并发。完成顺序不可预测——id 才是真正的标识符。

一个常见 bug：按调用列表顺序而非完成顺序返回结果。由于模型只关心 `tool_call_id`，这通常也能工作；但如果结果被丢弃或重复，乱序提交会让调试更困难。建议按完成顺序返回结果，并显式带上 id。

### 流式工具调用

当模型以流式输出时，`arguments` 会分段到达。三次并行调用的三股独立流在传输中交错。你需要为每个 id 准备一个累加器（accumulator）。

各提供商的数据形态：

- **OpenAI。** 每个数据块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。数据块带 `index`（调用列表中的位置）。你按 index 累加，在 id 首次出现时读取它，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件依次为 `message_start`，然后每个块触发一次 `content_block_start`，类型为 `tool_use`（包含 id、name 和空 input）。`content_block_delta` 事件携带 `input_json_delta` 片段。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）发出的数据块带 `functionCallId`，因此调用可以干净地交错。Gemini 3 之前，流式返回一次只给出一个完整调用。

### 部分 JSON 与过早解析陷阱

在 `arguments` 完整之前不能解析它。部分 JSON 如 `{"city": "Beng` 无效，会抛出异常。正确的门槛是各提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`、或 Gemini 的流结束事件。只有到这时才尝试 `json.loads`。更稳健的做法是使用增量 JSON 解析器，在结构完成时产出事件；OpenAI 的流式指南推荐这种方式，用于展示实时“思考中”指示器的用户体验。花括号计数不可靠（字符串或转义内容中的花括号会导致误报），只能当作非正式的调试启发式。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

宿主回复仍然必须引用这些 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

对 OpenAI 或 Anthropic 而言，回复中的顺序不影响正确性。Gemini 也接受任意顺序，只要 id 匹配即可。

### 基准测试：串行 vs 并行

`code/main.py` 中的封装模拟了三个延迟分别为 400 ms、600 ms 和 800 ms 的执行器。串行执行总共需要 1800 ms。并行执行只需要 max(400, 600, 800) = 800 ms。这个差值是恒定的，而非成比例的，因此随着工具数量增加，节省时间也会增加。

现实世界注意事项：并行调用会给下游 API 带来压力。对速率受限的服务进行 10 路扇出会失败。Phase 13 · 17 讲解网关级别的背压（backpressure）；重试语义计划在后续阶段中介绍。

### 流式扇出的墙上时钟时间

如果模型本身以流式输出，你可以在某一调用的参数完整后立即开始执行，而不必等所有调用都完成。这是 OpenAI 文档中提到的优化，但并非所有 SDK 都开放。本节课的封装做到了这一点：一旦模拟流产生一个完整的参数对象，宿主就立即启动该调用。

## 动手尝试

`code/main.py` 分为两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行地运行三次模拟天气调用，并打印墙上时钟时间。第二部分重放一个伪造的流式响应——三个并行调用的 `arguments` 片段交错在一个流中——并通过 `StreamAccumulator` 按 id 重组。没有真实 LLM，也没有网络，只有重组逻辑。

观察要点：

- 串行计时约为 1.8 秒。在相同伪造延迟下，并行计时约为 0.8 秒。
- 累加器通过按 id 缓冲，只在每个调用的 JSON 完整后才解析，从而处理乱序到达的数据块。
- 执行器在某个 id 的参数一完成就启动，而不是等所有流都结束。

## 交付成果

本节课会生成 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表（tool registry），该技能会审计哪些工具可以安全并行、哪些存在顺序依赖、哪些会压垮下游速率限制——并返回一个带每个工具 `parallel_safe` 标志的修订版注册表。

## 练习

1. 运行 `code/main.py` 并调整模拟延迟。确认并行与串行的比例大致为 `max/sum`（实际运行会略偏离理想值，因为线程调度、序列化和封装开销的存在）。在什么延迟分布下，并行不再有意义？

2. 扩展累加器以处理“调用在流中途被取消”的情况：丢弃该调用的缓冲区并发出 `cancelled` 事件。哪个提供商在文档中明确说明了这种情况？查看 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 将线程池替换为 `asyncio.gather` 并对两者进行基准测试。由于上下文切换开销更低，异步版本通常会有小幅提升，但前提是执行器真正执行 I/O。

4. 挑选两个不应并行的工具（例如 `create_file` 然后 `write_file`）。在注册表中添加 `ordering_dependency` 图，并根据该图控制并行扇出。这是依赖感知调度的最小机制，后续的智能体工程阶段会将其形式化。

5. 阅读 OpenAI 的 parallel-function-calling 章节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 明确建议关闭并行的那一类真实工具类型。（提示：对同一资源产生重大变更的修改操作。）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 并行工具调用（parallel tool calls） | “一个回合内的扇出” | 模型在单条 assistant 消息中发出多个工具调用 |
| `parallel_tool_calls` | “OpenAI 的开关” | 启用或禁用多调用发射 |
| `disable_parallel_tool_use` | “Anthropic 的反相开关” | 退出并行的标志；默认开启并行 |
| 工具调用 id（tool call id） | “关联句柄” | 结果消息必须回显的每次调用唯一标识 |
| 累加器（accumulator） | “流缓冲区” | 按 id 保存部分 `arguments` 片段的字符串缓冲区 |
| 乱序完成（out-of-order completion） | “快的先返回” | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| 依赖图（dependency graph） | “顺序约束” | 某些工具的输出会作为其他工具的输入；不能并行 |
| 过早解析陷阱（parse-early trap） | “JSON.parse 崩了” | 试图解析尚未完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | “Gemini 3 特性” | 每次调用带唯一 id 的流式参数片段 |
| 按完成顺序回复（completion-order reply） | “不用等全部” | 结果按到达顺序返回，并用 id 作为键 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为与退出标志
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 与结果批处理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 的 id 关联并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流式响应的分段参数重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`
