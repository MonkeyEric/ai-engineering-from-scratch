# 函数调用（Function Calling）深入解析 —— OpenAI、Anthropic、Gemini

> 2024 年，三家前沿大模型供应商在工具调用（tool-call）循环上殊途同归，随后又在其他所有细节上分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 并通过唯一 ID 进行关联。本课将三者的差异并排对比，避免你的代码从一家供应商迁移到另一家时意外损坏。

**Type:** 实战构建
**Languages:** Python（标准库、schema 转换器）
**Prerequisites:** Phase 13 · 01（工具接口）
**Time:** 约 75 分钟

## 学习目标

- 说明 OpenAI、Anthropic 和 Gemini 函数调用（function calling）负载在声明（declaration）、调用（call）、结果（result）三种形态上的差异。
- 将一个工具声明（tool declaration）转换为三家供应商各自的格式，并预测严格模式（strict mode）约束会在哪些地方不同。
- 在每家供应商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 掌握每家供应商的硬性限制（工具数量、schema 深度、参数长度），以及超出限制时各自返回的错误特征。

## 问题背景

函数调用请求的格式因供应商而异。以下是 2026 年生产栈中的三个真实示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型返回的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是必须解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码（constrained decoding）强制要求符合 schema。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已经解析完毕（是对象，不是字符串）。你回复一条新的 `user` 消息，其中包含 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及以上版本中唯一，用于并行调用的关联。你回复 `{functionResponse: {name, id, response}}`。

循环逻辑相同，但字段名、嵌套方式、字符串与对象的约定、关联机制各不相同。一个在 OpenAI 上写出天气智能体的团队，迁移到 Anthropic 要花两天做适配，再迁移到 Gemini 又要花一天，时间都耗费在基础管道上。

本课将构建一个转换器，把三种格式统一为一种规范的工具声明（canonical tool declaration），并在边界处进行路由。Phase 13 · 17 会把同一模式泛化为 LLM 网关（gateway）。

## 核心概念

### 共同结构

每家供应商都需要五样东西：

1. **工具列表（Tool list）。** 每个工具的名称、描述和输入 schema。
2. **工具选择（Tool choice）。** 强制指定工具、禁止工具，或让模型自行决定。
3. **调用输出（Call emission）。** 结构化输出，包含工具名和参数。
4. **调用 ID（Call id）。** 将响应与正确的调用关联（对并行调用尤为重要）。
5. **结果注入（Result injection）。** 将结果绑定回对应调用的消息或块。

### 逐字段形态对比

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明封装 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant 消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化 JSON | 已解析对象 | 已解析对象 |
| ID 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | role `tool`、`tool_call_id` | 包含 `tool_result`、`tool_use_id` 的 `user` | 匹配 `id` 的 `functionResponse` |
| 强制指定工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 即契约（始终强制） | 请求级别的 `responseSchema` |

### 你实际会碰到的限制

- **OpenAI。** 每次请求最多 128 个工具。Schema 深度 5。参数字符串 ≤ 8192 字节。严格模式要求：无 `$ref`、无重叠的 `oneOf`/`anyOf`/`allOf`，每个属性都必须在 `required` 中列出。
- **Anthropic。** 每次请求最多 64 个工具。Schema 深度理论上无上限，但实际建议 ≤ 10。没有严格模式开关；schema 即契约，模型通常会遵守。
- **Gemini。** 每次请求最多 64 个函数。Schema 类型属于 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有差异）。Gemini 3 起为并行调用分配唯一 ID。

### `tool_choice` 行为

三家都支持的三种模式，只是命名不同。

- **自动（Auto）。** 模型选择调用工具或输出文本。默认值。
- **必需 / 任意（Required / Any）。** 模型必须至少调用一个工具。
- **无（None）。** 模型不得调用工具。

每家供应商还有一个独有的模式：

- **OpenAI。** 按名称强制指定工具。
- **Anthropic。** 按名称强制指定工具；`disable_parallel_tool_use` 标志区分单工具与多工具。
- **Gemini。** `mode: "VALIDATED"` 无视模型意图，将每个响应都经过 schema 校验器。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）允许一条 assistant 消息发出多个调用。你同时执行它们，然后回复一条批量的 tool-role 消息，每个 `tool_call_id` 对应一条记录。Anthropic 历史上是单调用模式；`disable_parallel_tool_use: false`（Claude 3.5 起默认）启用多调用。Gemini 2 支持并行调用，但没有稳定的 ID；Gemini 3 增加了 UUID，乱序响应也能干净地关联。

### 流式响应

三家都支持流式工具调用。但底层格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量分片（delta chunks）逐步到达。持续累积，直到 `finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop 事件。`input_json_delta` 分片携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出的分片带有 `functionCallId`，多个并行调用可以交错出现。

Phase 13 · 03 将深入讲解并行 + 流式重组。本课聚焦声明和单次调用的形态。

### 错误与修复

参数无效的错误形态也不同。

- **OpenAI（非严格模式）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，于是注入错误消息并重新调用。
- **OpenAI（严格模式）。** 校验发生在解码阶段；不可能出现无效 JSON，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含未声明的字段；schema 仅供参考。请在服务端自行校验。
- **Gemini。** OpenAPI 3.0 的怪癖：对象字段上的 `enum` 会被静默忽略；请自行校验。

### 转换器模式

代码中的规范工具声明（canonical tool declaration）可以这样定义（具体形态由你决定）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数把它转换成三家供应商各自的形态。`code/main.py` 中的 harness 正是这样做的：它把一个伪造的工具调用依次经过每家供应商的响应形态往返一遍。不需要网络——本课教你的是形态，而不是 HTTP。

生产团队会把这种转换器包装进 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）。Phase 13 · 17 将交付一个网关（gateway），在三者中的任意一家前面暴露 OpenAI 风格的 API。

## 动手使用

`code/main.py` 定义了一个规范的 `Tool` 数据类，以及三个转换器，分别生成 OpenAI、Anthropic 和 Gemini 的声明 JSON。随后它把每种形态的构造响应解析为同一个规范调用对象，证明表层之下语义相同。运行它，并并排对比三种声明。

重点关注：

- 三个声明块只在封装和字段名上有区别。
- 三个响应块的区别在于调用所在的位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 项）。
- 一个 `canonical_call()` 函数即可从三种响应形态中提取 `{id, name, args}`。

## 交付成果

本课产出 `outputs/skill-provider-portability-audit.md`。针对面向某一家供应商的函数调用集成，该技能会生成可移植性审计（portability audit）：它依赖哪些供应商限制、哪些字段需要重命名、迁移到其他供应商时会出什么错。

## 练习

1. 运行 `code/main.py`，验证三家供应商的声明 JSON 都序列化自同一个底层 `Tool` 对象。修改规范工具，添加一个 enum 参数，并确认只有 Gemini 转换器需要处理 OpenAPI 的这个小怪癖。

2. 为每家供应商添加一个 `ListToolsResponse` 解析器，用于提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有该能力；注意这种不对称性。

3. 实现 `tool_choice` 转换：把规范的 `ToolChoice(mode="force", tool_name="x")` 映射到三家供应商的形态。然后映射 `mode="any"` 和 `mode="none"`。对照本课的对比表。

4. 选择三家供应商之一，通读其函数调用指南。找出其 schema 规范中其他两家不支持的一个字段。候选：OpenAI 的 `strict`、Anthropic 的 `disable_parallel_tool_use`、Gemini 的 `function_calling_config.allowed_function_names`。

5. 编写一个测试向量（test vector）：参数违反声明 schema 的工具调用。用它依次跑过每家供应商的校验器（Lesson 01 中的标准库校验器可作为代理），记录哪些错误被触发。记录你会在生产中为严格性选择哪家供应商。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 函数调用（Function calling） | "Tool use" | 供应商层面的结构化工具调用发射 API |
| 工具声明（Tool declaration） | "Tool spec" | 名称 + 描述 + JSON Schema 输入负载 |
| `tool_choice` | "Force / forbid" | 自动 / 必需 / 禁止 / 指定名称等模式 |
| 严格模式（Strict mode） | "Schema enforcement" | OpenAI 的标志位，通过约束解码让输出匹配 schema |
| `tool_use` 块 | "Anthropic's call shape" | 包含 id、name、input 的内联内容块 |
| `functionCall` 部分 | "Gemini's call shape" | 包含 name、args 和 id 的 `parts[]` 项 |
| 字符串参数（Arguments-as-string） | "Stringified JSON" | OpenAI 将参数作为 JSON 字符串返回，而非对象 |
| 并行工具调用（Parallel tool calls） | "Fan-out in one turn" | 一条 assistant 消息中的多个工具调用 |
| 拒绝（Refusal） | "Model declines" | 仅在严格模式下出现的拒绝块，代替工具调用 |
| OpenAPI 3.0 子集 | "Gemini schema quirk" | Gemini 使用一种类似 JSON Schema 的方言，存在细微差异 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 包含严格模式和并行调用的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 与 `tool_result` 块的语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 与 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级接口
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制的细节
