# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个智能体（agent）调用五个工具、三个 MCP 服务器和两个子智能体（sub-agent）。你需要一个贯穿这一切的追踪（trace）。OpenTelemetry GenAI 语义约定（semantic conventions，在 v1.37 及以上版本中为稳定属性）是 2026 年的标准，得到 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 的原生支持。本课会列出必需的属性，演示跨度（span）层级（智能体 → LLM → 工具），并提供一个标准库跨度发射器，可接入任何 OTel 导出器。

**Type:** Build
**Languages:** Python（标准库、OTel 跨度发射器）
**Prerequisites:** Phase 13 · 07（MCP 服务器）、Phase 13 · 08（MCP 客户端）
**Time:** 约 75 分钟

## 学习目标

- 说出 LLM 跨度和工具执行（tool-execution）跨度所需的 OTel GenAI 属性。
- 构建一个追踪层级，覆盖智能体循环、LLM 调用、工具调用和 MCP 客户端分发。
- 决定捕获（主动开启）哪些内容，以及默认情况下脱敏哪些内容。
- 无需重写工具代码，即可将跨度发送到本地收集器（Jaeger、Langfuse）。

## 问题背景

2026 年 2 月的一次调试：用户反馈“我的智能体有时 30 秒才响应，有时只要 3 秒。”没有追踪。日志里有 LLM 调用，但没有工具分发、没有 MCP 服务器往返、也没有子智能体。你只能猜测。最后发现：某个 MCP 服务器偶尔在冷启动时卡住。

没有端到端追踪，你根本发现不了这个问题。OTel GenAI 可以修复这一点。

这些约定在 2025–2026 年间由 OpenTelemetry 语义约定小组确定。它们定义了稳定的属性名称，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的跨度。一次埋点，即可发送到任何后端。

## 核心概念

### 跨度层级

```
agent.invoke_agent  （顶层，INTERNAL 跨度）
 ├── llm.chat       （CLIENT 跨度）
 ├── tool.execute   （INTERNAL）
 │    └── mcp.call  （CLIENT 跨度）
 ├── llm.chat       （CLIENT 跨度）
 └── subagent.invoke （INTERNAL）
```

整个结构都嵌套在同一个追踪 ID 之下。跨度 ID 将父子关系关联起来。

### 必需属性

根据 2025–2026 年的语义约定：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际提供服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 用于关联的提供商响应 ID。

对于工具跨度：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体的调用 ID。
- `gen_ai.tool.description` — 工具描述（可选）。

对于智能体跨度：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### 跨度类型

- `SpanKind.CLIENT`：跨越进程边界的调用（LLM 提供商、MCP 服务器）。
- `SpanKind.INTERNAL`：智能体自身循环步骤和工具执行。

### 可选的内容捕获

默认情况下，跨度只携带指标和时间信息，不包含提示词（prompt）和生成结果（completion）。大负载和个人身份信息（PII）默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 以及特定的内容捕获环境变量才能包含内容。在生成环境启用前请仔细审查。

### 跨度事件

可以在跨度上添加 token 级别的事件：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件按时间顺序排列在跨度内，可用于详细回放。

### 导出器

OTel 跨度可导出到：

- **Jaeger / Tempo。** 开源，可本地部署。
- **Langfuse。** 面向 LLM 可观测性；可视化 token 使用量。
- **Arize Phoenix。** 评估（evals）与追踪结合。
- **Datadog。** 商业方案；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；便于查询。

它们都使用 OTLP 作为传输格式。你的代码无需关心后端。

### 跨 MCP 传播

当 MCP 客户端调用服务器时，将 W3C `traceparent` 请求头注入请求。可流式 HTTP 支持标准请求头。Stdio 本身不携带 HTTP 请求头；该规范 2026 年路线图讨论在 JSON-RPC 调用中添加 `_meta.traceparent` 字段。

在该功能落地前：手动在每个请求的 `_meta` 中包含 `traceparent`。服务器会记录该追踪 ID。

### 指标

除了跨度，GenAI 语义约定还定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

对于不需要每次调用细节的面板，可以使用这些指标。

### AgentOps 层

AgentOps（成立于 2024 年）专注于生成式 AI 可观测性。它封装了主流框架（LangGraph、Pydantic AI、CrewAI）以自动发出 OTel 跨度。如果你的技术栈使用受支持的框架，这会很有用；否则请使用手动埋点。

## 动手使用

`code/main.py` 会 stdout 输出 OTel 格式的跨度（近似 OTLP-JSON 格式），对应一个调用 LLM、分发两个工具、并进行一次 MCP 往返的智能体。没有真实的导出器——本课关注的是跨度形态和属性集合。你可以把输出粘贴到兼容 OTLP 的查看器中，或直接阅读。

需要关注：

- 所有跨度共享同一个追踪 ID。
- 父子关系通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已被填充。
- 内容捕获默认关闭；其中一个场景通过环境变量开启。

## 交付产物

本课会产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该技能会生成一份埋点方案：在哪里添加跨度、填充哪些属性、以及目标导出器是哪些。

## 练习题

1. 运行 `code/main.py`。统计跨度数量，并识别哪些是 CLIENT，哪些是 INTERNAL。

2. 开启内容捕获（环境变量），确认出现 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件。注意 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并在每次调用时作为直方图样本发出。

4. 将 `traceparent` 从父智能体跨度传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器看到的是同一个追踪 ID。

5. 阅读 OTel GenAI 语义约定规范。找出一个本课代码未发出的属性，并将其添加进去。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|-----------|----------|
| OTel | "OpenTelemetry" | traces、metrics、logs 的开放标准 |
| GenAI semconv | "GenAI semantic conventions" | LLM / 工具 / 智能体跨度的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "Timed operation" | 具有开始、结束和属性的工作单元 |
| Trace | "Cross-span ancestry" | 共享同一追踪 ID 的跨度树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于跨度方向的提示 |
| OTLP | "OpenTelemetry Line Protocol" | 导出器的传输格式 |
| 可选内容 | "Prompt / completion capture" | 默认关闭；通过环境变量开启 |
| traceparent | "W3C header" | 在服务间传播追踪上下文 |
| Exporter | "Backend-specific shipper" | 将跨度发送到 Jaeger / Datadog 等后端的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI 跨度、指标和事件的权威约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行跨度属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级 `invoke_agent` 跨度
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 上的事实来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产环境集成实战
