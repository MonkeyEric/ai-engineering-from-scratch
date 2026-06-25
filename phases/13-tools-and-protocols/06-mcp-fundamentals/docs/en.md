# MCP 基础 — 原语、生命周期、JSON-RPC 基础

> MCP 之前的每一次集成都是一次性的。模型上下文协议（Model Context Protocol，MCP）由 Anthropic 于 2024 年 11 月首次发布，现由 Linux 基金会的 Agentic AI Foundation 托管，它规范了发现（discovery）和调用（invocation），使任何客户端都能与任何服务器通信。2025-11-25 版规范定义了六种原语（primitive，服务端三种、客户端三种）、三阶段生命周期以及 JSON-RPC 2.0 的传输格式。掌握这些之后，本阶段 MCP 章节的其余内容就只剩下阅读了。

**类型：** Learn
**语言：** Python（stdlib、JSON-RPC parser）
**前置知识：** Phase 13 · 01 至 05（工具接口与函数调用）
**时长：** 约 45 分钟

## 学习目标

- 说出全部六种 MCP 原语（服务端：tools、resources、prompts；客户端：roots、sampling、elicitation），并各给出一个用例。
- 梳理三阶段生命周期（initialize、operation、shutdown），并说明每一阶段由哪一方发送哪些消息。
- 解析并生成 JSON-RPC 2.0 的请求、响应和通知信封。
- 解释 `initialize` 时的能力协商是什么，以及缺少它会怎样导致系统失灵。

## 问题背景

MCP 出现之前，每个使用工具的代理都有自己的协议。Cursor 有一套形似 MCP 但互不兼容的工具系统；Claude Desktop 用的是另一种；VS Code 的 Copilot 扩展则是第三种。一个团队要写“Postgres 查询”工具，就得为三种不同的宿主 API 各写一遍。复用它意味着复制代码。

结果是一次性集成如寒武纪大爆发般涌现，生态演进速度也遇到了天花板。

MCP 通过标准化传输格式解决了这个问题。一个 MCP 服务器可以在任何 MCP 客户端中运行：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，截至 2026 年 4 月已有 300 多个客户端。每月 SDK 下载量达 1.1 亿次，公开服务器超过 1 万个。Linux 基金会于 2025 年 12 月在新成立的 Agentic AI Foundation 下接管了该协议。

本阶段使用的规范版本是 **2025-11-25**。它增加了异步任务（async Tasks，SEP-1686）、URL 模式引导输入（URL-mode elicitation，SEP-1036）、带工具的采样（sampling with tools，SEP-1577）、增量范围授权（incremental scope consent，SEP-835）以及 OAuth 2.1 资源指示符语义。阶段 13 · 09 至 16 会讲解这些扩展。本节课只涉及基础部分。

## 核心概念

### 三种服务端原语

1. **Tools（工具）。** 可调用的动作。与阶段 13 · 01 中的四步循环相同。
2. **Resources（资源）。** 暴露的数据。只读内容，可通过 URI 寻址：`file:///path`、`db://query/...` 或自定义协议。
3. **Prompts（提示）。** 可复用的模板。宿主 UI 中的斜杠命令；服务端提供模板，客户端填充参数。

### 三种客户端原语

4. **Roots（根）。** 服务器被允许访问的 URI 集合。由客户端声明，服务端遵守。
5. **Sampling（采样）。** 服务端请求客户端模型执行一次补全。它让服务端托管的代理循环无需服务端侧 API 密钥即可运行。
6. **Elicitation（引导输入）。** 服务端在运行中途请求客户端用户进行结构化输入。可以是表单或 URL（SEP-1036）。

MCP 中的每一项能力都恰好属于这六种之一。阶段 13 · 10 至 14 会深入讲解每一种。

### 传输格式：JSON-RPC 2.0

每条消息都是一个 JSON 对象，包含以下字段：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — 没有 `id`，不期待响应。

基础规范约有 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务端发往客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：initialize。**

客户端发送 `initialize`，附带其 `capabilities` 和 `clientInfo`。服务端以自身 `capabilities`、`serverInfo` 以及它支持的规范版本作为响应。客户端消化完响应后发送 `notifications/initialized`。自此之后，双方都可以按照协商好的能力发送请求。

**阶段 2：operation。**

双向进行。客户端调用 `tools/list` 进行发现，再调用 `tools/call` 执行。如果服务端声明了该能力，它可以发送 `sampling/createMessage`。当工具集合发生变化时，服务端可以发送 `notifications/tools/list_changed`；当用户修改根范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：shutdown。**

任意一方关闭传输层。MCP 中没有结构化的关闭方法；传输层（stdio 或 Streamable HTTP，阶段 13 · 09）负责承载连接结束信号。

### 能力协商

`initialize` 握手时的 `capabilities` 就是契约。下面是一个服务端的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务端声明它可以发出 `tools/list_changed` 通知，并支持 `resources/subscribe`。客户端通过声明自己的能力来响应：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务端就不得调用 `sampling/createMessage`。反之亦然：如果服务端没有声明 `resources.subscribe`，客户端也不得尝试订阅。

这正是防止生态分裂的机制。不支持采样的客户端仍然是合法的 MCP 客户端；不调用 `sampling` 的服务端仍然是合法的 MCP 服务端。它们只是在一起时不会使用该功能。

### 结构化内容与错误格式

`tools/call` 返回一个 `content` 数组，其中的块带类型：`text`、`image`、`resource`。阶段 13 · 14 还会把 MCP Apps（`ui://` 交互式 UI）加入这个列表。

错误使用 JSON-RPC 错误码。规范中新增的包括：`-32002`“Resource not found”、`-32603`“Internal error”，以及 MCP 特定的错误数据 `error.data`。

### 客户端能力与工具调用细节

一个常见误解：`capabilities.tools` 表示客户端是否支持工具列表变化通知。客户端“会不会”调用某个具体工具则是由模型决定的运行时选择，而非能力标志。能力标志是规范层面的契约，模型的选择与之正交。

### 为什么用 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一种轻量级双向协议。REST 由客户端发起。MCP 需要服务端主动发送的消息（采样、通知），因此 JSON-RPC 这种对称的请求/响应形态是自然之选。此外，JSON-RPC 可以干净地组合在 stdio 以及 WebSocket/Streamable HTTP 之上，无需重新发明 HTTP 的请求形态。

## 动手实践

`code/main.py` 提供了一个最小化的 JSON-RPC 2.0 解析器和生成器，并手动演示 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每一条消息。没有真实传输层，只展示消息形态。可对照“延伸阅读”中链接的规范来核对每个信封。

关注要点：

- `initialize` 双向声明能力；响应中包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目包含 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应的 `content` 是一个 `{type, text}` 块数组。

## 交付成果

本节课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一份类似 pcap 转录的 MCP 客户端-服务端交互记录，该技能会为每条消息标注其所属原语、生命周期阶段以及依赖的能力。

## 练习

1. 运行 `code/main.py`。找到发生能力协商的那一行，并描述如果服务端没有声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形态为：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在一个长时间运行的 `tools/call` 进行过程中发送该通知，并确认客户端处理程序会显示进度条。

3. 通读 MCP 2025-11-25 规范全文——整份文档约 80 页。找出大多数服务端都不需要的那一个能力标志。提示：它与资源订阅有关。

4. 在纸上勾勒一个假设的“定时任务（cron job）”功能应属于哪个原语。（提示：服务端希望客户端在预定时间调用它。目前的六种原语都不匹配。）MCP 2026 年路线图中有一份针对此事的 SEP 草案。

5. 从 GitHub 上一个公开的 MCP 服务端解析一份会话日志。统计请求、响应、通知消息的数量。计算生命周期流量与业务流量的比例。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MCP | “Model Context Protocol” | 用于模型到工具的发现与调用的开放协议 |
| 服务端原语 | “服务端暴露什么” | tools（动作）、resources（数据）、prompts（模板） |
| 客户端原语 | “客户端允许服务端使用什么” | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | “传输格式” | 对称的请求/响应/通知信封 |
| `initialize` 握手 | “能力协商” | 第一对消息；服务端与客户端声明各自支持的功能 |
| `tools/list` | “发现” | 客户端向服务端请求当前工具集合 |
| `tools/call` | “调用” | 客户端请求服务端用参数执行工具 |
| `notifications/*_changed` | “变更事件” | 服务端告知客户端其原语列表已变化 |
| 内容块 | “带类型的结果” | 工具结果中的 `{type: "text" | "image" | "resource" | "ui_resource"}` |
| SEP | “Spec Evolution Proposal” | 命名草案提案（例如 SEP-1686 用于异步任务） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六种原语的心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布博文
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾与 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 与 1724 的摘要
