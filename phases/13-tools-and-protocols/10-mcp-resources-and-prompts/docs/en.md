# MCP 资源（Resources）与提示词（Prompts）——超越工具（Tools）的上下文暴露

> 工具（tools）占据了 MCP 90% 的讨论热度。然而，另外两种服务器原语解决的问题截然不同：资源（resources）暴露可读数据；提示词（prompts）以“斜杠命令（slash-commands）”的形式暴露可复用模板。许多服务器本应以资源的形式暴露读取能力，而不是把它包装成工具；也应以提示词的形式暴露工作流，而不是把流程硬编码到客户端提示词里。本课给出决策规则，并带你走一遍 `resources/*` 与 `prompts/*` 消息。

**类型：** Build
**语言：** Python（stdlib，resource + prompt handler）
**先修：** Phase 13 · 07（MCP server）
**时长：** 约 45 分钟

## 学习目标

- 针对给定领域，判断应该把某项能力暴露为工具（tool）、资源（resource）还是提示词（prompt）。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 实现 `prompts/list` 与 `prompts/get`，支持参数模板。
- 识别宿主何时将提示词展示为斜杠命令，何时自动注入上下文。

## 问题所在

一个为笔记应用写的入门级 MCP 服务器会把所有能力都暴露成工具：`notes_read`、`notes_list`、`notes_search`。它把每一次数据访问都包装成模型驱动的工具调用。后果如下：

- 模型每次遇到可能受益于上下文的查询，都得决定是否调用 `notes_read`。
- 只读内容无法被订阅，也无法流式同步到宿主侧边栏。
- 客户端 UI（Claude Desktop 的资源附件面板、Cursor 的“包含文件”选择器）无法展示这些数据。

合理的划分方式：把数据暴露为资源；把会修改数据或经过计算的动作暴露为工具；把可复用的多步骤工作流暴露为提示词。每种原语都有各自的交互形态与访问模式。

## 核心概念

### 工具、资源与提示词——决策规则

| 能力 | 原语 |
|------|------|
| 用户想搜索、过滤或转换数据 | tool |
| 用户希望宿主把这份数据作为上下文包含进来 | resource |
| 用户想要一个可重复运行的模板化工作流 | prompt |

经验法则：如果模型在每次相关查询中调用它都会受益，那就是工具；如果用户在把它附加到对话时会受益，那就是资源；如果用户复用的基本单元是一整套多步骤工作流，那就是提示词。

### 资源（Resources）

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}`，返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义 scheme）
- `memory://session-2026-04-22/recent`（服务器专属 scheme）

`contents[]` 同时支持文本与二进制。二进制使用 `blob` 字段，内容为 base64 编码字符串，并附带 `mimeType`。

### 资源订阅（Resource subscriptions）

在能力声明中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}` 订阅。当资源变化时，服务器发送 `notifications/resources/updated {uri}`，客户端随后重新读取。

用例：一个笔记服务器的资源对应磁盘上的文件；文件监听器触发更新通知；当文件在宿主外部被编辑时，Claude Desktop 会重新拉取该文件到上下文中。

### 资源模板（Resource templates，2025-11-25 新增）

`resourceTemplates` 可让你暴露带参数的 URI 模式：例如 `notes://{id}`，其中 `id` 是补全目标。客户端可以在资源选择器中自动补全 id。

### 提示词（Prompts）

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}`，返回 `{description, messages: [{role, content}]}`。

提示词是一个模板，填充后得到一组消息，由宿主喂给它的模型。例如，一个 `code_review` 提示词接收 `file_path` 参数，返回三段消息：一条系统消息、一条附带文件内容的用户消息，以及一条带推理模板的助手开场消息。

### 宿主与提示词

Claude Desktop、VS Code 和 Cursor 会把提示词作为斜杠命令暴露在聊天 UI 中。用户输入 `/code_review` 并通过表单选择参数。服务器端的提示词就是“用户快捷方式”与“实际发送给模型的完整提示词”之间的契约。

并非所有客户端都支持提示词——需通过能力协商检查。服务器声明了提示词能力，但客户端若不支持，就看不到这些斜杠命令。

### “列表变更”通知

资源与提示词都会在集合发生变化时发出 `notifications/list_changed`。例如，笔记服务器刚导入 20 条新笔记时，会发出 `notifications/resources/list_changed`；客户端随后重新调用 `resources/list` 以获取新增项。

### 内容类型约定

- 文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
- 二进制：`image/png`、`application/pdf`，并附带 `blob` 字段。
- MCP Apps（第 14 课）：在 `ui://` URI 中使用 `text/html;profile=mcp-app`。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 每次读取都可以返回最新的五条笔记；`db://query/users/active` 可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端可以按 URI 缓存，那么该 URI 必须保持稳定。如果内容是一次性的，URI 应包含时间戳或 nonce，以免客户端缓存过期数据。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。不支持订阅的旧客户端或宿主则通过重新读取来轮询。两者都符合规范。服务器的能力声明会告知客户端自己支持哪一种。

订阅的成本：服务器需要维护每会话状态（谁在订阅什么）。保持订阅集合有界；断开的客户端应超时清理。

### 提示词 vs 系统提示词

MCP 中的提示词不是系统提示词（system prompt）。宿主的系统提示词（它自身的操作指令）与 MCP 提示词（由服务器提供、用户调用的模板）是并存的。一个行为良好的客户端绝不允许服务器提示词覆盖自己的系统提示词，而是将它们分层叠加。

## 动手实践

`code/main.py` 在第 07 课的笔记服务器基础上扩展了以下内容：

- 单条笔记资源（`notes://note-1` 等），支持 `resources/subscribe`。
- `review_note` 提示词，渲染为三段消息模板。
- 模拟文件监听器，当笔记被修改时发出 `notifications/resources/updated`。
- `notes://recent` 动态资源，始终返回最新的五条笔记。

运行演示即可看到完整流程。

## 交付成果

本课会生成 `outputs/skill-primitive-splitter.md`。针对一个拟议的 MCP 服务器，该交付物会对每项能力进行分类：应作为 tool / resource / prompt，并给出理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发一次笔记编辑，验证 `notifications/resources/updated` 事件是否触发。

2. 新增一个 `resources/list_changed` 发射器：当新笔记创建时发送通知，使客户端重新发现资源。

3. 为一个 GitHub MCP 服务器设计三个提示词：`summarize_pr`、`triage_issue`、`release_notes`。每个都要带参数模式（argument schemas），提示词体应可直接运行、无需二次编辑。

4. 从第 07 课的服务器中选取一个现有工具，判断它应继续作为 tool，还是拆分为“resource + tool”组合。用一句话说明理由。

5. 阅读规范中的 `server/resources` 与 `server/prompts` 章节。找出 `resources/read` 中一个“很少被填充但规范支持”的字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| Resource（资源） | “暴露的数据” | 宿主可读取的、可通过 URI 寻址的内容 |
| Resource URI（资源 URI） | “数据指针” | 带 scheme 前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | “监听变化” | 客户端可选的、针对特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | “资源已变更” | 通知客户端：某个已订阅资源有了新内容 |
| Resource template（资源模板） | “参数化 URI” | 带补全提示的 URI 模式，供宿主选择器使用 |
| Prompt（提示词） | “斜杠命令模板” | 带参数槽位的命名多消息模板 |
| Prompt arguments（提示词参数） | “模板输入” | 宿主在渲染前收集的带类型参数 |
| `prompts/get` | “渲染模板” | 服务器返回填充完成的消息列表 |
| Content block（内容块） | “类型化数据块” | `{type: text \| image \| resource \| ui_resource}` |
| Slash-command UX（斜杠命令交互） | “用户快捷方式” | 宿主以 `/` 开头的命令形式展示提示词 |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) —— 资源 URI、订阅与模板
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) —— 提示词模板与斜杠命令集成
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) —— 完整的 `resources/*` 消息参考
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) —— 完整的 `prompts/*` 消息参考
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) —— 社区指南，对官方文档的扩展
