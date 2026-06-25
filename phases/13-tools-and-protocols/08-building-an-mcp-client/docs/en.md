# 构建 MCP 客户端 —— 发现、调用与会话管理

> 大多数 MCP 内容都在推销服务器教程，而对客户端一笔带过。客户端代码才是复杂编排真正所在：进程派生、能力协商、跨多台服务器的工具列表合并、采样回调、重连，以及命名空间冲突解决。本节课将构建一个多服务器客户端，把三个不同的 MCP 服务器提升到一个扁平的工具命名空间中供模型使用。

**类型：** 构建
**语言：** Python（标准库，多服务器 MCP 客户端）
**前置知识：** Phase 13 · 07（构建 MCP 服务器）
**时间：** 约 75 分钟

## 学习目标

- 将 MCP 服务器作为子进程派生，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每台服务器的会话状态（capabilities、工具列表、最后看到的通知 id）。
- 将多台服务器的工具列表合并到一个命名空间中，并处理冲突。
- 将工具调用路由到拥有它的服务器，并重新组装响应。

## 问题背景

一个真正的智能体宿主（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端的职责：

1. 派生每个服务器。
2. 分别与每个服务器握手。
3. 对每个服务器调用 `tools/list`，并将结果扁平化。
4. 当模型发出 `notes_search` 时，在合并后的命名空间中查找，并路由到正确的服务器。
5. 在不阻塞的情况下处理来自任何服务器的通知（`tools/list_changed`）。
6. 在传输失败时重新连接。

所有这些都需要亲手实现，这正是“玩具”与“可用”之间的分水岭。官方 SDK 会封装这些逻辑，但你的心智模型必须属于自己。

## 核心概念

### 子进程派生

使用 `subprocess.Popen` 并设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。将 `bufsize` 设为 `1` 并使用文本模式以便逐行读取。每个服务器对应一个进程；客户端为每台服务器持有一个 `Popen` 句柄。

### 每台服务器的会话状态

为每台服务器维护一个 `Session` 对象，包含：

- `process` —— Popen 句柄。
- `capabilities` —— 服务器在 `initialize` 时声明的能力（capabilities）。
- `tools` —— 最新的 `tools/list` 结果。
- `pending` —— 请求 id 到等待响应的 promise/future 的映射。

请求本质上是异步的；在向服务器 A 发送 `tools/call` 的同时，服务器 B 正处于一次调用中间，这不能阻塞。可以使用线程加队列，也可以使用 asyncio。

### 合并命名空间

当客户端看到聚合后的工具列表时，名称可能发生冲突。两个服务器可能都暴露了 `search`。客户端有三种选择：

1. **按服务器名前缀。** `notes/search`、`files/search`。清晰但丑陋。
2. **静默先到优先。** 后到的服务器 `search` 覆盖先前的。有风险；会隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二台服务器，并通知用户。对安全敏感的宿主来说最安全。

Claude Desktop 使用按服务器名前缀。Cursor 使用冲突拒绝并给出清晰错误。VS Code MCP 也采用按服务器名前缀。

### 路由

合并后，一个调度表将 `tool_name -> session` 映射。模型按名称发出调用；客户端找到对应 session，然后向该服务器的 stdin 写入一条 `tools/call` 消息，并等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能会发送 `sampling/createMessage`，请求客户端运行自己的 LLM。客户端必须：

1. 阻塞对该服务器的后续请求，直到采样完成；如果实现支持并发，也可以流水线处理。
2. 调用自己的 LLM 提供方。
3. 将响应发回服务器。

第 11 课完整覆盖端到端采样。本节课为完整性起见只做占位实现。

### 通知处理

`notifications/tools/list_changed` 意味着需要重新调用 `tools/list`。`notifications/resources/updated` 意味着如果正在使用该资源，则需要重新读取。通知不产生响应 —— 不要尝试确认（ack）它们。

客户端常见 bug：在 `tools/call` 上阻塞读取循环，而通知还停留在流中。应使用后台读取线程，将每条消息推入队列；主线程从队列中取出并分发。

### 重连

传输可能失败：服务器崩溃、操作系统杀死进程、stdio 管道断裂。客户端检测到 stdout 上的 EOF，并将该会话标记为死亡。可选策略：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 将失败暴露给用户。适用于具有用户可见会话的状态型服务器。

Phase 13 · 09 会覆盖 Streamable HTTP 的重连语义；stdio 更简单。

### 保活与会话 id

Streamable HTTP 使用 `Mcp-Session-Id` 请求头。Stdio 没有会话 id —— 进程身份本身就是会话。保活 ping 是可选的；stdio 管道不会因空闲而断开。

## 使用它

`code/main.py` 将三个模拟的 MCP 服务器作为子进程派生，分别与它们握手，合并它们的工具列表，并将工具调用路由到正确的服务器。这些“服务器”实际上是运行简单响应程序的其他 Python 进程（没有真正的 LLM）。运行它可以看到：

- 三次初始化，每个都有自己的能力集。
- 三个 `tools/list` 结果合并成一个包含 7 个工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀避免的冲突。

重点观察：

- `Session` 数据类清晰地保存了每台服务器的状态。
- 后台读取线程从 stdout 上逐行取出数据，不会阻塞主线程。
- 调度表是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明相同名称时，后到的那个会被加上前缀重命名。

## 交付物

本节课生成 `outputs/skill-mcp-client-harness.md`。给定一个声明式的 MCP 服务器列表（名称、命令、参数），该技能会生成一个 harness：派生这些服务器、合并工具列表，并提供一个带冲突解决能力的路由函数。

## 练习

1. 运行 `code/main.py`，观察服务器派生日志。用 SIGTERM 杀死其中一个模拟服务器进程，观察客户端如何检测到 EOF 并将该会话标记为死亡。

2. 实现命名空间前缀。当两个服务器都暴露 `search` 时，将第二个重命名为 `<server>/search`。更新调度表，并验证工具调用能正确路由。

3. 为服务器重启添加连接池风格的退避：连续失败时指数退避，上限 30 秒，三次失败后向用户发送通知。

4. 草拟一个支持 100 个并发 MCP 服务器的客户端。什么数据结构会取代简单的调度字典？（提示：用于前缀命名空间的 trie，以及每个服务器工具数量的指标。）

5. 将该客户端移植到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应该从约 200 行缩减到约 40 行，同时保留多服务器路由能力。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| MCP 客户端（MCP client） | “智能体宿主” | 派生服务器并编排工具调用的进程 |
| 会话（Session） | “每台服务器的状态” | 能力、工具列表与待处理请求的簿记 |
| 合并命名空间（Merged namespace） | “一个工具列表” | 所有活跃服务器之间的扁平工具名称集合 |
| 命名空间冲突（Namespace collision） | “两个服务器有同名工具” | 客户端必须对重复项进行前缀、拒绝或先到优先处理 |
| 路由（Routing） | “这个调用给谁？” | 从工具名称分发到拥有它的服务器 |
| 后台读取器（Background reader） | “非阻塞 stdout” | 将服务器 stdout 抽到队列中的线程或任务 |
| 采样回调（Sampling callback） | “LLM 即服务” | 客户端处理来自服务器的 `sampling/createMessage` |
| `notifications/*_changed` | “原语发生变化” | 客户端必须重新发现或重新读取的信号 |
| 重连策略（Reconnection policy） | “服务器挂了怎么办” | 传输失败时的重启语义 |
| Stdio 会话（Stdio session） | “进程 = 会话” | 没有会话 id；子进程生命周期就是会话 |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) —— 规范客户端行为
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) —— 使用 Python SDK 的 hello-world 客户端教程
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) —— `ClientSession` 与 `stdio_client` 参考
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) —— TypeScript 对应版本
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) —— VS Code 如何在单一编辑器宿主中多路复用多个 MCP 服务器
