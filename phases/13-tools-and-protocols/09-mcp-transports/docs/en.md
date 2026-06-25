# MCP 传输层 —— stdio、Streamable HTTP 与 SSE 迁移

> stdio 只适用于本地。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输层已被弃用，并将在 2026 年中期移除。选错传输层意味着后续要迁移；选对传输层则能得到一个支持远程托管、会话保持和 DNS 重绑防护的 MCP 服务器。

**类型：** 学习  
**语言：** Python（标准库，Streamable HTTP 端点骨架）  
**前置条件：** Phase 13 · 07、08（MCP 服务器与客户端）  
**时长：** 约 45 分钟

## 学习目标

- 根据部署形态（本地 vs 远程、单进程 vs 集群）在 stdio 与 Streamable HTTP 之间做出选择。
- 实现 Streamable HTTP 单端点模式：POST 用于请求，GET 用于会话流。
- 强制执行 `Origin` 校验和会话 ID 语义，以防御 DNS 重绑攻击。
- 在 2026 年中期移除截止期限前，将旧版 HTTP+SSE 服务器迁移到 Streamable HTTP。

## 问题背景

最早的 MCP 远程传输层（2024-11）是 HTTP+SSE：两个端点，一个用于客户端 POST，另一个 Server-Sent-Events 通道用于服务器到客户端的流。它能跑。但也很笨拙：每个会话两个端点，某些 CDN 前的缓存会出问题，并且严重依赖长期存活的 SSE 连接，而某些 WAF 会激进地终止这些连接。

2025-03-26 的规范用 Streamable HTTP 取而代之：一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 头部。此后构建或迁移的每个服务器都使用 Streamable HTTP。旧的 SSE 模式正在被弃用 —— Atlassian Rovo 于 2026 年 6 月 30 日移除；Keboola 于 2026 年 4 月 1 日移除；大多数剩余的企业服务器将在 2026 年底前移除。

stdio 对于本地服务器仍然重要。Claude Desktop、VS Code 以及所有 IDE 形态的客户端都通过 stdio 派生服务器。正确的思维模型是：stdio 用于“本机”，Streamable HTTP 用于“跨网络”。不要混用。

## 核心概念

### stdio

- 子进程传输层。客户端派生服务器，通过 stdin/stdout 通信。
- 每行一个 JSON 对象，换行分隔。
- 没有会话 ID；进程身份就是会话。
- 无需认证（子进程继承父进程的信任边界）。
- 永远不要用于远程服务器 —— 如果要用 SSH 或 socat 做隧道，那不如直接用 Streamable HTTP。

### Streamable HTTP

单个端点 `/mcp`（或任意路径）。支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送 JSON-RPC 消息。服务器返回单个 JSON 响应，或者一个 SSE 流（包含一个或多个响应，适用于批量响应和与该请求相关的通知）。
- **GET /mcp。** 客户端打开一个长期存活的 SSE 通道。服务器用它向客户端发送请求（sampling、通知、elicitation）。
- **DELETE /mcp。** 客户端显式终止会话。

会话由 `Mcp-Session-Id` 头部标识，服务器在首次响应时设置该头部，客户端在后续每次请求时回传。会话 ID 必须是加密随机的（128 位以上）；出于安全考虑，客户端选择的 ID 会被拒绝。

### 单端点 vs 双端点

旧规范的双端点模式在 2026 年仍然可调用 —— 规范将其声明为“兼容遗留”。但所有新服务器都应为单端点。官方 SDK 输出单端点；仅当与未迁移的远程服务器通信时才使用遗留模式。

### `Origin` 校验与 DNS 重绑

浏览器（目前）不是 MCP 客户端，但攻击者可以构造一个网页，诱使浏览器向 `localhost:1234/mcp` 发起 POST —— 而用户的本地 MCP 服务器正好监听该地址。如果服务器不检查 `Origin`，浏览器的同源策略救不了它，因为 `Origin: http://evil.com` 是合法的跨域请求。

2025-11-25 规范要求服务器拒绝 `Origin` 不在白名单中的请求。白名单通常包含 MCP 客户端主机（`https://claude.ai`、`vscode-webview://*`）以及本地 UI 的 localhost 变体。

### 会话 ID 生命周期

1. 客户端首次请求不带 `Mcp-Session-Id`。
2. 服务器分配一个随机 ID，并在响应头部设置 `Mcp-Session-Id`。
3. 客户端在后续所有请求以及用于获取流的 `GET /mcp` 上回传该头部。
4. 服务器可以撤销会话；客户端在后续请求中看到 404，必须重新初始化。
5. 客户端可以显式 DELETE 会话，实现干净关闭。

### 保活与重连

SSE 连接会断开。客户端通过使用相同的 `Mcp-Session-Id` 重新 GET 来重建连接。服务器必须将中断期间错过的事件排队（在合理时间窗口内），并通过客户端回传的 `last-event-id` 头部进行重放。

Phase 13 · 13 将介绍 Tasks，它能让长时间运行的任务即使在完整会话重连后也能存活。

### 向后兼容探测

希望同时支持新旧服务器的客户端可以这样做：

1. 向 `/mcp` 发送 POST。
2. 如果响应是 `200 OK` 且为 JSON 或 SSE，则是 Streamable HTTP。
3. 如果响应是 `200 OK`、`Content-Type: text/event-stream` 并且带有指向二级端点的 `Location` 头部，则是旧版 HTTP+SSE；跟随 `Location`。

### Cloudflare、ngrok 与托管

2026 年的生产级远程 MCP 服务器运行在 Cloudflare Workers（配合其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 上。关键：你的托管环境必须支持 SSE GET 的长期 HTTP 连接。Vercel 免费版限制 10 秒，不适用。Cloudflare Workers 支持无限流。

### 网关组合

当你用网关（Phase 13 · 17）代理多个 MCP 服务器时，网关是一个单独的 Streamable HTTP 端点，负责重写会话 ID 并向上游多路复用。工具在网关层合并；客户端看到的是单一逻辑服务器。

### 传输层故障模式

- **stdio SIGPIPE。** 子进程在写入中途死亡会触发 SIGPIPE；服务器应干净退出。客户端应检测 EOF 并将会话标记为死亡。
- **HTTP 502 / 504。** Cloudflare、nginx 和其他代理在上游故障时返回这些状态码。Streamable HTTP 客户端应在短暂退避后重试一次。
- **SSE 连接断开。** TCP RST、代理超时或客户端网络变更都会关闭流。客户端使用 `Mcp-Session-Id` 和可选的 `last-event-id` 重新连接以恢复。
- **会话撤销。** 服务器使会话 ID 失效；客户端在下次请求时看到 404。客户端必须重新握手。
- **时钟偏移。** 客户端的资源 TTL 计算可能与服务器不一致。客户端应将服务器时间戳视为权威。

### 何时绕过 Streamable HTTP

某些企业在自有网络内部通过 gRPC 或消息队列传输层部署 MCP 服务器。这是非标准的 —— MCP 规范未正式定义这些。网关可以向 MCP 客户端暴露 Streamable HTTP 表面，同时在内部使用 gRPC。保持外部表面符合规范；网关拥有翻译职责。

## 动手实践

`code/main.py` 使用 `http.server`（标准库）实现了一个最小化的 Streamable HTTP 端点。它处理 `/mcp` 上的 POST、GET 和 DELETE，在首次响应时设置 `Mcp-Session-Id`，校验 `Origin`，并拒绝非白名单来源的请求。该处理程序复用了 Lesson 07 笔记服务器的分发逻辑。

观察重点：

- POST 处理程序读取 JSON-RPC 请求体、分发，并写入 JSON 响应（单响应变体；SSE 变体结构类似）。
- `Origin` 检查会拒绝默认的 `http://evil.example` 探测，但接受 `http://localhost`。
- 会话 ID 是随机的 128 位十六进制字符串；服务器在内存中保存每个会话的状态。

## 交付成果

本课产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（遗留）MCP 服务器，该技能会生成一份迁移计划，涵盖迁移到 Streamable HTTP、会话 ID 连续性、Origin 检查和向后兼容探测支持。

## 练习题

1. 运行 `code/main.py`。用 `curl` POST 一个 `initialize`，观察响应头中的 `Mcp-Session-Id`。POST 第二个请求并回传该头部，验证会话连续性。

2. 添加一个打开 SSE 流的 GET 处理程序。每五秒发送一个 `notifications/progress` 事件。使用相同的会话 ID 重新 GET 进行重连，确认服务器接受。

3. 实现 `last-event-id` 重放逻辑。重连时，重放自该 ID 以来生成的所有事件。

4. 扩展 `Origin` 校验以支持通配符模式（`https://*.example.com`），并确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表选取一个遗留 HTTP+SSE 服务器，草拟迁移方案：端点处理、会话 ID 生成和头部语义分别需要哪些改动。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| stdio transport | “本地子进程” | 基于 stdin/stdout 的 JSON-RPC，换行分隔 |
| Streamable HTTP | “远程传输层” | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | “遗留方案” | 双端点模型，将于 2026 年中期移除 |
| `Mcp-Session-Id` | “会话头部” | 服务器分配的随机 ID，客户端在后续每次请求中回传 |
| `Origin` allowlist | “DNS 重绑防御” | 拒绝 Origin 未经批准的请求 |
| Single endpoint | “一个 URL” | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | “SSE 重放” | 客户端用于恢复断开的流而不漏事件的头部 |
| Backwards-compat probe | “新旧检测” | 客户端根据响应形态自动选择传输层 |
| Long-lived HTTP | “SSE 流” | 服务器在一条 TCP 连接上推送事件数分钟甚至数小时 |
| Session revocation | “强制重新初始化” | 服务器使会话 ID 失效；客户端必须重新握手 |

## 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) —— stdio 与 Streamable HTTP 的权威参考
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) —— 引入 Streamable HTTP 的修订版本
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) —— Workers 托管的 Streamable HTTP 模式
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) —— 不同部署形态对比
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) —— 具体迁移截止期限示例
