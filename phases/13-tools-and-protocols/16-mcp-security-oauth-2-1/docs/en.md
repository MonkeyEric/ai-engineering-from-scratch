# MCP 安全 II — OAuth 2.1、资源指示符（Resource Indicators）与增量作用域（Incremental Scopes）

> 远程 MCP 服务器不仅需要认证（authentication），还需要授权（authorization）。2025-11-25 版规范与 OAuth 2.1 + PKCE + 资源指示符（resource indicators，RFC 8707）+ 受保护资源元数据（protected-resource metadata，RFC 9728）对齐。SEP-835 在 403 WWW-Authenticate 上增加了增量作用域同意（incremental scope consent）与升级授权（step-up authorization）。本节课把升级流程实现为状态机，让你看清每一步跳转。

**类型：** 构建
**语言：** Python（标准库，OAuth 状态机模拟器）
**前置知识：** 第 13 阶段 · 09（传输层），第 13 阶段 · 15（安全 I）
**时长：** 约 75 分钟

## 学习目标

- 区分资源服务器（resource server）与授权服务器（authorization server）的职责。
- 走一遍受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource` 参数（RFC 8707）和受保护资源元数据（RFC 9728）防止糊涂副手（confused deputy）攻击。
- 实现升级授权（step-up authorization）：服务器返回 403 并在 WWW-Authenticate 中要求更高作用域；客户端重新提示用户同意并重试。

## 问题背景

早期 MCP（2025 年前）的远程服务器要么使用临时 API 密钥，要么干脆不做认证。2025-11-25 版规范通过完整的 OAuth 2.1 配置文件补齐了这一短板。

三个真实场景需求：

- **普通远程服务器。** 用户安装了一个访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 + PKCE 是合适的形态。
- **作用域提升。** 被授予 `notes:read` 的笔记服务器之后可能因某个操作需要 `notes:write`。与其重新走完整流程，升级授权（step-up，SEP-835）可以只申请额外的作用域。
- **防止糊涂副手攻击。** 客户端持有一个受众（audience）限定给服务器 A 的令牌。服务器 A 是恶意的，试图把该令牌出示给服务器 B。资源指示符（RFC 8707）把令牌锁定到其预定受众。

OAuth 2.1 并不新鲜。新鲜的是 MCP 的配置文件（profile）：指定了必需的流程（仅授权码 + PKCE；禁止隐式授权，默认不允许客户端凭据），每次令牌请求都必须带资源指示符，并发布受保护资源元数据让客户端知道该去哪里。

## 核心概念

### 角色

- **客户端（Client）。** MCP 客户端（如 Claude Desktop、Cursor 等）。
- **资源服务器（Resource server）。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器（Authorization server）。** 负责签发令牌。可能与资源服务器是同一个服务，也可能是独立的身份提供方（IdP），如 Auth0、Keycloak、Cognito。

在 MCP 的配置文件中，资源服务器和授权服务器**可以（CAN）**是同一主机，但**应该（SHOULD）**通过 URL 区分。

### 授权码 + PKCE

流程如下：

1. 客户端生成 `code_verifier`（随机值）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器将验证器（verifier）的哈希与存储的挑战（challenge）比对，然后签发访问令牌。
6. 客户端使用该令牌：每次向资源服务器请求都带上 `Authorization: Bearer ...`。

PKCE 防止授权码被截获的攻击。资源指示符防止令牌在其他地方生效。

### 受保护资源元数据（RFC 9728）

资源服务器发布一份 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。这减少了配置——客户端只需要资源 URL。

### 资源指示符（RFC 8707）

令牌请求中的 `resource` 参数固定了令牌的预定受众（audience）。签发的令牌包含 `aud: "https://notes.example.com"`。收到该令牌的其他 MCP 服务器会检查 `aud` 并拒绝。

### 作用域模型

作用域是空格分隔的字符串。常见的 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 表示管理员能力（谨慎使用）
- `profile:read` 表示身份信息

选择作用域应遵循最小权限原则：现在需要什么就申请什么，需要更多时再升级。

### 升级授权（SEP-835）

用户授予了 `notes:read`。之后他们要求智能体删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 `insufficient_scope` 错误，弹出同意对话框请求额外作用域，执行一次小型 OAuth 流程，然后用新令牌重试请求。

### 令牌受众校验

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配则返回 401。这阻止了跨服务器令牌复用。

### 短效令牌与轮换

访问令牌（access token）**应该（SHOULD）**是短效的（默认 1 小时）。每次刷新时刷新令牌（refresh token）都会轮换。客户端在后台处理静默刷新。

### 禁止令牌透传

采样服务器（第 13 阶段 · 11）**禁止（MUST NOT）**把客户端的令牌透传给其他服务。采样请求就是边界。

### 防止糊涂副手攻击

令牌绑定到 `aud`。客户端绑定到 `client_id`。每个请求都针对这两者进行校验。规范明确禁止了 MCP 出现前远程工具生态中常见的“传令牌”模式。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 发布自己的元数据。授权服务器可以获取该客户端元数据文档，从而发现重定向 URI 和联系信息。这省去了手动客户端注册。

### 网关与 OAuth

第 13 阶段 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭据，发给客户端的令牌由网关签发，上游令牌永远不会离开网关。这颠覆了信任模型——用户只需向网关认证一次；网关负责 N 个服务器的授权。

## 使用它

`code/main.py` 将完整的 OAuth 2.1 升级流程模拟为状态机。它实现了：

- PKCE 验证码（code-verifier）/ 挑战（challenge）生成。
- 带资源指示符的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌校验。
- `insufficient_scope` 时的升级授权。

本节课没有 HTTP 服务器；状态机在内存中运行，方便你追踪每一步跳转。第 13 阶段 · 17 的网关课程会把它接到真实传输层上。

## 交付成果

本节课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带工具的远程 MCP 服务器，该技能会设计作用域集合、固定规则（pinning rules）和升级策略（step-up policy）。

## 练习

1. 运行 `code/main.py`。追踪双作用域升级流程。注意升级时哪些步骤会重复。

2. 加入刷新令牌轮换：每次刷新都签发新的刷新令牌并作废旧的。模拟轮换后被盗的刷新令牌被使用，确认它会失败。

3. 使用标准库 `http.server` 把受保护资源元数据端点实现为真实 HTTP 响应。参考第 09 课的 `/mcp` 端点。

4. 为一个 GitHub MCP 服务器设计作用域层级：读取仓库、创建 PR、批准 PR、合并 PR、管理员。在每一级之间使用升级授权。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 与 RFC 示例用法不同的那一个字段。（提示：与 `scopes_supported` 有关。）

## 关键术语

| 术语 | 大家的说法 | 实际含义 |
|------|------------|----------|
| OAuth 2.1 | “现代 OAuth” | 合并后的 RFC，强制要求 PKCE 并禁止隐式流程 |
| PKCE | “持有证明” | 验证码 + 挑战，用于挫败授权码截获攻击 |
| 资源指示符（Resource indicator） | “令牌受众” | RFC 8707 的 `resource` 参数，将令牌锁定到单一服务器 |
| 受保护资源元数据（Protected-resource metadata） | “发现文档” | RFC 9728 的 `.well-known/oauth-protected-resource` |
| 升级授权（Step-up authorization） | “增量同意” | SEP-835 按需添加作用域的流程 |
| `insufficient_scope` | “带 WWW-Authenticate 的 403” | 服务器发出的信号，要求重新同意更大作用域 |
| 糊涂副手（Confused deputy） | “跨服务复用令牌” | 可信持有者不适当地转发令牌所导致的攻击 |
| 短效令牌（Short-lived token） | “访问令牌存活时间” | 很快过期的 Bearer 令牌；由刷新令牌续期 |
| 作用域层级（Scope hierarchy） | “最小权限栈” | 分级的作用域集合，各级之间通过升级授权过渡 |
| 客户端 ID 元数据（Client ID metadata） | “客户端发现文档” | 客户端发布自身 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP OAuth 配置文件权威文档
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更的走读
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 固定受众的 RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档的 RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用的升级流程走读
