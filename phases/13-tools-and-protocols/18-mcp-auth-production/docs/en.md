# 生产环境 MCP 认证 —— 基于 iii 原语的 DCR、JWKS 轮换与受众固定令牌

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你交付给真实组织的每个 MCP 服务器背后都要有生产级认证：动态客户端注册（DCR，RFC 7591）、授权服务器元数据发现（RFC 8414）、不会在凌晨 3 点打断令牌验证的 JWKS 密钥轮换，以及拒绝混淆副手（confused-deputy）重用的受众固定令牌。本课将上述能力全部通过 iii 原语串联起来 —— 用 `iii.registerTrigger` 注册 HTTP 触发器与定时触发器，用 `iii.registerFunction` 承载认证逻辑，用 `state::set/get` 缓存密钥 —— 从而使认证面与引擎中的其他工作负载一样可观测、可重启、可回放。

**类型：** 实战构建
**语言：** Python（标准库，iii 原语以模拟形式嵌入课程环境）
**前置知识：** Phase 13 · 第 16 课（OAuth 2.1 状态机）、Phase 13 · 第 17 课（网关）
**时长：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并校验其合约。
- 实现 RFC 7591 动态客户端注册（DCR），让 MCP 客户端无需管理员介入即可注册。
- 使用定时触发器缓存并轮换 JWKS 密钥，使签名验证在密钥滚动期间仍然可用。
- 通过 RFC 8707 资源指示符将令牌固定到单一 MCP 资源，并拒绝混淆副手重用。
- 将每个端点与后台任务都实现为 iii 原语 —— HTTP 触发器、定时触发器、命名函数以及 `state::*` 读写 —— 从而一次重启即可重建整个认证面。
- 阅读 IdP 能力矩阵，当 IdP 无法满足 MCP 认证配置时拒绝部署。

## 问题背景

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境存在三个纯内存模拟器无法暴露的运维缺口。

第一个缺口是注册。真实组织运行着数百个 MCP 服务器和数千个 MCP 客户端。运维人员不可能为每个 Cursor 用户手动注册成 OAuth 客户端。RFC 7591 动态客户端注册（DCR）允许客户端向授权服务器 `POST /register`，并当场获得 `client_id`（以及可选的 `client_secret`）。服务器在 RFC 8414 元数据中公布 `registration_endpoint`；客户端无需带外配置即可发现它。

第二个缺口是密钥轮换。JWT 验证依赖授权服务器发布的签名密钥，即 JSON Web 密钥集（JWKS）。授权服务器按固定计划轮换这些密钥（通常每小时一次，应急响应时可能更快）。一个在启动时只拉取一次 JWKS 的 MCP 服务器，在轮换窗口前都能正常验证 —— 之后所有请求都会失败，直到重启。生产环境应将 JWKS 作为带缓存的值，并配套一个刷新任务，在前一批密钥过期前覆写缓存；同时还要在缓存未命中时同步回源拉取，以处理令牌由比缓存更新的密钥签名的场景。

第三个缺口是受众绑定。第 16 课介绍了 RFC 8707 资源指示符。在生产环境中，该指示符会变成每次请求的硬性声明检查。MCP 服务器将 `token.aud` 与自己的规范资源 URL 比对，不匹配者返回 HTTP 401。这是抵御上游 MCP 服务器（或持有本应发给某服务器的令牌的恶意客户端）在同一信任网格内将令牌重放到另一服务器的唯一协议层防线。

本课将这三个缺口都视为 iii 原语。元数据文档是一个返回函数输出的 HTTP 触发器。JWKS 轮换是一个调用 `auth::rotate-jwks` 的定时触发器，后者写入 `state::set("auth/jwks/<issuer>", ...)`。JWT 验证是一个可被其他代码通过 `iii.trigger("auth::validate-jwt", token)` 调用的函数。MCP 服务器本身也不过是另一个在分发前先调用验证的 HTTP 触发器。重启引擎：触发器注册表重建；状态持久保留；认证面自动恢复，无需人工对账。

## 核心概念

### RFC 8414 —— OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

拿到 MCP 资源 URL 的客户端会串联发现：先通过 RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）拿到 issuer，再通过本 RFC 的 `oauth-authorization-server` 拿到所有端点。客户端绝不硬编码授权 URL。

在为 MCP 信任某个 IdP 之前，必须校验以下合约：

- `code_challenge_methods_supported` 包含 `S256`（符合 RFC 7636 的 PKCE）。
- `grant_types_supported` 包含 `authorization_code`，且拒绝 `password` 与 `implicit`。
- `registration_endpoint` 存在（支持 RFC 7591）。
- `response_types_supported` 对 OAuth 2.1 而言必须恰好是 `["code"]`。

若其中任何一项缺失，MCP 服务器将拒绝针对该 IdP 部署。错的是部署清单，而不是代码。

### RFC 9728（回顾）—— 受保护资源元数据

第 16 课已涵盖 RFC 9728。在生产中的关键变化：这份文档是客户端查找*本* MCP 服务器所信任授权服务器的唯一位置。单个 MCP 服务器可能接受来自多个 IdP 的令牌（一个给员工，一个给合作伙伴）。RFC 9728 声明该集合；RFC 8414 则记录每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 —— 动态客户端注册

没有 DCR 时，每个 MCP 客户端（Cursor、Claude Desktop、自定义智能体）都需要与 IdP 管理员进行带外交换。有了 DCR，客户端只需提交：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器返回 `client_id` 以及用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

对于运行在用户设备上的 MCP 客户端，`token_endpoint_auth_method: none` 是正确的默认值。它们只获得 `client_id`，没有可泄露的 `client_secret`。PKCE 提供了公共客户端所需的持有证明（proof-of-possession）。

三个生产陷阱：

- 注册端点必须按源 IP 做速率限制。否则攻击者可以脚本化海量虚假注册，耗尽 `client_id` 命名空间。iii 让这变得简单：注册 HTTP 触发器在交给注册器前先调用 `auth::rate-limit` 函数。
- 部分企业 IdP 要求 `software_statement`（一份为客户端背书的签名 JWT）。本课的模拟跳过它；生产环境应接入验证步骤，拒绝来自非 localhost 重定向 URI 的未签名注册。
- `registration_access_token` 必须以哈希形式存储，而非明文。该令牌一旦泄露，攻击者就能重写客户端重定向 URI。

### RFC 8707（回顾）—— 资源指示符

第 16 课已确立其形态。生产规则：每次令牌请求都携带 `resource=<canonical-mcp-url>`，且 MCP 服务器在每次调用时校验 `token.aud` 是否匹配自身资源 URL。若 MCP 服务器可通过 `https://notes.example.com/mcp` 访问，则规范 URL 为 `https://notes.example.com` —— 路径部分被排除，以便单个服务器在单一受众下托管多条路径。

### RFC 7636（回顾）—— PKCE

PKCE 在 OAuth 2.1 中是强制性的。本课的授权码流程始终携带 `code_challenge` 与 `code_verifier`。服务器会拒绝任何没有 verifier 或 verifier 哈希结果与存储 challenge 不符的令牌请求。

### MCP 规范 2025-11-25 认证配置

MCP 规范（2025-11-25）对 MCP 服务器的授权层必须做什么有精确要求：

- 发布 `/.well-known/oauth-protected-resource`（RFC 9728）。
- 仅通过 `Authorization: Bearer ...` 接受令牌。
- 每次请求都校验 `aud`、`iss`、`exp` 以及必需的作用域（scope）。
- 对每个 401 和 403 响应返回携带 `Bearer error=...` 的 `WWW-Authenticate` 头，必要时包含 `scope=` 与 `resource=` 参数。
- 拒绝 `aud` 与规范资源不匹配的令牌。
- 拒绝 `iss` 不在受保护资源元数据 `authorization_servers` 列表中的令牌。

OAuth 2.1 草案是底层协议；RFC 8414/7591/8707/9728 加上 RFC 7636 是外表面；MCP 规范则是具体配置。

### IdP 能力矩阵

并非每个 IdP 都支持完整 MCP 配置。下表记录了截至 2025-11-25 规范的事实能力声明。它是*部署关卡*，而非推荐。

| IdP 类别 | RFC 8414 元数据 | RFC 7591 DCR | RFC 8707 资源 | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|
| 自建（Keycloak） | 支持 | 支持 | 支持（自 24.x） | 支持 | 本课 MCP 配置的参考 IdP；端到端支持所有 RFC。 |
| 企业 SSO（Microsoft Entra ID） | 支持 | 支持（高级租户） | 支持 | 支持 | DCR 可用性因租户级别而异；部署前请在目标租户中确认。 |
| 企业 SSO（Okta） | 支持 | 支持（Okta CIC / Auth0） | 支持 | 支持 | Auth0（现 Okta CIC）支持 DCR；经典 Okta org 需要管理员预注册。 |
| 社交登录 IdP（通用） | 不一 | 很少 | 很少 | 支持 | 多数社交 IdP 将客户端视为静态合作伙伴；不要依赖 DCR。仅作为身份源使用，并在上层叠加一个 MCP 感知的授权服务器。 |
| 定制 / 自研 | 视情况而定 | 视情况而定 | 视情况而定 | 视情况而定 | 如果你自研，必须完整实现上述配置。跳过四项 RFC 中任意一项都会破坏 MCP 认证合约。 |

部署清单的拒绝规则：如果所选 IdP 不返回 `registration_endpoint`，或未在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器拒绝启动。不存在降级模式。

### 基于 iii 的 JWKS 轮换模式

生产故障的典型模式是 JWKS 缓存过期。用定时触发器加 `state::*` 缓存解决：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时，定时触发器调用 `auth::rotate-jwks`，后者从 `<issuer>/.well-known/jwks.json` 拉取密钥并写入 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器从 `state::get` 读取。若令牌 `kid` 在缓存中缺失，会触发一次同步的 `auth::rotate-jwks` 作为回退。这同时覆盖两种场景：计划轮换（定时触发器）与密钥重叠窗口（同步回退）。

状态结构：

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时保留两把密钥是稳态。授权服务器轮换时会先引入下一密钥（`k_2026_04`），再退役上一密钥（`k_2026_03`），因此用旧密钥签发的令牌在过期前仍然有效。缓存保存它们的并集；验证器按 `kid` 选择。

### iii 原语接线（本课真正要讲的部分）

五个原语组合成认证面：

```python
# 1. RFC 8414 元数据文档
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 动态客户端注册
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT 验证作为可调用函数（资源服务器触发它）
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. 增量作用域的分步签发（来自第 16 课的 SEP-835）
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. 定时驱动的 JWKS 轮换
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP 服务器本身从不直接调用验证。它这样做：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这种间接是 iii 的核心赌注。明天你可以把验证器换成并行查询两个 IdP 的扇出，或增加 span 发射，或缓存正向验证结果。MCP 服务器无需改动。

### 受众绑定下的混淆副手攻击演练

服务器 A（`notes.example.com`）与服务器 B（`tasks.example.com`）都向同一授权服务器注册。服务器 A 被攻破。攻击者拿到用户的 notes 令牌并重放到服务器 B。

服务器 B 的验证器：

1. 解码 JWT，按 `kid` 取 JWKS，验证签名。
2. 检查 `iss` 是否在其受保护资源元数据的 `authorization_servers` 中。（通过 —— 同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败 —— 令牌 `aud` 是 `https://notes.example.com`。）
4. 返回 401，并附带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

受众声明是协议层抵御该攻击的唯一防线。为了性能而跳过它是最常见的生产错误；验证器必须在每次请求时都运行，而不是仅在会话开始时运行。

### 故障模式

- **JWKS 过期。** 密钥轮换后，验证器拒绝有效令牌。修复方案即上述定时+回退模式。绝不要只缓存 JWKS 而不配刷新任务。
- **缺失 `aud` 声明。** 部分 IdP 默认在令牌请求缺少 `resource` 时省略 `aud`。验证器必须拒绝缺失 `aud` 的令牌，而不能把缺失视为通配符。
- **作用域升级竞态。** 同一用户的两个并发 step-up 流都可能成功，产生两个作用域不同的访问令牌。验证器必须使用请求上呈现的令牌，而不是去查“用户当前作用域” —— 后者会引入 TOCTOU 窗口。
- **注册令牌泄露。** 泄露的 `registration_access_token` 可让攻击者重写重定向 URI。静态存储时要做哈希；客户端每次更新都需出示明文；一旦怀疑泄露立即轮换。
- **`iss` 未固定。** 接受任意 `iss` 的验证器会让攻击者架设自己的授权服务器，为目标受众注册客户端并签发令牌。受保护资源元数据中的 `authorization_servers` 列表就是允许列表；必须强制执行。

## 动手实践

`code/main.py` 用标准库 Python 和一个小型 `iii_mock` 注册表完整演练生产流程，模拟 `iii.registerFunction`、`iii.registerTrigger`、`iii.trigger` 和 `state::set/get`。流程如下：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点，发现注册端点。
3. MCP 客户端向 `/register` 提交（RFC 7591）并获得 `client_id`。
4. MCP 客户端使用携带 `resource` 指示符（RFC 8707）的 PKCE 保护授权码流程（RFC 7636）。
5. MCP 客户端用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器触发 `auth::validate-jwt`，后者从 `state::get` 读取 JWKS。
7. 定时触发器触发 `auth::rotate-jwks`，替换状态中的 JWKS。
8. 下一次调用即可用新密钥验证，无需重启。
9. 针对另一 MCP 资源的混淆副手尝试会收到受众不匹配 401。

本课的模拟 JWT 使用 HS256 与共享密钥（因此仅凭标准库即可运行）。生产环境使用 RS256 或 EdDSA 配合上述 JWKS 模式；验证逻辑本身完全一致。

## 交付产出

本课将生成 `outputs/skill-mcp-auth-iii.md`。给定 MCP 服务器配置与 IdP 能力集合后，该技能会输出需要注册的 iii 原语、JWKS 轮换计划、作用域映射，以及当 IdP 不支持完整 RFC 配置时应应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。跟踪上述 9 步流程。注意 `state::get` 在 `auth::rotate-jwks` 覆写前返回过期数据的位置，以及下一次请求如何基于新密钥通过验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加一个新的 IdP。签发由新 IdP 签名的令牌并确认验证器接受；再签发一个未列出 IdP 签名的令牌并确认验证器返回 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 将 `auth::rate-limit` 实现为一个 iii 函数，并在注册 HTTP 触发器内部、注册器运行前调用它。使用每个源 IP 的令牌桶（token bucket），保存在 `state::set("auth/ratelimit/<ip>", ...)` 中。

4. 阅读 RFC 7591，找出本课 `/register` 处理程序未校验的两个字段并加上校验。（提示：`software_statement` 与 `redirect_uris` 的 URI scheme。）

5. 阅读 MCP 规范 2025-11-25 授权章节。找出本课验证器目前未发出的关于 `WWW-Authenticate` 头的唯一规范性要求，并将其加入。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "自助客户端注册" | RFC 7591 `POST /register` 流程 |
| JWKS | "JWT 验证用的公钥" | JSON Web 密钥集（JSON Web Key Set），从 `jwks_uri` 拉取，按 `kid` 索引 |
| Resource indicator | "受众参数" | RFC 8707 `resource` 参数，将令牌固定到单一服务器 |
| `aud` claim | "受众" | JWT 声明，验证器将其与规范资源 URL 比对 |
| Confused deputy | "令牌重放" | 攻击者把发给服务器 A 的令牌拿到服务器 B 出示 |
| `iss` allow-list | "可信授权服务器" | 受保护资源元数据 `authorization_servers` 中列出的集合 |
| Key rotation | "滚动 JWKS" | 在重叠窗口内周期性地替换签名密钥 |
| Public client | "原生或浏览器客户端" | 没有 `client_secret` 的 OAuth 客户端；PKCE 作为补偿 |
| `WWW-Authenticate` | "401/403 响应头" | 携带 `Bearer error=...` 指令，驱动客户端恢复 |

## 延伸阅读

- [MCP —— Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) —— 本课实现的 MCP 认证配置
- [RFC 8414 —— OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) —— 发现合约
- [RFC 7591 —— OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) —— DCR
- [RFC 7636 —— Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) —— 公共客户端持有证明
- [RFC 8707 —— Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) —— 受众固定
- [RFC 9728 —— OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) —— 资源服务器发现
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) —— 整合后的 OAuth 底层协议
