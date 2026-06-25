# MCP 网关与注册表 —— 企业级控制平面

> 企业不能让每个开发者随意安装来源不明的 MCP 服务器。网关将认证、RBAC、审计、速率限制、缓存和工具投毒检测集中处理，然后把合并后的工具面暴露为单一 MCP 端点。官方 MCP 注册表（Official MCP Registry，由 Anthropic、GitHub、PulseMCP 与 Microsoft 共同维护，并通过命名空间验证）是规范上游。本课介绍网关在架构中的位置，带领实现一个最小化网关，并概览 2026 年的厂商格局。

**类型：** 学习
**语言：** Python（标准库，最小化网关）
**前置条件：** Phase 13 · 15（工具投毒），Phase 13 · 16（OAuth 2.1）
**时长：** 约 45 分钟

## 学习目标

- 解释 MCP 网关所处的位置（位于 MCP 客户端与多个后端 MCP 服务器之间）。
- 实现网关的五大职责：认证、RBAC、审计、速率限制、策略。
- 在网关层强制执行已固定工具哈希（pinned-tool-hash）清单。
- 区分官方 MCP 注册表与元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题背景

一家财富 500 强企业拥有 30 个已批准的 MCP 服务器、5000 名开发者、合规与审计要求，以及一个希望实现集中策略管理的安全团队。让每位开发者在其 IDE 中任意安装服务器显然不可行。

网关模式：

1. 网关以单一 Streamable HTTP 端点运行，开发者连接到这个端点。
2. 网关保存每个后端 MCP 服务器的凭据。
3. 每个开发者请求都通过网关自身的 OAuth 进行认证与作用域分配。
4. 网关将调用路由到后端服务器，并应用策略。
5. 所有调用均被记录以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway 都在 2025–2026 年间推出了网关或网关功能。

与此同时，官方 MCP 注册表作为规范上游发布：经过筛选、命名空间验证、采用反向 DNS 命名的服务器，网关可以从中拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）则聚合来自多个来源的服务器。

## 核心概念

### 网关的五大职责

1. **认证（auth）。** 使用 OAuth 2.1 识别开发者；映射到用户角色。
2. **RBAC（基于角色的访问控制）。** 每个用户的策略：允许访问哪些服务器、哪些工具、哪些作用域。
3. **审计（audit）。** 每次调用都记录谁、做了什么、何时、结果如何。
4. **速率限制（rate limit）。** 按用户 / 按工具 / 按服务器设置上限，防止滥用。
5. **策略（policy）。** 拒绝投毒描述、强制执行双步规则（Rule of Two）、脱敏个人身份信息（PII）。

### 网关作为单一端点

对开发者而言，网关看起来就是一个 MCP 服务器。其内部将请求路由到 N 个后端。会话 ID（Phase 13 · 09）在边界处被重写。

### 凭据托管（credential vaulting）

开发者永远看不到后端令牌。网关持有这些令牌（或代理到持有令牌的 identity provider）。一个只有网关层面 `notes:read` 权限的开发者，可能通过网关自身的后端凭据间接访问 notes MCP 服务器——但仅限于绑定该间接访问的策略之下。

### 在网关层固定工具哈希（tool-hash pinning）

网关维护一份已批准工具描述的清单（SHA256 哈希）。在发现阶段，它获取每个后端的 `tools/list`，将哈希与清单对比，并移除任何描述发生变化的工具。这就是在 Phase 13 · 15 中介绍的 rug-pull 防御机制的中心化应用。

### 策略即代码（policy-as-code）

高级网关使用 OPA/Rego、Kyverno 或 Styra 来表达策略。例如“用户 `alice` 只能在 `acme` 组织的仓库上调用 `github.open_pr`”这类规则以声明式方式编码。简单网关则使用手写 Python。两种形态都有效。

### 会话感知路由

当用户会话混合多个服务器时，网关进行多路复用：开发者的单一 MCP 会话持有 N 个后端会话，每个服务器一个。来自任意后端的通知都通过网关转发到开发者会话。

### 命名空间合并

网关合并所有后端的工具命名空间，通常在冲突时添加前缀。例如 `github.open_pr`、`notes.search`。这使得路由无歧义。

### 注册表

- **官方 MCP 注册表（Official MCP Registry，`registry.modelcontextprotocol.io`）。** 在 Anthropic、GitHub、PulseMCP、Microsoft 的共同管理下发布。通过命名空间验证（反向 DNS：`io.github.user/server`）。预先过滤基础质量。
- **Glama。** 以搜索为中心的元注册表，聚合多个来源。
- **MCPMarket。** 偏向商业的目录，包含厂商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 类似包管理器的安装流程。
- **LobeHub。** 集成在其 LobeChat 应用中的注册表。

企业网关默认从官方注册表拉取，允许管理员从元注册表中筛选添加，并拒绝任何未固定的服务器。

### 反向 DNS 命名

官方注册表要求公共服务器使用反向 DNS 名称：`io.github.alice/notes`。命名空间可防止抢注，并使信任委托更加清晰。

### 厂商概览，2026 年 4 月

| 厂商 | 优势 |
|--------|----------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；免费 tier |
| Kong AI Gateway | Kubernetes 原生；细粒度策略；日志输出到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | 面向 DevOps；指标优先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可定制过滤器 |

Phase 17（生产基础设施）将更深入探讨网关运维。

## 动手实践

`code/main.py` 提供了一个约 150 行的最小化网关：通过模拟的 Bearer token 认证用户、保存每个用户的 RBAC 策略、将请求路由到两个后端 MCP 服务器、把每次调用写入审计日志、强制执行速率限制，并拒绝任何描述哈希与固定清单不匹配的后端工具。

需要关注的内容：

- `RBAC` 字典以 `user_id` 为键，包含允许的 `server_tool` 项。
- `AUDIT_LOG` 是一个追加式事件列表。
- 速率限制为每个用户使用令牌桶（token bucket）。
- 固定清单是一个 `server::tool -> hash` 的字典。

## 产出交付

本课生成 `outputs/skill-gateway-bootstrap.md`。给定一份企业 MCP 方案（用户、后端、合规要求），该技能会输出一份网关配置规范。

## 练习

1. 运行 `code/main.py`。分别以被允许的用户、被禁止的用户以及触发速率限制的高频请求进行调用，验证三种流程。

2. 添加一条策略，在将结果返回客户端之前脱敏其中的 PII。可先用简单的正则匹配 SSN 格式的字符串；注意其局限性（邮箱、手机号等）。

3. 扩展审计日志，使其发出 OpenTelemetry GenAI span。Phase 13 · 20 会介绍具体属性。

4. 为 50 名开发者、五个后端（notes、github、postgres、jira、slack）设计一份 RBAC 策略。谁对每个后端只读？谁有写入权限？

5. 完整阅读 Cloudflare 的企业 MCP 文章。找出 Cloudflare 提供但本标准库网关未实现的一项功能。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------------|------------------------|
| Gateway（网关） | "MCP 代理" | 位于客户端与后端之间的集中式服务器 |
| Credential vaulting（凭据托管） | "后端令牌保留在服务端" | 开发者永远看不到上游令牌 |
| Session-aware routing（会话感知路由） | "多后端会话" | 网关为每个开发者会话多路复用 N 个后端会话 |
| Tool-hash pinning（工具哈希固定） | "已批准清单" | 每个已批准工具描述的 SHA256；集中阻止 rug-pull |
| RBAC（基于角色的访问控制） | "每个用户的策略" | 针对工具和服务器的角色访问控制 |
| Policy-as-code（策略即代码） | "声明式规则" | 在网关强制执行的 OPA/Rego、Kyverno、Styra 策略 |
| Audit log（审计日志） | "谁、做什么、何时" | 用于合规的追加式事件日志 |
| Rate limit（速率限制） | "每个用户的令牌桶" | 每分钟上限，防止滥用 |
| Official MCP Registry（官方 MCP 注册表） | "规范上游" | `registry.modelcontextprotocol.io`，命名空间验证 |
| Reverse-DNS naming（反向 DNS 命名） | "注册表命名空间" | `io.github.user/server` 约定 |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) —— 规范上游，命名空间验证
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) —— 网关模式，含 OAuth 与策略
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) —— 开源参考网关
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) —— 功能对比文章
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) —— IBM 企业级网关
