# 毕业设计 —— 构建完整的工具生态系统

> 第 13 阶段讲授了每一块内容。本毕业设计将它们连接成一个生产级形态的系统：一个包含工具、资源、提示、任务和 UI 的 MCP 服务器，边缘侧的 OAuth 2.1，RBAC 网关，多服务器客户端，A2A 子代理调用，接入收集器的 OTel 链路追踪，CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 打包。到末尾时，你能够捍卫每一个架构选择。

**类型：** 构建
**语言：** Python（标准库，端到端生态系统承载示例）
**先决条件：** 第 13 阶段 · 01 至 21
**时间：** 约 120 分钟

## 学习目标

- 组合一个 MCP 服务器，暴露工具、资源、提示，以及一个带有 `ui://` 应用的任务。
- 在服务器前方放置一个 OAuth 2.1 网关，执行 RBAC 并校验固定哈希。
- 编写一个多服务器客户端，端到端使用 OTel GenAI 属性进行追踪。
- 将部分工作负载委派给 A2A 子代理；验证不透明性得以保持。
- 使用 AGENTS.md + SKILL.md 打包整个技术栈，使其他代理可以驱动它。

## 问题

交付“研究与报告”系统：

- 用户提问：“总结 2026 年关于代理协议的被引用最多的三篇 arXiv 论文。”
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要工作委派给专门的写手代理；汇总结果；以 MCP Apps `ui://` 资源形式渲染交互式报告；将每一步记录到 OTel。

第 13 阶段的所有原语都会出现。这不是玩具——Anthropic（Claude Research 产品）、OpenAI（搭载 Apps SDK 的 GPT）以及第三方在 2026 年交付的生产级研究助手系统正是这个形态。

## 概念

### 架构

```
[用户] -> [客户端] -> [网关 (OAuth 2.1 + RBAC)] -> [研究 MCP 服务器]
                                                      |
                                                      +- MCP 工具：arxiv_search（纯工具）
                                                      +- MCP 资源：notes://recent
                                                      +- MCP 提示：/research_topic
                                                      +- MCP 任务：generate_report（长任务）
                                                      +- MCP Apps UI：ui://report/current
                                                      +- A2A 调用：writer-agent（tasks/send）
                                                      |
                                                      +- OTel GenAI 跨度
```

### 追踪层级

```
agent.invoke_agent
 ├── llm.chat （启动）
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions （内部不透明）
 ├── mcp.call -> tools/call generate_report （任务增强）
 │    └── tasks/status polling
 │    └── tasks/result （已完成，返回 ui:// 资源）
 └── llm.chat （最终综合）
```

一个追踪 ID。每个跨度都具有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，并通过资源指示符将受众（audience）固定到网关。
- 网关持有上游凭证；用户永远不会看到它们。
- RBAC：`alice` 拥有 `research:read`、`research:write`，可以调用所有工具。`bob` 只有 `research:read`，不能调用 `generate_report`。
- 固定的描述清单：若某服务器的工具哈希发生变化，则将其丢弃。
- 双因素审计（Rule of Two）：任何工具不得同时组合不受信任的输入、敏感数据和具有重大影响的操作。

### 渲染

最终的 `generate_report` 任务返回内容块以及一个 `ui://report/current` 资源。客户端宿主（如 Claude Desktop）在沙箱 iframe 中渲染交互式仪表盘。仪表盘包含排序后的论文列表、引用次数，以及一个按钮；当用户点击某篇论文时，会调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个系统以下列结构交付：

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶层工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                     # A2A 代理
  gateway/
    config.yaml                 # RBAC + 固定清单
```

用户通过 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` 技能来驱动该系统。

### 第 13 阶段各课时的贡献

| 课时 | 毕业设计使用的部分 |
|------|------------------------|
| 01-05 | 工具接口、提供方可移植性、并行调用、模式、静态检查 |
| 06-10 | MCP 原语、服务器、客户端、传输、资源 + 提示 |
| 11-14 | 采样、根目录与引导、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子代理委派 |
| 19 | OTel GenAI 链路追踪 |
| 20 | LLM 层路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 使用它

`code/main.py` 将之前课时的模式拼接成一个可运行的演示。全部使用标准库、全部在进程内运行，因此你可以端到端地阅读。它针对“研究与报告”场景运行完整流程：与网关握手、模拟 OAuth 2.1、合并 `tools/list`、将 `generate_report` 作为任务执行、A2A 调用写手、返回 `ui://` 资源、发出 OTel 跨度。

关注要点：

- 每个跃点共享同一个追踪 ID。
- 网关策略阻止第二个用户执行写操作。
- 任务生命周期从运行中变为已完成，并同时返回文本和 `ui://` 内容。
- A2A 调用的内部状态对编排器不可见。
- `AGENTS.md` 和 `SKILL.md` 是其他代理复现该工作流所需的唯一文件。

## 交付它

本课时生成 `outputs/skill-ecosystem-blueprint.md`。给定产品需求（研究、摘要、自动化），该技能产出完整架构：使用哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、以及怎样的打包方式。

## 练习

1. 运行 `code/main.py`。注意单个追踪 ID 以及跨度如何嵌套。统计该演示触及了第 13 阶段的多少个原语。

2. 扩展演示：添加第二个后端 MCP 服务器（例如 `bibliography`），并确认网关将其工具合并到同一名称空间。

3. 将模拟的 A2A 写手代理替换为在子进程中运行的真实代理。使用第 19 课时的测试架。

4. 在编排器与 LLM 之间的路由网关中添加 PII 脱敏步骤。确认用户查询中的电子邮件会被擦除。

5. 为将要维护该系统的队友编写一份 `AGENTS.md`。阅读时间应控制在五分钟内，并让他们掌握在 Cursor 或 Codex 中驱动本毕业设计所需的全部信息。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 毕业设计（Capstone） | “第 13 阶段集成演示” | 使用每一个原语的端到端系统 |
| 研究与报告（Research and report） | “场景” | 搜索、摘要、渲染模式 |
| 生态系统（Ecosystem） | “所有部分合在一起” | 服务器 + 客户端 + 网关 + 子代理 + 遥测 + 打包 |
| 追踪层级（Trace hierarchy） | “单个追踪 ID” | 每个跃点的跨度共享追踪；通过 span id 建立父子关系 |
| 网关签发令牌（Gateway-issued token） | “传递式授权” | 客户端只看到网关的令牌；网关持有上游凭证 |
| 合并命名空间（Merged namespace） | “所有工具在一个扁平列表中” | 在网关处合并多服务器工具，冲突时加前缀 |
| 不透明边界（Opacity boundary） | “A2A 调用隐藏内部” | 子代理的推理对编排器不可见 |
| 三层栈（Three-layer stack） | “AGENTS.md + SKILL.md + MCP” | 项目上下文 + 工作流 + 工具 |
| 纵深防御（Defense-in-depth） | “多层安全” | 固定哈希、OAuth、RBAC、双因素审计、审计日志 |
| 规范合规矩阵（Spec compliance matrix） | “我们交付的规范所要求的内容” | 将交付物映射到 2025-11-25 要求的检查清单 |

## 延伸阅读

- [MCP — 2025-11-25 规范](https://modelcontextprotocol.io/specification/2025-11-25) — 整合参考
- [MCP 博客 — 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 标准追踪约定
- [Anthropic — Claude Agent SDK 概览](https://code.claude.com/docs/en/agent-sdk/overview) — 生产级代理运行时模式
