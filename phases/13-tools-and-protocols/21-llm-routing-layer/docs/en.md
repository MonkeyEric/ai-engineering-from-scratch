# LLM 路由层 — LiteLLM、OpenRouter、Portkey

> 供应商锁定（provider lock-in）代价高昂。不同的工具调用负载适合不同的模型。路由网关（routing gateway）提供统一的 API 接口、重试、故障转移、成本追踪和护栏（guardrails）。2026 年主流方案分三类：LiteLLM（开源、可自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本节课说明选型标准，并带你实现一个基于标准库的路由网关。

**Type:** 学习
**Languages:** Python（标准库，路由 + 故障转移 + 成本追踪）
**Prerequisites:** Phase 13 · 02（function calling），Phase 13 · 17（gateways）
**Time:** 约 45 分钟

## 学习目标

- 区分自托管、托管和生产级路由方案。
- 实现按定义优先级顺序在供应商故障时重试的降级链（fallback chain）。
- 跨供应商追踪每次请求的成本与 token 使用量。
- 针对具体生产约束在 LiteLLM、OpenRouter 与 Portkey 之间做出选择。

## 问题背景

供应商路由发挥作用的场景：

1. **成本。** Claude Sonnet 的价格是 Haiku 的 3 倍。分类任务用 Haiku 就够了；综合任务才值得用 Sonnet。按请求路由。

2. **故障转移。** OpenAI 遇到糟糕的一小时，所有请求都失败。你希望无需重新部署就能自动回退到 Anthropic。

3. **延迟。** 实时聊天 UI 需要首 token 快速返回，批量摘要器则不需要。按延迟 SLA 路由。

4. **合规。** 欧盟用户必须留在欧盟区域。按地区路由。

5. **实验。** 对同一负载做两个模型的 A/B 测试。按测试分桶路由。

每次集成都手写一遍非常重复。路由网关提供 OpenAI 兼容的统一 API，其余事情由它处理。

## 核心概念

### OpenAI 兼容的代理形态

大家都使用 OpenAI 的接口形态。路由网关暴露 `/v1/chat/completions`，接收 OpenAI 的 schema，内部代理到 Anthropic / Gemini / Cohere / Ollama / 任何供应商。客户端无需关心。

### 模型别名（model aliases）

代码里不用写 `claude-3-5-sonnet-20251022`，而是写 `our_smart_model`。网关把别名映射到真实模型。当 Anthropic 发布 Claude 4 时，你只需在服务端改别名；业务代码完全不用动。

### 降级链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义这一逻辑。重试会计入预算，避免降级级联导致成本爆炸。

### 语义缓存（semantic caching）

相同或近似相同的提示词命中缓存，而不是再次调用供应商。在重复的 agent 循环中，节省幅度可达 30% 到 60%。缓存键基于嵌入（embedding）；语义相近的提示词共享同一个缓存槽。

### 护栏（Guardrails）

网关层能力：

- **PII 脱敏。** 发送前通过正则或 ML 方式进行扫描脱敏。
- **策略违规检测。** 拒绝包含违禁内容的提示词。
- **输出过滤。** 清洗补全结果，防止信息泄露。

Portkey 与 Kong 都内置了成熟的护栏。LiteLLM 则将其作为可选项。

### 按 key 限流

一个 API key 对应一个团队。按 key 的预算可以防止某个团队耗尽共享配额。大多数网关都支持。

### 自托管与托管的权衡

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产级） |
|--------|----------------------|----------------------|----------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 部署 | 部署代理 | 注册即用 | 两者皆可 |
| 供应商 | 100+ | 300+ | 100+ |
| 计费 | 你自己的 key | OpenRouter 额度 | 你自己的 key |
| 可观测性 | OpenTelemetry | 仪表盘 | 完整 OTel + PII 脱敏 |
| 最适合 | 需要完全控制的团队 | 快速原型 | 生产与合规 |

如果你有 SRE 团队并希望数据主权，选 LiteLLM。如果你想一张信用卡、零基础设施，选 OpenRouter。如果你需要开箱即用的护栏与合规，选 Portkey。

### 成本追踪

每次请求携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以各模型每 token 单价（网关维护的价格表）。再按用户 / 团队 / 项目聚合。

### MCP 与路由结合

网关既能路由 LLM 调用，也能路由 MCP sampling 请求。当 sampling 请求的 `modelPreferences` 倾向某个模型时，网关将其翻译到正确的后端。这正是 Phase 13 · 17（MCP 网关）与本节课的路由网关有时会合并为一个服务的地方。

### 路由策略

- **静态优先级。** 列表中第一个；出错时降级。
- **负载均衡。** 轮询或加权。
- **成本感知。** 选择满足延迟 / 质量要求的最便宜模型。
- **延迟感知。** 选择最近 N 分钟内最快的模型。
- **任务感知。** 提示词分类器把代码类路由到重智能的模型，把摘要类路由到重速度的模型。

## 动手实践

`code/main.py` 用约 150 行实现了一个路由网关：接收 OpenAI 形态的请求，翻译到各供应商桩（stub），执行优先级降级链，追踪每次请求成本，并在转发前对输入做一次 PII 脱敏。运行三个场景：正常请求、主供应商故障触发降级、PII 泄露被脱敏拦截。

关注重点：

- `ROUTES` 字典：alias -> 按优先级排列的具体供应商列表。
- 降级循环在 5xx 时重试。
- 成本追踪将 token 使用量乘以各模型费率。
- PII 脱敏器在转发前擦除类似 SSN 的格式。

## 交付成果

本节课产出 `outputs/skill-routing-config-designer.md`。给定负载画像（延迟、成本、合规），该技能会在 LiteLLM / OpenRouter / Portkey 中做出选择并生成路由配置。

## 练习题

1. 运行 `code/main.py`。触发故障场景；确认降级到第二个供应商，且成本被正确归属。

2. 添加语义缓存：以提示词的 SHA256 作为查找键；缓存命中立即返回。测量重复调用节省的成本。

3. 添加提示词分类器：把以 "code ..." 开头的提示词路由到偏重智能的别名，把 "summarize ..." 开头的提示词路由到偏重速度的别名。

4. 设计团队级预算：每个团队有月度消费上限；达到上限后网关拒绝请求。选择 enforcement 粒度（按请求或按窗口）。

5. 并行阅读 LiteLLM、OpenRouter 与 Portkey 的文档。指出每家独有而其他两家没有的一项功能。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 路由网关（routing gateway） | "LLM 代理" | 位于多个供应商前方的统一 API 层 |
| OpenAI 兼容 | "使用 OpenAI schema" | 接收 `/v1/chat/completions` 形态，翻译到任意后端 |
| 模型别名（model alias） | "our_smart_model" | 代码中使用的名称，由网关映射到具体模型 |
| 降级链（fallback chain） | "重试列表" | 失败时按顺序尝试的供应商列表 |
| 语义缓存（semantic caching） | "提示词嵌入缓存" | 键是提示词的嵌入；近似重复共享一次缓存命中 |
| 护栏（guardrails） | "输入/输出过滤器" | 脱敏 PII、拒绝策略违规 |
| 按 key 限流 | "团队预算" | 绑定到 API key 的配额 |
| 成本追踪 | "每次请求花费" | token 使用量 × 模型单价，再聚合 |
| LiteLLM | "开源代理" | 可自托管的开源路由网关 |
| OpenRouter | "托管 SaaS" | 按额度计费的主机网关 |
| Portkey | "生产级选项" | 开源 + 托管，内置护栏 |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — 自托管路由网关
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS
- [Portkey — docs](https://portkey.ai/docs) — 带护栏的生产路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 选型指南
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — 供应商调研
