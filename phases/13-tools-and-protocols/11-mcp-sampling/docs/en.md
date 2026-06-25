# MCP Sampling — 服务器请求的 LLM 补全与智能体循环

> 大多数 MCP 服务器只是“笨执行器”：接收参数、运行代码、返回内容。采样（sampling）让服务器翻转方向：它请求客户端的大语言模型（LLM）做出决策。这让服务器能在不持有任何模型凭证（model credentials）的情况下托管智能体循环（agent loops）。SEP-1577 于 2025-11-25 合并，在采样请求中加入了工具，从而可在循环中引入更深入的推理。漂移风险说明：SEP-1577 的“采样内工具”形态在 2026 年第一季度前仍处于实验阶段，SDK API 仍在定型。

**Type:** 构建
**Languages:** Python（标准库、采样骨架）
**Prerequisites:** Phase 13 · 07（MCP server）、Phase 13 · 10（resources 与 prompts）
**Time:** ~75 分钟

## Learning Objectives

- 解释 `sampling/createMessage` 解决的问题（服务器托管循环而无需服务器端 API 密钥（API key））。
- 实现一个服务器，向客户端请求对多轮提示词（multi-turn prompt）采样并返回补全。
- 使用 `modelPreferences`（成本 / 速度 / 智能优先级）引导客户端模型选择。
- 构建 `summarize_repo` 工具，它在内部通过采样迭代而非硬编码行为。

# The Problem

一个用于代码摘要工作流的有用 MCP 服务器需要：遍历文件树、选择要读取的文件、合成摘要并返回。那么大语言模型（LLM）的推理应该放在哪里？

选项 A：服务器自己调用大语言模型。需要 API 密钥（API key）、按服务器端计费、对每位用户来说都很昂贵。

选项 B：服务器返回原始内容；由客户端的智能体完成推理。这能工作，但会把服务器逻辑移到客户端提示词里，比较脆弱。

选项 C：服务器通过 `sampling/createMessage` 请求客户端的大语言模型。服务器保留算法（读取哪些文件、做几轮），客户端保留计费和模型选择权。服务器完全不需要任何凭证。

采样就是选项 C。它是一种受信任的服务器在不成为完整大语言模型主机的情况下托管智能体循环（agent loops）的机制。

# The Concept

### `sampling/createMessage` request

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行自己的大语言模型，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数，加起来为 1.0：

- `costPriority`：优先选择更便宜的模型。
- `speedPriority`：优先选择更快的模型。
- `intelligencePriority`：优先选择能力更强的模型。

再加上 `hints`：服务器偏好的命名模型。客户端可以采纳也可以忽略这些提示（hints）；最终由客户端的用户配置决定。

### `includeContext`

三个取值：

- `"none"` —— 仅使用服务器提供的消息。默认值。
- `"thisServer"` —— 包含来自该服务器会话的先前消息。
- `"allServers"` —— 包含所有会话上下文。

自 2025-11-25 起，`includeContext` 已被软弃用，因为它会泄露跨服务器上下文，存在安全风险。建议优先使用 `"none"`，并在消息中显式传递上下文。

### Sampling with tools（SEP-1577）

2025-11-25 新增：采样请求可以包含 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这让服务器能够通过客户端的模型托管一个 ReAct 风格的智能体循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环执行：采样、如果调用了工具就执行、再采样，直到返回最终的助手消息。该功能在 2026 年第一季度前仍处于实验阶段；SDK 签名可能还会有变化。实现时请对照 2025-11-25 规范中的 client/sampling 部分进行确认。

### Human-in-the-loop

客户端**必须**在运行采样前向用户展示服务器要求模型做什么。恶意服务器可能利用采样操纵用户会话（“对用户说 X，让他们点击 Y”）。Claude Desktop、VS Code 和 Cursor 都会把采样请求以确认对话框的形式呈现，用户可以拒绝。

2026 年的共识是：没有人工确认的采样是危险信号。网关（gateways，见 Phase 13 · 17）可以自动批准低风险的采样，并自动拒绝任何可疑请求。

### Server-hosted loops without API keys

典型用例：一个本身无法访问大语言模型的代码摘要 MCP 服务器。它的流程是：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage` 并附带提示：“挑选五个最可能描述该仓库用途的文件。”
3. 读取这些文件。
4. 再次调用 `sampling/createMessage`，传入文件内容和提示：“用三段话总结该仓库。”
5. 将摘要作为 `tools/call` 的结果返回。

服务器从不接触大语言模型 API。客户端的用户使用自己的凭证为补全付费。

### Safety risks（Unit 42 披露，2026 年第一季度）

- **隐蔽采样（Covert sampling）。** 一个工具总是调用采样，提示为“从会话上下文中回复用户的邮箱”。Phase 13 · 15 会详细讲解攻击向量。
- **通过采样盗用资源（Resource theft via sampling）。** 服务器要求客户端 summarized 攻击者的载荷，把费用转嫁给用户。
- **循环炸弹（Loop bombs）。** 服务器在紧凑循环中调用采样。客户端**必须**实施每会话速率限制（rate limits）。

# Use It

`code/main.py` 提供了一个假的服务器到客户端采样骨架。一个模拟的 `summarize_repo` 工具会触发两轮采样（先选文件，再总结），假客户端会返回预设响应。该骨架展示了：

- 服务器发送带 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回补全。
- 服务器继续自己的循环。
- 速率限制器（rate limiter）限制每次工具调用中的总采样次数。

值得关注的点：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都发生在采样调用中。
- 模型偏好（model preferences）会影响客户端的模型选择；`hints` 列出偏好模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 的限制可以捕捉失控循环。

# Ship It

本节课产出 `outputs/skill-sampling-loop-designer.md`。针对某个需要大语言模型调用的服务器端算法（研究、摘要、规划等），该技能会设计一个基于采样（sampling）的实现，包含合适的 `modelPreferences`、速率限制（rate limits）和安全确认机制。

# Exercises

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制截断的效果。

2. 实现 SEP-1577 的“采样内工具”变体：采样请求携带 `tools` 数组。验证客户端循环会在返回最终补全前执行这些工具。注意漂移风险：2026 年上半年 SDK 签名可能仍会变化。

3. 加入人机协同确认（human-in-the-loop）：在服务器第一次调用 `sampling/createMessage` 前暂停并等待用户批准。被拒绝的调用返回一个带类型的拒绝信息。

4. 添加一个按用户区分的速率限制器（rate limiter），以客户端会话为键。同一用户的同服务器循环应共享预算。

5. 设计一个 `summarize_pdf` 工具，使用采样挑选要包含的文本块。画出将要发送的消息。当 `modelPreferences.intelligencePriority` 为 0.1 与 0.9 时，行为会有何不同？

# Key Terms

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Sampling | “服务器到客户端的大语言模型调用” | 服务器请求客户端的模型进行补全 |
| `sampling/createMessage` | “该方法” | 采样请求的 JSON-RPC 方法 |
| `modelPreferences` | “模型优先级” | 成本 / 速度 / 智能权重以及模型名称提示（hints） |
| `includeContext` | “跨会话泄露” | 已被软弃用的上下文包含模式 |
| SEP-1577 | “采样中的工具” | 允许在采样请求中包含工具，以支持服务器托管的 ReAct 循环 |
| Human-in-the-loop | “用户确认” | 客户端在运行采样前将请求展示给用户确认 |
| Loop bomb | “失控采样” | 服务器端的无限采样循环；客户端必须进行速率限制 |
| Covert sampling | “隐藏推理” | 恶意服务器在采样提示词中隐藏真实意图 |
| Resource theft | “占用用户的大语言模型预算” | 服务器强迫客户端把预算花在它不需要的采样上 |
| `stopReason` | “生成停止的原因” | `endTurn`、`stopSequence` 或 `maxTokens` |

# Further Reading

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样的高层概述
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — `sampling/createMessage` 的规范形态
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 采样内工具的规范演进提案（实验性）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽采样与资源盗用模式
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带客户端代码示例的逐步讲解
