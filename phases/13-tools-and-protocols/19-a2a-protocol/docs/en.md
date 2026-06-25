# A2A — 智能体到智能体协议（Agent-to-Agent Protocol）

> MCP 是智能体到工具（agent-to-tool）的协议。A2A（Agent2Agent，智能体到智能体）则是智能体到智能体的开放协议，让构建在不同框架上的不透明智能体能够协同工作。它由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0，获得 AWS、Cisco、Microsoft、Salesforce、SAP、ServiceNow 等 150 多家支持者。A2A 吸纳了 IBM 的 ACP，并加入了 AP2 支付扩展。本节课将带你了解智能体卡片（Agent Card）、任务（Task）生命周期以及两种传输绑定。

**类型：** Build
**语言：** Python（标准库，Agent Card + Task 示例）
**前置课程：** Phase 13 · 06（MCP 基础）、Phase 13 · 08（MCP 客户端）
**时长：** 约 75 分钟

## 学习目标

- 区分 agent-to-tool（MCP）与 agent-to-agent（A2A）的使用场景。
- 在 `/.well-known/agent.json` 发布一张包含技能与端点元数据的智能体卡片（Agent Card）。
- 理解任务（Task）生命周期：submitted → working → input-required → completed / failed / canceled / rejected。
- 使用由 Parts（text、file、data）组成的消息（Message）以及作为输出的产物（Artifact）。

## 问题背景

一个客服智能体需要把撰写报告的工作委托给一个专门的写作智能体。在 A2A 出现之前，可选方案如下：

- 自定义 REST API。可行，但每一对智能体都要单独对接。
- 共享代码库。要求两个智能体运行在同一框架上。
- MCP。不合适：MCP 用于调用工具，而不是让两个智能体在保留各自不透明内部推理的情况下协作。

A2A 填补了这一空白。它把交互建模为一个智能体向另一个智能体发送任务（Task），包含生命周期、消息和产物。被调用智能体的内部状态保持不透明——调用方只能看到任务状态转换和最终输出。

A2A 是“让跨框架的智能体相互通信”的协议。它不会取代 MCP；二者是互补关系。

## 核心概念

### 智能体卡片（Agent Card）

每个符合 A2A 规范的智能体都会在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现机制基于 URL：拉取卡片，获知 A2A 端点地址，枚举技能。

### 签名智能体卡片（AP2）

AP2 扩展（2025 年 9 月）为智能体卡片增加了加密签名。发布者用 JWT 对自己的卡片签名，消费者进行验证，以防止冒充。

### 任务生命周期（Task lifecycle）

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

客户端通过 `tasks/send` 发起请求。被调用智能体在各状态之间转换；客户端通过 SSE 订阅状态更新，或采用轮询。

### 消息与部件（Messages and Parts）

一条消息包含一个或多个 Part：

- `text` —— 纯文本内容。
- `file` —— 带 mimeType 的 base64 二进制数据。
- `data` —— 类型化的 JSON 负载（被调用智能体的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### 产物（Artifacts）

输出不是原始字符串，而是产物（Artifact）。产物是一种具名、类型化的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

产物可以分块流式传输，调用方负责累加。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST 请求，可选 SSE 流式传输。这是默认绑定。
2. **gRPC。** 适用于原生使用 gRPC 的企业环境。

两种绑定承载相同的逻辑消息结构。

### 不透明性保护（Opacity preservation）

一项关键设计原则：被调用智能体的内部状态是不透明的。调用方只能看到任务状态和产物。被调用智能体的思维链（chain-of-thought）、工具调用、子智能体委托——全部不可见。这与 MCP 不同，在 MCP 中工具调用是透明的。

理由：A2A 让竞争对手也能在不必暴露内部实现的情况下协作。可以是“调用这个客服智能体”，而调用方无需了解该智能体如何提供服务。

### 发展时间线

- **2025-04-09。** Google 发布 A2A。
- **2025-06-23。** 捐赠给 Linux 基金会。
- **2025-08。** 吸纳 IBM 的 ACP。
- **2025-09。** AP2 扩展（智能体支付）发布。
- **2026-04。** v1.0 发布，获得 150 多家支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 使用场景 | 智能体到工具（Agent-to-tool） | 智能体到智能体（Agent-to-agent） |
| 不透明性 | 工具调用透明 | 内部推理不透明 |
| 典型调用方 | 智能体运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 带生命周期的任务 |
| 授权 | OAuth 2.1（Phase 13 · 16） | JWT 签名的智能体卡片（AP2） |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

需要调用某个具体工具时使用 MCP；需要把完整任务委托给另一个智能体时使用 A2A。许多生产系统会同时使用两者：智能体用 MCP 作为工具层，用 A2A 作为协作层。

## 动手实践

`code/main.py` 实现了一个最小化的 A2A 示例：一个研究智能体发布自己的卡片，一个写作智能体接收包含 PDF 和文本指令的 `tasks/send` 请求，经过 working → input_required → working → completed 的状态转换，最终返回一个文本产物。全部使用标准库，并通过内存内传输来聚焦消息结构。

需要关注的地方：

- 智能体卡片 JSON 的结构。
- 任务 id 的分配与状态转换。
- 混合类型 Parts 的消息。
- 任务中段的 input-required 分支。
- 完成时返回 Artifact。

## 产出物

本节课会生成 `outputs/skill-a2a-agent-spec.md`。对于需要被其他智能体调用的新智能体，该技能会产出 Agent Card JSON、技能模式（skills schema）和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的任务生命周期，包括被调用智能体请求澄清时的 input-required 暂停。

2. 添加签名智能体卡片。对卡片的规范化 JSON 进行 HMAC 签名，编写验证器并确认在卡片被篡改后验证失败。

3. 实现任务流式传输：写作智能体通过 SSE 发出三个渐进的产物分块，调用方将其累加。

4. 设计一个包装 MCP 服务器的 A2A 智能体。把每个 MCP 工具映射为一个 A2A 技能。注意其中的权衡——会损失哪些不透明性？

5. 阅读 A2A v1.0 的公告，找出截至 2026 年 4 月尚未被任何框架实现的一项特性。（提示：它与多跳任务委托有关。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | 用于不透明智能体协作的开放协议 |
| Agent Card（智能体卡片） | "`.well-known/agent.json`" | 描述智能体技能与端点的公开元数据 |
| Skill（技能） | "A callable unit" | 智能体支持的具名操作（类似 MCP 工具） |
| Task（任务） | "Unit of delegation" | 具有生命周期和最终产物的工作项 |
| Message（消息） | "Task input" | 携带 Parts（text、file、data） |
| Part（部件） | "Typed chunk" | 消息中的 `text` / `file` / `data` 元素 |
| Artifact（产物） | "Task output" | 完成时返回的具名、类型化输出 |
| AP2 | "Agent Payments Protocol" | 用于信任与支付的签名智能体卡片扩展 |
| Opacity（不透明性） | "Black-box collaboration" | 被调用智能体的内部对调用方隐藏 |
| Input-required | "Task pause" | 智能体需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) —— A2A 规范官方文档
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) —— 参考实现与 SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) —— 2025 年 6 月治理移交新闻稿
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) —— 路线图与合作伙伴进展
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) —— v1.0 发布说明与向后兼容指南
