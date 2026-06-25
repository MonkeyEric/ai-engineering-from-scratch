# 聊天机器人——从基于规则到神经网络再到大语言模型智能体

> ELIZA 通过模式匹配回复。DialogFlow 映射意图。GPT 从权重中生成答案。Claude 调用工具并验证结果。每一个时代都解决了前一个时代最糟糕的失败。

**类型：** 学习
**语言：** Python
**先修知识：** 第 5 阶段 · 13（问答系统），第 5 阶段 · 14（信息检索）
**时间：** 约 75 分钟

## 问题背景

用户说：“我想改签航班。”系统必须弄清楚用户的意图、缺少哪些信息、如何获取这些信息，以及如何完成操作。然后用户又说：“等等，如果我要取消呢？”系统必须记住上下文、切换任务并保留状态。

对话对机器学习系统来说很难。输入是开放式的。输出必须在多轮对话中保持连贯。系统可能需要对现实世界采取行动（改签航班、刷卡扣款）。每一步出错都会直接暴露给用户。

聊天机器人架构经历了四种范式，每一种都是因为前一种失败得太明显而被引入。本课按顺序介绍它们。2026 年的生产环境是最后两种的混合体。

## 核心概念

![聊天机器人演进：基于规则 → 检索 → 神经网络 → 智能体](../assets/chatbot.svg)

**基于规则（ELIZA、AIML、DialogFlow）。** 人工编写的模式匹配用户输入并生成回复。意图分类器将请求路由到预定义的流程。槽位填充状态机收集所需信息。在其设计的狭窄范围内表现出色。一旦超出范围立即失效。至今仍部署在安全关键领域（银行身份验证、航班预订），因为这些场景无法容忍幻觉。

**基于检索。** 一种 FAQ 风格的系统。将每一对（话语，回复）编码。运行时，对用户消息编码并检索最相似的存储回复。类似于 Zendesk 经典的“相似文章”功能。比规则更能处理同义改写。不生成新内容，因此不会幻觉。

**神经网络（seq2seq）。** 在对话日志上训练的编码器-解码器模型。从零生成回复。流畅但容易产生通用输出（“我不知道”）和事实漂移。无法可靠地保持主题。这就是 2016–2019 年间 Google、Facebook 和微软 的聊天机器人都令人失望的原因。

**大语言模型智能体。** 一个被包装在循环中的语言模型，能够规划、调用工具并验证结果。它不是带长提示词的聊天机器人，而是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。检索优先的 grounding（RAG）防止幻觉。工具调用让它能够实际执行操作。这就是 2026 年的架构。

这四种范式不是依次替代的关系。2026 年的生产级聊天机器人会同时经过四种路径：基于规则用于身份验证和破坏性操作，检索用于 FAQ，神经网络生成用于自然措辞，大语言模型智能体用于模糊开放式查询。

## 动手实现

### 步骤 1：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行代码实现 ELIZA。反射技巧（“I feel sad” → “Why do you feel sad”）是 Weizenbaum 于 1966 年提出的经典心理治疗师演示，至今仍具有启发性。

### 步骤 2：基于检索（FAQ）

下面的示例代码需要执行 `pip install sentence-transformers`（会同时安装 torch）。本课的 `code/main.py` 使用的是标准库中的 Jaccard 相似度，因此课程无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级处理。

### 步骤 3：神经网络生成（基线）

使用小型指令微调的编码器-解码器模型（FLAN-T5）或微调后的对话模型。2026 年仅凭它自身无法用于生产（会出现矛盾、离题漂移、事实错误），但会作为混合系统的一部分用于自然措辞。DialoGPT 风格的仅解码器模型需要显式的轮次分隔符和 EOS 处理才能生成连贯回复；FLAN-T5 的 text2text 流水线在教学示例中可以开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 步骤 4：大语言模型智能体循环

2026 年生产的典型形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

有三点需要说明。工具是大语言模型可以调用的可调用函数。当大语言模型返回最终答案而非工具调用时，循环终止。步骤预算可防止在模糊任务上陷入无限循环。

真实生产还会增加：检索优先 grounding（每次调用大语言模型前注入相关文档）、护栏（未经确认拒绝破坏性操作）、可观测性（记录每一步）以及评估（自动检查智能体行为是否符合规范）。

### 步骤 5：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

其模式是：对任何破坏性操作使用确定性规则，对固定 FAQ 使用检索，对其他所有内容使用大语言模型智能体。这就是 2026 年客户服务系统的实际部署方式。

## 应用场景

2026 年的技术栈：

| 使用场景 | 架构 |
|---------|---------------|
| 预订、支付、身份验证 | 基于规则的状态机 + 槽位填充 |
| 客户支持 FAQ | 基于精选答案的检索 |
| 开放式帮助对话 | 带 RAG 和工具调用的大语言模型智能体 |
| 内部工具 / IDE 助手 | 带工具调用的大语言模型智能体（搜索、读取、写入） |
| 陪伴 / 角色聊天机器人 | 带角色系统提示词的微调大语言模型，基于知识检索 |

生产中始终使用混合路由。没有任何单一架构能处理好每一种请求。路由层本身通常是一个小型意图分类器。

## 仍会存在的失效模式

- **自信捏造。** 大语言模型智能体声称完成了一个实际并未完成的操作。缓解措施：验证结果、记录工具调用、绝不允许大语言模型在没有成功工具返回的情况下声称已完成某事。
- **提示注入。** 用户插入文本覆盖系统提示词。在 OWASP 2025 年大语言模型应用十大风险中排名第一（LLM01）。分为两种：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、邮件或工具输出中）。

  攻击成功率因场景而异。在通用工具使用和编程基准测试中，前沿模型的成功率为约 0.5%–8.5%。某些高风险场景（针对 AI 编程智能体的自适应攻击、脆弱的编排层）已达到约 84%。生产环境 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）—— 由攻击者控制的邮件触发 Microsoft 365 Copilot 中的零点击数据泄露漏洞。

  缓解措施：将整个循环中的用户输入视为不可信；在工具调用前进行清理；将工具输出与主提示词隔离；采用 Plan-Verify-Execute（PVE）模式，即智能体先规划，然后在执行前验证每个动作是否符合该规划（这可阻止工具结果注入新的未计划动作）；对破坏性操作要求用户确认；对工具作用域应用最小权限原则。

  仅靠提示工程无法完全消除该风险。需要外部运行时防御层（LLM Guard、白名单验证、语义异常检测）。
- **范围蔓延。** 智能体因为工具调用返回了略微相关的信息而偏离任务。缓解措施：缩小工具契约；保持系统提示词聚焦；增加离题率评估。
- **无限循环。** 智能体反复调用同一工具。缓解措施：步骤预算、工具调用去重、用大语言模型评判“是否正在取得进展”。
- **上下文窗口耗尽。** 长对话将最早的几轮挤出上下文。缓解措施：总结较早轮次、按相似度检索相关历史轮次，或使用长上下文模型。

## 交付产出

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 用 10 条模式实现上文中的基于规则回复，用于一个咖啡店点餐机器人。测试边界情况：重复点单、修改订单、取消订单、意图不清。
2. **中等。** 构建一个 FAQ + 大语言模型回退的混合系统。为一款 SaaS 产品准备 50 条 FAQ，用大语言模型基于文档站点检索进行回退。在 100 个真实支持问题上测量拒绝率和准确率。
3. **困难。** 用三个工具（搜索、读取用户数据、发送邮件）实现上文中的智能体循环。用 50 个测试场景运行评估，包括提示注入攻击。报告离题率、任务失败率和任何注入成功的情况。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Intent（意图） | 用户想要什么 | 类别标签（book_flight、reset_password），用于路由到处理程序。 |
| Slot（槽位） | 一条信息 | 机器人需要的参数（日期、目的地）。槽位填充就是依次询问的过程。 |
| RAG | 检索加生成 | 检索相关文档，然后为大语言模型的回复提供依据。 |
| Tool call（工具调用） | 函数调用 | 大语言模型发出带名称和参数的结构化调用，运行时执行并返回结果。 |
| Agent loop（智能体循环） | 规划、执行、验证 | 控制器交替运行大语言模型调用和工具调用，直到任务完成。 |
| Prompt injection（提示注入） | 用户攻击提示词 | 试图覆盖系统提示词的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 最早的基于规则聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 晚期的神经网络聊天机器人论文，正处于大语言模型智能体接管之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名智能体循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年的生产指南，到 2026 年仍然适用。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 将提示注入列为首要安全风险的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实用的编排层防御，包括 Plan-Verify-Execute 和用户确认流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 间接提示注入导致零点击数据泄露的典型 CVE。说明具有写入权限的智能体为何需要运行时防御的参考案例。
