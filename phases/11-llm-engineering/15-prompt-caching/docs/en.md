# 提示词缓存（Prompt Caching）与上下文缓存（Context Caching）

> 你的系统提示词有 4,000 词元（tokens），检索增强生成（RAG）上下文有 20,000 词元，而每次请求都要把两者一起发送，并且每次都按全价付费。提示词缓存让提供方在服务端保留这段前缀并保持“温热”，再次使用时只按正常费率的 10% 计费。使用得当，它可将推理（inference）成本降低 50–90%，并将首词元延迟（first-token latency）降低 40–85%。

**类型：** 实战构建
**语言：** Python
**前置知识：** 第 11 阶段 · 01（提示工程）、第 11 阶段 · 05（上下文工程）、第 11 阶段 · 11（缓存与成本）
**时间：** 约 60 分钟

## 问题所在

一个编程智能体在对话的每一轮都向 Claude 发送相同的 15,000 词元系统提示词。20 轮对话，按输入词元 $3/M 计算，仅输入成本就高达 $0.90——这还不算用户真正的消息。乘以每天 10,000 次对话，仅这段从未变化的文本就产生每天 $9,000 的账单。

你无法压缩提示词而不损害质量。你也无法避免发送它——模型在每一轮都需要它。唯一的办法，就是不要再为提供方已经见过的前缀支付全价。

这个办法就是提示词缓存。Anthropic 于 2024 年 8 月推出该功能（2025 年推出了 1 小时延长存活时间的变体），OpenAI 同年稍晚实现了自动缓存，Google 在 Gemini 1.5 发布时同步上线了显式上下文缓存；如今三家都把缓存作为前沿模型的一等公民功能。

## 核心概念

![提示词缓存：一次写入，多次低价读取](../assets/prompt-caching.svg)

**工作原理。** 当某次请求的前缀与近期请求匹配时，提供方会直接复用上一次运行产生的 KV 缓存（KV-cache），而不再重新编码这些词元。你第一次只需支付少量写入溢价（write premium），之后每次都能享受大幅的读取折扣。

**2026 年三家提供方的三种风格。**

| 提供方 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存长度 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在内容块上显式标注 `cache_control` 标记 | 输入费用 90% 减免 | 25% 加价 | 5 分钟（可延长至 1 小时） | 1,024 词元（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入费用 50% 减免 | 无 | 最长 1 小时（尽力而为） | 1,024 词元 |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读取约为正常费率的 25% | 按 token·小时收取存储费 | 用户自定义（默认 1 小时） | 4,096 词元（Flash），32,768（Pro） |

**不变原则。** 三家都只缓存前缀。只要请求之间有任何词元不同，从第一个差异词元开始往后的所有内容都会失效。因此要把*稳定*部分放在顶部，把*可变*部分放在底部。

### 对缓存友好的布局

```
[系统提示词]          <-- 缓存此处
[工具定义]            <-- 缓存此处
[少样本示例]          <-- 缓存此处
[检索文档]            <-- 若复用则缓存，否则不缓存
[对话历史]            <-- 缓存至上一轮
[当前用户消息]        <-- 永不缓存（每次都不同）
```

违反这个顺序——比如把用户消息放在系统提示词之前，或在少样本示例之间穿插动态检索——缓存就永远不会命中。

### 盈亏平衡（break-even）计算

Anthropic 的 25% 写入溢价意味着：一个缓存块至少要被读取两次才能真正省钱。1 次写入 + 1 次读取，平均每次成本为 0.675 倍（节省 32%）；1 次写入 + 10 次读取，平均每次成本为 0.205 倍（节省 80%）。经验法则：只要预期在 TTL 内会被复用至少 3 次，就应该缓存。

## 动手实现

### 步骤 1：使用显式标记的 Anthropic 提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将该块保存 5 分钟。在该时间窗口内复用即可命中；过期后再复用会重新写入。

**响应中的用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25 倍计费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1 倍计费
```

在 CI 中同时检查这两个字段——如果多轮请求的 `cache_read_input_tokens` 始终为 0，说明你的缓存键（cache key）在漂移。

### 步骤 2：1 小时延长 TTL

对于长时间运行的批处理任务，默认 5 分钟会在任务之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是原来的 2 倍（比基线贵 50% 而非 25%），但只要批处理复用前缀超过 5 次，就能快速回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 不需要任何配置。任何超过 1,024 词元且与近期请求匹配的前缀都会自动享受 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 被折扣的部分
```

同样的缓存友好布局规则适用。但有两件事会杀死 OpenAI 的缓存，却不会杀死 Anthropic 的缓存：更改 `user` 字段（它被用作缓存键组件）以及重新排序工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 把缓存当作一个可以创建和命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按 token·小时对缓存存活时间收取存储费，读取费率约为正常输入费率的 25%。这种模式适合在多天、多会话中复用同一份巨型提示词。

### 步骤 5：在生产环境中监控命中率

请参阅 `code/main.py`，其中包含一个模拟三家提供方的记账器，可跟踪写入、读取、未命中次数，并计算每 1,000 次请求的混合成本。部署前应设置目标命中率门槛——大多数生产级 Anthropic 配置在预热后读取比例应超过 80%。

## 2026 年仍在犯的陷阱

- **顶部放动态时间戳。** 系统提示词顶部写 `"Current time: 2026-04-22 15:30:02"`。每次请求都会未命中。把时间戳移到缓存断点下方。
- **工具顺序不稳定。** 以稳定顺序序列化工具——部署之间字典顺序打乱会让所有命中失效。
- **近似的自由文本。** “You are helpful.” 与 “You are a helpful assistant.” 差一个字节就是完全未命中。
- **块太小。** Anthropic 强制要求至少 1,024 词元（Haiku 为 2,048）。更小的块会被静默地不缓存。
- **盲目的成本仪表板。** 把“输入词元”拆分为已缓存与未缓存。否则流量下降看起来就像缓存收益。

## 如何使用

2026 年的缓存选型栈：

| 场景 | 选择 |
|-----------|------|
| 智能体拥有稳定的 10k+ 系统提示词，且多轮对话 | Anthropic `cache_control`，5 分钟 TTL |
| 批处理任务 30 分钟以上复用同一前缀 | Anthropic `ttl: "1h"` |
| GPT-5 无服务器端点，无自定义基础设施 | OpenAI 自动缓存（只需让前缀稳定且足够长） |
| 跨多天复用巨型代码/文档语料 | Gemini 显式 `CachedContent` |
| 跨提供方降级 | 让可缓存前缀布局在各提供方保持一致，以便任何一方都能命中 |

与用户消息层结合时，请搭配语义缓存（semantic caching）（第 11 阶段 · 11）：提示词缓存处理*词元完全相同*的复用，语义缓存处理*语义完全相同*的复用。

## 交付

保存到 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: 设计对缓存友好的提示词布局，并选择合适的提供方缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

给定一份提示词（系统提示词 + 工具 + 少样本示例 + 检索 + 历史 + 用户消息）和使用画像（每小时请求数、所需 TTL、提供方），输出：

1. 布局。重新排序后的各段落，并标出唯一缓存断点；解释哪些段落稳定、哪些段落易变。
2. 提供方模式。Anthropic cache_control、OpenAI 自动缓存或 Gemini CachedContent。从 TTL 和复用模式出发给出理由。
3. 盈亏平衡。TTL 内每次写入对应的预期读取次数；与无缓存方案相比的净成本及计算过程。
4. 验证计划。CI 断言：第二次相同请求的 cache_read_input_tokens > 0；仪表板按已缓存与未缓存词元拆分。
5. 失效模式。列出该设置下缓存最可能未命中的三个原因（动态时间戳、工具重排、近似重复文本）以及各自的预防措施。

拒绝交付任何将动态字段放在缓存断点上方的缓存方案。拒绝在未证明 1 小时 TTL 的 2 倍写入溢价能够回本的情况下启用 1 小时 TTL。
```

## 练习

1. **简单。** 针对一个 10 轮对话，使用 5,000 词元系统提示词与 Claude 交互。先不用 `cache_control` 运行一次，再用 `cache_control` 运行一次。报告两种情况的输入词元费用。
2. **中等。** 编写一个测试框架：给定一个提示词模板和请求日志，计算各提供方（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、Gemini 显式）的预期命中率与美元节省。
3. **困难。** 构建一个布局优化器：给定一个提示词和字段列表（每个字段标记 `stable=True/False`），重写提示词，将单一缓存断点放在最大缓存友好位置且不丢失信息。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 提示词缓存（Prompt caching） | “让长提示词变便宜” | 复用提供方端的 KV 缓存以匹配前缀；重复输入词元可享受 50–90% 折扣。 |
| `cache_control` | “Anthropic 的标记” | 内容块属性，声明“到此为止的一切都可缓存”；格式为 `{"type": "ephemeral"}`。 |
| 缓存写入（Cache write） | “支付溢价” | 首次填充缓存的请求；Anthropic 按约 1.25 倍输入费率计费，OpenAI 免费。 |
| 缓存读取（Cache read） | “享受折扣” | 后续匹配前缀的请求；Anthropic 按 10% 计费，OpenAI 按 50%，Gemini 约 25%。 |
| TTL | “能活多久” | 缓存保持温热状态的秒数；Anthropic 默认 5 分钟（可延长 1 小时），OpenAI 尽力最长 1 小时，Gemini 用户自定义。 |
| 延长 TTL（Extended TTL） | “Anthropic 的 1 小时缓存” | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价为 2 倍，但适合批处理复用。 |
| 前缀匹配（Prefix match） | “为什么我的缓存没命中” | 只有从开头到断点的每个词元都字节相同时，缓存才会命中。 |
| 上下文缓存（Context caching，Gemini） | “显式那种” | Google 的命名式、按存储计费的缓存对象；最适合跨多天复用大型语料。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1 小时 TTL、盈亏平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 原始发布博文，包含延迟数据。
- 第 11 阶段 · 05（上下文工程）—— 如何切分提示词，让缓存得以落地。
- 第 11 阶段 · 11（缓存与成本）—— 将提示词缓存与用户消息的语义缓存配对使用。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 提示词缓存向用户暴露的 KV 缓存内存模型；解释为什么已缓存前缀的重新读取成本约是重新计算的 1/10。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — 预填充（prefill）是提示词缓存所绕过的阶段；本文解释为什么缓存命中时 TTFT 大幅下降，而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 提示词缓存与投机解码（speculative decoding）、Flash Attention、MQA/GQA 同属弯曲推理成本曲线的杠杆；想深入了解其余三项可读此文。
