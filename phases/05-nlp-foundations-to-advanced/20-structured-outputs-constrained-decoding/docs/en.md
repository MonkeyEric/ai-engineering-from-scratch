# 结构化输出与约束解码

> 让 LLM 输出 JSON。大多数时候你能得到 JSON。但在生产环境中，“大多数时候”就是问题所在。约束解码通过在采样前修改 logits，把“大多数时候”变成“总是”。

**类型：** 构建
**语言：** Python
**前置知识：** 第 5 阶段 · 第 17 课（聊天机器人），第 5 阶段 · 第 19 课（子词分词）
**时长：** 约 60 分钟

## 问题所在

一个分类器向 LLM 提问：“返回 {positive, negative, neutral} 中的一个。”模型却返回：“The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...”。你的解析器崩溃了，分类器的 F1 变成 0.0。

自由格式生成不是契约，只是建议。生产系统需要契约。

2026 年存在三个层级。

1. **提示工程。** 礼貌地要求。"Return only the JSON object." 在前沿模型上大约 80% 有效，在更小的模型上效果更差。
2. **原生结构化输出 API。** OpenAI 的 `response_format`、Anthropic 的 tool use、Gemini 的 JSON mode。在支持的 schema 上可靠，但会绑定厂商。
3. **约束解码。** 在每一步生成时修改 logits，让模型*无法*输出无效 token。按构造保证 100% 有效，适用于任何本地模型。

本课建立对三者的直觉，并说明何时选择哪一种。

## 核心概念

![约束解码在每一步屏蔽无效 token](../assets/constrained-decoding.svg)

**约束解码的工作原理。** 在每一步生成中，LLM 会输出覆盖整个词表（约 10 万 token）的 logit 向量。一个 *logit processor* 位于模型与采样器之间，根据当前在目标语法（JSON Schema、正则表达式、上下文无关语法）中的位置，计算哪些 token 是合法的，并将所有非法 token 的 logit 设为负无穷。对剩余 logit 做 softmax 后，概率质量只落在合法的续接 token 上。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译成有限状态机。每个 token 都能以 O(1) 时间查询合法的下一个 token。基于 FSM，因此递归 schema 需要展平。
- **XGrammar / llguidance。** 上下文无关语法引擎。能处理递归 JSON Schema。解码开销接近零。OpenAI 在 2025 年的结构化输出实现中致谢了 llguidance。
- **vLLM guided decoding。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，后端可选 Outlines、XGrammar 或 lm-format-enforcer。
- **Instructor。** 基于 Pydantic 的 LLM 封装。验证失败时自动重试。跨厂商，但不修改 logits —— 它依赖重试与结构化输出感知提示。

### 反直觉的结果

约束解码通常比无约束生成*更快*。两个原因。首先，它缩小了下一个 token 的搜索空间。其次，巧妙的实现会跳过强制 token 的生成（脚手架部分如 `{"name": "` —— 每个字节都是确定的）。

### 让你付出代价的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 前面，模型会在思考之前就确定答案。JSON 是有效的，答案却是错的，没有任何验证能捕获这一点。

```json
// 不好
{"answer": "yes", "reasoning": "because ..."}

// 好
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑问题，不是格式问题。

## 动手实现

### 第一步：从零实现基于正则表达式的约束生成

参见 `code/main.py` 中的独立 FSM 实现。核心思想用 30 行概括：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪当前已满足的语法部分。`valid_tokens(state, tokenizer)` 计算哪些词表 token 能让 FSM 沿接受路径前进。

### 第二步：使用 Outlines 处理 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误，永远。FSM 让无效输出不可达。

### 第三步：使用 Instructor 实现跨厂商的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不修改 logits。它将 schema 格式化进提示，解析输出，并在验证失败时重试（默认 3 次）。适用于任何厂商。重试会增加延迟与成本。跨厂商可移植性是它的卖点。

### 第四步：原生厂商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。在支持的 schema 上与 Outlines 可靠性相当。无需本地模型管理。但会绑定厂商。

## 陷阱

- **递归 schema。** Outlines 会把递归展平到固定深度。树状输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **巨型枚举。** 一万个选项的枚举编译缓慢或超时。改用检索器：先预测 top-k 候选，再约束到这些候选。
- **语法过于严格。** 强制 `date: "YYYY-MM-DD"` 的正则，模型就无法为缺失日期输出 `"unknown"`。模型会补偿性地编造一个日期。应允许 `null` 或哨兵值。
- **过早承诺。** 见上文字段顺序陷阱。永远把 reasoning 放在最前。
- **厂商 JSON mode 不带 schema。** 纯 JSON mode 只保证 JSON 语法有效，不保证符合你的用途。始终提供完整 schema。

## 使用建议

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生厂商结构化输出 |
| 任意厂商，Pydantic 工作流，可接受重试 | Instructor |
| 本地模型，需要 100% 有效性，扁平 schema | Outlines（FSM） |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务 | vLLM guided decoding |
| 批处理，可接受重试 | Instructor + 最便宜的模型 |

## 交付

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: 选择一种结构化输出方法、schema 设计与验证方案。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定一个用例（厂商、延迟预算、schema 复杂度、失败容忍度），输出：

1. 机制。原生厂商结构化输出、Instructor 重试、Outlines FSM 或 XGrammar CFG。一句话说明理由。
2. Schema 设计。字段顺序（reasoning 在前，answer 在后）、为“未知”设置可空字段、enum 还是 regex、必填字段。
3. 失败策略。最大重试次数、降级模型、优雅的 `null` 处理、分布外拒绝。
4. 验证方案。Schema 合规率（目标 100%）、语义有效性（LLM 评委）、字段覆盖率、延迟 p50/p99。

拒绝任何把 `answer` 或 `decision` 放在 reasoning 字段之前的设计。拒绝不带 schema 的裸 JSON mode。对仅支持 FSM 的库使用递归 schema 时发出警告。
```

## 练习

1. **简单。** 对一个小型开源权重模型（例如 Llama-3.2-3B）不使用约束解码，直接要求输出 `Review(sentiment, confidence, evidence_span)`。在 100 条评论上测量能解析为合法 JSON 的比例。
2. **中等。** 在同一批语料上使用 Outlines JSON mode。比较合规率、延迟与语义准确率。
3. **困难。** 从零实现一个基于正则表达式的约束解码器，匹配电话号码（`\d{3}-\d{3}-\d{4}`）。在 1000 个样本上验证 0 个无效输出。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Constrained decoding | 强制输出合法 | 在每一步生成中屏蔽非法 token 的 logits。 |
| Logit processor | 负责约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示；O(1) 查询合法下一个 token。 |
| CFG | 上下文无关语法 | 能处理递归的语法；比 FSM 慢但表达能力更强。 |
| Schema field order | 它重要吗？ | 重要 —— 第一个字段会让模型先承诺；永远把 reasoning 放在 answer 前面。 |
| Guided decoding | vLLM 里的叫法 | 同一概念，集成在推理服务中。 |
| JSON mode | OpenAI 早期版本 | 保证 JSON 语法；不保证匹配 schema。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) —— Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) —— 快速的基于 CFG 的约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) —— 推理服务集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) —— API 参考与注意事项。
- [Instructor library](https://python.useinstructor.com/) —— 跨厂商的 Pydantic + 重试。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) —— 对 6 个约束解码框架的基准测试。
