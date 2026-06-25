# 问答系统

> 三类系统塑造了现代问答。抽取式模型定位答案片段，检索增强式模型将答案锚定在文档中，生成式模型直接生成答案。当今每一个 AI 助手都是这三者的混合体。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 11（机器翻译），Phase 5 · 10（注意力机制）
**时间：** 约 75 分钟

## 问题

用户输入 "When did the first iPhone launch?"，期望得到 "June 29, 2007." 而不是 "Apple's history is long and varied." 也不是孤零零的 "2007"。需要一个直接、有依据且正确的答案。

过去十年中，三种架构主导了问答领域。

- **抽取式 QA。** 给定一个问题和一段已知包含答案的文本，找出答案片段在文本中的起始和结束索引。SQuAD 是最具代表性的基准。
- **开放域 QA。** 不给定文本。先检索相关段落，再抽取或生成答案。这是当今所有 RAG 流程的基石。
- **生成式 / 闭卷 QA。** 大型语言模型完全依靠参数化记忆作答，不做检索。推理最快，但事实可靠性最低。

2026 年的趋势是混合式：先检索最佳的几段文本，再提示生成模型基于这些文本作答。这就是 RAG，第 14 课会深入讲解检索部分。本课构建 QA 部分。

## 概念

![QA 架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式。** 用 Transformer（BERT 系列）将问题和段落一起编码。训练两个预测答案起始与结束 token 索引的头部。损失是对有效位置上的分类交叉熵。输出是段落中的一个片段。按构造不会幻觉，但按构造也无法处理段落无法回答的问题。

**检索增强式（RAG）。** 分为两个阶段。首先，检索器从语料库中找出 top-`k` 段落；其次，阅读器（抽取式或生成式）基于这些段落生成答案。检索器—阅读器的拆分让两者可以独立训练与评估。现代 RAG 通常还会在中间加入重排序器。

**生成式。** 仅解码器的大语言模型（GPT、Claude、Llama）基于已学习的权重作答，没有检索步骤。在常识问题上表现优异，在罕见或新近事实上则严重翻车。幻觉率与事实在预训练数据中的出现频率成反比。

## 动手实现

### 步骤 1：使用预训练模型做抽取式 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，该数据集包含无法回答的问题。默认情况下，`question-answering` 管道会返回得分最高的片段，即使模型的空答案得分更高——它**不会**自动返回空答案。若要获得明确的“无答案”行为，请在调用管道时传入 `handle_impossible_answer=True`：仅当空答案得分超过所有片段得分时，管道才会返回空答案。无论如何都要检查返回的 `score` 字段。

### 步骤 2：检索增强式管道（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段管道。密集检索器（Sentence-BERT）通过语义相似度找出相关段落；抽取式阅读器（RoBERTa-SQuAD）从合并后的 top 段落中抽取答案片段。该方法适用于小型语料库。对于百万级文档语料库，请使用 FAISS 或向量数据库。

### 步骤 3：基于 RAG 的生成式

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模板至关重要。明确告诉模型必须基于上下文作答，并在上下文不足时返回 "I don't know"，相比朴素提示可将幻觉率降低 40–60%。更复杂的模板还会加入引用、置信分数和结构化抽取。

### 步骤 4：反映真实世界的评估

SQuAD 使用**精确匹配（EM）**和**token 级 F1**。EM 是经过归一化（小写、去除标点、去掉冠词）后的严格匹配——预测结果要么完全一致得 1 分，否则得 0。F1 基于预测与参考之间的 token 重叠计算，给予部分 credit。两者都会低估改写形式："June 29, 2007" 与 "June 29th, 2007" 通常 EM 为 0（序数词破坏了归一化），但仍能获得可观的 F1。

生产环境 QA 需要关注：

- **答案准确率**（由 LLM 或人工评判，因为自动指标无法捕捉语义等价）。
- **引用准确率。** 所引用的段落是否真的支持答案？通过生成引用与检索段落之间的字符串匹配即可轻松自动检查。
- **拒答校准。** 当答案不在检索到的段落中时，系统能否正确地说 "I don't know"？衡量虚假自信率。
- **检索召回。** 在评估阅读器之前，先衡量检索器是否把正确段落送入了 top-`k`。阅读器无法修复缺失的段落。

### RAGAS：2026 年生产环境评估框架

`RAGAS` 是专为 RAG 系统设计的评估框架，也是 2026 年的生产默认选择。它在无需标准答案的情况下从四个维度打分：

- **忠实度（Faithfulness）。** 答案中的每个论断是否都来自检索到的上下文？基于 NLI 的蕴含关系衡量。这是你的主要幻觉指标。
- **答案相关性（Answer relevance）。** 答案是否回应了问题？通过从答案生成假设问题并与真实问题比较来衡量。
- **上下文精确度（Context precision）。** 检索到的片段中，实际相关的占多少？精确度低意味着提示中噪声多。
- **上下文召回（Context recall）。** 检索集合是否包含了回答问题所需的全部信息？召回低意味着阅读器不可能成功。

无参考评分让你可以在实时生产流量上评估，无需人工整理的标准答案。对于开放式问题，可在之上叠加 LLM-as-judge，因为精确匹配指标几乎无效。

`pip install ragas`。接入你的检索器 + 阅读器。每个查询得到四个标量。对退化设置告警。

## 使用建议

2026 年的技术栈。

| 使用场景 | 推荐方案 |
|---------|-------------|
| 给定段落，找出答案片段 | `deepset/roberta-base-squad2` |
| 固定语料库，不可接受闭卷答案 | RAG：密集检索器 + LLM 阅读器 |
| 在文档存储上实时查询 | RAG + 混合检索（BM25 + 密集）+ 重排序器（第 14 课） |
| 对话式 QA（追问） | 带对话历史的 LLM + 每轮 RAG |
| 高度事实性、受监管领域 | 对权威语料库做抽取式；绝不单独使用生成式 |

2026 年，抽取式 QA 已不再时髦，因为基于 LLM 的 RAG 能处理更多场景。但在需要逐字引用的场景中它仍然上线运行：法律研究、合规监管、审计工具。

## 交付物

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 段维基百科文本上搭建上述 SQuAD 抽取式管道。手工编写 10 个问题。统计答案正确的频率。如果段落和问题都清晰，你应该能看到 7–9 个正确。
2. **中等。** 加入拒答分类器。当 top 检索得分低于阈值（例如余弦相似度 0.3）时，直接返回 "I don't know" 而不调用阅读器。在留出集上调整阈值。
3. **困难。** 在你选择的 10,000 份文档语料库上搭建 RAG 管道。实现混合检索（BM25 + 密集）和 RRF 融合（见第 14 课）。测量有无混合步骤时的答案准确率，并记录哪些问题类型受益最大。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive QA | 找出答案片段 | 预测答案在给定段落中的起始和结束索引。 |
| Open-domain QA | 语料库上的 QA | 不给出段落；必须先检索再作答。 |
| RAG | 先检索再生成 | 检索增强生成。检索器 + 阅读器管道。 |
| SQuAD | 权威基准 | 斯坦福问答数据集。使用 EM + F1 指标。 |
| Hallucination | 编造答案 | 阅读器输出未被检索上下文支持。 |
| Refusal calibration | 知道何时闭嘴 | 系统无法在上下文中找到答案时正确返回 "I don't know"。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) —— 该基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) —— DPR，QA 领域最具代表性的密集检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) —— 提出 RAG 名称的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) —— 全面的 RAG 综述。
