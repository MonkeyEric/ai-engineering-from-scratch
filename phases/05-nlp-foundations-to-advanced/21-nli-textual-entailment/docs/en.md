# 自然语言推理——文本蕴含

> "t 蕴含 h" 意味着阅读 t 的人类会得出 h 为真的结论。NLI 的任务是预测蕴含 / 矛盾 / 中立。表面上平平无奇，在生产中却举足轻重。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 5 · 05（情感分析）、阶段 5 · 13（问答系统）
**时间：** ~60 分钟

## 问题背景

你做了一个摘要器。它生成了一段摘要。你怎么知道摘要里没有幻觉？

你做了一个聊天机器人。它回答了"是"。你怎么知道答案有检索到的段落作为支撑？

你要给一万篇新闻文章按主题分类，但没有训练标签。你能复用某个模型吗？

这三个问题都可以归结为自然语言推理。NLI 问的是：给定前提 `t` 和假设 `h`，`h` 是被 `t` 所蕴含、相矛盾，还是中立的（无关）？

- **幻觉检测：** `t` = 源文档，`h` = 摘要中的某个断言。不是蕴含 = 幻觉。
- **有依据问答：** `t` = 检索到的段落，`h` = 生成的答案。不是蕴含 = 捏造。
- **零样本分类：** `t` = 文档，`h` = 用自然语言表达的标签（"This is about sports"）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架背后都内置了一个 NLI 模型。

## 核心概念

![NLI: 三分类，前提 vs 假设](../assets/nli.svg)

**三个标签。**

- **蕴含。** `t` → `h`。"The cat is on the mat" 蕴含 "There is a cat."
- **矛盾。** `t` → ¬`h`。"The cat is on the mat" 与 "There is no cat" 矛盾。
- **中立。** 两边都推不出来。
"The cat is on the mat" 对 "The cat is hungry" 是中立的。

**不是逻辑蕴含。** NLI 是*自然*语言推理——指典型人类读者会做出的推断，而非严格逻辑。"John walked his dog" 在 NLI 中蕴含 "John has a dog"，但严格的一阶逻辑只有在你把"拥有"公理化之后才会承认。

**数据集。**

- **SNLI**（2015）。57 万人工标注对，前提来自图像说明。领域较窄。
- **MultiNLI**（2017）。43.3 万对，覆盖 10 种文体。2026 年的标准训练语料。
- **ANLI**（2019）。对抗式 NLI。人类专门写出用来攻破现有模型的例子。更难。
- **DocNLI、ConTRoL**（2020–21）。文档级前提。测试多跳与长程推理。

**模型结构。** Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示送入一个 3 路 softmax。在 MNLI 上训练，在留出基准上评估，分布内准确率达到 90% 以上。

**通过 NLI 做零样本分类。** 给定文档和候选标签，把每个标签转成假设（"This text is about sports"）。计算每个假设的蕴含概率，取最大值。这就是 Hugging Face `zero-shot-classification` pipeline 背后的机制。

## 动手实现

### 步骤 1：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 返回所有标签；替代已弃用的 return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

在生产中，NLI 的开源默认选择是 `facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli`。DeBERTa-v3 在排行榜上表现最好。

### 步骤 2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是 "This example is about {label}."，可通过 `hypothesis_template` 自定义。不需要训练数据。不需要微调。开箱即用。

### 步骤 3：RAG 忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度的核心。把生成答案拆成原子主张，每个主张与检索上下文做 NLI 检查，返回蕴含主张的比例。

### 步骤 4：手搓 NLI 分类器（概念版）

参考 `code/main.py` 中的纯标准库玩具实现：通过词汇重叠 + 否定检测来比较前提与假设。打不过 Transformer 模型——但它展示了这个任务的形态：两个文本输入，3 路标签输出，损失 = `{entail, contradict, neutral}` 上的交叉熵。

## 常见陷阱

- **仅假设捷径。** 模型只凭假设本身就能在 SNLI 上达到约 60% 的准确率，因为 "not"、"nobody"、"never" 等词与矛盾标签相关。这是检测标签泄露的有力基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）能通过 SNLI，但在 HANS/ANLI 上失效。请使用对抗性基准。
- **文档长度性能衰减。** 单句 NLI 模型在文档级前提上 F1 会下降 20 分以上。长上下文请使用在 DocNLI 上训练过的模型。
- **零样本模板敏感性。** "This example is about {label}"、"{label}" 与 "The topic is {label}" 可能让准确率波动 10 个百分点以上。要调模板。
- **领域不匹配。** MNLI 训练于通用英语。法律、医学、科学文本需要领域专用 NLI 模型（如 SciNLI、MedNLI）。

## 实际使用

2026 年的技术栈：

| 使用场景 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / 边缘端 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 幻觉检测 | RAGAS / DeepEval 内部的 NLI 层 |

2026 年的元模式：NLI 是文本理解的万能胶带。每当你需要判断"A 是否支持 B？"或"A 是否反驳 B？"——先考虑 NLI，再考虑调用另一个大模型。

## 交付产物

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **简单。** 在 20 个手工构造的（前提、假设、标签）三元组上运行 `facebook/bart-large-mnli`，覆盖全部三类。测量准确率。加入对抗性"子序列启发式"陷阱（如 "I did not eat the cake" vs "I ate the cake"），观察模型是否会崩溃。
2. **中等。** 在 100 条 AG News 标题上比较零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"`，报告准确率波动。
3. **困难。** 搭建一个 RAG 忠实度检查器：原子主张拆分 + 每个主张做 NLI。在 50 条带黄金上下文的 RAG 生成答案上评估，相对于人工标注测量假正率与假负率。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| NLI | Natural Language Inference | 前提-假设关系的三分类任务。 |
| RTE | Recognizing Textual Entailment | NLI 的旧称；任务相同。 |
| Entailment | "t implies h" | 典型读者在已知 t 时会认为 h 为真。 |
| Contradiction | "t rules out h" | 典型读者在已知 t 时会认为 h 为假。 |
| Neutral | "undecided" | t 与 h 之间没有推断关系。 |
| Zero-shot classification | NLI as classifier | 把标签表达为假设，取最大蕴含概率。 |
| Faithfulness | 答案是否有支撑？ | 在（检索上下文，生成答案）上做 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) —— SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) —— MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) —— ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) —— NLI 作为分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) —— 2026 年 NLI 的主力模型。
