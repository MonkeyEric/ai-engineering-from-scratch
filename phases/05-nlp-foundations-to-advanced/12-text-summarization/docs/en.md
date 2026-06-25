# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者想表达什么。任务不同，陷阱也不同。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 02（词袋 + TF-IDF）、阶段 5 · 11（机器翻译）
**时间：** 约 75 分钟

## 问题描述

一篇 2,000 词的新闻文章出现在你的信息流中，你需要一段 120 词的摘要。你可以选择从文章中挑选三个最重要的句子（抽取式），也可以用自己的话重写内容（生成式）。两者都被称为摘要，但它们是完全不同的问题。

抽取式摘要是一个排序问题。为每个句子打分，返回前 `k` 个。输出总是语法正确的，因为它是原样摘取的。风险在于可能遗漏分散在文章各处的信息。

生成式摘要是一个生成问题。Transformer 根据输入生成新文本。输出流畅且压缩度高，但可能幻觉出源文本中没有的事实。风险在于它会自信地编造内容。

本节课将构建两种摘要系统，并明确它们各自拥有的失败模式。

## 核心概念

![抽取式 TextRank 与生成式 Transformer 对比](../assets/summarization.svg)

**抽取式。** 将文章视为一张图，节点是句子，边是句子之间的相似度。在图上运行 PageRank（或类似算法），根据每个句子与其他句子的连接程度打分。得分最高的句子构成摘要。经典实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式。** 在文档-摘要对上微调 Transformer 编码器-解码器模型（BART、T5、Pegasus）。推理时，模型通过交叉注意力逐词阅读文档并生成摘要。Pegasus 尤其采用 gap-sentence 预训练目标，使其即使不经太多微调也能胜任摘要任务。

使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）进行评估。ROUGE-1 和 ROUGE-2 分别衡量一元词组和二元词组的重叠；ROUGE-L 衡量最长公共子序列。分数越高越好；ROUGE-L 达到 40 算“不错”，50 算“优秀”。每篇论文都会同时报告这三个指标。使用 `rouge-score` 包。

## 动手实现

### 步骤 1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两个细节值得一提。相似度函数使用对数归一化的词重叠，这是 TextRank 的原始变体。TF-IDF 向量的余弦相似度同样有效。阻尼系数 0.85 和迭代次数是 PageRank 的默认设置。

### 步骤 2：使用 BART 实现生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上进行了微调，开箱即可生成新闻风格的摘要。对于其他领域（科学论文、对话、法律文本），请使用对应的 Pegasus 检查点，或在自己的目标数据上微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

一定要使用词干还原。没有它的话，"running" 和 "run" 会被算作不同的词，导致 ROUGE 低估。

### 超越 ROUGE（2026 年的摘要评估）

二十年来 ROUGE 一直是摘要任务的主导指标，但在 2026 年它本身已经不够用了。一项针对 NLG 论文的大规模元分析显示：

- **BERTScore**（上下文嵌入相似度）自 2023 年起逐渐被广泛采用，现在大多数摘要论文都会与 ROUGE 一起报告。
- **BARTScore** 将评估视为生成任务：通过预训练 BART 对给定源文档生成该摘要的概率来打分。
- **MoverScore**（基于上下文嵌入的推土机距离）在 2025 年的摘要基准测试中登顶，因为它比 ROUGE 更能捕捉语义重叠。
- **FactCC** 和 **基于 QA 的忠实度评估** 在 2021–2023 年很常见，现在常被 **G-Eval** 取代（一种 GPT-4 提示链，通过思维链推理对连贯性、一致性、流畅性和相关性打分）。
- 当评分标准设计良好时，**G-Eval** 等 LLM 裁判方法与人类判断的一致性约为 80%。

生产建议：报告 ROUGE-L 以便与历史结果对比，报告 BERTScore 以衡量语义重叠，报告 G-Eval 以衡量连贯性和事实性。用 50–100 条人工标注的摘要进行校准。

### 步骤 4：事实性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险要低得多，因为输出是从源文本原样摘取的——但如果源句被断章取义、已经过时或顺序被打乱，仍然可能误导读者。这正是生产系统在处理合规相关内容时仍倾向于使用抽取式方法的最大原因。

需要命名的幻觉类型：

- **实体替换。** 源文本说 "John Smith"，摘要却说 "John Brown"。
- **数字漂移。** 源文本说 "25,000"，摘要却说 "25 million"。
- **极性翻转。** 源文本说 "rejected the offer"，摘要却说 "accepted the offer"。
- **事实捏造。** 源文本根本没提到 CEO，摘要却说 CEO 批准了。

有效的评估方法：

- **FactCC。** 一个在源句与摘要句之间的蕴含关系上训练的二分类器，预测事实性 / 非事实性。
- **基于 QA 的事实性评估。** 向 QA 模型提问，答案应在源文本中。如果摘要支持不同的答案，则标记为有问题。
- **实体级 F1。** 比较源文本与摘要中的命名实体。仅出现在摘要中的实体值得怀疑。

对于任何面向用户且事实性重要的场景（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式摘要必须在循环中加入事实性检查。

## 如何使用

2026 年的技术栈：

| 使用场景 | 推荐方案 |
|---------|-------------|
| 新闻，3–5 句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或微调后的 T5 |
| 多文档、长文本 | 任何支持 32k+ 上下文的 LLM，通过提示完成 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，幻觉风险天然较低 | TextRank 或 `sumy` 的 LSA / LexRank |

当计算资源不受限时，具备长上下文的 LLM 在 2026 年常常能击败专用模型。代价是成本和可复现性；专用模型的输出更稳定一致。

## 交付物

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: 选择抽取式或生成式，指定库名称，加入事实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定一个任务（文档类型、合规要求、长度、计算预算），输出：

1. 方法。抽取式或生成式。用一句话解释原因。
2. 起始模型 / 库。明确命名。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed` 或 LLM 提示。
3. 评估计划。ROUGE-1、ROUGE-2、ROUGE-L（使用带词干还原的 rouge-score）。如果是生成式，额外加入事实性检查。
4. 一个需要探测的失败模式。实体替换是生成式新闻摘要中最常见的；标记源实体未出现在摘要中的样本。

对于医疗、法律、金融或受监管内容，若无事实性关卡，应拒绝使用生成式摘要。若输入超过模型的上下文窗口，应标记为需要分块的 map-reduce 摘要，而不是简单截断。
```

## 练习

1. **简单。** 对 5 篇新闻文章运行 TextRank。将前 3 句与参考摘要对比，测量 ROUGE-L。在 CNN/DailyMail 风格的文章上，你应该能看到 30–45 的 ROUGE-L。
2. **中等。** 实现实体级事实性评估：从源文本和摘要中提取命名实体（spaCy），计算源实体在摘要中的召回率，以及摘要实体相对源文本的精确率。高精确率低召回率意味着安全但简略；低精确率意味着存在幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上对比 BART-large-CNN 与某个 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（按实体 F1）以及每条摘要的成本。记录各自在哪些情况下表现更好。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive | 挑选句子 | 从源文本中原样返回句子。不会幻觉。 |
| Abstractive | 重写 | 根据源文本生成新文本。可能幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考摘要之间的 N-gram / LCS 重叠。 |
| TextRank | 基于图的抽取式方法 | 在句子相似度图上运行 PageRank。 |
| Factuality | 是否正确 | 摘要中的论断是否被源文本支持。 |
| Hallucination | 编造内容 | 摘要中源文本未支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) —— 抽取式摘要的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) —— BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) —— Pegasus 与 gap-sentence 目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) —— ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) —— 事实性研究综述论文。
