# 多语言自然语言处理

> 一个模型，100 多种语言，其中大多数语言无需训练数据。跨语言迁移是 2020 年代最实用的奇迹。

**类型：** 学习
**语言：** Python
**先修：** 第 5 阶段 · 04（GloVe、FastText、子词），第 5 阶段 · 11（机器翻译）
**时长：** 约 45 分钟

## 问题背景

英语拥有数十亿带标注样本。乌尔都语只有数千条。迈蒂利语几乎没有。任何面向全球用户的实用 NLP 系统，都必须能在长尾语言上工作，而这些语言往往没有任务专属的训练数据。

多语言模型通过在多种语言上同时训练一个模型来解决这个问题。共享的表示让模型把在高资源语言中学到的能力迁移到低资源语言。用英语情感分析微调模型后，它在乌尔都语上直接就能给出相当不错的情感预测。这就是零样本跨语言迁移，它已经重塑了 NLP 在全球落地的方式。

本课会说明其中的权衡、经典模型，以及新手团队在多语言工作中最容易踩坑的一点：选择用于迁移的源语言。

## 核心概念

![通过共享多语言嵌入空间实现跨语言迁移](../assets/multilingual.svg)

**共享词表。** 多语言模型使用在全部目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词表是共享的：相同的子词单元在不同但相关的语言中表示相同的语素。英语和意大利语里的 `anti-` 会得到同一个 token。

**共享表示。** 在多种语言上做掩码语言建模预训练的 Transformer 会发现，不同语言中语义相近的句子会产生相近的隐藏状态。mBERT、XLM-R 和 NLLB 都表现出这一特性。英语中 "cat" 的嵌入与法语 "chat"、西班牙语 "gato" 聚在一起，整句嵌入也是如此。

**零样本迁移。** 用一种语言（通常是英语）的标注数据微调模型。推理时直接用于模型支持的任何其他语言。不需要目标语言的标注。类型相近的语言效果好，差异大的语言效果差。

**少样本微调。** 加入 100-500 条目标语言标注样本，分类任务准确率通常能达到英语基线的 95-98%。这是多语言 NLP 中性价比最高的单一杠杆。

## 经典模型

| 模型 | 年份 | 覆盖语言 | 说明 |
|------|------|----------|------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。第一个实用的多语言语言模型。低资源表现弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（规模远大于 Wikipedia）。确立了跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R 的 100 万 token 词表（原 25 万）。低资源语言上更好。 |
| mT5 | 2020 | 101 种语言 | T5 架构的多语言生成模型。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 开放的多语言 176B 大语言模型。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言大语言模型。阿拉伯语、印地语、斯瓦希里语上表现强。 |

按用例选择。分类任务稳妥默认用 XLM-R-base。生成任务看是翻译还是开放生成，分别用 mT5 或 NLLB。大模型风格的工作可搭配 Aya-23 或 Claude，并显式使用多语言提示。

## 源语言选择（2026 年研究）

大多数团队默认用英语做微调源语言。近期研究（2026 年）表明这通常是错的。

语言相似度比原始语料规模更能预测迁移质量。对斯拉夫语目标，德语或俄语常优于英语。对印度语目标，印地语常优于英语。基于世界语言结构地图集（WALS）特征的 **qWALS** 相似度指标（2026 年）可以量化这一点。**LANGRANK**（Lin 等人，ACL 2019）是另一种更早的方法，综合语言相似度、语料规模和谱系关系对候选源语言排序。

实用规则：如果目标语言有一个类型学上相近的高资源亲缘语言，先尝试用它微调，再与英语微调对比。

## 动手实现

### 步骤 1：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一套 API。在 NLI 数据上训练的 XLM-R 通过蕴含技巧很好地迁移到分类任务。

### 步骤 2：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译后的句子在嵌入空间中距离很近，而不同的英语句子距离更远。这正是跨语言检索、聚类和相似度计算得以成立的原因。

### 步骤 3：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 条目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是稳妥默认值。学习率过高会导致多语言对齐崩塌，模型退化为仅懂英语。

## 真正有效的评估

- **按语言分开的留出集准确率。** 不要只报总体指标。总体指标会掩盖长尾问题。
- **与单语基线对比。** 对于数据充足的语言，从零训练的单语模型有时优于多语言模型。需要实测。
- **实体级测试。** 目标语言中的命名实体。多语言模型对非拉丁文字的切分通常较弱。
- **跨语言一致性。** 同一含义用两种语言表达应得到相同预测。量化这个差距。

## 应用指南

2026 年推荐技术栈：

| 任务 | 推荐方案 |
|-----|-------------|
| 分类，100 种语言 | 微调的 XLM-R-base（约 270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 生成式多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相近高资源语言上做领域微调 |

如果性能重要，总要为目标语言微调预留预算。零样本只是起点，不是最终答案。

### 分词器代价（低资源语言容易出问题的地方）

多语言模型在所有语言间共享一个分词器。这个词表是在英语、法语、西班牙语、中文、德语占主导的语料上训练的。对于主导集合之外的语言，三种代价会悄然叠加：

- ** fertility 代价。** 低资源语言文本每个词会被切分成比英语多得多的 token。一句印地语可能需要同等英语句子 3-5 倍的 token。这 3-5 倍会吃掉上下文窗口、训练效率和推理延迟。
- **变体恢复代价。** 每个拼写错误、变音符号变体、Unicode 归一化不一致或大小写差异，都会在嵌入空间中变成冷启动的无关序列。模型无法学会母语者眼中显而易见的拼写对应关系。
- **容量溢出代价。** 前两种代价会占用上下文位置、层深度和嵌入维度。留给真正推理的资源，系统性地少于高资源语言从同一模型中获得的资源。

实际症状：模型在印地语上训练正常，loss 曲线看起来对，评估困惑度也合理，但线上输出却微妙地出错。句中形态会崩解，罕见变位无法恢复。**分词器坏了，靠堆数据是解决不了的。**

缓解措施：选择对目标语言覆盖好的分词器（XLM-V 的 100 万 token 词表是直接修复）；训练前在目标语言留出文本上验证分词 fertility；对真正长尾的文字使用字节级回退（SentencePiece `byte_fallback=True`、GPT-2 风格的字节级 BPE），确保没有任何字符是 OOV。

## 交付

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语、阿拉伯语上各运行 10 句零样本分类。分别报告每种语言的准确率。你会看到法语很强、印地语尚可、阿拉伯语波动较大。
2. **中等。** 用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型多语言语料上构建跨语言检索器。用英语查询，检索任意语言的文档。测量 recall@5。
3. **困难。** 对一项印地语分类任务，对比英语源微调和印地语源微调。在两种设定下都用 500 条目标语言样本做少样本微调。报告哪种源语言得到更高的印地语准确率，高出多少。这是 LANGRANK 论点的微缩版。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Multilingual model | 一个模型，多种语言 | 跨语言共享词表和参数。 |
| Cross-lingual transfer | 用一种语言训练，在另一种语言上运行 | 在源语言微调，无目标语言标注地在目标语言评估。 |
| Zero-shot | 没有目标语言标注 | 不在目标语言上微调的直接迁移。 |
| Few-shot | 少量目标语言标注 | 用 100-500 条目标语言样本进行微调。 |
| mBERT | 第一个多语言语言模型 | 在 Wikipedia 上预训练的 104 种语言 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练的 100 种语言 RoBERTa。 |
| NLLB | Meta 的 200 种语言机器翻译 | No Language Left Behind。包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) —— XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) —— 开启跨语言迁移研究路线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) —— NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) —— Aya，Cohere 的多语言大语言模型。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) —— qWALS / LANGRANK 源语言论文。
