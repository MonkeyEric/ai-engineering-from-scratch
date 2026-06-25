# 指代消解（Coreference Resolution）

> "她给他打了电话。他没接。医生正在吃午饭。" 三个人称，两拨人，却无人被点名。指代消解负责搞清楚谁是谁。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 5 · 06（NER）、Phase 5 · 07（POS & Parsing）
**时长：** 约 60 分钟

## 问题背景

从一篇 300 词的文章中抽取出 Apple Inc. 的每一次提及。当文章说 "Apple" 时很容易；但当它说 "这家公司"、"他们"、"Cupertino 的科技巨头" 或 "Jobs 的公司" 时就难了。如果不把这些提及都归到同一个实体，你的 NER 管线会漏掉 60-80% 的提及。

指代消解把所有指向同一现实世界实体的表达链接到一个簇中。它是表层 NLP（NER、句法分析）与下游语义任务（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

2026 年它为何重要：

- 摘要："CEO 宣布了……" 与 "Tim Cook 宣布了……" —— 摘要应当点出 CEO 的名字。
- 问答："她给谁打了电话？" 需要先消解代词 "she"。
- 信息抽取：知识图谱里 "PER1 创立了 Apple" 和 "Jobs 创立了 Apple" 作为两条独立条目是错误的。
- 跨文档信息抽取：把同一事件的多篇文章中的提及合并起来，就是跨文档指代消解。

## 核心概念

![指代聚类：提及 → 实体](../assets/coref.svg)

**任务。** 输入：一篇文档。输出：提及（span）的聚类，每个簇对应一个实体。

**提及类型。**

- **命名实体。** "Tim Cook"
- **名词短语。** "CEO"、"这家公司"
- **代词。** "他"、"她"、"他们"、"它"
- **同位语。** "Tim Cook，Apple 的 CEO，"

**架构。**

1. **基于规则的方法（Hobbs, 1978）。** 利用句法树和语法规则进行代词消解。不错的基线。在代词上出人意料地强劲。
2. **提及对分类器。** 对每一对提及（m_i, m_j），预测它们是否共指。然后通过传递闭包聚类。2016 年前的标准做法。
3. **提及排序。** 对每个提及，为候选先行词（包括“无先行词”）打分，选择得分最高的。
4. **基于 span 的端到端（Lee et al., 2017）。** Transformer 编码器。枚举长度上限内的所有候选 span。预测提及分数。为每个 span 预测先行词概率。贪婪聚类。现代默认方案。
5. **生成式方法（2024+）。** 提示 LLM："列出这段文本中每个代词及其先行词。" 在简单案例上效果不错，在长文档和罕见指代上表现吃力。

**评估指标。** 五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为单一指标无法完全衡量聚类质量。通常报告前三个的平均值作为 CoNLL F1。2026 年在 CoNLL-2012 上的最先进水平：约 83 F1。

**已知的难点。**

- 指代数页之前引入的实体的定指描述。
- 桥接回指（"轮子" → 之前提到的一辆车）。
- 汉语、日语等语言中的零代词回指。
- 预指（代词在指代对象之前）："当 **她** 走进来时，玛丽笑了。"

## 动手实践

### 步骤 1：预训练神经指代消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长的文档上，你会得到类似：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### 步骤 2：基于规则的代词消解器（教学用）

参见 `code/main.py` 中的纯标准库实现：

1. 提取提及：命名实体（首字母大写的片段）、代词（字典查找）、定指描述（"the X"）。
2. 对每个代词，查看前 K 个提及并按以下规则打分：
   - 性别/数一致（启发式）
   - 就近原则（越近越好）
   - 句法角色（主语优先）
3. 链接得分最高的先行词。

无法与神经模型竞争。但它能展示搜索空间以及端到端模型必须做出的决策。

### 步骤 3：使用 LLM 做指代消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

需要注意两种失效模式。第一，LLM 会过合并（把指向两个不同人的 "him" 和 "her" 归为一类）。第二，LLM 会在长文档中静默遗漏提及。务必用 span offset 校验。

### 步骤 4：评估

标准 CoNLL-2012 脚本会计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，先从带标注测试集上的 span-level 精确率与召回率开始，再加入 mention-linking F1。

## 常见陷阱

- **单例爆炸。** 某些系统把每个提及都报告为独立簇。B³ 对此较宽容，MUC 会惩罚。务必三个指标都检查。
- **长上下文中的代词。** 在超过 2000 个 token 的文档上性能下降约 15 F1。要谨慎分块。
- **性别假设。** 硬编码的性别规则在非二元指代、机构、动物等场景下会失效。使用学习模型或中性打分。
- **LLM 在长文档上的漂移。** 单次 API 调用无法可靠地聚类跨越 50 多个段落的提及。使用滑动窗口 + 合并。

## 应用

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 英文单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP neural coref |
| 多语言 | 在 OntoNotes 或多语言 CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件指代 | 专用端到端模型（2025–26 SOTA） |
| 快速 LLM 基线 | GPT-4o / Claude，配合结构化输出的指代消解提示 |
| 生产对话系统 | 规则兜底 + 神经主模型 + 关键槽位人工复核 |

2026 年实际落地的集成模式：先跑 NER，再跑指代消解，然后把指代簇合并进 NER 实体。下游任务看到的是每个簇一个实体，而不是每个提及一个实体。

## 交付

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在 5 个手工编写的段落上运行 `code/main.py` 中的基于规则的消解器。对照真实标签测量 mention-link 准确率。
2. **中等。** 在新闻文章上使用预训练神经指代消解模型。将聚类结果与你的人工标注对比。它在哪里出错？
3. **困难。** 构建一个指代增强的 NER 管线：先做 NER，再通过指代簇合并。在 100 篇文章上测量实体覆盖率相比纯 NER 的提升。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Mention | 一次指称 | 指向实体的文本片段（名称、代词、名词短语）。 |
| Antecedent | "it" 指代什么 | 后文提及与之共指的先前提及。 |
| Cluster | 某个实体的所有提及 | 全部指向同一现实世界实体的提及集合。 |
| Anaphora | 回指 | 后文指向前文（"he" → "John"）。 |
| Cataphora | 预指 | 前文指向後文（"当他到达时，约翰……"）。 |
| Bridging | 隐式指代 | "我买了一辆车。轮子很糟糕。"（那辆车的轮子。） |
| CoNLL F1 | 榜单上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 第 26 章 — 指代消解与实体链接](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 权威教材章节。
- [Lee et al. (2017). 端到端神经指代消解](https://arxiv.org/abs/1707.07045) — 基于 span 的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 提升指代消解的预训练。
- [Pradhan et al. (2012). CoNLL-2012 共享任务](https://aclanthology.org/W12-4501/) — 基准测试。
- [Hobbs (1978). 代词指代消解](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典方法。
