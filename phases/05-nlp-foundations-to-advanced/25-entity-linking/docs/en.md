# 实体链接与消歧

> NER 识别出了 "Paris"。实体链接要判断：Paris, France？Paris Hilton？Paris, Texas？Paris（特洛伊王子）？没有链接，你的知识图谱将始终充满歧义。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 06（NER），Phase 5 · 24（Coreference Resolution）
**时间：** ~60 分钟

## 问题背景

句子写道："Jordan beat the press." 你的 NER 把 "Jordan" 标注为 PERSON，很好。但*是哪一个* Jordan？

- Michael Jordan（篮球运动员）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利机器学习教授——在 ML 论文里这种混淆真实存在）？
- Jordan（国家）？
- Jordan（希伯来语名字）？

实体链接（EL）将每个提及解析为知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域 KB。它包含两个子任务：

1. **候选生成。** 给定 "Jordan"，哪些 KB 条目是合理的？
2. **消歧。** 给定上下文，哪个候选是正确的？

两个步骤都可以学习，也都有基准测试。组合 pipeline 已经稳定运行了十年——变化的是消歧器的质量。

## 核心概念

![实体链接 pipeline：mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**候选生成。** 给定提及的表面形式（"Jordan"），在别名索引中查找候选。Wikipedia 别名字典覆盖了大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、JFK（电影）。典型索引每个 mention 返回 10–30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、无需训练。
2. **基于嵌入（ESS / REL / BLINK）。** 编码 mention + 上下文；编码每个候选的描述；选择余弦相似度最大者。2020–2024 年的默认方案。
3. **生成式（GENRE, 2021；基于 LLM, 2023+）。** 逐 token 解码实体的规范名称。通过 trie 约束，保证输出一定是有效的 KB id。

**端到端 vs 流水线。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一个前向过程中完成 NER + 候选生成 + 消歧。流水线系统仍主导生产环境，因为你可以单独替换组件。

### 两项评估指标

- **提及召回（候选生成）。** 正确 KB 条目出现在候选列表中的黄金提及比例。这是整个 pipeline 的下限。
- **消歧准确率 / F1。** 在正确候选存在的前提下，top-1 选对的频率。

两个都要报告。一个在 80% 候选召回上达到 99% 消歧的系统，实际只是 80% 的 pipeline。

## 动手实现

### 步骤 1：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：约 1800 万（alias, entity）对。从 Wikidata dump 下载，存储为倒排索引。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠只是一个 toy 示例。请替换为基于嵌入的余弦相似度（参见 `code/main.py` 中的 step-2 transformer 版本）。

### 步骤 3：基于嵌入的方法（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引阶段，每个 KB 实体只编码一次；在查询阶段，对 mention + 上下文编码一次，与候选池做点积，选择最大值。

### 步骤 4：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（参见第 20 课）确保只输出有效标题。它与基于 KB 的 trie 紧密结合。现代后继包括 REL-GEN 以及使用结构化输出的 LLM 提示式 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年最容易落地的 EL pipeline。

### 步骤 5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准的 EL 基准：1,393 篇 Reuters 文章，34k 个提及，Wikipedia 实体。报告 in-KB 准确率（`P@1`）和 out-of-KB 的 NIL 检测率。

## 常见陷阱

- **NIL 处理。** 有些提及不在 KB 中（新兴实体、冷门人物）。系统必须预测 NIL，而不是胡乱猜一个错误实体。需单独评估。
- **提及边界错误。** 上游 NER 漏掉部分跨度（如把 "Bank of America" 只标成 "Bank"）会导致 EL 召回下降。
- **流行度偏差。** 训练后的系统倾向于预测高频实体。在 ML 论文中提到 "Michael I. Jordan" 时，常被链接到篮球 Jordan。
- **跨语言 EL。** 将中文文本中的提及映射到英文 Wikipedia 实体。需要多语言编码器或翻译步骤。
- **KB 过时。** 新公司、事件、人物不会出现在去年的 Wikipedia dump 中。生产 pipeline 需要刷新机制。

## 如何使用

2026 年的技术栈：

| 场景 | 推荐方案 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM 友好，提及量小 | 用候选列表 + 约束 JSON 提示 Claude/GPT-4 |
| 领域 KB（医学、法律） | 自定义 BERT + KB 感知检索，在领域 AIDA 风格数据集上微调 |
| 极低延迟 | 仅使用精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年可落地的生产模式：NER → 共指消解 → 对每个提及做 EL → 将聚类 collapsed 为每个聚类一个规范实体。输出：文档中每个实体一个 KB id，而不是每个提及一个。

## 交付物

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: 设计一个实体链接 pipeline——KB、候选生成器、消歧器、评估。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

给定一个用例（领域 KB、语言、规模、延迟预算），输出：

1. Knowledge base。Wikidata / Wikipedia / 自定义 KB。版本日期。刷新周期。
2. Candidate generator。别名索引、嵌入或混合。目标 mention recall @ K。
3. Disambiguator。先验 + 上下文、基于嵌入、生成式或 LLM 提示。
4. NIL strategy。对 top score 设阈值、分类器或显式 NIL 候选。
5. Evaluation。Mention recall @ 30、top-1 准确率、NIL 检测 F1 在留出集上。

拒绝任何没有 mention-recall 基线的 EL pipeline（如果候选生成没有把正确实体放进列表，你就无法评估消歧器）。拒绝任何未将输出约束为有效 KB id 的 LLM 提示式 EL pipeline。标记那些因流行度偏差而影响少数实体（例如同名冲突）且未做领域微调的系统。
```

## 练习

1. **简单。** 在 `code/main.py` 中为 10 个有歧义的提及（Paris、Jordan、Apple）实现先验 + 上下文消歧器。手工标注正确实体并测量准确率。
2. **中等。** 用 sentence transformer 编码 50 个有歧义的提及，并为每个候选的描述生成嵌入。比较基于嵌入的消歧与 Jaccard 上下文重叠。
3. **困难。** 构建一个包含 1k 实体的领域 KB（例如公司里的员工 + 产品）。实现端到端 NER + EL，在 100 句留出句子上测量精确率和召回率。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Entity linking (EL) | 链接到 Wikipedia | 将一个提及映射到唯一的 KB 条目。 |
| Candidate generation | 可能是谁？ | 返回提及对应的合理 KB 条目短名单。 |
| Disambiguation | 选对那个 | 利用上下文给候选打分并选出胜者。 |
| Alias index | 查找表 | 从表面形式 → 候选实体的映射。 |
| NIL | 不在 KB 中 | 显式预测没有 KB 条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域 KB。 |
| AIDA-CoNLL | 基准 | 1,393 篇带 gold 实体链接的 Reuters 文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) —— 基础性的先验 + 上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) —— 基于嵌入的主力方法。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) —— 带约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) —— 基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) —— 开源生产级 stack。
