# 关系抽取与知识图谱构建

> NER 找出了实体，实体链接确定了它们的指代，关系抽取则发现它们之间的边。知识图谱就是节点、边及其来源的总和。

**类型：** Build
**语言：** Python
**前置条件：** Phase 5 · 06（NER），Phase 5 · 25（实体链接）
**时间：** ~60 分钟

## 问题背景

一位分析师读到：“Tim Cook became CEO of Apple in 2011.” 其中包含四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（RE）将自由文本转换为结构化的三元组 `(subject, relation, object)`。跨语料聚合后就形成了知识图谱。再加以聚合与查询，就能为 RAG、分析或合规审计提供推理基础。

2026 年的问题在于：大语言模型抽取关系时过于积极。它们会幻想出源文本并未支持的三元组。没有来源依据，就无法区分真实三元组和看似合理的虚构信息。2026 年的答案是 AEVS 风格的“锚定-验证”流程。

## 核心概念

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系可以来自封闭本体（Wikidata 属性、FIBO、UMLS），也可以是开放集合（OpenIE 风格，任意关系）。

**三种抽取方法。**

1. **基于规则 / 模式。** Hearst 模式："X such as Y" → `(Y, isA, X)`，再加上手工正则。脆弱、精确、可解释。
2. **监督分类器。** 给定句子中的两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 年的标准做法。
3. **生成式大语言模型。** 通过提示让模型输出三元组。开箱即用，但必须有来源依据，否则会幻想出看似合理的垃圾结果。

**AEVS（Anchor-Extraction-Verification-Supplement，2026）。** 当前用于缓解幻觉的框架：

- **Anchor（锚定）。** 识别每一个实体片段和关系短语片段，并记录其精确位置。
- **Extract（抽取）。** 生成与锚定片段相关联的三元组。
- **Verify（验证）。** 将每个三元组元素与源文本匹配；拒绝任何没有文本支持的内容。
- **Supplement（补充）。** 通过覆盖性检查确保没有锚定片段被遗漏。

幻觉会大幅下降。计算成本更高，但结果可审计。

**开放 vs 封闭的权衡。**

- **封闭本体。** 固定属性列表（例如 Wikidata 的 11,000 多个属性）。可预测、可查询、难以随意发明。
- **开放信息抽取（Open IE）。** 任何动词短语都可作为关系。召回率高、精确率低、查询困难。

生产级知识图谱通常混合使用：先用开放 IE 发现关系，再将其规范化为封闭本体，最后合并入主图。

## 动手实现

### 步骤 1：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的小型抽取器见 `code/main.py`。Hearst 模式仍会用于领域特定流程，因为它们易于调试。

### 步骤 2：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一种 seq2seq 关系抽取器：文本进，三元组出，且已使用 Wikidata 属性编号。在远程监督数据上微调。是标准的开源基线模型。

### 步骤 3：带锚定的大语言模型提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

将返回的每个片段与源文本进行验证。如果 `text[start:end] != triple_entity`，则拒绝。这就是 AEVS 中“验证”步骤的最简形式。

### 步骤 4：规范化为封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化通常占工程工作量的 60%–80%。务必为此预留预算。

### 步骤 5：构建小型图谱并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是所有“基于知识图谱的 RAG”系统的最小单元。扩展到 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储即可。

## 常见陷阱

- **关系抽取前需做指代消解。** “He founded Apple”——关系抽取需要知道“he”是谁。先运行指代消解（第 24 课）。
- **实体规范化。** “Apple Inc”和“Apple”必须解析为同一节点。先做实体链接（第 25 课）。
- **幻觉三元组。** 大语言模型会输出文本不支持的三元组。必须执行片段验证。
- **关系规范化漂移。** 开放 IE 的关系表述不一致（"was born in"、"came from"、"is a native of"）。必须折叠为规范编号，否则图谱无法查询。
- **时间错误。** “Tim Cook is CEO of Apple”现在为真，但在 2005 年为假。许多关系具有时间边界。使用限定符（Wikidata 中的 `P580` 起始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医学和科学文本通常需要领域微调的关系抽取模型。

## 实际应用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速上线、通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域特定（生物医学、法律） | SciREX 风格领域微调 + 自定义本体 |
| 大模型提示抽取、可审计输出 | AEVS 流程：锚定 → 抽取 → 验证 → 补充 |
| 高吞吐量新闻信息抽取 | 基于模式 + 监督模型混合 |
| 从零构建知识图谱 | 开放 IE + 人工规范化环节 |
| 时态知识图谱 | 抽取时附带限定符（起始/结束时间、时间点） |

集成流程：NER → 指代消解 → 实体链接 → 关系抽取 → 本体映射 → 图谱加载。每个阶段都是潜在的质量关卡。

## 交付物

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习题

1. **简单。** 对 5 句新闻文章句子运行 `code/main.py` 中的模式抽取器。手工检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或一个小型大语言模型）。比较三元组。哪个抽取器精确率更高？召回率更高？
3. **困难。** 构建 AEVS 流程：用大语言模型抽取 + 对源文本验证片段。在 50 句 Wikipedia 风格句子上，测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Triple（三元组） | Subject-relation-object | 知识图谱的原子单位 `(s, r, o)`。 |
| Open IE（开放信息抽取） | 抽取任何关系 | 开放词汇的动词关系短语；高召回、低精确。 |
| Closed ontology（封闭本体） | 固定模式 | 有限的关系类型集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization（规范化） | 全部归一化 | 将表面名称 / 关系统一映射到规范编号。 |
| AEVS | 有依据的抽取 | Anchor-Extraction-Verification-Supplement 流程（2026）。 |
| Provenance（来源依据） | 源-真相链接 | 每个三元组都带有文档编号和字符片段指向其源文本。 |
| Distant supervision（远程监督） | 廉价标签 | 将文本与已有知识图谱对齐以生成训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) —— 远程监督奠基论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) —— seq2seq 关系抽取主力模型。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) —— 联合信息抽取。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) —— 2026 年缓解幻觉的设计框架。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) —— 规范化图谱查询教程。
