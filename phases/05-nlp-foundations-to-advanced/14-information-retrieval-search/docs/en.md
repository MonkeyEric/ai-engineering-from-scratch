# 信息检索与搜索

> BM25 精确但脆弱。密集检索覆盖面广但会漏掉关键词。混合检索是 2026 年的默认方案。其余都是调优。

**类型：** 构建
**语言：** Python
**先修知识：** 阶段 5 · 02（词袋 + TF-IDF），阶段 5 · 04（GloVe、FastText、子词）
**时长：** 约 75 分钟

## 问题背景

用户输入 "what happens if someone lies to get money"，并期望找到实际相关的法条："Section 420 IPC"。关键词搜索完全失效（没有共享词汇）。语义搜索也失效，如果嵌入不是用法律文本训练的。真正的搜索必须同时兼顾两者。

信息检索（IR）是每个 RAG 系统、每个搜索框、每个文档站点模糊查找背后的管道。2026 年能在生产环境落地的架构不是单一方法，而是一系列互补方法的组合，每一种都能弥补前一种的失败。

本课逐步构建每个模块，并指出每种方法能捕捉哪些失败场景。

## 核心概念

![混合检索：BM25 + 密集检索 + RRF + 交叉编码器重排序](../assets/retrieval.svg)

四层结构。按需选择。

1. **稀疏检索（BM25）。** 速度快，精确匹配表现好，语义理解差。基于倒排索引运行。在数百万文档上每次查询低于 10 毫秒。能准确找到法条引用、产品代码、错误信息、命名实体。
2. **密集检索。** 将查询和文档编码成向量。通过最近邻搜索查找。能捕捉改写和语义相似性。对只差一个字符的精确关键词匹配会失效。使用 FAISS 或向量数据库时每次查询 50-200 毫秒。
3. **融合。** 合并稀疏和密集检索的排序结果。倒数排序融合（RRF）是简单的默认选择，因为它忽略原始分数（不同方法的分数尺度不同），只使用排名位置。当某个信号在你的领域占主导时，也可以选择加权融合。
4. **交叉编码器重排序。** 取融合后的前 30 个结果。运行交叉编码器（将查询和文档一起输入，为每对打分）。保留前 5 个。交叉编码器每对推理比双编码器慢，但准确率高得多。只对前 30 个运行，可以摊平成本。

三路检索（BM25 + 密集 + 学习式稀疏如 SPLADE）在 2026 年的基准测试中优于两路，但需要为学习式稀疏索引提供基础设施。对大多数团队来说，两路加交叉编码器重排序是最佳平衡点。

## 动手构建

### 步骤 1：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数。`k1=1.5` 控制词频饱和度；值越大，重复词项权重越高。`b=0.75` 控制长度归一化；0 表示忽略文档长度，1 表示完全归一化。默认值来自原始论文中 Robertson 的推荐，通常无需调整。

### 步骤 2：用双编码器实现密集检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入做 L2 归一化，使点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，速度快，对大多数英文检索已经足够强。多语言场景使用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高准确率时，使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤 3：倒数排序融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 这个常数来自原始 RRF 论文。`k` 越大，排名差异带来的贡献越平缓；`k` 越小，靠前的排名越占主导。60 是文献中的默认推荐，通常无需调整。

### 步骤 4：混合搜索 + 重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段的组合。BM25 找到词汇匹配。密集检索找到语义匹配。RRF 无需分数校准即可合并两个排序。交叉编码器对前 30 个结果使用查询-文档对重新打分，能捕捉双编码器遗漏的细粒度相关性。保留前 5 个。

### 步骤 5：评估

| 指标 | 含义 |
|------|------|
| Recall@k | 在存在正确文档的查询中，正确文档出现在前 k 个的比例是多少？ |
| MRR（平均倒数排名） | 第一个相关文档排名的倒数平均值。 |
| nDCG@k | 考虑相关性等级，而不仅仅是二元的相关/不相关。 |

对于 RAG 来说，检索器的 **Recall@k** 是最重要的数字。如果正确的段落不在检索结果中，阅读器就无法回答。

调试提示：对于失败的查询，对比稀疏和密集排序结果。如果其中一个找到了正确文档而另一个没有，说明是词汇不匹配（修复：补上缺失的一半）或语义歧义（修复：更好的嵌入或重排序器）。

## 应用

2026 年的技术栈：

| 规模 | 技术栈 |
|------|--------|
| 1k-100k 文档 | 内存中的 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。不需要独立数据库。 |
| 100k-1000 万文档 | FAISS 或 pgvector 做密集检索 + Elasticsearch / OpenSearch 做 BM25。并行运行。 |
| 1000 万+ 文档 | Qdrant / Weaviate / Vespa / Milvus 等支持混合检索的系统。对前 30 个结果做交叉编码器重排序。 |
| 最佳质量前沿 | 三路（BM25 + 密集 + SPLADE）+ ColBERT 后期交互重排序 |

无论你选择哪种方案，都要为评估预留预算。在基准测试端到端 RAG 准确率之前，先基准测试检索召回率。阅读器无法修复检索器漏掉的内容。

### 2026 年生产 RAG 来之不易的经验

- **80% 的 RAG 失败可以追溯到数据摄取和分块，而不是模型。** 团队花数周时间更换大模型和调整提示词，而检索每三个查询就悄悄返回错误的上下文。先修复分块问题。
- **分块策略比分块大小更重要。** 固定大小的切分会破坏表格、代码和嵌套标题。句子感知是默认方案；语义分块或基于大模型的分块对技术文档和产品手册更有价值。
- **父文档模式。** 检索小的"子"块以获得精度。当同一父章节的多个子块出现时，替换为完整的父块以保留上下文。这能在不重新训练的情况下持续提升回答质量。
- **k_rerank=3 通常最优。** 超过这个数量的每个额外块都会增加 token 成本和生成延迟，却不会提升回答质量。如果 k=8 对你来说仍然比 k=3 好，说明重排序器表现不佳。
- **HyDE / 查询扩展。** 从查询生成一个假设答案，对该答案做嵌入，然后进行检索。弥合短问题与长文档之间的措辞鸿沟。无需训练即可免费提升精度。
- **上下文预算控制在 8K token 以内。** 持续达到这个上限意味着重排序阈值过松。
- **所有东西都要版本化。** 提示词、分块规则、嵌入模型、重排序器。任何漂移都会悄悄破坏回答质量。在忠实度、上下文精度和未回答问题率上设置 CI 门禁，防止用户受到影响。
- **三路检索（BM25 + 密集 + 学习式稀疏如 SPLADE）优于两路**，在 2026 年的基准测试中尤其擅长处理混合了专有名词和语义的查询。当基础设施支持 SPLADE 索引时，应投入使用。

根据 2026 年的行业测算，合理的检索设计可将幻觉降低 70-90%。大多数 RAG 性能提升来自更好的检索，而不是模型微调。

## 交付

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: 根据给定的语料库和查询模式选择检索技术栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

给定需求（语料库规模、查询模式、延迟预算、质量要求、基础设施限制），输出：

1. 技术栈。仅 BM25、仅密集检索、混合（BM25 + 密集 + RRF）、混合 + 交叉编码器重排序，或三路（BM25 + 密集 + 学习式稀疏）。
2. 密集编码器。指定具体模型。根据语言、领域和上下文长度进行匹配。
3. 重排序器。如果使用，指定具体的交叉编码器模型。注意重排序会在前 30 个结果上增加 30-100ms 延迟。
4. 评估计划。Recall@10 是检索器的主要指标。多答案场景使用 MRR。先建立基线，再针对基线衡量增量改进。

对于包含命名实体、错误代码或产品 SKU 的语料库，除非用户有证据表明密集检索能处理精确匹配，否则拒绝推荐纯密集检索。对于高风险检索（法律、医疗），如果最终前 5 个结果决定用户答案，则拒绝跳过重排序。
```

## 练习

1. **简单。** 在上述 500 篇文档的语料库上实现 `hybrid_search`。测试 20 个查询。比较 BM25 单独、密集单独和混合三者的 Recall@5。
2. **中等。** 添加 MRR 计算。对于每个已知有正确文档的测试查询，找出正确文档在 BM25、密集和混合排序中的排名。报告每种方法的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在领域数据上微调密集编码器。从 500 个查询-文档对构建训练集。比较微调前后的召回率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| BM25 | 关键词搜索 | Okapi BM25。根据词频、IDF 和文档长度对文档打分。 |
| 密集检索 | 向量搜索 | 将查询和文档编码成向量，找到最近邻。 |
| 双编码器 | 嵌入模型 | 独立编码查询和文档。查询时速度快。 |
| 交叉编码器 | 重排序模型 | 将查询和文档一起编码。慢但准确。 |
| RRF | 排序融合 | 通过求和 `1/(k + rank)` 合并两个排序。 |
| Recall@k | 检索指标 | 相关文档出现在前 k 个的查询比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威综述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典的双编码器。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩小与密集检索差距的学习式稀疏检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 后期交互检索。
