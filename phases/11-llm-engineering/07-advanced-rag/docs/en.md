# 高级 RAG：分块、重排序与混合搜索

> 基础 RAG 只检索最相似的 top-k 分块。对于简单问题这没问题，但在多跳推理、模糊查询和大型语料库面前会崩溃。高级 RAG 决定了一个系统到底是只能在 10 份文档上运行的演示，还是能在 1 000 万份文档上稳定运行的生产系统。

**类型：** 构建
**语言：** Python
**先修知识：** Phase 11, Lesson 06（RAG）
**时长：** 约 90 分钟
**相关：** Phase 5 · 23（Chunking Strategies for RAG）涵盖了六种分块算法——递归分块、语义分块、句子分块、父文档分块、迟分块、上下文增强检索——并附有 Vectara/Anthropic 的评测基准。本课在此基础上继续深入：混合搜索、重排序、查询转换。

## 学习目标

- 实现高级分块策略（语义分块、递归分块、父子分块），保留文档结构与上下文
- 构建混合搜索流水线，结合 BM25 关键词匹配、语义向量搜索与交叉编码器重排序
- 应用查询转换技术（HyDE、多查询、回退）来改善模糊或复杂问题的检索效果
- 诊断并修复常见 RAG 失败：检索到错误分块、答案不在上下文中、多跳推理断裂

## 问题所在

你在第 06 课搭建了一个基础 RAG 流水线，它能回答小规模语料库上的直接问题。现在试试这些：

**模糊查询**："What was revenue last quarter?" 语义搜索会返回关于收入战略、收入预测、以及 CFO 对收入增长看法的分块。它们都与"revenue"这个词语义相似，但没有一个包含实际数字。正确的分块写着 "$47.2M in Q3 2025"，却使用了 "earnings" 而不是 "revenue"。嵌入模型认为 "revenue strategy" 比 "Q3 earnings were $47.2M" 更接近查询。

**多跳问题**："Which team had the highest customer satisfaction score improvement?" 这需要找出每个团队的满意度分数，进行比较并找出最大值。没有一个单独的分块包含答案，信息分散在多个团队报告中。

**大型语料库问题**：你有 200 万个分块，正确答案在第 1,847,293 号分块里。你的 top-5 检索却拉回了第 14、89,201、1,200,000、44 和 901,333 号分块。它们在嵌入空间里接近查询，但都不包含答案。在这种规模下，近似最近邻（approximate nearest neighbor）搜索引入的误差足以把相关结果挤出 top-k。

基础 RAG 失败的原因是：向量相似度（vector similarity）不等于相关性（relevance）。一个分块可能与查询语义相似，却对回答问题毫无帮助。高级 RAG 通过四种技术解决这一问题：混合搜索（hybrid search，增加关键词匹配）、重排序（reranking，更仔细地给候选打分）、查询转换（query transformation，搜索前先修正查询）和更好的分块（chunking，以正确粒度检索）。

## 核心概念

### 混合搜索：语义 + 关键词

语义搜索（向量相似度）擅长理解含义。"How do I cancel my subscription?" 能匹配 "Steps to terminate your plan"，尽管它们没有共享任何单词。但它会漏掉精确匹配。"Error code E-4021" 可能不会匹配包含 "E-4021" 的分块，因为嵌入模型把它当作噪声处理。

关键词搜索（BM25）正好相反。它擅长精确匹配。"E-4021" 能完美匹配。但 "cancel my subscription" 如果文档写的是 "terminate your plan"，就会返回零结果。

混合搜索同时运行两者，然后合并结果。

**BM25**（Best Matching 25）是标准的关键词搜索算法，自 1990 年代以来一直是搜索引擎的支柱。公式如下：

```
BM25(q, d) = sum over terms t in q:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中 tf(t,d) 是词项 t 在文档 d 中的词频（term frequency），IDF(t) 是逆文档频率（inverse document frequency），|d| 是文档长度，avgdl 是平均文档长度，k1 控制词频饱和度（默认 1.2），b 控制长度归一化（默认 0.75）。

简单说：BM25 对包含查询词项（尤其是稀有词项）的文档给予更高分数，但重复词项带来的收益递减。一份出现 50 次 "revenue" 的文档，并不会比只出现 1 次的文档相关 50 倍。

### 倒数秩融合（Reciprocal Rank Fusion, RRF）

你现在有两份排序结果：一份来自向量搜索，一份来自 BM25。如何合并？倒数秩融合（RRF）是标准做法。

```
RRF_score(d) = sum over rankings R:
    1 / (k + rank_R(d))
```

其中 k 是一个常数（通常取 60），用于防止排名第一的结果主导最终分数。

一份在向量搜索排第 1、BM25 排第 5 的文档得分：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

一份在向量搜索排第 3、BM25 排第 2 的文档得分：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

RRF 天然平衡两种信号。在两个列表中都排名靠前的文档得分最高；在一个列表中排名第 1 却在另一个列表中缺失的文档只得到中等分数。这种方法稳健，因为它使用排名而非原始分数，因此两个系统之间分数分布的差异无关紧要。

### 重排序（Reranking）

检索（无论是向量、关键词还是混合）速度快但不够精确。它使用双编码器（bi-encoder）：查询和每个文档分别被嵌入，然后比较。嵌入只需计算一次并缓存，因此能扩展到数百万文档。

重排序使用交叉编码器（cross-encoder）：查询和候选文档一起输入模型，模型输出相关性分数。模型同时看到两段文本，能捕捉它们之间的细粒度交互。交叉编码器可以理解 "What were Q3 earnings?" 与包含 "$47.2M in Q3" 的分块高度相关，即使双编码器漏掉了这层联系。

代价是：交叉编码器比双编码器慢 100–1000 倍，因为它要联合处理查询-文档对。你无法为一百万个文档预计算交叉编码器分数。解决方案是：先用混合搜索召回更大的候选集（top-50），再用交叉编码器重排序得到最终的 top-5。

```mermaid
graph LR
    Q["查询"] --> H["混合搜索"]
    H --> C50["前 50 个候选"]
    C50 --> RR["交叉编码器重排序"]
    RR --> C5["最终前 5 个结果"]
    C5 --> P["构建提示词"]
    P --> LLM["生成答案"]
```

常见重排序模型（2026 年阵容）：
- Cohere Rerank 3.5：托管 API，多语言，在混合语料库上召回增益最佳
- Voyage rerank-2.5：托管 API，延迟最低
- Jina-Reranker-v2 Multilingual：开放权重，支持 100+ 语言
- bge-reranker-v2-m3：开放权重，强劲基线
- cross-encoder/ms-marco-MiniLM-L-6-v2：开放权重，可在 CPU 上运行，适合原型
- ColBERTv2 / Jina-ColBERT-v2：迟交互（late-interaction）多向量重排序器——打分复杂度为 O(词元数) 而非 O(文档数)

### 查询转换（Query Transformation）

有时问题不在检索，而在查询本身。"What was that thing about the new policy change?" 是一个糟糕的搜索查询：没有具体词项，嵌入模糊，任何检索系统都难以找到正确文档。

**查询重写（Query rewriting）**：把用户查询改写成更好的搜索查询。LLM 可以完成：

```
User: "What was that thing about the new policy change?"
Rewritten: "Recent policy changes and updates"
```

**HyDE（假设文档嵌入，Hypothetical Document Embeddings）**：不是直接用查询去搜索，而是生成一个假设答案，嵌入它，再用它搜索相似的真实文档。

```
Query: "What is the refund policy for enterprise?"
Hypothetical answer: "Enterprise customers are eligible for a full refund
within 60 days of purchase. Refunds are pro-rated based on the remaining
subscription period and processed within 5-7 business days."
```

嵌入这个假设答案，并搜索与其相似的真实文档。直觉是：假设答案在嵌入空间中比原始问题更接近真实答案。问题与答案的语言结构不同，通过生成假设答案，可以弥合嵌入中"问题空间"与"答案空间"之间的鸿沟。

HyDE 在检索前增加一次 LLM 调用，延迟会增加 500–2000 毫秒。当原始查询检索质量较差时，这是值得的。

### 父子分块（Parent-Child Chunking）

标准分块迫使你在"小分块精确检索"和"大分块上下文充分"之间取舍。父子分块消除了这种取舍。

索引小分块（128 词元）用于检索。当一个小分块被检索到时，返回其父分块（512 词元）用于提示词。小分块精确匹配查询，父分块为 LLM 生成答案提供足够上下文。

```mermaid
graph TD
    P["父分块（512 词元）<br/>退款政策完整章节"]
    C1["子分块（128 词元）<br/>标准版：30 天退款"]
    C2["子分块（128 词元）<br/>企业版：60 天按比例"]
    C3["子分块（128 词元）<br/>处理时间：5-7 天"]
    C4["子分块（128 词元）<br/>如何提交申请"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["查询：enterprise refund?"] -.->|"匹配子分块"| C2
    C2 -.->|"返回父分块"| P
```

查询 "enterprise refund?" 精确匹配子分块 C2，但提示词收到的是完整父分块 P，其中包含处理时间和申请流程等 surrounding context。

### 元数据过滤（Metadata Filtering）

在执行向量搜索前，先按元数据过滤语料库：日期、来源、类别、作者、语言。这能缩小搜索空间，防止返回不相关结果。

"What changed in the security policy last month?" 应只搜索最近 30 天内 security 类别的文档。如果没有元数据过滤，你会搜索整个语料库，可能召回一份两年前的安全文档，仅仅因为它语义上相似。

生产级 RAG 系统会为每个分块保存元数据：来源文档、创建日期、类别、作者、版本。向量数据库支持在相似度搜索前按元数据预过滤（pre-filtering），这对大规模性能至关重要。

### 评估

你搭建了一个 RAG 系统，怎么知道它是否有效？三个指标：

**检索相关性（Recall@k）**：对于一组已知相关文档的测试问题，相关文档出现在 top-k 结果中的百分比是多少？如果某问题的答案在第 47 号分块，那么第 47 号分块是否出现在 top-5 中？

**忠实度（Faithfulness）**：生成答案是否基于检索到的文档？如果检索到的分块说 "60-day refund window"，而模型说 "90-day refund window"，这就是忠实度失败——模型在已有正确上下文的情况下仍然产生了幻觉（hallucination）。

**答案正确性（Answer correctness）**：生成答案是否与预期答案一致？这是端到端指标，结合了检索质量与生成质量。

一个简单的忠实度检查：取生成答案中的每个断言，验证其（在实质上）是否出现在检索到的分块中。如果答案包含某个检索分块中不存在的事实，那它很可能是幻觉。

```mermaid
graph TD
    subgraph "评估框架"
        Q["测试问题<br/>+ 预期答案<br/>+ 相关文档 ID"]
        Q --> Ret["检索评估<br/>Recall@k：正确<br/>文档是否被召回？"]
        Q --> Faith["忠实度评估<br/>答案是否基于<br/>检索到的文档？"]
        Q --> Correct["正确性评估<br/>答案是否与<br/>预期答案一致？"]
    end
```

## 动手构建

### 步骤 1：BM25 实现

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 步骤 2：倒数秩融合

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### 步骤 3：混合搜索流水线

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### 步骤 4：简单重排序器

在生产环境中，你会使用交叉编码器模型。这里我们构建一个重排序器，用词重叠、词项重要性和短语匹配来为查询-文档相关性打分。

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### 步骤 5：HyDE（假设文档嵌入）

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### 步骤 6：父子分块

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### 步骤 7：忠实度评估

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## 实际使用

使用真正的交叉编码器进行重排序：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

使用 Cohere 托管重排序器：

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

使用真实 LLM 实现 HyDE：

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

使用 Weaviate 的生产级混合搜索：

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

alpha 参数控制平衡：0.0 = 纯关键词（BM25），1.0 = 纯向量，0.5 = 等权重。大多数生产系统使用 0.3 到 0.7 之间的 alpha。

## 交付物

本课产出：
- `outputs/prompt-advanced-rag-debugger.md` —— 用于诊断和修复 RAG 质量问题的提示词
- `outputs/skill-advanced-rag.md` —— 用于构建生产级混合搜索与重排序 RAG 的技能

## 练习

1. 在样本文档上比较 BM25、向量搜索与混合搜索。对 5 个测试查询，记录哪种方法在第一位返回了最相关的分块。混合搜索至少应在 5 个中的 3 个上获胜。

2. 实现元数据过滤。为每份文档添加 "category" 字段（security、billing、api、product）。在执行向量搜索前，过滤到相关类别。用 "What encryption is used?" 测试，验证它只搜索 security 类别的分块。

3. 使用第 06 课的简单生成函数构建完整 HyDE 流水线。比较直接查询搜索与 HyDE 搜索在所有 5 个测试查询上的检索质量（top-3 相关性）。对于模糊查询，HyDE 应有所改善。

4. 在样本文档上实现父子分块策略。使用 child_size=30 和 parent_size=100。用子分块搜索，但向提示词返回父分块。将生成答案与 chunk_size=50 的标准分块进行比较。

5. 创建评估数据集：10 个已知答案分块的问题。测量 (a) 仅向量搜索、(b) 仅 BM25、(c) 混合搜索、(d) 混合搜索 + 重排序 的 Recall@3、Recall@5 和 Recall@10。绘制结果并找出重排序帮助最大的场景。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| BM25 | "关键词搜索" | 一种概率排序算法，根据词频、逆文档频率和文档长度归一化为文档打分 |
| 混合搜索（Hybrid search） | "两全其美" | 并行运行语义（向量）搜索与关键词（BM25）搜索，再用秩融合合并结果 |
| 倒数秩融合（Reciprocal Rank Fusion） | "合并排序列表" | 通过对每个文档在所有列表中的 1/(k + rank) 求和来合并多份排序结果 |
| 重排序（Reranking） | "第二轮打分" | 使用更昂贵的交叉编码器模型对初步检索的候选集重新打分 |
| 交叉编码器（Cross-encoder） | "联合查询-文档模型" | 将查询和文档作为单一输入，输出相关性分数的模型；比双编码器更准确，但对全语料库搜索太慢 |
| 双编码器（Bi-encoder） | "独立嵌入模型" | 独立嵌入查询和文档的模型；嵌入可预计算，因此速度快，但准确性不如交叉编码器 |
| HyDE | "用假答案搜索" | 生成查询的假设答案，嵌入它，然后搜索与其相似的真实文档 |
| 父子分块（Parent-child chunking） | "小分块搜索，大分块上下文" | 索引小分块以精确检索，但返回更大的父分块以提供充足上下文 |
| 元数据过滤（Metadata filtering） | "搜索前先收窄" | 在执行向量搜索前按属性（日期、来源、类别）过滤文档，以缩小搜索空间 |
| 忠实度（Faithfulness） | "是否保持有依据" | 生成答案是否由检索到的文档支持，而非从模型训练数据中幻觉出来 |

## 延伸阅读

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) —— BM25 的权威参考，解释公式背后的概率基础
- Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009) —— 原始 RRF 论文，证明它比更复杂的融合方法更有效
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) —— HyDE 论文，证明假设文档嵌入无需任何训练数据即可提升检索
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) —— 证明在 BM25 之上使用交叉编码器重排序能显著提升检索质量
- [Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023)](https://arxiv.org/abs/2310.03714) —— 将提示词构造与权重选择视为检索流水线的优化问题；想从"提示 LLM"转向"编程 LLM"请读这篇
- [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research 2024)](https://arxiv.org/abs/2404.16130) —— GraphRAG 论文：实体关系抽取 + Leiden 社区检测，用于查询聚焦摘要；全局检索与局部检索的区别
- [Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)](https://arxiv.org/abs/2310.11511) —— 通过反思词元进行自我评估的 RAG；超越静态"检索-生成"的智能体前沿
- [LangChain Query Construction blog](https://blog.langchain.dev/query-construction/) —— 如何将自然语言查询转换为结构化数据库查询（Text-to-SQL、Cypher），作为检索前的预处理步骤
