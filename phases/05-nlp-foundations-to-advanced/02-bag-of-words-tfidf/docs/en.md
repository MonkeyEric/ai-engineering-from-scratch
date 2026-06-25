# 词袋模型、TF-IDF 与文本表示

> 先计数，再思考。在定义明确的任务上，TF-IDF 到 2026 年仍然能打败嵌入模型。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 01（文本处理），Phase 2 · 02（从零实现线性回归）
**时间：** ~75 分钟

## 问题

模型需要数字。你手头只有字符串。

每个 NLP 流水线都必须回答同一个问题：如何把长度可变的 token 流转换成分类器可以消费的固定长度向量？这个领域最早找到的答案是“能用就行里最笨的那个”——数词，然后做成向量。

这个向量支撑过的生产级 NLP 应用比任何嵌入模型都多：垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、前十年学术界的 NLP 基准测试。到 2026 年，从业者在做狭窄的文本分类任务时仍然会首先想到它。它快、可解释，而且在“词是否存在”起决定作用的任务上，往往与 4 亿参数的嵌入模型没有差别。

本课将从零实现词袋模型，然后实现 TF-IDF；接着用三行 scikit-learn 代码完成同样的事；最后指出那个会让你转向嵌入模型的失败场景。

## 概念

**词袋模型（Bag of Words, BoW）** 丢弃顺序。对每个文档，统计每个词在词汇表中出现多少次。向量长度等于词汇表大小，位置 `i` 表示词 `i` 的计数。

**TF-IDF** 重新加权 BoW。出现在所有文档里的词没有信息量，因此要缩小权重；在整个语料中罕见但在某个文档中频繁出现的词是信号，因此要放大权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是词在文档中的词频，`df` 是文档频率（包含该词的文档数），`N` 是总文档数。`log` 的作用是让随处可见的词权重保持有界。

关键特性：二者都产生稀疏向量，且每一维都可解释。你可以查看训练好的分类器权重，直接读出哪些词把文档推向哪个类别。这一点是 768 维 BERT 嵌入做不到的。

## 动手实现

### 步骤 1：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：已分词的文档列表（任何词级分词器都可以；本课 `code/main.py` 使用简化的小写版本）。输出：`{word: index}` 字典。稳定的插入顺序意味着索引 0 对应第一个文档中首次出现的词。约定不唯一；scikit-learn 会按字母顺序排序。

### 步骤 2：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行表示文档，列表示词汇表索引。条目 `[i][j]` 表示“词 `j` 在文档 `i` 中出现多少次”。文档 1 中 `cat` 出现了两次，因为它确实出现了两次；文档 0 中 `ran` 为 0，因为它没有出现。

### 步骤 3：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

有两个平滑技巧值得说明。`(n+1)/(d+1)` 避免 `log(x/0)`；末尾的 `+1` 保证出现在所有文档中的词 IDF 为 1 而不是 0，与 scikit-learn 的默认值一致。其他实现可能使用原始 `log(N/df)`，两种都可以；带平滑的版本更友好。

### 步骤 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有文档中，所以 IDF 很低；`dog` 只出现在一篇文档中，所以 IDF 很高。向量是稀疏的（大多数项很小），有区分力的词会被放大。

### 步骤 5：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不做归一化，长文档的向量会更大，并在相似度计算中占主导地位。L2 归一化把所有文档都放到单位超球面上，此时行之间的余弦相似度就是点积。

## 使用现成的工具

scikit-learn 提供了生产级实现。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一次调用完成分词、构建词汇表和 BoW。`TfidfVectorizer` 再加上 IDF 加权和 L2 归一化。二者都返回稀疏矩阵。对于 10 万篇文档，稠密矩阵放不下；尽量保持稀疏，直到分类器强制要求稠密输入。

几个改变结果的参数：

| 参数 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含二元词组。通常能提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 个文档中的词。对噪声数据可以压缩词汇表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。无需硬编码停用词表即可近似停用词过滤。 |
| `stop_words="english"` | scikit-learn 内置的英文停用词表。是否使用取决于任务——情感分析*不应*丢弃否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 代替原始 `tf`。当某个词在单篇文档中反复出现时很有帮助。 |

### TF-IDF 仍然占优的场景（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。这些任务看的是“词是否出现”，语义细枝末节无关紧要。
- 低数据场景（只有几百条标注样本）。TF-IDF 加逻辑回归没有预训练成本。
- 任何对延迟敏感的地方。TF-IDF 加线性模型以微秒级响应；把文档嵌入 transformer 需要 10–100 毫秒。
- 需要解释预测的系统。查看分类器系数，最正向的词就是理由。

### TF-IDF 失败的地方

语义失明。看这两个文档：

- “The movie was not good at all.”
- “The movie was excellent.”

一个是负面评价，一个是正面评价。它们的 TF-IDF 重叠正好是 `{the, movie, was}`。基于词袋的分类器必须死记硬背：`not` 出现在 `good` 附近会翻转标签。在足够数据下它能学会，但永远不会像真正理解句法的模型那样优雅。

另一个失败点：推理时遇到未登录词。一个在 IMDb 影评上训练的 BoW 模型，如果 `Zoomer-approved` 这个 token 在训练时从未出现，它就完全不知道如何处理。子词嵌入（第 04 课）能处理这种情况，TF-IDF 不能。

### 混合方案：TF-IDF 加权嵌入

2026 年中等数据量分类的务实默认做法：用 TF-IDF 权重作为对词嵌入的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你得到了嵌入的语义能力，也得到了 TF-IDF 对罕见词的强调。分类器在这个池化后的向量上训练。在情感、主题和意图分类等任务中，当标注样本低于约 5 万时，这种混合方案通常比单独使用任一种方法效果更好。

## 交付产物

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: 给定一个文本分类任务，推荐 BoW、TF-IDF、嵌入或混合方案。
phase: 5
lesson: 02
---

你推荐文本向量化策略。给定任务描述后，输出：

1. 表示方法（BoW、TF-IDF、transformer 嵌入或混合方案），并用一句话解释原因。
2. 具体的向量化器配置。指明库名，并引用参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 上线前需要测试的一个失败点。

当用户标注样本不足 500 时，除非他们能证明 TF-IDF 基线存在语义失败，否则拒绝推荐嵌入。情感分析任务拒绝移除停用词（否定词携带信号）。类别不平衡不能只靠换向量化器解决，需额外指出。

示例输入："Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

示例输出：

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化后的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证完全相同文档得分为 1.0，词汇完全不重叠的文档得分为 0.0。
2. **中等。** 给 `bag_of_words` 增加 `n-gram` 支持。参数 `n` 表示对 `n`-gram 计数。测试 `n=2` 时 `["the", "cat", "sat"]` 能正确输出 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 用 GloVe 100d 向量实现上面的 TF-IDF 加权嵌入混合方案（下载一次并缓存）。在 20 Newsgroups 数据集上比较它与纯 TF-IDF 以及纯均值池化嵌入的分类准确率，报告各自在什么情况下获胜。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 单个文档中词汇表词语的计数。丢弃顺序。 |
| TF | 词频 | 一个词在文档中出现的次数，可选择用文档长度归一化。 |
| DF | 文档频率 | 至少包含该词的文档数量。 |
| IDF | 逆文档频率 | 平滑后的 `log(N / df)`。降低随处可见词的权重。 |
| 稀疏向量 | 大部分为零 | 词汇表通常 1 万–10 万词；任一文档只包含其中很小一部分。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示完全相同，0 表示正交。 |

## 扩展阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 权威 API 参考，包含对每个参数的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 让 TF-IDF 在十年间成为默认方法的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年对“旧方法何时仍占优及原因”的解读。
