# 词嵌入 —— 从零实现 Word2Vec

> 一个词的意义由它周围的词决定。基于这个想法训练一个浅层网络，几何结构便会自然涌现。

**类型：** 构建
**语言：** Python
**前置知识：** 第 5 阶段 · 02（词袋模型 + TF-IDF），第 3 阶段 · 03（从零实现反向传播）
**时间：** 约 75 分钟

## 问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词，却不知道它们的意思几乎相同。一个在 `dog` 上训练出来的分类器，无法泛化到关于 `puppy` 的评论。你可以通过罗列同义词来掩盖这个问题，但遇到罕见词、领域术语以及所有你未预料到的语言时都会失效。

你想要一种表示，让 `dog` 和 `puppy` 在空间中靠得很近，让 `king - man + woman` 落在 `queen` 附近，让在 `dog` 上训练的模型能自动将部分信号迁移给 `puppy`。

Word2Vec 给了我们这样的空间。两层神经网络，万亿 token 的训练规模，发表于 2013 年。它的架构简单到几乎令人尴尬，却深刻改变了此后十年自然语言处理的面貌。

## 概念

**分布假说**（Firth，1957）：“要认识一个词，就看它常与哪些词为伴。” 如果两个词出现在相似的上下文中，它们很可能意思相近。

Word2Vec 有两种形式，都利用了上述思想。

- **Skip-gram。** 给定中心词，预测周围的词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（连续词袋）。** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对罕见词效果更好，因此成为默认选择。

网络只有一个隐藏层，没有非线性激活。输入是词汇表上的 one-hot 向量，输出是词汇表上的 softmax。训练完成后，丢弃输出层，隐藏层的权重就是词嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

技巧在于：对 10 万个词做 softmax 开销巨大。Word2Vec 使用 **负采样** 将其转化为二分类任务：预测“这个上下文词是否出现在该中心词附近”。对每个训练样本，采样少量负例（未共同出现的词），而不必在整个词汇表上计算 softmax。

## 从零构建

### 步骤 1：从语料库中提取训练样本

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内的每个（中心词，上下文词）配对都是一个正例训练样本。

### 步骤 2：嵌入表

两个矩阵。`W` 是中心词嵌入表（最终保留的那个）。`W'` 是上下文词表（常被丢弃，有时会与 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

使用较小的随机初始化。词汇量 1 万、维度 100 是现实规模；教学时，50 个词 × 16 维就足以观察到几何结构。

### 步骤 3：负采样目标函数

对每个正例样本 `(center, context)`，从词汇表中随机采样 `k` 个词作为负例。训练模型，使得正例的点积 `W[center] · W'[context]` 较大，负例的点积较小。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

核心公式：正例对上的 logistic 损失（希望 sigmoid 接近 1）加上负例对上的 logistic 损失（希望 sigmoid 接近 0）。梯度同时流回两个矩阵。完整推导见原论文；若想真正掌握，建议用纸笔亲自推一遍。

### 步骤 4：在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料库上训练足够多的轮次后，共享上下文的词会拥有相似的中心词嵌入。在玩具语料库上，这种效果很微弱；而在数十亿 token 上，效果会非常明显。

### 步骤 5：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。这不是因为模型懂得“王室”是什么，而是因为向量 `(king - man)` 捕捉到了类似“王室”的方向，把它加到 `woman` 上就会落在“王室女性”区域附近。

## 使用它

从零实现 Word2Vec 是为了教学。生产环境中的 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

实际工作中，你几乎不会自己训练 Word2Vec，而是下载预训练向量。

- **GloVe** —— 斯坦福基于共现矩阵分解的方法。提供 50 维、100 维、200 维、300 维的检查点，通用覆盖较好。第 04 课专门讲解 GloVe。
- **fastText** —— Facebook 对 Word2Vec 的扩展，嵌入字符 n-gram。通过组合子词来处理未登录词。第 04 课。
- **Google News 上的预训练 Word2Vec** —— 300 维，300 万词词汇表，2013 年发布，至今每天仍被大量下载。

### Word2Vec 在 2026 年仍然占优的场景

- 轻量领域特定检索。在一台笔记本电脑上用一小时基于医学摘要训练，得到通用模型无法捕捉的领域专用向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。将其从其他词中减去，可获得性别中立轴。公平性研究中仍在使用。
- 可解释性。100 维足够小，可以通过 PCA 或 t-SNE 绘制出来，真正观察到聚类形成。
- 任何需要在无 GPU 设备上本地推理的场景。Word2Vec 查表只是一次行读取。

### Word2Vec 的不足

多义词的壁垒。`bank` 只有一个向量，`river bank`（河岸）和 `financial bank`（银行）共享它。`table`（电子表格 vs. 家具）也共享一个向量。下游分类器无法从向量中区分这些不同义项。

上下文嵌入（ELMo、BERT 以及之后的所有 Transformer）通过基于周围上下文为每个词出现位置生成不同向量解决了这个问题。这就是从 Word2Vec 到 BERT 的跨越：从静态到上下文相关。Transformer 部分在第 7 阶段讲解。

另一个缺陷是未登录词问题。如果训练数据中不存在 `Zoomer-approved`，Word2Vec 就从未见过它，也没有回退机制。fastText 通过子词组合解决了这一问题（第 04 课）。

## 交付

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: 检查 word2vec 模型。运行类比、查找近邻、诊断质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你通过探查训练好的词嵌入来验证它们是否正常工作。给定一个 `gensim.models.KeyedVectors` 对象和词汇表，你运行：

1. 三项经典类比测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其余弦相似度。
2. 五个用户提供的领域特定词的最近邻测试。打印 top-5 近邻及其余弦相似度。
3. 一项对称性检查。在浮点精度范围内验证 `similarity(a, b) == similarity(b, a)`。
4. 一项退化检查。如果任意嵌入的范数低于 0.01 或高于 100，说明模型存在训练缺陷，需标记出来。

不要仅凭类比准确率就判定模型良好。类比基准可被投机取巧，且无法迁移到下游任务。建议同时进行内在评估与下游评估。
```

## 练习

1. **简单。** 在一个小型语料库（20 句关于猫和狗的句子）上运行训练循环。200 轮后验证 `nearest(vocab, W, W[vocab["cat"]])` 的 top 3 中是否包含 `dog`。如果没有，增加轮数或词汇量。
2. **中等。** 添加高频词子采样。频率高于 `10^-5` 的词按与频率成正比的概率从训练样本中丢弃。测量这对罕见词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练一个模型。计算两个偏见轴：`he - she` 和 `doctor - nurse`。将职业词投影到这两个轴上，报告偏见差距最大的职业。这是公平性研究者常用的探针方法。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Word embedding | 将词表示为向量 | 从上下文中学到的密集低维（通常为 100–300 维）表示。 |
| Skip-gram | Word2Vec 技巧 | 根据中心词预测上下文词。比 CBOW 慢，但对罕见词效果更好。 |
| Negative sampling | 训练捷径 | 用与 `k` 个随机词的二分类替代对整个词汇表的 softmax。 |
| Static embedding | 每个词一个向量 | 无论上下文如何都使用相同向量。无法处理多义词。 |
| Contextual embedding | 上下文敏感向量 | 根据周围词为每个出现位置生成不同向量。Transformer 所产生的结果。 |
| OOV | 未登录词 | 训练时未见过的词。Word2Vec 无法为其生成向量。 |

## 延伸阅读

- [Mikolov 等（2013）。Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) —— 负采样论文。简短易读。
- [Rong, X.（2014）。word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) —— 最清晰的梯度推导，如果你觉得原论文的数学过于晦涩，可以读这篇。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html) —— 真正可用的生产级训练设置。
