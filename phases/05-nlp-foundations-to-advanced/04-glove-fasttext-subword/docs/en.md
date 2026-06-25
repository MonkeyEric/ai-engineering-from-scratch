# GloVe、FastText 与子词嵌入

> Word2Vec 为每个词训练一个嵌入。GloVe 对共现矩阵进行分解。FastText 把词拆成片段嵌入。BPE 为 Transformer 架起了桥梁。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 03（从零实现 Word2Vec）
**时间：** ~45 分钟

## 问题背景

Word2Vec 留下了两个未解决的问题。

第一，当时还有另一条研究路线，直接对共现矩阵进行分解（LSA、HAL），而不是像 skip-gram 那样做在线更新。Word2Vec 的迭代方法本质上更好，还是差异只是两种方法处理计数方式不同造成的产物？**GloVe** 回答了这个问题：只要损失函数选取得当，矩阵分解的效果可以匹敌甚至超越 Word2Vec，而且训练成本更低。

第二，这两种方法对从未见过的词都没有办法。`Zoomer-approved`、`dogecoin`、上周新造的专有名词、某个稀有词根的每一种屈折变化。**FastText** 通过嵌入字符 n-gram 解决了这个问题：一个词是其组成部分的总和，包括词素，因此即使未登录词也能得到一个合理的向量。

第三，Transformer 出现之后，问题又发生了变化。词级词表上限大约在一百万条；而真实语言比这更开放。**字节对编码（BPE）** 及其变体通过学习一组高频子词单元覆盖了所有内容。每一个现代大语言模型的分词器都是子词分词器。

本课将依次讲解这三种方法，然后说明在不同场景下该选哪一种。

## 核心概念

**GloVe（Global Vectors，全局向量）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 出现在词 `i` 上下文窗口中的次数。训练向量，使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失加权，避免高频词对主导训练。完成。

**FastText。** 一个词是其字符 n-gram 加上词本身向量的总和。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量就是这些组成部分向量的和。训练方式与 Word2Vec 相同。好处是：未登录词（如 `whereupon`）可以由已知的 n-gram 组合而成。

**BPE（Byte-Pair Encoding，字节对编码）。** 从单个字节（或字符）的词表开始。统计语料中每一对相邻符号的出现次数。将最频繁的相邻对合并为一个新 token。重复 `k` 次。结果：词表大小为 `k + 256`，其中高频序列（`ing`、`tion`、`the`）是单个 token，而罕见词被拆成熟悉的片段。任意句子都能被分词成已知 token。

## 动手实现

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个细节值得说明。加权函数 `f(x) = (x/x_max)^alpha` 会降低高频词对（例如 `(the, and)`）的权重，避免它们主导损失。最终的嵌入是中心词表 `W` 与上下文词表 `W_tilde` 之和。将两者相加是一个已发表的技巧，通常比单独使用其中任意一个效果更好。

### FastText：子词感知嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词都由其 n-gram 集合表示（通常字符长度为 3 到 6）。词嵌入就是其 n-gram 嵌入的总和。对于 skip-gram 训练，只需把 Word2Vec 中使用单个向量的地方替换为这种表示。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未登录词，只要它的某些 n-gram 已知，就仍能得到一个向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，因此它们在向量空间中彼此靠近。

### BPE：学习子词词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一轮迭代会合并最频繁的相邻字符对。经过足够多的迭代后，高频子串（`low`、`est`、`tion`）会变成单个 token，而罕见词会被干净地拆成已知片段。

真实的 GPT / BERT / T5 分词器会学习 3 万到 10 万次合并。结果是：任何文本都能被分词成固定长度的已知 ID 序列，永远不会出现 OOV。

## 实际使用

在实践中，你很少会自己训练这些方法。你通常会加载预训练权重。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 Transformer 时代，使用 BPE 风格的子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（这是 GPT-2 的约定）。每个现代分词器都是 BPE 的变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 如何选择

| 场景 | 选择 |
|-----------|------|
| 预训练通用词向量，不需要处理 OOV | GloVe 300d |
| 预训练通用词向量，必须处理拼写错误 / 新词 / 形态丰富语言 | FastText |
| 输入 Transformer（训练或推理） | 使用该模型自带的分词器，永远不要替换 |
| 从头训练自己的语言模型 | 先在语料上训练 BPE 或 SentencePiece 分词器 |
| 生产环境的线性文本分类模型 | 仍然使用 TF-IDF。参见第 02 课。 |

## 交付产物

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: 为一种新的语言模型或文本流水线选择分词方案。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

给定任务和数据集描述，你输出：

1. 分词策略（词级、BPE、WordPiece、SentencePiece、字节级）。一句话说明理由。
2. 词表大小目标（例如，仅英文语言模型用 32k，多语言用 64k-100k）。
3. 具体的训练命令和调用的库。给出库名，并引用参数。
4. 一个可复现性陷阱。分词器与模型不匹配是最常见且最隐蔽的生产环境 bug；必须指出哪两者必须配套使用。

当用户要对预训练大语言模型进行微调时，拒绝推荐训练自定义分词器。对于任何面向生产推理的模型，拒绝推荐词级分词。对于非英文 / 多文字语料，必须标注为需要使用支持字节回退的 SentencePiece。
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠。你会看到大量共享片段（`pla`、`lay`、`play`），这也是 FastText 能在形态变化之间迁移得很好的原因。
2. **中等。** 扩展 `learn_bpe`，追踪词表增长过程。绘制“每个语料字符对应的 token 数”随合并次数变化的曲线。你会看到最初压缩速度很快，最终渐近在约 2-3 个字符每个 token。
3. **困难。** 在莎士比亚全集上训练一个 1k 次合并的 BPE。比较常见词与罕见专有名词的分词结果。测量合并前后的平均每个词的 token 数。写下让你意外的发现。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 共现矩阵 | 词-词频表 | `X[i][j]` = 词 `j` 出现在词 `i` 上下文窗口中的次数。 |
| 子词 | 词的一部分 | 字符 n-gram（FastText）或学习得到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 迭代合并最频繁的相邻符号对，直到词表达到目标大小。 |
| OOV | 未登录词 | 模型从未见过的词。Word2Vec/GloVe 无法处理。FastText 和 BPE 可以处理。 |
| 字节级 BPE | 在原始字节上的 BPE | GPT-2 的方案。词表从 256 个字节开始，因此永远不会有 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) —— GloVe 论文，七页，对损失函数的推导至今仍是最清晰的。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) —— FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— 将 BPE 引入现代 NLP 的开创性论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) —— BPE、WordPiece 和 SentencePiece 在实际使用中的真正区别。
