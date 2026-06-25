# Transformer 之前的文本生成——N-gram 语言模型

> 如果一个词出人意料，说明模型很差。困惑度把“意外”变成了一个数字。平滑让它保持有限。

**类型：** Build
**语言：** Python
**前置要求：** Phase 5 · 01（Text Processing），Phase 2 · 14（Naive Bayes）
**时长：** 约 45 分钟

## 问题背景

在 Transformer、RNN、词嵌入出现之前，语言模型通过统计一个词紧跟在前 `n-1` 个词之后出现的次数来预测下一个词。例如，"the cat" 后面出现 "sat" 47 次，出现 "jumped" 12 次，出现 "refrigerator" 0 次。归一化后得到概率分布。

这就是 n-gram 语言模型。从 1980 年到 2015 年，它驱动了所有的语音识别器、拼写检查器和基于短语的机器翻译系统。如今在需要低成本设备端语言建模时，它仍然在被使用。

有趣的问题是如何处理未出现的 n-gram。原始基于计数的模型会给任何未见过的序列分配零概率，这会带来灾难性后果，因为句子很长，几乎每个长句都至少包含一个未在训练集中出现过的序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑是这些研究的结晶，而现代深度学习也继承了它的经验主义传统。

## 核心概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常 trigram 取 3，4-gram 取 4）。从计数中计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何在训练集中未出现过的 n-gram 都会得到零概率。2007 年一项针对 Brown 语料库的研究发现，即使是 4-gram 模型，也有 30% 的 held-out 4-gram 在训练集中未出现。没有平滑，就无法在任何真实文本上进行评估。

**按复杂程度排序的平滑方法：**

1. **Laplace（加一平滑）。** 给每个计数加 1。简单，但在罕见事件上表现很差。
2. **Good-Turing。** 根据频率的频率，把概率质量从高频事件重新分配给未见事件。
3. **插值（Interpolation）。** 用可学习的权重把 n-gram、(n-1)-gram 等估计值结合起来。
4. **回退（Backoff）。** 如果 n-gram 计数为零，则回退到 (n-1)-gram。Katz 回退对其做了归一化处理。
5. **绝对折扣（Absolute discounting）。** 从所有计数中减去一个固定折扣 `D`，再把这部分质量重新分配给未见事件。
6. **Kneser-Ney。** 在绝对折扣的基础上，对低阶模型做了一个精巧的选择：用*延续概率*（continuation probability，即一个词出现在多少个不同上下文里）代替原始频次。

Kneser-Ney 的洞察非常深刻。"San Francisco" 是一个常见 bigram。Unigram "Francisco" 几乎总是出现在 "San" 之后。朴素的绝对折扣会给 "Francisco" 很高的 unigram 概率（因为计数很高）。Kneser-Ney 注意到 "Francisco" 只出现在一种上下文里，因此会相应降低它的延续概率。结果是：一个以 "Francisco" 结尾的新 bigram 会得到合理的低概率。

**评估指标：困惑度（perplexity）。** 在 held-out 测试集上，每个词的平均负对数似然的指数。越低越好。困惑度为 100 意味着模型像在 100 个词中均匀选择一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 动手实现

### 第一步：trigram 计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表。输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 表示句子边界。

### 第二步：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给每个计数加 1。虽然能平滑，但会把过多概率质量分配给未见事件，同时也会损害已知罕见事件。

### 第三步：Kneser-Ney（bigram，插值形式）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个关键部分。`continuation_prob` 捕获“这个词出现在多少种不同上下文中？”（Kneser-Ney 的创新点）。`lambda_prev` 是折扣释放出来的概率质量，用来给回退项加权。最终概率是打过折扣的主项加上加权后的延续项。

### 第四步：基于采样的文本生成

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。不同种子会产生不同输出。如果需要类似 beam search 的输出，可以在每一步取 argmax（贪心），并加入一个小随机性旋钮（温度）。

### 第五步：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。在 Brown 语料库上，调优后的 4-gram KN 模型困惑度约为 140。而 Transformer 语言模型在相同测试集上可以达到 15-30。差距约 10 倍。这就是领域转向神经网络的原因。

## 应用场景

- **经典 NLP 教学。** 这是接触平滑、最大似然估计和困惑度最清晰的方式。
- **KenLM。** 工业级 n-gram 库。在语音和机器翻译等对延迟敏感的系统中用作重排序器。
- **设备端自动补全。** 键盘里的 trigram 模型。至今仍在使用。
- **基线模型。** 在宣称神经网络语言模型很好之前，一定要先算一个 n-gram LM 的困惑度。如果你的 Transformer 没有大幅击败 KN，那一定哪里出了问题。

## 交付产物

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: 在训练神经网络语言模型之前，构建一个可复现的 n-gram 语言模型基线。
phase: 5
lesson: 16
---

给定语料库和目标用途（下一个词预测、重排序、困惑度基线），输出：

1. N-gram 阶数。通用英语用 trigram，语料大用 4-gram，语音重排序用 5-gram。
2. 平滑方法。默认用 Modified Kneser-Ney，教学场景可用 Laplace。
3. 工具库。生产用 `kenlm`，教学用 `nltk.lm`，仅为了学习才手写实现。
4. 评估方式。使用 held-out 困惑度，训练集和测试集必须采用一致的 tokenization。

拒绝报告在 tokenization 不一致的系统之间计算出的困惑度——困惑度数字只有在完全相同的 tokenization 下才具有可比性。同时标注测试集上的 OOV 比例；KN 对 OOV 处理较差，除非在训练时预留特殊的 <UNK> token。
```

## 练习题

1. **简单。** 在一个 1000 句的莎士比亚语料库上训练 trigram LM。生成 20 个句子。它们会在局部合理，但全局不连贯。这是经典演示。
2. **中等。** 在莎士比亚的 held-out 划分上实现 KN 模型的困惑度，并与 Laplace 比较。你应该看到 KN 降低 30-50% 的困惑度。
3. **困难。** 构建一个 trigram 拼写纠错器：给定一个拼写错误的词及其上下文，生成候选修正，并按 LM 下的上下文概率排序。在公开的 Birkbeck 拼写语料库上评估。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| N-gram | 词序列 | `n` 个连续 token 组成的序列。 |
| Smoothing | 避免零概率 | 重新分配概率质量，使未见事件获得非零概率。 |
| Perplexity | 语言模型质量指标 | 在 held-out 数据上的 `exp(-average log-prob)`。越低越好。 |
| Backoff | 回退到更短上下文 | 如果 trigram 计数为零，就使用 bigram。Katz 回退对其做了形式化。 |
| Kneser-Ney | 最佳 n-gram 平滑 | 绝对折扣 + 对低阶模型使用延续概率。 |
| Continuation probability | KN 特有 | `P(w)` 按 `w` 出现的上下文数量加权，而不是按原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) —— n-gram 语言模型与平滑的经典教材。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) —— 奠定 Kneser-Ney 为最佳 n-gram 平滑器的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) —— 原始 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) —— 快速工业级 n-gram LM，2026 年仍用于延迟敏感场景。
