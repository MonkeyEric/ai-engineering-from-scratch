# 从零实现自注意力（Self-Attention）

> 注意力（attention）就像一张查询表，每个词都在问“谁对我重要？”——并学会回答。

**类型：** Build
**语言：** Python
**前置知识：** 第 3 阶段（深度学习核心）、第 5 阶段第 10 课（序列到序列）
**时长：** ~90 分钟

## 学习目标

- 仅使用 NumPy 从零实现缩放点积自注意力（scaled dot-product self-attention），包括查询（query）/键（key）/值（value）投影和 softmax 加权求和
- 构建多头注意力（multi-head attention）层，完成分头、并行计算注意力并拼接结果
- 追踪注意力矩阵如何捕捉词元（token）之间的关系，并解释为什么用 sqrt(d_k) 缩放可以防止 softmax 饱和
- 应用因果掩码（causal masking），将双向注意力转换为自回归（解码器风格）注意力

## 问题背景

循环神经网络（RNN）一次处理一个词元。当到达第 50 个词元时，第 1 个词元的信息已经被压缩了 50 次。长程依赖被挤压进固定大小的隐藏状态——这是即使大量 LSTM 门控也无法彻底解决的瓶颈。

2014 年 Bahdanau 的注意力论文给出了解法：让解码器回顾每一个编码器位置，并决定当前步哪些位置重要。但它仍然依附于 RNN。2017 年的《Attention Is All You Need》提出了一个更尖锐的问题：如果注意力是*唯一*的机制呢？没有循环，没有卷积，只有注意力。

自注意力让序列中的每个位置都能在一步并行地关注其他所有位置。这正是 Transformer 快速、可扩展且占据主导地位的原因。

## 核心概念

### 数据库查询类比

把注意力想象成一次“软”数据库查询：

```
传统数据库：
  查询（Query）: "capital of France"  -->  精确匹配  -->  "Paris"

注意力：
  查询（Query）: "capital of France"  -->  与所有键（key）计算相似度  -->  对所有值（value）加权混合
```

每个词元生成三个向量：
- **查询（Query, Q）**：“我在找什么？”
- **键（Key, K）**：“我包含什么？”
- **值（Value, V）**：“如果被选中，我提供什么信息？”

查询与所有键的点积产生注意力分数（attention score）。高分表示“这个键与我的查询匹配”。这些分数对值进行加权。输出就是值的加权和。

### Q、K、V 的计算

每个词元的嵌入（embedding）都通过三个可学习的权重矩阵进行投影：

```
输入嵌入（n 个词元的序列，每个 d 维）：

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

三个权重矩阵：

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

投影结果：

  Q = X @ Wq    shape: (n, dk)      每个词元的查询
  K = X @ Wk    shape: (n, dk)      每个词元的键
  V = X @ Wv    shape: (n, dv)      每个词元的值
```

直观地看，单个词元的处理流程：

```
             Wq
  x_i ------[*]------> q_i    "我在找什么？"
       |
       |     Wk
       +----[*]------> k_i    "我包含什么？"
       |
       |     Wv
       +----[*]------> v_i    "我提供什么？"
```

### 注意力矩阵

得到所有词元的 Q、K、V 后，注意力分数构成一个矩阵：

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- q1 对每个键的关注程度
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

每一行：一个词元对整个序列的注意力分布
```

### 为什么要缩放？

点积的数值会随着维度 d_k 增长而变大。如果 d_k = 64，点积可能达到几十，这会把 softmax 推到梯度消失的区域。解决方法：除以 sqrt(d_k)。

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

这样可以让数值保持在 softmax 仍能产生有效梯度的范围内。

### Softmax 把分数变成权重

Softmax 把原始分数转换为每一行上的概率分布：

```
q1 的原始分数:   [2.1, 0.3, 0.1, 0.8, 0.2]
                        |
                     softmax
                        |
注意力权重:   [0.52, 0.09, 0.07, 0.14, 0.08]   （总和约等于 1.0）
```

现在每个词元都拥有一组权重，表示它应该关注其他每个词元的程度。

### 值的加权和

每个词元的最终输出是所有值向量（value vector）的加权和：

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

以词元 1 为例：
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整流程

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

一行公式表示：

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 动手实现

### 第 1 步：从零实现 Softmax

Softmax 把原始对数几率（logits）转换为概率。为数值稳定性，先减去最大值。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 第 2 步：缩放点积注意力

核心函数。接收 Q、K、V 矩阵，返回注意力输出和权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 第 3 步：带可学习投影的自注意力类

一个完整的自注意力模块，包含 Wq、Wk、Wv 权重矩阵，并按类似 Xavier 的方式初始化。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 第 4 步：在句子上运行

为一个句子构造假嵌入，并观察注意力权重。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 第 5 步：用 ASCII 热力图可视化注意力

把权重映射成字符，快速可视化。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 实际应用

PyTorch 的 `nn.MultiheadAttention` 正是我们刚才构建内容的工业级实现，额外增加了分头和输出投影：

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

关键区别：多头注意力并行运行多组注意力函数，每组使用自己的 Q、K、V 投影（d_k = d_model / n_heads），最后拼接结果。这让模型能同时捕捉不同类型的关系。

## 交付物

本课产出：
- `outputs/prompt-attention-explainer.md`——一个通过数据库查询类比解释注意力的提示词

## 练习题

1. 修改 `scaled_dot_product_attention`，使其接受一个可选的掩码矩阵，在 softmax 之前将某些位置设为负无穷（这就是因果/解码器掩码的工作原理）
2. 从零实现多头注意力：把 Q、K、V 分成 `n_heads` 份，分别计算注意力，再拼接，最后通过输出权重矩阵 Wo 投影
3. 取两句长度相同的不同句子，输入同一个 `SelfAttention` 实例，比较它们的注意力模式。哪些变了？哪些保持不变？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 查询（Query, Q） | “问题向量” | 输入的一个可学习投影，表示该词元在寻找什么信息 |
| 键（Key, K） | “标签向量” | 一个可学习投影，表示该词元包含什么信息，用于与查询匹配 |
| 值（Value, V） | “内容向量” | 一个可学习投影，携带实际信息，并根据注意力分数聚合 |
| 缩放点积注意力（Scaled dot-product attention） | “注意力公式” | softmax(QK^T / sqrt(d_k)) @ V——缩放可防止高维下 softmax 饱和 |
| 自注意力（Self-attention） | “词元看自己和其他词元” | Q、K、V 都来自同一序列的注意力，让每个位置都能关注其他所有位置 |
| 注意力权重（Attention weights） | “关注程度” | 对位置上的概率分布，由缩放点积的 softmax 产生 |
| 多头注意力（Multi-head attention） | “并行注意力” | 运行多组不同投影的注意力函数，再拼接结果以获得更丰富的表示 |

## 延伸阅读

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)——原始 Transformer 论文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)——完整架构的最佳可视化讲解
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)——逐行 PyTorch 实现及解释
