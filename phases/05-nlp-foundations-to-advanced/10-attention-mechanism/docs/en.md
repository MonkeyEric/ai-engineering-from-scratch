# 注意力机制 — 突破性进展

> 解码器不再眯眼盯着一份压缩摘要，而是开始查看整个源序列。此后的一切，都是注意力机制加工程技巧。

**类型：** 动手构建
**语言：** Python
**前置知识：** 第 5 阶段 · 第 09 课（序列到序列模型）
**时间：** 约 45 分钟

## 问题所在

第 09 课以一个有所节制的失败收尾。一个基于 GRU 的编码器-解码器在玩具复制任务上，序列长度为 5 时准确率可达 89%，长度到 80 时却跌近随机水平。原因是结构性的，并非训练 bug：编码器提取的所有信息都必须塞进一个固定大小的隐藏状态，而解码器永远只能看到它。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行代码级别的修复。他们不再只把最终编码器状态传给解码器，而是保留每一个编码器状态。在解码器的每一步，计算所有编码器状态的加权平均，权重回答“此刻解码器需要多看编码器位置 `i` 多少？”这个加权平均就是上下文（context），并且每一步都会变化。

这就是全部思想。Transformer 扩展了它。自注意力把它用到单个序列上。多头注意力并行运行它。但 2014 年的版本已经打破了瓶颈；一旦理解它，转向 Transformer 主要是工程，而非概念。

## 核心概念

![Bahdanau 注意力：解码器查询所有编码器状态](../assets/attention.svg)

在解码器每一步 `t`：

1. 用前一个解码器隐藏状态 `s_{t-1}` 作为 **query（查询）**。
2. 把它与每一个编码器隐藏状态 `h_1, ..., h_T` 打分，每个编码器位置得到一个标量。
3. 对分数做 softmax，得到注意力权重 `α_{t,1}, ..., α_{t,T}`，其和为 1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`，即编码器状态的加权平均。
5. 解码器接收 `c_t` 和前一个输出 token，生成下一个 token。

关键就是这个加权平均。当解码器需要把 "Je" 翻译成 "I" 时，它会高亮编码器上 "Je" 对应的状态，压低其他状态；当需要 "not" 时，它会高亮 "pas"。上下文向量在每一步都被重新塑造。

## 形状（让每个人都头疼的地方）

这是每个注意力实现第一次都会出错的地方。请慢慢读。

| 量 | 形状 | 说明 |
|-------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 若用 BiLSTM，则 `d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 做 softmax 后 |
| 上下文向量 `c_t` | `(d_h,)` | 与单个编码器状态同形状 |

**Bahdanau（加性）打分。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 形状为 `(d_s,)`，`h_i` 形状为 `(d_h,)`。
- `W_a` 形状为 `(d_attn, d_s)`。`U_a` 形状为 `(d_attn, d_h)`。
- tanh 内部求和后形状为 `(d_attn,)`。
- `v_α` 形状为 `(d_attn,)`。与 `v_α` 做内积后坍缩为标量。**这就是 `v_α` 的作用。** 它并不神奇，只是把 attention 维度的向量投影成标量分数的投影。

**Luong（乘性）打分。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬性约束。若编码器是双向的，就别用这个。
- `general`：`e_{t,i} = s_t^T * W * h_i`，`W` 形状为 `(d_s, d_h)`。去掉了维度相等的约束。
- `concat`：本质上就是 Bahdanau 形式。很少用，因为前两种更便宜。

**一个值得单独指出的 Bahdanau / Luong 陷阱。** Bahdanau 用的是 `s_{t-1}`（生成当前词**之前**的解码器状态）。Luong 用的是 `s_t`（生成当前词**之后**的状态）。混淆二者会产生微妙错误的梯度，极难调试。选定一篇论文就坚持它的约定。

## 动手实现

### 第 1 步：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的表格检查形状。`encoder_states` 形状为 `(T_enc, d_h)`。`projected_enc` 形状为 `(T_enc, d_attn)`。`projected_dec` 形状为 `(d_attn,)`，会广播。`combined` 形状为 `(T_enc, d_attn)`。`scores` 形状为 `(T_enc,)`。`weights` 形状为 `(T_enc,)`。`context` 形状为 `(d_h,)`。搞定。

### 第 2 步：Luong dot 与 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个都三行。这就是 Luong 论文为何影响深远。大多数任务上准确率相同，代码却少得多。

### 第 3 步：一个完整的数值示例

给定三个编码器状态（大致对应 "cat"、"sat"、"mat"）和一个与第一个对齐最强的解码器状态，注意力分布会集中在位置 0。如果解码器状态转向与最后一个对齐，注意力就会移到位置 2。上下文向量随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后把解码器状态移近第三个编码器状态，观察权重如何偏移。就是这样。注意力就是显式的对齐。

### 第 4 步：为什么这是通往 Transformer 的桥梁

把上面的语言翻译成 Q/K/V：

- **Query** = 解码器状态 `s_{t-1}`
- **Key** = 编码器状态（我们用来打分的对象）
- **Value** = 编码器状态（我们加权求和的对象）

在经典注意力中，key 和 value 是同一个东西。自注意力把它们分开：你可以让序列自己查询自己，K 和 V 用不同的可学习投影。多头注意力并行运行多组不同的可学习投影。Transformer 则把整层结构堆叠多次，并去掉 RNN。

数学是一样的。形状是一样的。从 Bahdanau 注意力到缩放点积注意力的教学跨越，主要是符号不同。

## 使用现成的

PyTorch 和 TensorFlow 都直接提供注意力。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层。query batch 有 5 个位置，key/value batch 有 10 个位置，维度均为 128，8 个头。`output` 是融入上下文后的新 query。`weights` 是可以可视化的 5×10 对齐矩阵。

### 经典注意力为何仍然重要

- 教学价值。单头、单层、基于 RNN 的版本让每一个概念都清晰可见。
- 在设备端 Transformer 放不下的序列任务中。
- 任何 2014–2017 年的论文。不了解 Bahdanau 的约定就会误读它们。
- 机器翻译中的细粒度对齐分析。原始注意力权重即使在 Transformer 模型上也是一种可解释性工具，读懂它们需要先知道它们是什么。

### 注意力权重作为解释的陷阱

注意力权重看起来很可解释。它们是跨位置求和为 1 的权重；可以画图；数值高意味着“看了这里”。审稿人很喜欢。

但它们不像看起来那么可解释。Jain 和 Wallace（2019）指出，对某些任务，注意力分布可以被置换、替换为任意替代分布而不改变模型预测。永远不要把注意力权重当作推理证据来报告，除非有消融或反事实检验支持。

## 交付物

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。** 实现 `softmax` 掩码，让编码器中的 padding token 注意力权重为零。在包含变长序列的 batch 上测试。
2. **中等。** 为 Luong 的 `general` 形式加入多头注意力。把 `d_h` 分成 `n_heads` 组，逐头做注意力，再拼接。验证单头情况与之前实现一致。
3. **困难。** 用带 Bahdanau 注意力的 GRU 编码器-解码器训练第 09 课的玩具复制任务。绘制准确率随序列长度变化的曲线，与无注意力基线对比。你应该看到随着长度增加，差距扩大，从而验证注意力机制确实缓解了瓶颈。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Attention | 看东西 | 对 value 序列的加权平均，权重由 query-key 相似度计算得到。 |
| Query, Key, Value | QKV | 三个投影：Q 提问，K 是被匹配对象，V 是被返回对象。 |
| Additive attention | Bahdanau | 前馈分数：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong dot / general | 分数为 `q^T k` 或 `q^T W k`。更便宜，大多数任务准确率相同。 |
| Alignment matrix | 漂亮的图 | 注意力权重排成 `(T_dec, T_enc)` 网格。读它可以看到模型关注了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原始论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种打分变体及对比。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 关于可解释性的警示。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 可运行的 PyTorch 逐步教程。
