# 多头注意力（Multi-Head Attention）

> 一个注意力头每次只学习一种关系。八个头就能学习八种关系。头是免费的，多来几个。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 · 02（从零实现自注意力）
**时间：** ~75 分钟

## 问题

单个自注意力头只计算一个注意力矩阵。这个矩阵只能捕捉一种关系——通常是能最小化训练损失的那种。如果你的数据里同时混杂了主谓一致、共指消解、长距离语篇和句法分块，单个头会把它们全部抹平成一个 softmax 分布，从而损失掉一半信号。

2017 年 Vaswani 论文给出的解决方案：并行运行多个注意力函数，每个函数有自己的 Q、K、V 投影，最后把输出拼接起来。每个头在维度为 `d_model / n_heads` 的子空间里工作。总参数量不变，表达能力却提升了。

到 2026 年，多头注意力已经成为每一款 Transformer 的默认配置。唯一的争论是*用多少个头*，以及键（key）和值（value）是否共享投影（分组查询注意力、多查询注意力、多头潜在注意力）。

## 概念

![多头注意力：拆分、注意力、拼接](../assets/multi-head-attention.svg)

**拆分。** 输入 `X` 形状为 `(N, d_model)`。先投影得到 Q、K、V，形状均为 `(N, d_model)`。再 reshape 成 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置后得到 `(n_heads, N, d_head)`。

**并行注意力。** 在每个头内部运行缩放点积注意力。每个头输出 `(N, d_head)`。各头在嵌入（embedding）的不同子空间上工作，在注意力计算本身期间互不交流。

**拼接并投影。** 把各头重新堆叠成 `(N, d_model)`，再乘以一个可学习的输出矩阵 `W_o`，形状为 `(d_model, d_model)`。`W_o` 是各头“混合信息”的地方。

**为什么有效。** 每个头都可以专门化，而不用和其他头争夺表征预算。2019–2024 年的探针研究显示出了不同的头角色：位置头、指向前一个 token 的头、复制头、命名实体头、归纳头（induction head，它们构成了上下文学习的基础）。

**2026 年的主要变体谱系：**

| 变体 | Q 头数 | K/V 头数 | 代表模型 |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2、BERT、T5 |
| Multi-query (MQA) | N | 1 | PaLM、Falcon |
| Grouped-query (GQA) | N | G（例如 N/8） | Llama 2 70B、Llama 3+、Qwen 2+、Mistral |
| Multi-head latent (MLA) | N | 压缩到低秩 | DeepSeek-V2、V3 |

GQA 是现代默认方案，因为它把 KV 缓存（KV-cache）内存缩减为原来的 `G/N`，同时几乎不损失质量。MLA 更进一步，把 K/V 压缩到一个潜在空间（latent space），在计算时再投影回来——消耗更多 FLOPs，但能节省更多内存。

## 动手实现

### 第一步：在第 02 课单头注意力的基础上拆分多头

把第 02 课的 `SelfAttention` 包上一层 split/combine。完整 NumPy 实现见 `code/main.py`；核心逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape 加一次 transpose，没有循环。这正是 PyTorch 的 `nn.MultiheadAttention` 底层做的事情。

### 第二步：对每个头运行缩放点积注意力

每个头拿到 Q、K、V 各自的切片。注意力变成批量化矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 就是一个 `bmm`。GPU 看到的是单次批量化矩阵乘法，形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`。增加头是免费的。

### 第三步：分组查询注意力（GQA）变体

只有键和值的投影方式会改变。Q 仍然分成 `n_heads` 组；K 和 V 分成 `n_kv_heads < n_heads` 组，然后通过重复扩展到与 Q 匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理阶段，这能节省内存，因为 KV 缓存里只存 `n_kv_heads` 份，而不是 `n_heads` 份。Llama 3 70B 使用 64 个查询头和 8 个 KV 头——缓存缩小了 8 倍。

### 第四步：探查每个头学到了什么

用 4 个头的 MHA 跑一个短句子。对每个头打印它的 `(N, N)` 注意力矩阵。你会发现即使随机初始化，不同头也会捕捉不同的结构——这有一部分是真实信号，一部分是子空间中旋转对称性造成的。

## 使用

在 PyTorch 里，一行代码就够了：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 中的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention 在 CUDA 上会自动调度到 Flash Attention。
# 对于 GQA，传入 Q 形状为 (B, n_heads, N, d_head)，K、V 形状为
# (B, n_kv_heads, N, d_head)。PyTorch 会自动处理重复。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**用多少个头？** 2026 年生产模型的一些经验法则：

| 模型规模 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是 64 或 128。它衡量了一个头能“看到”多少信息。低于 32 时，头会开始和缩放因子 `sqrt(d_head)` 较劲；高于 256 时，又会失去“多个小专家”的好处。

## 交付

参见 `outputs/skill-mha-configurator.md`。该技能会根据参数预算、序列长度和部署目标，为新 Transformer 推荐头数、KV 头数和投影策略。

## 练习

1. **简单。** 使用 `code/main.py` 中的 MHA，固定 `d_model=64`，把 `n_heads` 从 1 改到 16。在一个 tiny 单层模型的合成复制任务上绘制损失曲线。更多头会帮助收敛、达到平台期，还是会损害性能？
2. **中等。** 实现 MQA（所有查询头共享一个 KV 头）。测量相比完整 MHA 参数量下降多少，并计算在 N=2048 时推理 KV 缓存大小缩小了多少。
3. **困难。** 实现一个极简版多头潜在注意力（MLA）：把 K、V 压缩到秩为 `r` 的潜在向量，KV 缓存里存这个潜在向量，注意力计算时再解压。当 `r` 取多少时，缓存内存会低于完整 MHA 的 1/8，同时验证集困惑度（ppl）仍保持在 1 bit 以内？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Head | “一个注意力电路” | 维度为 `d_head = d_model / n_heads` 的一组 Q/K/V 投影，拥有自己的注意力矩阵。 |
| d_head | “头维度” | 每个头的隐藏宽度；生产环境中几乎总是 64 或 128。 |
| Split / combine | “reshape 技巧” | 在注意力前后把 `(N, d_model)` 与 `(n_heads, N, d_head)` 互相转换的 reshape + transpose。 |
| W_o | “输出投影” | 拼接各头后应用的 `(d_model, d_model)` 矩阵；头是这里混合的。 |
| MQA | “一个 KV 头” | 多查询注意力（Multi-Query Attention）：单一共享的 K/V 投影。KV 缓存最小，质量略有损失。 |
| GQA | “Llama 2 以来的默认” | 分组查询注意力（Grouped-Query Attention）：`n_kv_heads < n_heads`，通过重复与 Q 匹配。 |
| MLA | “DeepSeek 的 trick” | 多头潜在注意力（Multi-head Latent Attention）：K、V 被压缩到低秩潜在向量，在注意力时解压。 |
| Induction head | “上下文学习背后的电路” | 一对头，能检测先前出现的模式并复制其后的内容。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) —— 原始多头注意力规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) —— MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) —— 如何在训练后将 MHA 转换为 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) —— MLA 以及它在缓存内存上为何优于 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) —— 从机制角度探究头到底在做什么。
