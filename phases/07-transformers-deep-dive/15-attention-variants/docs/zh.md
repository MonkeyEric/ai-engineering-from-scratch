# 注意力变体 —— 滑动窗口、稀疏与差分注意力

> 完整注意力是一个圆：每个词元（token）都能看到所有词元，而显存（memory）为此买单。四种变体弯曲了这个圆的形状，挽回了一半开销。

**类型：** 动手实践  
**语言：** Python  
**先决条件：** Phase 7 · 02（自注意力），Phase 7 · 03（多头注意力），Phase 7 · 12（KV 缓存 / Flash Attention）  
**时长：** 约 60 分钟

## 问题背景

完整注意力（attention）在序列长度上的显存与计算量（compute）开销均为 `O(N²)`。对于 128K 上下文的 Llama 3 70B，每层就有 160 亿个注意力项，再乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活显存，但并未改变算术开销——每个 token 仍然要关注其他所有 token。

三类变体改变了注意力矩阵本身的拓扑结构：

1. **滑动窗口注意力（Sliding Window Attention, SWA）。** 每个 token 只关注固定大小的邻近窗口，而非完整前缀。显存与计算量降至 `O(N · W)`，其中 `W` 为窗口大小。应用于 Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **稀疏 / 分块注意力（Sparse / Block Attention）。** 只有选定的 `(i, j)` 对会被打分，其余项被强制置为零权重。代表工作包括 Longformer、BigBird、OpenAI sparse transformer。
3. **差分注意力（Differential Attention）。** 用独立的 Q/K 投影计算两份注意力图，再相互相减。它能消除把权重泄露给前几个 token 的“注意力汇聚（attention sink）”。代表为微软的 DIFF Transformer（2024）。

这些变体可以共存。2026 年的前沿模型通常会混合使用：大部分层采用 SWA-1024，每五层设一层全局完整注意力，还有少量差分头负责净化检索结果。Gemma 3 的 5:1 SWA 与全局注意力比例已成为当前教科书级默认配置。

## 概念

### 滑动窗口注意力（SWA）

因果 SWA 中，位置 `i` 处的每个查询（query）只关注 `[i - W, i]` 内的位置；双向 SWA 中则关注 `[i - W/2, i + W/2]`。窗口外的 token 在分数矩阵中会被赋值为 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

当 `N = 8192`、`W = 1024` 时，分数矩阵期望有 1024 × 8192 个非零项——相当于 8 倍的缩减。

**KV 缓存（KV cache）随 SWA 缩减。** 每层只需保留最后 `W` 个 token 的 K 和 V。以类 Gemma-3 配置（窗口 1024、上下文 128K）为例，KV 缓存可缩减 128 倍。

**质量代价。** 纯 SWA 的 Transformer 在长距离检索上表现吃力。解决办法是将 SWA 层与完整注意力层交错排列：Gemma 3 采用 5:1 的 SWA:全局 比例。Mistral 7B 则使用因果 SWA 堆栈，让信息通过重叠窗口“向前流动”——每层将有效感受野（effective receptive field）扩展 `W`，经过 `L` 层后，模型最多能回看到 `L × W` 个 token。

### 稀疏 / 分块注意力

预先选定一个 `N × N` 的稀疏模式。三种经典形状：

- **局部 + 步长稀疏（OpenAI sparse transformer）。** 关注最近的 `W` 个 token，以及此前每隔 `stride` 个 token 采样一次的位置。以 `O(N · √N)` 的计算量同时捕捉局部与长距离信息。
- **Longformer / BigBird。** 局部窗口 + 少量全局 token（例如 `[CLS]`）与所有 token 互相 attend + 随机稀疏连接。在同等质量下可实证支持 2 倍上下文。
- **原生稀疏注意力（Native Sparse Attention，DeepSeek，2025）。** 学习哪些 `(Q, K)` 分块重要，在核函数层面跳过零块。与 FlashAttention 兼容。

稀疏注意力本质上是核工程（kernel engineering）的故事。数学很简单（给分数矩阵加掩码），收益来自永远不把零项加载进 SRAM。FlashAttention-3 和 2026 年的 FlexAttention API 让自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力（DIFF Transformer，2024）

常规注意力存在一个“注意力汇聚（attention sink）”问题：softmax 强制每一行求和为 1，导致那些本来不想关注任何特定内容的 token，会把权重倾倒给第一个（或前几个）token。这会侵占本应分配给真正内容的容量。

差分注意力通过计算**两份**注意力图并相减来解决该问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是可学习的标量（通常为 0.5–0.8）。A1 捕捉真实内容的权重，A2 捕捉汇聚项。相减后汇聚被抵消，权重重新分配给相关 token。

微软（2024）报告的结果：困惑度（perplexity）降低 5–10%，在相同训练长度下有效上下文延长 1.5–2 倍，大海捞针（needle-in-haystack）检索更敏锐。

### 变体对比

| 变体 | 计算量 | KV 缓存 | 相比完整注意力的质量 | 生产应用 |
|------|--------|---------|----------------------|----------|
| 完整注意力 | O(N²) | 每层 O(N) | 基线 | 各模型默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | -0.1 ppl，配合全局层表现良好 | Gemma 2/3、Phi-3-Long |
| 局部 + 步长稀疏 | O(N·√N) | 混合 | 与 SWA 相近 | OpenAI sparse transformer、Longformer |
| BigBird（局部 + 全局 + 随机） | 约 O(N) | 混合 | 在 2 倍上下文下与完整注意力相当 | 早期长上下文 BERT |
| 原生稀疏注意力（DeepSeek-V3.2） | O(N · 活跃比例) | 每层 O(N) | 与基线差距 < 0.05 ppl | DeepSeek-V3.2，2025 |
| 差分注意力 | O(2·N²) | 每层 O(2N) | -5% 至 -10% ppl | DIFF Transformer、2026 年初的模型 |

## 动手实现

参见 `code/main.py`。我们实现了一个因果掩码比较器，在玩具序列上并排展示完整注意力、SWA、局部+步长稀疏以及差分注意力。

### 步骤 1：完整因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第 07 课的基线。下三角；对角线以上权重为零。

### 步骤 2：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

只有一个参数——`window`。当 `window >= n` 时，退化为完整因果注意力；当 `window = 1` 时，每个 token 只关注自身。

### 步骤 3：局部 + 步长稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集的局部窗口加上从序列开头起每隔 `stride` 个 token 采样一次的位置。随着层数增加，感受野（receptive field）以对数步长增长。

### 步骤 4：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力前向计算，用可学习的混合系数相减。代码中会比较单层注意力与差分注意力的 attention-sink 热力图，观察汇聚现象如何消失。

### 步骤 5：KV 缓存大小

打印 `N = 131072` 时每种变体每层的缓存大小。SWA 与稀疏变体可减少 10–100 倍，差分注意力则翻倍。请谨慎支付这笔显存账单。

## 实际使用

2026 年的生产级用法：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 以 5:1 的比例混合 SWA（窗口=1024）与全局层。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 接受一个掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

它会被编译成自定义 Triton 核函数。对于常见模式，速度在 FlashAttention-3 的 10% 以内，且掩码函数就是一个 Python 可调用对象。

**如何选择：**

- **纯完整注意力** —— 适用于约 16K 以内的上下文，或检索质量至关重要的场景。
- **SWA + 全局混合** —— 长上下文（>32K）、训练与推理受显存限制。这是 2026 年 32K 以上的默认选择。
- **稀疏分块注意力** —— 自定义核函数、自定义模式。仅用于特殊工作负载（检索、音频）。
- **差分注意力** —— 任何受 attention-sink 污染影响的工作负载（长上下文 RAG、大海捞针检索）。

## 交付

参见 `outputs/skill-attention-variant-picker.md`。该技能会根据目标上下文长度、检索需求以及训练/推理的计算配置，为新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 会将每行最后 4 个 token 之外的所有位置置零。验证 `window=n` 能与完整因果注意力逐位一致。
2. **中等。** 在第 07 课大作业基础上实现因果 SWA，`window=1024`。在 tinyshakespeare 上训练 1,000 步。与完整注意力相比，验证损失（val loss）回退了多少？峰值显存降低了多少？
3. **困难。** 在大作业模型中实现 Gemma-3 风格的 5:1 层混合（5 层 SWA，1 层全局）。在参数量相同的情况下，与纯 SWA 和纯全局基线比较损失、显存与生成质量。
4. **困难。** 实现每个头（head）可学习 `λ` 的差分注意力。在合成检索任务（一根 needle，2,000 个干扰项）上训练。在参数量相同的情况下，与单层注意力基线比较检索准确率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 滑动窗口注意力（SWA） | “局部注意力” | 每个查询只关注最近的 `W` 个 token；KV 缓存（KV cache）缩减至 `O(W)`。 |
| 有效感受野 | “模型能回看多远” | 在窗口为 `W` 的 `L` 层 SWA 堆栈中，最多可回看到 `L × W` 个 token。 |
| Longformer / BigBird | “局部 + 全局 + 随机” | 包含少量始终参与 attention 的全局 token 的稀疏模式；早期的长上下文方案。 |
| 原生稀疏注意力 | “DeepSeek 的核技巧” | 学习块级稀疏性；在核函数层面跳过零块，同时保持质量。 |
| 差分注意力 | “两张图，一相减” | DIFF Transformer：从第一张注意力图中减去可学习的 `λ` 倍的第二张图，以抵消 attention sink。 |
| 注意力汇聚（attention sink） | “权重泄露给 token 0” | softmax 归一化强制每行和为 1；信息不足的查询会把权重倾倒到位置 0。 |
| FlexAttention | “掩码即 Python” | PyTorch 2.5+ API，将任意掩码函数编译成 FlashAttention 形态的核函数。 |
| 层类型混合 | “5:1 SWA 对全局” | 在堆栈中交错稀疏注意力层与完整注意力层，以更低显存保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) —— 滑动窗口 + 全局 token 的经典论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) —— 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) —— OpenAI 的局部 + 步长模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) —— 1:1 的 SWA:全局混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) —— 窗口 1024 的 5:1 混合，如今已成教科书级默认。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) —— DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) —— DeepSeek-V3.2 的可学习稀疏注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) —— “实际使用”节中 mask-as-callable 模式的 API 参考。
