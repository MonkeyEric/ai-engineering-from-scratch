# 为什么是 Transformer —— RNN 的痛点

> RNN 逐个处理 token。Transformer 一次性处理所有 token。这一单一的架构赌注改变了 2017 年后深度学习的所有扩展曲线。

**类型：** 学习
**语言：** Python
**先修：** Phase 3（深度学习核心）、Phase 5 · 09（序列到序列）、Phase 5 · 10（注意力机制）
**时长：** 约 45 分钟

## 问题所在

在 2017 年之前，全球所有最先进的序列模型——语言、翻译、语音——都是循环神经网络（RNN）。LSTM 和 GRU 在长达半年的时间里霸占了相当于 ImageNet 地位的翻译基准测试。它们是人们手边唯一的工具。

但它们有三个致命弱点。顺序计算意味着你无法在时间轴上并行化：token `t+1` 需要 token `t` 的隐藏状态（hidden state）。一个 1,024 个 token 的序列，在单周期可完成 1,000,000 次浮点运算的 GPU 上却要串行 1,024 步。训练墙上时间随序列长度线性增长，而硬件却是为并行设计的。

梯度消失（vanishing gradients）意味着 50 个 token 之前的信息已经被压缩了 50 次非线性变换。门控循环单元（LSTM、GRU）缓解了这种挤压，但从未根除。长程依赖（long-range dependencies）——例如“我去年夏天在飞往京都的飞机上读的那本书是……”—— routinely 失败。

固定宽度的隐藏状态意味着编码器（encoder）必须把整个源序列压缩成单个向量后，解码器（decoder）才能看到任何内容。无论源序列是 5 个 token 还是 500 个 token，瓶颈的形状都一样。

2017 年的论文《Attention Is All You Need》提出了一个激进的想法：彻底抛弃循环。让每个位置同时关注所有其他位置。用一次大规模矩阵乘法（matrix multiplication）取代 1,024 次顺序乘法。

其结果是到 2026 年统治了所有模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物（AlphaFold 3）、机器人（RT-2）。同一个模块，不同的输入。

## 核心概念

![RNN 顺序计算 vs Transformer 并行注意力](../assets/rnn-vs-transformer.svg)

**循环作为瓶颈。** RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步都依赖前一步。你无法在 `h_4` 之前计算 `h_5`。在现代拥有 10,000+ 并行核心的 GPU 上，这在长序列上浪费了 99% 的硅片算力。

**注意力（attention）作为广播。** 自注意力（self-attention）同时为每一对 `(i, j)` 计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N 的注意力矩阵在一次批处理矩阵乘法中填满。没有任何一步依赖其他步。GPU 非常喜欢这种计算。

**加速不是常数级别的。** 它是 `O(N)` 串行深度与 `O(1)` 串行深度的差别。实践中，在相同硬件和 N=512 的情况下，Transformer 每个 epoch 的训练速度快 5–10 倍，并且这个差距随序列长度 widening，直到撞上注意力的 `O(N²)` 内存墙（后来 Flash Attention 修复了这一点——参见第 12 课）。

**Transformer 的代价。** 注意力的内存按 `O(N²)` 缩放。对于 2K 上下文没问题。对于 128K 上下文，你需要滑动窗口、RoPE 外推、Flash Attention 分块或线性注意力变体。RNN 的时间和内存都是 `O(N)`；Transformer 用时间换内存，然后通过并行化再把时间赢回来。

**归纳偏置（inductive bias）的转变。** RNN 假设局部性和近因性。Transformer 不做任何假设——每一对位置都有可能被关注。这就是为什么 Transformer 需要更多数据才能训练好，但一旦数据充足就能扩展得更远。Chinchilla（2022）将这一点形式化：给定足够的 token，Transformer 总是能在同等参数量（parameter count）下击败 RNN。

## 动手实现

这里不搭建真实神经网络——我们用数值方式模拟核心瓶颈，让你在笔记本上切身感受到差距。

### 第一步：测量串行深度

参见 `code/main.py`。我们构建两个函数。一个把序列编码成加法链（串行，类似 RNN）。一个把序列编码成并行归约（广播，类似注意力）。数学相同，依赖图不同。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # 无法并行化：h 依赖于前一个 h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # 每个 x 相互独立
```

我们对最长 100,000 个元素的序列进行计时。RNN 版本是 O(N) 且单 CPU 流水线。即便在纯 Python 中，当长度 ≥ 1,000 时，注意力风格的归约也会更快，因为 Python 的 `sum()` 是用 C 实现的，每步没有解释器开销。

### 第二步：统计理论运算量

两个算法都执行 N 次加法。差别在于*依赖深度*：在开始下一步之前必须先顺序执行多少操作。RNN 深度 = N。注意力深度 = 树形归约的 log(N)，或并行扫描的 1。决定 GPU 时间的是深度，而不是操作数。

### 第三步：长序列上的经验缩放

我们打印一张计时表，让 O(N) 的差距一目了然。在 2026 年的 Mac 笔记本上，长度小于 1,000 的序列快得无法测量。长度 100,000 时呈现出清晰的线性扫描。把这个现象放大到 16,384 个 token、12 层等效 LSTM 的 Transformer，你就会明白为什么 2016 年的训练墙上时间是一道关卡。

## 何时使用

2026 年仍然选择 RNN 的场景：

| 场景 | 选择 |
|------|------|
| 流式推理，逐 token 输出，恒定内存 | RNN 或状态空间模型（Mamba、RWKV） |
| 超长序列（>100 万 token），注意力内存爆炸 | 线性注意力、Mamba 2、Hyena |
| 没有矩阵乘法加速器的边缘设备 | 深度可分离 RNN 仍在 FLOPs/瓦特上占优 |
| 其他所有情况（训练、批量推理、最长 128K 上下文） | Transformer |

状态空间模型（state-space model，SSM）如 Mamba，本质上是用结构化参数化获得两者优点的 RNN：`O(N)` 的扫描内存，通过选择性扫描（selective scan）实现并行训练。它们能恢复 Transformer 约 90% 的质量，并拥有更好的长上下文缩放能力。2026 年，大多数前沿实验室训练的是 SSM+Transformer 混合模型（例如 Jamba、Samba）——循环并未消亡，而是成为一种组件。

## 交付物

参见 `outputs/skill-architecture-picker.md`。该技能会根据序列长度、吞吐（throughput）和训练预算约束，为新序列问题挑选架构。当训练量超过 10 亿 token 时，它应当始终拒绝推荐纯 RNN，除非明确说明权衡。

## 练习

1. **简单。** 从 `code/main.py` 中取出 `rnn_style`，把标量隐藏状态替换成长度为 64 的隐藏状态向量。重新测量。串行开销随隐藏状态维度（hidden-state dimension）增长多少？
2. **中等。** 用纯 Python 实现并行前缀和（Hillis-Steele 扫描）。验证它在长度 1024 上与串行扫描输出数值相同。统计深度。
3. **困难。** 把注意力风格的归约用 PyTorch 移植到 GPU。在序列长度从 64 扫到 65,536 的过程中对两者计时。绘制并解释曲线形状。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 循环（Recurrence） | “RNN 是串行的” | 第 `t` 步依赖第 `t-1` 步的计算，迫使时间轴上串行执行。 |
| 串行深度（Serial depth） | “计算图有多深” | 依赖操作的最长链；即便硬件无限，也限制了墙上时间。 |
| 注意力（Attention） | “让 token 互相看” | 加权求和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 与 j 的相似度得分。 |
| 上下文窗口（Context window） | “模型能看多远” | 注意力层能作为输入的位置数量；二次内存代价在这里缩放。 |
| 归纳偏置（Inductive bias） | “架构内置的假设” | 关于数据形态的先验；CNN 假设平移不变性，RNN 假设近因性。 |
| 状态空间模型（State-space model） | “有代数背景的 RNN” | 通过结构化状态空间矩阵参数化，实现并行训练的循环。 |
| 二次瓶颈（Quadratic bottleneck） | “上下文为什么这么贵” | 注意力内存 = `O(N²)` 序列长度；Flash Attention 隐藏了常数，而非缩放趋势。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 终结主流 NLP 中循环地位的论文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 注意力的诞生之地，彼时还被铆接在 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) —— 原始 LSTM 论文，留作记录。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) —— 对 Transformer 的现代循环式回应。
