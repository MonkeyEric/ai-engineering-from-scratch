# 差分注意力（V2）

> Softmax 注意力会把少量概率分配给每一个非匹配词元。在 10 万级别的序列长度上，这些噪声会不断累积并淹没真实信号。差分 Transformer（Differential Transformer，Ye 等人，ICLR 2025）通过计算两个 softmax 的差值来解决这个问题，从而减去共享的噪声基底。DIFF V2（Microsoft，2026 年 1 月）是面向生产栈的重构版本：解码延迟与基线 Transformer 持平、无需自定义算子、兼容 FlashAttention。本节课从 V1 到 V2 完整贯通，并提供一个可在标准库 Python 中运行的差分算子玩具实现。

**类型：** 构建
**语言：** Python（标准库）
**前置知识：** Phase 7 · 02（自注意力），Phase 7 · 15（注意力变体），Phase 10 · 14（架构概览）
**时间：** 约 60 分钟

## 学习目标

- 准确说明为什么 softmax 注意力存在噪声基底，以及为什么它随上下文长度增长。
- 推导差分注意力公式，并解释减法为什么能抵消共享噪声分量同时保留信号。
- 梳理 V1 到 V2 的改进：什么变快了、什么变简单了、什么更稳定了，以及为什么每一项改动对生产预训练都是必需的。
- 用纯 Python 从零实现差分注意力，并在合成信号加噪声查询上实证验证噪声抵消特性。

## 问题背景

标准 softmax 注意力有一个数学性质，在规模化时会变成工程痛点。对于查询 `q`，注意力权重为 `softmax(qK^T / sqrt(d))`。Softmax 永远无法输出精确的零——每个非匹配词元都会获得一定的正质量。这部分残余质量就是噪声，并且随上下文长度增长。在 128k 词元场景下，即使每个非匹配词元只获得 0.001% 的概率，127,999 个词元合计也会贡献约 12% 的总概率。模型不得不学习绕过这个随上下文增长的噪声基底。

 经验上这会表现为注意力头之间的干扰：长上下文 RAG 中出现幻觉引用、10 万词元检索任务中的“中间迷失”（lost-in-the-middle）失败，以及超过 32k 后的“大海捞针”基准准确率下降。Differential Transformer 论文（arXiv:2410.05258，ICLR 2025）测量了这一差距：在同等规模下，DIFF Transformer 达到更低的困惑度、更高的长上下文准确率，并产生更少的幻觉。

DIFF V1 存在三个问题，使其无法进入前沿预训练流水线。它的值缓存（value cache）在每个解码步需要加载两次，需要破坏 FlashAttention 兼容性的自定义 CUDA 核函数，并且其每头 RMSNorm 在 70B 以上规模的长周期训练中 变得不稳定。DIFF V2（Microsoft unilm blog，2026 年 1 月 20 日）修复了这三个问题。本节课将同时介绍两个版本，构建差分算子，并在一个玩具查询上基准测试噪声抵消效果。

## 核心概念

### Softmax 的噪声基底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 会精确为零。如果 `k_i` 与 `q` 完全无关，得分 `q . k_i` 不为 0——它以方差 `||q||^2 / d` 在零附近波动。经过 softmax 归一化后，每个无关词元仍然对加权和贡献 `O(1/N)`。所有无关词元的总贡献为 `O((N-1)/N) = O(1)`——这并不是一个小量。

模型真正想要的是类似硬 top-k 的行为：在匹配词元上权重高，其余位置接近零。Softmax 本身过于平滑，无法直接做到这一点。

### 差分思想

将每个注意力头的 Q 和 K 投影拆成两份：Q = (Q_1, Q_2)、K = (K_1, K_2)。计算两张注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法可以抵消两张图共享的噪声分布。如果两张图在 127k 个无关词元上都大致呈均匀分布（在随机初始化时确实如此），这些噪声会被抵消。信号——即在少数真正相关词元上的峰值权重——只有当它以相同幅度同时出现在两张图中时才会被抵消，而模型训练后这种情况不会发生。

`lambda` 是每个头可学习的标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负数。`lambda_init` 默认取一个较小的正数，例如 0.8。

### 为什么这相当于逐头降噪

想象两支有噪声的麦克风同时录制同一个人的声音。两者都拾取了说话声，以及相关的背景噪声。将其中一个减去另一个，共享的噪声就会下降。声音能够保留，是因为两个信号在相位或幅度上存在足够差异，避免了完全抵消。每个头的 `lambda` 学习的就是这种平衡。

### V1 与 V2：差异对比

V1 为了与基线 Transformer 保持相同参数量，将每个头的维度减半。这损害了头的表达能力，并且——更痛苦地——使每个头的值缓存减半。解码时每个步必须加载两次值缓存（每个 softmax 分支一次）。结果是：即使参数量相同，解码速度仍慢于基线。

V2 则将查询头数量翻倍，同时保持 KV 头数量不变（从升维投影中借用参数）。头的维度与基线保持一致。减法完成后，额外维度被投影回与基线 Transformer 的 O_W 投影相同的维度。这一改动同时带来三件事：

1. 解码速度与基线持平（KV 缓存只加载一次）。
2. FlashAttention 无需修改即可运行（不需要自定义核函数）。
3. 解码时的算术强度提高（每从 HBM 加载一字节数据进行更多计算）。

V2 还去除了 V1 用于稳定减法的每头 RMSNorm。在 70B 级别的预训练规模下，该 RMSNorm 会在训练后期导致不稳定。V2 用更简单的初始化方案替代它，在不引入额外模块的情况下保持训练稳定。

### 何时使用它

| 工作负载 | 收益 |
|----------|------|
| 长上下文 RAG（64k+） | 注意力图更干净，幻觉引用更少 |
| 大海捞针基准 | 超过 32k 后准确率显著提升 |
| 多文档问答 | 跨文档干扰更少 |
| 8k 代码补全 | 收益有限，不值得改动架构 |
| 短对话（< 4k） | 与基线几乎没有区别 |

价值随上下文长度增长。在 4k 词元时，噪声基底足够小，标准注意力即可胜任。在 128k 时，它已经在损害你的模型。

### 与其他 2026 年常用技术如何组合

| 特性 | 是否与 DIFF V2 兼容？ |
|------|----------------------|
| GQA | 是（V2 增加 Q 头数量，不改变 KV 头数量） |
| MLA（DeepSeek） | 原则上可以，尚无已发表论文将二者结合 |
| MoE | 是（注意力与 MLP 块相互独立） |
| RoPE | 是（保持不变） |
| YaRN / 长上下文缩放 | 是（正是 DIFF 最能发挥作用的场景） |
| FlashAttention | V2 支持（V1 不支持） |
| 投机解码（Speculative decoding） | 是（注意力改动对投机解码循环完全透明） |

## 动手实现

`code/main.py` 用纯 Python 实现了差分注意力。一个带有已知信号加噪声结构的玩具查询，让你可以直接测量噪声抵消比率。

### 步骤 1：标准 softmax 注意力

标准库矩阵运算：列表的列表、手动矩阵乘法、通过减去最大值实现数值稳定的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤 2：将 Q、K 拆成两半

V1 风格：将头维度减半。V2 风格：保持头维度不变，将头数量翻倍。玩具实现采用 V1 风格以方便教学——数学完全相同，只有张量形状不同。

### 步骤 3：两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负数。这没有问题——值缓存仍然可以处理带符号的贡献。后续的 V 投影会吸收这些符号。

### 步骤 4：噪声抵消测量

构建长度为 1024 的合成序列。将信号词元放在已知位置，其余填充噪声。分别计算（a）标准 softmax 注意力在信号位置上的权重，以及（b）差分注意力在信号位置上的权重。测量各自的信噪比（signal-to-noise ratio）。DIFF 注意力可靠地产生比标准注意力高 3 到 10 倍的信噪比，具体取决于两个分支被训练到多大程度分化。

### 步骤 5：V1 与 V2 的参数量核算

给定配置（hidden=4096，heads=32，d_head=128），打印：

- 基线 Transformer：Q、K、V 每个大小为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 每个大小为 `hidden * hidden`，V 大小为 `hidden * hidden`（不变），内部头维度减半。额外增加每头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 大小为 `2 * hidden * hidden`，K 大小为 `hidden * hidden`，V 大小为 `hidden * hidden`。额外维度在 O_W 之前投影回去。同样增加 `lambda` 参数。

玩具代码会测量 V2 的额外参数开销（每个注意力块大约额外 `hidden * hidden`）并打印出来。

## 应用场景

截至 2026 年 4 月，DIFF V2 尚未在所有生产推理服务中上线，但 vLLM 和 SGLang 的集成正在进行中。同时，这种模式已经出现在：

- Microsoft 内部长上下文生产模型。
- 多个面向 256k+ 上下文的开源模型训练复现。
- 混合架构中，将 DIFF 注意力与滑动窗口注意力交替使用。

2026 年你会考虑使用它的场景：

- 从头训练面向 64k+ 有效上下文的新模型。从一开始就加入差分注意力；后续重训成本高昂。
- 微调长上下文模型，且“中间迷失”失败主导了你的评估。在 Q 投影上使用 LoRA 可以近似 DIFF 结构。

你不会考虑使用的场景：

- 你正在部署一个长上下文表现稳定的预训练稠密模型。对已有权重来说，重训成本很少能回本。
- 你的上下文始终在 16k 以下。噪声基底可以忽略。

## 交付物

本节课会生成 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉画像和训练预算，它会产出一份集成方案，说明如何在新预训练运行或 LoRA 微调中加入差分注意力。

## 练习题

1. 运行 `code/main.py`。验证差分注意力在合成查询上报告的信噪比高于标准 softmax 注意力。改变噪声幅度，找出标准注意力变得不可用的临界点。

2. 计算从基线到 DIFF V1、以及从基线到 DIFF V2 的参数量变化，针对一个 7B 级别模型（hidden=4096，heads=32，d_head=128，32 层）。说明哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文的 Section 3（arXiv:2410.05258）和 DIFF V2 Hugging Face blog 的 Section 2。用两句话解释为什么 V1 的每头 RMSNorm 是必需的，以及为什么 V2 可以在不导致训练发散的情况下去除它。

4. 实现一个消融实验：用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完整减法）计算差分注意力。在合成查询上测量信噪比如何随参数变化，找出使信噪比最大的 `lambda`。

5. 将玩具实现扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。证明 KV 缓存大小与具有相同 (8, 32) 配置的基线 GQA 模型一致。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 差分注意力（Differential attention） | “两个 softmax 相减” | 将 Q、K 各拆成两半，计算两张 softmax 图，用第一张减去第二张（按 lambda 缩放），再乘以 V |
| 噪声基底（Noise floor） | “Softmax 的非零尾部” | Softmax 给每个无关词元分配的 O(1/N) 权重，在长上下文上求和为 O(1) |
| lambda | “减法缩放系数” | 每头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负数 |
| DIFF V1 | “ICLR 2025 版本” | 原始差分 Transformer；将头维度减半以保持参数量，需要自定义核函数，解码更慢 |
| DIFF V2 | “2026 年 1 月修复版” | Q 头翻倍、保持 KV 头；解码速度与基线持平，且兼容 FlashAttention |
| 每头 RMSNorm（Per-head RMSNorm） | “V1 的稳定器” | V1 在差分后应用的额外归一化；V2 为预防训练后期不稳定而移除 |
| 信噪比（Signal-to-noise ratio） | “注意力浪费了多少” | 真实信号位置上的权重与无关位置平均权重之比 |
| 中间迷失（Lost in the middle） | “长上下文失败模式” | 长上下文中间文档检索准确率下降的经验现象——差分注意力可缓解此问题 |
| 算术强度（Arithmetic intensity） | “每加载一字节的 FLOPs” | V2 通过每次加载 KV 时翻倍查询数量，在解码阶段提升的比率；对内存受限的解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) —— 原始论文，包含噪声抵消理论与长上下文消融
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) —— 生产栈重构，匹配基线解码速度，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) —— 理论分析：减法为何能恢复预训练注意力结构
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) —— 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) —— DIFF 所减去的基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) —— 差分注意力针对的长上下文基准
