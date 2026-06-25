# 视觉自回归建模（VAR）：下一尺度预测

> 扩散（diffusion）模型按时间步迭代采样（去噪步骤），而 VAR 按尺度迭代采样——它先预测 1×1 词元，再预测 2×2、4×4，直至最终分辨率，每一尺度都以之前的尺度为条件。2024 年的论文表明，VAR 在图像生成上符合 GPT 风格的缩放定律（scaling law），并在相同计算预算下超越了 DiT。本课将构建其核心机制。

**类型：** 构建
**语言：** Python（使用 PyTorch）
**前置知识：** 第 7 阶段第 03 课（Multi-Head Attention），第 8 阶段第 06 课（DDPM）
**时间：** 约 90 分钟

## 问题背景

自回归（autoregressive）生成之所以主导语言建模，是因为它的扩展具有可预测性：更多计算、更多参数、更低的困惑度（perplexity）、更好的输出。在 2024 年之前，图像生成有两种主要的自回归尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（基于 VQ-VAE 码逐词元）。

两者都受制于生成顺序问题。像素和词元分布在二维网格中，但自回归模型必须按一维光栅顺序依次访问。一个早期角落像素根本不知道图像最终会变成什么样。生成质量的扩展性不如文本上的 GPT，在相同计算量下也从未达到扩散模型的质量。

VAR 通过改变生成对象来解决生成顺序问题。它不再在空间中逐个预测图像词元，而是按逐步提升的分辨率预测整张图像。第 1 步：预测 1×1 词元（图像整体的“摘要”）。第 2 步：预测 2×2 词元网格（较粗的特征）。第 3 步：预测 4×4 网格。第 K 步：预测最终的 (H/8)×(W/8) 网格。

每个尺度关注所有之前的尺度（按“尺度顺序”保持因果性），而在同一尺度内部并行计算。顺序问题随之消失：尺度 k 上的整张图像在一次 Transformer 前向传播中即可完成。

## 核心概念

### 多尺度 VQ-VAE 分词器

VAR 需要一个**多尺度离散分词器（tokenizer）**。对于图像 x，它生成一系列分辨率逐步提升的词元网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用同一个码本（codebook，典型大小为 4096–16384）。每个尺度的词元化不是独立的——训练目标是让各尺度的残差相加后重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是**残差 VQ（residual VQ）**的一种变体。尺度 k 捕捉的是尺度 1..k-1 遗漏的信息。解码器取所有尺度嵌入（embedding）之和并生成图像。

多尺度 VQ 分词器只需训练一次（类似 VQGAN），然后冻结。所有生成工作都由其上层的自回归模型完成。

### 下一尺度预测

生成模型是一个 Transformer，它接收所有之前尺度的词元，并预测下一尺度的词元。

输入序列结构如下：

```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入（position embedding）同时编码尺度索引和尺度内的空间位置。注意力（attention）在尺度顺序上保持因果性：尺度 k、位置 (i, j) 的词元可以关注尺度 1..k 的所有词元，以及尺度 k 中按某种内部顺序排在它之前的词元（VAR 使用固定的位置注意力，尺度内部没有因果性——同一尺度内的所有位置都并行预测）。

训练损失（loss）：在每个尺度 k，基于所有先前尺度的词元预测 z_k。对离散 VQ 码使用交叉熵损失（cross-entropy loss）。整体结构与 GPT 相同，只是这里的“序列”按尺度组织。

### 生成过程

推理（inference）时：

```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

当 K = 10 个尺度时，生成只需 10 次 Transformer 前向传播。每次前向传播都并行地输出整个尺度——尺度内部不存在逐词元的自回归。对于 256×256 图像，这大约只需 10 次传播，而 DiT 需要 28–50 次。

### 为什么下一尺度优于下一词元

三个结构性优势：

1. **由粗到细符合自然图像统计。** 人类视觉感知和图像数据集都表现出尺度相关的规律性：低频结构稳定且可预测，高频细节依赖于低频内容。下一尺度预测正是利用了这一特性。

2. **尺度内并行生成。** 与 GPT 风格的词元自回归不同，VAR 一步就能生成某个尺度的所有词元。有效生成长度从线性变为对数尺度。

3. **没有生成顺序偏差。** 尺度 k 的词元能看到整个尺度 k-1；不存在“左侧”或“上方”的偏差，避免早期词元在晚期上下文尚未确定时就被迫确定。

### 缩放定律

Tian 等人证明，VAR 在 ImageNet 上的 FID 遵循幂律缩放曲线——就像 GPT 对困惑度所做的那样。参数或计算量翻倍，误差大约减半。这是首个像语言模型一样清晰地展现出这种缩放行为的图像生成模型。因此，对 VAR 尺度扩展的预测可以从计算量推导，而不再是对每个架构的经验猜测。

### 与扩散模型的关系

VAR 与扩散（diffusion）模型共享同样的数据压缩思路：两者都把生成问题拆分成一系列更简单的子问题。

- 扩散：逐步加噪，学习撤销一步。
- VAR：逐步增加分辨率，学习预测下一尺度。

它们是穿越同一问题的不同轴线。两者都能得到可处理的条件分布（conditional distribution）。经验上，VAR 推理（inference）更快（传播次数更少，且尺度内完全并行），在类别条件（class-conditional）ImageNet 上达到或超过 DiT 的表现。文本条件（text-conditional）VAR（VARclip、HART）是一个活跃的研究方向。

## 动手实现

在 `code/main.py` 中，你将：

1. 在合成“图像”数据（二维高斯环）上构建一个微型**多尺度 VQ 分词器**。
2. 训练一个 **VAR 风格 Transformer**，对词元进行下一尺度预测。
3. 通过调用 Transformer 4 次（4 个尺度）进行采样并解码。
4. 验证按尺度顺序训练确实能让生成在尺度内部并行。

这是一个玩具级实现。重点是观察按尺度组织的注意力掩码和尺度内并行生成是否真正生效。

## 交付成果

本课将生成 `outputs/skill-var-tokenizer-designer.md`——一项关于设计多尺度分词器的技能，包括：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融实验。** 用 4、6、8、10 个尺度训练 VAR。衡量重建质量与自回归传播次数的权衡。更多尺度 = 更精细的残差 = 更好质量，但传播次数更多。

2. **码本大小。** 用 512、4096、16384 的码本大小训练分词器。更大的码本重建更好，但预测更难。找到拐点。

3. **尺度内并行性检查。** 对训练好的 VAR 显式测量注意力模式。在尺度 k 内，模型是否只关注跨尺度位置而不关注尺度内部？验证掩码实现。

4. **VAR 与 DiT 的缩放对比。** 在相同的 ImageNet 类别条件任务上，在相同参数量预算（如 33M、130M、458M）下训练 VAR 和 DiT。绘制 FID 与计算量的关系。VAR 在每个尺寸上都应领先 DiT——在小规模下复现论文结果。

5. **文本条件。** 将 VAR 扩展为通过 adaLN 接收文本嵌入（CLIP 池化输出）作为额外条件输入。这就是 HART 的做法。在文本对齐采样中，FID 提升了多少？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| VAR | “视觉自回归（Visual AutoRegressive）” | 在 VQ 词元网格金字塔上通过下一尺度预测进行图像生成 |
| 下一尺度预测 | “先粗后细” | 模型以所有先前尺度为条件，预测分辨率逐步提升的尺度的词元 |
| 多尺度 VQ 分词器 | “残差 VQ（Residual VQ）” | 生成 K 个分辨率递增的词元网格的 VQ-VAE，解码器将所有尺度相加 |
| 尺度 k | “金字塔第 k 层” | K 个分辨率层级之一，从 k=1 的 1×1 到 k=K 的 (H/p)×(W/p) |
| 尺度内并行 | “每尺度一次前向传播” | 尺度 k 的所有词元在一次 Transformer 前向传播中预测，而不是自回归地 |
| 跨尺度因果 | “按尺度排序的注意力” | 尺度 k 的词元可以关注尺度 1..k，但不能关注 k+1..K |
| 残差 VQ | “叠加式词元化” | 每个尺度的词元编码低尺度留下的残差；解码器将所有尺度嵌入相加 |
| VAR 缩放定律 | “图像版 GPT 缩放” | FID 随计算量遵循可预测的幂律，类似语言模型的困惑度 |
| HART | “VAR + 文本的混合” | 文本条件（text-conditional）VAR 变体，将 MaskGIT 风格的迭代解码与 VAR 的尺度结构相结合 |
| 尺度位置嵌入 | “（尺度，行，列）三元组” | 位置编码同时携带尺度索引和尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，权威参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散模型对比基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度分词器所扩展的分词器家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像词元化的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR
