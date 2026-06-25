# Transfusion：一个 Transformer 中的自回归文本 + 扩散图像

> Chameleon 和 Emu3 把所有赌注都压在离散token上。它们能跑通，但量化瓶颈显而易见——图像质量在连续空间扩散模型之下触顶。Transfusion（Meta，Zhou 等，2024 年 8 月）则反向下注：保持图像连续，彻底去掉 VQ-VAE，并用两个损失训练一个 transformer。文本token用下一token预测（next-token prediction，NTP）；图像patch用流匹配（flow matching）/ 扩散损失。两个目标函数同时优化同一套权重。Stable Diffusion 3 背后的架构（MMDiT）是它的近亲。本节课将解读 Transfusion 的核心思想，构建一个 toy 双损失训练器，并画出让同一个 transformer 同时完成两项任务的注意力掩码（attention mask）。

**Type:** Build
**Languages:** Python（标准库，MNIST 规模的 toy 双损失训练器）
**Prerequisites:** 第 12 阶段 · 11（Chameleon）、第 8 阶段（生成式 AI）
**Time:** 约 180 分钟

## Learning Objectives

- 搭建一个主干网络同时跑两个损失（文本token做 NTP，图像patch做扩散 MSE）。
- 解释为什么图像patch之间用双向注意力、文本token之间用因果注意力是正确的掩码选择。
- 从计算量、图像质量和代码复杂度三方面，比较 Transfusion 风格（连续图像、扩散损失）与 Chameleon 风格（离散图像、NTP）。
- 说明 MMDiT 的贡献：每个 block 内模态专属权重，并在残差流（residual stream）上做联合注意力。

## The Problem

离散图像token与连续图像表示之争比 LLM 还要古老。连续表示（原始像素、VAE 隐空间）保留细节；离散token（VQ 索引）适配 transformer 的原生词表，却在量化步骤丢失细节。

Chameleon / Emu3 走了离散路线：单一损失、单一架构，但图像保真度受限于 tokenizer 质量。

扩散模型（diffusion models）走了连续路线：图像质量极佳，但模型与 LLM 分离，噪声调度工程复杂，且无法与文本生成 cleanly 集成。

Transfusion 的问题是：能不能两者兼得？保持图像连续，仍然只训练一个模型，把两个损失缝合进同一个梯度步（gradient step）。

## The Concept

### 双损失架构

一个纯解码器 transformer 处理包含以下内容的序列：

- 文本token（离散的，来自 BPE 词表）。
- 图像patch（连续的，16×16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入层相同）。
- `<image>` 和 `</image>` 标签，标记连续patch所在位置。

前向传播只跑一遍。损失按token二选一：

- 文本token：标准交叉熵（cross-entropy），接词表 logits 头。
- 图像patch：连续patch上的扩散损失——预测每个patch上添加的噪声。

梯度流（gradient flow）经过共享的 transformer 主体。两个损失同时改进共享权重。

### 注意力掩码：因果文本 + 双向图像

文本token必须是因果的——不能让某个文本token看到未来文本，否则教师强制（teacher forcing）会失效。图像patch则代表同一个快照；同一张图像块内部的patch应互相双向 attending。

掩码规则：

```
M[i, j] = 1 如果满足：
  (i 是文本且 j 是文本且 j <= i)   # 文本之间因果
  OR (i 是图像且 j 是图像且 same_image_block(i, j))   # 同一张图像内双向
  OR (i 是文本且 j 是图像且 j < i_image_end)   # 文本关注之前的图像
  OR (i 是图像且 j 是文本且 j < i_image_start)   # 图像关注前面的文本
```

训练与推理时都实现为块三角掩码（block-triangular mask）。

### Transformer 内部的扩散损失

扩散损失是标准做法：给图像patch加噪声，让模型预测噪声（或等价地预测干净patch）。Transfusion 的版本使用流匹配——预测从带噪样本到干净样本的速度场（velocity field）。

训练时：
1. 对每个图像patch x0，随机采样时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列中的文本 NTP 损失一起反向传播。

推理时，生成过程如下：
- 文本token：标准自回归采样。
- 图像patch：以先前文本token为条件的扩散采样循环（通常 10–30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等，2024 年 3 月）与 Transfusion 同期发布了 MMDiT（Multimodal Diffusion Transformer，多模态扩散 Transformer）。二者架构上是兄弟关系。

MMDiT 的关键差异：

- 每个 block 有模态专属权重。每个 transformer block 对文本token和图像patch分别使用独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其余部分按模态分离。
- 整流流（rectified flow）训练。流匹配的一种具体变体，采样方式更简单，数学上比 DDPM 更简洁。
- 规模。MMDiT 是 SD3 的主干（20 亿和 80 亿参数两个版本）。Transfusion 论文则扩展到 70 亿参数。

两者殊途同归：同一个 transformer 对文本跑 NTP，对连续图像表示跑扩散。

### 为什么它能击败 Chameleon 风格

连续扩散与离散 NTP 在图像生成上的质量差距是可量化的。Transfusion 论文报告：

- 在 70 亿参数下，同等规模的 Chameleon 风格模型 FID 低 3–5 分。
- 无需训练 tokenizer——图像编码器更简单（线性投影到隐藏维度，与 ViT 输入层相同）。
- 图像patch去噪可以并行化，而自回归图像token不行。

缺点：Transfusion 是双损失模型，训练动态更棘手。损失权重需要调参；NTP 与扩散之间的时间表不匹配可能导致某个头主导训练。

### 下游还有什么

Janus-Pro（第 12.15 课）进一步发展 Transfusion 思想，将理解用的视觉编码器与生成用的编码器解耦——理解用 SigLIP，生成用 VQ——同时共享 transformer 主体。Show-o（第 12.14 课）则把扩散换成离散扩散（掩码预测）。统一生成家族在 Transfusion 之后快速分支。

2026 年能输出图像的生产级 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定使用了这一家族的某种后代。具体细节属于商业机密。

## Use It

`code/main.py` 在一个 MNIST 级别的 toy 问题上构建了一个 toy Transfusion：

- 文本caption是描述数字（0–9）的短整数序列。
- 图像是 4×4 的字节网格。
- 一对共享权重的线性投影作为 transformer 的简化替代品；文本上跑 NTP 损失，带噪patch上跑 MSE 损失。
- 训练循环交替两个损失，注意力掩码显式构造。
- 生成时在一个前向过程中同时产出文本caption和 4×4 图像。

这个 transformer 是 toy。真正的收获是双损失的管线、注意力掩码构造和推理循环。

## Ship It

本节课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本+图像、文本+音频、文本+视频），它会设计双损失调度（损失权重、掩码形状、共享 vs 模态专属 block）并标注实现风险。

## Exercises

1. 一个 Transfusion 风格模型训练时 70% 是文本token、30% 是图像patch。图像扩散损失的幅度约为文本 NTP 损失的 10 倍。应设置什么损失权重才能平衡二者？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块三角掩码。把每个位置标成 0 或 1。

3. MMDiT 使用模态专属的 QKV 权重。与 Transfusion 的完全共享 transformer 相比，这增加了多少参数量开销？在 70 亿参数规模下，是否值得？

4. 生成：给定一个文本提示，模型先跑 NTP 生成 50 个token，然后遇到 `<image>`，再在 20 个去噪步内对 256 个patch做扩散。总共需要多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述什么是整流流（rectified flow），以及为什么它比 DDPM 在更少的推理步内收敛。

## Key Terms

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 双损失训练（two-loss training） | "NTP + diffusion" | 同一个 transformer 在同一个梯度步中同时优化文本token的交叉熵与连续图像patch的 MSE |
| 流匹配（flow matching） | "Rectified flow" | 一种扩散变体，预测从噪声到干净数据的速度场；数学上比 DDPM 更简洁 |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3 的架构：联合注意力，模态专属的 MLP 与归一化层 |
| 块三角掩码（block-triangular mask） | "Causal text + bidirectional image" | 跨文本因果、在图像区域内双向的注意力掩码 |
| 连续图像表示（continuous image representation） | "No VQ" | 图像patch是实值向量，而非整数码本索引 |
| 速度预测（velocity prediction） | "v-parameterization" | 网络输出的是噪声与数据之间的速度场，而不是噪声本身 |

## Further Reading

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
