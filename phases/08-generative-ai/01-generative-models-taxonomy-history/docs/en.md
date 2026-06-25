# 生成模型——分类与历史

> 每一款图像模型、文本模型、视频模型和 3D 模型，都可以归入五个类别之一。选错类别，你会和数学纠缠数周；选对类别，过去十二年的进展会清晰地在脑海中堆叠起来。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 2 阶段（机器学习基础）、第 3 阶段（深度学习核心）、第 7 阶段 · 第 14 课（Transformer）
**Time:** ~45 分钟

## 问题所在

生成模型（generative model）只做一件事：给定从某个未知分布 `p_data(x)` 中抽取的训练样本，输出看起来像来自同一分布的新样本。人脸、句子、MIDI 文件、蛋白质结构——如果你眯起眼睛看，都是同一个问题。

难点在于，`p_data` 存在于数百万维的空间中（一张 512×512 的 RGB 图像约有 78.6 万维），样本位于该空间中的一个薄流形（manifold）上，而你可能只有约 1000 万个样本。暴力求解密度毫无希望。每个生成模型都是一种妥协：用一个稍微简单一点的问题替换一个困难的问题。

过去十二年里有五个家族幸存下来。知道每个家族做出了哪种妥协，你就能明白它为什么在某些任务上胜出，而在另一些任务上崩溃。

## 核心概念

![五类生成模型——按建模对象分类](../assets/taxonomy.svg)

**1. 显式密度（explicit density），可精确计算。** 把 `log p(x)` 写成可以实际求值的形式。自回归模型（autoregressive model）（PixelCNN、WaveNet、GPT）将 `p(x) = ∏ p(x_i | x_<i)` 分解；标准化流（normalizing flow）（RealNVP、Glow）把 `p(x)` 构造成简单基础分布的可逆变换。优点：精确似然（likelihood）、干净的训练损失（loss）。缺点：自回归推理是顺序的（长序列慢），流需要可逆架构（架构上受限）。

**2. 显式密度，近似。** 从下界约束 `log p(x)`（ELBO），然后优化这个下界。变分自编码器（VAE, variational autoencoder）（Kingma 2013）使用带变分后验的编码器-解码器结构。扩散模型（diffusion model）（DDPM、Ho 2020）训练一个去噪器，隐式地优化加权 ELBO。到 2026 年，扩散模型是图像、视频和 3D 领域的主导骨干网络。

**3. 隐式密度（implicit density）。** 完全跳过密度；学习一个生成器 `G(z)` 来生成样本，再学习一个判别器 `D(x)` 来区分真假。生成对抗网络（GAN, generative adversarial network）（Goodfellow 2014）。推理速度快（一次前向传播），但训练 notoriously 不稳定。StyleGAN 1/2/3 即使在 2026 年，仍然是固定域照片级真实（photoreal）图像（人脸、卧室）的标杆。

**4. 基于分数 / 连续时间。** 直接学习对数密度的梯度 `∇_x log p(x)`（即分数，score）。Song & Ermon（2019）证明分数匹配（score matching）可以把扩散模型推广到随机微分方程（SDE）。流匹配（flow matching）（Lipman 2023）是 2024–2026 年的热门方向：无需模拟的训练、更直的路径、比 DDPM 快 4–10 倍的采样。Stable Diffusion 3、Flux、AudioCraft 2 都使用流匹配。

**5. 基于离散编码的词元自回归。** 用 VQ-VAE 或残差量化器把高维数据压缩成短的离散词元（token）序列，然后用 Transformer 建模该序列。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch 分词器都采用这种方式。这是第 1 类加上一个可学习的分词器（tokenizer）。

## 简史

| 年份 | 模型 | 为什么重要 |
|------|-------|-----------------|
| 2013 | VAE（Kingma） | 第一个拥有可用训练损失的深度生成模型。 |
| 2014 | GAN（Goodfellow） | 隐式密度，没有似然——却生成惊人锐利的样本。 |
| 2015 | DRAW、PixelCNN | 顺序图像生成。 |
| 2017 | Glow、RealNVP | 可逆流；精确似然伴随深度增长。 |
| 2017 | Progressive GAN | 首个百万像素人脸生成。 |
| 2019 | StyleGAN / StyleGAN2 | 照片级真实人脸，在该单一领域仍然难以超越。 |
| 2020 | DDPM（Ho） | 扩散模型变得实用。 |
| 2021 | CLIP、DALL-E 1、VQGAN | 文本到图像成为主流。 |
| 2022 | Imagen、Stable Diffusion 1、DALL-E 2 | 潜在扩散 + 文本条件 =  commodity。 |
| 2022 | ControlNet、LoRA | 对预训练扩散模型的精细控制。 |
| 2023 | SDXL、Midjourney v5、Flow matching | 规模 + 更好的训练动态。 |
| 2024 | Sora、Stable Diffusion 3、Flux.1 | 视频扩散；流匹配胜出。 |
| 2025 | Veo 2、Kling 1.5、Runway Gen-3、Nano Banana | 生产级视频。 |
| 2026 | Consistency + Rectified Flow | 从扩散骨干网络实现单步采样。 |

## 五个问题的快速分诊

当一篇新的生成模型论文发布时，在阅读方法部分之前，先回答这五个问题。

1. **建模对象是什么？** 像素、潜变量、离散词元、3D 高斯、网格、波形？
2. **密度是显式还是隐式？** 他们是否写下了 `log p(x)`？
3. **采样：一次完成还是迭代？** 迭代意味着推理更慢；一次完成通常意味着对抗或蒸馏。
4. **条件控制：无条件、类别、文本、图像、姿态？** 这决定了损失和架构脚手架。
5. **评估指标：FID、CLIP score、IS、人类偏好、任务准确率？** 每项都有已知的失效模式（见第 14 课）。

在本阶段每一课中，你都会重新回答这五个问题。到最后，它们会变成本能。

## 动手实现

本课的代码是一个轻量级可视化：用三种玩具方法（核密度估计、离散直方图、最近样本“类 GAN”生成器）从一维高斯混合分布的样本中拟合分布，让你在一个屏幕就能打印的问题上，直观看到显式密度与隐式密度的区别。

运行 `code/main.py`。它从双峰高斯混合中抽取 2000 个样本，然后打印：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两种方法可以回答“这个点有多大概率？”第三种不能。这就是*显式 vs 隐式*的区别，它将在未来每一课中都很重要。

## 应用指南

2026 年，哪种家族适合哪种任务？

| 任务 | 最佳家族 | 原因 |
|------|-------------|-----|
| 照片级真实人脸，窄域 | StyleGAN 2/3 | 仍然最锐利，推理最快。 |
| 通用文本到图像 | 潜在扩散 + 流匹配 | SD3、Flux.1、DALL-E 3。 |
| 快速文本到图像 | 整流流 + 蒸馏 | SDXL-Turbo、SD3-Turbo、LCM。 |
| 文本到视频 | 扩散 Transformer + 流匹配 | Sora、Veo 2、Kling。 |
| 语音 + 音乐 | 基于词元的自回归（AudioLM、VALL-E、MusicGen）或流匹配（AudioCraft 2） | 离散词元成本更低、更易扩展。 |
| 3D 场景 | 高斯泼溅拟合，扩散先验 | 3D-GS 用于重建，扩散用于新视角合成。 |
| 密度估计（无需采样） | 流（flows） | 唯一拥有精确 `log p(x)` 的家族。 |
| 仿真 / 物理 | 流匹配、分数 SDE | 直线路径、平滑向量场。 |

## 交付物

保存为 `outputs/skill-model-chooser.md`。

该技能接收任务描述，并输出：(1) 应使用哪个家族，(2) 三个开源和三个托管选项的排序列表，(3) 需要注意的潜在失效模式，(4) 计算/时间预算。

## 练习

1. **简单。** 对于以下五款产品，分别识别其家族和骨干网络：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自公开技术报告。
2. **中等。** 明天你要读的那篇论文声称比扩散模型快 100 倍。写下三个问题，用来检验这种加速在条件控制和高分辨率下是否仍然成立。
3. **困难。** 选取一个你关心的领域（例如蛋白质结构、CAD、分子、轨迹）。为该领域当前的 SOTA 模型回答上述五个分诊问题，并勾勒出一个更好的模型会改变什么。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 生成模型（generative model） | “它能生成新东西” | 学习对 `p_data(x)` 的采样器，可选择性地暴露 `log p(x)`。 |
| 显式密度（explicit density） | “你可以计算它” | 模型提供闭式或可计算的 `log p(x)`。 |
| 隐式密度（implicit density） | “GAN 风格” | 只有采样器——无法评估给定点的 `p(x)`。 |
| ELBO | “证据下界” | `log p(x)` 的可计算下界；VAE 和扩散模型优化它。 |
| 分数（score） | “对数密度的梯度” | `∇_x log p(x)`；扩散模型和 SDE 模型学习这个场。 |
| 流形假设（manifold hypothesis） | “数据生活在某个曲面上” | 高维数据集中在低维流形上；这就是降维有效的原因。 |
| 自回归（autoregressive） | “预测下一块” | 将联合分布分解为条件分布的乘积。 |
| 潜变量（latent） | “压缩编码” | 低维表示，解码器可以从中重建输入。 |

## 生产须知：五个家族，五种推理形态

每个家族对应不同的推理服务成本曲线。生产推理文献把大语言模型推理框定为 prefill + decode；同样的分解也适用于这里：

- **自回归（第 1 和第 5 类）。** 顺序解码主导延迟；KV 缓存（KV-cache）、连续批处理、投机解码都直接适用。
- **VAE / 扩散 / 流匹配（第 2 和第 4 类）。** 没有大语言模型意义上的 decode。成本 = `num_steps × step_cost`，而 `step_cost` 是在完整潜在分辨率下运行一次 Transformer 或 U-Net 前向传播。生产调参点是步数（DDIM / DPM-Solver / 蒸馏）、批大小和精度（bf16 / fp8 / int4）。
- **GAN（第 3 类）。** 一次前向传播。没有调度，没有 KV 缓存。TTFT ≈ 总延迟。这就是 StyleGAN 在窄域 UX 中仍然占优的原因。

当你在论文摘要里看到“比扩散模型更快”时，请把它翻译成“更少的步数 × 相同的单步成本”或“相同的步数 × 更便宜的单步成本”。其余都是营销。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— GAN 论文。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) —— VAE 论文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) —— DDPM 论文。
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) —— 扩散作为 SDE。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) —— 流匹配论文。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— Stable Diffusion 3。
