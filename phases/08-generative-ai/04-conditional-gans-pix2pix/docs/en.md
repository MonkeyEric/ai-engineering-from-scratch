# 条件生成对抗网络（Conditional GANs）与 Pix2Pix

> 2014–2017 年间生成对抗网络的第一项重大突破是：控制模型生成什么。可以附加一个标签、一张图片，或一句话。Pix2Pix 做的是“图像到图像”版本，在狭义的图像转换任务上，它至今仍优于所有通用的文本到图像模型。

**类型：** 构建
**语言：** Python
**先修知识：** 第 8 阶段 · 03（GANs）、第 4 阶段 · 06（U-Net）、第 3 阶段 · 07（CNNs）
**时长：** 约 75 分钟

## 问题背景

无条件 GAN 会随机采样各种人脸——适合做演示，但无法落地。你想要的是：*把草图转成照片*、*把地图转成航拍图*、*把白天场景变成夜晚*、*给灰度图像上色*。在这些任务中，输入都是一张图像 `x`，必须输出与其语义对应的 `y`。同一个 `x` 可以有很多合理的 `y`。均方误差（mean-squared error）会把这些可能解压成模糊的“平均像”；而对抗损失不会，因为“看起来真实”这一标准天然是锐利的。

条件生成对抗网络（conditional GAN，Mirza & Osindero，2014）把条件 `c` 同时输入给生成器 `G` 和判别器 `D`。Pix2Pix（Isola et al.，2017）进一步专门化了这一框架：条件是一张完整输入图像，生成器采用 U-Net，判别器是基于*图像块*的分类器（PatchGAN），损失函数是对抗损失加 L1 损失。这套组合即使在 2026 年，仍能在狭义的图像到图像领域击败从零训练的文本到图像模型，因为它使用*成对数据（paired data）*——你拥有恰好需要的监督信号。

## 核心概念

![Pix2Pix：U-Net 生成器与 PatchGAN 判别器](../assets/pix2pix.svg)

**条件生成器（Conditional G）。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（不额外输入噪声——Isola 发现显式噪声会被忽略）。

**条件判别器（Conditional D）。** `D(x, y) → [0, 1]`。输入是*（条件，输出）*成对图像。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不只是 `y` 本身是否真实。

**U-Net 生成器。** 编码器-解码器结构，瓶颈层两侧有跳跃连接（skip connection）。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节会消失。

**PatchGAN 判别器。** 不再输出单一的真/假分数，而是输出一个 `N×N` 网格，每个格子判断一个约 70×70 像素的感受野区域，最后取平均。这等价于马尔可夫随机场假设：真实感是局部的。训练更快、参数量更少、输出更锐利。

**损失函数。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练，并把生成器推向已知目标。相比 L2，L1 产生更锐利的边缘（因为它接近中位数，而非均值）。Pix2Pix 默认取 `λ = 100`。

## CycleGAN——没有成对数据时怎么办

Pix2Pix 需要成对的 `(x, y)` 数据。CycleGAN（Zhu et al.，2017）放宽了这一要求，代价是增加一个*循环一致性（cycle consistency）*损失。它使用两个生成器 `G: X → Y` 和 `F: Y → X`，训练目标是让 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这样无需成对样本，也能实现马↔斑马、夏天↔冬天等转换。

到了 2026 年，无配对图像到图像任务大多由扩散模型完成（如 ControlNet、IP-Adapter），而非 CycleGAN；但循环一致性的思想仍几乎出现在每一篇无配对域适应论文中。

## 动手实现

`code/main.py` 实现了一个一维数据上的小型条件 GAN。条件 `c` 是类别标签（0 或 1）。任务：为给定类别生成一个来自条件分布的样本。

### 步骤 1：把条件同时拼接到 G 和 D 的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码（one-hot encoding）是最简单的方式。更大的模型会使用可学习的嵌入（embedding）、FiLM 调制，或交叉注意力（cross-attention）。

### 步骤 2：训练条件 GAN

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件下*的真实分布，而不是边缘分布。

### 步骤 3：验证每类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略。** 生成器学到忽略条件，判别器也不会惩罚，因为条件信号太弱。修复方法：更早、更激进地把条件输入 D（例如第一层，而不是最后一层），或使用投影判别器（projection discriminator，Miyato & Koyama 2018）。
- **L1 权重过低。** 生成器会漂向任意看起来像真的输出，而不忠于输入。Pix2Pix 风格任务建议从 `λ≈100` 开始。
- **L1 权重过高。** 因为 L1 仍是 L_p 范数，生成器会输出模糊结果。训练稳定后可逐渐退火降低。
- **判别器中的真值泄漏。** 要把 `(x, y)` 拼接作为 D 的输入，而不是只输入 `y`。否则 D 无法检查一致性。
- **每类各自模式坍塌。** 每个类别都可能独立坍塌。要做基于类别的多样性检查。

## 如何使用

2026 年图像到图像任务的最佳实践：

| 任务 | 最佳方案 |
|------|---------|
| 草图 → 照片，同域，有成对数据 | Pix2Pix / Pix2PixHD（仍然快、仍然锐利） |
| 草图 → 照片，无成对数据 | 带 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2，或 SD + ControlNet-Seg |
| 风格迁移 | 使用 IP-Adapter 或 LoRA 的扩散模型；GAN 方法已成为 legacy |
| 深度 → 照片 | 基于 Stable Diffusion 的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus，或 SD-Upscale（扩散） |
| 上色 | ColTran、基于扩散的上色器，或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方案 |

Pix2Pix 仍是正确选择，当且仅当：(a) 你有数千组成对样本；(b) 任务狭窄且可重复；(c) 需要快速推理。在通用开放域任务上，扩散模型更胜一筹。

## 交付成果

保存 `outputs/skill-img2img-chooser.md`。该技能接收任务描述、数据可用性（成对/非成对、样本数 N）以及延迟/质量预算，输出：方案选择（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据需求、推理成本、评估协议（LPIPS、FID、任务专用指标）。

## 练习题

1. **简单。** 修改 `code/main.py`，加入第三个类别。确认 G 仍能把每个类别的噪声映射到正确模式。
2. **中等。** 在一维设置中用感知风格损失替换 L1（例如用一个固定的小型 D 作为特征提取器）。它是否会改变条件分布的锐利程度？
3. **困难。** 在一维设置中草拟一个 CycleGAN：两个分布、两个生成器、循环损失。证明它无需成对数据就能学会双向映射。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 条件 GAN（Conditional GAN） | “带标签的 GAN” | G(z, c)、D(x, c)，两个网络都看到条件。 |
| Pix2Pix | “图像到图像的 GAN” | 成对条件 GAN：U-Net 生成器 + PatchGAN 判别器 + L1 损失。 |
| U-Net | “带跳跃连接的编码器-解码器” | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | “局部真实感分类器” | 判别器输出每块分数，而非单一全局分数。 |
| CycleGAN | “无配对图像转换” | 两个生成器 + 循环一致性损失；无需成对数据。 |
| SPADE | “GauGAN” | 用语义图对中间激活做归一化；用于分割图到图像。 |
| FiLM | “特征级线性调制（Feature-wise linear modulation）” | 来自条件的逐特征仿射变换；低成本条件注入。 |

## 生产提示：Pix2Pix 作为延迟受限场景下的基线

当你有成对数据且任务狭窄（草图 → 渲染、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的单步推理在延迟上比扩散模型低一个数量级。生产环境中常见的对比：

| 路径 | 步数 | 在单张 L4 上 512² 的典型延迟 |
|------|------|-----------------------------|
| Pix2Pix（U-Net 前向） | 1 | 约 30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | 约 1.2 s |
| SDXL-Turbo Img2Img | 1-4 | 约 0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | 约 3-5 s |

Pix2Pix 在静态批次（每个请求 FLOPs 相同）中吞吐量占优。扩散模型在质量和泛化性上占优。现代常见做法是：为狭窄任务部署一个 Pix2Pix 风格的蒸馏模型作为主力，并用扩散模型处理尾部输入。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) —— 条件 GAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) —— Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) —— CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) —— Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) —— SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) —— 投影判别器。
