# GAN——生成器与判别器

> Goodfellow 在 2014 年的窍门是彻底绕开密度估计。两个网络。一个负责造假，一个负责打假。它们相互对抗，直到假样本与真实样本无法区分。按理说这不该奏效，而且很多时候确实不奏效。但一旦成功，其生成的样本在特定窄域内仍然是文献中最锐利的。

**类型：** 实践构建
**语言：** Python
**前置知识：** Phase 3 · 02（反向传播）、Phase 3 · 08（优化器）、Phase 8 · 02（VAE）
**时间：** 约 75 分钟

## 问题

变分自编码器（VAE）生成的样本模糊，因为其均方误差（MSE）解码器损失对*均值*图像是贝叶斯最优的——而多个合理数字的均值是一个模糊的数字。你需要的是一种奖励*合理性*而非像素级接近某个单一目标的损失。合理性没有闭式表达，只能学习它。

Goodfellow 的思路：训练一个分类器 `D(x)` 来区分真实图像与伪造图像；训练生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号就是 `D` 当前认为“看起来像真实”的东西。随着 `G` 的提升，这个信号也会变化，追逐一个移动的目标。如果两个网络都收敛，`G` 就学会了数据分布，却根本不需要写下 `log p(x)`。

这就是对抗训练。其数学形式是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

到 2026 年，GAN 已不再是生成模型的 SOTA（扩散模型和流匹配夺走了这顶王冠）。但 StyleGAN 2/3 仍然是已发布的最锐利的人脸模型，GAN 判别器被用作扩散训练中的*感知损失（perceptual loss）*，而对抗训练支撑着快速一步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让实时扩散真正可落地。

## 概念

![GAN 训练：生成器与判别器的极小极大博弈](../assets/gan.svg)

**生成器（Generator）`G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射为样本 `x̂`。一个解码器形状的网络（全连接或转置卷积）。

**判别器（Discriminator）`D(x)`。** 将样本映射为一个标量概率（或分数）。真实样本 → 1，伪造样本 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。对 real=1、fake=0 做二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和（non-saturating）*形式（原始的 `log(1 - D(G(z)))` 在 `D` 很自信时会饱和并掐灭梯度）。

**训练循环。** 一步 `D`，一步 `G`。重复。

**为什么有效。** 如果 `G` 完美匹配 `p_data`，`D` 只能随机猜测，到处输出 0.5；`G` 不再有梯度。达到均衡。

**为什么失效。** 模式坍塌（mode collapse，`G` 找到一个 `D` 分不清的模式并永远重复它）、梯度消失（`D` 学得太快导致 `log D` 饱和）、训练不稳定（学习率、批次大小，什么都可能出问题）。

## 让 GAN 真正可用的变体

| 年份 | 创新 | 解决的问题 |
|------|------------|-----|
| 2015 | DCGAN | 卷积/反卷积、批归一化、LeakyReLU——第一个稳定架构。 |
| 2017 | WGAN、WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替代 BCE。修复梯度消失。 |
| 2017 | 谱归一化（Spectral normalization） | 对判别器做 Lipschitz 约束。2026 年的判别器仍在使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再逐层添加。首个百万像素成果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化（AdaIN）。固定域照片级真实感的最佳模型。 |
| 2021 | StyleGAN3 | 无混叠、平移等变——2026 年仍是人脸生成的黄金标准。 |
| 2022 | StyleGAN-XL | 有条件、类别感知、更大规模。 |
| 2024 | R3GAN | 以更强的正则化重新定位；无需技巧即可在 1024² 上训练。 |

## 动手实现

`code/main.py` 在一个 1 维数据上训练一个微型 GAN：两个高斯分布的混合。生成器和判别器都是单隐藏层 MLP。我们手工实现前向、反向传播和极小极大循环。目标是亲眼看到两种关键失效模式（模式坍塌 + 梯度消失）是如何发生的。

### 第一步：非饱和损失

原始 Goodfellow 损失 `log(1 - D(G(z)))` 在 `D` 以高置信度把 `G` 的伪造样本判为假时会趋于 0。此时 `G` 的梯度基本为零——`G` 无法改进。非饱和形式 `-log D(G(z))` 具有相反的渐近行为：当 `D` 很自信时它会放大，从而给 `G` 强烈的信号。

```python
def g_loss(d_fake):
    # 最大化 log D(G(z))  <=>  最小化 -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 第二步：每个生成器步对应一个判别器步

```python
for step in range(steps):
    # 训练 D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # 训练 G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 新的伪造样本
    update_G(fake_batch)
```

`G` 使用新的伪造样本，否则梯度是过时的。

### 第三步：观察模式坍塌

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实模式中的一个不再被生成。判别器也不再纠正，因为它从未被当作伪造样本见过。

## 常见陷阱

- **判别器太强。** 把 `D` 的学习率降低 2–5 倍，或在输入上加实例/层噪声。如果 `D` 准确率超过 95%，`G` 就死了。
- **生成器记住了一个模式。** 给 `D` 的输入加噪声，使用 minibatch-discrimination 层，或切换到 WGAN-GP。
- **批归一化泄露统计量。** 真实批次与伪造批次经过同一 BN 层会混合统计量。改用实例归一化（instance norm）或谱归一化。
- **Inception Score 被操纵。** FID 和 IS 在低样本量时很吵。评估时至少使用 1 万个样本。
- **条件任务的一次性采样是谎言。** 你仍然需要 CFG scale、截断技巧和重采样才能得到可用输出。

## 如何使用

2026 年的 GAN 技术栈：

| 场景 | 选择 |
|-----------|------|
| 照片级真实人脸，固定姿态 | StyleGAN3（最锐利、最小） |
| 动漫 / 风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN（Phase 8 · 04）或 ControlNet（Phase 8 · 08） |
| 快速一步文本到图像 | 扩散模型的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 作为扩散训练器内部的感知损失 | 图像块上的小 GAN 判别器 |
| 任何多模态、开放域任务 | 不要选 GAN——用扩散或流匹配 |

GAN 锐利但狭窄。一旦领域打开——照片、任意文本提示、视频——就切换到扩散。对抗技巧作为组件继续存在（感知损失、蒸馏），但不再作为独立生成器。

## 交付

保存 `outputs/skill-gan-debugger.md`。该技能接收一次失败的 GAN 运行（损失曲线、样本网格、数据集大小），输出可能原因的有序列表、一行修复建议以及重新运行协议。

## 练习

1. **简单。** 用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 再运行。`G` 的损失多快会坍塌为常数？
2. **中等。** 把 Goodfellow 的 BCE 损失替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并把 `D` 的权重裁剪到 `[-0.01, 0.01]`。训练是否更稳定？比较实际墙钟收敛时间。
3. **困难。** 把 1 维示例扩展到 2 维数据（环上的 8 个高斯混合）。跟踪生成器在 1k、5k、10k 步时捕获了多少个模式。实现 minibatch discrimination 并重新测量。

## 关键术语

| 术语 | 俗称 | 实际含义 |
|------|-----------------|-----------------------|
| 生成器（Generator） | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器（Discriminator） | "D" | 分类器 `D: x → [0, 1]`，判断真实 vs 伪造。 |
| 极小极大（Minimax） | "博弈" | 联合目标下的 `min_G max_D`。 |
| 非饱和损失（Non-saturating loss） | "修复" | 对 `G` 使用 `-log D(G(z))` 而非 `log(1 - D(G(z)))`。 |
| 模式坍塌（Mode collapse） | "G 只记住了一样东西" | 生成器在数据丰富的情况下只产生少数几种输出。 |
| WGAN | "Wasserstein" | 用推土机距离（Earth-Mover）+ 梯度惩罚替代 BCE；梯度更平滑。 |
| 谱归一化（Spectral norm） | "Lipschitz 技巧" | 约束 `D` 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "那个能用的" | 映射网络 + AdaIN；人脸领域的最佳模型，2026 年仍是。 |

## 生产提示：一次性推理是 GAN 的持久优势

GAN 在开放域生成上已不再赢得样本质量，但仍赢得推理成本。在生产推理文献的术语中，GAN 具有以下特点：

- **没有预填充（prefill），没有解码阶段。** 单次 `G(z)` 前向传播。TTFT 约等于总延迟。
- **没有 KV 缓存压力。** 唯一的状态就是权重。批次大小受激活内存限制，而非缓存。
- **连续批处理极其简单。** 因为每个请求消耗相同的固定 FLOPs，服务器目标占用率下的静态批次通常就是最优的。不需要在途调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）在 2026 年成为快速文本到图像的主流技术：它把 20–50 步的扩散流程压缩成 1–4 次 GAN 式前向传播，同时保留扩散基础模型的分布。对抗损失作为一种训练时的调节手段存活下来，用于把慢速生成器变成快速生成器。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) —— 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) —— WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) —— SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) —— StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) —— StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) —— SDXL-Turbo。
