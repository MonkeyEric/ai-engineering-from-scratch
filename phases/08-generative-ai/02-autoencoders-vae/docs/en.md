# 自编码器与变分自编码器（Autoencoders & Variational Autoencoders, VAE）

> 普通自编码器先压缩、再重构。它只会记忆，不会生成。加入一个小技巧——强制隐变量服从高斯分布——你就得到了一个采样器。这个名为重参数化（reparameterization）的技巧，即 `z = μ + σ·ε`，正是 2026 年你使用的所有潜空间扩散（latent-diffusion）和流匹配（flow-matching）图像模型都以 VAE 作为输入层的原因。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 3 · 02（反向传播），Phase 3 · 07（卷积神经网络），Phase 8 · 01（生成式 AI 分类）
**时长：** 约 75 分钟

## 问题

把一张 784 像素的 MNIST 数字压缩成 16 维的编码，再重构出来。普通自编码器能把重构均方误差（MSE）压得很低，但隐空间（code space）却凹凸不平、杂乱无章。在隐空间里随机挑一个点解码，得到的只是噪声。它没有采样能力，本质上只是一个乔装成生成模型的压缩模型。

你真正想要的是：（a）隐空间是一个干净、平滑、可以采样的分布——比如各向同性高斯分布 `N(0, I)`；（b）从该分布中任意采样并解码，都能得到一个合理的数字；（c）编码器与解码器仍具备良好的压缩能力。三个目标，一种架构，一个损失函数。

Kingma 在 2013 年提出的 VAE 通过以下方式解决这一问题：训练编码器输出一个分布 `q(z|x) = N(μ(x), σ(x)²)`，利用 KL 散度惩罚项将该分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 中采样隐变量 `z`。推理时，丢弃编码器，直接采样 `z ~ N(0, I)` 并解码。正是这个 KL 惩罚项迫使隐空间变得结构化。

到了 2026 年，独立的 VAE 已经很少单独发布——在原始图像质量上，它们已被扩散模型超越——但它们是每一个潜空间扩散模型（SD 1/2/XL/3、Flux、AudioCraft）首选的编码器。学懂了 VAE，你就学懂了当今所有图像生成管线中那层看不见的“第一层”。

## 概念

![自编码器与 VAE 的对比：重参数化技巧](../assets/vae.svg)

**自编码器（Autoencoder）。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。隐空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。二者共同定义 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧（Reparameterization trick）。** 直接从 `q(z|x)` 采样是不可导的。将采样改写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上一个不含参数的噪声，梯度可以畅通地流过 `μ` 和 `σ`。

**损失函数。** 证据下界（Evidence Lower BOund, ELBO），由两项组成：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重构项（reconstruction）把 `x̂` 推向 `x`；KL 项把 `q(z|x)` 推向先验。二者此消彼长。较小的 β（<1）会得到更锐利的样本，但隐空间没那么像高斯；较大的 β（>1）会让隐空间更规整，但样本会更模糊。β-VAE（Higgins, 2017）让这颗“旋钮”名声大噪，并开启了解耦表示学习（disentanglement）的研究热潮。

**采样。** 推理时：从 `N(0, I)` 采样 `z`，前向通过解码器。只需一次前向传播——不像扩散模型那样需要迭代采样。

## 动手实现

`code/main.py` 实现了一个微型 VAE，完全不依赖 numpy 或 torch。输入是 8 维合成数据，来自一个 8 维空间中的 2 成分高斯混合分布。编码器和解码器都是单隐藏层多层感知机（MLP）。我们实现了 tanh 激活、前向传播、损失函数以及手写反向传播。这不是生产代码，而是教学代码。

### 步骤 1：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

这里输出 `log σ²` 而不是 `σ`，因此网络输出不受约束（用 σ 的 softplus 会在 σ ≈ 0 时让梯度消失，是个陷阱）。

### 步骤 2：重参数化与解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 步骤 3：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两边都是高斯分布，KL 有闭式解，不要数值积分。到了 2026 年，仍有人发布用蒙特卡洛估计 KL 的代码——它慢 3 倍，毫无必要。

### 步骤 4：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 常见陷阱

- **后验崩塌（Posterior collapse）。** KL 项过于激进地把 `q(z|x)` 推向 `N(0, I)`，导致 `z` 不再携带关于 `x` 的信息。解决方法：β 退火（β-annealing，β 从 0 逐渐升到 1）、free bits，或跳过不活跃维度的 KL 项。
- **样本模糊。** 高斯解码器似然对应 MSE 重构，而 L2 的贝叶斯最优解是均值——一堆合理数字的均值就是一个模糊数字。解决方法：使用离散解码器（VQ-VAE、NVAE），或者只把 VAE 当编码器用，在隐变量上叠加扩散模型（Stable Diffusion 就是这样做的）。
- **β 初始太大或升温太快。** 参见“后验崩塌”。建议从 β≈0.01 开始并逐渐提升。
- **隐变量维度太小。** MNIST 用 16 维即可，ImageNet 256² 常用 256 维，ImageNet 1024² 可用 2048 维。Stable Diffusion 的 VAE 把 512×512×3 压缩到 64×64×4（空间面积下采样 32 倍，通道数也压缩 32 倍）。

## 应用场景

2026 年的 VAE 技术栈：

| 场景 | 选择 |
|-----------|------|
| 作为扩散模型的图像隐空间编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频隐空间编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频隐空间 | Sora 的时空块（spatiotemporal patches）、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散隐变量（用于 Transformer 建模） | VQ-VAE、RVQ（ResidualVQ） |
| 用于生成的连续隐变量 | 普通 VAE，再在其隐空间上训练流/扩散模型 |

潜空间扩散模型（latent-diffusion model）就是“编码器与解码器之间夹着一个扩散模型”的 VAE。VAE 负责粗粒度压缩，扩散模型负责繁重的生成工作。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen Transformer）也遵循同一范式。

## 交付物

保存为 `outputs/skill-vae-trainer.md`。

该技能接受：数据集画像 + 目标隐变量维度 + 下游用途（重构、采样或潜空间扩散输入）；输出：架构选择（plain/β/VQ/RVQ）、β 调度方案、隐变量维度、解码器似然（高斯 vs 分类分布）以及评估计划（重构 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 将 `code/main.py` 中的 `β` 分别改为 `0.01`、`0.1`、`1.0`、`5.0`，记录最终重构 MSE 和 KL。对于你的合成数据，哪个 β 是帕累托最优的？
2. **中等。** 将高斯解码器似然替换为伯努利似然（交叉熵损失），在同一合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展为迷你 VQ-VAE：把连续的 `z` 替换为在 K=32 条目的码本（codebook）中寻找最近邻。比较重构 MSE，并报告有多少码本条目实际被使用（码本崩塌是真实存在的）。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 自编码器（Autoencoder） | 编码-解码网络 | `x → z → x̂`，学习 MSE。不是生成模型。 |
| VAE | 带采样器的自编码器 | 编码器输出分布，KL 惩罚项塑造隐空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`；当 `q = p(z|x)` 时紧致。 |
| 重参数化（Reparameterization） | `z = μ + σ·ε` | 将随机节点改写为确定性部分 + 纯噪声，使反向传播能穿过采样过程。 |
| 先验（Prior） | `p(z)` | 隐变量的目标分布，通常是 `N(0, I)`。 |
| 后验崩塌（Posterior collapse） | “KL 项赢了” | 编码器忽略 `x`，直接输出先验；解码器只能凭空捏造。 |
| β-VAE | 可调的 KL 权重 | `loss = recon + β·KL`。β 越高表示越解耦，但样本越模糊。 |
| VQ-VAE | 离散隐变量 | 用最近的码本向量替换连续 `z`，便于 Transformer 建模。 |

## 生产提示：VAE 是扩散服务中最热门的瓶颈路径

在 Stable Diffusion / Flux / SD3 的推理管线中，每个请求会调用 VAE 两次——一次编码（用于 img2img / 图像修复）和一次解码。在 1024² 分辨率下，解码过程往往是整个管线中激活内存占用最大的单一峰值，因为它需要把 `128×128×16` 的隐变量上采样回 `1024×1024×3`。这带来两个实际影响：

- **分片或分块解码。** `diffusers` 提供了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块解码以轻微接缝伪影为代价，把内存复杂度从 `O(H·W)` 降到 `O(tile²)`。对于 1024² 及以上的消费级 GPU，这是必不可少的。
- **解码用 bf16，最终 resize 保留 fp32 数值精度。** SD 1.x 的 VAE 以 fp32 发布，*在 1024²+ 直接转成 fp16 会静默产生 NaN*。SDXL 提供了 `madebyollin/sdxl-vae-fp16-fix`——务必优先使用该 fp16-fix 变体，或直接使用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) —— VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) —— 解耦 β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) —— VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) —— 当时的图像 VAE  state-of-the-art。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) —— Stable Diffusion；VAE 作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) —— Encodec，音频 VAE 的事实标准。
