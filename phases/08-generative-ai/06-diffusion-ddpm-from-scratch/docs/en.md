# 扩散模型 —— 从零实现 DDPM

> Ho、Jain、Abbeel（2020）为领域提供了一套“用过就回不去”的配方：用上千个小步骤把数据逐步摧毁成噪声，训练一个神经网络预测噪声，再在推理时把过程倒转。如今主流的图像、视频、3D 和音乐模型都运行在这个循环之上，可能还会在其基础上叠加流匹配（flow matching）或一致性（consistency）等技巧。

**类型：** 构建  
**语言：** Python  
**前置知识：** Phase 3 · 02（反向传播），Phase 8 · 02（VAE）  
**时间：** 约 75 分钟

## 问题

你希望构建一个能从 `p_data(x)` 采样的采样器（sampler）。GAN 常常陷入极小极大博弈而发散；VAE 从高斯解码器生成模糊样本。你真正想要的是一个满足以下三点的训练目标：（a）单一且稳定的损失（loss）（没有鞍点、没有极小极大）；（b）`log p(x)` 的下界（因而有可解释的似然）；（c）样本质量达到 SOTA。

Sohl-Dickstein 等人（2015）在理论上给出答案：定义一条逐步加入高斯（Gaussian）噪声的马尔可夫链（Markov chain）`q(x_t | x_{t-1})`，并训练一条反向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho、Jain、Abbeel（2020）则把损失简化为一句话——预测噪声——并整理清楚了数学。2020 年它只是一个新奇事物；2021 年它产出了 SOTA 样本；2022 年它演变成了 Stable Diffusion；到 2026 年，它已成为基础架构。

## 概念

![DDPM：前向加噪，反向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个小步骤中逐步加入高斯噪声。数学上可解的关键在于：累计步也是高斯分布：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，对应一条 `β_t` 调度。让 `β_t` 在 T=1000 步内从 1e-4 线性增长到 0.02，`x_T` 就近似 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)`，预测被加入的噪声。给定 `x_t`，按如下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 可以是 `sqrt(β_t)`，也可以是学习到的方差。这个式子看起来很复杂，但只是代数运算——用后验 `q(x_{t-1} | x_t, x_0)` 解出 `x_{t-1}`，再用基于噪声预测估计出的 `x_0` 替换进去。

**训练损失（loss）。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选 `t`，采样 `ε ~ N(0, I)`，通过闭式一次性算出带噪的 `x_t`，然后对噪声做回归。一个损失函数，没有极小极大、没有 KL、没有重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始，从 `t = T` 迭代到 `1` 执行反向步骤。完成。

## 为什么有效

三点直觉：

1. **去噪容易，生成难。** 在 `t=T` 时，数据已是纯噪声——网络要解决的问题很平凡。在 `t=0` 时，网络只需清理少量像素。在中间 `t`，问题虽难，但同一套权重能从所有噪声层级获得大量梯度。

2. **伪装成噪声预测的分数匹配（score matching）。** Vincent（2011）证明，预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*分数（score）*。反向 SDE 利用这个分数沿密度梯度向上走——一场朝着高概率区域的有引导的随机游走。

3. **ELBO 退化为简单的 MSE。** 完整变分下界（ELBO）在每个时间步都有一个 KL 项。在 DDPM 的参数化下，这些 KL 项会简化为对噪声预测的 MSE，并带有一些系数；Ho 把这些系数去掉（称为“simple”损失），结果质量反而*提升*了。

## 动手实现

`code/main.py` 实现了一个一维 DDPM。数据是一个双峰混合分布。“网络”是一个小型 MLP，输入 `(x_t, t)`，输出预测的噪声。训练就是那一行损失函数。采样则迭代反向链。

### 步骤 1：前向调度（闭式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 步骤 2：一次性采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 步骤 3：一次训练步骤

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 步骤 4：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于 1 维问题、40 个时间步、24 个单元的 MLP，这个实现大约 200 个 epoch 就能学会双峰混合分布。

## 时间条件

网络需要知道当前正在对哪个时间步去噪。两种常见选择：

- **正弦嵌入（sinusoidal embedding）。** 类似 Transformer 的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。先过一个 MLP，再广播到网络中。
- **FiLM / group-norm 条件。** 把嵌入投影为每个通道的缩放/偏置（FiLM），加在每个块中。

我们的玩具代码使用正弦嵌入 → 拼接。生产级 U-Net 使用 FiLM。

## 注意事项

- **调度（schedule）很重要。** 线性 `β` 是 DDPM 默认，但余弦调度（Nichol & Dhariwal, 2021）在相同计算量下能得到更好的 FID。如果质量遇到瓶颈，就换调度。
- **时间步嵌入很脆弱。** 直接把原始 `t` 作为浮点数传入在 1 维玩具问题上可以，但对图像会失败；务必使用合适的嵌入。
- **V-prediction vs ε-prediction。** 在极端区域（`t` 非常小或非常大）时，`ε` 的信噪比很差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 都使用它。
- **无分类器引导（classifier-free guidance）。** 推理时同时计算有条件 `ε` 和无条件 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。第 08 课会详细讲。
- **1000 步太多了。** 生产环境使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见第 12 课。

## 应用场景

| 角色 | 2026 年的典型技术栈 |
|------|---------------------|
| 图像像素空间扩散（小型、玩具级） | DDPM + U-Net |
| 图像潜在扩散 | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频潜在扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜在扩散 | Encodec + 扩散 Transformer |
| 科学领域（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散模型是通用的生成式骨干网络。流匹配（flow matching，第 13 课）是 2024-2026 年的竞争者，通常在相同质量下推理速度更快。

## 交付

保存 `outputs/skill-diffusion-trainer.md`。该 skill 接收一个数据集 + 计算预算，输出：调度（线性/余弦/S 型（sigmoid））、预测目标（ε/v/x）、步数、引导尺度、采样器族，以及评估协议。

## 练习

1. **简单。** 在 `code/main.py` 中把 T 从 40 改成 10。样本质量（输出可视化直方图）如何退化？双峰结构在多大的 T 下会崩塌？
2. **中等。** 从 ε-prediction 切换到 v-prediction。重新推导反向步骤。比较最终样本质量。
3. **困难。** 加入无分类器引导。以类别标签 `c ∈ {0, 1}` 为条件，训练时 10% 的概率丢弃它，采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。测量 `w = 0, 1, 3, 7` 时的条件模式命中率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 前向过程（forward process） | “加噪” | 固定的马尔可夫链 `q(x_t | x_{t-1})`，用于逐步破坏数据。 |
| 反向过程（reverse process） | “去噪” | 学习到的链 `p_θ(x_{t-1} | x_t)`，用于重建数据。 |
| β 调度（β schedule） | “噪声阶梯” | 每步方差；可选线性、余弦或 S 型（sigmoid）。 |
| α̅ | “Alpha bar” | 累计乘积 `∏(1 - β)`；给出从 `x_0` 到 `x_t` 的闭式。 |
| 简单损失（simple loss） | “对噪声的 MSE” | `||ε - ε_θ(x_t, t)||²`；所有变分推导都退化为它。 |
| ε-prediction | “预测噪声” | 输出即被加入的噪声；标准 DDPM。 |
| V-prediction | “预测速度” | 输出为 `α·ε - σ·x`；在 `t` 的全范围内条件更稳定。 |
| DDPM | “那篇论文” | Ho 等人 2020；线性 β、1000 步、U-Net。 |
| DDIM | “确定性采样器” | 非马尔可夫采样器，20-50 步，训练目标相同。 |
| 无分类器引导（classifier-free guidance） | “CFG” | 混合有条件与无条件噪声预测，以放大条件作用。 |

## 生产提示：扩散推理是一个步数问题

DDPM 论文运行 T=1000 步反向过程。没人会在生产里直接上线这个。每个真实推理栈都会选择以下三种策略之一——而每种都清晰对应到生产里“延迟从哪里来”的分析框架：

1. **更快采样器，模型不变。** DDIM（20-50 步）、DPM-Solver++（10-20 步）、UniPC（8-16 步）。直接替换反向循环；训练好的 `ε_θ` 权重不动。延迟降低 20-50 倍。
2. **蒸馏（distillation）。** 训练学生网络在更少步数内匹配老师：渐进蒸馏（2 → 1）、一致性模型（任意 → 1-4 步）、LCM、SDXL-Turbo、SD3-Turbo。延迟再降 5-10 倍，但需要重新训练。
3. **缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA attention、bf16 权重。每步延迟降低约 2 倍。可与（1）和（2）叠加。

对于一个生产级扩散服务，预算讨论与 LLM 的生产文献所述相同：延迟 = `num_steps × step_cost + VAE_decode`，吞吐量 = `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（只需一步）；从用户视角看 TPOT 等价于完整响应时间，因为图像生成是“一次性完成”的。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) —— 扩散模型的奠基论文，超前于时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) —— DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) —— DDIM，更少步数。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) —— 余弦调度、学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) —— 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) —— CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) —— 统一记号、最清晰的配方。
