# 图像生成——扩散模型

> 扩散模型学习的是去噪。训练它从一张带噪图像中去除一点点噪声，然后反向重复一千次，你就得到了一个图像生成器。

**类型：** Build
**语言：** Python
**前置知识：** 阶段 4 第 07 课（U-Net）、阶段 1 第 06 课（概率论）、阶段 3 第 06 课（优化器）
**时长：** ~75 分钟

## 学习目标

- 推导前向加噪过程 `x_0 -> x_1 -> ... -> x_T`，并解释为什么闭式解 `q(x_t | x_0)` 对任意 t 都成立
- 实现一个 DDPM 风格的训练目标：回归每一步加入的噪声，并实现一个从纯噪声走回图像的采样器
- 构建一个时间条件化的 U-Net（小到能在 CPU 上训练），用于预测任意时间步的噪声
- 解释 DDPM 与 DDIM 采样的区别，以及各自适用的场景（第 23 课将深入讲解流匹配与 Rectified Flow）

## 问题背景

GAN 是“一步生成”：噪声进去，图像出来，一次前向传播完成。它们快，但难训练。扩散模型是“迭代生成”：从纯噪声开始，一小步一小步地去噪，图像逐渐浮现。它们慢，但容易训练。在过去五年里，后一个特性占据了主导：任何小团队都能训练扩散模型并获得不错的样本；而 GAN 训练是一门需要多年失败经验才能掌握的技艺。

除了训练稳定性之外，扩散模型的迭代结构也成就了现代图像生成的所有能力：文本条件化、图像修复、图像编辑、超分辨率、可控风格。采样循环的每一步都是一个可以注入新约束的钩子。正因如此，Stable Diffusion、Imagen、DALL-E 3、Midjourney，以及你将要使用的每一个可控图像模型，都是基于扩散的。

本课构建一个最简版 DDPM：前向加噪、反向去噪、训练循环。下一课（Stable Diffusion）会把它接入一个生产系统，包含 VAE、文本编码器和分类器自由引导（classifier-free guidance）。

## 核心概念

### 前向过程

取一张图像 `x_0`。加入少量高斯噪声得到 `x_1`。再加一点得到 `x_2`。持续 T 步，直到 `x_T` 几乎与纯高斯噪声无法区分。

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)
```

`beta_t` 是一个较小的方差调度表，通常在线性区间内从 0.0001 增加到 0.02，共 T=1000 步。每一步都轻微压缩信号并注入新的噪声。

### 闭式跳跃

逐步加噪是一个马尔可夫链，但数学上可以折叠：你可以直接从 `x_0` 一步采样出 `x_t`。

```
Define alpha_t = 1 - beta_t
Define alpha_bar_t = prod_{s=1..t} alpha_s

Then:
  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I)

Equivalently:
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  where epsilon ~ N(0, I)
```

这一单个方程就是扩散模型之所以可行的全部原因。训练时，你随机选一个 `t`，直接从 `x_0` 采样 `x_t`，然后一步完成训练——无需模拟整个马尔可夫链。

### 反向过程

前向过程是固定的。反向过程 `p(x_{t-1} | x_t)` 由神经网络学习。扩散模型并不直接预测 `x_{t-1}`；它们预测在步骤 t 加入的噪声 `epsilon`，然后由数学公式从中推导出 `x_{t-1}`。

```mermaid
flowchart LR
    X0["x_0<br/>(干净图像)"] --> Q1["q(x_t|x_0)<br/>加噪"]
    Q1 --> XT["x_t<br/>(带噪图像)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["预测的 epsilon"]
    EPS --> LOSS["与真实 epsilon<br/>的 MSE 损失"]

    XT -.->|采样| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|重复 1000 次| X0S["x_0（生成样本）"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb
```

### 训练损失

每个训练步骤如下：

1. 采样一张真实图像 `x_0`。
2. 从 [1, T] 中均匀采样一个时间步 `t`。
3. 采样噪声 `epsilon ~ N(0, I)`。
4. 计算 `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`。
5. 用网络预测 `epsilon_theta(x_t, t)`。
6. 最小化 `|| epsilon - epsilon_theta(x_t, t) ||^2`。

就是这样。神经网络学习预测任意时间步的噪声。损失是 MSE。没有对抗博弈，没有模式崩溃，没有振荡。

### 采样器（DDPM）

生成时：从 `x_T ~ N(0, I)` 开始，一步一步反向推进。

```
for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    where z ~ N(0, I) if t > 1, else 0
return x_0
```

关键在于，尽管反向条件分布在一般情况下没有闭式解，但对于这个特定的高斯前向过程，它是有闭式解的。那些看起来复杂的系数正是贝叶斯法则给出的结果。

### 为什么要 1000 步

前向噪声调度表的设计目标是每一步只加入足够的噪声，使得反向步骤近似高斯分布。步数太少，反向步骤会远离高斯分布，网络无法很好地建模。步数太多，采样开销增加而收益递减。T=1000 配合线性调度表是 DDPM 的默认设置。

### DDIM：快 20 倍的采样

训练方式不变。采样方式改变。DDIM（Song 等人，2020）定义了一个确定性的反向过程，可以在不重训练的情况下跳过时间步。用 DDIM 以 50 步采样，就能获得接近 DDPM 1000 步的质量。每个生产系统都会使用 DDIM 或更快的变体（DPM-Solver、Euler ancestral）。

### 时间条件化

网络 `epsilon_theta(x_t, t)` 需要知道它正在对哪个时间步进行去噪。现代扩散模型通过正弦时间嵌入（与 Transformer 中的位置编码思想相同）注入 `t`，并在 U-Net 的每一层加到特征图上。

```
t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)
```

没有时间条件化时，网络必须自己从图像中猜测噪声级别，这虽然也能工作，但样本效率低得多。

## 动手实现

### 步骤 1：噪声调度表

```python
import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))
```

预先计算一次，训练和采样时按索引取用。

### 步骤 2：前向扩散（q_sample）

```python
def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise
```

一行闭式解。`t` 是一个时间步批次，批次中的每张图像对应一个时间步。

### 步骤 3：一个极小的时间条件化 U-Net

```python
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)
```

两层 U-Net，在瓶颈处注入时间条件化。处理真实图像时需要增加深度和宽度。

### 步骤 4：训练循环

```python
def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

这就是整个训练循环。没有 GAN 博弈，没有特殊损失，只有一个 MSE 调用。

### 步骤 5：采样器（DDPM）

```python
@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

生成一批样本需要 1000 次前向传播。在实际代码中，你会把它换成 DDIM 50 步采样器。

### 步骤 6：DDIM 采样器（确定性，约快 20 倍）

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x
```

`eta=0` 时完全确定性（相同噪声输入总是产生相同输出）。`eta=1` 时退化为 DDPM。

## 如何使用

在生产环境中，使用 `diffusers`：

```python
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)
```

该库提供了现成的调度器（DDPM、DDIM、DPM-Solver、Euler、Heun）、可配置的 U-Net、文本到图像与图像到图像的 pipeline，以及 LoRA 微调工具。

做研究时，`k-diffusion`（Katherine Crowson）拥有最忠实的参考实现和最好的采样变体。

## 交付物

本课产出：

- `outputs/prompt-diffusion-sampler-picker.md` —— 一个根据质量目标、延迟预算和条件类型选择 DDPM / DDIM / DPM-Solver / Euler 的提示词
- `outputs/skill-noise-schedule-designer.md` —— 一个根据 T 和目标破坏程度生成线性、余弦或 sigmoid beta 调度表，并绘制信噪比随时间变化诊断图的技能

## 练习

1. **（简单）** 可视化前向过程：取一张图像，绘制 `t in [0, 100, 250, 500, 750, 1000]` 时的 `x_t`。验证 `x_1000` 是否看起来像纯高斯噪声。
2. **（中等）** 在 synthetic-circles 数据集上训练 TinyUNet 20 个 epoch，并采样 16 个圆形。比较 DDPM（1000 步）和 DDIM（50 步）采样——在相同噪声种子下它们是否生成相似的图像？
3. **（困难）** 实现余弦噪声调度表（Nichol & Dhariwal, 2021）：`alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`。用线性和余弦调度表分别训练同一模型，并证明余弦调度表在较少步数下生成更好的样本。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 前向过程 | “随时间加噪” | 固定的马尔可夫链，在 T 步内将图像逐步破坏成高斯噪声 |
| 反向过程 | “逐步去噪” | 学习到的分布，从噪声逐步走回图像 |
| Epsilon 预测 | “预测噪声” | 训练目标：`epsilon_theta(x_t, t)` 预测第 t 步加入的噪声 |
| Beta 调度表 | “噪声量” | 由 T 个小方差组成的序列，定义每一步引入多少噪声 |
| alpha_bar_t | “累计保留因子” | 到时间 t 为止 (1 - beta_s) 的乘积；t 越大，剩余信号越少 |
| DDPM 采样器 | “祖先采样，随机” | 从每个条件高斯分布中采样 x_{t-1}；共 1000 步 |
| DDIM 采样器 | “确定性，快速” | 将采样重写为确定性常微分方程；20-100 步即可获得相近质量 |
| 时间条件化 | “告诉模型当前是第几步” | 将 t 的正弦嵌入注入 U-Net，使其知道当前噪声级别 |

## 延伸阅读

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) —— 让扩散模型变得实用并在 FID 上击败 GAN 的开创性论文
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) —— 余弦调度表与 v 参数化
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502) —— 确定性采样器，让实时推理成为可能
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) —— 对扩散模型所有设计选择的统一视角；当前最佳参考
