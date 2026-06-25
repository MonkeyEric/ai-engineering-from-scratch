# 流匹配（Flow Matching）与矫正流（Rectified Flows）

> 扩散模型需要 20–50 个采样步，因为它沿着一条弯曲的路径从噪声走向数据。流匹配（Lipman 等，2023）与矫正流（Liu 等，2022）则训练直线路径。更直的路径意味着更少的步数，也意味着更快的推理。Stable Diffusion 3、Flux.1 与 AudioCraft 2 都在 2024 年转向了流匹配。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 06（DDPM）、Phase 1 · 微积分
**时长：** 约 45 分钟

## 问题所在

DDPM 的反向过程是从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 把它压缩到 20–50 步确定性采样。你希望能更少——最好一步搞定。阻碍在于求解反向过程的 ODE 是刚性的，路径弯曲。

如果你能训练模型让从噪声到数据的路径是一条*直线*，那么从 `t=1` 到 `t=0` 的单步欧拉（Euler）采样就能奏效。流匹配直接构建这一点：定义一条从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练一个向量场 `v_θ(x, t)` 去匹配它的时间导数，然后在推理时积分。

矫正流（Liu 2022）更进一步：通过一种称为回流（reflow）的过程迭代地拉直路径，产生逐渐更接近线性的 ODE。经过两次回流迭代后，2 步采样器即可媲美 50 步 DDPM 的质量。

## 核心概念

![流匹配：噪声与数据之间的直线插值](../assets/flow-matching.svg)

### 直线流（Straight-line flow）

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data`、`x_1 ~ N(0, I)`。沿这条直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义一个神经向量场 `v_θ(x_t, t)`，并训练它去匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是**条件流匹配（conditional flow matching）**损失（Lipman 2023）。训练是无模拟（simulation-free）的：你无需展开 ODE，只需采样 `(x_0, x_1, t)` 并做回归。

### 采样

推理时，沿时间*反向*积分学到的向量场：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 出发，用欧拉步长下降到 `t=0`。

### 矫正流（Rectified flow，Liu 2022）

直线流虽然有效，但学到的路径*实际上并不直*——因为多个 `x_0` 可能映射到同一个 `x_1`，所以路径会弯曲。矫正流的回流步骤如下：

1. 用随机配对训练流模型 v_1。
2. 通过从 `x_1` 积分 v_1 到落点 `x_0`，采样 N 对 `(x_1, x_0)`。
3. 用这些配对样本训练 v_2。因为这些配对现在是“ODE 匹配”的，它们之间的直线插值真正更平坦。
4. 重复上述过程。

实践中，2 次回流迭代就能接近线性，从而支持 2–4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是从流匹配蒸馏而来的模型。

### 为什么 2024 年图像领域选择了它

三个原因：

1. **无模拟训练**——训练时不需要 ODE 展开，实现极其简单。
2. **更好的损失几何**——直线路径的信噪比一致，而 DDPM 的 ε-损失在调度两端信噪比很差。
3. **更快的推理**——SDXL-Turbo 质量只需 4–8 步；一致性蒸馏（consistency distillation）甚至只需 1 步。

## 流匹配与 DDPM 的精确联系

使用高斯条件路径的流匹配，本质上是用*特定噪声调度*的扩散。选取 `x_t = α(t) x_0 + σ(t) x_1` 这一调度，流匹配就等价于 Stratonovich 形式重写的扩散，其中 `v = α'·x_0 - σ'·x_1`。对于高斯路径，二者在代数上是等价的。

流匹配带来的新增价值：目标的*清晰性*（一个 plain velocity）、更干净的损失，以及尝试非高斯插值的自由。

## 动手实现

`code/main.py` 实现了一维流匹配，数据分布为双峰高斯混合。向量场 `v_θ(x, t)` 是一个小型 MLP，用直线目标训练。推理时分别用 1、2、4、20 个欧拉步积分，并比较样本质量。

### 步骤 1：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 步骤 2：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 步骤 3：比较步数

预期 4 步采样器已经能匹配 20 步质量——这对延迟来说是重大突破。

## 常见陷阱

- **时间参数化。** 流匹配使用 `t ∈ [0, 1]`，`t=0` 对应数据、`t=1` 对应噪声。DDPM 使用 `t ∈ [0, T]`，`t=0` 对应数据、`t=T` 对应噪声。方向相同，尺度不同。论文中经常把这个搞错。
- **调度选择。** 矫正流的直线是流匹配“那个”调度，但你也可以使用余弦（cosine）或对数正态（logit-normal）的 t 采样（SD3 就是如此）以获得更好的尺度覆盖。
- **回流成本。** 为回流生成配对数据集需要对每个样本做一次完整推理。只在真正需要 1–2 步推理时才做回流。
- **无分类器引导（Classifier-free guidance）仍然适用。** 只需把 ε 换成 v 做线性组合：`v_cfg = (1+w) v_cond - w v_uncond`。

## 应用场景

| 应用场景 | 2026 年技术栈 |
|----------|-----------|
| 文本生成图像，最佳质量 | 流匹配：SD3、Flux.1-dev |
| 文本生成图像，1–4 步 | 蒸馏流匹配：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 从流匹配基模做一致性蒸馏（LCM、PCM） |
| 音频生成 | 流匹配：Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | 流匹配与扩散混合（Sora、Veo、Stable Video） |
| 科学 / 物理（粒子轨迹、分子） | 流匹配 + 等变向量场 |

2025–2026 年只要论文说“比扩散更快”，几乎总是流匹配 + 蒸馏。

## 交付成果

保存 `outputs/skill-fm-tuner.md`。该技能接收一个扩散风格的模型规格，并转换为流匹配训练配置：调度选择、时间采样分布（uniform / logit-normal）、优化器、回流计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py`，比较 1 步与 20 步的 MSE 与真实数据分布。
2. **中等。** 把均匀的 `t` 采样换成对数正态（集中在中间 t）。模型质量是否提升？
3. **困难。** 实现一次回流迭代：用第一个模型生成配对的 (x_0, x_1)，在这些配对上训练第二个模型，并比较 1 步样本质量。

## 关键术语

| 术语 | 大家的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Flow matching | “直线扩散” | 训练 `v_θ(x, t)` 沿插值匹配 `x_1 - x_0`。 |
| Rectified flow | “Reflow” | 迭代拉直已学习流的流程。 |
| Velocity field | “v_θ” | 模型输出——移动 `x_t` 的方向。 |
| Straight-line interpolant | “路径” | `x_t = (1-t)·x_0 + t·x_1`；目标导数很简单。 |
| Euler sampler | “一阶 ODE 求解器” | 最简单的积分器；路径越直效果越好。 |
| Logit-normal t | “SD3 采样” | 把 `t` 采样集中在中间值，那里梯度最强。 |
| Consistency distillation | “1 步采样器” | 训练一个学生模型直接把任意 `x_t` 映射到 `x_0`。 |
| CFG with velocity | “v-CFG” | `v_cfg = (1+w) v_cond - w v_uncond`；同样的技巧，新的变量。 |

## 生产笔记：Flux.1-schnell 是流匹配的最快形态

流匹配在生产上的代表作是 Flux.1-schnell——一个流匹配 DiT，被蒸馏到 1–4 步推理，同时保持 Flux-dev 级别质量。Niels 的“在 8GB 机器上运行 Flux”笔记本是部署参考方案：T5 + CLIP 编码、量化的 MMDiT 去噪（schnell 4 步 vs dev 50 步）、VAE 解码。成本核算如下：

| 变体 | 步数 | 在 1024² L4 上的延迟 | 总 FLOPs（相对值） |
|---------|-------|------------------------|------------------------|
| Flux.1-dev（原始） | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08×（快 12 倍） |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

生产铁律：**流匹配基模 + 蒸馏 = 2026 年快速文本生成图像的默认方案。** 每家主要厂商都出货这种组合：SD3-Turbo（SD3 + 流匹配 + 蒸馏）、Flux-schnell（Flux-dev + 矫正流拉直）、CogView-4-Flash。纯扩散基模只存在于旧版检查点中。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) —— 矫正流。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) —— 流匹配。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— SD3，大规模矫正流。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) —— 涵盖流匹配与扩散的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) —— 扩散 / 流的一步步蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) —— Turbo 变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) —— 生产中的流匹配。
