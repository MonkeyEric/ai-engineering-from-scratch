# ControlNet、LoRA 与条件控制

> 仅靠文本是一种笨拙的控制信号。ControlNet 让你克隆一个预训练的扩散模型（diffusion model），并用深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 让你只训练一千万参数就能微调一个 20 亿参数的模型。两者结合，把 Stable Diffusion 从玩具变成了 2026 年每家工作室都在使用的图像管线（image pipeline）。

**类型：** 实践构建  
**语言：** Python  
**前置要求：** 第 8 阶段 · 07（潜变量扩散，latent diffusion），第 10 阶段（从零开始的大语言模型，LLMs from Scratch —— 作为 LoRA 的基础）  
**预计时间：** ~75 分钟

## 问题所在

像“一位身穿红色连衣裙的女士在繁忙街道上遛狗”这样的提示词（prompt），无法告诉模型狗在*哪里*、女士的*姿态*如何、街道的*视角*是什么。文本大约只能锁定图像所需信息的 10%。其余都是视觉信息，很难用文字高效描述。

为每一种信号（姿态、深度、Canny 边缘、分割）从头训练一个新的条件模型是不可承受的。你希望保持 26 亿参数的 SDXL 主干网络冻结，接入一个读取条件（conditioning）的小型旁路网络，由它去微调主干网络的中间特征（feature）。这就是 ControlNet。

你还希望在无需重新训练整个模型的情况下教会模型新概念（你的脸、你的产品、你的风格）。你想要一个缩小到 1/100 的增量。这就是 LoRA——低秩适配器（low-rank adapter），插入到现有的注意力（attention）权重中。

ControlNet + LoRA + 文本 = 2026 年从业者的标准工具箱。大多数生产级图像管线会在 SDXL / SD3 / Flux 基础模型之上叠加 2-5 个 LoRA、1-3 个 ControlNet，再加一个 IP-Adapter。

## 核心概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等人，2023）

取一个预训练的 SD。把 U-Net 的编码器（encoder）一半*克隆*出来。冻结原始网络。训练这个克隆体接受额外的条件输入（边缘、深度、姿态）。通过*零卷积（zero-convolution）*跳跃连接（1×1 卷积，初始化为零——开始时是无操作，再学习一个增量）把它接回原始网络的解码器（decoder）一半。

```
SD U-Net 解码器：   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 起初等价于恒等映射——即使还没训练也不会造成损害。用 100 万组（prompt、condition、image）三元组和标准扩散损失（diffusion loss）进行训练。

每种模态的 ControlNet 都作为小型旁路模型发布（SDXL 约 360MB，SD 1.5 约 70MB）。推理时可以组合使用：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等人，2021）

对于模型中任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力（attention）层通常秩（rank）为 4-16，重型微调则为 64-128。新增参数量为 `2 · d · r`，而非 `d²`。以 SDXL 的注意力层 `d=640`、`r=16` 为例：每个适配器只需 2 万参数，而不是 41 万——减少了 20 倍。放到整个模型看：一个 LoRA 通常只有 20-200MB，而基础模型是 5GB。

推理时可以缩放 LoRA：`W' = W + α · B @ A`。通常 `α = 0.5-1.5`。多个 LoRA 可以叠加（但要注意它们会以非线性方式相互影响）。

### IP-Adapter（Ye 等人，2023）

一种小型适配器，接受*图像*作为条件（与文本并列）。它使用 CLIP 图像编码器生成图像词元（image tokens），并把这些词元注入交叉注意力（cross-attention），与文本词元一起作用。每个基础模型约 20MB。无需 LoRA 即可实现“按这张参考图的风格生成图像”。

## 工具组合矩阵

| 工具 | 控制内容 | 大小 | 适用场景 |
|------|---------|------|---------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格化 |
| IP-Adapter | 来自参考图的风格或主体 | 20MB | 文字无法描述外观时 |
| Textual Inversion | 把单个概念变成新词元 | 10KB | 遗留方案，大多被 LoRA 取代 |
| DreamBooth | 对主体进行完整微调 | 2-5GB | 强身份保持、高算力 |
| T2I-Adapter | 更轻量的 ControlNet 替代方案 | 70MB | 边缘设备、推理预算受限 |

ControlNet ≈ 空间。LoRA ≈ 语义。两者一起用。

## 动手实现

`code/main.py` 在一维（1-D）上模拟这两种机制：

1. **LoRA。** 一个预训练线性层 `W`。冻结它。训练低秩 `B @ A`，使得 `W + BA` 逼近目标线性层。证明 `r = 1` 就足以完美学习一个秩为 1 的修正。

2. **ControlNet 精简版。** 一个“冻结的基础”预测器和一个读取额外信号的“旁路网络”。旁路网络的输出由一个初始化为零的可学习标量门控（我们这里的零卷积等价物）。训练并观察门控逐渐增大。

### 步骤 1：LoRA 的数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W 被冻结；A、B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步骤 2：零初始化的旁路网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate 初始化为 0
h = base(x) + gated
```

在第 0 步时，输出与基础模型完全一致。早期训练会缓慢更新 `gate`——不会出现灾难性漂移。

## 常见陷阱

- **LoRA 缩放过度。** `α = 2` 或 `α = 3` 是常见的“让它更强”的捷径，但会导致过度风格化或崩坏输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 姿态 ControlNet 权重设为 1.0 同时深度 ControlNet 权重也设为 1.0 通常会过冲。权重之和 ≈ 1.0 是一个安全的默认值。
- **LoRA 用错基础模型。** SDXL LoRA 在 SD 1.5 上会静默失效，因为注意力维度不匹配。Diffusers 0.30+ 会给出警告。
- **Textual Inversion 漂移。** 在某个检查点（checkpoint）上训练的词元迁移到另一个检查点会严重漂移。LoRA 更具可迁移性。
- **LoRA 权重合并与存储。** 你可以把 LoRA 烘焙进基础模型权重以加快推理（无需运行时相加），但会失去运行时缩放 `α` 的能力。建议两个版本都保留。

## 实际应用

| 目标 | 2026 年的管线 |
|------|--------------|
| 复现某品牌的艺术风格 | 在约 30 张精选图像上以秩 32 训练 LoRA |
| 把自己的脸放进生成图像 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 指定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 图像修复（第 09 课） |
| 快速 1 步风格化 | SDXL-Turbo 上的 LCM-LoRA |

## 交付任务

保存 `outputs/skill-sd-toolkit-composer.md`。该技能接收一个任务（输入资产：prompt、可选参考图、可选姿态、可选深度、可选涂鸦），输出工具栈、权重和可复现的种子协议（seed protocol）。

## 练习

1. **简单。** 在 `code/main.py` 中，把 LoRA 秩 `r` 从 1 改到 4。秩达到多少时，LoRA 能精确匹配一个秩为 2 的目标增量？
2. **中等。** 在两个目标变换上分别训练两个 LoRA。一起加载它们，展示它们的加性交互。什么时候交互会偏离线性？
3. **困难。** 使用 diffusers 叠加：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。测量随着堆叠权重变化时，FID 与提示词遵循度之间的权衡。

## 关键术语

| 术语 | 大家的说法 | 实际含义 |
|------|-----------|---------|
| ControlNet | “空间控制” | 克隆的编码器 + 零卷积跳跃连接；读取一张条件图像。 |
| Zero convolution | “从恒等开始” | 初始化为零的 1×1 卷积；ControlNet 起初是无操作。 |
| LoRA | “低秩适配器” | `W + B @ A`，`r << d`；参数量仅为完整微调的 1/100。 |
| 秩 r | “调节旋钮” | LoRA 的压缩度；通常 4-16，64+ 用于重型个性化。 |
| α | “LoRA 强度” | 对 LoRA 增量进行运行时缩放。 |
| IP-Adapter | “参考图像” | 通过 CLIP 图像词元进行图像条件控制的小型适配器。 |
| DreamBooth | “完整主体微调” | 在约 30 张主体图像上训练整个模型。 |
| Textual Inversion | “新词元” | 只学习一个新的词嵌入（embedding）；遗留方案，大多已被取代。 |

## 生产注意事项：LoRA 热切换、ControlNet 通道与多租户服务

一个真正的文本生成图像 SaaS 会在同一个基础检查点之上服务数百个 LoRA 和十几个 ControlNet。服务问题与大语言模型（LLM）的多租户（multi-tenant）非常相似（生产文献中在连续批处理、LoRAX / S-LoRA 的 LLM 场景下有大量讨论）：

- **热切换 LoRA，不要合并。** 把 `W' = W + α·B·A` 合并进基础权重可以让每步推理快 3-5%，但会固定 `α` 和基础模型。把 LoRA 作为秩为 r 的增量热驻留在显存（VRAM）中；diffusers 提供 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 来实现按请求激活。切换成本是 `2 · d · r · num_layers` 个权重——MB 级别，亚秒级。
- **ControlNet 作为第二条注意力通道。** 克隆的编码器与基础模型并行运行。两个权重各为 1.0 的 ControlNet 意味着每步多出两次前向传播，而不是一次合并传播。批次余量会呈二次下降。要为每个激活的 ControlNet 预留约 1.5 倍的单步成本。
- **LoRA 也可以量化。** 如果你把基础模型量化了（参见第 07 课，8GB 上的 Flux），LoRA 增量也能干净地量化到 8 位或 4 位。QLoRA 风格的加载让你能在 4 位 Flux 基础模型上叠加 5-10 个 LoRA，而不会爆显存。

Flux 特别提示：Niels 的 Flux-on-8GB notebook 把基础模型量化到 4 位；在此基础上加载风格 LoRA（`pipe.load_lora_weights("user/style-lora")`），并以 `weight_name="pytorch_lora_weights.safetensors"` 设置权重，仍然可以工作。这就是 2026 年大多数 SaaS 工作室交付的配方。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) —— ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) —— LoRA（最初用于大语言模型；后迁移到扩散模型）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) —— IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) —— ControlNet 的更轻量替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) —— DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter 文档](https://huggingface.co/docs/diffusers/training/controlnet) —— 参考管线。
