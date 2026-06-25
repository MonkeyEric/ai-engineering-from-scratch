# 潜在扩散与 Stable Diffusion

> 在 512×512 图像的像素空间（pixel space）里做扩散，简直是计算上的“战争罪”。Rombach 等人（2022）意识到：生成图像并不需要全部 78.6 万个维度——只需要足够捕捉语义结构的维度，剩下的交给单独的解码器。在 VAE 的潜在空间（latent space）里跑扩散，就是 Stable Diffusion 的核心思想。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02（VAE）、Phase 8 · 06（DDPM）、Phase 7 · 09（ViT）
**Time:** 约 75 分钟

## 问题所在

512² 像素空间扩散意味着 U-Net 要处理形状为 `[B, 3, 512, 512]` 的张量。对于一个 5 亿参数的 U-Net，每步采样约需 100 GFLOPS；50 步就是单张图 5 TFLOPS。如果用十亿张图片训练，算力账单高得离谱。

这些 FLOPS 大多浪费在推动感知上并不重要的细节上——也就是高频纹理，而一个带损 VAE 完全可以把它们压缩掉。Rombach 的想法是：先训练一个 VAE（*第一阶段*），然后冻结它，在 4 通道 64×64 的潜在空间里完整运行扩散（*第二阶段*）。同样的 U-Net，像素数只有 1/16，FLOPS 减少约 64 倍，质量却相当。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 使用 860M 的 U-Net 处理 `64×64×4` 的潜在变量；SDXL 使用 2.6B 的 U-Net 处理 `128×128×4`；SD3 把 U-Net 换成了带流匹配（flow matching）的 Diffusion Transformer（DiT）；Flux.1-dev（Black Forest Labs，2024）则搭载了一个 12B 参数的 DiT-MMDiT。它们都建立在同一个两阶段框架之上。

## 核心概念

![潜在扩散：VAE 压缩 + 在潜在空间里做扩散](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1. **第一阶段 —— VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴下采样 8 倍，并调整通道数，使潜在变量总大小约为像素数的 1/16。损失（loss）= 重建损失（L1 + LPIPS 感知损失）+ KL 散度（KL 权重很小，因此 `z` 不会被强迫太接近高斯分布，因为我们并不需要从 `z` 中精确采样）。通常还会加入对抗损失，让解码图像更锐利。

2. **第二阶段 —— 在 `z` 上做扩散。** 把 `z = E(x_real)` 当作数据。训练一个 U-Net（或 DiT）对 `z_t` 去噪。推理时：通过扩散采样得到 `z_0`，再计算 `x = D(z_0)`。

**文本条件。** 还有两个额外组件。一个是冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。另一个是交叉注意力（cross-attention）注入：每个 U-Net 块接收 `[Q = 图像特征, K = V = 文本 token]` 并将它们融合。文本 token 是文本影响图像的唯一途径。

**损失函数和第 06 课完全一样。** 同样是 DDPM / 流匹配中的噪声均方误差（MSE）。你只是把数据域换了一下。

## 架构变体

| Model | Year | Backbone | Latent shape | Text encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 tokens） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT（multimodal DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

趋势：用 DiT 取代 U-Net（在潜在 patch 上运行的 transformer）、放大文本编码器（T5 在 prompt 遵循度上优于 CLIP）、增加潜在通道数（4 → 16，为细节留出更多余量）。

## 动手实现

`code/main.py` 在第 06 课的 DDPM 之上堆叠了一个玩具级 1-D “VAE”（恒等编码器 + 解码器，仅用于演示；真实 VAE 会是卷积网络），并加入了类别条件与分类器无关引导（classifier-free guidance）。它展示了关键洞见：无论是在原始 1-D 数值上运行，还是在编码后的数值上运行，扩散损失都同样有效。

### 步骤 1：编码器 / 解码器

```python
def encode(x):    return x * 0.5          # 玩具级“压缩”到更小尺度
def decode(z):    return z * 2.0
```

真实 VAE 有训练好的权重。为了教学，这个线性映射足以说明：扩散在 `z` 上操作，而不关心原始数据空间。

### 步骤 2：在 `z` 空间做扩散

和 第 06 课 的 DDPM 相同。网络看到的数据是 `z = E(x)`。采样得到 `z_0` 后，用 `D(z_0)` 解码。

### 步骤 3：分类器无关引导

训练时，10% 的概率丢弃类别标签（替换为 null token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完全多样），`w = 3` = 默认值，`w = 7+` = 饱和 / 过度锐利。

### 步骤 4：文本条件（概念，不在代码中实现）

把类别标签替换为冻结文本编码器的输出。通过交叉注意力（cross-attention）把文本嵌入（text embedding）送入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类别条件扩散模型与 Stable Diffusion 之间唯一实质性的区别。

## 常见陷阱

- **VAE 尺度不匹配。** SD 1.x 的 VAE 在编码后会乘一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记它会让 U-Net 在方差完全错误的潜在变量上训练。每个 checkpoint 都附带这个参数。
- **文本编码器悄悄出错。** SD3 需要 T5-XXL 且 token 长度 ≥128，仅回退到 CLIP 会有损。务必检查 `use_t5=True`，否则 prompt 保真度会暴跌。
- **混用潜在空间。** SDXL、SD3、Flux 使用不同的 VAE。在 SDXL 潜在空间上训练的 LoRA 无法用于 SD3。Hugging Face diffusers 0.30+ 会拒绝加载不匹配的 checkpoint。
- **CFG 过高。** `w > 10` 会产生饱和、油腻的图像，并为了 prompt 贴合而牺牲多样性。甜点区是 `w = 3-7`。
- **负提示词泄漏。** 空负提示词会成为 null token；填写了内容的负提示词会成为 `ε_uncond`。二者并不相同；有些 pipeline 会默默默认使用 null token。

## 实际应用

2026 年的生产栈：

| Target | Recommended backbone |
|--------|----------------------|
| 窄领域、成对数据、从头训练模型 | SDXL 微调（LoRA / full）—— 最快交付 |
| 开放域文生图、开放权重 | Flux.1-dev（12B，Apache / 非商业）或 SD3.5-Large |
| 最快推理、开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳 prompt 遵循度、托管服务 | GPT-Image / DALL-E 3（仍是如此）、Midjourney v7、Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）—— 原生支持图像 + 文本 |
| 研究、基线 | SD 1.5 —— 古老但研究充分 |

## 交付

保存 `outputs/skill-sd-prompter.md`。该技能接收文本 prompt + 目标风格，输出：模型与 checkpoint、CFG scale、采样器、负提示词、分辨率、可选的 ControlNet/IP-Adapter 组合，以及每步 QA 检查清单。

## 练习

1. **简单。** 用 guidance `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每类的平均样本。在哪个 `w` 时，类均值会偏离真实数据均值？
2. **中等。** 把玩具级线性编码器换成带重建损失的 tanh-MLP 编码器/解码器对。在新的潜在变量上重新训练扩散。样本质量会变化吗？
3. **困难。** 用 diffusers 搭建真实 Stable Diffusion 推理：加载 `sdxl-base`，跑 30 步 Euler + CFG=7，计时。然后切换到 `sdxl-turbo`，4 步、CFG=0。同一主题，不同质量——描述变化及原因。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| First stage | “The VAE” | 训练好的编码器/解码器对；把 512² 压缩到 64²。 |
| Second stage | “The U-Net” | 在潜在空间上运行的扩散模型。 |
| CFG | “Guidance scale” | `(1+w)·ε_cond - w·ε_uncond`；调节条件强度。 |
| Null token | “Empty prompt embed” | 用于 `ε_uncond` 的无条件嵌入。 |
| Cross-attention | “How text gets in” | 每个 U-Net 块以文本 token 作为 K 和 V 进行注意力计算。 |
| DiT | “Diffusion Transformer” | 用 transformer 替代 U-Net，在潜在 patch 上运行；扩展性更好。 |
| MMDiT | “Multi-modal DiT” | SD3 的架构：文本和图像流联合注意力。 |
| VAE scaling factor | “Magic number” | 把潜在变量除以约 5.4，使扩散在单位方差空间运行。 |

## 生产笔记：在 8GB 消费级 GPU 上运行 Flux-12B

参考的 Flux 集成是经典的“我只有消费级 GPU，能上线吗？”解决方案。秘诀就是把生产推理文献里列出的三个旋钮用到扩散 DiT 上：

1. **错峰加载。** Flux 有三个网络无需同时驻留显存：T5-XXL 文本编码器（fp32 下约 10 GB）、CLIP-L（很小）、12B MMDiT 和 VAE。先编码 prompt，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。8GB 消费级 GPU 一次只能装下一个阶段。
2. **bitsandbytes 4-bit 量化。** 对 T5 编码器和 DiT 都使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存降低 8 倍，文生图质量下降几乎不可感知（参考 notebook 中 Aritra 的 benchmark 链接）。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 会在每次前向传播时自动在 CPU 和 GPU 之间切换模块。延迟增加 10-20%，但能让 pipeline 跑得起来。

内存估算：`10 GB T5 / 8 = 1.25 GB`（量化后），`12 B 参数 × 0.5 字节 = ~6 GB`（量化后的 DiT），再加上激活值。用 stas00 的话说，这是 TP=1 推理的极端情况——没有模型并行，最大化量化。生产环境你会在 H100 上跑 TP=2 或 TP=4；对于单台开发笔记本，这就是配方。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) —— Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) —— SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) —— DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— SD3、MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) —— CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) —— Flux.1 家族。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) —— 上述所有 checkpoint 的参考实现。
