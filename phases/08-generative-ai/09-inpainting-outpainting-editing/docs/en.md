# 图像修复、外扩与编辑

> 文生图创造新事物，图像修复则修正旧内容。在实际生产中，70% 的可计费图像工作都是编辑——换背景、去 Logo、扩展画布、重绘手部。图像修复才是扩散模型真正创造价值的地方。

**类型：** 动手实践
**语言：** Python
**前置知识：** 第 8 阶段 · 07（潜空间扩散），第 8 阶段 · 08（ControlNet 与 LoRA）
**时长：** 约 75 分钟

## 问题背景

客户发来一张完美的产品照片，但背景里有一块显眼的标识。你想抹掉它，同时让其他部分像素级保持一致。你不能从头跑文生图——结果会有不同的颜色、不同的光照、不同的产品角度。你只想重新生成被掩膜覆盖的区域，并且让重绘内容与周围上下文协调一致。

这就是图像修复（inpainting）。它的变体包括：

- **图像修复（Inpainting）。** 在掩膜（mask）内部重新生成，保留外部像素。
- **图像外扩（Outpainting）。** 在掩膜外部（或画布之外）重新生成，保留内部内容。
- **图像编辑（Image editing）。** 重新生成整图，但保持对原图的语义或结构保真度（如 SDEdit、InstructPix2Pix）。

2026 年的每一款扩散（diffusion）管线都内置了修复模式：Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们的原理相同。

## 核心概念

![图像修复：掩膜感知的去噪，并重新注入上下文信息](../assets/inpainting.svg)

### 朴素方法（以及为什么它不对）

给标准文生图加上一个掩膜。在每一步采样中，把含噪潜变量（noisy latent）里未掩膜区域替换成前向扩散后的清晰图像。它能跑……但效果很差。边界伪影会透出来，因为模型对掩膜区域内的内容一无所知。

### 正确的修复模型

训练一个改进版 U-Net，输入通道从 4 个变成 9 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

多出来的通道是 VAE 编码后的源图像副本，外加一个单通道掩膜。训练时，随机遮挡图像的一部分区域，训练模型只去噪被掩膜覆盖的区域，同时把未掩膜区域作为干净的条件信号。推理时，模型能够“看到”掩膜周围的上下文，从而生成连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都采用这种 9 通道（或类似）输入。在 Diffusers 中对应 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng 等，2022）——无需训练的编辑

将源图像加噪到某个中间时间步 `t`，然后用新的提示词从 `t` 反向采样到 0。无需重新训练。起始 `t` 的选择在保真度与创作自由度之间权衡：

- `t/T = 0.3` → 几乎与源图一致，仅有轻微风格变化
- `t/T = 0.6` → 适度编辑，保留粗粒度结构
- `t/T = 0.9` → 从接近噪声的状态生成，对原图保留很少

### InstructPix2Pix（Brooks 等，2023）

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时，同时以输入图像和文本指令（如“把它变成日落”“加一条龙”）作为条件。它有两组 CFG（classifier-free guidance）缩放系数：图像尺度与文本尺度。

### RePaint（Lugmayr 等，2022）

保留标准的无条件扩散模型。在反向采样的每一步中，重新采样——偶尔跳回到更噪的状态再重新去噪。这样可以避免边界伪影。适用于没有专门训练修复模型的情况。

## 动手实现

`code/main.py` 实现了一个在 5 维数据上的玩具级一维图像修复方案。我们在 5 维混合数据上训练一个 DDPM（去噪扩散概率模型），每个样本是从两个簇之一采样的 5 个浮点数。推理时，我们“掩膜”5 个维度中的 2 个，在每一步注入未掩膜 3 个维度的前向加噪版本，并只重新生成被掩膜的维度。

### 步骤 1：5 维 DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步骤 2：在所有 5 个维度上训练去噪器

标准 DDPM。网络对 5 维含噪输入输出 5 维噪声预测。

### 步骤 3：推理时进行掩膜感知的反向采样

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这就是朴素方法，在一维玩具数据上有效。真实图像修复使用 9 通道输入，因为纹理连贯性更重要。

### 步骤 4：图像外扩

图像外扩就是把掩膜反转后的修复：把新增（原先不存在）的画布区域掩膜，其余区域填入原图。训练目标完全相同。

## 常见陷阱

- **接缝（Seams）。** 朴素方法会留下可见边界，因为梯度信息不会跨掩膜传播。修复方法：将掩膜膨胀 8–16 像素，或使用专门的修复模型。
- **掩膜泄漏（Mask leakage）。** 如果条件图像中未掩膜区域质量低或含噪，会污染掩膜内部的生成。可轻度去噪或模糊。
- **CFG 与掩膜大小相关。** 小掩膜配高 CFG 会得到过饱和斑块。小编辑应降低 CFG。
- **SDEdit 保真度悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能丢失主体身份。需扫参并保存检查点。
- **提示词不匹配。** 提示词应描述*整张*图像，而不只是新内容。应写“一只猫坐在椅子上”，而不是“一只猫”。

## 应用指南

| 任务 | 管线 |
|------|----------|
| 移除小面积物体 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + “日落蓝天” |
| 扩展画布 | SDXL 外扩模式（8 像素羽化）或 Flux-Fill 外扩掩膜 |
| 重绘手部 / 面部 | SD-Inpaint，提示词重新描述主体 + ControlNet-Openpose |
| 改变局部风格 | 在掩膜区域用 SDEdit 取 `t/T=0.5` |
| “把它变成日落” | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 掩膜 → SD-Inpaint |
| 超高保真 | Flux-Fill 或 GPT-Image（托管）处理最难场景 |

SAM（Meta 的 Segment Anything，2023）+ 扩散修复是 2026 年背景移除的主流管线。SAM 2（2024）支持视频。

## 交付

保存 `outputs/skill-editing-pipeline.md`。该技能接收原始图像 + 编辑描述 + 可选掩膜（或 SAM 提示词），输出：掩膜生成方案、基础模型、CFG 缩放系数（图像 + 文本）、SDEdit-t 或修复模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，把被掩膜维度比例从 0.2 变到 0.8。在哪个比例下，修复质量（掩膜维度的残差）会与无条件生成相当？
2. **中等。** 实现 RePaint：每 10 个反向步骤回退 5 步（加噪）并重新去噪。测量它是否能降低掩膜边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill，在 20 个面部重绘任务上分别打分姿态遵循度与身份保持度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Inpainting | “补洞” | 在掩膜内重新生成；保留外部像素。 |
| Outpainting | “扩展画布” | 在画布外重新生成；保留内部内容。 |
| 9-channel U-Net | “专业修复模型” | 输入为 `noisy \| encoded-source \| mask` 的 U-Net。 |
| SDEdit | “带噪声水平的图生图” | 将图像加噪到时间步 `t`，再用新提示词去噪。 |
| InstructPix2Pix | “只用文本编辑” | 在（图像、指令、输出）三元组上微调的扩散模型。 |
| RePaint | “无需重训练” | 反向过程中周期性地重新加噪，以减少接缝。 |
| SAM | “Segment Anything” | 通过点击或框选生成掩膜；常与修复模型搭配。 |
| Flux-Kontext | “带上下文编辑” | 接受参考图像 + 指令进行编辑的 Flux 变体。 |

## 生产提示：编辑管线对延迟敏感

用户编辑图像时期望往返时间低于 5 秒。30 步 SDXL-Inpaint 在 1024² 分辨率下于 L4 上约 3–4 秒，加上 SAM 掩膜生成（约 200 毫秒）和 VAE 编解码（合计约 500 毫秒）。在生产视角下，这是 TTFT（time-to-first-token）受限而非吞吐量受限——batch 为 1、并发低，必须压缩每个阶段：

- **SAM-H 是慢的。** SAM-H 在 1024² 下约 200 毫秒；SAM-ViT-B 约 40 毫秒，质量损失很小。SAM 2（视频）增加时序开销；单图编辑不要用。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 会编码为潜变量。如果你已有上一步生成的潜变量（迭代式编辑 UI 常见），直接通过 `latents=...` 传入，跳过一次 VAE 编码。
- **掩膜膨胀也影响吞吐。** 小掩膜意味着大部分 U-Net 前向计算被浪费（未掩膜像素反正会被钳制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 仍跑完整 U-Net；只有真正的 9 通道修复变体才能利用掩膜计算。
- **Flux-Kontext 是 2025 年的答案。** 对 `(source_image, instruction)` 单次前向传播——无需单独掩膜，无需 SDEdit 噪声扫描。在 H100 上约 1.5 秒完成一次编辑。架构启示：把多个阶段压缩成一步。

## 延伸阅读

- [Lugmayr 等（2022）。RePaint：使用去噪扩散概率模型进行图像修复](https://arxiv.org/abs/2201.09865) —— 无需训练的修复方法。
- [Meng 等（2022）。SDEdit：基于随机微分方程的引导式图像合成与编辑](https://arxiv.org/abs/2108.01073) —— SDEdit。
- [Brooks、Holynski、Efros（2023）。InstructPix2Pix](https://arxiv.org/abs/2211.09800) —— 基于文本指令的编辑。
- [Kirillov 等（2023）。Segment Anything](https://arxiv.org/abs/2304.02643) —— SAM，掩膜来源。
- [Ravi 等（2024）。SAM 2：图像与视频中的 Segment Anything](https://arxiv.org/abs/2408.00714) —— 视频版 SAM。
- [Hertz 等（2022）。Prompt-to-Prompt：基于交叉注意力控制的图像编辑](https://arxiv.org/abs/2208.01626) —— 注意力层级的编辑。
- [Black Forest Labs（2024）。Flux.1-Fill 与 Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) —— 2024 年工具。
