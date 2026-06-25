# 视频生成

> 图像是二维张量。视频是三维张量。理论相同，但计算量要难上 10–100 倍。OpenAI 的 Sora（2024 年 2 月）证明了这是可行的。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 已能从文本生成 1080p 的商用视频——而开放权重栈（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）仅落后约 12 个月。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 07（潜在扩散），Phase 7 · 09（ViT），Phase 8 · 06（DDPM）
**时长：** ~45 分钟

## 问题

一段 10 秒、1080p、24fps 的视频包含 240 帧，每帧 1920×1080×3 像素。每段原始数据约 1.5 GB。在像素空间做扩散不可行。你需要：

1. **时空压缩。** 用 VAE 把视频（而非单帧）编码成一系列时空块（patch）。
2. **时间一致性。** 帧与帧之间需要在数秒内保持内容、光照和物体身份一致。网络必须学会建模运动。
3. **计算预算。** 对相同模型规模，视频训练比图像训练贵 10–100 倍。
4. **条件控制。** 文本、图像（首帧）、音频或另一段视频。大多数生产级模型都接受这四种输入。

解决这一问题的架构，是把**扩散 Transformer（Diffusion Transformer，DiT）**应用于时空块，并在海量（提示词、描述、视频）数据集上训练。损失函数与第 06 课的扩散损失相同。

## 概念

![视频扩散：分块、DiT、解码](../assets/video-generation.svg)

### 分块（Patchify）

用 3D VAE（学习得到的时空压缩）对视频编码。潜在变量形状为 `[T_latent, H_latent, W_latent, C_latent]`。再把它切分成大小为 `[t_p, h_p, w_p]` 的块。对于 Sora 风格的模型，`t_p = 1`（每帧分块）或 `t_p = 2`（每两帧分块）。一段 10 秒 1080p 的视频压缩后约有 20,000–100,000 个块。

### 时空 DiT

Transformer 处理展平后的块序列。每个块带有一个三维位置嵌入（时间 + y + x）。注意力通常会分解：

- **空间注意力：** 在每一帧的块之间进行。
- **时间注意力：** 在相同空间位置跨帧进行。
- **完整 3D 注意力：** 成本高 16–100 倍，仅在低分辨率或研究中使用。

### 文本条件

与大型文本编码器做交叉注意力（Sora 用 T5-XXL，CogVideoX-5B 也用 T5-XXL）。长提示词很重要——Sora 的训练集使用 GPT 生成的密集重标注描述，平均每段视频约 200 个 token。

### 训练

在时空潜在变量上使用标准扩散损失（ε 预测或 v 预测）。数据：网络视频 + 约 1 亿条精选片段 + 合成文本描述。计算量：即使是一次小规模研究训练也需要 10,000+ GPU 小时；Sora 级别则要 100,000+。

## 2026 年生产级格局

| 模型 | 发布时间 | 最大时长 | 最大分辨率 | 开放权重？ | 亮点 |
|-------|------|--------------|---------|---------------|---------|
| Sora（OpenAI） | 2024-02 | 60s | 1080p | 否 | 首个在大规模上展现出世界模拟器特性的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | 否 | 推理速度快 5 倍的生产级 Sora |
| Veo 2（Google） | 2024-12 | 8s | 4K | 否 | 2025 年画质最高、物理最合理 |
| Veo 3 | 2025 Q3 | 15s | 4K | 否 | 原生音频与更强的镜头控制 |
| Kling 1.5 / 2.1（快手） | 2024-2025 | 10s | 1080p | 否 | 2025 年 Q1 人体动作最佳 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 否 | 其上叠加了专业视频工具 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 否 | 角色一致性最强 |
| CogVideoX（THUDM） | 2024 | 10s | 720p | 是（2B、5B） | 首个开放的 5B 级视频模型 |
| HunyuanVideo（腾讯） | 2024-12 | 5s | 720p | 是（13B） | 2024 年末开放 SOTA |
| Mochi-1（Genmo） | 2024-10 | 5.4s | 480p | 是（10B） | 授权最宽松 |
| WAN 2.2（阿里巴巴） | 2025-07 | 5s | 720p | 是 | 2025 年中期最强的开放模型 |

开放权重正在比图像领域更快地缩小差距：到 2026 年中期，HunyuanVideo + WAN 2.2 的 LoRA 已经支撑了大部分开源工作流。

## 动手实现

`code/main.py` 模拟了核心时空 DiT 思想：对一个小的合成视频分块，为每个块加上位置嵌入，然后用类 Transformer 的注意力对整个序列去噪。不用 numpy，纯 Python。我们展示了：当相邻帧的块共享同一个去噪器并拥有位置嵌入时，即使在一维情况下也会出现时间一致性。

### 步骤 1：对合成的一维“视频”分块

```python
def make_video(T_frames=8, rng=None):
    # “视频”是一组沿平滑轨迹变化的一维数值序列
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 步骤 2：为每帧添加位置嵌入

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 步骤 3：去噪器看到整个序列

我们不再独立地为每一帧去噪，而是把这个小网络把所有帧的数值和它们的位置嵌入拼接起来，联合预测所有帧的噪声。

### 步骤 4：时间一致性测试

训练完成后，采样一段视频并测量帧间差值。如果模型学到了时间结构，这些差值会比独立采样每一帧时更小。

## 常见陷阱

- **独立逐帧采样 = 闪烁。** 如果你对每一帧单独运行图像扩散，输出会闪烁，因为每帧的噪声相互独立。视频扩散通过注意力或共享噪声把帧耦合起来，从而解决这个问题。
- **朴素 3D 注意力 = 显存溢出。** 对 10 秒 1080p 的潜在变量做完整 3D 注意力，运算量高达数千亿次。应分解为空间 + 时间注意力。
- **数据描述比数据量更重要。** Sora 相比之前工作的主要升级，是训练时使用了约 10 倍更详细的描述（GPT-4 重新标注片段）。OpenAI 的技术报告明确指出了这一点。
- **首帧条件控制。** 大多数生产级模型也接受一张图像作为首帧，这就是“图生视频（image-to-video）”模式；训练时会包含这种变体。
- **物理漂移。** 长片段（>10 秒）会累积细微不一致。滑动窗口生成 + 关键帧锚定可以缓解。

## 应用场景

| 使用场景 | 2026 年推荐 |
|----------|-----------|
| 最高质量文生视频，托管服务 | Veo 3 或 Sora |
| 镜头可控的电影级视频 | Runway Gen-3 + Motion Brush |
| 多片段角色一致性 | Pika 2.0 或 Kling 2.1 |
| 开放权重、快速微调 | WAN 2.2 + LoRA |
| 图生视频 | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音频驱动唇形同步 | Veo 3（原生音频）或专用唇形同步模型 |
| 视频编辑 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（静帧） |

在质量相当时，2024 到 2026 年间每秒视频成本下降了约 20 倍。

## 交付

保存 `outputs/skill-video-brief.md`。该技能接收一份视频简报（时长、宽高比、风格、镜头计划、主体一致性、音频），输出：模型 + 托管方案、提示词脚手架（镜头语言、主体描述、运动描述词）、种子与可复现协议，以及一份帧级 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中比较（a）独立逐帧采样和（b）联合序列采样的帧间差值。报告差值的均值与方差。
2. **中等。** 添加首帧条件：将第 0 帧固定为给定值，再采样其余帧。测量该固定值如何传播。
3. **困难。** 使用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。以 720p 生成一段 6 秒视频，计时 20 步推理。分析时空注意力的性能瓶颈。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Video VAE | “3-D VAE” | 将 `(T, H, W, C)` 压缩为时空潜在变量的编码器。 |
| Patches | “The tokens” | 潜在变量中固定大小的三维块；DiT 的输入。 |
| Factorized attention | “Spatial + temporal” | 先在空间上做注意力，再在时间上做注意力；跳过完整 3D 注意力。 |
| Image-to-video（I2V） | “Animate this photo” | 模型接收一张图像 + 文本，输出从该图像开始的视频。 |
| Keyframe conditioning | “Anchor frames” | 固定特定帧以控制视频整体走向。 |
| Motion brush | “Directional hint” | 用户在图像上绘制运动向量的 UI 输入。 |
| Re-captioning | “Dense captions” | 用 LLM 为训练片段重新标注详细提示词。 |
| Flicker | “Temporal artifact” | 帧间不一致；通过耦合去噪解决。 |

## 生产提示：视频潜在变量是内存带宽问题

一段 10 秒、1080p、24fps 的视频共有 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 原始像素。经过 4× 视频 VAE 压缩（`2× 空间 × 2× 时间`）后，每次请求的潜在变量约 100 MB。再用时空 DiT 跑 30 步、batch 为 1，每步要在 HBM 中搬运约 3 GB 数据——瓶颈是内存带宽，而非 FLOPs。

三个生产级调优手段，均直接来自生产推理文献的推理章节：

- **DiT 上的张量并行（TP）。** 文生视频模型通常 ≥10B 参数。4 张 H100 上做 TP=4 是标配；405B 级别模型则用 PP=2 × TP=2。每步延迟大致随 TP 线性下降，直到 all-reduce 墙。
- **帧批处理 = 连续批处理。** 在生成时，视频可以看作由注意力关联起来的一批帧。连续批处理（飞行中调度）同样适用：如果模型架构允许滑动窗口生成，就可以在返回第 `t-1` 帧的同时开始渲染第 `t+1` 帧。
- **片段级预填充缓存。** 对于图生视频，首帧条件控制类似于 LLM 的提示词预填充：计算一次，在时间解码的多轮传递中复用。这实际上就是视频的 KV 缓存。

## 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) —— Sora 技术报告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) —— CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) —— HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) —— Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) —— 2025 年中期开放 SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) —— 开创性视频扩散论文。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) —— Stable Video Diffusion 的前身。
