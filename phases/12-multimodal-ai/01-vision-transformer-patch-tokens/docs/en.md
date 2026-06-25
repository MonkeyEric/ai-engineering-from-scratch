# 视觉 Transformer 与 Patch Token 原语

> 在进入任何多模态（multimodal）系统之前，图像必须先变成 Transformer 能够消费的 token 序列。2020 年的 ViT 论文给出的答案是：16×16 像素的图像块（patch）、线性投影（linear projection）和位置嵌入（position embedding）。五年后的 2026 年，每一款前沿模型（Claude Opus 4.7 原生 2576px、Gemini 3.1 Pro、Qwen3.5-Omni）仍然从这里开始——编码器从 ViT 演进到了 DINOv2、再到了 SigLIP 2，register token 被加了进来，位置方案变成了 2D-RoPE，但这一原语始终未变。本节课从头到尾解读 patch-token 流程，并用标准库 Python 把它实现出来，让 Phase 12 的后续内容对“视觉 token”有一个具体的心智模型。

**Type:** 学习  
**Languages:** Python（stdlib，图像块分词器 + 几何计算器）  
**Prerequisites:** Phase 7（Transformers）、Phase 4（Computer Vision）  
**Time:** ~120 分钟

## 学习目标

- 将一张 H×W×3 的图像转换为带正确位置编码的 patch token 序列。
- 针对给定的（patch size、resolution、hidden dim、depth）ViT 配置，计算序列长度、参数量和前向 FLOPs。
- 说出让 ViT 从 2020 年研究走向 2026 年生产环境的三大升级：自监督预训练（DINO / MAE）、register token 和原生分辨率打包（native-resolution packing）。
- 在下游任务中选择使用 CLS 池化、均值池化（mean pooling）还是 register token。

## 问题

Transformer 操作的是向量序列。文本天然就是序列（字节或 token）。而图像是一个二维像素网格，带有三个颜色通道——它不是序列。如果你把每个像素都展平，一张 224×224 的 RGB 图像会变成 150,528 个 token，在这个长度上做自注意力（self-attention）完全不现实（复杂度与序列长度成平方关系）。

2020 年之前的方法通常是在前面接一个 CNN 特征提取器：ResNet 生成一个 7×7 的 2048 维特征图，把这 49 个 token 喂给 Transformer。这样可行，但继承了 CNN 的归纳偏置（平移等变性、局部感受野），也削弱了 Transformer 对规模的利用能力。

Dosovitskiy 等人（2020）提出了一个直接的问题：如果我们跳过 CNN 呢？把图像切分成固定大小的 patch（比如 16×16 像素），将每个 patch 线性投影（linear projection）成一个向量，加上位置嵌入（position embedding），然后把序列喂给普通的 Transformer。这在当时被视为异端——没有卷积的计算机视觉。但只要有足够的数据（JFT-300M，然后是 LAION），它就在 ImageNet 上击败了 ResNet，并且持续进步。

到 2026 年，ViT 原语已经成为毋庸置疑的基础。每个开放权重的视觉语言模型（VLM）的视觉塔（vision tower）都是它的某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是“要不要用 patch？”，而是“用多大的 patch、什么样的分辨率调度、什么样的预训练目标、什么样的位置编码”。

## 概念

### 将图像块视为 token

给定一张形状为 `(H, W, 3)` 的图像 `x` 和 patch size `P`，你可以把图像切分成 `(H/P) × (W/P)` 个互不重叠的网格。每个 patch 是一个 `P × P × 3` 的像素立方体。把每个立方体展平成一个 `3 P²` 维向量。再应用一个共享的线性投影矩阵 `W_E`，形状为 `(3 P², D)`，把每个 patch 映射到模型的隐藏维度 `D`。

以 ViT-B/16 的标准配置为例：
- 分辨率 224，patch size 16 → 网格 14×14 → 196 个 patch token。
- 每个 patch 是 `16 × 16 × 3 = 768` 个像素值，投影到 `D = 768`。
- 添加一个可学习的 `[CLS]` token → 序列长度 197。

patch 投影在数学上等价于一个二维卷积：kernel size 为 `P`、stride 为 `P`、输出通道为 `D`。生产代码就是这样实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。“线性投影”是概念视角；“卷积核”视角则更高效。

### 位置嵌入

patch 本身没有内在顺序——Transformer 把它们看作一个集合。早期 ViT 添加了可学习的 1D 位置嵌入（每个位置一个 768 维向量，共 197 个）。它有效，但会把模型绑定到训练分辨率：推理时如果网格大小改变，就必须对位置表做插值。

现代视觉骨干网络使用 2D-RoPE（如 Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或因子化 2D 位置。2D-RoPE 根据 patch 的（行，列）索引旋转查询（query）和键（key）向量，使模型从旋转角度推断出相对二维位置。不需要位置表，推理时可以处理任意网格大小。

### CLS token、池化输出与 register token

图像级别的表示是什么？目前有三种选择共存：

1. `[CLS]` token。在 patch 序列前prepend一个可学习的向量。经过所有 Transformer 块后，CLS token 的隐藏状态就是图像表示。这一做法继承自 BERT。原始 ViT 和 CLIP 使用它。
2. 均值池化（mean pooling）。对 patch token 的输出隐藏状态取平均。SigLIP、DINOv2 和大多数现代 VLM 使用它。
3. Register token。Darcet 等人（2023）观察到，没有显式 sink token 的 ViT 会产生高范数的“伪影”patch，并劫持自注意力。添加 4–16 个可学习的 register token 可以吸收这部分负担，并提升密集预测任务（分割、深度估计）的质量。DINOv2 和 SigLIP 2 都默认带有 register。

这个选择对下游任务很重要。CLS 适用于分类。对于把 patch token 喂给 LLM 的 VLM，则完全不做池化——每个 patch 都成为一个 LLM 输入 token。Register 在交给 LLM 之前会被丢弃（它们是脚手架，不是内容）。

### 预训练：监督、对比、掩码与自蒸馏

2020 年的 ViT 使用 JFT-300M 上的监督分类进行预训练，但很快就被以下方法超越：

- CLIP（2021）：在 4 亿图像-文本对上做对比学习。见第 12.02 课。
- MAE（2021，He 等人）：掩码 75% 的 patch 并重建像素。自监督，仅使用纯图像。
- DINO（2021）/ DINOv2（2023）：使用师生框架做自蒸馏，无需标签、无需 caption。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉骨干网络，也是“密集特征”场景的默认选择。
- SigLIP / SigLIP 2（2023、2025）：使用 sigmoid 损失和 NaFlex 原生长宽比的 CLIP。2026 年开放 VLM（Qwen、Idefics2、LLaVA-OneVision）的主流视觉塔。

你选择的预训练目标决定了骨干网络擅长什么：CLIP/SigLIP 用于与文本的语义匹配，DINOv2 用于密集视觉特征，MAE 可作为下游微调的起始点。

### 缩放定律

ViT 的缩放研究（Zhai 等人，2022）表明，ViT 的质量在模型大小、数据量和计算量上遵循可预测的缩放定律（scaling laws）。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- patch size 是序列长度与保真度之间的杠杆。patch 14（DINOv2/SigLIP SO400m 常见）每张图像产生比 patch 16 更多的 token；对 OCR 和密集任务更好，但速度更慢。
- 分辨率是另一个重要杠杆。从 224 提升到 384 再到 512 几乎总能带来提升，但 FLOPs 呈平方级增长。

ViT-g/14（10 亿参数，patch 14，分辨率 224 → 256 个 token）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开放 VLM 的两大主力编码器。

### ViT 的参数量

完整计算在 `code/main.py` 中。对于 224 分辨率下的 ViT-B/16：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 之前，先用这种方式估算每个 ViT 的体量。骨干网络的大小决定了任何下游 VLM 的显存（VRAM）下限。

### 2026 年生产环境配置

2026 年大多数开放 VLM 搭载的编码器是原生分辨率（NaFlex）下的 SigLIP 2 SO400m/14。它的配置如下：
- 4 亿参数。
- patch size 14，默认分辨率 384 → 每张图像 729 个 patch token。
- 图像级任务使用均值池化（mean pooling）；VQA 任务中全部 729 个 patch 都流入 LLM。
- 4 个 register token，在交给 LLM 之前被丢弃。
- 2D-RoPE，并带有图像级缩放以支持原生长宽比。

这一配置中的每一个决定都可以追溯到一篇你可以阅读的论文。

## 动手实践

`code/main.py` 是一个 patch 分词器和几何计算器。它接收（图像 H、W，patch P，隐藏维度 D，深度 L），并输出：

- patch 后的网格形状和序列长度。
- 一个合成 8×8 像素玩具图像的 token 序列（走一遍展平 + 投影路径）。
- 按 patch 嵌入、位置嵌入、Transformer 块和输出头分解的参数量。
- 目标分辨率下每次前向传播的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。把参数量与论文公布的数字对上。调整 patch size 和分辨率，感受 token 数量的代价。

## 交付成果

本节课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch size、resolution、hidden dim、depth），它会给出 token 数量、参数量和显存估算，并附带理由。每当你为 VLM 挑选视觉骨干网络时，都可以使用这项技能——它能避免“token 爆炸导致 LLM 上下文被占满”的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280×720 输入、patch size 14 时的 patch token 序列长度。与仅使用 CLS 表示相比如何？

2. 一帧 1080p 画面（1920×1080）在 patch 14 下产生多少个 token？以 30 FPS 播放 5 分钟视频，总共有多少视觉 token？哪种方式最能节省成本：池化、帧采样，还是 token 合并？

3. 用纯 Python 实现 patch token 上的均值池化（mean pooling）。验证对 DINOv2 输出的 196 个 token 做 mean-pool 的结果，与模型 `forward` 返回的池化嵌入一致。

4. 阅读《Vision Transformers Need Registers》（arXiv:2309.16588）的第 3 节。用两句话说明 register token 吸收了什么伪影，以及为什么它对下游密集预测任务很重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同分辨率的图像，生成一个打包后的单一序列和块对角注意力掩码（block-diagonal attention mask）。学到第 12.06 课时再与之对比验证。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| Patch | "16×16 像素方块" | 输入图像中固定大小的不重叠区域；成为一个 token |
| Patch embedding | "线性投影" | 一个共享的可学习矩阵（或 stride=P 的 Conv2d），把展平的 patch 像素映射到 D 维向量 |
| CLS token | "类别 token" | 前置的可学习向量，其最终隐藏状态代表整张图像；2026 年已非必需 |
| Register token | "Sink token" | 额外的可学习 token，用于吸收 ViT 在预训练中出现的高范数注意力伪影 |
| Position embedding | "位置信息" | 让每个位置具有顺序感知的向量或旋转；现代默认是 2D-RoPE |
| Grid | "Patch grid" | 给定分辨率和 patch size 下，由 `(H/P) × (W/P)` 个 patch 组成的二维数组 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的特性：单个模型无需重新训练即可服务多种长宽比和分辨率 |
| Backbone | "视觉塔" | 预训练的图像编码器，其 patch-token 输出在 VLM 中喂给 LLM |
| Pooling | "图像级摘要" | 把 patch token 变成单个向量的策略：CLS、mean、attention pool 或基于 register 的方式 |
| Patch 14 vs 16 | "更细 vs 更粗的网格" | Patch 14 每张图像产生更多 token，OCR 保真度更高但更慢；patch 16 是经典默认 |

## 延伸阅读

- [Dosovitskiy 等人 —《一张图像等价于 16×16 个词》(arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He 等人 —《掩码自编码器是可扩展的视觉学习者》(arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab 等人 — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无需标签。
- [Darcet 等人 —《Vision Transformers Need Registers》(arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 与伪影分析。
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认视觉塔。
- [Zhai 等人 —《Scaling Vision Transformers》(arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验缩放定律。
