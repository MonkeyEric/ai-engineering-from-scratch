# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实图像并不是 224×224 的方块。收据可能是 9:16，图表可能是 16:9，医学扫描可能是 4096×4096，手机截图可能是 9:19.5。2024 年之前的视觉语言模型（VLM）解决方案——把所有东西 resize 成固定正方形——丢弃了让 OCR、文档理解和高分辨率场景解析真正有效的信号。NaViT（Google，2023）展示了你可以把可变分辨率的图像块打包进单个 Transformer 批次，并通过块对角掩码（block-diagonal masking）实现隔离。Qwen2-VL 的 M-RoPE（2024）则彻底抛弃了绝对位置表。LLaVA-NeXT 的 AnyRes 把高分辨率图像切分成基础图 + 子图。SigLIP 2 的 NaFlex 变体（2025）如今已成为希望单一检查点服务所有宽高比的开放 VLM 的默认编码器。本节课从头实现 patch-n'-pack。

**Type:** Build  
**Languages:** Python（标准库，图像块打包器 + 块对角掩码）  
**Prerequisites:** Phase 12 · 01（ViT 图像块），Phase 12 · 05（LLaVA）  
**Time:** 约 120 分钟

## 学习目标

- 将一批不同分辨率图像的图像块打包成一个序列，并构建块对角注意力掩码。
- 针对给定任务，在 AnyRes 瓦片化（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做出选择。
- 在不 resize 的前提下，为 OCR、图表和摄影类图像计算令牌预算。
- 说出正方形 resize 的三种失效模式：文字被压扁、内容被裁剪、填充（padding）浪费令牌。

## 问题所在

Transformer 期望输入是一个序列。一个批次就是若干等长序列的堆叠。如果你的图像都是 224×224，那么每次都会得到 196 个 patch token，不需要 padding，任务完成。训练用 224，推理也用 224，再也不用考虑分辨率。

现实世界不会配合。文档是竖版的（8.5×11 英寸，约 2:3）。图表截图是横版的（16:9）。收据又窄又高（1:3）。医学影像常以 2048×2048 或更高分辨率输出。移动设备截图是 1170×2532（0.46:1）。

2024 年前的三种做法以及各自的问题：

1. **resize 到固定正方形**（224×224 或 336×336）。挤压会扭曲文字和人脸；下采样会破坏图表标签和 OCR 内容。在 LLaVA-1.5 之前这是标准做法。
2. **裁剪到固定宽高比**。你会丢弃图像的大部分内容，而选择裁剪位置本身就是另一个视觉问题。
3. **padding 到最长边**。避免了扭曲，但对于竖版图像会浪费 50% 以上的令牌在 padding 上。并且注意力要在所有这些 pad token 上付出二次方成本。

2024-2025 年的答案：让 Transformer 直接按图像原始分辨率吃图像块，并想办法把异构批次打包进一个序列而不浪费计算。

## 核心概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani 等，2023）是首篇在大规模上证明这条路可行的论文。思路很机械：

1. 对批次中的每张图像，按选定的图像块大小（例如 14）计算其原始图像块网格。
2. 把每张图像的图像块展平成可变长度的序列。
3. 把所有图像的序列拼接成一个长序列。
4. 构建块对角注意力掩码，使图像 A 的图像块只与图像 A 内部交互。
5. 携带每个图像块的二维位置信息（2D RoPE 或分数位置编码）。

一个包含三张图像的批次——336×336（576 token）、224×224（256 token）、448×336（768 token）——会变成一个 1600 token 的序列，并带有一个 1600×1600 的块对角掩码。没有 padding，没有浪费计算，Transformer 可以处理任意宽高比。

NaViT 还引入了训练时的**分数图像块丢弃**（fractional patch dropping）——随机丢弃批次中 50% 的图像块——既起到正则化作用，也加快了训练速度。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是一种务实的替代方案。给定一张高分辨率图像和一个固定编码器（CLIP 或 SigLIP，336 尺寸），对图像进行瓦片化：

1. 从预定义的网格集合中选择最符合图像宽高比的布局——(1×1)、(1×2)、(2×1)、(1×3)、(3×1)、(2×2) 等。
2. 把整张图像切分成该网格；每个瓦片变成 336×336 的裁剪图。
3. 同时生成一张缩略图：把整图 resize 到 336×336，作为全局上下文 token。
4. 每个瓦片都经过冻结的 336 编码器。把瓦片 token 和缩略图 token 拼接起来。

一张 672×672 的图像使用 2×2 网格加缩略图：4 × 576 + 576 = 2880 个视觉 token。代价高昂但有效——大语言模型既能看到局部细节，也能看到全局上下文。

当你的编码器被冻结且只支持单一分辨率时，AnyRes 是首选方案。但它会让大图像的 token 数量爆炸（1344×1344 的图像用 4×4 网格是 9216 + 576 ≈ 9800 token，几乎占满一个 8k LLM 上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了**多模态旋转位置编码**（Multimodal Rotary Position Embedding，M-RoPE）。与 NaViT 的分数位置或 AnyRes 的瓦片+缩略图不同，每个图像块携带一个三维位置（时间、高度、宽度）。查询/键（query/key）的旋转会处理任意 H、W 和时间长度。

M-RoPE 原生支持动态分辨率，无需重新训练。推理时你输入任意 H×W 的图像，图像块嵌入器会输出 H/14 × W/14 个 token，每个 token 得到 (t=0, r=行, c=列) 的位置，RoPE 用正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 延续了这一设计。InternVL3 的 V2PE 是同一思想，只是按模态使用可变编码。

与 AnyRes 不同，M-RoPE 在原始分辨率下是 O(H × W / P²) 个 token——没有乘法级的瓦片开销。与 NaViT 不同，它每次前向仍然只处理单张图像。跨分辨率批次仍然需要在之上做 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 检查点的 native-flex 模式。单个模型在推理时服务多种序列长度（256、729、1024 token）。内部它在训练时使用 NaViT 风格的 patch-n'-pack，并为每个图像块使用绝对分数位置。卖点是：一个检查点，根据任务在推理时选择你的令牌预算。

语义任务（分类、检索）用 256 token。OCR 或图表理解用 1024 token。无需重新训练。

### 打包掩码

块对角掩码是大多数实现容易绊倒的地方。对于总长度为 `N_total`、覆盖图像 `i=0..B-1`、各图像长度为 `n_i` 的打包序列，掩码 `M` 的形状为 `(N_total, N_total)`，当两个下标落在同一张图像的块内时值为 1，否则为 0。你可以用累积长度列表来构建它：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 当且仅当存在某个 b，使得 offsets[b] <= i < offsets[b+1] 且 offsets[b] <= j < offsets[b+1]
```

在 PyTorch 里，用 `torch.block_diag` 一行就能实现，或者通过显式 gather。FlashAttention 的可变长度路径（`cu_seqlens`）则完全跳过密集掩码，直接利用累积长度张量在各序列内部做注意力——对于典型批次，比密集掩码快约 10 倍。

### 令牌预算

按任务选择策略：

- **OCR / 文档**：1024-4096 token。SigLIP 2 NaFlex 用 1024，或 AnyRes 3×3 + 缩略图。
- **图表与 UI**：384-448 原始分辨率下 729-1024 token。Qwen2.5-VL 动态分辨率并设置最大像素上限。
- **自然照片**：256-576 token 就够了。下游 LLM 能捕捉到足够信息。在内容密度高的地方才为 token 付费。
- **视频**：空间池化后每帧 64-128 token，帧率 2-8 FPS。第 12.17 课会专门讲视频。

2026 年的生产规则：为每个任务设置一个最大像素上限，按原生宽高比编码到该上限，打包批次，并跳过 padding。Qwen2.5-VL 暴露的 `min_pixels` 和 `max_pixels` 正是为了这个旋钮。

## 动手实践

`code/main.py` 使用整数像素坐标，为一组异构分辨率的图像实现了 patch-n'-pack。它会：

- 接收一个 (H, W) 图像尺寸列表。
- 在图像块大小 14 下计算每张图像的 patch 序列长度。
- 把它们打包成一个总长度为 `sum(n_i)` 的序列。
- 构建块对角注意力掩码（密集形式，便于理解）。
- 对比打包成本与正方形 resize 和 AnyRes 瓦片化的成本。
- 为混合批次（收据、图表、截图、照片）打印一张令牌预算表。

运行它。得出的数字正是 2026 年每个开放 VLM 都在使用 patch-n'-pack 的原因。

## 交付成果

本节课产出 `outputs/skill-resolution-budget-planner.md`。给定一个混合宽高比的工作负载（OCR、图表、照片、视频帧）和总令牌预算，它会选择正确的策略（NaFlex、AnyRes、M-RoPE 或固定正方形），并输出每个请求的配置。当你为产品选型 VLM 时使用这项技能——它能避免那种会吞掉延迟预算的隐性 10 倍 token 暴涨。

## 练习题

1. 一张收据为 600×1500（1:2.5）。在图像块大小 14 下，原始分辨率有多少个 token？resize 到 336 正方形后有多少个？实际中哪种会损失更多 OCR 精度？

2. 为一个包含长度 256、576、729、1024 的四图像批次构建块对角掩码。验证注意力矩阵是 2585×2585，且非零条目数恰好为 `256² + 576² + 729² + 1024²`。

3. 对于一张 1792×896 的图像，图像块大小 14，比较：(a) resize 到 336 再编码，(b) AnyRes 2×1 + 缩略图，(c) M-RoPE 原生分辨率。哪种用的 token 最少？哪种保留的细节最多？

4. 实现分数图像块丢弃：给定一个打包序列，均匀随机丢弃 50% 的 token，并相应更新块对角掩码。测量掩码稀疏度的变化。

5. 阅读 Qwen2-VL 论文的 3.2 节（arXiv:2409.12191）。用两句话说明 `min_pixels` 和 `max_pixels` 控制什么，以及为什么上下界都很重要。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------|
| Patch-n'-pack | "NaViT-style packing" | 把不同图像的可变长度图像块序列拼接进同一个批次维度 |
| Block-diagonal mask | "Packing mask" | 注意力掩码，限制每张图像的图像块只能与自己交互，不能 attend 到打包中的邻居 |
| AnyRes | "LLaVA-NeXT tiling" | 把高分辨率图像切分成固定大小的瓦片网格，再加一张全局缩略图；每个瓦片都用固定编码器编码 |
| NaFlex | "SigLIP 2 native-flex" | 单个 SigLIP 2 检查点，在推理时无需重新训练即可服务 256/729/1024 token 预算 |
| M-RoPE | "Multimodal RoPE" | 三维旋转位置编码（时间、行、列），无需位置表即可处理任意 H、W、T |
| cu_seqlens | "FlashAttention packing" | 累积长度张量，FlashAttention 可变长度路径用它替代密集块对角掩码 |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL 的每个请求旋钮，用于限制极小或极大输入上的 token 数量 |
| Visual token budget | "How many tokens per image" | 每张图像发出的 patch token 粗略计数；决定 LLM 的提示预算和注意力成本 |

## 延伸阅读

- [Dehghani 等 — Patch n' Pack: NaViT（arXiv:2307.06304）](https://arxiv.org/abs/2307.06304)
- [Wang 等 — Qwen2-VL（arXiv:2409.12191）](https://arxiv.org/abs/2409.12191)
- [Laurençon 等 — What matters when building vision-language models?（Idefics2，arXiv:2405.02246）](https://arxiv.org/abs/2405.02246)
- [Tschannen 等 — SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report（arXiv:2502.13923）](https://arxiv.org/abs/2502.13923)
