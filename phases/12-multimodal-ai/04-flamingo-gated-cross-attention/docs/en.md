# Flamingo 与门控交叉注意力：少样本视觉语言模型

> DeepMind 的 Flamingo（2022）率先做到了两件事。它证明单一模型可以处理任意交错的图像、视频和文本序列；同时证明视觉语言模型（VLM）可以进行上下文学习——只需在提示词中给出三个示例（图像， caption），模型就能为新图像生成 caption，无需任何梯度更新。其核心机制是门控交叉注意力层：在冻结的 LLM 已有层之间插入新层，并通过一个可学习的 tanh 门控进行初始化，初始值为零，从而在初始化时完整保留 LLM 的文本能力。本课将详细讲解 Flamingo 的 Perceiver 重采样器（Perceiver resampler）和门控交叉注意力架构——它们是 Gemini 交错输入与 Idefics2 视觉 token 的先驱。

**Type:** 学习
**Languages:** Python（标准库，门控交叉注意力 + Perceiver resampler 演示）
**Prerequisites:** Phase 12 · 03（BLIP-2 Q-Former）
**Time:** 约 120 分钟

## 学习目标

- 解释门控交叉注意力如何通过 `tanh(gate) = 0` 在初始化时保留冻结 LLM 的文本能力。
- 逐步理解 Perceiver resampler：通过交叉注意力将 N 个图像 patch 映射为 K 个固定的“潜在（latent）”查询。
- 描述 Flamingo 如何处理交错的图像-文本序列，并使用因果掩码（causal masking）尊重图像在序列中的位置。
- 复现少样本多模态提示词结构（3 个图像-caption 示例，后跟一张查询图像）。

## 问题背景

BLIP-2 将 32 个视觉 token 输入冻结 LLM 的输入层，适用于每张提示词只有一张图像的场景。但如果你想输入多张与文本交错排列的图像，例如“这是图像 A，描述它；这是图像 B，描述它；现在这是图像 C，描述它”呢？此时 LLM 的自注意力需要在同一序列中同时处理图像 token 和文本 token，而“哪些位置可以 attend 到哪些图像”这一问题会变得非常棘手。

Flamingo 的解决方案是：完全不改变 LLM 的输入流。在现有 LLM 块之间插入额外的交叉注意力层。文本 token 仍然像以往一样通过 LLM 的因果自注意力流动。每隔几个 LLM 块，文本 token 还会通过一个新增的门控层交叉 attend 到图像特征。门控初始化为零，意味着在训练第 0 步时这些新层是“无操作（no-op）”的——模型行为与预训练 LLM 完全一致。随着训练推进，门控逐渐打开，视觉信息开始流入。

Flamingo 回答的第二个问题是：如何处理每张提示词中可变数量的图像（0、1 或多张）？答案是 Perceiver resampler——一个小的交叉注意力模块，接收任意数量的 patch，输出固定数量的视觉潜在 token。无论提示词中有多少张图像，LLM 的交叉注意力层看到的输入形状始终相同。

## 核心概念

### 冻结的 LLM

Flamingo 以冻结的 Chinchilla 70B LLM 为起点。全部 700 亿参数保持不变。原有的文本自注意力与 FFN 正常运作。

### Perceiver resampler

对于提示词中的每张图像，ViT 会生成 N 个 patch token。Perceiver resampler 拥有 K 个固定的可学习潜在向量（Flamingo 使用 K=64）。每个 resampler 块包含两个子步骤：

1. 交叉注意力：K 个潜在向量 attend 到 N 个 patch token（Q 来自潜在向量，K/V 来自 patch）。
2. 潜在向量之间的自注意力 + FFN。

经过 6 个 resampler 块后，输出为 K=64 个维度 1024 的视觉 token，与 ViT 产生了多少 patch 无关。224×224 的图像（196 个 patch）和 480×480 的图像（900 个 patch）都会输出 64 个 resampler token。

对于视频，resampler 会按时间帧应用：每一帧的 patch 产生 64 个潜在向量，并通过时间位置编码（temporal positional encoding）让模型区分 t=0 与 t=N。整段视频最终变为 T × 64 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个可学习标量，初始化为 0。
- `tanh(0) = 0`，因此初始化时门控分支贡献为零。
- 随着 `alpha` 偏离零，交叉注意力的贡献平滑增加。
- 残差连接意味着即使门控完全打开，也不会覆盖 LLM 的文本表示；它只是在文本表示之上叠加视觉信息。

这是 Flamingo 中最重要的设计选择：视觉条件化是加性的、门控的、并在初始化时为零。在第 0 步，Flamingo 在纯文本输入上就是一台完美的 Chinchilla 70B。

### 交错输入的掩码交叉注意力

在类似“<image A> caption A <image B> caption B <image C> ?”的提示词中，每个文本 token 应该只能看到序列中它之前出现的图像。交叉注意力掩码强制规定：位置 `t` 处的文本 token 只能 attend 到图像索引 `i < i_t` 的图像 resampler token，其中 `i_t` 是位置 `t` 之前最近的一张图像。“只能看到最近的前一张图像”或“能看到所有前面的图像”都是可行选择；Flamingo 选择了前者。

### 上下文少样本学习

Flamingo 的提示词看起来像这样：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到这一补全模式后会输出“bird”（或 image3 所展示的内容）。无需梯度更新。冻结 LLM 的上下文学习能力通过门控交叉注意力得以保留——这正是论文的核心结论，也是其重要意义所在。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：4300 万个图文交错的网页，按阅读顺序重建。
2. 图像-文本对（ALIGN + LTIP）：44 亿对。
3. 视频-文本对（VTP）：2700 万个短视频片段。

OBELICS（2023）是该交错网页语料库的开源复现版本，Idefics、Idefics2 以及大多数开源“类 Flamingo”模型都在其上训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开源复现版本。架构完全相同（Perceiver resampler + 在冻结的 LLaMA 或 MPT 上的门控交叉注意力）。提供 3B、4B、9B 检查点。由于基础 LLM 更小、数据更少，效果落后于 Flamingo。

Otter（2023）基于 OpenFlamingo，并在 MIMIC-IT（一个多模态指令数据集）上进行指令微调，证明门控交叉注意力同样适用于指令跟随场景。

### 后续演进

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力系列， progressively 更简单（Idefics2 放弃 resampler，改用直接 patch token 加自适应池化）。
- Flamingo 到 Chameleon 的过渡：到 2024 年，许多团队转向早期融合（early-fusion，见第 12.11 课）；但在需要冻结主干的场景中，Flamingo 式门控交叉注意力仍在生产中使用。
- Gemini 的交错输入：概念上继承了 Flamingo 的交错格式灵活性，但具体机制未公开。

### 与 BLIP-2 的对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥接 | 输入层使用一次 Q-Former | 每 M 层使用门控交叉注意力 |
| 视觉 token | 每张图像 32 个 | 每次交叉注意力每层每张图像 64 个 |
| LLM 是否冻结 | 是 | 是 |
| 少样本上下文学习 | 较弱 | 强——论文的核心亮点 |
| 交错输入 | 原生不支持 | 支持，是设计的核心目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错网页 |
| 训练参数量 | 1.88 亿 | 约 100 亿（交叉注意力层） |
| 计算资源 | 8 张 A100 训练数天 | 数千块 TPUv4 训练数周 |

预算有限且只需单图 VQA 时选 BLIP-2。需要交错输入、少样本或多图推理时选 Flamingo / Idefics2。

## 动手实践

`code/main.py` 演示了以下内容：

1. 在 36 个伪造 patch token 上使用 8 个可学习潜在向量的 Perceiver resampler（纯 Python 交叉注意力实现）。
2. 门控交叉注意力步骤：`alpha = 0` 时输出等于输入（LLM 不变），然后 `alpha = 2.0` 时视觉贡献被混合进来。
3. 一个交错掩码构建器，为“（图像 1）（文本 1）（图像 2）（文本 2）”序列生成二维注意力掩码。

## 交付成果

本课会生成 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开源 VLM 的配置（是否使用 resampler、交叉注意力频率、门控方案），它会识别其中的 Flamingo 血统元素并解释冻结策略。有助于调试微调后文本性能下降的原因（答案：门控开得太快太宽）。

## 练习题

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 1.4B 门控交叉注意力层 + 6400 万 resampler。训练参数占总参数的比例是多少？

2. 在 PyTorch 中实现门控残差 `y = tanh(alpha) * cross + x`。通过实验说明当 `alpha=0` 时，初始化阶段 `y==x` 严格成立。

3. 阅读 OpenFlamingo 论文第 3.2 节（arXiv:2308.01390），了解当同一 batch 中每个提示词的图像数量不同时，他们如何处理多张图像。描述其 padding 策略。

4. 为什么 Flamingo 的交叉注意力掩码只允许文本 token attend 到**最近的**前一张图像，而不是所有前面的图像？阅读 Flamingo 论文第 2.4 节并解释其中的权衡。

5. 上下文少样本：为一种新的 Flamingo 变体构造 4 个“图像 → 主要物体颜色”的示例提示词。描述当示例数量从 0 变化到 8 时，预期准确率的变化模式。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| Perceiver resampler | “固定潜在交叉注意力” | 将可变数量的输入 patch 映射为 K 个固定 token 的模块 |
| Gated cross-attention | “Tanh 门控桥接” | 残差层 `y = tanh(alpha)*cross + x`，可学习 alpha，初始化为 0 |
| Interleaved input | “混合序列” | 图像与文本按阅读顺序自由混合的提示词格式 |
| Frozen LLM | “LLM 无梯度” | 文本 LLM 的权重不更新，仅训练 resampler 与交叉注意力层 |
| Few-shot | “上下文示例” | 在提示词中给出少量（图像，答案）对，模型无需微调即可泛化 |
| OBELICS | “交错网页语料库” | 1.41 亿个按阅读顺序包含图像和文本的网页开源数据集 |
| Chinchilla | “70B 冻结基座” | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| Gate schedule | “alpha 如何变化” | 训练过程中交叉注意力门控打开的速度 |
| Cross-attn frequency | “每 M 层一次” | 门控交叉注意力块的插入频率；Flamingo 使用 M=4 |
| OpenFlamingo | “开源复现” | MosaicML/LAION 发布的 3-9B 开源检查点；架构与 Flamingo 相同 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料库。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 经指令微调的 Flamingo 后继模型。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化版本。
