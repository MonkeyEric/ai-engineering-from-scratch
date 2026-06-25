# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以支撑未来五年的核心思想：只用带有噪声的网页图像-标题对和一个对比损失（contrastive loss），就能把图像编码器（image encoder）和文本编码器（text encoder）对齐到同一个向量空间中。零监督标签。4 亿对数据。得到的嵌入空间（embedding space）可以做零样本分类、图文检索，并作为视觉塔（vision tower）接入 2026 年的每一个视觉-语言模型（VLM）。SigLIP 2（2025）用 sigmoid 替代了 softmax，以更低成本超越了 CLIP。本节课从 InfoNCE 的数学推导讲到 sigmoid 成对损失（sigmoid pairwise loss），并用标准库 Python 实现训练步骤。

**Type:** Build
**Languages:** Python（stdlib，InfoNCE + sigmoid loss 实现）
**Prerequisites:** Phase 12 · 01（ViT patches），Phase 7（Transformers）
**Time:** ~180 分钟

## 学习目标

- 从互信息出发推导 InfoNCE 损失，并实现一个数值稳定的向量化版本。
- 解释为什么 sigmoid 成对损失（SigLIP）能在 batch 达到 32768+ 时扩展，而无需 softmax 所要求的 all-gather 开销。
- 通过构建文本模板（`a photo of a {class}`）并在余弦相似度上取 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练的四个可调杠杆：batch size、温度（temperature）、提示模板（prompt template）、数据质量。

## 问题背景

在 CLIP 出现之前，视觉模型依赖监督学习。收集带标签数据集（ImageNet：120 万张图像、1000 个类别），训练 CNN，然后发布。标签昂贵，标签会让模型偏向标注者能达成一致的内容，而且标签无法迁移到新任务，除非进行微调（finetuning）。

而互联网上的图像-标题对超过十亿，是免费且松散的标注。一张金毛寻回犬的照片，附带 alt 文本“my dog Max in the park”，本身就包含监督信号——文字描述了图像。问题是如何把它变成可用的训练信号？

CLIP 的答案是：把图像-标题对当作匹配任务。给定 N 张图像和 N 个标题，学习把每张图像和自己的标题匹配起来，同时对抗另外 N-1 个干扰项。监督信号是“这两个东西应该在一起；其余 N-1 个不应该”。没有类别标签，没有人工标注，只有一个对比损失。

最终得到的嵌入空间能做比训练目标更多的事。ImageNet 零样本之所以有效，是因为“a photo of a cat”这个短语的嵌入会靠近那些从未被显式标注为“cat”的猫图片。这个赌注催生了 2026 年的所有 VLM。

## 核心概念

### 双塔编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，为每张图像输出一个 D 维向量。
- 文本编码器 `g`：一个小型 transformer，为每个标题输出一个 D 维向量。

两个塔都把输出归一化为单位长度。由于都是单位范数，相似度可以写成 `cos(f(x), g(y)) = f(x)^T g(y)`。

对于 N 个（图像，标题）对组成的 batch，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是可学习的温度（temperature，CLIP 初始化为 0.07，在 log 空间中学习）。

### InfoNCE 损失

CLIP 使用对行和列对称的交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正样本是它自己的标题
loss_t2i = CE(S^T, labels=identity)   # 每个标题的正样本是它自己的图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。交叉熵中的 softmax 迫使每张图像与自己的标题的相似度高于 batch 中所有其他标题。这些“负样本”就是 batch 中的其他样本。更大的 batch = 更多负样本 = 更强的信号。CLIP 在 batch size 为 32k 的条件下训练；规模很重要。

### 温度

`tau` 控制 softmax 的尖锐程度。温度低 → 分布尖锐，产生难负样本挖掘（hard negative mining）的效果。温度高 → 分布平缓，所有样本都参与贡献。CLIP 学习 `log(1/tau)`，并做裁剪以防止崩溃。SigLIP 2 固定初始 tau，改用可学习的偏置（bias）。

### 为什么 sigmoid 更易于扩展（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，你必须把每个嵌入 all-gather 到所有 replica，然后再做 softmax。通信量随 world size 呈二次增长。

SigLIP 用逐元素的 sigmoid 替换 softmax：对于每一对 `(i, j)`，损失是一个二分类问题——“这两个是匹配对吗？”正对标签在对角线上，其余都是负对。损失公式为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当 `i == j`，否则为 0。每一对的损失相互独立，无需 all-gather。每个 GPU 计算自己的局部块然后求和。SigLIP 2 可以以很低成本把 batch 扩展到 32k–512k，而 CLIP 则需要成比例地增加通信。

### 零样本分类

给定 N 个类别名称，为每个类别构建一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板，用图像编码器嵌入待分类图像。在余弦相似度上取 argmax 即得预测类别。无需在目标类别上训练。

提示模板（prompt template）很重要。CLIP 原始论文每个类别使用 80 个模板（普通、艺术、照片、绘画等）并平均嵌入，ImageNet 提升 3 个百分点。现代用法通常只选一个或两个模板。

### 线性探测与微调

零样本只是基线。线性探测（linear probe，在冻结的 CLIP 特征之上训练一个线性层）在域内任务上通常优于零样本。完整微调（full finetuning）在域内又优于线性探测，但可能损害零样本迁移能力。三种方案，三种权衡。

### SigLIP 2：NaFlex 与稠密特征

SigLIP 2（2025）增加了：
- NaFlex：单个模型处理任意长宽比和分辨率。
- 更强的稠密特征（dense features），用于分割和深度估计，目标是作为 VLM 中冻结的骨干网络。
- 多语言：训练数据覆盖 100 多种语言，而 CLIP 仅支持英语。
- 1B 参数规模，CLIP 最高只有 400M。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认的视觉塔。CLIP 仍是纯图像-文本检索任务中的默认选择，特别是当 LAION-2B 训练分布与你的查询模式匹配时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 思想相同，18 亿对数据，90% 是噪声。证明了噪声数据也能扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开源复现，多个规模，是首选的开源检查点。EVA-CLIP：基于掩码图像建模（masked image modeling）初始化，是 VLM 的强骨干。BASIC：Google 的 CLIP+ALIGN 混合体。它们都属于同一个家族，只是数据和调参不同。

### 零样本天花板

CLIP 类模型的 ImageNet 零样本准确率上限约为 76%（CLIP-G、OpenCLIP-G）。要继续提升，要么需要更多数据（SigLIP 2 达到 80%+），要么需要架构改动（监督头、更多参数）。这个基准正在饱和；真正的价值在于下游 VLM 所消耗的嵌入空间。

## 动手实现

`code/main.py` 实现了：

1. 一个玩具级双塔编码器（基于哈希的图像特征、基于字符的文本特征），让你无需 numpy 就能看到 InfoNCE 的结构。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定）。
3. 用于对比的 sigmoid 成对损失。
4. 零样本分类流程：对一组文本提示计算余弦相似度，取 argmax 得到预测。

运行它并观察损失曲线。绝对数值是玩具级别，但曲线形状与真实 CLIP 训练器一致。

## 交付成果

本节课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径传入）和目标类别列表，它使用 CLIP 模板构建文本提示，用指定检查点（例如 `openai/clip-vit-large-patch14`）对两侧进行嵌入，并返回 top-1 / top-5 预测及相似度分数。该技能不会对提示列表之外的类别做出断言。

## 练习题

1. 手动为 4 对样本实现 InfoNCE。构造 4×4 相似度矩阵，运行 softmax，取出对角线，计算交叉熵。用这个手算结果验证你的 Python 实现。

2. SigLIP 除了温度外还使用一个偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 中存在严重类别不平衡（每行负样本远多于正样本）时，`b` 起什么作用？请阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 为猫 vs 狗构建一个零样本分类器。尝试两个提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成是否优于单个模板？

4. 计算在 512 张 GPU、batch 32k 的情况下，softmax InfoNCE 与 sigmoid 成对损失的通信成本。哪个是 O(N) 扩展，哪个是 O(N²)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放定律论文（arXiv:2212.07143，Cherti 等）。从图中复现他们关于数据缩放的结论：在模型大小固定时，ImageNet 零样本准确率与训练数据量之间存在怎样的对数-线性关系？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| InfoNCE | “对比损失” | 在 batch 的相似度矩阵上做交叉熵；每个样本的正样本是它的配对项，负样本是其余所有项 |
| Sigmoid 损失 | “SigLIP 损失” | 逐对二分类交叉熵；没有 softmax，没有 all-gather，在分布式训练中扩展成本更低 |
| 温度（Temperature） | “tau” | 在 softmax/sigmoid 前缩放 logits 的标量，控制分布的尖锐程度 |
| 零样本（Zero-shot） | “无需微调的分类” | 用文本提示构建类别嵌入，通过余弦相似度分类；不在目标类别上训练 |
| 提示模板（Prompt template） | “a photo of a ...” | 围绕类别名的文本脚手架；会影响 1-5 个百分点的零样本准确率 |
| 双塔编码器（Dual encoder） | “Two-tower” | 一个图像编码器 + 一个文本编码器，输出到共享的 D 维空间 |
| 难负样本（Hard negative） | “难干扰项” | 与正样本足够相似的负样本，模型必须努力才能把它区分开 |
| 线性探测（Linear probe） | “冻结 + 一层” | 只在冻结特征之上训练一个线性分类器；用于衡量特征质量 |
| NaFlex | “原生灵活分辨率” | SigLIP 2 的能力，可在不调整尺寸的情况下接受任意长宽比和分辨率的图像 |
| 温度缩放（Temperature scaling） | “log 参数化 tau” | CLIP 用 `log(1/tau)` 参数化，使梯度行为更稳定；并裁剪防止 tau 崩溃到接近 0 |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用噪声网页数据扩展。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 缩放定律。
