# 视觉 Transformer（Vision Transformer, ViT）

> 图像是补丁（patch）组成的网格，句子是词元（token）组成的网格。同一个 Transformer 通吃两者。

**类型：** 动手实践
**语言：** Python
**前置知识：** 阶段 7 · 05（完整 Transformer）、阶段 4 · 03（CNN）、阶段 4 · 14（视觉 Transformer 入门）
**时长：** 约 45 分钟

## 问题背景

2020 年之前，计算机视觉（computer vision）几乎等同于卷积（convolution）。ImageNet、COCO 和检测基准上的每个最优模型（SOTA）都使用 CNN 骨干网络（CNN backbone）。Transformer 则是自然语言处理的专属。

Dosovitskiy 等人（2020）在《An Image is Worth 16x16 Words》中证明，完全可以去掉卷积。把图像切成固定大小的补丁（patch），将每个补丁线性投影（linearly project）为嵌入（embedding），再把序列喂给标准的 Transformer 编码器（transformer encoder）。在足够大的规模下（ImageNet-21k 预训练或更大），ViT 能达到或超过基于 ResNet 的模型。

ViT 开启了一种更广泛的范式：一种架构，多种模态（modality）。Whisper 将音频词元化（tokenize），ViT 将图像补丁化，机器人有动作词元，视频有像素词元。Transformer 并不关心输入类型——给它序列，它就能学习。

到 2026 年，ViT 及其衍生模型（DeiT、Swin、DINOv2、ViT-22B、SAM 3）已主导视觉领域。CNN 仍在边缘设备和延迟敏感任务中占优，其他场景基本都有 ViT 的身影。

## 核心概念

![图像 → 补丁 → 词元 → Transformer](../assets/vit.svg)

### 步骤 1 — 补丁化（patchify）

将一张 `H × W × C` 的图像拆成 `N × (P·P·C)` 的扁平补丁序列。典型配置：`224 × 224` 图像、`16 × 16` 补丁 → 196 个补丁，每个 768 维。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

补丁尺寸是杠杆。更小的补丁 = 更多词元、更高分辨率、二次增长的注意力成本。更大的补丁 = 更粗糙、更便宜。

### 步骤 2 — 线性嵌入（linear embedding）

用一个可学习的矩阵把每个扁平补丁投影到 `d_model`。这等价于卷积核大小为 `P`、步幅为 `P` 的卷积。在 PyTorch 里就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——两行代码实现。

### 步骤 3 — 前置 `[CLS]` 词元并添加位置嵌入

- 前置一个可学习的 `[CLS]` 词元（token）。它的最终隐藏状态就是用于分类的图像表示。
- 添加可学习的位置嵌入（positional embedding）（ViT 原版），或二维正弦位置编码（后续变体）。
- 2024 年以后，旋转位置编码（RoPE, Rotary Position Embedding）扩展到二维，有时不再使用显式嵌入。

### 步骤 4 — 标准 Transformer 编码器

堆叠 L 个 `LayerNorm → 自注意力（Self-Attention） → 残差连接 → LayerNorm → MLP → 残差连接` 块。与 BERT 完全相同，没有视觉专属层。这是论文的教学要点。

### 步骤 5 — 预测头（head）

分类任务：取 `[CLS]` 隐藏状态 → 线性层 → softmax。DINOv2 或 SAM 则丢弃 `[CLS]`，直接使用补丁嵌入（patch embedding）。

### 重要变体

| 模型 | 年份 | 改动 |
|-------|------|--------|
| ViT | 2020 | 原始模型。固定补丁大小，全局注意力。 |
| DeiT | 2021 | 知识蒸馏（distillation）；仅在 ImageNet-1k 上即可训练。 |
| Swin | 2021 | 分层（hierarchical）结构 + 移位窗口（shifted window）。固定亚二次成本。 |
| DINOv2 | 2023 | 自监督（无需标签）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 220 亿参数；缩放定律（scaling law）生效。 |
| SigLIP | 2023 | ViT + 语言编码器对，使用 sigmoid 对比损失。 |
| SAM 3 | 2025 | 任意分割；ViT-Large + 可提示掩码解码器。 |

### 为什么花了这么长时间

ViT 需要*大量*数据才能追平 CNN，因为它缺乏 CNN 的归纳偏置（inductive bias），如平移不变性（translation invariance）和局部性（locality）。没有超过 1 亿张标注图像或强自监督预训练时，CNN 在同等算力下仍然更优。DeiT 在 2021 年通过蒸馏技巧解决了这个问题；DINOv2 在 2023 年通过自监督永久解决了这个问题。

## 动手实现

参见 `code/main.py`。使用纯标准库实现补丁化、线性嵌入和正确性检查，不涉及训练——任何现实规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### 步骤 1：虚拟图像

一个 24 × 24 的 RGB 图像，表示为由行组成的列表，每行是 `(R, G, B)` 元组。使用 6×6 补丁 → 16 个补丁，每个 108 维的嵌入向量。

### 步骤 2：补丁化（patchify）

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：网格按行优先遍历。所有 ViT 都采用这种顺序。

### 步骤 3：线性嵌入

将每个扁平补丁乘以一个随机 `(patch_flat_size, d_model)` 矩阵。验证前置 `[CLS]` 后的输出形状为 `(N_patches + 1, d_model)`。

### 步骤 4：统计现实规模 ViT 的参数量

打印 ViT-Base 的参数量：12 层、12 头、d=768、补丁=16。与 ResNet-50（约 2500 万）对比，ViT-Base 约 8600 万，ViT-Large 约 3.07 亿，ViT-Huge 约 6.32 亿。

## 使用现成模型

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # 图像表示
```

**到 2026 年，DINOv2 嵌入是图像特征的默认选择。** 冻结骨干网络（backbone），只训练一个极小的头。这适用于分类、检索、检测、图像描述（captioning）。Meta 的 DINOv2 检查点（checkpoint）在所有非文本视觉任务上都优于 CLIP。

**补丁尺寸选择。** 小型模型用 16×16（ViT-B/16）。稠密预测任务（分割）用 8×8 或 14×14（SAM、DINOv2）。超大模型通常用 14×14。

## 交付应用

参见 `outputs/skill-vit-configurator.md`。该 skill 会根据数据集大小、分辨率和算力预算，为新视觉任务选择合适的 ViT 变体和补丁尺寸。

## 练习

1. **简单。** 运行 `code/main.py`。验证补丁数量等于 `(H/P) * (W/P)`，扁平补丁维度等于 `P*P*C`。
2. **中等。** 实现二维正弦位置嵌入——为每个补丁的 `row` 和 `col` 分别生成独立正弦编码并拼接。把它输入一个微型 PyTorch ViT，在 CIFAR-10 上与可学习位置嵌入对比准确率。
3. **困难。** 用 PyTorch 构建一个 3 层 ViT，在 1000 张 MNIST 图像上用 4×4 补丁训练，测量测试准确率。然后加入 DINOv2 预训练（简化版：仅训练编码器从掩码补丁预测补丁嵌入），准确率是否提升？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| 补丁（Patch） | “视觉 Transformer 的词元” | 图像中某个 `P × P × C` 区域像素值的扁平向量。 |
| 补丁化（Patchify） | “切块 + 展平” | 把图像切成不重叠的补丁，每个展平为向量。 |
| `[CLS]` 词元 | “图像摘要” | 前置的可学习词元；其最终嵌入即为图像表示。 |
| 归纳偏置（Inductive bias） | “模型假设了什么” | ViT 比 CNN 的先验更少，需要更多数据弥补。 |
| DINOv2 | “自监督 ViT” | 无需标签，通过图像增强 + 动量教师训练。2026 年最佳通用图像特征。 |
| SigLIP | “CLIP 的继任者” | 用 sigmoid 对比损失训练的 ViT + 文本编码器；同等算力下优于 CLIP。 |
| Swin | “窗口化 ViT” | 分层 ViT，局部注意力 + 移位窗口；亚二次复杂度。 |
| Register tokens | “2023 年的技巧” | 少量额外可学习词元，吸收注意力汇点（attention sink）；提升 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) —— ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) —— DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) —— Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) —— DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) —— DINOv2 的 register token 修正。
