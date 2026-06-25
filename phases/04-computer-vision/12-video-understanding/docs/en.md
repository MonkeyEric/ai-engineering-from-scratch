# 视频理解——时间建模

> 视频是一系列图像，以及连接它们的物理过程。每个视频模型要么将时间视为一个额外的轴（3D 卷积），要么视为需要关注的序列（transformer），要么视为一次性提取并池化的特征（2D+pool）。

**类型：** 学习 + 实践
**语言：** Python
**前置知识：** 第 4 阶段第 03 课（CNN）、第 4 阶段第 04 课（图像分类）
**时间：** ~45 分钟

## 学习目标

- 区分三种主要的视频建模方法（2D+pool、3D 卷积、时空 transformer），并预测它们的成本与精度权衡
- 在 PyTorch 中实现帧采样、时间池化以及 2D+pool 基线分类器
- 解释为什么 I3D 的“膨胀”3D 卷积核能很好地从 ImageNet 权重迁移，以及分解式 (2+1)D 卷积的不同之处
- 了解标准动作识别数据集与指标：Kinetics-400/600、UCF101、Something-Something V2；片段级别与视频级别的 top-1 准确率

## 问题

一段 30 秒、30 fps 的视频包含 900 张图像。朴素地，视频分类就是跑 900 次图像分类后再做某种聚合。当动作几乎在每一帧都可见时（体育、烹饪、健身视频），这种方法有效；但当动作本身由运动定义时，效果很差：在每一帧静止图像里，“把某物从左推到右”看起来只是两个静止物体。

每个视频架构的核心问题是：何时以及如何对时间结构进行建模？这个答案决定了其他一切——计算成本、预训练策略、是否能复用 ImageNet 权重、模型在什么数据集上训练。

本课比静态图像课程短。核心图像机制已经具备，视频理解主要讲时间故事：采样、建模和聚合。

## 概念

### 三大架构家族

```mermaid
flowchart LR
    V["Video clip<br/>(T frames)"] --> A1["2D + pool<br/>run 2D CNN per frame,<br/>average over time"]
    V --> A2["3D conv<br/>convolve over<br/>T x H x W"]
    V --> A3["Spatio-temporal<br/>transformer<br/>attention over<br/>(t, h, w) tokens"]

    A1 --> C["Logits"]
    A2 --> C
    A3 --> C

    style A1 fill:#dbeafe,stroke:#2563eb
    style A2 fill:#fef3c7,stroke:#d97706
    style A3 fill:#dcfce7,stroke:#16a34a
```

### 2D + pool

取一个 2D CNN（ResNet、EfficientNet、ViT），独立地对每个采样帧运行。对每帧嵌入做平均（或最大池化、注意力池化）。将池化后的向量送入分类器。

优点：
- 直接迁移 ImageNet 预训练权重。
- 实现最简单。
- 成本低：T 帧 × 单图推理成本。

缺点：
- 无法建模运动。动作 = 外观的聚合。
- 时间池化与顺序无关；“开门”和“关门”看起来一样。

适用场景：外观为主任务、小数据视频迁移学习、初始基线。

### 3D 卷积

将 2D (H, W) 卷积核替换为 3D (T, H, W) 卷积核。网络在时间和空间上同时卷积。早期代表：C3D、I3D、SlowFast。

I3D 技巧：取一个预训练的 2D ImageNet 模型，通过沿新时间轴复制每个 2D 卷积核来“膨胀”它。一个 3×3 的 2D 卷积变成 3×3×3 的 3D 卷积。这样 3D 模型就能拥有强预训练权重，而不是从头训练。

优点：
- 直接建模运动。
- I3D 膨胀带来免费迁移学习。

缺点：
- 计算量是 2D 对应模型的约 T/8 倍（3 层堆叠、时间核为 3 的情况下）。
- 时间核较小；长程运动需要金字塔或双流方法。

适用场景：运动是关键信号的动作识别任务（Something-Something V2、Kinetics 中运动占主导的类别）。

### 时空 transformer

将视频切分成空间-时间 patch 网格，并在所有 patch 上做注意力。TimeSformer、ViViT、Video Swin、VideoMAE。

重要的注意力模式：
- **Joint** —— 在 (t, h, w) 上做单一注意力。复杂度与 `T*H*W` 成二次关系；昂贵。
- **Divided** —— 每个块做两次注意力：一次沿时间，一次沿空间。接近线性扩展。
- **Factorised** —— 时间注意力与空间注意力在块之间交替。

优点：
- 在各大基准上取得 SOTA 精度。
- 通过 patch 膨胀从图像 transformer（ViT）迁移。
- 通过稀疏注意力支持长上下文视频。

缺点：
- 计算需求高。
- 需要谨慎选择注意力模式，否则运行时间会激增。

适用场景：大型数据集、高保真视频理解、多模态视频+文本任务。

### 帧采样

一段 10 秒、30 fps 的剪辑有 300 帧；把全部 300 帧输入任何模型都很浪费。标准策略：

- **均匀采样** —— 在剪辑中等间距选取 T 帧。2D+pool 的默认做法。
- **密集采样** —— 随机选取连续的 T 帧窗口。3D 卷积常用，因为运动需要相邻帧。
- **多片段** —— 从同一视频中采样多个 T 帧窗口，分别分类，测试时平均预测结果。

T 通常为 8、16、32 或 64。T 越大，获得的时间信号越多，计算也越大。

### 评估

两个层次：
- **片段级准确率** —— 模型看到一个 T 帧片段，报告 top-k。
- **视频级准确率** —— 对视频中多个片段的预测取平均；更高且更稳定。

两个都应报告。如果模型片段 / 视频得分是 78% / 82%，说明它严重依赖测试时平均；如果是 80% / 81%，则每个片段更鲁棒。

### 你会遇到的数据集

- **Kinetics-400 / 600 / 700** —— 通用动作数据集。40 万片段；YouTube URL（很多已失效）。
- **Something-Something V2** —— 由运动定义的动作（“把 X 从左移到右”）。无法被 2D+pool 解决。
- **UCF-101**、**HMDB-51** —— 更老、更小、仍有引用。
- **AVA** —— 在时空上进行动作*定位*；比分类更难。

## 实现

### 步骤 1：帧采样器

对帧列表（或视频张量）实现均匀采样与密集采样。

```python
import numpy as np

def sample_uniform(num_frames_total, T):
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total, T, rng=None):
    rng = rng or np.random.default_rng()
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    start = int(rng.integers(0, num_frames_total - T + 1))
    return list(range(start, start + T))
```

两个函数都返回 `T` 个索引，用于切片视频张量。

### 步骤 2：2D+pool 基线

对每一帧运行 2D ResNet-18，平均池化特征，然后分类。

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FramePool(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))  # global avg pool kept
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (N, T, 3, H, W)
        N, T = x.shape[:2]
        x = x.view(N * T, *x.shape[2:])
        feats = self.features(x).view(N, T, -1)
        pooled = feats.mean(dim=1)
        return self.head(pooled)

model = FramePool(num_classes=10)
x = torch.randn(2, 8, 3, 224, 224)
print(f"output: {model(x).shape}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
```

1100 万参数，ImageNet 预训练，逐帧运行、平均、分类。这个基线在外观为主的任务上通常只比真正的 3D 模型低 5-10 个百分点——有时甚至更好，因为它复用了更强的 ImageNet 骨干。

### 步骤 3：I3D 风格的膨胀 3D 卷积

通过沿新时间轴重复权重，将一个 2D 卷积变成 3D 卷积。

```python
def inflate_2d_to_3d(conv2d, time_kernel=3):
    out_c, in_c, kh, kw = conv2d.weight.shape
    weight_3d = conv2d.weight.data.unsqueeze(2)  # (out, in, 1, kh, kw)
    weight_3d = weight_3d.repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d = nn.Conv3d(in_c, out_c, kernel_size=(time_kernel, kh, kw),
                        padding=(time_kernel // 2, conv2d.padding[0], conv2d.padding[1]),
                        stride=(1, conv2d.stride[0], conv2d.stride[1]),
                        bias=False)
    conv3d.weight.data = weight_3d
    return conv3d

conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
conv3d = inflate_2d_to_3d(conv2d, time_kernel=3)
print(f"2D weight shape:  {tuple(conv2d.weight.shape)}")
print(f"3D weight shape:  {tuple(conv3d.weight.shape)}")
x = torch.randn(1, 3, 8, 56, 56)
print(f"3D output shape:  {tuple(conv3d(x).shape)}")
```

除以 `time_kernel` 可保持激活幅度大致不变——这对首次不破坏批归一化统计量很重要。

### 步骤 4：分解式 (2+1)D 卷积

将 3D 卷积拆分为 2D（空间）卷积和 1D（时间）卷积。感受野相同，参数更少，在某些基准上精度更高。

```python
class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        mid_c = (in_c * out_c * kernel_size * kernel_size * kernel_size) \
                // (in_c * kernel_size * kernel_size + out_c * kernel_size)
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
        self.bn = nn.BatchNorm3d(mid_c)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_c, out_c, kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0), bias=False)

    def forward(self, x):
        return self.temporal(self.act(self.bn(self.spatial(x))))

c = Conv2Plus1D(3, 64)
x = torch.randn(1, 3, 8, 56, 56)
print(f"(2+1)D output: {tuple(c(x).shape)}")
```

完整的 R(2+1)D 网络就是将 ResNet-18 中的每个 3x3 卷积替换为 `Conv2Plus1D`。

## 应用

两个库覆盖生产级视频工作：

- `torchvision.models.video` —— R(2+1)D、MViT、Swin3D，带有 Kinetics 预训练权重。与图像模型 API 相同。
- `pytorchvideo`（Meta）—— 模型库、Kinetics / SSv2 / AVA 数据加载器、标准变换。

对于视觉-语言视频模型（视频字幕、视频问答），使用 `transformers`（`VideoMAE`、`VideoLLaMA`、`InternVideo`）。

## 交付

本课产出：

- `outputs/prompt-video-architecture-picker.md` —— 一个根据外观与运动、数据规模、计算预算来选择 2D+pool / I3D / (2+1)D / transformer 的提示词。
- `outputs/skill-frame-sampler-auditor.md` —— 一个检查视频流水线采样器并标记常见 bug 的技能：索引差一、`num_frames < T` 时采样不均、缺少保宽高比裁剪等。

## 练习

1. **（简单）** 估算 FramePool 在 T=8 时与 I3D 风格 3D ResNet 在 T=8 时的 FLOPs。说明为什么 2D+pool 便宜 3-5 倍。
2. **（中等）** 生成合成视频数据集：随机小球朝随机方向运动，标签为运动方向（“从左到右”、“从右到左”、“对角向上”）。用 FramePool 训练它。证明它接近随机准确率，说明仅靠外观不足以完成运动任务。
3. **（困难）** 通过将 ResNet-18 中的每个 Conv2d 替换为 `Conv2Plus1D`，构建 R(2+1)D-18。从 ImageNet 预训练的 ResNet-18 膨胀第一个卷积的权重。在练习 2 的运动数据集上训练并击败 FramePool。

## 关键术语

| 术语 | 大家的说法 | 实际含义 |
|------|-----------|---------|
| 2D + pool | “逐帧分类器” | 在每次采样帧上运行 2D CNN，沿时间平均池化特征，再分类 |
| 3D 卷积 | “时空卷积核” | 在 (T, H, W) 上卷积的卷积核；可原生建模运动 |
| Inflation | “将 2D 权重提升到 3D” | 通过沿新时间轴重复 2D 卷积权重来初始化 3D 卷积权重，再除以 kernel_T 以保持激活尺度 |
| (2+1)D | “分解式卷积” | 将 3D 拆分为 2D 空间 + 1D 时间；参数更少，中间多一层非线性 |
| Divided attention | “先时间后空间” | Transformer 块每层做两次注意力：一次在同一帧的 token 上，一次在同一位置的 token 上 |
| Clip | “T 帧窗口” | 采样的 T 帧子序列；视频模型消费的单元 |
| Clip vs video accuracy | “两种评估设置” | Clip = 每个视频一个样本，video = 对多个采样片段的预测取平均 |
| Kinetics | “视频领域的 ImageNet” | 400-700 个动作类别，30 万+ YouTube 片段，标准视频预训练语料 |

## 进一步阅读

- [I3D: Quo Vadis, Action Recognition（Carreira & Zisserman，2017）](https://arxiv.org/abs/1705.07750) —— 引入膨胀方法与 Kinetics 数据集
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions（Tran 等，2018）](https://arxiv.org/abs/1711.11248) —— 分解式卷积，至今仍是强基线
- [TimeSformer: Is Space-Time Attention All You Need?（Bertasius 等，2021）](https://arxiv.org/abs/2102.05095) —— 首个强大的视频 transformer
- [VideoMAE（Tong 等，2022）](https://arxiv.org/abs/2203.12602) —— 面向视频的掩码自编码器预训练；当前主流的预训练方案
