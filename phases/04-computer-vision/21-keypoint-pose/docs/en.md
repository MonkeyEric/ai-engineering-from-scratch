# 关键点检测与姿态估计

> 姿态是一组有序的关键点。关键点检测器是热图回归器。其余都是工程细节。

**类型：** Build
**语言：** Python
**前置知识：** Phase 4 Lesson 06（目标检测）、Phase 4 Lesson 07（U-Net）
**时间：** 约 45 分钟

## 学习目标

- 区分自顶向下（top-down）与自底向上（bottom-up）姿态估计，并说明各自的适用场景
- 使用每个关键点一个高斯的目标热图，对 K 个关键点进行热图回归，并在推理时提取关键点坐标
- 解释部位亲和力场（PAFs）以及自底向上流程如何将关键点关联成实例
- 使用 MediaPipe Pose 或 MMPose 进行生产级关键点估计，并理解其输出格式

## 问题背景

关键点任务有许多名字：人体姿态（17 个关节）、人脸关键点（68 或 478 个点）、手部（21 个点）、动物姿态、机器人物体姿态、医学解剖学标志点。它们都共享同一个结构：在物体上检测 K 个离散点，并输出它们的 (x, y) 坐标。

姿态估计是动作捕捉、健身应用、体育分析、手势控制、动画、AR 试穿和机器人抓取的基础。2D 姿态已经相当成熟；3D 姿态（从单目相机估计关节在世界坐标系中的位置）仍是当前的研究前沿。

工程上的问题是规模。单图单人姿态是一个 20 毫秒的问题。人群中的多人姿态在 30 fps 下运行则是另一个问题，需要不同的架构。

## 核心概念

### 自顶向下 vs 自底向上

```mermaid
flowchart LR
    subgraph TD["Top-down pipeline"]
        A1["Detect person boxes"] --> A2["Crop each box"]
        A2 --> A3["Per-box keypoint model<br/>(HRNet, ViTPose)"]
    end
    subgraph BU["Bottom-up pipeline"]
        B1["One pass over image"] --> B2["All keypoint heatmaps<br/>+ association field"]
        B2 --> B3["Group keypoints into<br/>instances (greedy matching)"]
    end

    style TD fill:#dbeafe,stroke:#2563eb
    style BU fill:#fef3c7,stroke:#d97706
```

- **自顶向下（Top-down）** —— 先检测人体框，再对每个裁剪区域运行单人关键点模型。精度最高；耗时随人数线性增长。
- **自底向上（Bottom-up）** —— 一次前向传播预测所有关键点以及关联场，然后进行分组。无论人群多拥挤，耗时基本恒定。

自顶向下方法（HRNet、ViTPose）在精度上领先；自底向上方法（OpenPose、HigherHRNet）在拥挤场景下吞吐量更高。

### 热图回归

不直接回归 `(x, y)`，而是为每个关键点预测一个 `H x W` 的热图，在真实位置中心放置一个高斯 blob。

```
target[k, y, x] = exp(-((x - cx_k)^2 + (y - cy_k)^2) / (2 sigma^2))
```

在推理时，每个热图的 argmax 即为预测的关键点位置。

热图比直接回归更有效的原因：网络的空间结构（卷积特征图）与自然空间输出天然对齐。高斯目标还能起到正则作用——微小的定位误差只会产生较小的损失，而不是零损失。

### 亚像素定位

Argmax 给出整数坐标。要获得亚像素精度，可以对 argmax 及其邻域拟合抛物线，或者使用众所周知的偏移公式 `(dx, dy) = 0.25 * (heatmap[y, x+1] - heatmap[y, x-1], ...)`。

### 部位亲和力场（Part Affinity Fields, PAFs）

OpenPose 用于自底向上关联的 trick。对于每一对相连的关键点（例如左肩到左肘），预测一个双通道场，编码从一个关键点指向另一个关键点的单位向量。要将肩膀与手肘关联起来，就沿着候选点对之间的连线对 PAF 做积分；积分最高的点对即为匹配。

```
For each connection (limb):
  PAF channels: 2 (unit vector x, y)
  Line integral: sum over sample points of (PAF . line_direction)
  Higher integral = stronger match
```

这种方法优雅且可扩展到任意拥挤程度，无需按人裁剪。

### COCO 关键点

标准的人体姿态数据集：每个人 17 个关键点，使用 PCK（正确关键点百分比）和 OKS（对象关键点相似度）作为评估指标。OKS 是 IoU 的关键点版本，COCO mAP@OKS 报告的就是它。

### 2D vs 3D

- **2D 姿态** —— 图像坐标；已能在生产环境中达到高质量（MediaPipe、HRNet、ViTPose）。
- **3D 姿态** —— 世界 / 相机坐标；仍是活跃研究领域。常见方法：
  - 用一个小型 MLP 将 2D 预测提升到 3D（VideoPose3D）。
  - 从图像直接回归 3D（PyMAF、MHFormer）。
  - 多相机系统（CMU Panoptic）用于获取真值。

## 动手实现

### 步骤 1：高斯热图目标

```python
import numpy as np
import torch

def gaussian_heatmap(size, cx, cy, sigma=2.0):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)

hm = gaussian_heatmap(64, 32, 32, sigma=2.0)
print(f"peak: {hm.max():.3f} at ({hm.argmax() % 64}, {hm.argmax() // 64})")
```

按通道堆叠每个关键点的热图，即可得到完整的目标张量。

### 步骤 2：微型关键点头部网络

一个 U-Net 风格的模型，输出 K 个热图通道。

```python
import torch.nn as nn
import torch.nn.functional as F

class TinyKeypointNet(nn.Module):
    def __init__(self, num_keypoints=4, base=16):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, base, 3, 2, 1), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.up2 = nn.ConvTranspose2d(base, num_keypoints, 2, 2)

    def forward(self, x):
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.mid(h2)
        u1 = self.up1(h3)
        return self.up2(u1)
```

输入 `(N, 3, H, W)`，输出 `(N, K, H, W)`。损失为与目标高斯热图的逐像素 MSE。

### 步骤 3：推理——提取关键点坐标

```python
def heatmap_to_coords(heatmaps):
    """
    heatmaps: (N, K, H, W)
    returns:  (N, K, 2) float coordinates in image pixels
    """
    N, K, H, W = heatmaps.shape
    hm = heatmaps.reshape(N, K, -1)
    idx = hm.argmax(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    return torch.stack([xs, ys], dim=-1)

coords = heatmap_to_coords(torch.randn(2, 4, 32, 32))
print(f"coords: {coords.shape}")  # (2, 4, 2)
```

推理只需一行代码。如需亚像素精化，可在 argmax 附近插值。

### 步骤 4：合成关键点数据集

简单做法：在白色画布上画四个点，让模型学习预测它们。

```python
def make_synthetic_sample(size=64):
    img = np.ones((3, size, size), dtype=np.float32)
    rng = np.random.default_rng()
    kps = rng.integers(8, size - 8, size=(4, 2))
    for cx, cy in kps:
        img[:, cy - 2:cy + 2, cx - 2:cx + 2] = 0.0
    hms = np.stack([gaussian_heatmap(size, cx, cy) for cx, cy in kps])
    return img, hms, kps
```

数据集足够简单，微型模型一分钟内即可学会。

### 步骤 5：训练

```python
model = TinyKeypointNet(num_keypoints=4)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(200):
    batch = [make_synthetic_sample() for _ in range(16)]
    imgs = torch.from_numpy(np.stack([b[0] for b in batch]))
    hms = torch.from_numpy(np.stack([b[1] for b in batch]))
    pred = model(imgs)
    # Upsample pred to full resolution
    pred = F.interpolate(pred, size=hms.shape[-2:], mode="bilinear", align_corners=False)
    loss = F.mse_loss(pred, hms)
    opt.zero_grad(); loss.backward(); opt.step()
```

## 实际应用

- **MediaPipe Pose** —— Google 的生产级姿态估计器；提供 WebGL + 移动端运行时，延迟低于 10 毫秒。
- **MMPose**（OpenMMLab）—— 全面的研究代码库；包含所有 SOTA 架构及预训练权重。
- **YOLOv8-pose** —— 最快的实时多人姿态估计，单次前向传播即可完成。
- **transformers HumanDPT / PoseAnything** —— 新兴的视觉-语言方法，支持开放词汇姿态估计（任意物体、任意关键点集合）。

## 交付成果

本节课将产出：

- `outputs/prompt-pose-stack-picker.md` —— 一个根据延迟、人群规模和 2D/3D 需求选择 MediaPipe / YOLOv8-pose / HRNet / ViTPose 的提示词。
- `outputs/skill-heatmap-to-coords.md` —— 一个编写亚像素热图转坐标例程的技能，该例程被所有生产级姿态模型使用。

## 练习题

1. **（简单）** 在合成的 4 点数据集上训练微型关键点模型。报告 200 步后预测关键点与真实关键点之间的平均 L2 误差。
2. **（中等）** 添加亚像素精化：给定 argmax 位置，沿 x 和 y 方向利用相邻像素拟合一维抛物线。报告相比整数 argmax 的精度提升。
3. **（困难）** 构建一个 2 人合成数据集，每张图像包含两个 4 关键点模式的实例。训练一个带 PAFs 的自底向上流程，预测每个关键点属于哪个实例，并评估 OKS。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Keypoint（关键点） | "A landmark" | 物体上一个特定的有序点（关节、角点、特征点） |
| Pose（姿态） | "The skeleton" | 属于同一实例的一组有序关键点 |
| Top-down（自顶向下） | "Detect then pose" | 两阶段流程：人体检测器 + 每个裁剪区域的关键点模型；精度最高 |
| Bottom-up（自底向上） | "Pose first, group later" | 单次前向预测所有关键点并分组；耗时与人群规模无关 |
| Heatmap（热图） | "Gaussian target" | 每个关键点对应的 H x W 张量，峰值位于真实位置；首选回归目标 |
| PAF（部位亲和力场） | "Part Affinity Field" | 双通道单位向量场，编码肢体方向；用于将关键点分组为实例 |
| OKS（对象关键点相似度） | "Keypoint IoU" | Object Keypoint Similarity；COCO 姿态估计评估指标 |
| HRNet | "High-Resolution Net" | 主流的自顶向下关键点架构；全程保持高分辨率特征 |

## 延伸阅读

- [OpenPose (Cao et al., 2017)](https://arxiv.org/abs/1812.08008) —— 基于 PAFs 的自底向上方法；仍是该领域最清晰的论文
- [HRNet (Sun et al., 2019)](https://arxiv.org/abs/1902.09212) —— 自顶向下的参考架构
- [ViTPose (Xu et al., 2022)](https://arxiv.org/abs/2204.12484) —— 使用 plain ViT 作为姿态骨干网络；在多个基准上达到当前 SOTA
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) —— 生产级实时姿态估计；2026 年部署最广泛、速度最快的方案
