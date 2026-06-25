# 单目深度与几何估计

> 深度图是一幅单通道图像，每个像素表示该点到相机的距离。过去，仅凭单张 RGB 帧预测深度，没有立体相机或 LiDAR 是不可能的。到 2026 年，一个冻结的 ViT 编码器加上一个轻量头，就能让预测结果与真值只差几个百分点。

**类型：** 构建 + 使用  
**语言：** Python  
**前置知识：** Phase 4 Lesson 14（ViT）、Phase 4 Lesson 17（自监督视觉）、Phase 4 Lesson 07（U-Net）  
**时长：** 约 60 分钟  

## 学习目标

- 区分相对深度与度量深度，并能说明每个生产模型（MiDaS、Marigold、Depth Anything V3、ZoeDepth）解决的是哪一种
- 在没有标定的情况下，使用 Depth Anything V3（DINOv2 骨干网络）对任意单张图像进行深度预测
- 解释为什么单目深度仅凭单张图像就能工作（透视线索、纹理梯度、学习到的先验），以及它无法恢复什么（绝对尺度、被遮挡的几何）
- 利用深度图和针孔相机内参将 2D 检测结果提升为 3D 点

## 问题背景

深度是 2D 计算机视觉中缺失的维度。给定 RGB 图像，你知道物体在图像平面上的位置，但不知道它们有多远。深度传感器（立体相机、LiDAR、飞行时间）可以直接解决这个问题，但价格昂贵、易损坏且量程有限。

单目深度估计——从单张 RGB 帧预测深度——过去输出的结果模糊且不可靠。到 2026 年，大型预训练编码器改变了这一局面：Depth Anything V3 使用冻结的 DINOv2 骨干网络，生成的深度图可以泛化到室内、室外、医学和卫星等多个领域。Marigold 将深度估计重新表述为条件扩散问题。ZoeDepth 则回归真实的度量距离。

深度也是 2D 检测与 3D 理解之间的桥梁：将检测框中的像素乘以深度，就能把 2D 物体提升到 3D 点云中。这是每一个 AR 遮挡系统、每一条避障管线，以及每一个“捡起杯子”的机器人的核心。

## 核心概念

### 相对深度 vs 度量深度

- **相对深度** — 有序的 `z` 值，没有真实世界单位。“像素 A 比像素 B 更近，但距离之比并未锚定到米。”
- **度量深度** — 相机到物体的绝对距离，单位为米。要求模型已经学会图像线索与真实距离之间的统计关系。

MiDaS 和 Depth Anything V3 生成相对深度。Marigold 生成相对深度。ZoeDepth、UniDepth 和 Metric3D 生成度量深度。度量模型对相机内参敏感；相对模型则不然。

### 编码器-解码器结构

```mermaid
flowchart LR
    IMG["Image (H x W x 3)"] --> ENC["Frozen ViT encoder<br/>(DINOv2 / DINOv3)"]
    ENC --> FEATS["Dense features<br/>(H/14, W/14, d)"]
    FEATS --> DEC["Depth decoder<br/>(conv upsampler,<br/>DPT-style)"]
    DEC --> DEPTH["Depth map<br/>(H, W, 1)"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style DEC fill:#fef3c7,stroke:#d97706
    style DEPTH fill:#dcfce7,stroke:#16a34a
```

Depth Anything V3 冻结编码器，只训练 DPT 风格的解码器。编码器提供丰富的特征；解码器将其插值回图像分辨率并回归深度。

### 为什么单张图像也能产生深度

2D 图像包含许多与深度相关的单目线索：

- **透视** — 3D 中的平行线在 2D 中会收敛。
- **纹理梯度** — 远处的表面纹理更小、更密集。
- **遮挡顺序** — 近处物体会遮挡远处物体。
- **大小恒常性** — 已知大小的物体（汽车、人）可提供近似尺度。
- **大气透视** — 户外场景中，远处物体看起来更朦胧、更偏蓝。

在数十亿张图像上训练的 ViT 会内化这些线索。只要有足够的数据和强大的骨干网络，单目深度无需任何显式 3D 监督即可达到合理精度。

### 单目深度无法做到的事

- **没有内参或场景中已知物体时，无法恢复绝对度量尺度。** 网络可以预测“杯子比勺子远两倍”，却不知道杯子到底是 1 米还是 10 米远。
- **被遮挡的几何** — 椅子的背面不可见，无法可靠推断。
- **真正无纹理/反光的表面** — 镜子、玻璃、均匀墙面。网络会输出看似合理但错误的深度。

### 2026 年的 Depth Anything V3

- 编码器使用原版 DINOv2 ViT-L/14（冻结）。
- DPT 解码器。
- 在来自多种来源的已位姿图像对上训练（仅需光度一致性，无需显式深度监督）。
- 能够从**任意数量的视觉输入中预测空间一致的几何，无论是否已知相机位姿**。
- 在单目深度、任意视角几何、视觉渲染、相机位姿估计等任务上达到 SOTA。

这是 2026 年需要深度时的即插即用模型。

### Marigold — 用于深度估计的扩散模型

Marigold（Ke et al., CVPR 2024）将深度估计重新表述为条件图像到图像扩散。条件：RGB。目标：深度图。使用预训练的 Stable Diffusion 2 U-Net 作为骨干网络。输出的深度图在物体边界处异常清晰。代价：推理比前馈模型慢（10–50 个去噪步）。

### 内参与针孔相机模型

要将像素 `(u, v)` 连同深度 `d` 提升到相机坐标系中的 3D 点 `(X, Y, Z)`：

```
fx, fy, cx, cy = camera intrinsics
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d
```

内参来自 EXIF 元数据、标定板或单目内参估计器（Perspective Fields、UniDepth）。没有内参时，仍可假设 60–70° 视场角和中等分辨率主点来渲染点云——可用于可视化，但不能用于测量。

### 评估

两个常用指标：

- **AbsRel**（绝对相对误差）：`mean(|d_pred - d_gt| / d_gt)`。越低越好。生产模型通常在 0.05–0.1 之间。
- **delta < 1.25**（阈值精度）：满足 `max(d_pred/d_gt, d_gt/d_pred) < 1.25` 的像素比例。越高越好。SOTA 模型可达 0.9 以上。

对于相对深度（Depth Anything V3、MiDaS），评估使用这两个指标的尺度-平移不变版本。

## 动手实现

### 步骤 1：深度指标

```python
import torch

def abs_rel_error(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return (torch.abs(pred - target) / target.clamp(min=1e-6)).mean().item()


def delta_accuracy(pred, target, threshold=1.25, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    ratio = torch.maximum(pred / target.clamp(min=1e-6), target / pred.clamp(min=1e-6))
    return (ratio < threshold).float().mean().item()
```

评估前务必将无效深度像素（零、NaN、饱和）用掩码过滤掉。

### 步骤 2：尺度-平移对齐

对于相对深度模型，在计算指标前需将预测对齐到真值。用最小二乘拟合 `a * pred + b = target`：

```python
def align_scale_shift(pred, target, mask=None):
    if mask is not None:
        p = pred[mask]
        t = target[mask]
    else:
        p = pred.flatten()
        t = target.flatten()
    A = torch.stack([p, torch.ones_like(p)], dim=1)
    coeffs, *_ = torch.linalg.lstsq(A, t.unsqueeze(-1))
    a, b = coeffs[:2, 0]
    return a * pred + b
```

评估 MiDaS / Depth Anything 时，先运行 `align_scale_shift`，再运行 `abs_rel_error`。

### 步骤 3：将深度提升为点云

```python
import numpy as np

def depth_to_point_cloud(depth, intrinsics):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


depth = np.random.uniform(0.5, 4.0, (240, 320))
intr = (320.0, 320.0, 160.0, 120.0)
pc = depth_to_point_cloud(depth, intr)
print(f"point cloud shape: {pc.shape}  (H, W, 3)")
```

一个函数，搞定所有 3D 提升应用。将点云导出为 `.ply`，然后在 MeshLab 或 CloudCompare 中打开。

### 步骤 4：用合成深度场景进行冒烟测试

```python
def synthetic_depth(size=96):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    # 地面：从近（顶部）到远（底部）的线性渐变
    depth = 1.0 + (yy / size) * 4.0
    # 中间方块：更近
    mask = (np.abs(xx - size / 2) < size / 6) & (np.abs(yy - size * 0.6) < size / 6)
    depth[mask] = 2.0
    return depth.astype(np.float32)


gt = torch.from_numpy(synthetic_depth(96))
pred = gt + 0.3 * torch.randn_like(gt)  # 模拟预测
aligned = align_scale_shift(pred, gt)
print(f"before align  absRel = {abs_rel_error(pred, gt):.3f}")
print(f"after align   absRel = {abs_rel_error(aligned, gt):.3f}")
```

### 步骤 5：Depth Anything V3 使用方式（参考）

```python
import torch
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-v2-large")

image = Image.open("street.jpg").convert("RGB")
out = pipe(image)
depth_np = np.array(out["depth"])
```

三行代码。`out["depth"]` 是 PIL 灰度图；如需计算，转成 numpy。专门针对 Depth Anything V3，发布后只需替换模型 ID；API 保持不变。

## 实际应用

- **Depth Anything V3**（Meta AI / ByteDance，2024–2026）—— 相对深度的默认选择。生产环境中使用 ViT-large 骨干网络的最快模型。
- **Marigold**（ETH，2024）—— 视觉质量最高，推理较慢。
- **UniDepth**（ETH，2024）—— 度量深度，同时估计相机内参。
- **ZoeDepth**（Intel，2023）—— 度量深度；较老，但仍然可靠。
- **MiDaS v3.1** —— 经典且稳定；适合作为对比基线。

典型集成模式：

1. RGB 帧输入。
2. 深度模型生成深度图。
3. 检测器生成检测框。
4. 通过深度将检测框中心提升到 3D；如有可用的点云，再进行融合。
5. 下游应用：AR 遮挡、路径规划、物体尺寸估计、立体视觉替代。

实时场景下，Depth Anything V2 Small（INT8 量化）在 518×518 分辨率下可在消费级 GPU 上达到约 30 fps。

## 交付成果

本节课产出：

- `outputs/prompt-depth-model-picker.md` —— 根据延迟、度量/相对需求以及场景类型，在 Depth Anything V3、Marigold、UniDepth、MiDaS 之间进行选择。
- `outputs/skill-depth-to-pointcloud.md` —— 一个从深度图构建点云的技能，正确处理内参并导出为 `.ply`。

## 练习

1. **（简单）** 在任意 10 张你桌面的图片上运行 Depth Anything V2。将深度保存为灰度 PNG 并检查。找出一个预测深度明显错误的物体，并解释为什么单目线索失效了。
2. **（中等）** 给定 Depth Anything V2 输出的 RGB + 深度，将其提升到点云并用 `open3d` 渲染。比较两个场景（室内/室外），观察哪个更可信。
3. **（困难）** 拍摄五对仅有一个已知物体位置不同的图像（例如瓶子向镜头移动 30 cm）。在两张图像上都用 UniDepth 预测度量深度。报告预测距离差值与真实 30 cm 的对比。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| Monocular depth | “单图深度” | 从单张 RGB 帧估计深度，无需立体相机或 LiDAR |
| Relative depth | “有序深度” | 有序的 z 值，没有真实世界单位 |
| Metric depth | “绝对距离” | 以米为单位的深度；需要标定或在度量监督下训练的模型 |
| AbsRel | “绝对相对误差” | `|d_pred - d_gt| / d_gt` 的均值；标准深度指标 |
| Delta accuracy | “delta < 1.25” | 预测值在真值 25% 范围内的像素比例 |
| Pinhole camera | “fx, fy, cx, cy” | 用于将 (u, v, d) 提升为 (X, Y, Z) 的相机模型 |
| DPT | “Dense Prediction Transformer” | 在冻结 ViT 编码器之上用于深度估计的卷积解码器 |
| DINOv2 backbone | “它之所以有效的原因” | 无需深度标签即可跨领域泛化的自监督特征 |

## 延伸阅读

- [Depth Anything V3 paper page](https://depth-anything.github.io/) —— 使用 DINOv2 编码器的 SOTA 单目深度模型
- [Marigold (Ke et al., CVPR 2024)](https://marigoldmonodepth.github.io/) —— 基于扩散模型的深度估计
- [UniDepth (Piccinelli et al., 2024)](https://arxiv.org/abs/2403.18913) —— 带内参估计的度量深度
- [MiDaS v3.1 (Intel ISL)](https://github.com/isl-org/MiDaS) —— 经典的相对深度基线
- [DINOv3 blog post (Meta)](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) —— 提升深度精度的编码器系列
