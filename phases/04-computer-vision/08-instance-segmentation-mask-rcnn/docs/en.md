# 实例分割 — Mask R-CNN

> 在 Faster R-CNN 检测器上加一个轻量掩码分支，就得到了实例分割。难点在于 RoIAlign，而它比你想象的更难。

**类型：** 构建 + 学习
**语言：** Python
**前置知识：** 第 4 阶段第 06 课（YOLO）、第 4 阶段第 07 课（U-Net）
**时间：** 约 75 分钟

## 学习目标

- 端到端梳理 Mask R-CNN 架构：backbone、FPN、RPN、RoIAlign、box head、mask head
- 从零实现 RoIAlign，并解释为什么不再使用 RoIPool
- 使用 torchvision 的 `maskrcnn_resnet50_fpn_v2` 预训练模型生成生产级实例掩码，并正确读取其输出格式
- 在小规模自定义数据集上微调 Mask R-CNN，替换 box head 与 mask head 并冻结 backbone

## 问题背景

语义分割为每个类别输出一个掩码。实例分割为每个目标输出一个掩码，即使两个目标属于同一类别。统计个体数量、跨帧跟踪、测量尺寸（例如墙中每块砖的包围盒、显微图像中的每个细胞）都需要实例分割。

Mask R-CNN（He 等人，2017）通过将实例分割重新定义为“检测 + 掩码”解决了这一问题。其设计如此简洁，以至于此后五年内几乎所有实例分割论文都是 Mask R-CNN 的变体；时至今日，torchvision 的实现仍然是中小型数据集的生产默认选择。

工程上的核心难点是采样：如何从一个角点不与像素边界对齐的候选框中，裁剪出固定大小的特征区域？一旦出错，mAP 会处处损失零点几个点。RoIAlign 就是答案。

## 核心概念

### 网络架构

```mermaid
flowchart LR
    IMG["Input"] --> BB["ResNet<br/>backbone"]
    BB --> FPN["Feature<br/>Pyramid Network"]
    FPN --> RPN["Region<br/>Proposal<br/>Network"]
    FPN --> RA["RoIAlign"]
    RPN -->|"top-K proposals"| RA
    RA --> BH["Box head<br/>(class + refine)"]
    RA --> MH["Mask head<br/>(14x14 conv)"]
    BH --> NMS["NMS"]
    MH --> NMS
    NMS --> OUT["boxes +<br/>classes + masks"]

    style BB fill:#dbeafe,stroke:#2563eb
    style FPN fill:#fef3c7,stroke:#d97706
    style RPN fill:#fecaca,stroke:#dc2626
    style OUT fill:#dcfce7,stroke:#16a34a
```

需要理解的五个部分：

1. **Backbone** — 在 ImageNet 上训练的 ResNet-50 或 ResNet-101。生成步长为 4、8、16、32 的多级特征图。
2. **FPN（特征金字塔网络）** — 自顶向下 + 横向连接，使每一层都具有 C 个通道的语义丰富特征。检测时根据目标尺寸选择对应的 FPN 层。
3. **RPN（区域提议网络）** — 一个小的卷积头，在每个锚点位置预测“这里是否有目标？”以及“如何修正候选框？”。每张图像生成约 1000 个候选框。
4. **RoIAlign** — 从任意 FPN 层上的任意候选框中采样固定大小（例如 7x7）的特征块。使用双线性采样，不进行量化。
5. **Heads** — 两层的 box head 用于精修候选框并选择类别；一个小的卷积头为每个候选框输出一个 `28x28` 的二值掩码。

### 为什么用 RoIAlign，而不是 RoIPool

最早的 Fast R-CNN 使用 RoIPool：将候选框划分为网格，取每个单元格内的最大特征，并将所有坐标四舍五入为整数。这种取整会让特征图与输入像素坐标之间产生最多一个特征像素的对齐误差 — 在 224x224 图像上很小，但在步长为 32 的特征图上却是灾难性的。

```
RoIPool:
  box (34.7, 51.3, 98.2, 142.9)
  round -> (34, 51, 98, 142)
  split grid -> round each cell boundary
  misalignment accumulates at every step

RoIAlign:
  box (34.7, 51.3, 98.2, 142.9)
  sample at exact float coordinates using bilinear interpolation
  no rounding anywhere
```

RoIAlign 在 COCO 上能将 mask AP 免费提升 3-4 个点。如今所有关注定位精度的检测器都在使用它 — YOLOv7 seg、RT-DETR、Mask2Former 概莫能外。

### RPN 一句话解释

在特征图的每个位置放置 K 个不同尺寸和形状的锚框。为每个锚框预测一个目标性分数和一个回归偏移量，用于将锚框调整为更贴合目标的框。按分数保留前 ~1000 个框，以 IoU 0.7 做 NMS，将幸存者交给后续 head。RPN 使用自己的小型损失函数训练 — 结构与第 6 课中的 YOLO 损失相同，只是二分类（有目标 / 无目标）。

### Mask head

对于每个候选框（经过 RoIAlign 后），mask head 是一个小型 FCN：四个 3x3 卷积、一个 2x 反卷积、一个最终 1x1 卷积，输出 `num_classes` 个通道的 `28x28` 分辨率。只保留与预测类别对应的通道，其余通道被忽略。这实现了掩码预测与分类的解耦。

将 28x28 的掩码上采样到候选框的原始像素大小，得到最终二值掩码。

### 损失函数

Mask R-CNN 将四个损失相加：

```
L = L_rpn_cls + L_rpn_box + L_box_cls + L_box_reg + L_mask
```

- `L_rpn_cls`、`L_rpn_box` — RPN 候选框的目标性分类损失与框回归损失。
- `L_box_cls` — head 分类器在 (C+1) 个类别（含背景）上的交叉熵损失。
- `L_box_reg` — head 框精修的 smooth L1 损失。
- `L_mask` — 28x28 掩码输出上的逐像素二值交叉熵损失。

每项损失都有各自的默认权重；torchvision 实现通过构造函数参数暴露它们。

### 输出格式

`torchvision.models.detection.maskrcnn_resnet50_fpn_v2` 返回一个字典列表，每张图像对应一个：

```
{
    "boxes":  (N, 4) in (x1, y1, x2, y2) pixel coordinates,
    "labels": (N,) class IDs, 0 = background so indices are 1-based,
    "scores": (N,) confidence scores,
    "masks":  (N, 1, H, W) float masks in [0, 1] — threshold at 0.5 for binary,
}
```

掩码已经是完整图像分辨率。28x28 的 head 输出已在内部完成上采样。

## 动手实现

### 第一步：从零实现 RoIAlign

这是 Mask R-CNN 中一个用代码理解比用文字理解更简单的组件。

```python
import torch
import torch.nn.functional as F

def roi_align_single(feature, box, output_size=7, spatial_scale=1 / 16.0):
    """
    feature: (C, H, W) 单张图像特征图
    box: (x1, y1, x2, y2) 原始图像像素坐标
    output_size: 输出网格边长（box head 用 7，mask head 用 14）
    spatial_scale: 特征图步长的倒数
    """
    C, H, W = feature.shape
    x1, y1, x2, y2 = [c * spatial_scale - 0.5 for c in box]
    bin_w = (x2 - x1) / output_size
    bin_h = (y2 - y1) / output_size

    grid_y = torch.linspace(y1 + bin_h / 2, y2 - bin_h / 2, output_size)
    grid_x = torch.linspace(x1 + bin_w / 2, x2 - bin_w / 2, output_size)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    gx = 2 * (xx + 0.5) / W - 1
    gy = 2 * (yy + 0.5) / H - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(feature.unsqueeze(0), grid, mode="bilinear",
                            align_corners=False)
    return sampled.squeeze(0)
```

每个数值都通过双线性采样得到。没有取整、没有量化、没有梯度丢失。

### 第二步：与 torchvision 的 RoIAlign 对比

```python
from torchvision.ops import roi_align

feature = torch.randn(1, 16, 50, 50)
boxes = torch.tensor([[0, 10, 20, 100, 90]], dtype=torch.float32)  # (batch_idx, x1, y1, x2, y2)

ours = roi_align_single(feature[0], boxes[0, 1:].tolist(), output_size=7, spatial_scale=1/4)
theirs = roi_align(feature, boxes, output_size=(7, 7), spatial_scale=1/4, sampling_ratio=1, aligned=True)[0]

print(f"shape ours:   {tuple(ours.shape)}")
print(f"shape theirs: {tuple(theirs.shape)}")
print(f"max|diff|:    {(ours - theirs).abs().max().item():.3e}")
```

当 `sampling_ratio=1` 且 `aligned=True` 时，两者差异在 `1e-5` 以内。

### 第三步：加载预训练 Mask R-CNN

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
print(f"classes (including background): {len(model.roi_heads.box_predictor.cls_score.out_features * [0])}")
```

4600 万参数、91 个类别（COCO）。第一个类别（id 0）是背景；模型实际检测到的目标从 id 1 开始。

### 第四步：推理

```python
with torch.no_grad():
    x = torch.randn(3, 400, 600)
    predictions = model([x])
p = predictions[0]
print(f"boxes:  {tuple(p['boxes'].shape)}")
print(f"labels: {tuple(p['labels'].shape)}")
print(f"scores: {tuple(p['scores'].shape)}")
print(f"masks:  {tuple(p['masks'].shape)}")
```

掩码张量的形状为 `(N, 1, H, W)`。以 0.5 为阈值得到每个目标的二值掩码：

```python
binary_masks = (p['masks'] > 0.5).squeeze(1)  # (N, H, W) boolean
```

### 第五步：为自定义类别数替换 head

常见的微调方案：复用 backbone、FPN 和 RPN；替换两个分类 head。

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_custom_maskrcnn(num_classes):
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

custom = build_custom_maskrcnn(num_classes=5)
print(f"custom cls_score.out_features: {custom.roi_heads.box_predictor.cls_score.out_features}")
```

`num_classes` 必须包含背景类，因此一个有 4 个目标类别的数据集应使用 `num_classes=5`。

### 第六步：冻结不需要训练的层

在小数据集上，冻结 backbone 和 FPN。只让 RPN 的目标性 + 回归以及两个 head 参与学习。

```python
def freeze_backbone_and_fpn(model):
    # torchvision 的 Mask R-CNN 将 FPN 封装在 `model.backbone` 内部（即
    # `model.backbone.fpn`），因此遍历 `model.backbone.parameters()` 会同时覆盖
    # ResNet 特征层和 FPN 横向/输出卷积。
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model

custom = freeze_backbone_and_fpn(custom)
trainable = sum(p.numel() for p in custom.parameters() if p.requires_grad)
print(f"trainable after freeze: {trainable:,}")
```

在 500 张图像的数据集上，这一步决定了模型是能收敛还是会过拟合。

## 应用

torchvision 中 Mask R-CNN 的完整训练循环只有 40 行，且在不同任务之间几乎没有本质区别 — 换数据集即可。

```python
def train_step(model, images, targets, optimizer):
    model.train()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    return {k: v.item() for k, v in loss_dict.items()}
```

`targets` 列表必须包含每张图像的字典，其中有 `boxes`、`labels` 和 `masks`（形状为 `(num_instances, H, W)` 的二值张量）。训练时模型返回一个包含四项损失的字典；评估时返回预测列表，具体取决于 `model.training`。

`pycocotools` 评估器会同时输出 box 和 mask 的 mAP@IoU=0.5:0.95；你需要同时关注这两个数值，才能判断是 box head 还是 mask head 成为瓶颈。

## 交付

本课产出：

- `outputs/prompt-instance-vs-semantic-router.md` — 一个通过三个问题判断应使用实例分割、语义分割还是全景分割，并给出应首选模型的提示词。
- `outputs/skill-mask-rcnn-head-swapper.md` — 一个 skill，根据新的 `num_classes` 生成用于替换任意 torchvision 检测模型 head 的 10 行代码。

## 练习

1. **（简单）** 在 100 个随机框上验证你的 RoIAlign 与 `torchvision.ops.roi_align` 的一致性。报告最大绝对差值。同时运行 RoIPool（2017 年前的行为），并展示它在靠近边界的框上会偏差约 1-2 个特征像素。
2. **（中等）** 在 50 张图像的自定义数据集上微调 `maskrcnn_resnet50_fpn_v2`（任意两个类别：气球、鱼、坑洼、logo 均可）。冻结 backbone，训练 20 个 epoch，报告 mask AP@0.5。
3. **（困难）** 将 Mask R-CNN 的 mask head 替换为输出 56x56 而非 28x28 的版本。测量替换前后的 mAP@IoU=0.75。解释性能提升（或没有提升）为何符合边界精度 / 内存开销的权衡预期。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------------|----------------------|
| Mask R-CNN | “检测加掩码” | Faster R-CNN + 一个为每个候选框、每个类别预测 28x28 掩码的小型 FCN head |
| FPN | “特征金字塔” | 自顶向下 + 横向连接，使每个步长层级都拥有 C 个通道的语义丰富特征 |
| RPN | “区域提议网络” | 一个小的卷积头，每张图像生成约 1000 个有目标 / 无目标的候选框 |
| RoIAlign | “无取整裁剪” | 从任意浮点坐标框中双线性采样固定大小特征网格 |
| RoIPool | “2017 年前的裁剪” | 用途与 RoIAlign 相同，但会对框坐标取整；已过时 |
| Mask AP | “实例 mAP” | 使用掩码 IoU 而非框 IoU 计算的平均精度；COCO 实例分割指标 |
| 二值 mask head | “每类掩码” | 为每个候选框预测每个类别的一个二值掩码；只保留预测类别对应的通道 |
| 背景类 | “类别 0” | 通用的“无目标”类别；真实类别的索引从 1 开始 |

## 延伸阅读

- [Mask R-CNN（He 等人，2017）](https://arxiv.org/abs/1703.06870) — 原论文；第 3 节关于 RoIAlign 是必读内容
- [FPN：Feature Pyramid Networks（Lin 等人，2017）](https://arxiv.org/abs/1612.03144) — FPN 论文；现代检测器都在使用它
- [torchvision Mask R-CNN 教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) — 微调循环的权威参考
- [Detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) — 生产级实现，包含几乎所有检测与分割变体的训练权重
