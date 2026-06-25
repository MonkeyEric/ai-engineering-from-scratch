# 图像分类

> 分类器是一个从像素到类别概率分布的函数。其余一切都是工程细节。

**类型：** Build
**语言：** Python
**前置知识：** Phase 2 Lesson 09（模型评估）、Phase 3 Lesson 10（迷你框架）、Phase 4 Lesson 03（CNN）
**时间：** ~75 分钟

## 学习目标

- 在 CIFAR-10 上构建端到端图像分类流程：数据集、数据增强、模型、训练循环、评估
- 解释每个组件（数据加载器、损失函数、优化器、学习率调度器、数据增强）的作用，并预测其中任意一个出问题时会如何在损失曲线上表现出来
- 从零实现 mixup、cutout 和标签平滑，并论证各自何时值得加入
- 阅读混淆矩阵和逐类精确率/召回率表，以在总体准确率之外诊断数据集与模型的失败

## 问题背景

所有真正交付的视觉任务，在某种意义上都会归结为图像分类。检测是对区域进行分类。分割是对像素进行分类。检索是按与类别中心的相似度排序。把分类做对——数据集循环、增强策略、损失函数、评估方式——是贯穿本阶段所有其他任务的可迁移技能。

大多数分类 bug 并不在模型里，而是在流程中：归一化写错、训练集未打乱、增强扭曲了标签、验证集被训练数据污染、学习率在第 30 个 epoch 后悄然发散。一个配置正确时能在 CIFAR-10 上达到 93% 的 CNN，在配置错误时通常只能拿到 70–75%，而且损失曲线看起来一直都很合理。

本课手工搭建整个流程，使每个部分都可检查。你不会使用 `torchvision.datasets` 中任何可能隐藏 bug 的组件。

## 核心概念

### 分类流程

```mermaid
flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

流程中的每一行都是 bug 可能藏身的地方。交叉熵接收原始对数几率，而不是 softmax 输出，因此任何在损失前写 `model(x).softmax()` 的代码都会悄悄计算出错误的梯度。增强只作用于输入，不作用于标签——mixup 除外，它同时混合两者。`optimizer.zero_grad()` 每步只能调用一次；跳过它会导致梯度累积，看起来像学习率剧烈波动。这些 bug 都会让学习曲线变平，却不会抛出错误。

### 交叉熵、对数几率与 softmax

分类器每张图像输出 `C` 个数字，称为对数几率（logits）。应用 softmax 后它们变成概率分布：

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

交叉熵衡量正确类别的负对数概率：

```
CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )
```

右边是数值稳定形式（log-sum-exp）。PyTorch 的 `nn.CrossEntropyLoss` 将 softmax 与 NLL 融合为一个操作，并直接接收原始对数几率。先手动应用 softmax 几乎一定是 bug——你会计算 log(softmax(softmax(z)))，这是一个无意义的量。

### 为什么数据增强有效

CNN 对平移具有归纳偏置（来自权重共享），但对裁剪、翻转、颜色抖动或遮挡没有内置不变性。教会它这些不变性的唯一方法，是展示能触发这些变化的像素。训练时的每一次随机变换都在说：“这两张图像标签相同；去学习忽略差异的特征。”

```
原始裁剪：  "朝左的狗"
翻转：       "朝右的狗"       <- 相同标签，不同像素
旋转(+15)： "略有倾斜的狗"
颜色抖动： "暖光下的狗"
RandomErasing："缺失一块的狗"
```

规则是：增强必须保持标签不变。在数字数据集上，cutout 和旋转可能把 "6" 翻成 "9"；这时应使用更小的旋转范围，并选择尊重数字特定不变性的增强。

### Mixup 与 Cutmix

普通增强变换像素，但标签仍保持 one-hot。**Mixup** 和 **cutmix** 打破这一点，同时对输入和标签进行插值。

```
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  paste a random rectangle of x_j into x_i
  y = area-weighted mix of y_i and y_j
```

为什么有效：模型不再记忆尖锐的 one-hot 目标，而是学习在类别之间插值。训练损失上升，测试准确率上升。这是给任何分类器最便宜的鲁棒性升级。

### 标签平滑

Mixup 的近亲。不针对 `[0, 0, 1, 0, 0]` 训练，而是针对 `[eps/C, eps/C, 1-eps, eps/C, eps/C]`，其中 `eps` 取较小值如 0.1。阻止模型产生任意尖锐的对数几率，并以几乎零成本改善校准。自 PyTorch 1.10 起内置于 `nn.CrossEntropyLoss(label_smoothing=0.1)`。

### 超越准确率的评估

总体准确率会掩盖类别不平衡。一个 90-10 的二分类器如果总是预测多数类，准确率也是 90%。真正能告诉你发生什么的工具：

- **逐类准确率** —— 每个类别一个数字；立刻暴露表现不佳的类别。
- **混淆矩阵** —— C×C 网格，其中第 i 行第 j 列 = 真实类别 i 被预测为类别 j 的次数；对角线是正确预测，非对角线是模型犯错的地方。
- **Top-1 / Top-5** —— 正确类别是否在前 1 或前 5 个预测中；Top-5 对 ImageNet 很重要，因为 "Norwich terrier" 与 "Norfolk terrier" 等类别确实难以区分。
- **校准（ECE）** —— 置信度为 0.8 的预测是否真的有 80% 正确？现代网络系统性地过于自信；可通过温度缩放或标签平滑修正。

## 动手实现

### 第 1 步：确定性合成数据集

CIFAR-10 存储在磁盘上。为了让本课可复现且快速，我们构建一个看起来像 CIFAR 的合成数据集：32×32 RGB 图像，每类有特定结构需要模型学习。完全相同的流程可直接用于真实 CIFAR-10。

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

每个类别有自己的调色板和频率模式，加上高斯噪声迫使模型学习信号而非记忆像素。十个类别，每类一千张图像，并已随机打乱。

### 第 2 步：归一化与数据增强

所有视觉流程都包含的两种变换。

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

裁剪前先进行 reflect 填充，而不是零填充，因为黑色边框会成为模型学会以非有用方式忽略的信号。

### 第 3 步：Mixup

在训练步骤中混合两张图像和两个标签。实现为批次变换，因此它位于前向传播旁边，而不是数据集内部。

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` 是面向软标签分布的交叉熵。当目标恰好是 one-hot 时，它退化为通常情况。

### 第 4 步：训练循环

完整配方：遍历一次数据，每批梯度只算一次，每个 epoch 调度器步进一次。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # Training accuracy vs the un-mixed labels `y` is only an approximation
        # when mixup is on (the model saw soft targets, not y). Treat it as a
        # rough progress signal; rely on val accuracy for real performance.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

每次写训练循环时都要检查的五个不变式：

1. 训练前 `model.train()`，评估前 `model.eval()` —— 会改变 dropout 和 batchnorm 的行为。
2. `.zero_grad()` 在 `.backward()` 之前。
3. 累积指标时使用 `.item()`，避免保留计算图。
4. 评估时用 `@torch.no_grad()` —— 节省内存和时间，防止隐蔽错误。
5. 对原始对数几率取 argmax，而非 softmax —— 结果相同，少一次操作。

### 第 5 步：整合

使用上一课的 `TinyResNet`，训练几个 epoch，然后评估。

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet comes from the previous lesson (03-cnns-lenet-to-resnet).
# Adjust the import path to wherever you stored the previous lesson's code.
from cnns_lenet_to_resnet import TinyResNet  # example placeholder

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

在合成数据集上，这能在 5 个 epoch 内达到接近完美的验证准确率，这正是关键所在：流程正确，模型就能学到可学的东西。将数据集替换为真实 CIFAR-10，同样的循环无需修改即可训练到约 90%。

### 第 6 步：阅读混淆矩阵

仅靠准确率无法告诉你模型在哪里失败。混淆矩阵可以。

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

行是真实类别，列是预测。如果类别 3 和 5 之间出现一团非对角计数，说明模型会混淆这两个类别，并为你提供针对性收集数据或设计类别特定增强的起点。

## 使用它

`torchvision` 把上述所有内容封装成语义化的组件。对于真实 CIFAR-10，完整流程只需四行加训练循环。

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

注意两点：mean/std 是**数据集专属**的——它们在 CIFAR-10 训练集上计算，而不是 ImageNet——并且 reflect 填充是社区默认的裁剪策略。在这里直接复制 ImageNet 的统计值会导致约 1% 的准确率损失，除非有人去剖析模型，否则没人会发现。

## 交付

本课产出：

- `outputs/prompt-classifier-pipeline-auditor.md` —— 一个提示词，用于审计训练脚本是否满足上述五个不变式，并指出第一个违反项。
- `outputs/skill-classification-diagnostics.md` —— 一个 skill，给定混淆矩阵和类别名称列表后，总结逐类失败并提出单一最具影响力的修复建议。

## 练习

1. **（简单）** 在合成数据集上用相同模型分别训练 5 个 epoch，一次使用 mixup，一次不用。绘制两者的训练损失和验证损失。解释为什么使用 mixup 时训练损失更高，但验证准确率却相近或更好。
2. **（中等）** 实现 Cutout——在每张训练图像中随机清零一个 8×8 的正方形——并做消融实验：无增强、hflip+crop、hflip+crop+cutout、hflip+crop+mixup。报告每种情况的验证准确率。
3. **（困难）** 构建 CIFAR-100 流程（100 个类别，相同输入尺寸），复现 ResNet-34 训练，使其达到已发布准确率的 1% 范围内。加分项：扫描三个学习率和两个权重衰减，记录到本地 CSV，并生成最终的混淆矩阵-主要错误表。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| Logits | "Raw outputs" | 每张图像在 softmax 前的 C 维向量；交叉熵期望的是它，而不是 softmax 后的值 |
| Cross-entropy | "The loss" | 正确类别的负对数概率；将 log-softmax 与 NLL 融合为一个稳定操作 |
| DataLoader | "The batcher" | 用洗牌、分批和（可选）多进程加载包装数据集；一半的训练 bug 都怪到它头上 |
| Augmentation | "Random transforms" | 训练时任何保持标签不变的像素级变换；教会 CNN 本身不具备的不变性 |
| Mixup / Cutmix | "Mix two images" | 同时混合输入和标签，使分类器学习平滑插值而非硬边界 |
| Label smoothing | "Softer targets" | 用 (1-eps, eps/(C-1), ...) 替换 one-hot；改善校准并轻微提升准确率 |
| Top-k accuracy | "Top-5" | 正确类别在前 k 个最高概率预测中；用于类别确实模糊的数据集 |
| Confusion matrix | "Where errors live" | C×C 表格，其中条目 (i, j) 统计真实类别 i 被预测为 j 的次数；对角线表示正确，非对角线提示该修什么 |

## 延伸阅读

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) —— 单页最清晰的训练流程概览
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) —— 各种小技巧合在一起能为 ImageNet 上的 ResNet 带来 3-4% 准确率提升
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) —— 原始 mixup 论文；三页理论加令人信服的实验
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) —— 证明现代网络校准不佳并用一个标量参数修复的论文
