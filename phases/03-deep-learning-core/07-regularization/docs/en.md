# 正则化

> 你的模型在训练数据上达到 99%，在测试数据上只有 60%。它记住了而不是学会了。正则化是你对复杂度征收的税，用以强制泛化。

**类型：** 构建  
**语言：** Python  
**前置条件：** 第 03.06 课（优化器）  
**时间：** 约 75 分钟

## 学习目标

- 从零实现带反向缩放（inverted scaling）的 Dropout、L2 权重衰减、批归一化、层归一化和 RMSNorm
- 通过正则化实验衡量训练-测试准确率差距并诊断过拟合
- 解释为什么 Transformer 使用 LayerNorm 而不是 BatchNorm，以及为什么现代 LLM 更倾向使用 RMSNorm
- 根据过拟合的严重程度，应用正确的正则化技术组合

## 问题

一个具有足够参数的神经网络可以记住任何数据集。这并非假设——Zhang 等人（2017）通过在带有随机标签的 ImageNet 上训练标准网络证明了这一点。这些网络在完全随机的标签分配上达到了接近零的训练损失。它们记住了一百万个没有规律可学的随机输入-输出对。训练损失完美。测试准确率为零。

这就是过拟合问题，而且随着模型变大，问题会变得更严重。GPT-3 有 1750 亿个参数。训练集大约有 5000 亿个 token。拥有这么多参数，模型有足够的能力逐字记住训练数据的重要部分。没有正则化，它只会照搬训练样本，而不是学习可泛化的模式。

训练性能和测试性能之间的差距就是过拟合差距。本课中的每一项技术都从不同角度攻击这个差距。Dropout 强制网络不依赖任何单个神经元。权重衰减防止任何单个权重变得过大。批归一化使损失曲面更平滑，从而使优化器找到更平坦、更易泛化的最小值。层归一化做同样的事，但在批归一化失效的地方（小批次、变长序列）也能工作。RMSNorm 通过去掉均值计算使其速度快 10%。每项技术都很简单。合在一起，它们就是“记住的模型”和“泛化的模型”之间的区别。

## 概念

### 过拟合谱

每个模型都处于从欠拟合（太简单而无法捕捉模式）到过拟合（太复杂以至于捕捉到噪声）的谱系上的某个位置。最佳点在中间，正则化从过拟合一侧将模型推向那里。

```mermaid
graph LR
    Under["欠拟合<br/>训练: 60%<br/>测试: 58%<br/>模型太简单"] --> Good["良好拟合<br/>训练: 95%<br/>测试: 92%<br/>泛化良好"]
    Good --> Over["过拟合<br/>训练: 99.9%<br/>测试: 65%<br/>记住了噪声"]

    Dropout["Dropout"] -->|"向左推"| Over
    WD["权重衰减"] -->|"向左推"| Over
    BN["BatchNorm"] -->|"向左推"| Over
    Aug["数据增强"] -->|"向左推"| Over
```

### Dropout

最简单的正则化技术，拥有最优雅的解释。在训练期间，以概率 p 随机将每个神经元的输出置为零。

```
output = activation(z) * mask    其中 mask[i] ~ Bernoulli(1 - p)
```

当 p = 0.5 时，每次前向传播中有一半神经元被置零。网络必须学习冗余表示，因为它无法预测哪些神经元可用。这防止了协同适应——神经元学会依赖特定其他神经元的存在。

集成解释：一个有 N 个神经元并带有 dropout 的网络可以创建 2^N 个子网络（每个神经元开或关的每一种组合）。带 dropout 的训练近似同时训练全部 2^N 个子网络，每个子网络在不同的小批量上训练。在测试时，你使用所有神经元（无 dropout），并将输出缩放 (1 - p) 以匹配训练期间的期望值。这相当于对 2^N 个子网络的预测进行平均——单个模型产生的巨大集成。

在实践中，缩放是在训练期间而不是测试期间应用的（反向 dropout）：

```
训练期间：  output = activation(z) * mask / (1 - p)
测试期间：  output = activation(z)   （无需更改）
```

这样更干净，因为测试代码完全不需要知道 dropout。

默认比率：Transformer 用 p = 0.1，MLP 用 p = 0.5，CNN 用 p = 0.2-0.3。较高的 dropout = 较强的正则化 = 更大的欠拟合风险。

### 权重衰减（L2 正则化）

将所有权重的平方幅值加入损失：

```
total_loss = task_loss + (lambda / 2) * sum(w_i^2)
```

正则化项的梯度为 lambda * w。这意味着每一步，每个权重都按与其幅值成比例的量向零收缩。大权重受到更多惩罚。模型被推向没有一个权重占主导地位的解。

为什么这有助于泛化：过拟合模型往往有大权重，这些权重会放大训练数据中的噪声。权重衰减保持小权重，这限制了模型的有效容量，并迫使它依赖稳健、可泛化的特征，而不是记住的瑕疵。

lambda 超参数控制强度。典型值：

- Transformer 的 AdamW 用 0.01
- CNN 的 SGD 用 1e-4
- 严重过拟合的模型用 0.1

如第 06 课所述：权重衰减和 L2 正则化在 SGD 中等价，但在 Adam 中不等价。使用 Adam 训练时始终使用 AdamW（解耦权重衰减）。

### 批归一化

在将每一层的输出传递给下一层之前，在小批量维度上进行归一化。

对于某一层上的一个小批量激活值：

```
mu = (1/B) * sum(x_i)           （批次均值）
sigma^2 = (1/B) * sum((x_i - mu)^2)   （批次方差）
x_hat = (x_i - mu) / sqrt(sigma^2 + eps)   （归一化）
y = gamma * x_hat + beta        （缩放和平移）
```

Gamma 和 beta 是可学习参数，让网络在最优时撤消归一化。没有它们，你就是在强制每一层的输出为零均值单位方差，这可能不是网络想要的。

**训练与推理的分裂：** 训练期间，mu 和 sigma 来自当前小批量。推理期间，你使用训练期间累积的运行平均值（动量为 0.1 的指数移动平均，即 90% 旧值 + 10% 新值）。

BatchNorm 为什么有效仍然有争议。原始论文声称它减少了“内部协变量偏移”（层输入分布随前层更新而变化）。Santurkar 等人（2018）表明这种解释是错误的。实际原因：BatchNorm 使损失曲面更平滑。梯度更具预测性，Lipschitz 常数更小，优化器可以安全地迈出更大的步长。这就是为什么 BatchNorm 允许你使用更高的学习率并更快收敛。

BatchNorm 有一个根本限制：它依赖于批次统计。当批次大小为 1 时，均值和方差是没有意义的。当批次较小（< 32）时，统计量有噪声并损害性能。这对物体检测（内存限制批次大小）和语言建模（序列长度变化）等任务很重要。

### 层归一化

跨特征而非跨批次进行归一化。对于单个样本：

```
mu = (1/D) * sum(x_j)           （特征均值）
sigma^2 = (1/D) * sum((x_j - mu)^2)   （特征方差）
x_hat = (x_j - mu) / sqrt(sigma^2 + eps)
y = gamma * x_hat + beta
```

D 是特征维度。每个样本独立归一化——不依赖于批次大小。这就是为什么 Transformer 使用 LayerNorm 而不是 BatchNorm。序列长度可变，批次大小通常很小（或在生成时为 1），训练和推理之间的计算完全相同。

Transformer 中的 LayerNorm 应用在每个自注意力块和前馈块之后（Post-LN），或它们之前（Pre-LN，对训练更稳定）。

### RMSNorm

不带均值减法的 LayerNorm。由 Zhang & Sennrich（2019）提出。

```
rms = sqrt((1/D) * sum(x_j^2))
y = gamma * x / rms
```

仅此而已。没有均值计算，没有 beta 参数。观察发现：LayerNorm 中的重新中心化（均值减法）对模型性能贡献甚微，但消耗计算资源。移除它可以在相同准确率下节省约 10% 的开销。

LLaMA、LLaMA 2、LLaMA 3、Mistral 以及大多数现代 LLM 使用 RMSNorm 而不是 LayerNorm。在数十亿参数和数万亿 token 的规模下，这 10% 的节省是显著的。

### 归一化方法对比

```mermaid
graph TD
    subgraph "批归一化"
        BN_D["跨 BATCH 归一化<br/>对每个特征"]
        BN_S["批次: [x1, x2, x3, x4]<br/>特征 1: 归一化 [x1f1, x2f1, x3f1, x4f1]"]
        BN_P["需要 batch > 32<br/>训练与评估不同<br/>用于 CNN"]
    end
    subgraph "层归一化"
        LN_D["跨 FEATURES 归一化<br/>对每个样本"]
        LN_S["样本 x1: 归一化 [f1, f2, f3, f4]"]
        LN_P["与批次无关<br/>训练与评估相同<br/>用于 Transformer"]
    end
    subgraph "RMS 归一化"
        RN_D["类似于 LayerNorm<br/>但跳过均值减法"]
        RN_S["只除以 RMS<br/>无中心化"]
        RN_P["比 LayerNorm 快 10%<br/>准确率相同<br/>用于 LLaMA, Mistral"]
    end
```

### 数据增强作为正则化

不是模型修改，而是数据修改。在保持标签的同时变换训练输入：

- 图像：随机裁剪、翻转、旋转、颜色抖动、cutout
- 文本：同义词替换、回译、随机删除
- 音频：时间拉伸、音高偏移、添加噪声

效果与正则化相同：它增加了训练集的有效大小，使模型更难记住特定样本。一个只以原始形式看到每张图像一次的模型可以记住它。一个看到每张图像 50 个增强版本的模型被迫学习不变结构。

### 早停法

最简单的正则化器：当验证损失开始增加时停止训练。此时模型尚未过拟合。在实践中，你每轮跟踪验证损失，保存最佳模型，并继续训练一个“耐心”窗口（通常为 5-20 轮）。如果验证损失在耐心窗口内没有改善，你就停止并加载保存的最佳模型。

### 何时应用什么

```mermaid
flowchart TD
    Gap{"训练-测试<br/>准确率差距？"} -->|"> 10%"| Heavy["强正则化"]
    Gap -->|"5-10%"| Medium["中等正则化"]
    Gap -->|"< 5%"| Light["轻度正则化"]

    Heavy --> D5["Dropout p=0.3-0.5"]
    Heavy --> WD2["权重衰减 0.01-0.1"]
    Heavy --> Aug["激进的数据增强"]
    Heavy --> ES["早停法"]

    Medium --> D3["Dropout p=0.1-0.2"]
    Medium --> WD1["权重衰减 0.001-0.01"]
    Medium --> Norm["BatchNorm 或 LayerNorm"]

    Light --> D1["Dropout p=0.05-0.1"]
    Light --> WD0["权重衰减 1e-4"]
```

## 动手实现

### 第 1 步：Dropout（训练和评估模式）

```python
import random
import math


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training:
            return list(x)
        self.mask = []
        output = []
        for val in x:
            if random.random() < self.p:
                self.mask.append(0)
                output.append(0.0)
            else:
                self.mask.append(1)
                output.append(val / (1 - self.p))
        return output

    def backward(self, grad_output):
        grads = []
        for g, m in zip(grad_output, self.mask):
            if m == 0:
                grads.append(0.0)
            else:
                grads.append(g / (1 - self.p))
        return grads
```

### 第 2 步：L2 权重衰减

```python
def l2_regularization(weights, lambda_reg):
    penalty = 0.0
    for w in weights:
        penalty += w * w
    return lambda_reg * 0.5 * penalty

def l2_gradient(weights, lambda_reg):
    return [lambda_reg * w for w in weights]
```

### 第 3 步：批归一化

```python
class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.training = True
        self.num_features = num_features

    def forward(self, batch):
        batch_size = len(batch)
        if self.training:
            mean = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            for j in range(self.num_features):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            mean = list(self.running_mean)
            var = list(self.running_var)

        self.x_hat = []
        output = []
        for sample in batch:
            normalized = []
            out_sample = []
            for j in range(self.num_features):
                x_h = (sample[j] - mean[j]) / math.sqrt(var[j] + self.eps)
                normalized.append(x_h)
                out_sample.append(self.gamma[j] * x_h + self.beta[j])
            self.x_hat.append(normalized)
            output.append(out_sample)
        return output
```

### 第 4 步：层归一化

```python
class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)

        self.x_hat = []
        output = []
        for j in range(self.num_features):
            x_h = (x[j] - mean) / math.sqrt(var + self.eps)
            self.x_hat.append(x_h)
            output.append(self.gamma[j] * x_h + self.beta[j])
        return output
```

### 第 5 步：RMSNorm

```python
class RMSNorm:
    def __init__(self, num_features, eps=1e-6):
        self.gamma = [1.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        rms = math.sqrt(sum(xi * xi for xi in x) / len(x) + self.eps)
        output = []
        for j in range(self.num_features):
            output.append(self.gamma[j] * x[j] / rms)
        return output
```

### 第 6 步：带与不带正则化的训练

```python
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class RegularizedNetwork:
    def __init__(self, hidden_size=16, lr=0.05, dropout_p=0.0, weight_decay=0.0):
        random.seed(0)
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0 else None

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x, training=True):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        if self.dropout and training:
            self.dropout.training = True
            self.h = self.dropout.forward(self.h)
        elif self.dropout:
            self.dropout.training = False
            self.h = self.dropout.forward(self.h)

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * (d_out * self.h[i] + self.weight_decay * self.w2[i])
            for j in range(2):
                self.w1[i][j] -= self.lr * (d_h * self.x[j] + self.weight_decay * self.w1[i][j])
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def evaluate(self, data):
        correct = 0
        total_loss = 0.0
        for x, y in data:
            pred = self.forward(x, training=False)
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            if (pred >= 0.5) == (y >= 0.5):
                correct += 1
        return total_loss / len(data), correct / len(data) * 100

    def train_model(self, train_data, test_data, epochs=300):
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in train_data:
                pred = self.forward(x, training=True)
                self.backward(y)
                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            train_loss = total_loss / len(train_data)
            train_acc = correct / len(train_data) * 100
            test_loss, test_acc = self.evaluate(test_data)
            history.append((train_loss, train_acc, test_loss, test_acc))
            if epoch % 75 == 0 or epoch == epochs - 1:
                gap = train_acc - test_acc
                print(f"    轮次 {epoch:3d}: train_acc={train_acc:.1f}%, test_acc={test_acc:.1f}%, gap={gap:.1f}%")
        return history
```

## 如何使用（PyTorch）

PyTorch 以模块形式提供了所有归一化和正则化方法：

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),
)

model.train()
out_train = model(torch.randn(32, 784))

model.eval()
out_test = model(torch.randn(1, 784))
```

`model.train()` / `model.eval()` 切换至关重要。它打开/关闭 dropout，并告诉 BatchNorm 使用批次统计还是运行统计。在推理前忘记调用 `model.eval()` 是深度学习中常见的错误之一。你的测试准确率会随机波动，因为 dropout 仍然活跃，且 BatchNorm 在使用小批次统计量。

对于 Transformer，模式不同：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.ff(x))
        return x
```

LayerNorm，而不是 BatchNorm。Dropout p=0.1，而不是 p=0.5。这些是 Transformer 的默认值。

## 交付内容

本课程产出：
- `outputs/prompt-regularization-advisor.md` —— 一个诊断过拟合并推荐正确正则化策略的提示

## 练习

1. 实现 2D 数据的空间 dropout：不丢弃单个神经元，而是丢弃整个特征通道。将连续特征的分组作为通道处理并丢弃整组来模拟它。在 hidden_size=32 的圆形数据集上，将其与标准 dropout 的训练-测试差距进行比较。

2. 将第 05 课的标签平滑与本课的 dropout 结合。在四种配置下训练：两者都不用、仅 dropout、仅标签平滑、两者都用。测量每种配置的最终训练-测试准确率差距。哪种组合给出的差距最小？

3. 在圆形数据集网络的隐藏层和激活函数之间添加一个 BatchNorm 层。在 0.01、0.05 和 0.1 的学习率下，使用和不使用 BatchNorm 进行训练。BatchNorm 应允许在更高的学习率下稳定训练，而普通网络会发散。

4. 实现早停法：每轮跟踪测试损失，保存最佳权重，如果测试损失在 20 轮内没有改善则停止。将正则化网络训练 1000 轮。报告哪一轮的测试准确率最好，以及你节省了多少轮计算。

5. 在一个 4 层网络（不仅仅是 2 层）上比较 LayerNorm 与 RMSNorm。使用相同的权重初始化两者。训练 200 轮并比较最终准确率、训练速度（每轮时间）和第一层的梯度幅值。验证 RMSNorm 在相同准确率下更快。

## 关键术语

| 术语 | 人们常说的话 | 实际含义 |
|------|----------------|----------------------|
| 过拟合 | “模型记住了数据” | 当模型的训练性能显著超过测试性能时，表明它学到了噪声而非信号 |
| 正则化 | “防止过拟合” | 任何约束模型复杂度以改善泛化的技术：dropout、权重衰减、归一化、增强等 |
| Dropout | “随机删除神经元” | 在训练期间以概率 p 随机将神经元输出置零，强制冗余表示；等价于训练一个集成 |
| 权重衰减 | “L2 惩罚” | 每一步通过减去 lambda * w 将所有权重向零收缩；通过权重幅值惩罚复杂度 |
| 批归一化 | “按批次归一化” | 在批次维度上归一化层输出，训练时使用批次统计，推理时使用运行平均值 |
| 层归一化 | “按样本归一化” | 在每个样本内部跨特征归一化；与批次无关，用于批次大小变化的 Transformer |
| RMSNorm | “LayerNorm 去掉均值” | 均方根归一化；去掉 LayerNorm 中的均值减法，以获得相同准确率下 10% 的速度提升 |
| 早停法 | “在过拟合前停止” | 当验证损失停止改善时暂停训练；最简单的正则化器，常与其他方法配合使用 |
| 数据增强 | “从更少数据产生更多数据” | 变换训练输入（翻转、裁剪、噪声）以增加有效数据集大小并强制学习不变性 |
| 泛化差距 | “训练-测试差距” | 训练性能和测试性能之间的差异；正则化旨在最小化这一差距 |

## 进一步阅读

- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014) —— 原始的 dropout 论文，包含集成解释和大量实验
- Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (2015) —— 引入了 BatchNorm 及其训练过程，是引用最多的深度学习论文之一
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) —— 表明 RMSNorm 在降低计算量的同时匹配了 LayerNorm 的准确率；被 LLaMA 和 Mistral 采用
- Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization" (2017) —— 里程碑式论文，显示神经网络可以记忆随机标签，挑战了关于泛化的传统观点