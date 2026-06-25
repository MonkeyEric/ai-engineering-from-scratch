# 优化器

> 梯度下降告诉你该往哪个方向移动。但它从不说明该走多远或多快。SGD 是指南针。Adam 是带有实时路况信息的 GPS。

**类型：** 构建  
**语言：** Python  
**前置条件：** 第 03.05 课（损失函数）  
**时间：** 约 75 分钟

## 学习目标

- 用 Python 从零实现 SGD、带动量的 SGD、Adam 和 AdamW 优化器
- 解释 Adam 的偏置校正如何补偿训练初期零初始化的矩估计
- 证明为什么在相同任务上，AdamW 比带 L2 正则化的 Adam 产生更好的泛化能力
- 为 Transformer、CNN、GAN 和微调任务选择合适的优化器和默认超参数

## 问题

你已经计算出了梯度。你知道权重 #4,721 应该减少 0.003 以降低损失。但 0.003 是以什么为单位？按什么缩放？你应该在第 1 步和第 1,000 步移动相同的量吗？

普通梯度下降在每一步对每个参数应用相同的学习率：w = w - lr * gradient。这在实践中给神经网络训练带来了三个棘手的问题。

第一，振荡。损失曲面很少像一个光滑的碗。它更像一条又长又窄的山谷。梯度指向山谷的横向（陡峭方向），而不是沿着山谷（平缓方向）。梯度下降在狭窄维度上来回弹跳，而在有用的维度上却进展缓慢。你见过这种情况：损失快速下降然后停滞，不是因为模型收敛了，而是因为它在振荡。

第二，所有参数共用一个学习率是错误的。有些权重需要大的更新（它们处于早期欠拟合阶段）。另一些则需要微小的更新（它们接近最优值）。对前者有效的学习率会毁掉后者，反之亦然。

第三，鞍点。在高维空间中，损失曲面存在广阔的平坦区域，梯度接近零。普通 SGD 以梯度的速度穿过这些区域，而速度实际上就是零。模型看起来卡住了。它并没有卡住——它在一个平坦区域，另一侧是有用的下坡。但 SGD 没有机制能推动它穿过去。

Adam 解决了全部三个问题。它为每个参数维护两个移动平均——均值梯度（动量，处理振荡）和均方梯度（自适应学习率，处理不同尺度）。再结合前几步的偏置校正，它用一个优化器配合默认超参数就能解决 80% 的问题。本节课将从零构建它，让你准确理解它何时以及为什么在另外 20% 的问题上失效。

## 概念

### 随机梯度下降（SGD）

最简单的优化器。计算小批量上的梯度，并朝相反方向迈步。

```
w = w - lr * gradient
```

“随机”意味着你使用数据的随机子集（小批量）来估计梯度，而不是整个数据集。这种噪声实际上是有用的——它有助于逃离尖锐的局部最小值。但噪声也会引起振荡。

学习率是唯一的旋钮。太高：损失发散。太低：训练永远完成不了。最优值取决于架构、数据、批次大小和当前训练阶段。对于现代网络上的普通 SGD，典型值范围在 0.01 到 0.1 之间。但即便在单次训练运行中，理想的学习率也会变化。

### 动量

球滚下山的比喻虽然老套，但很准确。你不是仅仅依靠梯度来迈步，而是维持一个累积过去梯度的速度。

```
m_t = beta * m_{t-1} + gradient
w = w - lr * m_t
```

Beta（通常为 0.9）控制保留多少历史信息。当 beta = 0.9 时，动量大约是最新 10 个梯度的平均值（1 / (1 - 0.9) = 10）。

为什么这能解决振荡：指向同一方向的梯度会累积。方向翻转的梯度会相互抵消。在那个狭窄的山谷中，“横向”分量每步都改变符号并被抑制。“纵向”分量保持一致并被放大。结果是在有用的方向上平稳加速。

实际数字：在病态条件的损失曲面上，单独的 SGD 可能需要 10,000 步。带动量（beta=0.9）的 SGD 在相同问题上通常只需要 3,000-5,000 步。加速不是边际的。

### RMSProp

第一个真正有效的逐参数自适应学习率方法。由 Hinton 在 Coursera 讲座中提出（从未正式发表）。

```
s_t = beta * s_{t-1} + (1 - beta) * gradient^2
w = w - lr * gradient / (sqrt(s_t) + epsilon)
```

s_t 跟踪平方梯度的移动平均。具有持续大梯度的参数被一个大数除（有效学习率更小）。具有小梯度的参数被一个小数除（有效学习率更大）。

这解决了“所有参数共用一个学习率”的问题。一个已经得到大更新的权重可能接近目标——慢下来。一个只得到微小更新的权重可能训练不足——快起来。

Epsilon（通常为 1e-8）防止参数未被更新时除以零。

### Adam：动量 + RMSProp

Adam 结合了两种思想。它为每个参数维护两个指数移动平均：

```
m_t = beta1 * m_{t-1} + (1 - beta1) * gradient        （一阶矩：均值）
v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2       （二阶矩：方差）
```

**偏置校正**是大多数解释跳过的关键细节。在第 1 步，m_1 = (1 - beta1) * gradient。当 beta1 = 0.9 时，这只有 0.1 * gradient——小了十倍。移动平均还没有预热。偏置校正进行补偿：

```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

在第 1 步，beta1 = 0.9 时：m_hat = m_1 / (1 - 0.9) = m_1 / 0.1 = 实际梯度。在第 100 步：(1 - 0.9^100) 约等于 1.0，所以校正消失。偏置校正在前约 10 步很重要，约 50 步后就不相关了。

更新规则：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

Adam 默认值：lr = 0.001，beta1 = 0.9，beta2 = 0.999，epsilon = 1e-8。这些默认值对 80% 的问题有效。当它们无效时，首先改变 lr，然后是 beta2。几乎从不改变 beta1 或 epsilon。

### AdamW：正确的权重衰减

L2 正则化在损失中添加 lambda * w^2。在普通 SGD 中，这等价于权重衰减（每一步从权重中减去 lambda * w）。在 Adam 中，这种等价性被打破。

Loshchilov 和 Hutter 的洞察：当你把 L2 加入损失，然后 Adam 处理梯度时，自适应学习率也会缩放正则化项。梯度方差大的参数得到较少的正则化。方差小的参数得到更多的正则化。这不是你想要的——你希望无论梯度统计如何，正则化都是均匀的。

AdamW 通过在 Adam 更新之后直接将权重衰减应用于权重来修复这个问题：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * lambda * w
```

权重衰减项（lr * lambda * w）不被 Adam 的自适应因子缩放。每个参数得到相同的比例收缩。

这看起来像是一个小细节。事实并非如此。AdamW 在几乎所有任务上都比 Adam + L2 正则化收敛到更好的解。它是 PyTorch 中训练 Transformer、扩散模型和大多数现代架构的默认优化器。BERT、GPT、LLaMA、Stable Diffusion——全部使用 AdamW 训练。

### 学习率：最重要的超参数

```mermaid
graph TD
    LR["学习率"] --> TooHigh["太高 (lr > 0.01)"]
    LR --> JustRight["恰到好处"]
    LR --> TooLow["太低 (lr < 0.00001)"]

    TooHigh --> Diverge["损失爆炸<br/>权重 NaN<br/>训练崩溃"]
    JustRight --> Converge["损失稳定下降<br/>达到良好最小值<br/>泛化良好"]
    TooLow --> Stall["损失下降缓慢<br/>困在次优最小值<br/>浪费算力"]

    JustRight --> Schedule["通常需要调度"]
    Schedule --> Warmup["预热：从 0 线性上升到最大值<br/>前 1‑10% 的训练步数"]
    Schedule --> Decay["衰减：随时间降低<br/>余弦或线性"]
```

如果你只调一个超参数，那就调学习率。学习率改变 10 倍，比你做出的任何架构决策都更重要。常见默认值：

- SGD：lr = 0.01 到 0.1
- Adam/AdamW：lr = 1e-4 到 3e-4
- 微调预训练模型：lr = 1e-5 到 5e-5
- 学习率预热：在前 1‑10% 的步数线性上升

### 优化器对比

```mermaid
flowchart LR
    subgraph "优化路径"
        SGD_P["SGD<br/>穿越山谷时振荡<br/>慢但能找平坦最小值"]
        Mom_P["SGD + 动量<br/>路径更平滑<br/>比 SGD 快 3 倍"]
        Adam_P["Adam<br/>逐参数自适应<br/>收敛快"]
        AdamW_P["AdamW<br/>Adam + 正确的衰减<br/>泛化最好"]
    end
    SGD_P --> Mom_P --> Adam_P --> AdamW_P
```

### 每种优化器何时胜出

```mermaid
flowchart TD
    Task["你在训练什么？"] --> Type{"模型类型？"}

    Type -->|"Transformer / LLM"| AdamW["AdamW<br/>lr=1e-4, wd=0.01-0.1"]
    Type -->|"CNN / ResNet"| SGD_M["SGD + 动量<br/>lr=0.1, momentum=0.9"]
    Type -->|"GAN"| Adam2["Adam<br/>lr=2e-4, beta1=0.5"]
    Type -->|"微调"| AdamW2["AdamW<br/>lr=2e-5, wd=0.01"]
    Type -->|"还不知道"| Default["从 AdamW 开始<br/>lr=3e-4, wd=0.01"]
```

## 动手实现

### 第 1 步：普通 SGD

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
```

### 第 2 步：带动量的 SGD

```python
class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
```

### 第 3 步：Adam

```python
import math

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
```

### 第 4 步：AdamW

```python
class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]
```

### 第 5 步：训练对比

在第 05 课的圆形数据集上使用全部四种优化器训练相同的两层网络。比较收敛情况。

```python
import random

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


class OptimizerTestNetwork:
    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_grads(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)
                self.set_params(params)

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    轮次 {epoch:3d}: loss={avg_loss:.4f}, 准确率={accuracy:.1f}%")
        return losses
```

## 如何使用（PyTorch）

PyTorch 优化器处理参数组、梯度裁剪和学习率调度：

```python
import torch
import torch.optim as optim

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.randn(32, 784))
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

模式永远是：zero_grad、forward、loss、backward、（裁剪）、step、（调度）。记住这个顺序。弄错（例如在 optimizer.step() 之前调用 scheduler.step()）是微妙的 bug 的常见来源。

对于 CNN，许多从业者仍然更喜欢 SGD + 动量（lr=0.1, momentum=0.9, weight_decay=1e-4）配合分段或余弦调度。SGD 找到更平坦的最小值，这通常泛化更好。对于 Transformer 和 LLM，带预热 + 余弦衰减的 AdamW 是通用默认值。没有经过衡量的理由，不要与共识对抗。

## 交付内容

本课程产出：
- `outputs/prompt-optimizer-selector.md` —— 一个为任何架构选择正确优化器和学习率的决策提示

## 练习

1. 实现 Nesterov 动量，即在“前视”位置（w - lr * beta * v）而不是当前位置计算梯度。在圆形数据集上比较其与标准动量的收敛情况。

2. 实现学习率预热调度：在前 10% 的训练步数中从 0 线性上升到 max_lr，然后余弦衰减到 0。使用带预热和不带预热的 Adam 进行训练。测量在圆形数据集上达到 90% 准确率需要多少轮。

3. 在 Adam 训练期间跟踪每个参数的有效学习率。有效率为 lr * m_hat / (sqrt(v_hat) + eps)。绘制 10、50 和 200 步后有效率的分布。所有参数的更新速度是否相同？

4. 实现梯度裁剪（按全局范数裁剪）。将最大梯度范数设为 1.0。使用高学习率（对 Adam 用 lr=0.01）训练，对比有无裁剪。在 10 个随机种子上统计有多少次运行发散（损失变为 NaN）。

5. 在一个具有大权重的网络上比较 Adam 与 AdamW。将所有权重初始化为 [-5, 5] 范围内的随机值（比正常大得多）。使用 weight_decay=0.1 训练 200 轮。为两种优化器绘制训练过程中权重的 L2 范数。AdamW 应显示更快的权重收缩。

## 关键术语

| 术语 | 人们常说的话 | 实际含义 |
|------|----------------|----------------------|
| 学习率 | “步长” | 梯度更新上的标量乘数；训练中影响最大的超参数 |
| SGD | “基础梯度下降” | 随机梯度下降：通过减去 lr * gradient 来更新权重，在小批量上计算梯度 |
| 动量 | “滚球类比” | 过去梯度的指数移动平均；抑制振荡并加速一致方向 |
| RMSProp | “自适应学习率” | 将每个参数的梯度除以其最近梯度的均方根；均衡学习率 |
| Adam | “默认优化器” | 结合动量（一阶矩）和 RMSProp（二阶矩），并对初始步进行偏置校正 |
| AdamW | “正确的 Adam” | 带解耦权重衰减的 Adam；直接将正则化应用于权重而非通过梯度 |
| 偏置校正 | “移动平均预热” | 除以 (1 - beta^t) 以补偿 Adam 矩估计的零初始化 |
| 权重衰减 | “收缩权重” | 每一步减去权重值的一部分；一种惩罚大权重的正则化方法 |
| 学习率调度 | “随时间改变 lr” | 在训练期间调整学习率的函数；预热 + 余弦衰减是现代默认 |
| 梯度裁剪 | “限制梯度范数” | 当梯度向量范数超过阈值时将其缩小；防止梯度爆炸更新 |

## 进一步阅读

- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014) —— 原始 Adam 论文，包含收敛分析和偏置校正推导
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017) —— 证明了在 Adam 中 L2 正则化和权重衰减不等价，并提出了 AdamW
- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017) —— 引入了 LR 范围测试和循环调度，消除了调固定学习率的需要
- Ruder, "An Overview of Gradient Descent Optimization Algorithms" (2016) —— 所有优化器变体的最佳单篇综述，附有清晰的比较和直觉