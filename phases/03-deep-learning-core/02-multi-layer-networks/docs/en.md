# 多层网络与前向传播

> 一个神经元画一条直线。堆叠它们，你就能画出任何形状。

**类型：** 构建  
**语言：** Python  
**前置条件：** 第一阶段（数学基础），第 03.01 课（感知机）  
**时间：** 约 90 分钟

## 学习目标

- 使用 Layer 和 Network 类从零构建一个多层网络，完成完整的前向传播
- 追踪网络中每一层的矩阵维度，识别形状不匹配的问题
- 解释堆叠非线性激活函数如何使网络能够学习弯曲的决策边界
- 使用 2‑2‑1 架构和手工调整的 Sigmoid 权重求解 XOR 问题

## 问题

单个神经元是一条直线绘制器。仅此而已。一条笔直的线穿过你的数据。人工智能中每一个真正的问题——图像识别、语言理解、下围棋——都需要曲线。将神经元堆叠成层，就是你获得曲线的方式。

1969 年，Minsky 和 Papert 证明了这个局限是致命的：单层网络无法学习 XOR。不是“学得吃力”——而是数学上不可能。XOR 真值表将 [0,1] 和 [1,0] 放在一侧，[0,0] 和 [1,1] 放在另一侧。没有一条直线能分开它们。

这导致神经网络研究资金中断了十多年。事后看来，解决办法显而易见：不要再使用单层，而是将神经元堆叠成层。让第一层将输入空间雕刻成新的特征，让第二层将这些特征组合成任何单条直线都无法做出的决策。

这个堆叠就是多层网络。它是今天生产中每一个深度学习模型的基础。前向传播——数据从输入层通过隐藏层流向输出层——是在其他任何东西能工作之前，你需要构建的第一件事。

## 概念

### 层：输入层、隐藏层、输出层

一个多层网络有三种类型的层：

**输入层**——实际上并不是真正的层。它保存你的原始数据。两个特征意味着两个输入节点。这里不进行任何计算。

**隐藏层**——真正完成工作的地方。每个神经元接收前一层的每一个输出，应用权重和偏置，然后将结果通过激活函数。“隐藏”是因为你在训练数据中永远无法直接看到这些值。

**输出层**——最终答案。对于二分类，一个带 Sigmoid 的神经元。对于多分类，每个类别一个神经元。

```mermaid
graph LR
    subgraph Input["输入层"]
        x1["x1"]
        x2["x2"]
    end
    subgraph Hidden["隐藏层（3 个神经元）"]
        h1["h1"]
        h2["h2"]
        h3["h3"]
    end
    subgraph Output["输出层"]
        y["y"]
    end
    x1 --> h1
    x1 --> h2
    x1 --> h3
    x2 --> h1
    x2 --> h2
    x2 --> h3
    h1 --> y
    h2 --> y
    h3 --> y
```

这是一个 2‑3‑1 网络。两个输入，三个隐藏神经元，一个输出。每一条连接都带有一个权重。每一个神经元（除了输入层）都带有一个偏置。

每一层都产生一个称为隐藏状态的数值向量。对于文本，隐藏状态会增加维度——将一个词编码为 768 个数字以捕捉语义含义。对于图像，它们会降低维度——将数百万像素压缩成一个可管理的表示。隐藏状态正是学习所在之处。

### 神经元和激活函数

每个神经元做三件事：

1. 将每个输入乘以其对应的权重
2. 将所有乘积相加并加上偏置
3. 将总和通过激活函数

目前，激活函数是 Sigmoid：

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Sigmoid 将任何数压入 (0, 1) 范围内。大的正输入趋向于 1。大的负输入趋向于 0。零映射到 0.5。这条平滑曲线使学习成为可能——与感知机的硬阶跃不同，Sigmoid 在每个地方都有梯度。

### 前向传播：数据如何流动

前向传播将输入数据逐层推过网络，直到到达输出。在前向传播过程中不发生学习。它纯粹是计算：乘、加、激活、重复。

```mermaid
graph TD
    X["输入: [x1, x2]"] --> WH["乘以权重矩阵 W1 (2×3)"]
    WH --> BH["加上偏置向量 b1 (3,)"]
    BH --> AH["对每个元素应用 sigmoid"]
    AH --> H["隐藏输出: [h1, h2, h3]"]
    H --> WO["乘以权重矩阵 W2 (3×1)"]
    WO --> BO["加上偏置向量 b2 (1,)"]
    BO --> AO["应用 sigmoid"]
    AO --> Y["输出: y"]
```

在每一层，三个操作依次发生：

```
z = W * input + b       （线性变换）
a = sigmoid(z)          （激活）
```

一层的输出成为下一层的输入。这就是整个前向传播。

### 矩阵维度

追踪维度是深度学习中最重要的调试技能。下面是 2‑3‑1 网络：

| 步骤 | 操作 | 维度 | 结果形状 |
|------|------|------|----------|
| 输入 | x | -- | (2,) |
| 隐藏层线性 | W1 * x + b1 | W1: (3, 2), b1: (3,) | (3,) |
| 隐藏层激活 | sigmoid(z1) | -- | (3,) |
| 输出层线性 | W2 * h + b2 | W2: (1, 3), b2: (1,) | (1,) |
| 输出层激活 | sigmoid(z2) | -- | (1,) |

规则：第 k 层的权重矩阵 W 的形状为（第 k 层神经元数，第 k‑1 层神经元数）。行匹配当前层，列匹配前一层。如果形状对不上，那就是有 bug。

### 通用近似定理

1989 年，George Cybenko 证明了一个了不起的结果：一个带单隐藏层且有足够神经元的神经网络，可以以任意期望的精度逼近任意连续函数。

这并不意味着单隐藏层总是最好的。它意味着这种架构在理论上是可行的。在实践中，更深的网络（更多层，每层更少的神经元）用远少于浅而宽网络的总参数就能学会相同的函数。这就是深度学习有效的原因。

直觉是：隐藏层中的每个神经元学习一个“凸起”或特征。在正确位置放置足够多的凸起，就能近似任意平滑曲线。神经元越多，凸起越多，近似效果越好。

```mermaid
graph LR
    subgraph FewNeurons["4 个隐藏神经元"]
        A["粗糙近似"]
    end
    subgraph MoreNeurons["16 个隐藏神经元"]
        B["较接近的近似"]
    end
    subgraph ManyNeurons["64 个隐藏神经元"]
        C["近乎完美的拟合"]
    end
    FewNeurons --> MoreNeurons --> ManyNeurons
```

### 可组合性

神经网络是可组合的。你可以堆叠它们、串联它们、并行运行它们。Whisper 模型使用编码器网络处理音频，并使用单独的解码器网络生成文本。现代 LLM 是仅解码器。BERT 是仅编码器。T5 是编码器‑解码器。架构的选择定义了模型能做什么。

## 动手实现

纯 Python。不使用 NumPy。每个矩阵运算都从零编写。

### 第 1 步：Sigmoid 激活函数

```python
import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
```

将输入限制在 [-500, 500] 之间可以防止溢出。`math.exp(500)` 很大但有限。`math.exp(1000)` 是无穷大。

### 第 2 步：Layer 类

深度学习中最重要的运算就是矩阵乘法。每一层、每一个注意力头、每一次前向传播——归根结底都是矩阵乘法。线性层接受一个输入向量，将其乘以一个权重矩阵，再加上一个偏置向量：y = Wx + b。这个单一的方程占据了神经网络中 90% 的计算量。

一个层包含一个权重矩阵和一个偏置向量。它的 forward 方法接受一个输入向量并返回激活后的输出。

```python
class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output
```

权重矩阵的形状为 (n_neurons, n_inputs)。每一行是一个神经元在所有输入上的权重。forward 方法遍历神经元，计算加权和加偏置，应用 sigmoid，并收集结果。

### 第 3 步：Network 类

网络是一个层的列表。前向传播将它们串联起来：第 k 层的输出馈入第 k+1 层。

```python
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current
```

这就是整个前向传播。四行逻辑。数据进入，流经每一层，从另一端出来。

### 第 4 步：使用手工调整权重的 XOR

在第 01 课中，我们通过组合 OR、NAND 和 AND 感知机解决了 XOR。现在用我们的 Layer 和 Network 类做同样的事。2‑2‑1 架构：两个输入，两个隐藏神经元，一个输出。

```python
hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[[20.0, 20.0], [-20.0, -20.0]],
    biases=[-10.0, 30.0],
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[[20.0, 20.0]],
    biases=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (四舍五入: {predicted}, 期望: {expected})")
```

大的权重（20, -20）使 sigmoid 表现得像阶跃函数。第一个隐藏神经元近似 OR。第二个近似 NAND。输出神经元将它们组合成 AND，也就是 XOR。

### 第 5 步：圆分类

一个更难的问题：将二维点分类为在圆心为原点、半径为 0.5 的圆内还是圆外。这需要弯曲的决策边界——单个感知机无法做到。

```python
import random
import math

random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1),
])
```

使用随机权重，网络分类效果不会好。但前向传播仍然能运行。这就是关键——前向传播只是计算。学习正确的权重是反向传播，将在第 03 课中介绍。

```python
correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

print(f"随机权重的准确率: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")
```

随机权重给出的准确率很低——通常比猜多数类还差。在训练之后（第 03 课），这个具有 8 个隐藏神经元的相同架构将画出一个弯曲的边界，将圆内和圆外分开。

## 如何使用（PyTorch）

PyTorch 用四行代码完成以上所有工作：

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
output = model(x)
print(output)
```

`nn.Linear(2, 8)` 就是你的 Layer 类：形状为 (8, 2) 的权重矩阵，形状为 (8,) 的偏置向量。`nn.Sigmoid()` 就是你的按元素应用的 sigmoid 函数。`nn.Sequential` 就是你的 Network 类：按顺序串联层。

区别在于速度和规模。PyTorch 在 GPU 上运行，可以处理数百万样本的批次，并自动计算反向传播的梯度。但前向传播的逻辑与你刚才从零构建的完全相同。

## 交付内容

本课程产出一个用于设计网络架构的可复用提示：

- `outputs/prompt-network-architect.md`

当你需要为特定问题决定使用多少层、每层多少个神经元以及哪些激活函数时，可以使用它。

## 练习

1. 构建一个 2‑4‑2‑1 网络（两个隐藏层），并在 XOR 数据上使用随机权重运行前向传播。打印中间隐藏层的输出，观察表示在各层之间如何变换。

2. 将圆分类器中的隐藏层大小从 8 改为 2，再改为 32。每次使用随机权重运行前向传播。隐藏神经元的数量会改变输出的范围或分布吗？为什么？

3. 在 Network 类上实现一个 `count_parameters` 方法，返回可训练的权重和偏置的总数。在 784‑256‑128‑10 网络（经典的 MNIST 架构）上测试它。它有多少个参数？

4. 为一个 3‑4‑4‑2 网络构建前向传播。输入 RGB 颜色值（归一化到 0‑1），观察两个输出。这是一个用于二分类的简单颜色分类器架构。

5. 用“泄漏阶跃”函数替换 sigmoid：如果 z < 0 则返回 0.01 * z，否则返回 1.0。使用第 4 步中相同的手工调整权重在 XOR 上运行前向传播。它仍然有效吗？为什么平滑的 sigmoid 比硬截断更受欢迎？

## 关键术语

| 术语 | 人们常说的话 | 实际含义 |
|------|----------------|----------------------|
| 前向传播 | “运行模型” | 将输入推过每一层——乘以权重、加偏置、激活——产生输出 |
| 隐藏层 | “中间部分” | 输入层和输出层之间的任何层，其值在数据中无法直接观察到 |
| 多层网络 | “深度神经网络” | 按顺序堆叠的神经元层，每一层的输出作为下一层的输入 |
| 激活函数 | “非线性” | 在线性变换之后应用的函数，为决策边界引入曲线 |
| Sigmoid | “S 形曲线” | sigma(z) = 1/(1+e^(-z))，将任意实数压入 (0,1)，处处平滑可微 |
| 权重矩阵 | “参数” | 形状为（当前层神经元数，前一层神经元数）的矩阵 W，包含可学习的连接强度 |
| 偏置向量 | “偏移量” | 在矩阵乘法后加入的向量，使神经元即使在所有输入为零时也能激活 |
| 通用近似 | “神经网络能学会任何东西” | 具有足够神经元的单隐藏层可以近似任意连续函数——但“足够”可能意味着数十亿 |
| 线性变换 | “矩阵乘法步骤” | z = W * x + b，激活之前的计算，将输入映射到一个新空间 |
| 决策边界 | “分类器切换的地方” | 输入空间中网络输出跨越分类阈值的那一个曲面 |

## 进一步阅读

- Michael Nielsen, "Neural Networks and Deep Learning", 第 1‑2 章 (http://neuralnetworksanddeeplearning.com/) —— 关于前向传播和网络结构的最清晰免费解释，附带交互式可视化
- Cybenko, "Approximation by Superpositions of a Sigmoidal Function" (1989) —— 原始通用近似定理论文，出奇地易读
- 3Blue1Brown, "But what is a neural network?" (https://www.youtube.com/watch?v=aircAruvnKk) —— 20 分钟的层、权重和前向传播可视化讲解，帮助建立正确的思维模型
- Goodfellow, Bengio, Courville, "Deep Learning", 第 6 章 (https://www.deeplearningbook.org/) —— 多层网络的标准参考书，免费在线阅读