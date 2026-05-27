# 信息论

> 信息论衡量的是惊讶程度。损失函数就建立在这个基础上。

**类型：** 学习  
**语言：** Python  
**前置要求：** 阶段1，第06课（概率）  
**时间：** 约60分钟

## 学习目标

- 从零计算熵、交叉熵和 KL 散度，并解释它们之间的关系
- 推导为什么最小化交叉熵损失等价于最大化对数似然
- 计算特征与目标之间的互信息，以对特征重要性进行排序
- 将困惑度解释为语言模型从中选择的有效词汇表大小

## 问题描述

你在训练的每个分类模型中都调用 `CrossEntropyLoss()`。你在每一篇语言模型论文中都看到“困惑度”。你在 VAE、知识蒸馏和 RLHF 中读到 KL 散度。这些并不是互不相关的概念。它们是同一个想法，只是披着不同的外衣。

信息论为你提供了推理不确定性、压缩和预测的语言。克劳德·香农在 1948 年为了解决通信问题而发明了它。事实证明，训练一个神经网络也是一个通信问题：模型试图通过学习到的权重这个噪声信道来传输正确的标签。

本课程从零推导每一个公式，让你看到它们的来源以及为什么有效。

## 概念讲解

### 信息量（惊讶程度）

当不太可能发生的事情发生时，它携带的信息量更大。硬币落地是正面？不惊讶。中了彩票？非常惊讶。

一个概率为 p 的事件的信息量为：

```
I(x) = -log(p(x))
```

以 2 为底的对数给出比特（bits）。使用自然对数给出奈特（nats）。相同的概念，不同的单位。

| 事件 | 概率 | 惊讶程度（比特） |
|------|------|-----------------|
| 公平硬币正面 | 0.5 | 1.0 |
| 掷出 6 点 | 0.167 | 2.58 |
| 千分之一事件 | 0.001 | 9.97 |
| 必然事件 | 1.0 | 0.0 |

必然事件携带零信息。你已经知道它们会发生。

### 熵（平均惊讶程度）

熵是一个分布所有可能结果上的期望惊讶程度。

```
H(P) = -Σ p(x) * log(p(x))  对所有 x
```

对于二元变量，公平硬币具有最大熵：1 比特。有偏硬币（99% 正面）具有低熵：0.08 比特。你已经知道会发生什么，因此每次抛掷几乎不会告诉你有任何新信息。

```
公平硬币：   H = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0 比特
有偏硬币：   H = -(0.99 * log2(0.99) + 0.01 * log2(0.01)) = 0.08 比特
```

熵衡量一个分布中不可约的不确定性。你无法将其压缩到低于熵。

### 交叉熵（你每天都在用的损失函数）

交叉熵衡量当你使用分布 Q 来编码实际来自分布 P 的事件时，平均的惊讶程度。

```
H(P, Q) = -Σ p(x) * log(q(x))  对所有 x
```

P 是真实分布（标签）。Q 是你的模型预测。如果 Q 与 P 完全匹配，交叉熵等于熵。任何不匹配都会使它更大。

在分类问题中，P 是一个独热向量（真实类别概率为 1，其他为 0）。这将交叉熵简化为：

```
H(P, Q) = -log(q(真实类别))
```

这就是分类问题中整个交叉熵损失公式。最大化正确类别的预测概率。

### KL 散度（分布之间的距离）

KL 散度衡量使用 Q 而不是 P 所带来的额外惊讶程度。

```
D_KL(P || Q) = Σ p(x) * log(p(x) / q(x))  对所有 x
             = H(P, Q) - H(P)
```

交叉熵是熵加上 KL 散度。由于训练期间真实分布的熵是常数，最小化交叉熵等价于最小化 KL 散度。你是在推动模型的分布向真实分布靠近。

KL 散度不是对称的：D_KL(P || Q) ≠ D_KL(Q || P)。它不是真正的距离度量。

### 互信息

互信息衡量知道一个变量能告诉你关于另一个变量的多少信息。

```
I(X; Y) = H(X) - H(X|Y)
        = H(X) + H(Y) - H(X, Y)
```

如果 X 和 Y 独立，互信息为零。知道其中一个不会告诉你关于另一个的任何信息。如果它们完全相关，互信息等于任一变量的熵。

在特征选择中，特征与目标之间的高互信息意味着该特征有用。低互信息意味着它是噪声。

### 条件熵

H(Y|X) 衡量在观察到 X 之后，Y 还剩下多少不确定性。

```
H(Y|X) = H(X,Y) - H(X)
```

两个极端：
- 如果 X 完全决定了 Y，那么 H(Y|X) = 0。知道 X 消除了关于 Y 的所有不确定性。例如：X = 摄氏度温度，Y = 华氏度温度。
- 如果 X 不能告诉你关于 Y 的任何信息，那么 H(Y|X) = H(Y)。知道 X 完全没有减少你的不确定性。例如：X = 抛硬币，Y = 明天的天气。

条件熵总是非负的，且永远不会超过 H(Y)：

```
0 <= H(Y|X) <= H(Y)
```

在机器学习中，条件熵出现在决策树中。在每次分裂时，算法选择最小化 H(Y|X) 的特征——即消除关于标签 Y 最多不确定性的特征。

### 联合熵

H(X,Y) 是 X 和 Y 一起的联合分布的熵。

```
H(X,Y) = -Σ Σ p(x,y) * log(p(x,y))  对所有 x, y
```

关键性质：

```
H(X,Y) <= H(X) + H(Y)
```

当 X 和 Y 独立时取等号。如果它们共享信息，联合熵小于各自熵的和。“缺失”的熵恰好就是互信息。

```mermaid
graph TD
    subgraph "信息论维恩图"
        direction LR
        HX["H(X)"]
        HY["H(Y)"]
        MI["I(X;Y)<br/>互信息"]
        HXgY["H(X|Y)<br/>= H(X) - I(X;Y)"]
        HYgX["H(Y|X)<br/>= H(Y) - I(X;Y)"]
        HXY["H(X,Y) = H(X) + H(Y) - I(X;Y)"]
    end

    HXgY --- MI
    MI --- HYgX
    HX -.- HXgY
    HX -.- MI
    HY -.- MI
    HY -.- HYgX
    HXY -.- HXgY
    HXY -.- MI
    HXY -.- HYgX
```

关系：
- H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
- I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
- H(X,Y) = H(X) + H(Y) - I(X;Y)

### 互信息（深入）

互信息 I(X;Y) 量化了知道一个变量能减少关于另一个变量的多少不确定性。

```
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
       = Σ Σ p(x,y) * log(p(x,y) / (p(x) * p(y)))
```

性质：
- I(X;Y) >= 0 总是成立。你永远不会因为观测到某物而丢失信息。
- I(X;Y) = 0 当且仅当 X 和 Y 独立。
- I(X;Y) = I(Y;X)。它是对称的，与 KL 散度不同。
- I(X;X) = H(X)。一个变量与自身共享其所有信息。

**用于特征选择的互信息。** 在机器学习中，你希望特征能够提供关于目标的信息。互信息为你提供了一种有原则的特征排序方法：

1. 对每个特征 X_i，计算 I(X_i; Y)，其中 Y 是目标变量。
2. 按互信息分数对特征排序。
3. 保留前 k 个特征。

这适用于特征与目标之间的任何关系——线性、非线性、单调或非单调。相关性只捕捉线性关系。互信息捕捉一切。

| 方法 | 检测的关系 | 计算成本 | 处理分类数据？ |
|------|-----------|---------|---------------|
| 皮尔逊相关 | 线性关系 | O(n) | 否 |
| 斯皮尔曼相关 | 单调关系 | O(n log n) | 否 |
| 互信息 | 任意统计依赖 | O(n log n)（带分箱）| 是 |

### 标签平滑与交叉熵

标准分类使用硬目标：[0, 0, 1, 0]。真实类别概率为 1，其他为 0。标签平滑将其替换为软目标：

```
soft_target = (1 - epsilon) * hard_target + epsilon / num_classes
```

epsilon = 0.1，4 个类时：
- 硬目标：  [0, 0, 1, 0]
- 软目标：  [0.025, 0.025, 0.925, 0.025]

从信息论的角度看，标签平滑增加了目标分布的熵。硬独热目标的熵为 0 —— 没有不确定性。软目标具有正熵。

为什么这有帮助：
- 防止模型将 logits 推向极端值（需要无限大的 logits 才能在交叉熵下完美匹配独热目标）
- 起到正则化作用：模型无法 100% 确定
- 改善校准：预测概率更好地反映真实的不确定性
- 减少训练与推理行为之间的差距

带标签平滑的交叉熵损失变为：

```
L = (1 - epsilon) * CE(硬目标, 预测) + epsilon * H_均匀(预测)
```

第二项惩罚偏离均匀分布的预测——这是对置信度的直接正则化。

### 为什么交叉熵是分类问题的最优损失函数

三个视角，同一结论。

**信息论视角。** 交叉熵衡量你使用模型分布而非真实分布所浪费的比特数。最小化它使你的模型成为现实的最有效编码器。

**最大似然视角。** 对于 N 个训练样本，真实类别为 y_i：

```
似然         = ∏ q(y_i)
对数似然     = Σ log(q(y_i))
负对数似然   = -Σ log(q(y_i))
```

最后一行就是交叉熵损失。最小化交叉熵 = 最大化训练数据在你模型下的似然。

**梯度视角。** 交叉熵对 logits 的梯度就是简单的（预测值 - 真实值）。干净、稳定、计算快。这就是为什么它与 softmax 完美配对。

### 比特 vs 奈特

唯一的区别是对数的底。

```
以 2 为底的对数 -> 比特（信息论传统）
以 e 为底的对数 -> 奈特（机器学习惯例）
以 10 为底的对数 -> 哈特利（很少使用）
```

1 奈特 = 1/ln(2) 比特 = 1.4427 比特。PyTorch 和 TensorFlow 默认使用自然对数（奈特）。

### 困惑度

困惑度是交叉熵的指数。它告诉你模型在多少个等可能的选项中感到困惑的有效数量。

```
困惑度 = 2^H(P,Q)   （如果使用比特）
困惑度 = e^H(P,Q)   （如果使用奈特）
```

困惑度为 50 的语言模型，平均而言就像它必须从 50 个可能的下一个 token 中均匀随机选择一样困惑。越低越好。

GPT-2 在常见基准上达到约 30 的困惑度。现代模型在代表性良好的领域已经达到个位数。

## 动手实现

### 步骤 1：信息量与熵

```python
import math

def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float('inf') if p <= 0 else 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )

fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1/6] * 6

print(f"公平硬币熵:   {entropy(fair_coin):.4f} 比特")
print(f"有偏硬币熵:   {entropy(biased_coin):.4f} 比特")
print(f"公平骰子熵:   {entropy(fair_die):.4f} 比特")
```

### 步骤 2：交叉熵与 KL 散度

```python
def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"真实分布熵:           {entropy(true_dist):.4f} 比特")
print(f"交叉熵（好模型）:     {cross_entropy(true_dist, good_model):.4f} 比特")
print(f"交叉熵（坏模型）:     {cross_entropy(true_dist, bad_model):.4f} 比特")
print(f"KL 散度（好模型）:    {kl_divergence(true_dist, good_model):.4f} 比特")
print(f"KL 散度（坏模型）:    {kl_divergence(true_dist, bad_model):.4f} 比特")
```

### 步骤 3：交叉熵作为分类损失

```python
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:      {logits}")
print(f"Softmax:     {[f'{p:.4f}' for p in probs]}")
print(f"真实类别:    {true_class}")
print(f"损失:        {loss:.4f} 奈特")
print(f"困惑度:      {math.exp(loss):.2f}")
```

### 步骤 4：交叉熵等于负对数似然

```python
import random

random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"交叉熵损失:      {ce_loss:.6f}")
print(f"负对数似然:      {nll:.6f}")
print(f"差异:            {abs(ce_loss - nll):.2e}")
```

### 步骤 5：互信息

```python
def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi

independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"互信息（独立）: {mutual_information(independent):.4f} 比特")
print(f"互信息（依赖）: {mutual_information(dependent):.4f} 比特")
```

## 使用 NumPy

在实际使用中，你会用 NumPy 实现相同的概念：

```python
import numpy as np

def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()

def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()

def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)

true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])
print(f"熵:       {np_entropy(true):.4f} 奈特")
print(f"交叉熵:   {np_cross_entropy(true, pred):.4f} 奈特")
print(f"KL 散度:  {np_kl_divergence(true, pred):.4f} 奈特")
```

你从零构建了 `torch.nn.CrossEntropyLoss()` 内部所做的事情。现在你知道为什么训练期间损失会下降：你的模型预测分布正在向真实分布靠近，以浪费信息的奈特数来衡量。

## 练习

1. 假设英文字母均匀分布（26 个字母），计算其熵。然后使用实际的字母频率估计熵。哪个更高？为什么？

2. 一个模型对某个样本输出 logits [5.0, 2.0, 0.5]，真实类别为 1。手动计算交叉熵损失，然后用你的 `cross_entropy_loss` 函数验证。什么样的 logits 会给出零损失？

3. 证明 KL 散度不是对称的。选取两个分布 P 和 Q，计算 D_KL(P || Q) 和 D_KL(Q || P)。解释它们为什么不同。

4. 构建一个函数，计算一个 token 预测序列的困惑度。给定一个 (真实_token索引, 预测_logits) 对的列表，返回该序列的困惑度。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| 信息量 | “惊讶程度” | 编码一个事件所需的比特（或奈特）数：-log(p) |
| 熵 | “随机性” | 一个分布所有结果的平均惊讶程度。衡量不可约的不确定性。 |
| 交叉熵 | “损失函数” | 使用模型分布 Q 来编码来自真实分布 P 的事件时的平均惊讶程度。 |
| KL 散度 | “分布之间的距离” | 使用 Q 而不是 P 所浪费的额外比特数。等于交叉熵减熵。不对称。 |
| 互信息 | “X 和 Y 有多相关” | 知道 Y 后对 X 的不确定性的减少。为零意味着独立。 |
| Softmax | “将 logits 转为概率” | 指数化并归一化。将任意实值向量映射为有效的概率分布。 |
| 困惑度 | “模型有多困惑” | 交叉熵的指数。模型在每一步从中选择的有效词汇表大小。 |
| 比特 | “香农的单位” | 以 2 为底的对数衡量的信息。一比特解决一次公平抛硬币的不确定性。 |
| 奈特 | “机器学习的单位” | 以自然对数衡量的信息。PyTorch 和 TensorFlow 默认使用。 |
| 负对数似然 | “NLL 损失” | 对于独热标签，与交叉熵损失相同。最小化它等于最大化正确预测的概率。 |

## 延伸阅读

- [Shannon 1948：通信的数学理论](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) —— 原始论文，至今仍然可读
- [可视化信息论 (Chris Olah)](https://colah.github.io/posts/2015-09-Visual-Information/) —— 关于熵和 KL 散度的最佳可视化解释
- [PyTorch CrossEntropyLoss 文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) —— 框架如何实现你刚刚构建的内容