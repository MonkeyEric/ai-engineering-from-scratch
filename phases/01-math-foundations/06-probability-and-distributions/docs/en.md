# 概率与分布

> 概率是 AI 用来表达不确定性的语言。

**类型：** 学习  
**语言：** Python  
**前置要求：** 阶段1，第01-04课  
**时间：** 约75分钟

## 学习目标

- 从零实现伯努利分布、分类分布、泊松分布、均匀分布和正态分布的 PMF 和 PDF
- 计算期望值和方差，并使用中心极限定理解释为什么高斯分布无处不在
- 使用数值稳定技巧（减去最大 logit）构建 softmax 和 log-softmax 函数
- 从 logits 计算交叉熵损失，并将其与负对数似然联系起来

## 问题描述

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型从 50,000 个候选中选择下一个词。一个扩散模型通过对学习到的分布进行采样来生成图像。这些都是概率的实际应用。

模型做出的每一个预测都是一个概率分布。每一个损失函数都衡量预测分布与真实分布之间的差距。每一个训练步骤都调整参数，使一个分布看起来更像另一个分布。没有概率，你就无法阅读任何一篇机器学习论文，无法调试任何模型，也无法理解为什么你的训练损失是 NaN。

## 概念讲解

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的一个子集。概率将事件映射到 0 到 1 之间的数字。

```
抛硬币：
  S = {H, T}
  P(H) = 0.5， P(T) = 0.5

掷单个骰子：
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三个公理定义了整个概率论：
1. 对任何事件 A，有 P(A) ≥ 0
2. P(S) = 1（总有什么事会发生）
3. 当 A 和 B 不能同时发生时，P(A 或 B) = P(A) + P(B)

其他所有内容（贝叶斯定理、期望、分布）都源自这三条规则。

### 条件概率与独立性

P(A|B) 是在 B 已经发生的前提下 A 发生的概率。

```
P(A|B) = P(A and B) / P(B)

示例：一副扑克牌
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

当知道一个事件对另一个事件不提供任何信息时，这两个事件是独立的：

```
独立：   P(A|B) = P(A)
等价于： P(A and B) = P(A) * P(B)
```

抛硬币是独立的。无放回抽牌则不是。

### 概率质量函数 vs 概率密度函数

离散随机变量具有概率质量函数（PMF）。每个结果都有一个可以直接读出的特定概率。

```
PMF: P(X = k)

公平骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量具有概率密度函数（PDF）。单个点处的密度不是概率。概率来自对密度在一个区间上的积分。

```
PDF: f(x)

P(a <= X <= b) = 从 a 到 b 积分 f(x) dx

f(x) 可以大于 1（密度，不是概率）
从 -∞ 到 +∞ 积分 f(x) dx = 1
```

这个区别在机器学习中很重要。分类输出是 PMF（离散选择）。VAE 的潜在空间使用 PDF（连续）。

### 常见分布

**伯努利分布：** 一次试验，两种结果。用于二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p， 方差 = p(1-p)
```

**分类分布：** 一次试验，k 种结果。用于多分类（softmax 输出）。

```
P(X = i) = p_i， 其中 sum p_i = 1
示例：P(猫) = 0.7， P(狗) = 0.2， P(鸟) = 0.1
```

**均匀分布：** 所有结果等可能。用于随机初始化。

```
离散：P(X = k) = 1/n， k ∈ {1, ..., n}
连续：f(x) = 1/(b-a)， x ∈ [a, b]
```

**正态分布（高斯分布）：** 钟形曲线。由均值 μ 和方差 σ² 参数化。

```
f(x) = (1 / sqrt(2*π*σ²)) * exp(-(x - μ)² / (2*σ²))

标准正态：μ = 0, σ = 1
  68% 的数据落在 1 个标准差内
  95% 落在 2 个标准差内
  99.7% 落在 3 个标准差内
```

**泊松分布：** 固定区间内稀有事件的计数。用于事件率建模。

```
P(X = k) = (λ^k * e^(-λ)) / k!
均值 = λ， 方差 = λ
```

### 期望值与方差

期望值是结果的加权平均。

```
离散：   E[X] = Σ x_i * P(X = x_i)
连续：   E[X] = ∫ x * f(x) dx
```

方差衡量围绕均值的散布程度。

```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
标准差 = sqrt(Var(X))
```

在机器学习中，期望值以损失函数的形式出现（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度的高方差意味着训练不稳定。

### 联合分布与边缘分布

联合分布 P(X, Y) 描述两个随机变量在一起的情况。

联合 PMF 示例（X = 天气，Y = 雨伞）：

| | Y=0（无雨伞） | Y=1（有雨伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨天） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布通过对另一个变量求和得到：

```
P(X = x) = Σ 对所有 y 求和 P(X = x, Y = y)
```

上表中的行总和与列总和就是边缘概率。

### 为什么正态分布无处不在

中心极限定理：许多独立随机变量的和（或平均值）收敛于正态分布，无论原始分布是什么。

```
掷 1 个骰子：均匀分布（平坦）
2 个骰子的平均值：三角形（中间高）
30 个骰子的平均值：近乎完美的钟形曲线

这对任何起始分布都成立。
```

这就是为什么：
- 测量误差近似正态（许多小的独立来源）
- 神经网络中的权重初始化使用正态分布
- SGD 中的梯度噪声近似正态（许多样本梯度的和）
- 在给定均值和方差的情况下，正态分布是最大熵分布

### 对数概率

原始概率会导致数值问题。将许多小概率相乘会很快下溢为零。

```
P(句子) = P(词1) * P(词2) * ... * P(词n)
       = 0.01 * 0.003 * 0.02 * ...
       -> 0.0（约 30 项后就下溢）
```

对数概率解决了这个问题。乘法变成加法。

```
log P(句子) = log P(词1) + log P(词2) + ... + log P(词n)
           = -4.6 + -5.8 + -3.9 + ...
           -> 有限数（不会下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总是 ≤ 0（因为 0 < P ≤ 1）
- 越负表示可能性越小
- 交叉熵损失就是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 将它们转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / Σ exp(z_j) 对所有 j

性质：
  - 所有输出都在 (0, 1) 之间
  - 所有输出之和为 1
  - 保持输入的相对顺序
  - exp() 会放大 logits 之间的差异
```

Softmax 技巧：在取指数之前减去最大 logit，以防止溢出。

```
z = [100, 101, 102]
exp(102) 会溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1 （安全）

结果相同，没有溢出。
```

Log-softmax 结合了 softmax 和对数，以获得数值稳定性。PyTorch 内部使用它来实现交叉熵损失。

### 采样

采样是指从分布中随机抽取值。在机器学习中：
- Dropout 随机采样哪些神经元被置零
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要一些技术，如逆变换采样、拒绝采样或重参数化技巧（用于 VAE）。

## 动手实现

### 步骤 1：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 步骤 2：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 步骤 3：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"骰子: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 步骤 4：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 步骤 5：Softmax 和对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 步骤 6：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步骤 7：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整的实现以及所有可视化代码在 `code/probability.py` 中。

## 使用 NumPy / SciPy

使用 NumPy 和 SciPy，上述所有操作都可以用一行代码完成：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你从零实现了这些。现在你知道库函数在做什么了。

## 练习

1. 实现指数分布的逆变换采样。通过采样 10,000 个值并将直方图与真实 PDF 比较来验证。

2. 构建两个有偏骰子的联合分布表。计算边缘分布，并检查骰子是否独立。

3. 对于一个 5 分类分类器，当正确类别是索引 3 时，输出的 logits 为 `[2.0, 0.5, -1.0, 3.0, 0.1]`，计算其交叉熵损失。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，接受一个对数概率列表，返回最可能的序列、总对数概率以及等效的原始概率。用一个 50 个词的句子进行测试，每个词的概率为 0.01。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| 样本空间 | “所有可能性” | 一个实验所有可能结果的集合 S |
| PMF | “概率函数” | 给出每个离散结果的确切概率的函数，所有概率之和为 1 |
| PDF | “概率曲线” | 连续变量的密度函数。在一个区间上积分得到概率 |
| 条件概率 | “给定某条件下的概率” | P(A\|B) = P(A and B) / P(B)。贝叶斯思维和贝叶斯定理的基础 |
| 独立性 | “它们不互相影响” | P(A and B) = P(A) * P(B)。知道一个事件对另一个事件不提供信息 |
| 期望值 | “平均值” | 所有结果的概率加权和。损失函数就是一个期望值 |
| 方差 | “散布程度” | 与均值的平方偏差的期望。高方差意味着噪声大、估计不稳定 |
| 正态分布 | “钟形曲线” | f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))。由于中心极限定理无处不在 |
| 中心极限定理 | “平均值变成正态分布” | 许多独立样本的均值收敛于正态分布，无论原始分布是什么 |
| 联合分布 | “两个变量一起” | P(X, Y) 描述 X 和 Y 所有结果组合的概率 |
| 边缘分布 | “对另一个变量求和” | P(X) = Σ_y P(X, Y)。从联合分布中恢复一个变量的分布 |
| 对数概率 | “概率的对数” | log P(x)。将乘积变为和，防止长序列中的数值下溢 |
| Softmax | “将分数变成概率” | softmax(z_i) = exp(z_i) / Σ exp(z_j)。将实数 logits 映射为有效的概率分布 |
| 交叉熵 | “损失函数” | -Σ p_true * log(p_predicted)。衡量两个分布之间的差异。越低越好 |
| Logits | “模型原始输出” | softmax 之前的未归一化分数。以 logistic 函数命名 |
| 采样 | “抽取随机值” | 根据概率分布生成值。模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown：但什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) —— 平均值为何变成正态分布的直观证明
- [斯坦福 CS229 概率复习讲义](https://cs229.stanford.edu/section/cs229-prob.pdf) —— 涵盖这里及更多内容的简明参考
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) —— 为什么数值稳定性很重要以及如何实现