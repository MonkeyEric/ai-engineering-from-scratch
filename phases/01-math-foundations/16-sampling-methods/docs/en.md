# 采样方法

> 采样是人工智能探索可能性空间的方式。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段1，第06-07课（概率、贝叶斯定理）
**时间：** 约120分钟

## 学习目标

- 仅使用均匀随机数从头实现逆CDF、拒绝采样和重要性采样
- 为语言模型的词元生成构建温度、top-k 和 top-p（核）采样
- 解释重参数化技巧及其为何能使梯度通过 VAE 中的采样反向传播
- 运行 Metropolis-Hastings MCMC 从未归一化的目标分布中采样

## 问题描述

一个语言模型处理完你的提示后，产生了一个包含 50,000 个 logits 的向量。词汇表中的每个词元对应一个。现在它必须选出一个。怎么选？

如果它总是选概率最高的词元，那么每次回答都一模一样。确定性的、无聊的。如果它完全均匀随机地选，输出就是胡言乱语。答案存在于这两个极端之间的某处，而那个“某处”由采样控制。

采样不仅限于文本生成。强化学习通过对轨迹采样来估计策略梯度。VAE 通过对学习到的分布进行采样并反向传播通过随机性来学习潜在表示。扩散模型通过对噪声采样并迭代去噪来生成图像。蒙特卡洛方法估计没有封闭形式解的积分。MCMC 算法探索无法枚举的高维后验分布。

每一个生成式 AI 系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本节课将从均匀随机数开始，逐步构建每一种主要的采样方法，最终涵盖驱动现代大语言模型和生成式模型的技术。

## 核心概念

### 为什么采样很重要

采样在 AI 和机器学习的四个基本角色中出现：

**生成。** 语言模型、扩散模型和 GAN 都通过采样产生输出。采样算法直接控制创造性、连贯性和多样性。温度、top-k 和核采样是工程师每天都会调节的旋钮。

**训练。** 随机梯度下降对 mini-batch 采样。Dropout 采样要失活的神经元。数据增强采样随机变换。重要性采样在强化学习（PPO, TRPO）中对样本重新加权，以降低梯度方差。

**估计。** 机器学习中的许多量没有封闭形式的解。数据分布上的期望损失、基于能量的模型的分区函数、贝叶斯推断中的边缘似然。蒙特卡洛估计通过对样本取平均来近似所有这些量。

**探索。** MCMC 算法探索贝叶斯推断中的后验分布。进化策略采样参数扰动。汤普森采样在 bandit 问题中平衡探索与利用。

核心挑战：你只能直接从简单的分布（均匀、正态）中采样。对于其他所有分布，你需要一种方法，将简单分布的采样转换为目标分布的采样。

### 均匀随机采样

每个采样方法都从这里开始。一个均匀随机数生成器产生 [0, 1) 区间内的值，其中每个等长子区间具有相等的概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    对于 0 <= a <= b <= 1

性质：
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个项目的离散集中均匀采样，生成 U 并返回 floor(n * U)。要从连续区间 [a, b] 中采样，计算 a + (b - a) * U。

关键洞察：一个均匀随机数恰好包含从一个分布中产生一个样本所需的随机性。诀窍在于找到正确的变换。

### 逆 CDF 方法（逆变换采样）

累积分布函数（CDF）将值映射为概率：

```
F(x) = P(X <= x)

性质：
  F 是非递减的
  F(-inf) = 0
  F(+inf) = 1
  F 将实数轴映射到 [0, 1]
```

逆 CDF 将概率映射回值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从目标分布。

```
算法：
  1. 生成 u ~ Uniform(0, 1)
  2. 返回 F_inverse(u)

为什么有效：
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

解方程 F(x) = u 得：
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

由于 (1 - U) 与 U 同分布：
  x = -ln(u) / lambda
```

当你能够写出 F_inverse 的封闭形式时，这个方法完美工作。对于正态分布，没有封闭形式的逆 CDF，因此我们使用其他方法（Box-Muller，或数值近似）。

**离散版本：** 对于离散分布，将 CDF 构建为累积和，生成 U，然后找到累积和首次超过 U 的那个索引。这就是第 06 课中 `sample_categorical` 的工作方式。

### 拒绝采样

当你无法对 CDF 求逆，但可以计算目标 PDF（可能没有归一化）时，拒绝采样可以工作。

```
目标分布：p(x)   （可以计算，可能未归一化）
提议分布：q(x)   （可以从中采样）
边界 M：使得对所有 x 有 p(x) <= M * q(x)

算法：
  1. 从 q(x) 采样 x
  2. 采样 u ~ Uniform(0, 1)
  3. 如果 u < p(x) / (M * q(x))，则接受 x
  4. 否则，拒绝并回到步骤 1

接受率 = 1 / M
```

边界 M 越紧，接受率越高。在低维（1-3维）中，拒绝采样效果不错。在高维中，接受率呈指数下降，因为提议的大部分体积被拒绝。这就是拒绝采样的维度灾难。

**示例：从截断正态分布采样。** 使用截断范围内的均匀提议。包络 M 是该范围内正态 PDF 的最大值。

**示例：从半圆采样。** 在外接矩形内均匀提议。如果点落在半圆内则接受。这就是蒙特卡洛计算 pi 的方式：接受率等于面积比 pi/4。

### 重要性采样

有时你并不需要来自目标分布 p(x) 的样本，而是需要估计在 p(x) 下的期望，并且你拥有来自另一个分布 q(x) 的样本。

```
目标：估计 E_p[f(x)] = ∫ f(x) * p(x) dx

改写：
  E_p[f(x)] = ∫ f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

其中 w(x) = p(x) / q(x) 是重要性权重。

估计量：
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    ，其中 x_i ~ q(x)
```

这在强化学习中至关重要。在 PPO（近端策略优化）中，你在旧策略 pi_old 下收集轨迹，但想优化新策略 pi_new。重要性权重是 pi_new(a|s) / pi_old(a|s)。PPO 会对这些权重进行裁剪，以防止新策略偏离旧策略太远。

重要性采样估计量的方差取决于 q 与 p 的相似程度。如果 q 与 p 差异很大，少数样本会获得巨大的权重并主导估计。自归一化重要性采样通过除以权重之和来减少这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡洛估计

蒙特卡洛估计通过对随机样本取平均来近似积分。大数定律保证收敛性。

```
目标：估计 I = ∫ g(x) dx 积分区域 D

方法：
  1. 从 D 中均匀采样 x_1, ..., x_N
  2. I ~ (Volume(D) / N) * sum(g(x_i))

误差：O(1 / sqrt(N))   与维度无关
```

误差率与维度无关。这就是蒙特卡洛方法在高维中主导的原因，而在高维中基于网格的积分是不可能的。

**估计 pi：**

```
从 [-1, 1] x [-1, 1] 中均匀采样 (x, y)
计数落在单位圆内的个数：x^2 + y^2 <= 1
pi ~ 4 * (内部计数) / (总计数)
```

**估计期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    其中 x_i ~ p(x)

样本均值收敛到真实期望。
估计量的方差 = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings

MCMC 构建一个马尔可夫链，其平稳分布就是目标分布 p(x)。经过足够多的步数后，链中的样本（近似）来自 p(x)。

```
目标：p(x)   （已知至多一个归一化常数）
提议：q(x'|x)  （给定当前状态如何提议下一个状态）

Metropolis-Hastings 算法：
  1. 从某个 x_0 开始
  2. 对于 t = 1, 2, ..., T：
     a. 提议 x' ~ q(x'|x_t)
     b. 计算接受比率：
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. 以概率 min(1, alpha) 接受：
        - 如果 u < alpha（u ~ Uniform(0,1)）：x_{t+1} = x'
        - 否则：x_{t+1} = x_t
  3. 丢弃前 B 个样本（烧入期）
  4. 返回剩余样本
```

对于对称提议（q(x'|x) = q(x|x')），比率简化为 p(x')/p(x)。这就是原始的 Metropolis 算法。

**为什么有效。** 接受规则确保了细致平衡：位于 x 并移动到 x' 的概率等于位于 x' 并移动到 x 的概率。细致平衡蕴含 p(x) 是链的平稳分布。

**实际考虑：**
- 烧入：在链达到平衡之前丢弃早期样本
- 稀疏化：每 k 个样本保留一个，以减少自相关
- 提议尺度：太小则链移动缓慢（高接受率，慢探索）；太大则大多数提议被拒绝（低接受率，卡在原地）
- 高维中高斯提议的最优接受率约为 0.234

### Gibbs 采样

Gibbs 采样是 MCMC 对多元分布的一个特例。它不是一次更新所有维度，而是每次只从一个变量的条件分布中更新它。

```
目标：p(x_1, x_2, ..., x_d)

算法：
  对每次迭代 t：
    采样 x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    采样 x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    采样 x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样要求你能够从每个条件分布 p(x_i | x_{-i}) 中采样。这对许多模型是直接的：
- 贝叶斯网络：条件分布遵循图结构
- 高斯混合模型：条件分布是高斯分布
- Ising 模型：每个自旋的条件分布只依赖于其邻居

接受率始终为 1（每个提议都被接受），因为从精确条件分布中采样自动满足细致平衡。

**局限性。** 当变量高度相关时，Gibbs 采样混合缓慢，因为一次只更新一个变量无法在分布中做出大的对角移动。

### 温度采样（用于大语言模型）

语言模型对词汇表中的每个词元输出 logits z_1, ..., z_V。Softmax 将其转换为概率。温度在 softmax 之前对 logits 进行缩放：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0：标准 softmax（原始分布）
T -> 0： argmax（确定性，总是选最高 logit）
T -> inf：均匀（所有词元等可能）
T < 1.0：使分布变陡（更自信，更少多样性）
T > 1.0：使分布变平（更不自信，更多多样性）
```

**为什么有效。** 将 logits 除以 T < 1 会放大 logits 之间的差异。如果 z_1 = 2 且 z_2 = 1，除以 T = 0.5 得到 z_1/T = 4 和 z_2/T = 2，拉大了差距。经过 softmax 后，最高 logit 的词元获得更大的份额。

**实际使用：**
- T = 0.0：贪心解码，最适合事实性问答
- T = 0.3-0.7：稍有创造性，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，很少有用

温度不改变哪些词元是可能的。它改变分配给每个词元的概率质量。

### Top-k 采样

Top-k 采样将候选集限制为概率最高的 k 个词元，然后重新归一化并从该受限集中采样。

```
算法：
  1. 计算所有 V 个词元的 softmax 概率
  2. 按概率降序排序词元
  3. 只保留前 k 个词元
  4. 重新归一化：p_i' = p_i / sum(p_j for j in top-k)
  5. 从重新归一化的分布中采样

k = 1：  贪心解码
k = V：  无过滤（标准采样）
k = 40：典型设置，去除长尾中不太可能的词元
```

Top-k 防止模型选择那些存在于词汇分布长尾中的极不可能词元（错别字、无意义词）。问题在于：k 是固定的，与上下文无关。当模型自信时（某个词元概率为 95%），k = 40 仍然允许 39 个替代选项。当模型不确定时（概率分散在 1000 个词元上），k = 40 会切断合理的选项。

### Top-p（核）采样

Top-p 采样动态调整候选集的大小。它不是保留固定数量的词元，而是保留累积概率超过 p 的最小词元集。

```
算法：
  1. 计算所有 V 个词元的 softmax 概率
  2. 按概率降序排序词元
  3. 找到最小的 k，使得前 k 个词元的概率之和 >= p
  4. 只保留这 k 个词元
  5. 重新归一化并采样

p = 0.9： 保留覆盖 90% 概率质量的词元
p = 1.0： 无过滤
p = 0.1： 非常严格，接近贪心
```

当模型自信时，核采样保留很少的词元（可能 2-3 个）。当模型不确定时，它保留很多（可能 200 个）。这种自适应行为是核采样通常比 top-k 产生更好文本的原因。

**常见组合：**
- 温度 0.7 + top-p 0.9：良好的通用设置
- 温度 0.0（贪心）：最适合确定性任务
- 温度 1.0 + top-k 50：Fan 等人（2018）原始论文的设置

Top-k 和 top-p 可以结合使用。先应用 top-k，然后在剩余集合上应用 top-p。

### 重参数化技巧（用于 VAE）

变分自编码器通过将输入编码到潜在空间中的分布、从该分布采样、然后将样本解码回输入来学习。问题：你无法通过采样操作进行反向传播。

```
标准采样（不可微分）：
  z ~ N(mu, sigma^2)

  随机性阻碍了梯度流动。
  d/d_mu [从 N(mu, sigma^2) 中采样的样本] = ???
```

重参数化技巧将随机性与参数分离：

```
重参数化采样：
  epsilon ~ N(0, 1)          （固定的随机噪声，无参数）
  z = mu + sigma * epsilon   （参数的确定性函数）

  现在 z 是关于 mu 和 sigma 的确定性、可微函数。
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  梯度可以流过 mu 和 sigma。
```

这之所以有效，是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 同分布。关键洞察：将随机性移动到一个无参数源（epsilon），然后将样本表示为参数的可微变换。

**在 VAE 训练循环中：**
1. 编码器对每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 反向传播通过步骤 4、3、2、1（可行，因为步骤 3 是可微的）

没有重参数化技巧，VAE 无法用标准的反向传播训练。这一个洞察使得 VAE 变得实用。

### Gumbel-Softmax（可微分的分类采样）

重参数化技巧适用于连续分布（高斯）。对于离散分类分布，我们需要不同的方法。Gumbel-Softmax 提供了对分类采样的可微分近似。

**Gumbel-Max 技巧（不可微分）：**

```
要从具有对数概率 log(p_1), ..., log(p_k) 的分类分布中采样：
  1. 为每个类别采样 g_i ~ Gumbel(0, 1)
     （g = -log(-log(u))，其中 u ~ Uniform(0, 1)）
  2. 返回 argmax(log(p_i) + g_i)

这会生成精确的分类样本。
```

**Gumbel-Softmax（可微分近似）：**

```
用 soft 的 softmax 替换硬 argmax：
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau（温度）控制近似程度：
  tau -> 0：  趋近于一个 one-hot 向量（硬分类）
  tau -> inf：趋近于均匀 (1/k, 1/k, ..., 1/k)
  tau = 1.0： 软近似
```

Gumbel-Softmax 产生离散样本的连续松弛。输出是一个概率向量（软 one-hot）而不是硬 one-hot。梯度通过 softmax 流动。在训练的前向传播中，你可以使用“直通”估计器：前向传播使用硬 argmax，但反向传播使用软 Gumbel-Softmax 梯度。

**应用：**
- VAE 中的离散潜在变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 具有离散动作的强化学习

### 分层采样

标准蒙特卡洛采样可能会在样本空间中偶然留下空隙。分层采样通过将空间划分为层并从每层中采样来强制均匀覆盖。

```
标准蒙特卡洛：
  从 [0, 1] 中均匀采样 N 个点
  某些区域可能有聚类，其他区域有间隙

分层采样：
  将 [0, 1] 划分为 N 个相等的层：[0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  在每个层内均匀采样一个点
  x_i = (i + u_i) / N   其中 u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

分层采样的方差总是低于或等于标准蒙特卡洛：

```
Var(分层) <= Var(标准蒙特卡洛)

当 f(x) 平滑变化时，改进最大。
对于分段常数函数，分层采样是精确的。
```

**应用：**
- 数值积分（准蒙特卡洛）
- 训练数据划分（确保每个折中的类别平衡）
- 带分层的重要性采样（结合两种技术）
- NeRF（神经辐射场）使用沿相机射线的分层采样

### 与扩散模型的联系

扩散模型通过一个采样过程生成图像。前向过程在 T 步中向图像添加高斯噪声，直到变成纯噪声。反向过程学习去噪，逐步恢复原始图像。

```
前向过程（已知）：
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  其中 epsilon ~ N(0, I)

  T 步后：x_T ~ N(0, I)  （纯噪声）

反向过程（学习得到）：
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  其中 z ~ N(0, I)

  每个去噪步骤都是一个采样步骤。
```

与本节课方法的联系：
- 每个去噪步骤都使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声调度 {alpha_t} 控制一种形式的温度退火
- 训练使用蒙特卡洛估计来近似 ELBO（证据下界）
- 扩散模型中的祖先采样是一个马尔可夫链（每一步只依赖于当前状态）

整个图像生成过程是迭代采样：从噪声开始，在每一步，以学习到的去噪模型为条件，采样一个噪声略低的版本。

## 动手实现

### 步骤1：均匀采样和逆 CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数样本，并验证均值是 1/lambda。

### 步骤2：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从一个截断正态分布中采样。通过直方图验证形状。

### 步骤3：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀提议估计正态分布下的 E[X^2]。与已知答案（mu^2 + sigma^2）比较。

### 步骤4：蒙特卡洛估计 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤5：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从一个双峰分布（两个高斯的混合）中采样。可视化链的轨迹。

### 步骤6：Gibbs 采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤7：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组词元 logits 的输出分布。

### 步骤8：Top-k 和 top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤9：重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度如何通过重参数化样本流动，但不通过直接采样流动。

### 步骤10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出趋近于 one-hot 向量。

包含所有可视化的完整实现在 `code/sampling.py` 中。

## 使用它

使用 NumPy 和 SciPy 的生产版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"指数均值: {exponential_samples.mean():.4f} (期望 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"在 1.96 处的 CDF: {normal.cdf(1.96):.4f}")
print(f"0.975 处的逆 CDF: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"采样的词元索引: {token}")
```

对于大规模 MCMC，使用专用库：
- PyMC：带 NUTS（自适应 HMC）的全功能贝叶斯建模
- emcee：集成 MCMC 采样器
- NumPyro/JAX：GPU 加速的 MCMC

你从头构建了这些。现在你知道库调用在做什么了。

## 练习

1. 为柯西分布实现逆 CDF 采样。其 CDF 为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，并绘制直方图与真实 PDF 对比。注意重尾（远离中心的极端值）。

2. 使用拒绝采样，以 Uniform(0, 1) 为提议，从 Beta(2, 5) 分布中生成样本。绘制接受的样本与真实 Beta PDF 的对比图。理论接受率是多少？

3. 使用蒙特卡洛方法，以 1,000、10,000 和 100,000 个样本估计 sin(x) 从 0 到 pi 的积分。比较每个样本量下的误差。验证误差按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings 从一个二维分布 p(x, y) ∝ exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。绘制样本和链的轨迹。尝试不同的提议标准差。

5. 构建一个完整的文本生成演示：给定一个包含 10 个词的词汇表及其 logits，使用 (a) 贪心、(b) 温度=0.7、(c) top-k=3、(d) top-p=0.9 生成 20 个词元的序列。比较 5 次运行的输出多样性。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 采样 | "抽取随机值" | 根据概率分布生成值。所有生成式 AI 背后的机制 |
| 均匀分布 | "所有等可能" | [a, b] 中的每个值具有相等的概率密度 1/(b-a)。所有采样方法的起点 |
| 逆 CDF | "概率变换" | F_inverse(U) 将均匀样本转换为任何已知 CDF 的分布中的样本。精确且高效 |
| 拒绝采样 | "提议并接受/拒绝" | 从简单的提议分布生成，以目标/提议比例的概率接受。精确但浪费样本 |
| 重要性采样 | "重新加权样本" | 使用来自 q(x) 的样本估计在 p(x) 下的期望，通过用 p(x)/q(x) 对每个样本加权。强化学习中 PPO 的核心 |
| 蒙特卡洛 | "平均随机样本" | 将积分近似为样本平均。误差 O(1/sqrt(N))，与维度无关 |
| MCMC | "收敛的随机游走" | 构建一个马尔可夫链，其平稳分布为目标分布。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | "上坡接受，有时下坡也接受" | 提议移动，基于密度比接受。细致平衡确保收敛到目标分布 |
| Gibbs 采样 | "一次一个变量" | 每次从一个变量的条件分布中更新它，保持其他变量固定。接受率 100% |
| 温度 | "置信度旋钮" | 在 softmax 之前将 logits 除以 T。T<1 使分布变陡（更自信），T>1 使分布变平（更多样） |
| Top-k 采样 | "保留最好的 k 个" | 将除概率最高的 k 个词元外的所有词元概率置零，重新归一化，采样。固定候选集大小 |
| 核采样 (top-p) | "保留那些可能的" | 保留累积概率超过 p 的最小词元集。自适应候选集大小 |
| 重参数化技巧 | "将随机性移到外部" | 将 z 写成 mu + sigma * epsilon，其中 epsilon ~ N(0,1)。使采样可微分。VAE 训练的关键 |
| Gumbel-Softmax | "软分类采样" | 使用 Gumbel 噪声 + 带温度的 softmax 对分类采样进行可微分近似 |
| 分层采样 | "强制覆盖" | 将样本空间划分为层，从每层中采样。方差总是低于朴素蒙特卡洛 |
| 烧入期 | "预热期" | 在链达到其平稳分布之前丢弃的初始 MCMC 样本 |
| 细致平衡 | "可逆性条件" | p(x) * T(x->y) = p(y) * T(y->x)。p 是马尔可夫链平稳分布的充分条件 |
| 扩散采样 | "迭代去噪" | 从噪声开始，应用学习到的去噪步骤生成数据。每一步都是一个条件采样操作 |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - 原始 Gumbel-Softmax 论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 核（top-p）采样论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入重参数化技巧的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 将采样与图像生成联系起来