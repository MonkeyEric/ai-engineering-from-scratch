# 特征选择

> 特征越多并不越好。**正确的特征**才是更好的。

**类型：** 构建  
**语言：** Python  
**前置条件：** 第二阶段，第 01–09 课，第 08 课（特征工程）  
**时间：** 约 75 分钟

## 学习目标

- 从零实现过滤式方法（方差阈值、互信息、卡方检验）和封装式方法（RFE、前向选择）
- 解释为什么互信息能够捕捉相关分析所忽略的非线性特征-目标关系
- 比较 L1 正则化（嵌入式选择）与 RFE（封装式选择），并评估它们的计算权衡
- 构建一个结合多种方法的特征选择流水线，并展示在留出数据上泛化能力的提升

## 问题

你有 500 个特征。模型训练缓慢，持续过拟合，没有人能解释它学到了什么。你添加更多特征希望提升性能，结果却更糟。

这就是“维度灾难”的实际表现。随着特征数量增加，特征空间的体积呈爆炸式增长。数据点变得稀疏，点之间的距离趋于收敛。模型需要指数级更多的数据才能找到真正的模式。噪声特征淹没信号特征，过拟合成为默认状态。

特征选择是解药。剥离噪声，去除冗余，保留那些携带目标真实信息的特征。结果是：更快的训练、更好的泛化能力，以及你能真正解释的模型。

目标不是使用所有可用信息，而是使用**正确的信息**。

## 概念

### 特征选择的三大类别

每种特征选择方法都属于以下三类之一：

```mermaid
flowchart TD
    A[特征选择方法] --> B[过滤式方法]
    A --> C[封装式方法]
    A --> D[嵌入式方法]

    B --> B1["方差阈值"]
    B --> B2["互信息"]
    B --> B3["卡方检验"]
    B --> B4["相关性过滤"]

    C --> C1["递归特征消除"]
    C --> C2["前向选择"]
    C --> C3["后向消除"]

    D --> D1["L1 / Lasso 正则化"]
    D --> D2["基于树的重要性"]
    D --> D3["弹性网络"]
```

**过滤式方法** 使用统计度量独立地对每个特征进行评分。它们不使用模型。速度快，但会遗漏特征之间的交互。

**封装式方法** 通过训练模型来评估特征子集。它们将模型性能作为评分依据。效果更好，但代价昂贵，因为需要多次重新训练模型。

**嵌入式方法** 在模型训练过程中选择特征。L1 正则化将权重推向零；决策树在最有用特征上分裂。选择发生在拟合过程中，而不是单独的一步。

### 方差阈值

最简单的过滤方法。如果某个特征在各样本间几乎不变化，那么它几乎不携带任何信息。

考虑一个特征，在 1000 个样本中有 999 个为 0.0。它的方差接近于零。没有任何模型能利用它区分类别。删除它。

```
variance(x) = mean((x - mean(x))^2)
```

设定一个阈值（例如 0.01），丢弃所有方差低于该阈值的特征。这会在完全不查看目标变量的情况下，移除常量或近似常量特征。

使用时机：作为其他方法之前的预处理步骤。以近乎零的成本捕获明显无用的特征。

局限性：特征可能有高方差却仍然是纯噪声。方差阈值是必要条件，但不是充分条件。

### 互信息

互信息衡量的是，知道特征 X 的值能在多大程度上减少目标 Y 的不确定性。

```
I(X; Y) = sum_x sum_y p(x, y) * log(p(x, y) / (p(x) * p(y)))
```

如果 X 和 Y 独立，则 p(x, y) = p(x) * p(y)，因此对数项为零，I(X; Y) = 0。X 告诉你关于 Y 的信息越多，互信息就越高。

相对于相关性的主要优势：互信息能够捕捉非线性关系。一个特征可能与目标的相关性为零，但互信息很高，因为关系是二次的或周期性的。

对于连续特征，先分箱（基于直方图估计）。分箱数量会影响估计结果——箱太少会丢失信息，箱太多会引入噪声。常见选择：sqrt(n) 个箱，或 Sturges 规则（1 + log2(n)）。

```mermaid
flowchart LR
    A[特征 X] --> B[离散化为分箱]
    B --> C["计算联合分布 p(x,y)"]
    C --> D["计算 MI = sum p(x,y) * log(p(x,y) / p(x)p(y))"]
    D --> E[按 MI 评分对特征排序]
    E --> F[选择前 K 个]
```

### 递归特征消除（RFE）

RFE 是一种封装式方法。它利用模型自身的特征重要性进行迭代剪枝：

1. 使用所有特征训练模型
2. 按重要性对特征排序（线性模型使用系数，树模型使用不纯度减少量）
3. 移除最不重要的特征
4. 重复直到达到所需特征数量

```mermaid
flowchart TD
    A["开始：全部 N 个特征"] --> B["训练模型"]
    B --> C["对特征重要性排序"]
    C --> D["移除最不重要的特征"]
    D --> E{"特征数 == 目标数量?"}
    E -->|否| B
    E -->|是| F["返回所选特征"]
```

RFE 会考虑特征交互，因为模型会同时看到所有剩余特征。移除一个特征会改变其他特征的重要性。这使得它比过滤式方法更彻底。

代价：需要训练模型 N - target 次。对于 500 个特征、目标为 10 的情况，就是 490 次训练运行。对于昂贵的模型，这很慢。可以通过每轮移除多个特征来加速（例如，每轮移除最不重要的 10%）。

### L1（Lasso）正则化

L1 正则化将权重的绝对值添加到损失函数中：

```
loss = prediction_error + alpha * sum(|w_i|)
```

参数 alpha 控制特征剪枝的激进程度。alpha 越大，越多的权重会被精确地置为零。

为什么能精确为零？L1 惩罚在权重空间中产生了一个菱形约束区域。最优解往往落在该菱形的一个角上，此时一个或多个权重为零。L2 正则化（岭回归）产生圆形约束，权重会收缩但很少恰好为零。

这是嵌入式特征选择：模型在训练过程中学习忽略哪些特征。权重为零的特征实际上被移除了。

优点：单次训练运行，能处理相关特征（从中选一个，其余置零），在大多数线性模型实现中内置。

局限性：仅适用于线性模型，无法捕捉非线性特征重要性。

### 基于树的重要性

决策树及其集成方法（随机森林、梯度提升）天然地对特征进行排序。每次分裂都会减少不纯度（分类用基尼或熵，回归用方差）。能带来更大不纯度减少的特征更重要。

对于包含 T 棵树的随机森林：

```
importance(feature_j) = (1/T) * sum over all trees of
    sum over all nodes splitting on feature_j of
        (n_samples * impurity_decrease)
```

这会为每个特征给出归一化的重要性分数。它能够自动处理非线性关系和特征交互。

注意：基于树的重要性偏向于具有较多唯一值（高基数）的特征。一个随机的 ID 列会因为能完美分割每个样本而显得很重要。可以使用排列重要性作为合理性检查。

### 排列重要性

一种模型无关的方法：

1. 训练模型，在验证数据上记录基线性能
2. 对每个特征：随机打乱其取值，测量性能下降程度
3. 下降越大，特征越重要

如果打乱某个特征不损害性能，说明模型不依赖它；如果性能崩溃，则该特征至关重要。

排列重要性避免了基于树的重要性的基数偏差。但速度较慢：每个特征需要一次完整评估，且为稳定性需重复多次。

### 对比表

| 方法               | 类型       | 速度   | 非线性 | 特征交互 |
|--------------------|------------|--------|--------|----------|
| 方差阈值           | 过滤式     | 极快   | 否     | 否       |
| 互信息             | 过滤式     | 快     | 是     | 否       |
| 相关性过滤         | 过滤式     | 快     | 否     | 否       |
| RFE                | 封装式     | 慢     | 取决于模型 | 是    |
| L1 / Lasso         | 嵌入式     | 快     | 否（线性）| 否      |
| 树重要性           | 嵌入式     | 中等   | 是     | 是       |
| 排列重要性         | 模型无关   | 慢     | 是     | 是       |

### 决策流程图

```mermaid
flowchart TD
    A[开始：特征选择] --> B{特征数量？}
    B -->|"< 50"| C["从方差阈值 + 互信息开始"]
    B -->|"50-500"| D["方差阈值，然后 L1 或树重要性"]
    B -->|"> 500"| E["方差阈值，然后互信息过滤，再对幸存者做 RFE"]

    C --> F{使用线性模型？}
    D --> F
    E --> F

    F -->|是| G["使用 L1 正则化做最终选择"]
    F -->|否 - 树模型| H["树重要性 + 排列重要性"]
    F -->|否 - 其他| I["使用你的模型做 RFE"]

    G --> J[验证：比较所选特征与全部特征]
    H --> J
    I --> J

    J --> K{性能提升？}
    K -->|是| L["使用所选特征发布"]
    K -->|否| M["尝试不同方法或保留全部特征"]
```

## 动手实现

### 第 1 步：生成具有已知特征结构的合成数据

```python
import numpy as np


def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)

    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)

    informative = np.column_stack([x1, x2, x3, x4, x5])

    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])

    noise = rng.randn(n_samples, 10) * 0.5

    X = np.hstack([informative, correlated, noise])
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)

    feature_names = (
        [f"info_{i}" for i in range(5)]
        + [f"corr_{i}" for i in range(5)]
        + [f"noise_{i}" for i in range(10)]
    )

    return X, y, feature_names
```

我们已知真实情况：特征 0-4 是有信息的（其中 3 和 4 是 0 和 1 的相关副本），特征 5-9 与有信息特征相关，特征 10-19 是纯噪声。一个好的选择方法应将 0-4 排在最前面，10-19 排在最后。

### 第 2 步：方差阈值

```python
def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances
```

### 第 3 步：互信息（离散）

```python
def discretize(x, n_bins=10):
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned


def mutual_information(X, y, n_bins=10):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples

    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))

        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi

    return mi_scores
```

### 第 4 步：递归特征消除

```python
def simple_logistic_importance(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        w -= lr * (X.T @ error) / n_samples
        b -= lr * np.mean(error)

    return w, b


def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total

    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)

        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)

    for idx in remaining:
        rankings[idx] = 1

    selected_mask = rankings == 1
    return selected_mask, rankings
```

### 第 5 步：L1 特征选择

```python
def soft_threshold(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y

        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)

        w -= lr * gradient_w
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b

    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w
```

### 第 6 步：基于树的重要性（简单决策树）

```python
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_idx):
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0

    best_threshold = None
    best_gain = -1.0
    parent_gini = gini_impurity(y)
    n = len(y)

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            continue

        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[right_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def tree_importance(X, y, n_trees=50, max_depth=5, seed=42):
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    for _ in range(n_trees):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        feature_subset = rng.choice(n_features, size=max(1, int(np.sqrt(n_features))), replace=False)

        X_boot = X[sample_idx]
        y_boot = y[sample_idx]

        tree_imp = _build_tree_importance(X_boot, y_boot, feature_subset, max_depth)
        importances += tree_imp

    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


def _build_tree_importance(X, y, feature_subset, max_depth, depth=0):
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    if depth >= max_depth or len(np.unique(y)) <= 1 or len(y) < 4:
        return importances

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    for f in feature_subset:
        threshold, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_threshold = threshold

    if best_feature is None or best_gain <= 0:
        return importances

    importances[best_feature] += best_gain * len(y)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    importances += _build_tree_importance(X[left_mask], y[left_mask], feature_subset, max_depth, depth + 1)
    importances += _build_tree_importance(X[right_mask], y[right_mask], feature_subset, max_depth, depth + 1)

    return importances
```

### 第 7 步：运行所有方法并进行比较

代码文件在同一合成数据集上运行所有五种方法，并打印比较表格，显示每种方法选择了哪些特征。

## 如何使用（scikit-learn）

使用 scikit-learn 时，特征选择已内置到流水线中：

```python
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)

mi_scores = mutual_info_classif(X, y)
top_k = np.argsort(mi_scores)[-10:]

rfe_selector = RFE(LogisticRegression(), n_features_to_select=10)
rfe_selector.fit(X, y)
X_rfe = rfe_selector.transform(X)

lasso_selector = SelectFromModel(Lasso(alpha=0.01))
lasso_selector.fit(X, y)
X_lasso = lasso_selector.transform(X)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

从零实现的代码准确展示了每种方法内部的工作机制。方差阈值只是计算 `var(X, axis=0)` 并应用掩码。互信息是计数联合频数和边际频数构成的列联表。RFE 是一个训练、排序、剪枝的循环。L1 是带软阈值步骤的梯度下降。树重要性是在多次分裂中累积不纯度减少量。没有魔法——只有统计和循环。

scikit-learn 版本增加了鲁棒性（例如 `mutual_info_classif` 使用 k-NN 密度估计而非分箱）、速度（C 实现）和流水线集成。

## 交付内容

本课程产出：
- `outputs/skill-feature-selector.md` —— 一份快速参考决策树，用于选择正确的特征选择方法

## 练习

1. **前向选择**：实现 RFE 的反向操作。从零个特征开始，每一步添加使模型性能提升最大的特征，直到添加特征不再有帮助。将所选特征与 RFE 结果进行比较。哪个更快？哪个结果更好？

2. **稳定性选择**：运行 L1 特征选择 50 次，每次使用数据的随机 80% 子样本，并采用略有不同的 alpha 值。统计每个特征被选中的次数。在超过 80% 的运行中被选中的特征称为“稳定”特征。将稳定特征与单次 L1 选择的结果进行比较。哪个更可靠？

3. **多重共线性检测**：计算所有特征的相关矩阵。实现一个函数，给定相关性阈值（例如 0.9），从每个高度相关的配对中移除一个特征（保留与目标互信息更高的那个）。在合成数据集上测试，验证它能移除冗余的相关特征。

4. **特征选择流水线**：将方差阈值、互信息过滤和 RFE 串联成一个流水线。首先移除近零方差特征，然后按互信息保留前 50%，最后对幸存者运行 RFE。将此流水线与单独对所有特征运行 RFE 进行比较。流水线更快吗？准确率是否相当？

5. **从零实现排列重要性**：实现排列重要性。对每个特征，打乱其取值 10 次，测量 F1 分数的平均下降量。将排序结果与基于树的重要性进行比较。找出它们不一致的案例并解释原因（提示：相关特征）。

## 关键术语

| 术语 | 人们常说的话 | 实际含义 |
|------|----------------|----------------------|
| 过滤式方法 | “独立地对特征评分” | 一种特征选择方法，使用统计度量对特征排序，不训练模型，单独评估每个特征 |
| 封装式方法 | “用模型来挑选特征” | 一种特征选择方法，通过训练模型并使用其性能作为选择标准来评估特征子集 |
| 嵌入式方法 | “模型在训练过程中选择特征” | 特征选择作为模型拟合的一部分发生，例如 L1 正则化将权重推向零 |
| 互信息 | “一个变量能告诉你关于另一个变量的多少信息” | 衡量在知道 X 的情况下 Y 的不确定性减少量，能够捕捉线性和非线性依赖关系 |
| 递归特征消除 | “训练、排序、剪枝、重复” | 一种迭代的封装式方法，训练模型、移除最不重要的特征，重复直到达到目标数量 |
| L1 / Lasso 正则化 | “能杀死特征的惩罚项” | 将权重绝对值之和加入损失函数，使不重要特征的权重精确为零 |
| 方差阈值 | “移除常量特征” | 丢弃方差低于指定阈值的特征，过滤掉不携带任何信息的特征 |
| 特征重要性 | “哪些特征最重要” | 表示每个特征对模型预测贡献程度的分数，通过分裂增益（树）或系数大小（线性）计算 |
| 排列重要性 | “打乱并衡量损害” | 通过随机打乱每个特征的取值并衡量模型性能下降来评估特征重要性 |
| 维度灾难 | “特征太多，数据不足” | 添加特征会指数级增加特征空间体积，使数据稀疏且距离无意义的现象 |

## 进一步阅读

- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) —— 特征选择方法的基础综述，至今仍被广泛引用
- [scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html) —— 带有代码示例的过滤式、封装式和嵌入式方法实用参考
- [Stability Selection (Meinshausen & Buhlmann, 2010)](https://arxiv.org/abs/0809.2932) —— 将子采样与特征选择结合，以获得稳健且可重复的结果
- [Beware Default Random Forest Importances (Strobl et al., 2007)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25) —— 展示了基于树的重要性的基数偏差，并提出条件重要性作为替代方案