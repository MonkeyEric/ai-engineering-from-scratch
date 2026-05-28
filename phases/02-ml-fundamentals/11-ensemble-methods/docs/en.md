# 集成方法

> 一群弱学习器，正确组合，就能变成一个强学习器。这不是比喻。这是一个定理。

**类型：** 构建
**语言：** Python
**先修知识：** 第二阶段，第10课（偏差-方差权衡）
**时间：** 约120分钟

## 学习目标

- 从零实现 AdaBoost 和梯度提升，并解释 Boosting 如何顺序地减少偏差
- 构建一个 Bagging 集成，并演示平均去相关模型如何在不增加偏差的情况下减少方差
- 比较 Bagging、Boosting 和 Stacking 在减少不同误差成分方面的特点
- 评估集成多样性，并解释随着弱学习器独立性的增加，多数投票准确率为何会提高

## 问题描述

单棵决策树训练快且易于解释，但它容易过拟合。单个线性模型在复杂边界上会欠拟合。你可能会花费数天时间精心设计完美的模型架构。或者，你也可以将一堆不完美的模型组合起来，得到一个比其中任何一个都更好的模型。

集成方法正是这样做的。它们是赢得表格数据 Kaggle 比赛最可靠的技术，驱动着大多数生产级机器学习系统，并生动地展示了偏差-方差权衡。Bagging 减少方差。Boosting 减少偏差。Stacking 学习在不同的输入上信任哪个模型。

## 核心概念

### 为什么集成方法有效

假设你有 N 个独立的分类器，每个的准确率 p > 0.5。多数投票的准确率为：

```
P(多数正确) = 对 k > N/2 求和 C(N,k) * p^k * (1-p)^(N-k)
```

对于 21 个准确率为 60% 的分类器，多数投票准确率约为 74%。当有 101 个分类器时，准确率会上升到 84%。当模型犯不同的错误时，错误会相互抵消。

关键要求是**多样性**。如果所有模型都犯相同的错误，组合它们毫无帮助。集成之所以有效，是因为它们通过以下方式产生多样化的模型：

- 不同的训练子集（Bagging）
- 不同的特征子集（随机森林）
- 顺序错误纠正（Boosting）
- 不同的模型家族（Stacking）

### Bagging（自助聚合）

Bagging 通过在每个模型上使用不同的训练数据自助采样来创造多样性。

```mermaid
flowchart TD
    D[训练数据] --> B1[自助样本 1]
    D --> B2[自助样本 2]
    D --> B3[自助样本 3]
    D --> BN[自助样本 N]

    B1 --> M1[模型 1]
    B2 --> M2[模型 2]
    B3 --> M3[模型 3]
    BN --> MN[模型 N]

    M1 --> V[平均或多数投票]
    M2 --> V
    M3 --> V
    MN --> V

    V --> P[最终预测]
```

自助样本是从原始数据中有放回地抽取的，大小与原始数据相同。每个自助样本中出现约 63.2% 的唯一样本。剩下的 36.8%（袋外样本）提供了一个免费的验证集。

Bagging 在不显著增加偏差的情况下减少了方差。每棵树都对其自助样本过拟合，但每棵树的过拟合不同，因此平均会抵消掉噪声。

**随机森林** 是 Bagging 的一个变种：在每个分裂点，只考虑一个随机的特征子集。这迫使树之间产生更多多样性。典型的候选特征数量是分类问题的 `sqrt(n_features)` 和回归问题的 `n_features / 3`。

### Boosting（顺序错误纠正）

Boosting 顺序地训练模型。每个新模型都专注于前序模型预测错误的样本。

```mermaid
flowchart LR
    D[带权重的数据] --> M1[模型 1]
    M1 --> E1[找出错误]
    E1 --> W1[增加错误样本的权重]
    W1 --> M2[模型 2]
    M2 --> E2[找出错误]
    E2 --> W2[增加错误样本的权重]
    W2 --> M3[模型 3]
    M3 --> F[所有模型的加权和]
```

Boosting 减少偏差。每个新模型都会纠正当前集成模型的系统性错误。最终预测是所有模型的加权和，其中表现更好的模型获得更高的权重。

权衡：如果运行太多轮次，Boosting 可能会过拟合，因为它会持续拟合更难（可能是噪声）的样本。

### AdaBoost

AdaBoost（自适应 Boosting）是第一个实用的 Boosting 算法。它适用于任何基学习器，典型的基学习器是决策树桩（深度为1的决策树）。

算法流程：

```
1. 初始化样本权重: w_i = 1/N 对所有 i

2. 对于 t = 1 到 T:
   a. 在带权数据上训练弱学习器 h_t
   b. 计算加权误差:
      err_t = sum(w_i * I(h_t(x_i) != y_i)) / sum(w_i)
   c. 计算模型权重:
      alpha_t = 0.5 * ln((1 - err_t) / err_t)
   d. 更新样本权重:
      w_i = w_i * exp(-alpha_t * y_i * h_t(x_i))
   e. 归一化权重，使其和为 1

3. 最终预测: H(x) = sign(sum(alpha_t * h_t(x)))
```

误差越低的模型获得越高的 alpha。被错误分类的样本获得更高的权重，以便下一个模型重点关注它们。

### 梯度提升

梯度提升将 Boosting 推广到任意损失函数。它不是重新加权样本，而是让每个新模型拟合当前集成模型预测的残差（损失函数的负梯度）。

```
1. 初始化: F_0(x) = argmin_c sum(L(y_i, c))

2. 对于 t = 1 到 T:
   a. 计算伪残差:
      r_i = -dL(y_i, F_{t-1}(x_i)) / dF_{t-1}(x_i)
   b. 拟合一个树 h_t 来预测残差 r_i
   c. 找到最佳步长:
      gamma_t = argmin_gamma sum(L(y_i, F_{t-1}(x_i) + gamma * h_t(x_i)))
   d. 更新:
      F_t(x) = F_{t-1}(x) + learning_rate * gamma_t * h_t(x)

3. 最终预测: F_T(x)
```

对于平方误差损失，伪残差就是实际残差：`r_i = y_i - F_{t-1}(x_i)`。每棵树都直接拟合前一个集成模型的误差。

学习率控制着每棵树的贡献大小。较小的学习率需要更多的树，但泛化能力更好。典型值：0.01 到 0.3。

### XGBoost：为什么它主宰表格数据

XGBoost（极致梯度提升）是带有工程优化的梯度提升，使其快速、准确且抗过拟合：

- **正则化目标：** 对叶节点权重的 L1 和 L2 惩罚，防止单棵树过于自信
- **二阶近似：** 使用损失的一阶和二阶导数，提供更好的分裂决策
- **稀疏感知分裂：** 通过在每个分裂点学习缺失数据的最佳方向，原生处理缺失值
- **列采样：** 像随机森林一样，在每个分裂点对特征进行采样以增加多样性
- **加权分位数草图：** 为分布式数据上的连续特征高效找到分裂点
- **缓存感知块结构：** 针对 CPU 缓存行优化的内存布局

对于表格数据，XGBoost（及其继任者 LightGBM）的表现 consistently 优于神经网络。这种情况在短期内不会改变。如果你的数据可以放入一个行和列的表格中，从梯度提升开始。

### Stacking（元学习）

Stacking 使用多个基础模型的预测作为元学习器的特征。

```mermaid
flowchart TD
    D[训练数据] --> M1[模型 1: 随机森林]
    D --> M2[模型 2: SVM]
    D --> M3[模型 3: 逻辑回归]

    M1 --> P1[预测 1]
    M2 --> P2[预测 2]
    M3 --> P3[预测 3]

    P1 --> META[元学习器]
    P2 --> META
    P3 --> META

    META --> F[最终预测]
```

元学习器学习对于不同的输入应该信任哪个基础模型。如果随机森林在某些区域表现更好，而 SVM 在其他区域表现更好，元学习器将学会相应地分配任务。

为了避免数据泄露，基础模型的预测必须通过在训练集上进行交叉验证来生成。你绝不能在相同的数据上既训练基础模型又生成元特征。

### 投票

最简单的集成方法。直接组合预测。

- **硬投票：** 对类别标签进行多数投票。
- **软投票：** 平均预测概率，选择平均概率最高的类别。由于使用了置信度信息，通常效果更好。

## 动手实现

### 步骤 1：决策树桩（基学习器）

`code/ensembles.py` 中的代码从零实现了所有内容。我们从决策树桩开始：一个只有一次分裂的树。

```python
class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float("inf")

        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for thresh in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    pred[polarity * X[:, f] < polarity * thresh] = -1
                    error = np.sum(weights[pred != y])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = f
                        self.threshold = thresh
                        self.polarity = polarity

    def predict(self, X):
        n = X.shape[0]
        pred = np.ones(n)
        idx = self.polarity * X[:, self.feature_idx] < self.polarity * self.threshold
        pred[idx] = -1
        return pred
```

### 步骤 2：从零实现 AdaBoost

```python
class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.full(n, 1 / n)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            pred = stump.predict(X)

            err = np.sum(weights[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)
            weights *= np.exp(-alpha * y * pred)
            weights /= weights.sum()

            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        total = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.sign(total)
```

### 步骤 3：从零实现梯度提升

```python
class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        current_pred = np.full(len(y), self.initial_pred)

        for _ in range(self.n_estimators):
            residuals = y - current_pred
            tree = SimpleRegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            current_pred += self.lr * update
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
```

### 步骤 4：与 sklearn 对比

代码验证了我们从零实现的版本产生的准确率与 sklearn 的 `AdaBoostClassifier` 和 `GradientBoostingClassifier` 相似，并并排比较了所有方法。

## 使用示例

### 何时使用每种方法

| 方法 | 主要减少 | 最佳适用场景 | 注意事项 |
|---|---|---|---|
| Bagging / 随机森林 | 方差 | 噪声数据，多特征 | 对偏差无效 |
| AdaBoost | 偏差 | 干净数据，简单基学习器 | 对离群值和噪声敏感 |
| 梯度提升 | 偏差 | 表格数据，竞赛 | 训练慢，不经调优易过拟合 |
| XGBoost / LightGBM | 两者 | 生产级表格机器学习 | 超参数多 |
| Stacking | 两者 | 追求最后 1-2% 的准确率 | 复杂，元学习器有过拟合风险 |
| 投票 | 方差 | 快速组合多样化模型 | 仅在模型多样化时有效 |

### 表格数据的生产级技术栈

对于大多数表格预测问题，按以下顺序尝试：

1.  **LightGBM 或 XGBoost**，使用默认参数
2.  调整 n_estimators, learning_rate, max_depth, min_child_weight
3.  如果需要追求最后的 0.5%，使用 3-5 个多样化模型构建一个 Stacking 集成
4.  全程使用交叉验证

尽管研究界持续尝试，但神经网络在表格数据上的表现几乎总是逊于梯度提升。TabNet、NODE 等架构偶尔能与之匹敌，但很少能超越调优良好的 XGBoost。

## 交付成果

本课程产出 `outputs/prompt-ensemble-selector.md` —— 一个帮助您为给定数据集选择合适的集成方法的提示。描述您的数据（大小、特征类型、噪声水平、类别平衡）和要解决的问题。该提示会引导您完成一个决策清单，推荐一种方法，建议起始超参数，并警告该方法的常见错误。同时产出 `outputs/skill-ensemble-builder.md`，包含完整的选择指南。

## 练习

1.  修改 AdaBoost 实现，跟踪每轮之后的训练准确率。绘制准确率随估计器数量变化的曲线。它何时收敛？

2.  通过在回归树中添加随机特征子采样，从零实现一个随机森林。使用 `max_features=sqrt(n_features)` 训练 100 棵树，并平均预测值。将方差减少效果与单棵树进行比较。

3.  在梯度提升实现中添加早停：每轮后跟踪验证损失，当连续 10 轮没有改善时停止。它实际需要多少棵树？

4.  构建一个包含三个基础模型（逻辑回归、决策树、K近邻）和一个逻辑回归元学习器的 Stacking 集成。使用 5 折交叉验证生成元特征。与每个单独的基础模型进行比较。

5.  在相同数据集上使用默认参数运行 XGBoost。将其准确率与您从零实现的梯度提升进行比较。对两者计时。速度差异有多大？

## 关键术语表

| 术语 | 人们通常说 | 实际含义 |
|---|---|---|
| Bagging | "在随机子集上训练" | 自助聚合：在自助样本上训练模型，平均预测值以减少方差 |
| Boosting | "关注困难样本" | 顺序训练模型，每个模型纠正当前集成模型的错误，以减少偏差 |
| AdaBoost | "重新加权数据" | 通过更新样本权重进行 Boosting；错误分类的点在下个学习器中获得更高权重 |
| 梯度提升 | "拟合残差" | 通过让每个新模型拟合损失函数的负梯度来进行 Boosting |
| XGBoost | "Kaggle 大杀器" | 带有正则化、二阶优化和系统级速度技巧的梯度提升 |
| Stacking | "模型叠模型" | 使用基础模型的预测作为元学习器的输入特征 |
| 随机森林 | "许多随机树" | 对决策树进行 Bagging，并在每个分裂点添加随机特征子采样以增加多样性 |
| 集成多样性 | "犯不同的错误" | 模型必须在错误上不相关，集成才能比单个模型更好 |
| 袋外误差 | "免费验证" | 不在自助样本中的样本（约 36.8%）可作为验证集，无需单独留出 |

## 延伸阅读

- [Schapire & Freund: Boosting: Foundations and Algorithms](https://mitpress.mit.edu/9780262526036/) —— AdaBoost 创建者所著的书
- [Friedman: Greedy Function Approximation: A Gradient Boosting Machine (2001)](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) —— 原始的梯度提升论文
- [Chen & Guestrin: XGBoost (2016)](https://arxiv.org/abs/1603.02754) —— XGBoost 论文
- [Wolpert: Stacked Generalization (1992)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) —— 原始的 Stacking 论文
- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) —— 实用参考