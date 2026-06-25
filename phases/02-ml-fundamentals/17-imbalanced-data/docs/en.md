# 处理不平衡数据

> 当你的数据中99%都是“正常”时，准确率就是一个谎言。

**类型：** 构建
**语言：** Python
**前置要求：** 第二阶段，第01-09课（特别是评估指标）
**时间：** 约90分钟

## 学习目标

- 从零实现SMOTE，并解释合成过采样与随机复制有何不同
- 使用F1分数、AUPRC和Matthews相关系数来评估不平衡分类器，而非使用准确率
- 比较类别权重、阈值调优和重采样策略，并针对给定的不平衡比率选择合适的方法
- 构建一个完整的不平衡数据处理流水线，结合SMOTE、类别权重和阈值优化

## 问题所在

你构建了一个欺诈检测模型。它达到了99.9%的准确率。你庆祝了一番。然后你发现它把每一笔交易都预测为“非欺诈”。

这不是一个bug。当只有0.1%的交易是欺诈时，这是模型做出的理性行为。模型学习到，总是猜测多数类可以最小化整体误差。这在技术上是正确的，但完全没用。

这种事情在真正重要的分类问题中随处可见。疾病诊断：1%的阳性率。网络入侵：0.01%的攻击。制造缺陷：0.5%的次品。垃圾邮件过滤：20%的垃圾邮件。流失预测：5%的流失客户。少数类越重要，它往往就越稀有。

准确率之所以失效，是因为它平等地对待所有正确的预测。正确标记一笔合法交易和正确抓住一次欺诈，在准确率上都只算一分。但抓住欺诈才是模型存在的全部理由。我们需要能够强迫模型关注那个稀有但重要的类别的指标、技术和训练策略。

## 核心概念

### 为什么准确率会失效

考虑一个包含1000个样本的数据集：990个负类，10个正类。一个始终预测为负类的模型：

|  | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际正类 | 0 (TP) | 10 (FN) |
| 实际负类 | 0 (FP) | 990 (TN) |

准确率 = (0 + 990) / 1000 = 99.0%

该模型抓住了零个欺诈。零个疾病。零个缺陷。但准确率却显示99%。这就是为什么在不平衡问题上准确率是危险的。

### 更好的指标

**精确率** = TP / (TP + FP)。在所有被标记为正类的样本中，有多少是真正例？高精确率意味着很少的误报。

**召回率** = TP / (TP + FN)。在所有实际为正类的样本中，我们抓住了多少？高召回率意味着很少的漏报。

**F1 分数** = 2 * 精确率 * 召回率 / (精确率 + 召回率)。这是精确率和召回率的调和平均数。相比算术平均数，它更能惩罚精确率和召回率之间的极端不平衡。

**F-beta 分数** = (1 + beta^2) * 精确率 * 召回率 / (beta^2 * 精确率 + 召回率)。当beta > 1时，召回率更重要。当beta < 1时，精确率更重要。F2在欺诈检测中常用（漏掉欺诈比误报更糟糕）。

**AUPRC**（精确率-召回率曲线下面积）。类似于AUC-ROC，但对于不平衡数据信息更丰富。一个随机分类器的AUPRC等于正类比率（不像ROC是0.5）。这使得改进更容易被观察到。

**Matthews相关系数** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。取值范围从-1到+1。仅当模型在两个类别上都表现良好时才会给出高分。即使类别大小差异很大，它也是平衡的。

对于上面那个“总是预测负类”的模型：精确率 = 0/0（未定义，通常设为0），召回率 = 0/10 = 0，F1 = 0，MCC = 0。这些指标正确地识别出该模型毫无价值。

### 不平衡数据处理流水线

```mermaid
flowchart TD
    A[不平衡数据集] --> B{不平衡比率如何？}
    B -->|轻度: 80/20| C[类别权重]
    B -->|中度: 95/5| D[SMOTE + 阈值调优]
    B -->|严重: 99/1| E[SMOTE + 类别权重 + 阈值调优]
    C --> F[训练模型]
    D --> F
    E --> F
    F --> G[使用 F1 / AUPRC / MCC 评估]
    G --> H{足够好了吗？}
    H -->|否| I[尝试不同策略]
    H -->|是| J[部署并持续监控]
    I --> B
```

### SMOTE: 合成少数类过采样技术

随机过采样会复制现有的少数类样本。这有效，但有过度拟合的风险，因为模型会反复看到完全相同的点。

SMOTE创建新的、合理的但非副本的合成少数类样本。该算法如下：

1. 对于每一个少数类样本 x，在其它少数类样本中找出它的 k 个最近邻
2. 随机挑选一个邻居
3. 在 x 和该邻居之间的线段上创建一个新样本

公式：`新样本 = x + random(0, 1) * (邻居 - x)`

这是通过在真实少数类点之间进行插值，在特征空间的同一区域创建样本，而不仅仅是复制现有数据。

```mermaid
flowchart LR
    subgraph Original["原始少数类点"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE 生成过程"]
        direction TB
        S1["挑选 x1，邻居 x2"]
        S2["随机参数 t = 0.4"]
        S3["新点 = x1 + 0.4*(x2-x1)"]
        S4["新点 = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["增强后的数据集"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["合成点 (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### 采样策略对比

**随机过采样**：复制少数类样本以匹配多数类数量。
- 优点：简单，无信息丢失
- 缺点：完全相同的副本导致过拟合，增加训练时间

**随机欠采样**：移除多数类样本以匹配少数类数量。
- 优点：训练速度快，简单
- 缺点：丢弃可能有用的多数类数据，方差较高

**SMOTE**：通过插值创建合成的少数类样本。
- 优点：生成新的数据点，相比随机过采样减少了过拟合
- 缺点：可能在决策边界附近产生噪声样本，不考虑多数类分布

| 策略 | 数据改变方式 | 风险 | 何时使用 |
|----------|-------------|------|-------------|
| 过采样 | 复制少数类 | 过拟合 | 小数据集，轻度不平衡 |
| 欠采样 | 移除多数类 | 信息丢失 | 大数据集，需要快速训练 |
| SMOTE | 添加合成少数类 | 边界噪声 | 中度不平衡，有足够少数类样本用于k-NN |

### 类别权重

不改变数据，而是改变模型对待错误的方式。给误分类少数类赋予更高的权重。

对于一个有950个负类和50个正类的二分类问题：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类获得了19倍的权重。误分类一个正类样本的代价相当于误分类19个负类样本。模型被迫去关注少数类。

在逻辑回归中，这会修改损失函数：

```
加权损失 = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中 w_i 取决于样本 i 的类别。

在期望意义下，类别权重在数学上等价于过采样，但不需要创建新的数据点。这使得它更快，并且避免了重复样本带来的过拟合风险。

### 阈值调优

大多数分类器输出一个概率。默认阈值是0.5：如果 P(正类) >= 0.5，则预测为正类。但0.5是主观决定的。当类别不平衡时，最优阈值通常要低得多。

流程如下：
1. 训练模型
2. 在验证集上获取预测概率
3. 从0.0到1.0遍历阈值
4. 计算每个阈值下的F1（或你选择的指标）
5. 挑选最大化该指标的阈值

```mermaid
flowchart LR
    A[模型] --> B[预测概率]
    B --> C[遍历阈值 0.0 至 1.0]
    C --> D[计算每个阈值下的F1]
    D --> E[挑选最佳阈值]
    E --> F[在生产环境中使用]
```

一个模型对于某笔欺诈交易可能输出 P(欺诈) = 0.15。在阈值0.5下，这会被归类为非欺诈。在阈值0.10下，它会被正确捕获。概率校准的重要性不及排序——只要欺诈交易获得的概率高于非欺诈交易，就存在一个能够将它们分开的阈值。

### 代价敏感学习

这是类别权重的泛化。不是使用统一的代价，而是赋予具体的误分类代价：

| | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际正类 | 0（正确） | C_FN = 100 |
| 实际负类 | C_FP = 1 | 0（正确） |

漏掉一笔欺诈交易（FN）的代价是误报（FP）的100倍。模型优化的是总代价，而不是总错误数。

当你能够估算真实世界的代价时，这是最具原则性的方法。错过一次癌症诊断与一次导致额外活检的误报，其代价截然不同。明确这些代价会迫使做出正确的权衡。

### 决策流程图

```mermaid
flowchart TD
    A[开始: 面对不平衡数据集] --> B{不平衡程度如何？}
    B -->|"< 70/30"| C["轻度：优先尝试类别权重"]
    B -->|"70/30 至 95/5"| D["中度：SMOTE + 类别权重"]
    B -->|"> 95/5"| E["严重：结合多种策略"]
    C --> F{数据量够吗？}
    D --> F
    E --> F
    F -->|"< 1000 样本"| G["过采样或SMOTE，避免欠采样"]
    F -->|"1000-10000"| H["SMOTE + 阈值调优"]
    F -->|"> 10000"| I["欠采样可行，或用类别权重"]
    G --> J[训练 + 使用F1/AUPRC评估]
    H --> J
    I --> J
    J --> K{召回率够高吗？}
    K -->|否| L[降低阈值]
    K -->|是| M{精确率可接受吗？}
    M -->|否| N[提高阈值或增加特征]
    M -->|是| O[部署上线]
```

## 动手构建

### 步骤1: 生成一个不平衡数据集

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### 步骤2: 从零实现SMOTE

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### 步骤3: 随机过采样和欠采样

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### 步骤4: 带类别权重的逻辑回归

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### 步骤5: 阈值调优

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 步骤6: 评估函数

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### 步骤7: 对比所有方法

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 基线：不做任何处理
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# 过采样
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# 类别权重
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# 阈值调优（在预留的验证集上调优，而非测试集）
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

代码文件在一个脚本中运行以上所有内容并打印结果。

## 实际使用

借助scikit-learn和imbalanced-learn，这些技术都可以一行代码搞定：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

从零开始的实现精确地展示了每种技术到底在做什么。SMOTE只不过是在少数类上进行k-NN插值。类别权重就是给损失函数乘以系数。阈值调优就是一个在阈值上循环的过程。没有魔法。

## 交付成果

本课产出：
- `outputs/skill-imbalanced-data.md` —— 一份处理不平衡分类问题的决策清单

## 练习

1. **Borderline-SMOTE**：修改SMOTE实现，使其仅为靠近决策边界的少数类点生成合成样本（即那些k近邻中包含多数类样本的点）。在类别有重叠的数据集上，与标准SMOTE比较结果。

2. **代价矩阵优化**：实现代价敏感学习，其中代价矩阵是一个参数。创建一个函数，接收代价矩阵并返回最小化期望代价的最优预测结果。用不同的代价比率（1:10, 1:100, 1:1000）进行测试，绘制精确率-召回率权衡的变化图。

3. **阈值校准**：实现Platt缩放（对模型的原始输出拟合一个逻辑回归，以产生校准后的概率）。对比校准前后的精确率-召回率曲线。证明校准不会改变排序（AUC保持不变），但会让概率更有意义。

4. **平衡装袋集成**：训练多个模型，每个模型都在一个平衡的自助样本（所有少数类 + 随机抽样的多数类子集）上训练。平均它们的预测结果。将此方法与使用SMOTE的单一模型进行比较。衡量多次运行的性能和方差。

5. **不平衡比率实验**：取一个平衡数据集，逐步增加不平衡比率（50/50, 70/30, 90/10, 95/5, 99/1）。对每个比率，分别使用和不使用SMOTE进行训练。绘制两种方法的F1分数相对于不平衡比率的变化图。在什么比率下SMOTE开始产生有意义的差异？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|----------------------|
| 类别不平衡 | “某个类的样本多得多” | 数据集中类别的分布显著偏斜，导致模型偏向多数类 |
| SMOTE | “合成过采样” | 通过在现有少数类样本与其k个最近少数类邻居之间插值，创建新的少数类样本 |
| 类别权重 | “让稀有类别上的错误代价更高” | 用特定类别的权重乘以损失函数，使模型更严重地惩罚少数类的误分类 |
| 阈值调优 | “移动决策边界” | 将分类的概率阈值从默认的0.5更改为能优化所需指标的值 |
| 精确率-召回率权衡 | “两者不可兼得” | 降低阈值能捕获更多正例（更高召回率），但也会标记更多误报（更低精确率），反之亦然 |
| AUPRC | “PR曲线下面积” | 将精确率-召回率曲线总结为一个数字；当类别严重不平衡时，比AUC-ROC更具信息量 |
| Matthews相关系数 | “平衡的指标” | 预测标签与实际标签之间的相关性，仅当模型在两个类别上都表现良好时才产生高分 |
| 代价敏感学习 | “不同的错误代价不同” | 将真实世界的误分类代价纳入训练目标，使模型优化总代价，而非错误计数 |
| 随机过采样 | “复制少数类” | 重复少数类样本以平衡类别数量；简单但有对重复点过拟合的风险 |

## 扩展阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) —— 原始的SMOTE论文，仍然是不平衡学习领域引用最多的作品
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) —— 综合综述，涵盖采样、代价敏感和算法方法
- [imbalanced-learn 文档](https://imbalanced-learn.org/stable/) —— Python库，提供SMOTE变体、欠采样策略和流水线集成
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) —— 何时以及为何在不平衡问题上应优先选择PR曲线而非ROC曲线