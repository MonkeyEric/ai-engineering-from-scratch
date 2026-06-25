# 情感分析

> 最经典的 NLP 任务。关于经典文本分类，你所需知道的大部分内容都能在这里找到。

**类型：** 构建
**语言：** Python
**先修：** Phase 5 · 02（词袋模型 + TF-IDF），Phase 2 · 14（朴素贝叶斯）
**时长：** 约 75 分钟

## 问题定义

"The food was not great." 是正面还是负面？

情感分析听起来很简单。评论者说他喜欢或不喜欢某样东西，给句子打个标签。它之所以成为最经典的 NLP 任务，是因为每个看似简单的案例背后都藏着一个困难的案例。否定会翻转语义，讽刺会反转语义。"Not bad at all" 是正面的，尽管它包含两个带负面色彩的词。表情符号携带的信号往往比周围文字更强。领域词汇也很重要（`tight` 在音乐评论里和时尚评论里含义不同）。

情感分析是经典 NLP 的一个活实验室。如果你理解为什么每个朴素基线都有特定的失效模式，你就理解了为什么人们会发明更复杂的模型。本节课从零实现一个朴素贝叶斯基线，加入逻辑回归，并指出那些让生产级情感分析成为合规级难题的陷阱。

## 核心概念

经典情感分析分两步。

1. **表示。** 把文本转换成特征向量。词袋模型、TF-IDF 或 n-gram。
2. **分类。** 在有标注样本上拟合一个线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是"最笨但能用"的模型。假设每个特征在给定标签下相互独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时把概率相乘。"朴素"的独立性假设错得离谱，结果却出奇地好。原因在于：面对稀疏文本特征和中等规模数据时，分类器更关心每个词偏向哪一侧，而不是词与词之间的依赖关系。

逻辑回归修正了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为一个 bigram 特征会学到负权重，而朴素贝叶斯无法为从未标注过的 bigram 做到这一点。

## 动手实现

### 步骤 1：一个真实的微型数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意设得很小。实际工作中会使用数万个样本（IMDb、SST-2、Yelp polarity）。数学原理完全相同。

### 步骤 2：从零实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加法平滑（alpha=1.0）即拉普拉斯平滑。没有它，类别中未出现的词概率为零，对数会爆炸。实践中常用 `alpha=0.01`，`alpha=1.0` 是教学默认值。

### 步骤 3：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

这里 L2 正则化很重要。文本特征稀疏，没有 L2 模型会记住训练样本。从 `0.01` 开始并调参。

### 步骤 4：处理否定（失效模式）

考虑 "not good" 和 "not bad"。词袋分类器看到的是 `{not, good}` 和 `{not, bad}`，只能从训练集中哪个出现更多来学习。Bigram 分类器看到的是 `not_good` 和 `not_bad`，把它们当作不同特征来学习。通常这就够了。

当你没有 bigram 时，一个更粗糙但有效的做法是**否定范围标注**：在否定词之后、下一个标点之前，给每个 token 加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 成了不同特征，分类器可以给它们相反的权重。三行预处理，就能在情感基准上带来可测量的准确率提升。

### 步骤 5：真正重要的评估指标

类别不平衡时，仅看准确率会误导。真实情感语料库通常是 70-80% 正面或 70-80% 负面；一个永远预测多数类的分类器也能拿到 80% 的准确率，但毫无价值。以下每一项都要报告：

- **每类精确率与召回率。** 每个类别各一对。对它们做宏平均，得到一个尊重类别平衡的单一数值。
- **Macro-F1（不平衡数据的首选指标）。** 每类 F1 的等权平均。类别不平衡时用它代替准确率。
- **Weighted-F1（替代指标）。** 与 macro 类似，但按类别频次加权。当不平衡本身具有业务含义时，可与 macro-F1 一起报告。
- **混淆矩阵。** 原始计数。在相信任何单一指标之前，务必先查看混淆矩阵；它能揭示模型混淆的是哪两个类别。
- **每类错误样本。** 每个类别抽取 5 个错误预测并亲自阅读。没有什么能替代阅读真实错误。

对于严重不平衡的数据（比例 > 95:5），应报告 **AUROC** 和 **AUPRC**，而非准确率。AUPRC 对少数类更敏感，而这通常正是你关心的（垃圾信息、欺诈、罕见情感）。

**需要避免的常见错误。** 在不平衡数据上报告 micro-F1 会得到一个看起来很高的数字，因为它被多数类主导。Macro-F1 会强迫你看到少数类的表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 应用

scikit-learn 用六行就能正确实现。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三件事。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 加入 bigram，让 `not_good` 成为特征。`sublinear_tf=True` 减弱重复词的影响。这三个参数就是把 SST-2 上 75% 准确率基线提升到 85% 的关键。

### 什么时候该用 Transformer

- 讽刺检测。经典模型在这里完全失效。
- 长评论，情感在文档中间发生转折。
- 基于方面的情感分析。"Camera was great but battery was terrible." 你需要把情感归因到不同方面。只有 Transformer 或结构化输出模型能做到。
- 非英语、低资源语言。多语言 BERT 能免费提供零样本基线。

如果你需要以上任何一种，跳到 phase 7（Transformer 深入）。否则，TF-IDF + bigram + 否定处理的朴素贝叶斯或逻辑回归，就是你 2026 年的生产级基线。

### 可复现性陷阱（再次强调）

重新训练情感模型是常规操作，重新评估却不是。论文中报告的准确率来自特定的划分、特定的预处理、特定的分词器。如果你在比较新模型和基线时没有使用完全相同的流程，得到的差异就会具有误导性。始终要在你的流程上重新生成基线，而不是直接引用论文里的数字。

## 交付

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: 为新数据集设计情感分析基线。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、规模、标签粒度、延迟预算），你输出：

1. 特征提取方案。指定分词器、n-gram 范围、停用词策略（通常保留）、否定处理（范围前缀或 bigram）。
2. 分类器。基线用朴素贝叶斯，生产用逻辑回归，仅在领域需要讽刺/方面/跨语言时才用 Transformer。
3. 评估计划。报告精确率、召回率、F1、混淆矩阵和每类错误样本（不要只报标量）。
4. 上线后需要监控的一种失效模式。领域漂移和讽刺是两大首要问题。

拒绝在情感任务中建议去除停用词。当类别不平衡时（例如 90% 正面），拒绝把准确率作为唯一指标。对子词丰富的语言，应标注为需要 FastText 或 Transformer 嵌入，而非词级 TF-IDF。
```

## 练习

1. **简单。** 在 scikit-learn 流程中加入 `apply_negation` 作为预处理步骤，并在小型情感数据集上测量 F1 变化。
2. **中等。** 实现类别加权逻辑回归（向 scikit-learn 传入 `class_weight="balanced"`，或自己推导梯度）。在合成的 90-10 类别不平衡数据上测量其影响。
3. **困难。** 通过训练第二个分类器来拟合情感模型的残差，构建一个讽刺检测器。记录实验设置。当准确率低于随机水平时提醒读者（二分类讽刺检测的随机水平约为 50%，而大多数第一次尝试都会落在那里）。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Polarity（极性） | 正面或负面 | 二分类标签；有时也会扩展到中性或细粒度（5 星）。 |
| Aspect-based sentiment（基于方面的情感分析） | 每个方面的极性 | 把情感归因到文本中提到的具体实体或属性。 |
| Negation scoping（否定范围） | 翻转附近 token | 在 "not" 之后直到标点的 token 加上 `NOT_` 前缀。 |
| Laplace smoothing（拉普拉斯平滑） | 计数加 1 | 防止朴素贝叶斯中出现零概率特征。 |
| L2 regularization（L2 正则化） | 收缩权重 | 向损失中加入 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) —— 奠基性综述。很长，但前四节涵盖了所有经典内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) —— 这篇论文证明了 bigram + 朴素贝叶斯在短文本上很难被击败。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— `CountVectorizer`、`TfidfVectorizer` 以及你要调节的每个参数的参考文档。
