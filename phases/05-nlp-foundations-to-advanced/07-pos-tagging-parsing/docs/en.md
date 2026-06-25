# 词性标注与句法分析

> 语法曾一度不受待见。后来，几乎每个大语言模型（LLM）流水线都需要验证结构化抽取，它又回来了。

**类型：** 实践构建
**语言：** Python
**前置知识：** 第 5 阶段 · 01（文本处理），第 2 阶段 · 14（朴素贝叶斯）
**时长：** 约 45 分钟

## 问题背景

第 01 课提到，词形还原需要词性标签。如果不知道 `running` 是动词，词形还原器就无法将其还原为 `run`；如果不知道 `better` 是形容词，也无法还原为 `good`。

这句话背后藏着一个完整的子领域。词性标注（POS tagging）为每个词分配语法类别；句法分析则还原句子的树状结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典 NLP 花了二十年时间精进这两者。随后深度学习将它们压缩为预训练 Transformer 之上的 token 分类任务，研究社区也随之转向。

但应用社区没有。每个结构化抽取流水线在底层仍在使用词性标注和依存树。LLM 生成的 JSON 需要依据语法约束进行校验；问答系统利用依存分析来分解查询；机器翻译质量评估器会检查分析树的对齐。

值得了解。本课将介绍标签集、基线方法，以及何时应该停止从零实现、转而调用 spaCy。

## 核心概念

**词性标注**为每个 token 标注一个语法类别。**宾州树库（Penn Treebank, PTB）**标签集是英语默认标准，包含 36 个标签，其中一些区分在普通读者看来过于精细：`NN` 单数名词、`NNS` 复数名词、`NNP` 专有名词单数、`VBD` 动词过去式、`VBZ` 动词第三人称单数现在时，等等。**通用依存（Universal Dependencies, UD）**标签集更粗粒度（17 个标签）且语言无关，已成为跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**生成一棵树。主要有两种形式：

- **成分句法分析。** 名词短语、动词短语、介词短语相互嵌套。输出是一棵非终结符类别（NP、VP、PP）的树，词作为叶子节点。
- **依存句法分析。** 每个词都有一个它所依赖的中心词（head），边标注语法关系。输出是一棵树，每条边都是一个（head, dependent, relation）三元组。

依存分析在 2010 年代胜出，因为它能干净地泛化到不同语言，尤其是语序自由的语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 动手实现

### 步骤 1：最频繁词性基线

能工作的最简单词性标注器：对每个词，预测它在训练集中出现最频繁的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这一基线能达到约 85% 的准确率。不算好，但任何严肃模型都不应低于这个下限。

### 步骤 2：二元 HMM 词性标注器

对序列的联合概率建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（给定前一个标签的当前标签概率）和发射概率（给定标签的词概率）。用带拉普拉斯平滑的计数估计两者，再用 Viterbi 算法（在标签网格上做动态规划）解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 语料库上的二元 HMM 可达约 93% 准确率。从 85% 到 93% 的提升主要归功于转移概率——模型学到了 `DET NOUN` 很常见，而 `NOUN DET` 很罕见。

### 步骤 3：现代标注器为何更强

转移概率和发射概率都是局部的，无法捕捉 `saw` 在 "I bought a saw" 中是名词、在 "I saw the movie" 中是动词的现象。使用任意特征（后缀、词形、前后词、当前词）的 CRF 可达约 97%；BiLSTM-CRF 或 Transformer 可达 98% 以上。

这项任务的天花板由标注者分歧决定。在 Penn Treebank 上，人工标注者的一致性约为 97%。超过 98% 的模型很可能在过拟合测试集。

### 步骤 4：依存分析概览

从零完整实现依存分析超出本课范围；经典教材处理见 Jurafsky 和 Martin。需要了解的两大经典范式：

- **基于转移**的解析器（arc-eager、arc-standard）像移进-归约解析器一样工作：读入 token，将其压入栈，再应用创建弧的归约动作。贪心解码很快。经典实现是 MaltParser。现代神经网络版本：Chen 和 Manning 的基于转移解析器。
- **基于图**的解析器（Eisner 算法、Dozat-Manning 双仿射）为所有可能的 head-dependent 边打分，并选择最大生成树。较慢但更准确。

对大多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构就清晰了。

## 应用

每个生产级 NLP 库都提供词性和依存分析器，作为标准流水线的一部分。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，与分词、NER、词形还原集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。Stanford 对 CoreNLP 的继任者，在 60 多种语言上达到最先进水平。
- **trankit**。基于 Transformer，UD 准确率优秀。
- **NLTK**。`pos_tag`。可用、较慢、较老。适合教学。

### 2026 年这仍然重要的场景

- **词形还原。** 第 01 课需要词性才能正确还原。始终如此。
- **从 LLM 输出中做结构化抽取。** 验证生成句是否符合语法约束（如主谓一致、必要的修饰语）。
- **基于方面的情感分析。** 依存分析能告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 可通过分析树分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系是语言无关的，可对新语言进行零样本结构化分析。
- **低算力流水线。** 如果你无法部署 Transformer，词性标注 + 依存分析 + 地名表仍能走得很远。

## 交付

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: 为下游 NLP 任务设计一套经典词性 + 依存分析流水线。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定一个下游任务（信息抽取、改写验证、查询分解、词形还原），你输出：

1. 要使用的标签集。仅英语遗留流水线用 Penn Treebank，多语言或跨语言任务用 Universal Dependencies。
2. 使用的库。生产环境通常用 spaCy，学术级多语言用 stanza，最高 UD 准确率用 trankit。需给出具体模型 ID。
3. 集成模式。写出调用库并消费所需属性（`.pos_`、`.dep_`、`.head`）的 3-5 行代码。
4. 要测试的失效模式。名动歧义（`saw`、`book`、`can`）和 PP 附着歧义是经典陷阱。采样 20 条输出并人工检查。

不要推荐从零自研解析器。从零构建解析器是研究项目，不是应用任务。如果某流水线消费词性标签却不处理大小写变体，请标记其为脆弱实现。
```

## 练习

1. **简单。** 在一个小型标注语料库（例如 NLTK 的 Brown 子集）上运行最频繁词性基线，在留出句子上测量准确率，验证约 85% 的结果。
2. **中等。** 训练上述二元 HMM，报告每个标签的精确率/召回率。HMM 最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存分析从 1000 句样本中提取主谓宾三元组，在 50 条人工标注三元组上评估，记录提取失败的场景（通常是被动句、并列结构和省略主语）。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| 词性标签 | 词的类别 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标签集 | 英语专用。包含细粒度的动词时态和名词数。 |
| Universal Dependencies | 多语言标签集 | 比 PTB 更粗粒度；语言无关；跨语言工作的默认选择。 |
| 依存分析 | 句子树 | 每个词有一个 head，每条边有一个语法关系。 |
| Viterbi | 动态规划 | 给定发射概率和转移概率，寻找概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky 和 Martin —《Speech and Language Processing》第 8、18 章](https://web.stanford.edu/~jurafsky/slp3/) — 词性与句法分析的经典教材。
- [Universal Dependencies 项目](https://universaldependencies.org/) — 每个多语言解析器都在使用的跨语言标签集和树库集合。
- [spaCy 语言特征指南](https://spacy.io/usage/linguistic-features) — `Token` 上每个暴露属性的实用参考。
- [Chen 和 Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经网络解析器带入主流的论文。
