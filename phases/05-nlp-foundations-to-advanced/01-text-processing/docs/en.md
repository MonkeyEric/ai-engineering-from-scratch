# 文本处理 —— 分词、词干提取、词形还原

> 语言是连续的，模型是离散的，预处理就是两者之间的桥梁。

**类型：** 实践
**语言：** Python
**前置知识：** Phase 2 · 14（朴素贝叶斯）
**时间：** ~45 分钟

## 问题

模型读不懂 "The cats were running."。它读的是整数。

每个 NLP 系统开篇都要回答三个问题：单词从哪里开始；单词的词根是什么；什么时候该把 "run"、"running"、"ran" 当作同一个词，什么时候又该把它们区分开。

分词出错，模型就会从垃圾数据中学习。如果你的分词器把 `don't` 当作一个 token，却把 `do n't` 当作两个，训练分布就会分裂。如果你的词干提取器把 `organization` 和 `organ` 压缩成同一个词干，主题建模就会失效。如果你的词形还原器需要词性上下文，但你没有传入，动词就会被当作名词处理。

本节课从零构建三种预处理原语，然后展示 NLTK 和 spaCy 如何做同样的工作，让你看清其中的取舍。

## 概念

三种操作，各有职责，也各有失效模式。

**分词（Tokenization）** 把字符串拆成 token。"Token" 故意保持模糊，因为合适的粒度取决于任务：经典 NLP 用词级别，Transformer 用子词级别，没有空格的语言用字符级别。

**词干提取（Stemming）** 按规则砍掉后缀。快、激进、但笨拙。`running -> run`。`organization -> organ`。后者就是它的失效模式。

**词形还原（Lemmatization）** 利用语法知识把单词还原为词典形式。慢、准确，需要查表或形态分析器。`ran -> run`（需要知道 ran 是 run 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：速度优先且能容忍噪声时用词干提取（搜索索引、粗略分类）；语义重要时用词形还原（问答、语义搜索、任何用户会读到的文本）。

## 动手实现

### 第一步：基于正则表达式的分词器

最简单实用的分词器按非字母数字字符切分，同时把标点符号保留为独立 token。不完美，也不终极，但一行就能跑。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排列：带可选内部撇号的单词（如 `don't`、`it's`）、纯数字、任何非空白非字母数字的单个字符作为独立 token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失效模式。`3pm` 被拆成 `['3', 'pm']`，因为我们让字母连续段和数字连续段交替匹配。对大多数任务来说够用了。URL、邮箱、话题标签都会失效。生产环境中，应在通用模式之前加入专门模式。

### 第二步：Porter 词干提取器（仅 step 1a）

完整的 Porter 算法有五阶段规则。仅 step 1a 就覆盖了英语最常见的后缀，足以展示其模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

自上而下阅读规则。`ies -> i` 规则导致 `ponies -> poni`，而不是 `pony`。真正的 Porter 算法会用 step 1b 修复这个问题。规则之间会竞争，排在前面的规则优先，顺序比任何单条规则都重要。

### 第三步：基于查表的词形还原器

真正的词形还原需要形态学知识。一个适合教学的可行版本使用小型词元表和回退策略。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键的教学点。`watched` 不在表中，而且我们的回退只处理 `ing`。真正的词形还原会覆盖 `ed`、不规则动词、比较级形容词、以及发音变化的复数（如 `children -> child`）。这就是为什么生产系统使用 WordNet、spaCy 的形态分析器或完整形态分析器。

### 第四步：把它们串成管道

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺省的部分是词性标注器。Phase 5 · 07（词性标注）会构建一个。目前先把所有词默认标为 `NOUN`，并承认这一局限。

## 使用现成工具

NLTK 和 spaCy 都自带生产级实现。各自只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 能处理缩写、Unicode 以及你的正则漏掉的边界情况。`PorterStemmer` 会运行全部五个阶段。`WordNetLemmatizer` 需要把 NLTK 的 Penn Treebank 词性标记转换为 WordNet 的缩写集合。上面的转换代码是许多教程都会跳过的那部分接线工作。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整个流程隐藏在 `nlp(text)` 背后。分词、词性标注、词形还原一次性完成。规模化运行时比 NLTK 更快，开箱即用也更准确。代价是你不容易单独替换其中某个组件。

### 如何选择

| 场景 | 选择 |
|-----------|------|
| 教学、研究、需要更换组件 | NLTK |
| 生产、多语言、速度优先 | spaCy |
| Transformer 流程（反正你会用模型自带的分词器） | 使用 `tokenizers` / `transformers`，跳过传统预处理 |

### 没人提醒你的两种失效模式

大多数教程讲完算法就结束了。但真实的预处理管道会有两件事咬你一口，而且几乎没人讲。

**可复现性漂移。** NLTK 和 spaCy 在不同版本之间会改变分词和词形还原行为。spaCy 2.x 输出 `['do', "n't"]` 的句子，在 3.x 可能输出 `["don't"]`。你的模型是在一种分布上训练的，而推理现在运行在另一种分布上。准确率悄悄下降，却没人知道为什么。要在 `requirements.txt` 中固定库版本；写一份预处理回归测试，冻结 20 个样本句子的期望分词结果；每次升级都跑一遍。

**训练 / 推理不匹配。** 训练时用了激进预处理（小写、去停用词、词干提取），部署时却直接喂原始用户输入，性能会暴跌。这是生产 NLP 中最常见的单点故障。如果你在训练时做了预处理，推理时也必须运行完全相同的函数。把预处理作为函数封装在模型包里，而不是让 serving 团队重写的某个 notebook 单元格。

## 交付

一个可复用的提示词模板，帮助工程师在不读三本教科书的情况下选出预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为 NLP 任务推荐分词、词干提取与词形还原方案。
phase: 5
lesson: 01
---

你负责经典 NLP 预处理建议。给定任务描述后，输出：

1. 分词选择（正则、NLTK word_tokenize、spaCy 或 Transformer 分词器），并说明原因。
2. 是否使用词干提取、词形还原、两者都用或都不用，并说明原因。
3. 具体的库调用。说出函数名。如果涉及 NLTK，给出词性标注转换代码。
4. 一个用户应该测试的失效模式。

禁止为用户可见文本推荐词干提取。禁止在没有词性标注的情况下推荐词形还原。遇到非英语输入时，应标记为需要不同流程。
```

## 练习

1. **简单。** 扩展 `tokenize`，使 URL 保持为单个 token。测试：`tokenize("Visit https://example.com today.")` 应生成一个 URL token。
2. **中等。** 实现 Porter step 1b。如果单词包含元音且以 `ed` 或 `ing` 结尾，则删除该后缀。处理双写辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个词形还原器，以 WordNet 作为查表依据，WordNet 无条目时回退到你的 Porter 词干提取器。在标注语料上测量准确率，并与纯 WordNet 和纯 Porter 对比。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| Token | 一个词 | 模型消费的基本单位。可以是词、子词、字符或字节。 |
| Stem | 词根 | 基于规则去掉后缀后的结果。不一定是真实存在的单词。 |
| Lemma | 词典形式 | 你会去查词典的形式。需要语法上下文才能正确计算。 |
| POS tag | 词性 | 如 NOUN、VERB、ADJ 等类别。词形还原准确需要它。 |
| Morphology | 词形规则 | 单词如何根据时态、数、格改变形态。词形还原依赖于此。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) —— 原始论文，仅五页，至今仍是讲解最清晰的资料。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) —— 真实管道是如何连接的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) —— 你还没想到过的分词边界情况。
