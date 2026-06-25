# 命名实体识别

> 把名字抽出来。听起来简单，直到你遇到歧义边界、嵌套实体和领域术语。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 02（词袋 + TF-IDF）、Phase 5 · 03（词嵌入）
**时间：** ~75 分钟

## 问题

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（也许算）、US（GPE）。一个好的 NER 系统会把它们全部抽取出来并赋予正确类型；差的系统会漏掉 iPhone，把 Apple 和水果搞混，还把 "US" 标成 PERSON。

NER 是每个结构化抽取流水线的幕后主力。简历解析、合规日志扫描、病历匿名化、搜索查询理解、聊天机器人回复的 grounding、法律合同抽取——你几乎看不到它，却无时无刻不依赖它。

本课沿着经典路径（基于规则、HMM、CRF）走向现代方法（BiLSTM-CRF，再到 Transformer）。每一步都在解决前一步的特定局限。这个演进模式就是本课的核心。

## 概念

**BIO 标注**（或 BILOU）把实体抽取转化为序列标注问题：为每个词符标注 `B-TYPE`（实体开始）、`I-TYPE`（实体内部）或 `O`（不属于任何实体）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多词符实体连成链：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽取任意跨度的实体。

架构演进：

- **基于规则。** 正则 + 地名词典查找。对已知实体精度高，对新实体零覆盖。
- **HMM。** 隐马尔可夫模型。词符给定标签的发射概率、标签到标签的转移概率，用 Viterbi 解码。基于标注数据训练。
- **CRF。** 条件随机场。与 HMM 类似，但是判别式模型，因此可以混合任意特征（词形、大小写、相邻词）。在 2026 年仍是低资源部署的经典生产主力。
- **BiLSTM-CRF。** 用神经网络特征替代手工特征。LSTM 双向读取句子，上层接 CRF 保证标签序列一致。
- **基于 Transformer。** 用 BERT 等模型加 token 分类头微调。精度最高，算力也最大。

## 动手实现

### 步骤 1：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤 2：手工特征

对于经典（非神经网络）NER，特征决定一切。以下是常用特征：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`，`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对专有名词具有很强的区分信号。

### 步骤 3：简单的基于规则 + 词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级地名词典可能有数百万条目，从 Wikipedia 和 DBpedia 抓取而来。覆盖率很高，但消歧能力（`Apple` 公司 vs 水果）很差。这正是统计模型最终获胜的原因。

### 步骤 4：CRF 步骤（概要，非完整实现）

50 行内从零实现完整 CRF，没有概率论基础并不直观。改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 分别是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学会非法序列（例如 `I-ORG` 出现在 `O` 之后）的概率很低，这正是 CRF 无需手动编写约束即可保证 BIO 一致性的方式。

### 步骤 5：BiLSTM-CRF 增加了什么

特征变成可学习。输入是词符嵌入（GloVe 或 fastText）。LSTM 从左到右、从右到左读取句子，拼接后的隐状态经过 CRF 输出层。CRF 仍然强制标签序列一致；LSTM 则用可学习特征替代了手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于 CRF 层，使用 `torchcrf.CRF`（通过 `pip install pytorch-crf` 安装）。相比手工特征 CRF 的提升是可测量的，但除非你有数万条标注句子，否则提升通常比你想象的要小。

## 使用

spaCy 开箱即提供生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标为 `ORG` 而非 `PRODUCT`——spaCy 的小模型对产品实体覆盖较弱。大型模型（`en_core_web_lg`）表现更好，Transformer 模型（`en_core_web_trf`）更好。

使用 Hugging Face 进行基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的 B-X、I-X 词符合并为一个跨度。不设该参数会得到词符级标签，需要自行合并。

### 基于 LLM 的 NER（2026 年的选择）

零样本和少样本 LLM NER 目前在许多领域已与微调模型相当，在标注数据稀缺时优势尤为明显。

- **零样本提示。** 给 LLM 一个实体类型列表和示例模式，要求输出 JSON。开箱即用，但在新领域上精度中等。
- **ZeroTuneBio 风格提示。** 把任务拆成候选抽取 → 语义解释 → 判断 → 复核。多阶段提示（而非一次性提示）能显著提升生物医学 NER 的精度，法律、金融、科学领域同理。
- **基于 RAG 的动态提示。** 每次推理时从少量标注种子集中检索最相似的样例，动态构建少样本提示。2026 年的基准测试显示，这比静态提示能把 GPT-4 在生物医学 NER 上的 F1 提高 11–12%。
- **按实体类型分解。** 对长文档，单次调用同时抽取所有实体类型会随着长度增加而召回下降。为每种实体类型单独跑一轮抽取：推理成本更高，但精度大幅提升。这是临床病历和法律合同中的标准做法。

2026 年的生产建议：在收集训练数据之前，先建立 LLM 零样本基线。很多时候它的 F1 已经足够好，根本无需微调。

### 经典 NER 仍然占优的场景

即便有 LLM，经典 NER 在以下场景仍然胜出：

- 延迟预算低于 50ms。
- 拥有数千条标注样本且需要 98%+ F1。
- 领域具有稳定的本体，预训练 CRF 或 BiLSTM 能很好迁移。
- 监管要求本地部署、非生成式模型。

### 它会失效的地方

- **领域迁移。** 在法律合同上直接跑 CoNLL 训练的 NER，可能还不如一个地名词典。必须在目标领域微调。
- **嵌套实体。** "Bank of America Tower" 同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠跨度，需要嵌套 NER（多轮或基于跨度的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation"。基于词符的模型有时会把它切分。使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医学 NER 中的 DRUG_BRAND、ADVERSE_EVENT、DOSE 等标签，通用模型完全不了解。那里应从 SciSpaCy 和 BioBERT 开始。

## 交付

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: 为给定的抽取任务选择合适的 NER 方法。
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

给定任务描述（领域、标签集、语言、延迟、数据量），输出：

1. 方法。基于规则 + 地名词典、CRF、BiLSTM-CRF，还是 Transformer 微调。
2. 起始模型。给出具体名称（spaCy 模型 ID、Hugging Face checkpoint ID，或 "custom, trained from scratch"）。
3. 标注策略。BIO、BILOU 还是基于跨度。用一句话说明理由。
4. 评估。使用 `seqeval`，始终报告实体级 F1（而非词符级 F1）。

如果标注样本少于 500 条且用户没有预训练领域模型，拒绝推荐 Transformer 微调。若存在嵌套实体，标记为需要基于跨度或多轮模型。如果用户提到 "production scale" 且标签仍与 CoNLL-2003 一致，要求进行地名词典审计。
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在 10 句话上验证往返一致性。
2. **中等。** 用上面的 sklearn-crfsuite CRF 在 CoNLL-2003 英文 NER 数据集上训练，用 `seqeval` 报告每类实体 F1。典型结果：~84 F1。
3. **困难。** 在领域专属 NER 数据集（医学、法律或金融）上微调 `distilbert-base-cased`，并与 spaCy 小模型对比。记录数据泄漏检查，并写下最让你意外的发现。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| NER | 抽取名字 | 为词符跨度标注类型（PERSON、ORG、GPE、DATE 等）。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 继续，`O` 在外部。 |
| BILOU | 更好的 BIO | 增加 `L-X`（结尾）和 `U-X`（单个词符），边界更清晰。 |
| CRF | 结构化分类器 | 建模标签之间的转移，而不仅是发射概率，强制生成合法序列。 |
| 嵌套 NER | 重叠实体 | 一个跨度与另一个跨度的子跨度属于不同实体类型，BIO 无法表达。 |
| 实体级 F1 | 正确的 NER 指标 | 预测跨度必须与真实跨度完全匹配；词符级 F1 会高估准确率。 |

## 延伸阅读

- [Lample et al. (2016). 命名实体识别的神经架构](https://arxiv.org/abs/1603.01360) —— BiLSTM-CRF 论文，经典之作。
- [Devlin et al. (2018). BERT：深度双向 Transformer 的预训练](https://arxiv.org/abs/1810.04805) —— 提出后来成为标准的 token 分类范式。
- [spaCy 语言特征 —— 命名实体](https://spacy.io/usage/linguistic-features#named-entities) —— `Doc.ents` 和 `Span` 所有属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) —— 正确的指标库，永远用它。
