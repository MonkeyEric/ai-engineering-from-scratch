# 用于文本的 CNN 与 RNN

> 卷积学习 n-gram。循环网络记忆。两者都被注意力机制取代。但在资源受限的硬件上，两者仍然重要。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 3 · 11（PyTorch 入门）、阶段 5 · 03（词嵌入）、阶段 4 · 02（从零实现卷积）
**时间：** 约 75 分钟

## 问题

TF-IDF 和 Word2Vec 生成的是忽略词序的扁平向量。基于它们构建的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时承载着关键信号。

在 Transformer 出现之前，两类架构填补了这一空白。

**用于文本的卷积神经网络（TextCNN）。** 对词嵌入序列应用一维卷积。宽度为 3 的滤波器是一个可学习的 trigram 检测器：它覆盖三个词并输出一个分数。堆叠不同宽度（2、3、4、5）以检测多尺度模式。通过最大池化得到固定大小的表示。扁平、并行、快速。

**循环神经网络（RNN、LSTM、GRU）。** 逐个处理 token，维护一个向前传递信息的隐藏状态。顺序处理、带有记忆、可处理可变长度输入。从 2014 年到 2017 年主导了序列建模，然后注意力机制出现了。

本节课将构建这两类模型，并指出促使注意力机制诞生的问题。

## 概念

**TextCNN**（Kim，2014）。首先对 token 进行嵌入。宽度为 `k` 的一维卷积在连续的 `k`-gram 嵌入上滑动一个滤波器，生成特征图。对该特征图进行全局最大池化，挑选最强激活。将多个滤波器宽度的池化输出拼接起来，送入分类头。

为什么有效。一个滤波器就是一个可学习的 n-gram。最大池化具有位置不变性，因此 "not good" 无论是在评论开头还是中间都会触发相同的特征。三个滤波器宽度，每个宽度 100 个滤波器，就得到 300 个学习得到的 n-gram 检测器。训练是并行的，没有时间步之间的顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 在不同时间步之间共享。时间步 `T` 的隐藏状态是整个前缀的摘要。对于分类任务，可以对 `h_1 ... h_T` 进行池化（最大池化、平均池化或取最后一个状态）。

普通 RNN 存在梯度消失问题。**LSTM** 增加了门控来决定遗忘什么、存储什么、输出什么，从而在长序列上稳定梯度。**GRU** 将 LSTM 简化为两个门，参数量更少但性能相近。

**双向 RNN** 同时运行一个前向和一个后向 RNN，并拼接它们的隐藏状态。每个 token 的表示都能看到其左右两侧的上下文。对于标注任务至关重要。

## 构建

### 步骤 1：PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间维度视为通道。池化后的输出是固定大小的，与输入长度无关。

### 步骤 2：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列做最大池化，而不是取最后一个状态。对于分类任务，最大池化通常优于取最后一个隐藏状态，因为在长序列末端的信息往往会主导最后一个状态。

### 步骤 3：梯度消失演示（直观理解）

没有门控的普通 RNN 无法学习长期依赖。考虑一个玩具任务：预测 token `A` 是否出现在序列中的任何位置。如果 `A` 位于位置 1，而序列长度为 100，那么损失产生的梯度必须反向流经循环权重 99 次。如果权重小于 1，梯度就会消失；如果大于 1，梯度就会爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 当权重为 0.9，步长为 100 时：
#   0.9 ^ 100 ≈ 2.7e-5
# 从第 100 步到第 1 步的梯度实际上为零。
```

LSTM 通过**细胞状态**解决了这个问题：细胞状态以加法交互的方式贯穿网络（遗忘门会对它做乘法缩放，但梯度仍能沿着这条“高速公路”流动）。GRU 以更少的参数实现了类似效果。两者都能让你在 100+ 步的序列上稳定训练。

### 步骤 4：为什么这仍然不够

即使有了 LSTM，仍然存在三个问题。

1. **顺序瓶颈。** 在长度为 1000 的序列上训练 RNN 需要 1000 个串行的前向/反向步骤，无法在时间维度上并行。
2. **编码器-解码器结构中的固定大小上下文向量。** 解码器只能看到编码器的最终隐藏状态，这个状态是整个输入的压缩。长输入会丢失细节。第 09 课会专门讲这个问题。
3. **远距离依赖的精度上限。** LSTM 优于普通 RNN，但在 200+ 步的距离上传输特定信息仍然困难。

注意力机制解决了这三个问题。Transformer 完全去除了循环。第 10 课是转折点。

## 应用

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都已达到生产可用级别。训练代码是标准的。

Hugging Face 提供了预训练嵌入，可以直接作为输入层接入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

在约束合适时使用它们的对照清单。

- **边缘 / 端侧推理。** 使用 GloVe 嵌入的 TextCNN 比 Transformer 小 10-100 倍。如果你的部署目标是手机，这就是合适的方案。
- **流式 / 在线分类。** RNN 每次处理一个 token；Transformer 需要完整序列。对于实时到来的文本，LSTM 仍然占优。
- **极小基线模型。** 在新任务上快速迭代。用 CPU 在 5 分钟内训练一个 TextCNN。
- **数据有限的序列标注。** BiLSTM-CRF（第 06 课）对于有 1k-10k 标注句子的 NER 任务来说，仍然是生产级架构。

其他情况都交给 Transformer。

## 交付

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: 给定一组约束条件，选择合适的文本编码器架构。
phase: 5
lesson: 08
---

给定约束条件（任务、数据量、延迟预算、部署目标、计算预算），输出：

1. 编码器架构：TextCNN、BiLSTM、BiLSTM-CRF、Transformer 微调，或“使用预训练 Transformer 作为冻结编码器 + 小型分类头”。
2. 嵌入输入：随机初始化、冻结的 GloVe / fastText，或上下文化 Transformer 嵌入。
3. 5 行训练方案：优化器、学习率、批量大小、训练轮数、正则化。
4. 一个监控信号。对于 RNN/CNN 模型：缺少注意力机制意味着它们会遗漏长距离依赖；检查按长度划分的准确率。对于 Transformer：学习率过高会导致微调崩溃；检查训练损失。

当标注数据少于约 500 条时，拒绝推荐微调 Transformer，除非先证明 TextCNN / BiLSTM 基线已经陷入平台期。将边缘部署标记为需要优先考虑架构。
```

## 练习

1. **简单。** 在一个三类玩具数据集上训练 TextCNN（你可以自己构造数据）。验证滤波器宽度 (2, 3, 4) 的平均 F1 是否优于单一宽度 (3)。
2. **中等。** 为 LSTM 分类器实现最大池化、平均池化和最后一个状态池化。在一个小数据集上比较结果；记录哪种池化效果最好，并推测原因。
3. **困难。** 构建一个 BiLSTM-CRF NER 标注器（结合第 06 课与本课内容）。在 CoNLL-2003 上训练。与第 06 课仅 CRF 的基线以及 BERT 微调进行比较。报告训练时间、内存占用和 F1。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| TextCNN | 用于文本的 CNN | 对词嵌入做一维卷积堆叠，再经全局最大池化。Kim（2014）。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控 RNN | 增加输入/遗忘/输出门和一个细胞状态。能在长序列上稳定训练。 |
| GRU | 更简化的 LSTM | 两个门而非三个。准确率相近，参数更少。 |
| Bidirectional | 双向 | 前向与后向 RNN 拼接。每个 token 都能看到其上下文的两侧。 |
| Vanishing gradient | 训练信号消失 | 普通 RNN 中反复乘以小于 1 的权重，使早期时间步的梯度实际上变为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) —— TextCNN 论文。八页。易读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) —— LSTM 论文。出奇地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) —— 让 LSTM 变得人人可懂的图解。
