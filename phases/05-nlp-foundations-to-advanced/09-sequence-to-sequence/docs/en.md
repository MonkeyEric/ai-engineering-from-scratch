# 序列到序列模型

> 两个 RNN 假装成翻译器。它们遇到的瓶颈正是注意力机制存在的原因。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 08（用于文本的 CNN + RNN），Phase 3 · 11（PyTorch 入门）
**时间：** ~75 分钟

## 问题

分类任务将变长序列映射为单个标签。翻译任务则将变长序列映射为另一个变长序列。输入和输出属于不同的词表，可能是不同的语言，长度也不一定相同。

seq2seq 架构（Sutskever、Vinyals、Le，2014）用一个刻意简单的方案解决了这个问题：两个 RNN。一个读取源句并生成固定大小的上下文向量；另一个读取该向量并逐 token 生成目标句。本质上就是你为第 08 课写的代码，只是换了一种拼接方式。

研究它有两个原因。第一，上下文向量瓶颈是 NLP 中最具教学意义的失败案例，它解释了注意力机制和 Transformer 擅长做的一切。第二，训练技巧（教师强制、计划采样、推理时的束搜索）仍然适用于包括大语言模型在内的所有现代生成系统。

## 概念

**编码器。** 读取源句的 RNN。它的最终隐藏状态就是**上下文向量**——对整个输入的固定大小摘要。理论上要“不丢失任何信息”。

**解码器。** 另一个从上下文向量初始化的 RNN。每一步它以上一步生成的 token 作为输入，输出在目标词表上的分布。通过采样或取 argmax 选择下一个 token，再把它反馈回去。重复直到生成 `<EOS>` token 或达到最大长度。

**训练：** 解码器每一步都计算交叉熵损失，并在序列上求和。两个网络都通过标准的随时间反向传播进行训练。

**教师强制。** 训练时，解码器在步骤 `t` 的输入是位置 `t-1` 的*真实* token，而不是解码器自己上一步的预测。这能稳定训练；否则早期错误会逐级放大，模型根本学不到东西。推理时只能用模型自己的预测，因此训练与推理的分布永远存在差异，这种差异称为**暴露偏差**。

**瓶颈。** 编码器学到的关于源句的所有信息都必须被压缩进那一个上下文向量。长句会丢失细节，罕见词会变得模糊，语序调整（chat noir vs. black cat）只能死记硬背，而不是被真正计算出来。

注意力机制（第 10 课）通过让解码器查看*每一个*编码器隐藏状态，而不只是最后一个，直接解决了这个问题。这就是它的全部价值所在。

## 动手实现

### 第一步：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状是 `[batch, seq_len, hidden_dim]`，即每个输入位置对应一个隐藏状态。`hidden` 的形状是 `[1, batch, hidden_dim]`，代表最后一步。第 08 课说“对 outputs 做池化用于分类”。这里我们把最后一个隐藏状态作为上下文向量，并忽略每步的输出。

### 第二步：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次调用一步。输入是一批单 token 和当前隐藏状态；输出是下一个 token 的词表 logits 和更新后的隐藏状态。

### 第三步：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有两个值得指出的参数。`ignore_index=0` 会跳过填充 token 的损失。`teacher_forcing_ratio` 是在每一步使用真实 token 还是模型预测的概率。初始可设为 1.0（完全教师强制），并在训练过程中退火到约 0.5，以缩小暴露偏差差距。

### 第四步：推理循环（贪心解码）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码每一步都选择概率最高的 token。它可能一头栽错：一旦选定某个 token 就无法收回。**束搜索**会保留前 `k` 个部分序列，最后选择得分最高的完整序列。束宽 3-5 是常见设置。

### 第五步：展示瓶颈

在玩具复制任务上训练模型：源序列 `[a, b, c, d, e]`，目标序列 `[a, b, c, d, e]`。逐渐增加序列长度，观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损记住 40 个 token 的输入。信息其实存在于编码器的每一步，但解码器只能看到最后一个状态。注意力机制直接解决了这一点。

## 使用现成工具

PyTorch 提供了 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库则提供了完整的编码器-解码器模型（BART、T5、mBART、NLLB），它们都在数十亿 token 上训练过。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器模型用 Transformer 替换了 RNN。高层结构（编码器、解码器、逐 token 生成）与 2014 年的 seq2seq 论文完全一致，只是每个块内部的机制不同。

### 什么时候仍然选择基于 RNN 的 seq2seq

在新项目中几乎从不。具体例外：

- 流式翻译：每次只消费一个 token，且内存有界。
- 端侧文本生成：Transformer 的内存成本过高。
- 教学用途。理解编码器-解码器瓶颈，是理解 Transformer 为何获胜的最快路径。

### 暴露偏差及其缓解方法

- **计划采样。** 在训练过程中逐渐降低教师强制比例，让模型学会从自己的错误中恢复。
- **最小风险训练。** 以句子级 BLEU 分数而不是 token 级交叉熵作为训练目标，更接近真实需求。
- **强化学习微调。** 用某个指标奖励序列生成器。现代 LLM 的 RLHF 就在做这件事。

这三种方法同样适用于基于 Transformer 的生成系统。

## 交付成果

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: 为给定任务设计一个序列到序列流水线。
phase: 5
lesson: 09
---

给定一个任务（翻译、摘要、改写、问题重写），输出：

1. 架构。默认使用预训练的 Transformer 编码器-解码器（BART、T5、mBART、NLLB）。仅在特定约束下才使用基于 RNN 的 seq2seq。
2. 起始检查点。明确写出名称（`facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`），并让检查点与任务及语言覆盖范围匹配。
3. 解码策略。需要确定性输出时用贪心解码；追求质量时用束搜索（宽度 4-5）；需要多样性时用带温度的采样。每种策略用一句话说明理由。
4. 发布前需要验证的一种失败模式。暴露偏差在较长输出上会表现为生成漂移；抽取 20 条位于 90 百分位长度的输出进行人工检查。

如果平行语料不足一百万对，拒绝推荐从头训练 seq2seq。任何面向用户的流水线若使用贪心解码，都要标记为脆弱（贪心容易产生重复和循环）。
```

## 练习

1. **简单。** 实现玩具复制任务。在输入输出相等的样本上训练一个 GRU seq2seq，测量长度为 5、10、20 时的准确率，复现瓶颈现象。
2. **中等。** 添加束宽为 3 的束搜索解码。在一个小型平行语料上与贪心解码对比 BLEU，记录束搜索在哪些地方有效（通常是最后几个 token）以及在哪些地方没有提升。
3. **困难。** 在 1 万对样本的改写数据集上微调 `facebook/bart-base`。对比微调模型与基础模型在留出数据上的束宽 4 输出。报告 BLEU，并挑选 10 个定性示例。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Encoder | 输入 RNN | 读取源句。输出每步隐藏状态和一个最终的上下文向量。 |
| Decoder | 输出 RNN | 从上下文向量初始化。逐 token 生成目标序列。 |
| Context vector | 摘要 | 编码器的最终隐藏状态。固定大小。注意力机制要解决的瓶颈。 |
| Teacher forcing | 用真实 token | 训练时输入前一步的真实 token。稳定学习过程。 |
| Exposure bias | 训练/测试差距 | 模型在真实 token 上训练，从未练习过从自己错误中恢复。 |
| Beam search | 更好的解码 | 每步保留前 k 个部分序列，而不是贪心立即确定。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) —— 原始 seq2seq 论文，仅四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) —— 提出 GRU 与编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 注意力论文。本课之后应立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) —— 可实际搭建的 seq2seq + 注意力代码。
