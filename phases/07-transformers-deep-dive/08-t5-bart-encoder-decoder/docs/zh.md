# T5、BART —— 编码器-解码器模型

> 编码器负责理解，解码器负责生成。把它们组合起来，就得到了为“输入 → 输出”任务而生的模型：翻译、摘要、改写、转录。

**Type:** 学习  
**Languages:** Python  
**Prerequisites:** Phase 7 · 05（完整 Transformer）、Phase 7 · 06（BERT）、Phase 7 · 07（GPT）  
**Time:** 约 45 分钟

## 问题背景

仅解码器的 GPT 和仅编码器的 BERT 都是为了不同目标而对 2017 年原始架构做的裁剪。但很多任务天然就是输入到输出的形式：

- 翻译：英语 → 法语。
- 摘要：5,000 词元的文章 → 200 词元的摘要。
- 语音识别：音频词元 → 文本词元。
- 结构化抽取：散文 → JSON。

对于这些任务，编码器-解码器（encoder-decoder）结构是最自然的 fit。编码器为源序列生成稠密表示，解码器在每一步生成都通过交叉注意力（cross-attention）关注该表示。训练时在输出端做移位一位（shift-by-one），损失（loss）与 GPT 相同，只是以编码器输出为条件。

两篇论文定义了现代编码器-解码器范式的标准做法：

1. **T5**（Raffel et al. 2019）。《Text-to-Text Transfer Transformer》。所有 NLP 任务都被重新定义为“文本进、文本出”：单一架构、单一词表、单一损失。以掩码跨度预测（masked span prediction）做预训练：在输入中损坏若干跨度，让解码器输出这些跨度。
2. **BART**（Lewis et al. 2019）。《Bidirectional and Auto-Regressive Transformer》。去噪自编码器：用多种方式损坏输入（打乱、掩码、删除、旋转），让解码器重建原始文本。

到了 2026 年，编码器-解码器结构仍在以下场景发光：

- Whisper（语音 → 文本）。
- Google 的翻译技术栈。
- 一些具有明显“上下文 + 修改”结构的代码补全 / 修复模型。
- Flan-T5 及其变体，用于结构化推理任务。

聚光灯被仅解码器模型抢走了，但编码器-解码器从未退场。

## 核心概念

![编码器-解码器与交叉注意力](../assets/encoder-decoder.svg)

### 前向流程

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键在于：编码器对每个输入只运行一次；解码器以自回归（autoregressive）方式运行，但每一步都交叉关注到**同一个**编码器输出。缓存编码器输出对长输入来说是一个免费的加速。

### T5 预训练 —— 跨度损坏（span corruption）

从输入中随机选取若干跨度（平均长度 3 个词元，共占 15%）。每个跨度替换为一个唯一的哨兵标记（sentinel token）：`<extra_id_0>`、`<extra_id_1>` 等。解码器只输出带哨兵前缀的损坏跨度：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

这种方式比预测整个序列成本更低。T5 论文中的消融实验表明，它与 MLM（BERT）和 prefix-LM（UniLM）具有竞争力。

### BART 预训练 —— 多重噪声去噪

BART 尝试了五种噪声函数：

1. 词元掩码（token masking）。
2. 词元删除（token deletion）。
3. 文本填充（text infilling）：掩码一个跨度，解码器需插入正确长度。
4. 句子重排（sentence permutation）。
5. 文档旋转（document rotation）。

组合“文本填充 + 句子重排”在下游任务上取得了最佳效果。解码器始终重建完整原始序列，因此 BART 的预训练计算量高于 T5。

### 推理

与 GPT 一样采用自回归生成。贪心 / 束搜索（beam search）/ top-p 采样都适用。束搜索（宽度 4–5）是翻译和摘要的标准选择，因为这类任务的输出分布比对话更窄。

### 2026 年如何选择变体

| 任务 | 是否用编码器-解码器？ | 原因 |
|------|----------------------|------|
| 翻译 | 通常是 | 源序列明确；输出分布固定；束搜索效果好 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器负责处理音频特征 |
| 聊天 / 推理 | 否，用仅解码器 | 没有持久的“输入”——对话本身就是序列 |
| 代码补全 | 通常不用 | 长上下文下的仅解码器更占优；Qwen 2.5 Coder 等代码模型都是仅解码器 |
| 摘要 | 两者皆可 | BART、PEGASUS 曾击败早期的仅解码器基线；现代仅解码器大模型已能匹敌 |
| 结构化抽取 | 两者皆可 | T5 的“文本 → 文本”形式很干净，任何输出格式都能吸收 |

自 2022 年以来的趋势：仅解码器模型逐渐接管了编码器-解码器曾主导的任务，因为 (a) 经指令微调的仅解码器大模型可通过提示泛化到任意任务，(b) 单一架构更易扩展，(c) RLHF 默认面向解码器架构。编码器-解码器仍在输入模态不同（语音、图像）或束搜索质量至关重要的场景守住阵地。

## 动手实现

参见 `code/main.py`。我们为一个玩具语料库实现 T5 风格的跨度损坏——这是本课最有用的单一知识点，因为它在之后的每一个编码器-解码器预训练配方中都会出现。

### 第一步：跨度损坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式遵循 T5 约定：`<sent0> span0 <sent1> span1 ...`。被损坏的输入把未改变的词元与跨度位置上的哨兵标记交错排列。

### 第二步：验证可逆性

给定被损坏的输入和目标，重建原始句子。如果损坏是可逆的，前向过程就是良定义的。这是一个合理性检查——真实训练不会这样做，但测试成本很低，能帮你发现跨度索引中的差一错误。

### 第三步：BART 噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。把其中两个组合起来并展示结果。

## 实际使用

HuggingFace 参考实现：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的 trick：任务名称直接写在输入文本里。同一个模型可以处理几十种任务，因为每个任务都是“文本进、文本出”。到 2026 年，这一模式已被指令微调的仅解码器模型泛化，但 T5 最先将其系统化。

## 落地应用

参见 `outputs/skill-seq2seq-picker.md`。该技能会根据输入-输出结构、延迟和质量目标，为新任务选择编码器-解码器还是仅解码器。

## 练习题

1. **简单。** 运行 `code/main.py`，对一句 30 词元的句子应用跨度损坏，验证把源中的非哨兵词元与解码出的目标跨度拼接后能还原原文。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` 词元替换随机跨度，解码器必须推断出正确的跨度长度和内容。展示一个示例。
3. **困难。** 在一个 200 对的小规模英语 → 儿童黑话（pig-Latin）语料上微调 `flan-t5-small`，在 50 对的留出测试集上测 BLEU。与在相同数据、相同算力下微调 `Llama-3.2-1B` 做对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 编码器-解码器（encoder-decoder） | “Seq2seq transformer” | 两个堆叠：双向编码器处理输入，带交叉注意力的因果解码器生成输出。 |
| 交叉注意力（cross-attention） | “源端与目标端对话的地方” | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一通道。 |
| 跨度损坏（span corruption） | “T5 的预训练 trick” | 用哨兵标记替换随机跨度；解码器输出这些跨度。 |
| 去噪目标（denoising objective） | “BART 的游戏” | 对输入施加某种噪声函数，训练解码器重建干净序列。 |
| 哨兵标记（sentinel token） | “`<extra_id_N>` 占位符” | 在源端标记损坏跨度、在目标端重新标记它们的特殊词元。 |
| Flan | “指令微调的 T5” | 在 1,800+ 任务上微调的 T5；让编码器-解码器在指令跟随方面具备了竞争力。 |
| 束搜索（beam search） | “解码策略” | 每步保留 top-k 条部分序列；翻译/摘要的标准做法。 |
| 教师强制（teacher forcing） | “训练时的输入” | 训练时把真实的上一个输出词元喂给解码器，而不是采样得到的词元。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) —— T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) —— BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) —— Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) —— Whisper，2026 年最典型的编码器-解码器模型。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) —— 参考实现。
