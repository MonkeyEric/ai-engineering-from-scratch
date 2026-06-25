# BERT —— 掩码语言建模

> GPT 预测下一个词，BERT 预测缺失的词。一句话的差别，却造就了此后半个时代所有与嵌入相关的一切。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer）、Phase 5 · 02（文本表示）
**时长：** 约 45 分钟

## 问题背景

2018 年，每项自然语言处理任务——情感分析、命名实体识别、问答、蕴含推理——都要从头在各自的标注数据上训练独立模型。没有预训练好的“懂英语”的检查点可供微调。ELMo（2018）证明可以用双向 LSTM 预训练上下文嵌入；它有所帮助，但泛化能力有限。

BERT（Devlin 等，2018）提出：如果我们拿一个 Transformer 编码器，用互联网上的每一句话训练它，并强制它根据左右上下文预测缺失的词，会怎样？之后只需为下游任务加一个微调头。参数效率上的突破令人惊叹。

结果是：18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）霸榜了所有 NLP 排行榜。到 2020 年，地球上的每个搜索引擎、内容审核管道和语义搜索系统内部都运行着 BERT。

到了 2026 年，仅编码器模型仍然是分类、检索和结构化抽取的最佳选择——每 token 速度比解码器快 5–10 倍，其嵌入也是所有现代检索栈的骨干。ModernBERT（2024 年 12 月）将架构推进到 8K 上下文，结合 Flash Attention、RoPE 与 GeGLU。

## 核心概念

![掩码语言建模：选取 token，将其掩码，预测原始 token](../assets/bert-mlm.svg)

### 训练信号

以句子 `the quick brown fox jumps over the lazy dog` 为例。

随机掩码 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型预测掩码位置上的原始 token。由于编码器是双向的，预测位置 1 的 `[MASK]` 时可以利用位置 2+ 的 `brown fox jumps`。这正是 GPT 无法做到的。

### BERT 掩码规则

在被选中的 15% token 中：

- 80% 替换为 `[MASK]`。
- 10% 替换为随机 token。
- 10% 保持不变。

为什么不总是用 `[MASK]`？因为 `[MASK]` 在推理时不会出现。如果在 100% 的掩码位置都让模型看到 `[MASK]`，会在预训练与微调之间产生分布偏移。10% 随机替换 + 10% 保持不变让模型保持“诚实”。

### 下一句预测（Next Sentence Prediction，NSP）—— 以及为何被弃用

原始 BERT 还训练了 NSP 任务：给定句子 A 和 B，判断 B 是否紧跟 A。RoBERTa（2019）通过消融实验证明 NSP 不仅无益，反而有害。现代编码器均已跳过该任务。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的基础组件重构了编码器块：

| 组件 | 原始 BERT（2018） | ModernBERT（2024） |
|-----------|----------------------|-------------------|
| 位置编码 | 可学习绝对位置编码 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | 前置归一化 RMSNorm |
| 注意力 | 全稠密注意力 | 局部（128）与全局交替 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与 2018 年的架构不同，ModernBERT 原生支持 Flash Attention。在 8K 序列长度下，推理速度比 DeBERTa-v3 快 2–3 倍，GLUE 分数也更高。

### 2026 年仍选择编码器的应用场景

| 任务 | 编码器优于解码器的原因 |
|------|---------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每个 token 的嵌入质量更高 |
| 分类（情感、意图、毒性） | 一次前向传播，无生成开销 |
| 命名实体识别 / token 标注 | 逐位置输出，天然双向 |
| 零样本蕴含推理（NLI） | 在编码器上加分类头 |
| RAG 重排序器 | 交叉编码器打分，比 LLM 重排序器快 10 倍 |

## 动手实现

### 第一步：掩码逻辑

参见 `code/main.py`。函数 `create_mlm_batch` 接收 token ID 列表、词表大小和掩码概率，返回应用掩码后的输入 ID 与标签（仅在掩码位置有值，其余为 -100，遵循 PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 保持原始 token 不变
    return input_ids, labels
```

### 第二步：在极小语料上运行 MLM 预测

用 200 个句子、20 个词的词表训练一个 2 层编码器 + MLM 头。不计算梯度，仅做前向传播的合理性检查。完整训练需要 PyTorch。

### 第三步：对比掩码类型

展示三元掩码规则如何让模型在没有 `[MASK]` 时仍然可用。分别在未掩码句子和已掩码句子上预测，两者都应产生合理的 token 分布，因为训练时模型见过这两种模式。

### 第四步：微调任务头

将 MLM 头替换为玩具情感数据集上的分类头。只训练头，编码器冻结。这是所有 BERT 应用的标准范式。

## 实际使用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型就是微调后的 BERT。** `sentence-transformers` 中的 `all-MiniLM-L6-v2` 等模型就是使用对比损失训练的 BERT。编码器相同，改变的是损失函数。

**交叉编码器重排序器也是微调后的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上做配对分类。查询与文档之间的双向注意力正是交叉编码器质量优于双编码器的关键。

**2026 年不应选择 BERT 的场景。** 任何生成任务。编码器无法以自回归方式合理生成 token。另外，在不到 10 亿参数的模型规模下，小型解码器可能以更灵活的架构达到相当质量（如 Phi-3-Mini、Qwen2-1.5B）。

## 交付

参见 `outputs/skill-bert-finetuner.md`。该技能文档定义了针对新分类或抽取任务进行 BERT 微调的范围，包括骨干选择、头设计、数据、评估与早停策略。

## 练习

1. **简单。** 运行 `code/main.py`，打印 10,000 个 token 上的掩码分布。确认约 15% 被选中，其中约 80% 变为 `[MASK]`。
2. **中等。** 实现整词掩码：如果一个词被切分成多个子词，则要么一起掩码，要么都不掩码。在 500 句语料上测量这是否能提升 MLM 准确率。
3. **困难。** 在公开数据集的 10,000 个句子上训练一个微型（2 层，d=64）BERT。用 `[CLS]` token 微调 SST-2 情感分类。与参数量相当的仅解码器基线对比，哪个更好？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| MLM | “掩码语言建模（masked language modeling）” | 训练信号：随机将 15% 的 token 替换为 `[MASK]`，预测原始 token。 |
| 双向（Bidirectional） | “两边都看” | 编码器注意力没有因果掩码——每个位置都能看到其他所有位置。 |
| `[CLS]` | “汇聚 token（pooler token）” | 每个序列前添加的特殊 token；其最终嵌入用作句子级表示。 |
| `[SEP]` | “分段分隔符（segment separator）” | 分隔成对序列（如 query/doc、句子 A/B）。 |
| NSP | “下一句预测（next sentence prediction）” | BERT 的第二项预训练任务；RoBERTa 证明其无效，2019 年后被弃用。 |
| 微调（Fine-tuning） | “适配到具体任务” | 编码器基本冻结，在其顶部训练一个小型任务头。 |
| 交叉编码器（Cross-encoder） | “重排序器（reranker）” | 将 query 和 doc 同时输入 BERT，输出相关性分数。 |
| ModernBERT | “2024 年升级版” | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K 上下文重构的编码器。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) —— 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) —— 如何正确训练 BERT；推翻 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) —— 替换 token 检测在同等计算量下优于 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) —— ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) —— 标准编码器实现参考。
