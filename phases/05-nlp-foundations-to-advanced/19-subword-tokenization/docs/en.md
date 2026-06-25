# 子词分词 —— BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到未登录词会束手无策，字符级分词器会让序列长度爆炸。子词分词器折中处理：每个现代大语言模型都离不开它。

**类型：** 学习
**语言：** Python
**前置知识：** 第 5 阶段 · 01（文本处理）、第 5 阶段 · 04（GloVe / FastText / 子词）
**时长：** 约 60 分钟

## 问题所在

你的词表有 50,000 个词。用户输入了 "untokenizable"。分词器返回 `[UNK]`。模型对这个词完全拿不到信号。更糟的是：语料中 90 分位数的文档包含 40 个生僻词，也就是说每篇文档会丢掉 40 处信息。

子词分词解决了这个问题。常见词保持为单个词元；生僻词拆解成有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据因此能覆盖所有内容，因为任何字符串最终都是字节序列。

2026 年的所有前沿大模型都采用这三种算法之一（BPE、Unigram、WordPiece），封装在三个库之一（tiktoken、SentencePiece、HF Tokenizers）中。没有分词器的选择，就无法交付语言模型。

## 核心概念

![BPE、Unigram、WordPiece 的逐字符对比](../assets/subword-tokenization.svg)

**BPE（字节对编码，Byte-Pair Encoding）。** 从字符级词表开始。统计每一对相邻符号。将出现频率最高的一对合并成一个新词元。重复此过程直到达到目标词表大小。主流算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral 均使用。

**字节级 BPE。** 同样的算法，但直接在原始字节（256 个基础词元）而非 Unicode 字符上运行。保证零 `[UNK]` —— 任何字节序列都能被编码。GPT-2 使用 50,257 个词元（256 字节 + 50,000 次合并 + 1 个特殊词元）。

**Unigram。** 从一个非常大的候选词表开始。为每个词元赋予一个一元概率。迭代地剔除那些移除后最小幅增加语料库对数似然的词元。推理时具有概率性：可以采样多种分词结果（通过子词正则化做数据增强很有用）。T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并能最大化训练语料库似然的符号对，而非原始频率。BERT、DistilBERT、ELECTRA 使用。

**SentencePiece 与 tiktoken。** SentencePiece 是直接在原始 Unicode 文本上*训练*词表（BPE 或 Unigram）的库，将空格编码为 `▁`。tiktoken 是 OpenAI 针对预建词表的快速*编码器*；它不能训练。

经验法则：

- **训练新词表：** SentencePiece（多语言、无需预分词）或 HF Tokenizers。
- **针对 GPT 词表做快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者都要：** HF Tokenizers —— 一个库同时完成训练与部署。

## 动手实现

### 步骤 1：从零实现 BPE

见 `code/main.py`。循环逻辑如下：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法体现了三个关键点。`</w>` 标记词尾，因此 "low"（作为后缀）和 "lower"（作为前缀）保持区分。按频率加权使高频对优先合并。合并列表是有序的 —— 推理时按训练顺序依次应用。

### 步骤 2：用学到的合并规则编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现复杂度为 O(n·|merges|)。生产级实现（tiktoken、HF Tokenizers）使用合并优先级查找和优先队列，运行时间接近线性。

### 步骤 3：实践中的 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # 对 CJK 可更低（英语 0.9995，日语 0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格被编码为 `▁`，`character_coverage` 控制生僻字符被保留还是映射到 `<unk>` 的激进程度。

### 步骤 4：用于 OpenAI 兼容词表的 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码，不训练。速度极快（Rust 后端）。与 GPT-4/5 的分词结果完全一致，可用于字节计数、成本估算、上下文窗口预算。

## 2026 年仍会出现的陷阱

- **分词器漂移。** 用词表 A 训练，却用词表 B 部署。词元 ID 不同，模型输出会乱套。在 CI 中检查 `tokenizer.json` 的哈希值。
- **空格歧义。** BPE 对 "hello" 和 " hello" 会产生不同词元。务必显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足。** 英语偏重语料训练出的词表会把非拉丁文字拆成 5-10 倍的词元。同样提示在 GPT-3.5 上，日语/阿拉伯语的成本要高出 5-10 倍。o200k_base 已部分改善。
- **表情符号拆分。** 单个表情符号可能占 5 个词元。预算上下文时务必检查表情符号处理。

## 如何使用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 从头训练单语模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供 OpenAI 兼容 API | tiktoken（GPT-4+ 用 `o200k_base`） |
| 领域专用词表（代码、数学、蛋白质） | 在领域语料上训练自定义 BPE，再与基础词表合并 |
| 边缘推理、小模型 | Unigram（较小词表效果更好） |

词表大小是缩放决策，不是固定常数。粗略经验：<1B 参数用 32k，1-10B 用 50-100k，多语言/前沿模型用 200k 以上。

## 交付

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: 针对给定语料库与部署目标，选择分词算法、词表大小与库。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

给定语料库（规模、语言、领域）和部署目标（从头训练 / 微调 / API 兼容推理），输出：

1. 算法。BPE、Unigram 或 WordPiece。一句话说明理由。
2. 库。SentencePiece、HF Tokenizers 或 tiktoken。说明理由。
3. 词表大小。取整到最近的 1k。理由需关联模型规模与语言覆盖。
4. 覆盖设置。`character_coverage`、`byte_fallback`、特殊词元列表。
5. 验证计划。在留出集上计算平均每个词所需词元数、OOV 率、压缩比、往返解码一致性。

若语料包含稀有文字，拒绝训练 `character_coverage < 0.995` 的分词器。拒绝交付 CI 中没有冻结 `tokenizer.json` 哈希检查的模型。将任何小于 16k 词表的单语分词器标记为可能欠规格。
```

## 练习

1. **简单。** 在 `code/main.py` 的微型语料上训练一个 500 次合并的 BPE。编码三个留出词。其中多少个恰好产生 1 个词元，多少个产生多于 1 个词元？
2. **中等。** 在 100 句英语维基百科文本上比较 `cl100k_base`、`o200k_base` 与你自训的 vocab=32k SentencePiece BPE 的词元数。报告各自的压缩比。
3. **困难。** 用同一语料分别训练 BPE、Unigram 和 WordPiece。在小型情感分类器上测量下游准确率。分词器选择是否会让 F1 变化超过 1 个百分点？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 贪婪合并最频繁的字符对，直到达到目标词表大小。 |
| Byte-level BPE | 永远不会出现未知词元 | 在原始 256 字节上运行的 BPE；GPT-2 / Llama 使用。 |
| Unigram | 概率分词器 | 从大型候选集中基于对数似然剪枝；T5、Gemma 使用。 |
| SentencePiece | 处理空格那个 | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 最快的那个 | OpenAI 用 Rust 实现的预建词表 BPE 编码器。不能训练。 |
| Merge list | 神秘数字 | `(a, b) → ab` 合并的有序列表；推理时按顺序应用。 |
| Character coverage | 多稀有算太稀有 | 分词器必须覆盖的训练语料字符比例；通常约 0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) —— Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) —— SentencePiece 库论文。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) —— 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) —— 教程与编码列表。
