# 从零构建分词器

> 第 01 课给了你一件玩具。这一课给你一件真正的武器。

**类型：** 实践
**语言：** Python
**前置要求：** 第 10 阶段，第 01 课（分词器：BPE、WordPiece、SentencePiece）
**时长：** 约 90 分钟

## 学习目标

- 构建一个生产级的 BPE 分词器，能够处理 Unicode、空白符归一化和特殊词元
- 实现字节级回退机制，使分词器能够编码任意输入（包括表情符号、中日韩文字和代码）而不产生未知词元
- 添加预分词正则表达式，在应用 BPE 合并前按词边界拆分文本
- 在语料上训练自定义分词器，并在多语言文本上评估其压缩率与 tiktoken 的对比

## 问题所在

你在第 01 课实现的 BPE 分词器在英文文本上运行良好。现在换成日文、表情符号，或者混用制表符和空格的 Python 代码试试。

它会崩溃。

不是因为 BPE 错了——而是实现不够完整。一个生产级分词器需要处理任意编码的原始字节、拆分前归一化 Unicode、管理不会参与合并的特殊词元、将预分词与子词拆分串联起来，并且速度要足够快，不能成为处理 15 万亿词元的训练流水线的瓶颈。

GPT-2 的分词器有 50,257 个词元。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些都不是玩具数字。这些词汇表背后的合并表是在数百 GB 文本上训练出来的，而周围的配套机制——归一化、预分词、特殊词元注入、对话模板格式化——才是区分一个只能处理 "hello world" 的分词器和一个能处理整个互联网的分词器的关键。

你将亲手构建这些机制。

## 核心概念

### 完整流水线

生产级分词器不是单一算法，而是由五个阶段组成的流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有明确的职责：

| 阶段 | 作用 | 重要性 |
|------|------|--------|
| 归一化（Normalize） | NFKC Unicode 归一化，可选小写化，可选去除重音符号 | "fi" 连字（U+FB01）会变成 "fi"（两个字符）。不做此处理，同一个单词会得到不同的词元。 |
| 预分词（Pre-Tokenize） | 在 BPE 前将文本拆分成块 | 防止 BPE 跨词边界合并。"the cat" 永远不应产生词元 "e c"。 |
| BPE 合并（BPE Merge） | 对字节序列应用学习到的合并规则 | 核心压缩步骤，将原始字节转换为子词词元。 |
| 特殊词元（Special Tokens） | 注入 [BOS]、[EOS]、[PAD]、对话模板标记等 | 这些词元拥有固定 ID，从不参与 BPE 合并。模型依靠它们识别结构。 |
| ID 映射（ID Mapping） | 将词元字符串转换为整数 ID | 模型看到的是整数，而不是字符串。 |

### 字节级 BPE

第 01 课的分词器已经在 UTF-8 字节上操作，这是正确的选择。但我们忽略了一个重要问题：当这些字节不是合法 UTF-8 时会发生什么？

字节级 BPE 通过将每个可能的字节值（0-255）都视为合法词元来解决这个问题。基础词汇表正好 256 项。任何文件——文本、二进制、损坏数据——都可以被分词，且不会产生未知词元。

GPT-2 使用了一个技巧：将每个字节映射到一个可打印的 Unicode 字符，使词汇表保持人类可读。例如它们的映射中，字节 0x20（空格）变成了字符 "G"。这纯粹是为了可读性，算法本身并不关心。

真正的威力在于：字节级 BPE 能处理地球上的所有语言。一个汉字占用 3 个 UTF-8 字节。日文可以是 3-4 字节。阿拉伯文、天城文、表情符号——都是字节序列。BPE 算法在这些字节序列中寻找模式的方式，与在英文 ASCII 字节中寻找模式的方式完全相同。

### 预分词

在 BPE 接触文本之前，你需要先将其拆分成块。这可以防止合并算法生成跨越词边界的词元。

GPT-2 使用一个正则表达式来拆分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

该模式会在缩写处拆分（"don't" 变成 "don" + "'t"），将带有可选前导空格的单词、数字、标点符号和空白符分别拆开。前导空格保留在单词上——因此 "the cat" 变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，完全跳过正则表达式。它将原始字节流视为一个长序列，让 BPE 算法自行学习边界。这种方式更简单，但给了 BPE 更多创建跨词词元的自由。

这个选择很重要。GPT-2 的正则表达式会阻止分词器学习前一个单词末尾的 "the" 与下一个单词开头的 "the" 应该合并。SentencePiece 允许这种情况，有时能获得更高的压缩效率，但词元的可解释性更差。

### 特殊词元

每个生产级分词器都会为结构性标记保留词元 ID：

| 词元 | 用途 | 使用模型 |
|------|------|----------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3、GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批次对齐填充 | BERT、T5 |
| `[UNK]` | 未知词元（字节级 BPE 消除了它） | BERT、WordPiece |
| `<\|im_start\|>` | 对话消息边界开始 | ChatGPT、Qwen |
| `<\|im_end\|>` | 对话消息边界结束 | ChatGPT、Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | 助手轮次标记 | Llama 3 |

特殊词元不会被 BPE 拆分。它们在合并算法运行前被精确匹配，替换为固定 ID，然后周围的文本再按正常方式分词。

### 对话模板

这是大多数人感到困惑、大多数实现也容易出错的地方。

当你向对话模型发送消息时，API 接收的是一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的不是 JSON，而是一个扁平的词元序列。对话模板使用特殊词元将消息列表转换为这个扁平序列。每个模型的做法都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板一旦写错，模型就会输出乱码。模型是在一种精确的格式上训练的。任何偏差——缺少换行、交换词元、多余空格——都会让输入偏离训练分布。

### 速度

对于生产级分词，Python 太慢了。

tiktoken（OpenAI）是用 Rust 编写并带有 Python 绑定的。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。它们比纯 Python 实现快 10-100 倍。

作为参考：以每秒 100 万词元（较快的 Python）的速度，为 Llama 3 预训练分词 15 万亿词元需要 174 天。以每秒 1 亿词元（Rust）的速度，只需要 1.7 天。

你用 Python 实现是为了理解算法。在生产环境中，你会使用编译后的实现，只通过 Python 包装器调用。

## 动手实现

### 步骤 1：字节级编码

这是基础。将任意字符串转换为字节序列，将每个字节映射到一个可打印字符以便显示，并实现反向转换。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

用多语言文本测试，观察字节数量：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 字节。"你好" 是 6 字节（每个字符 3 字节）。火焰表情是 4 字节。字节级分词器不关心语言。字节就是字节。

### 步骤 2：基于正则的预分词器

使用 GPT-2 的正则表达式将文本拆分成块。每个块由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持，因此我们回退到 ASCII 字符类。对于生产级多语言分词器，请安装 `regex`。

试试看：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保留在单词上。缩写在撇号处拆分。标点符号成为独立的块。BPE 不会跨这些边界合并词元。

### 步骤 3：字节序列上的 BPE

来自第 01 课的核心算法，但现在独立地对每个预分词块进行操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 步骤 4：特殊词元处理

特殊词元需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 步骤 5：完整的分词器类

将所有步骤串联起来：归一化、按特殊词元拆分、预分词、BPE 合并、映射为 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 步骤 6：多语言测试

真正的考验。用英文、中文、表情符号和代码来测试它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

每个汉字产生 3 字节。表情符号产生 4 字节。这些都不会让分词器崩溃，也不会产生未知词元。这就是字节级 BPE 的力量。

## 投入使用

### 对比真实分词器

加载 Llama 3、GPT-4 和 Mistral 的真实分词器，观察它们如何处理同一段多语言文本。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到同一段文本产生了不同的词元数量。Llama 3 拥有 128K 词汇表，更积极地合并常见模式。GPT-4 的 100K 处于中间。Mistral 的 32K 产生更多词元，但嵌入层更小。

权衡始终相同：词汇表越大，序列越短，但参数量也越多。

## 交付成果

本课将生成一个用于构建和调试生产级分词器的提示词。详见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加一个 `get_token_bytes(id)` 方法，显示任意词元 ID 对应的原始字节。用它查看最常见的合并词元实际代表什么。
2. **中等：** 实现 Llama 风格的预分词器，按空白符和数字拆分但保留前导空格。在相同语料上与 GPT-2 正则表达式方法对比词汇表差异。
3. **困难：** 添加一个对话模板方法，接收 `{"role": ..., "content": ...}` 消息列表，并生成 Llama 3 对话格式的正确词元序列。与 HuggingFace 实现进行对比测试。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 字节级 BPE（Byte-level BPE） | "在字节上工作的分词器" | 基础词汇表为 256 个字节值的 BPE——可处理任意输入而不产生未知词元 |
| 预分词（Pre-tokenization） | "BPE 之前的拆分" | 基于正则或规则的拆分，防止 BPE 跨词边界合并 |
| NFKC 归一化（NFKC normalization） | "Unicode 清理" | 先规范分解再做兼容组合——"fi" 连字变成 "fi"，全角 "A" 变成 "A" |
| 对话模板（Chat template） | "消息如何变成词元" | 将角色/内容消息列表转换为扁平词元序列的精确格式——因模型而异，必须匹配训练格式 |
| 特殊词元（Special tokens） | "控制词元" | 绕过 BPE 的保留词元 ID——[BOS]、[EOS]、[PAD]、对话标记——在合并前精确匹配 |
| 生育能力（Fertility） | "每词词元数" | 输出词元与输入单词的比率——GPT-4 英文约 1.3，韩语 2-3，越高意味着上下文浪费越严重 |
| tiktoken | "OpenAI 分词器" | 带有 Python 绑定的 Rust BPE 实现——比纯 Python 快 10-100 倍 |
| 合并表（Merge table） | "词汇表" | 训练过程中学习到的字节对合并的有序列表——这就是分词器学到的知识 |

## 延伸阅读

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken) —— GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) —— 支持 BPE、WordPiece、Unigram 的 Rust 分词器库
- [Llama 3 论文（Meta, 2024）](https://arxiv.org/abs/2407.21783) —— 关于 128K 词汇表和分词器训练的详情
- [SentencePiece（Kudo & Richardson, 2018）](https://arxiv.org/abs/1808.06226) —— 语言无关的分词方法
- [GPT-2 分词器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) —— 最初的字节到 Unicode 映射
