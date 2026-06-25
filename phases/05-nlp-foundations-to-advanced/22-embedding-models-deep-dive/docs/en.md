# 嵌入模型 —— 2026 年深度解析

> Word2Vec 为每个词提供一个向量。现代嵌入模型为每个段落提供一个向量，支持跨语言，并同时提供稀疏、稠密和多向量视图，维度可按索引规模裁剪。选错了，你的 RAG 就会检索到错误的内容。

**类型：** 学习
**语言：** Python
**先修：** Phase 5 · 03（Word2Vec）、Phase 5 · 14（信息检索）
**时长：** 约 60 分钟

## 问题所在

你的 RAG 系统有 40% 的时间检索到了错误的段落。罪魁祸首很少是向量数据库或提示词，而是嵌入模型。

2026 年选择嵌入模型意味着要在五个维度上做取舍：

1. **稠密 vs 稀疏 vs 多向量。** 每个段落一个向量，或每个 token 一个向量，或带权重的稀疏词袋。
2. **语言覆盖范围。** 仅英文任务上，单语英语模型仍然占优；语料混杂时，多语言模型更优。
3. **上下文长度。** 512 token vs 8,192 vs 32,768 —— 而真实有效容量通常只有标称最大值的 60-70%。
4. **维度预算。** 3,072 维全精度浮点数 = 每个向量 12 KB。1 亿向量时，存储成本为每月 1,300 美元。Matryoshka 截断可将存储降低 4 倍。
5. **开源 vs 托管。** 开源权重意味着你掌控全栈和数据；托管意味着用控制权换取始终最新的模型。

本课将逐一说明这些权衡，让你能基于证据而非上季度的流行趋势做选择。

## 核心概念

![稠密、稀疏与多向量嵌入](../assets/embedding-modes.svg)

**稠密嵌入。** 每个段落一个向量（通常 384-3,072 维）。余弦相似度按语义接近程度排序。代表：OpenAI `text-embedding-3-large`、BGE-M3 稠密模式、Voyage-3。默认首选。

**稀疏嵌入。** SPLADE 风格。Transformer 为词表中每个 token 预测一个权重，然后把大部分置零。结果是一个 |vocab| 大小的稀疏向量。能捕获词汇匹配（类似 BM25），但权重是学出来的。对关键词密集型查询很强。

**多向量（后期交互）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。用 MaxSim 打分：对每个查询 token，找到最相似的文档 token，累加分数。存储和打分成本更高，但在长查询和领域特定语料上表现更好。

**BGE-M3：三种同时输出。** 单个模型同时输出稠密、稀疏和多向量表示。每种都可以独立检索；分数通过加权求和融合。2026 年当你想从一个检查点获得灵活性时的默认选择。

**Matryoshka 表示学习。** 训练目标使得向量的前 N 维本身构成可用的独立嵌入。把 1,536 维截断到 256 维，只需付出约 1% 的准确率代价，却能节省 6 倍存储。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 均支持。

### MTEB 排行榜只讲了部分事实

Massive Text Embedding Benchmark —— 发布时（2022 年）包含 8 类任务共 56 项，MTEB v2 扩展到 100 多项。2026 年初，Gemini Embedding 2 在检索榜居首（67.71 MTEB-R），Cohere embed-v4 领跑综合榜（65.2 MTEB），BGE-M3 在开源多语言榜上领先（63.0）。排行榜是必要但不充分的条件 —— 务必在你自己的领域上评测。

### 三层模式

| 使用场景 | 模式 |
|----------|------|
| 快速初筛 | 稠密双编码器（BGE-M3、text-3-small） |
| 提升召回 | 稀疏（SPLADE、BGE-M3 sparse）+ RRF 融合 |
| top-50 精度 | 多向量（ColBERTv2）或交叉编码器重排器 |

大多数生产系统会同时用上三种。

## 动手实现

### 步骤 1：基线 —— 用 Sentence-BERT 生成稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 让点积等价于余弦相似度。务必始终设置。

### 步骤 2：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 经过训练，前几层截断几乎无损。非 Matryoshka 模型（原始 Sentence-BERT）截断后会急剧退化。

### 步骤 3：BGE-M3 多功能嵌入

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

一次推理得到三种索引。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

权重需在你的领域上调优。

### 步骤 4：在自定义任务上运行 MTEB 评测

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*有代表性*的子集上运行候选模型。不要只信任排行榜排名 —— 你的领域更重要。

### 步骤 5：从零手写余弦相似度

参见 `code/main.py`。平均 Hashing Trick 嵌入（仅标准库）。无法与 Transformer 嵌入竞争，但展示了完整流程：分词 → 向量 → 归一化 → 点积。

## 常见陷阱

- **查询和文档用同一个模型。** 某些模型（Voyage、Jina-ColBERT）采用非对称编码 —— 查询和文档经过不同路径。务必查看模型卡。
- **漏加前缀。** `bge-*` 模型需要在查询前拼接 `"Represent this sentence for searching relevant passages: "`。忘记的话召回会下降 3-5 个百分点。
- **Matryoshka 过度截断。** 1,536 → 256 通常安全，1,536 → 64 则不一定。请在评测集上验证。
- **上下文截断。** 大多数模型会静默截断超过最大长度的输入。长文档需要分块（见第 23 课）。
- **忽视长尾延迟。** MTEB 分数掩盖了 p99 延迟。6 亿参数模型可能比 3.35 亿参数模型高 2 分，但单次查询成本高 3 倍。

## 如何选择

2026 年的推荐栈：

| 场景 | 选择 |
|-----------|------|
| 仅英文、快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重、英文 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（1.37 亿参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集型查询 | 增加 SPLADE 稀疏，与稠密做 RRF 融合 |

2026 年套路：先用 BGE-M3 或 text-3-large，在你的领域上用 MTEB 评测，若某个领域专用模型领先超过 3 分再替换。

## 交付成果

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习题

1. **简单。** 用 `bge-small-en-v1.5` 对 100 个句子以完整维度（384）编码，再以 Matryoshka 128 维编码。在 10 个查询上测量 MRR 下降。
2. **中等。** 在你的领域 500 个段落上对比 BGE-M3 的稠密、稀疏和 colbert 模式。哪种在 recall@10 上胜出？RRF 融合能否击败最佳单模式？
3. **困难。** 在三个候选模型上运行 MTEB，覆盖你最重要的两项领域任务。报告 MTEB 分数、100 条查询批次上的 p99 延迟，以及每百万次查询成本。选择帕累托最优者。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Dense embedding | 向量 | 每个文本一个固定长度向量，用余弦相似度排序。 |
| Sparse embedding | 学出来的 BM25 | 词表中每个 token 一个权重，大部分是零，端到端训练。 |
| Multi-vector | ColBERT 风格 | 每个 token 一个向量；MaxSim 打分；索引更大，召回更高。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维本身就是可用的小尺寸嵌入。 |
| MTEB | 那个基准 | Massive Text Embedding Benchmark —— 发布时 56 项任务，v2 超过 100 项。 |
| BEIR | 检索基准 | 18 个零样本检索任务；常被用来衡量跨领域鲁棒性。 |
| Asymmetric encoding | 查询 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) —— 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) —— 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) —— 统一三种模式的模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) —— 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) —— 生产中的后期交互。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) —— 实时排名。
