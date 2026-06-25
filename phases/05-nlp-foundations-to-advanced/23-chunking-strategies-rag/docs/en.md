# RAG 的分块策略

> 分块配置对检索质量的影响不亚于嵌入模型的选择（Vectara NAACL 2025）。分块出错时，再多的重排序也救不回来。

**类型：** Build  
**语言：** Python  
**前置条件：** Phase 5 · 14（Information Retrieval）、Phase 5 · 22（Embedding Models）  
**时长：** ~60 分钟

## 问题所在

你把一份 50 页的合同输入 RAG 系统。用户问：“终止条款是什么？” 检索器却返回封面页。为什么？因为模型是在 512 token 的分块上训练的，而终止条款位于第 20 页，跨页断开，且没有局部关键词把它和查询联系起来。

解决方案不是“买个更好的嵌入模型”，而是分块。多大？是否重叠？在哪里切分？是否带上下文？

2026 年 2 月的基准测试结果出人意料：

- Vectara 2026 年研究：递归式 512 token 分块以 69% 对 54% 的准确率击败语义分块。
- 在 Natural Questions 上，SPLADE + Mistral-8B：重叠没有带来任何可衡量的收益。
- 上下文悬崖：上下文达到约 2,500 token 时，回复质量急剧下降。

“显而易见”的答案（语义分块、20% 重叠、1000 token）往往是错的。本节课构建对六种策略的直觉，并告诉你该在何时使用哪一种。

## 核心概念

![在一段文本上可视化六种分块策略](../assets/chunking.svg)

**固定分块。** 每 N 个字符或 token 切一次。最简单的基线。会打断句子。压缩率高，连贯性差。

**递归分块。** LangChain 的 `RecursiveCharacterTextSplitter`。优先尝试按 `\n\n` 切分，然后 `\n`、`.`、空格。优雅回退。2026 年的默认选择。

**语义分块。** 对每个句子做嵌入。计算相邻句子的余弦相似度。在相似度低于阈值处切分。保留主题连贯性。较慢；有时产生仅 40 token 的碎片，损害检索。

**句子分块。** 按句子边界切分。每个 chunk 一个句子，或 N 个句子的窗口。在成本小得多的情况下，效果与语义分块相当（最长约 5k token）。

**父文档分块。** 同时存储用于检索的小子块和用于上下文的大父块。按子块检索，返回父块。即使子块质量差，也能稳妥地返回合理的父块。

**迟分块 / 晚期分块（Late Chunking，2024）。** 先对整个文档在 token 级别做嵌入，再把 token 嵌入汇聚成 chunk 嵌入。保留跨 chunk 上下文。适用于长上下文嵌入器（BGE-M3、Jina v3）。计算成本更高。

**上下文检索（Contextual Retrieval，Anthropic，2024）。** 在每个 chunk 前拼接一段 LLM 生成的、说明其在文档中位置的摘要（“本 chunk 位于终止条款的第 3.2 节……”）。在 Anthropic 自己的基准测试中，检索提升 35–50%。索引成本高昂。

### 超越所有默认值的规则

根据查询类型匹配分块大小：

| 查询类型 | 分块大小 |
|------------|-----------|
| 事实型（“CEO 叫什么名字？”） | 256-512 token |
| 分析型 / 多跳 | 512-1024 token |
| 整段理解 | 1024-2048 token |

NVIDIA 2026 年基准测试。chunk 应足够大以包含答案及其局部上下文，又足够小，使得检索器的 top-K 结果聚焦于答案而非上下文噪声。

## 动手实现

### 步骤 1：固定分块与递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域上调整 `threshold`。太高 → 碎片；太低 → 一个巨大 chunk。

### 步骤 3：父文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察：对父块去重。多个子块可能映射到同一父块；全部返回会浪费上下文。

### 步骤 4：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引带有上下文的 chunks。在查询时，检索会受益于额外的周围信息。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终进行基准测试。对你的语料库来说，“最佳”策略可能与任何博客文章都不一样。

## 常见陷阱

- **只在事实型查询上评估分块。** 多跳查询会揭示完全不同的胜者。使用按查询类型分层的评估集。
- **没有最小尺寸的语义分块。** 会产生损害检索的 40 token 碎片。始终强制设置 `min_tokens`。
- **把重叠当作 Cargo Cult。** 2026 年的研究发现重叠通常毫无收益，却会让索引成本翻倍。要测量，不要假设。
- **没有最小/最大限制。** 5 token 或 5000 token 的 chunk 都会破坏检索。必须夹紧限制。
- **跨文档分块。** 绝不允许 chunk 跨越两个文档。始终按文档分块，然后再合并。

## 如何使用

2026 年推荐栈：

| 场景 | 策略 |
|-----------|----------|
| 首次构建、未知语料库 | 递归分块，512 token，无重叠 |
| 事实型 QA | 递归分块，256-512 token |
| 分析型 / 多跳 | 递归分块，512-1024 token + 父文档分块 |
| 强交叉引用（合同、论文） | 迟分块（Late Chunking）或上下文检索 |
| 对话 / 对话语料 | 回合级分块 + 说话人元数据 |
| 短文本（推文、评论） | 一篇文档 = 一个 chunk |

从递归 512 开始。在 50 条查询的评估集上测量 Recall@5。再据此调优。

## 交付

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: 为给定的语料库和查询分布选择分块策略、大小和重叠量。
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

给定语料库（文档类型、平均长度、领域）和查询分布（事实型 / 分析型 / 多跳），输出：

1. 策略。递归 / 句子 / 语义 / 父文档 / 迟分块 / 上下文检索。理由。
2. 分块大小。Token 数量。理由需结合查询类型。
3. 重叠量。默认 0；若大于 0 需说明理由。
4. 最小/最大限制。设置 `min_tokens`、`max_tokens` 保护。
5. 评估计划。在 50 条查询的分层评估集（事实型、分析型、多跳）上测量 Recall@5。

拒绝任何未设置最小/最大 chunk 大小的分块策略。若未做消融实验证明有帮助，拒绝超过 20% 的重叠。对没有最小 token 下限的语义分块建议发出警告。
```

## 练习

1. **简单。** 用 `fixed(512, 0)`、`recursive(512, 0)`、`recursive(512, 100)` 对一份 20 页文档做分块。比较 chunk 数量和边界质量。
2. **中等。** 在 5 份文档上构建 30 条查询的评估集。测量递归分块、语义分块和父文档分块的 Recall@5。哪个胜出？是否与博客文章一致？
3. **困难。** 实现上下文检索。测量相比递归基线的 MRR 提升。报告索引成本（LLM 调用）与准确率提升的对比。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| Chunk | 文档片段 | 被嵌入、索引和检索的子文档单元。 |
| Overlap | 安全边距 | 相邻 chunk 共享的 N 个 token；在 2026 年的基准测试中通常没用。 |
| Semantic chunking | 智能分块 | 在相邻句子的嵌入相似度下降处切分。 |
| Parent-document | 两级检索 | 检索小子块，返回更大的父块。 |
| Late chunking | 嵌入后再分块 | 先在 token 级别嵌入完整文档，再汇聚成 chunk 向量。 |
| Contextual retrieval | Anthropic 的技巧 | 在索引前为每个 chunk 前置 LLM 生成的摘要。 |
| Context cliff | 2500 token 墙 | 在 RAG 中观察到上下文约 2.5k token 时质量下降（2026 年 1 月）。 |

## 延伸阅读

- [Yepes 等 / LangChain — 递归字符切分文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产中的默认选择。
- [Vectara (2024, NAACL 2025). 分块配置分析](https://arxiv.org/abs/2410.13070) — 分块的重要性不亚于嵌入模型选择。
- [Jina AI — 长上下文嵌入模型中的迟分块（2024）](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 迟分块论文。
- [Anthropic — 上下文检索](https://www.anthropic.com/news/contextual-retrieval) — 使用 LLM 生成上下文前缀带来 35-50% 的检索提升。
- [NVIDIA 2026 分块大小基准测试 — Premai 摘要](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型选择分块大小。
