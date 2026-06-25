# 长上下文评估 —— NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣传支持 1000 万 token 上下文。但在 100 万 token 时，8-needle MRCR 降至 26.3%。宣传值 ≠ 可用值。长上下文评估告诉你，你实际部署的模型的真实容量。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 5 · 13（问答系统），Phase 5 · 23（分块策略）
**时间：** 约 60 分钟

## 问题所在

你有一份 200 页的合同。模型声称支持 100 万 token 上下文。你把合同贴进去，问：“终止条款是什么？”模型给出了答案——但它回答的是封面页，因为终止条款埋在 12 万 token 深处，超过了模型实际关注的范围。

这就是 2026 年的上下文容量鸿沟。规格表上写着 100 万或 1000 万。现实是，通常只有 60-70% 可用，而“可用”还取决于任务类型。

- **检索（单针大海捞针）：** 在前沿模型上，直到宣传的最大长度都接近完美。
- **多跳 / 聚合：** 在大多数模型上，超过约 128k 后急剧下降。
- **对分散事实进行推理：** 最先失败的任务。

长上下文评估衡量这些维度。本课介绍各个基准测试、它们实际测量什么，以及如何为你的领域构建自定义的 needle 测试。

## 核心概念

![NIAH 基线、RULER 多任务、LongBench 整体评估](../assets/long-context-eval.svg)

**Needle-in-a-Haystack（NIAH，2023）。** 将一个事实（“magic word is pineapple”）放在长上下文中的可控深度，让模型检索它。遍历深度 × 长度。这是最早的长上下文基准。前沿模型现在已在此饱和；它是必要但不充分的基线。

**RULER（Nvidia，2024）。** 4 个类别下的 13 种任务类型：检索（单键 / 多键 / 多值）、多跳追踪（变量跟踪）、聚合（常见词频率）、问答。上下文长度可配置（4k 到 128k+）。能揭示在 NIAH 上饱和但在多跳任务上失败的模型。在 2024 年的发布中，17 个声称支持 32k+ 上下文的模型中，只有一半在 32k 上保持了质量。

**LongBench v2（2024）。** 503 道选择题，8k-200 万词上下文，六个任务类别：单文档问答、多文档问答、长上下文上下文学习、长对话、代码仓库、长结构化数据。真实世界长上下文行为的生产级基准。

**MRCR（多轮共指消解）。** 大规模多轮共指。8-needle、24-needle、100-needle 变体。暴露模型在注意力退化前能同时处理多少事实。

**NoLiMa。** “非词面 needle。”needle 与查询没有字面重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接多个文档，从任意一个文档中提问。测试选择性注意力。

**BABILong。** 将 bAbI 推理链嵌入无关的 haystack 中。测试“haystack 中的推理”，而不仅是检索。

### 实际应该报告什么

- **宣传的上下文窗口。** 规格表上的数字。
- **有效检索长度。** NIAH 在某个阈值（例如 90%）下通过。
- **有效推理长度。** 多跳或聚合任务在该阈值下通过。
- **退化曲线。** 准确率随上下文长度变化，按任务类型绘制。

给你的规格表两个数字：检索有效长度和推理有效长度。通常推理有效长度只有宣传窗口的 25-50%。

## 动手实现

### 步骤 1：为你的领域构建自定义 NIAH

参见 `code/main.py`。骨架如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

遍历 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。这就是你目标模型的 NIAH 报告卡。

### 步骤 2：多 needle 变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像“三个 magic word 是什么？”这样的问题需要全部检索出来。单 needle 的成功无法预测多 needle 的成功。

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

回答需要串联三次赋值。在 128k 长度下，前沿模型在此任务上的准确率常降至 50-70%。

### 步骤 4：在你的技术栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。聚合分数会掩盖任务层面的巨大差异。

## 常见陷阱

- **只做 NIAH 评估。** 在 100 万 token 上通过 NIAH 不能说明多跳能力。务必运行 RULER 或自定义多跳测试。
- **均匀的深度采样。** 许多实现只测试 depth=0.5。应测试 depth=0、0.25、0.5、0.75、1.0——“中间迷失”效应是真实存在的。
- **与 filler 的词面重叠。** 如果 needle 与 filler 共享关键词，检索会变得过于简单。使用 NoLiMa 风格的非重叠 needle。
- **忽视延迟。** 100 万 token 的 prompt 预填充需要 30-120 秒。在测量准确率的同时，也要测量首 token 时间。
- **相信厂商自报数字。** OpenAI、Google、Anthropic 都会发布自己的分数。务必在你的用例上独立复测。

## 如何使用

2026 年的工具栈：

| 场景 | 基准测试 |
|-----------|-----------|
| 快速 sanity check | 自定义 NIAH，3 个深度 × 3 个长度 |
| 生产环境模型选型 | RULER（13 个任务），在你的目标长度上运行 |
| 真实世界问答质量 | LongBench v2 单文档问答子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话 / 交互场景 | MRCR 8-needle，在你的目标长度上运行 |
| 模型升级回归测试 | 固定内部 NIAH + RULER 测试套件，每次新模型都跑 |

生产经验法则：在目标长度上完成 NIAH + 1 项推理任务之前，永远不要相信上下文窗口。

## 交付成果

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: 为给定模型和用例设计一套长上下文评估方案。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

给定目标模型、目标上下文长度和用例，输出：

1. 测试项。NIAH 深度 × 长度网格；RULER 多跳；自定义领域任务。
2. 采样。每个长度取深度 0、0.25、0.5、0.75、1.0。
3. 指标。检索通过率；推理通过率；首 token 时间；单次查询成本。
4. 截断标准。有效检索长度（90% 通过）和有效推理长度（70% 通过）。两者都报告。
5. 回归测试。固定测试套件，每次模型升级都重跑，并展示差异。

拒绝只凭模型卡就相信上下文窗口。拒绝为任何多跳工作负载只做 NIAH 评估。拒绝把厂商自报的长上下文分数当作独立证据。
```

## 练习

1. **简单。** 构建一个 NIAH，3 个深度（0.25、0.5、0.75）× 3 个长度（1k、4k、16k）。在任意模型上运行。将通过率绘制成 3×3 热力图。
2. **中等。** 添加 3-needle 变体。测量在每个长度上检索全部 3 个 needle 的能力。与相同长度下的单 needle 通过率对比。
3. **困难。** 构造一个变量追踪任务（X1 → X2 → X3，共 3 跳），嵌入 64k 的 filler 中。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | 大海捞针 | 在 filler 中埋入一个事实，让模型检索它。 |
| RULER | 加强版 NIAH | 13 种任务类型，覆盖检索 / 多跳 / 聚合 / 问答。 |
| 有效上下文 | 真实容量 | 准确率仍高于阈值的上下文长度。 |
| 中间迷失 | 深度偏差 | 模型对长输入中间内容关注不足。 |
| 多 needle | 同时处理多个事实 | 多个埋入点；测试注意力调度，而不仅是检索。 |
| MRCR | 多轮共指 | 8、24 或 100-needle 共指；暴露注意力饱和点。 |
| NoLiMa | 非词面 needle | needle 与查询没有字面 token 重叠；需要推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) —— 原始 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) —— 多任务基准。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) —— 真实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) —— 更难的 needle。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) —— haystack 中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) —— 深度偏差论文。
