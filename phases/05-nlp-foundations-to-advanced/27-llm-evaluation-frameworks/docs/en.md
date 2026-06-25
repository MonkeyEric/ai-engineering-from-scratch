# LLM 评估 —— RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 无法捕捉语义等价。人工审核无法规模化。LLM-as-judge 是生产级答案——只要有足够的校准，就可以信任这个数字。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 5 · 第 13 课（问答系统），阶段 5 · 第 14 课（信息检索）  
**时长：** 约 75 分钟

## 问题背景

你的 RAG 系统回答："June 29th, 2007."  
黄金参考是："June 29, 2007."  
精确匹配得分为 0，F1 约 75%。人工会打 100 分。

现在乘以 10,000 个测试用例；再乘以检索器、分块、提示词或模型的每一次改动。你需要一个能理解语义、可廉价规模化运行、不会谎报回退、并能暴露正确失效模式的评估器。

2026 年有三个框架主导了这个问题。

- **RAGAS。** 检索增强生成评估（Retrieval-Augmented Generation ASsessment）。四项 RAG 指标（忠实性、答案相关性、上下文精确率、上下文召回率），后端使用 NLI + LLM 裁判。有研究背书、轻量。
- **DeepEval。** 面向 LLM 的 Pytest。G-Eval、任务完成度、幻觉、偏见指标。原生支持 CI/CD。
- **G-Eval。** 一种方法（也是 DeepEval 的指标）：基于思维链的 LLM 裁判、自定义标准、0-1 分数。

三者都依赖 LLM-as-judge。本节课将建立对该方法及其信任层的直觉。

## 核心概念

![四个评估维度、LLM-as-judge 架构](../assets/llm-evaluation.svg)

**LLM-as-judge。** 用一个根据评分标准（rubric）给输出打分的 LLM 来替代静态指标。给定 `(query, context, answer)`，向裁判 LLM 提问："在忠实性上给出 0-1 分。" 返回分数。

它为何有效：LLM 以极低的成本逼近人类判断。GPT-4o-mini 每次评分约 $0.003，1000 个样本的回退评估运行成本不到 $5。

它为何悄无声息地失效：

1. **裁判偏见。** 裁判更偏爱更长的答案、来自同一模型家族的答案、与提示风格一致的答案。
2. **JSON 解析失败。** 错误 JSON → NaN 分数 → 在聚合中被静默排除。RAGAS 用户深知此痛。用 try/except 捕获并显式标记失效模式。
3. **模型版本漂移。** 升级裁判会改变所有指标。冻结裁判模型 + 版本。

**RAG 四项指标。**

| 指标 | 问题 | 后端 |
|------|------|------|
| Faithfulness | 答案中的每个主张是否都来自检索到的上下文？ | 基于 NLI 的蕴涵 |
| Answer relevance | 答案是否回答了问题？ | 从答案生成假设问题，再与真实问题比较 |
| Context precision | 检索到的块中有多大比例是相关的？ | LLM 裁判 |
| Context recall | 检索是否返回了所有需要的内容？ | 对照黄金答案的 LLM 裁判 |

**G-Eval。** 定义自定义标准："答案是否引用了正确的来源？" 框架会自动扩展为思维链评估步骤，然后给出 0-1 分。适用于 RAGAS 未涵盖的特定领域质量维度。

**校准。** 在与人工标签建立相关性之前，永远不要信任原始裁判分数。运行 100 个人工标注样本。绘制裁判分 vs 人工分。计算 Spearman rho。若 rho < 0.7，说明你的裁判评分标准需要改进。

## 动手实现

### 步骤 1：使用 NLI 评估忠实性（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案拆分为原子主张；用 NLI 逐一检查每个主张是否被检索到的上下文支持。忠实性 = 被支持的主张比例。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与所问问题不同，相关性就会下降。

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分标准。显式步骤比隐式"给出 0-1 分"提示更稳定。

### 步骤 4：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

将其作为 pytest 文件提交。在每个 PR 上运行。出现回退时阻止合并。

### 步骤 5：从零开始的玩具评估器

参见 `code/main.py`。仅使用标准库近似实现忠实性（答案主张与上下文的重叠）和相关性（答案 token 与问题 token 的重叠）。非生产级。用于展示基本形态。

## 常见陷阱

- **未校准。** 与人工标签相关性仅 0.3 的裁判只是噪声。上线前必须做一次校准运行。
- **自我评估。** 用同一个 LLM 既生成又裁判会使分数虚高 10-20%。裁判应使用不同的模型家族。
- **成对裁判中的位置偏见。** 裁判更偏好先出现的选项。始终随机化顺序并两种顺序都跑。
- **原始聚合掩盖失败。** 平均分 0.85 往往掩盖了 5% 的灾难性失败。务必检查底部分位数。
- **黄金数据集腐化。** 未版本化的评估集会随时间漂移，破坏纵向比较。每次改动都要给数据集打标签。
- **LLM 成本。** 规模化后，裁判调用占主导成本。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 如何使用

2026 年技术栈：

| 使用场景 | 框架 |
|----------|------|
| RAG 质量监控 | RAGAS（4 项指标） |
| CI/CD 回退门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线实时流量监控 | RAGAS 无参考模式 |
| 人机协同抽检 | LangSmith 或 Phoenix（带标注 UI） |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型组合：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。三个都跑；它们之间的不一致本身就有价值。

## 交付产物

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: 设计一个带有校准裁判和 CI 门控的 LLM 评估方案。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

给定一个使用场景（RAG / 智能体 / 生成式任务），输出：

1. 指标。忠实性 / 相关性 / 上下文精确率 / 上下文召回率，以及任何带标准的自定义 G-Eval 指标。
2. 裁判模型。指定模型 + 版本，并说明成本与准确率的权衡理由。
3. 校准。人工标注集大小，目标 Spearman rho vs 人工 > 0.7。
4. 数据集版本控制。标签策略、变更日志、分层策略。
5. CI 门控。每项指标的阈值、回退窗口逻辑、底部分位数告警。

拒绝依赖未在 ≥50 个人工标注样本上测试过的裁判。拒绝自我评估（同一模型既生成又裁判）。拒绝只报告聚合分数而不暴露底部 10%。标记任何未做并行基线评估就升级裁判的流水线。
```

## 练习

1. **简单。** 在 10 个已知存在幻觉的 RAG 示例上使用 RAGAS。验证忠实性指标能捕捉每一个幻觉。
2. **中等。** 对 50 个 QA 答案按正确性人工标注 0-1。用 G-Eval 打分。测量裁判与人工之间的 Spearman rho。
3. **困难。** 用 DeepEval 构建 pytest CI 门控。故意让检索器回退。验证门控失败。通过对最低 10% 设置阈值检查来添加底部分位数告警。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| LLM-as-judge | 用 LLM 打分 | 向裁判模型发出提示，根据评分标准给输出打出 0-1 分。 |
| RAGAS | RAG 指标库 | 带有 4 项无参考 RAG 指标的开源评估框架。 |
| Faithfulness | 答案是否有依据？ | 被检索上下文蕴涵的答案主张比例。 |
| Context precision | 检索到的块是否相关？ | 实际起作用的 top-K 块比例。 |
| Context recall | 检索是否找全了？ | 被检索块支持的黄金答案主张比例。 |
| G-Eval | 自定义 LLM 裁判 | 评分标准 + 思维链评估步骤 + 0-1 分。 |
| Calibration | 信任但验证 | 裁判分数与人工分数之间的 Spearman 相关性。 |

## 延伸阅读

- [Es 等（2023）。RAGAS：检索增强生成的自动评估](https://arxiv.org/abs/2309.15217) —— RAGAS 论文。
- [Liu 等（2023）。G-Eval：利用 GPT-4 进行更好人类对齐的 NLG 评估](https://arxiv.org/abs/2303.16634) —— G-Eval 论文。
- [DeepEval 文档](https://deepeval.com/docs/metrics-introduction) —— 开放的生产级技术栈。
- [Zheng 等（2023）。用 MT-Bench 和 Chatbot Arena 审视 LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) —— 偏见、校准与局限。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) —— 集成 RAGAS、DeepEval、Phoenix 的统一框架。
