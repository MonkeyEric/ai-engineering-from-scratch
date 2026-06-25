# 投机解码（Speculative Decoding）与 EAGLE

> 前沿大语言模型（LLM）生成一个 token 需要对数百亿参数做一次完整前向传播。这次前向传播其实严重超配了：大多数情况下，一个小得多的模型就能正确猜出接下来的 3-5 个 token，而大模型只需要*验证*这个猜测。如果猜测正确，你就用一次前向传播的代价换来了 5 个 token。投机解码（Leviathan 等，2023）让这一过程变得精确无误，而 EAGLE-3（2025）将接受率推高到每次验证约 4.5 个 token——在保持相同输出分布的前提下实现了 4-5 倍加速。

**类型：** Build
**语言：** Python（使用 numpy）
**前置要求：** Phase 10 Lesson 12（推理优化）、Phase 10 Lesson 04（预训练 Mini-GPT）
**时间：** 约 75 分钟

## 问题所在

在 H100 上，700 亿参数级别模型的解码吞吐通常只有每秒 40-80 个 token。每个 token 都需要一次完整前向传播，从 HBM 读取全部模型权重。你不能在不改变输出的前提下把模型变小，也无法超过显存限制继续增大批大小。你陷入了僵局——除非能让模型每次前向传播输出不止一个 token。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但其中存在并发机会。如果你有一个廉价的预测器说“接下来 4 个 token 很可能是 [a, b, c, d]”，你就可以在**大模型的一次前向传播**中同时验证这 5 个位置，并接受最长的匹配前缀。

Leviathan、Kalai、Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过一个巧妙的接受/拒绝规则实现了精确的分布保持。同样的输出分布，快 2-4 倍。

## 核心概念

### 双模型设置

- **目标模型（target model）** `M_p`：你想要采样结果的大而慢、高质量的模型。分布：`p(x)`。
- **草稿模型（draft model）** `M_q`：小而快、质量较低的模型。分布：`q(x)`。通常小 5-30 倍。

每步流程：

1. 草稿模型自回归地提出 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型对所有 `K+1` 个位置并行运行**一次**前向传播，为每个被提议的 token 生成 `p(x_k)`。
3. 通过下面修改后的拒绝采样规则从左到右接受/拒绝每个 token，接受最长匹配前缀。
4. 如果有 token 被拒绝，则从修正后的分布中采样替换 token 并停止；否则从 `p(· | x_1...x_K)` 中采样一个额外 token。

如果草稿与目标完全一致，你每次目标前向传播就能得到 K+1 个 token；如果草稿在位置 1 就错了，你只能得到 1 个 token。

### 精确性规则

投机解码在**分布上与直接从 p 采样完全等价**。拒绝规则如下：

```
对每个草稿 token x_t：
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        接受 x_t
    else:
        从残差分布中采样替换：(p - q)+ / ||(p - q)+||_1
        停止
```

其中 `(p - q)+` 表示逐点差值的正部。当草稿与目标一致时（`p ≈ q`），接受率接近 1。当二者不一致时，残差分布的构造保证了最终样本仍然精确服从 `p`。

**贪心（greedy）情况。** 对于 temperature=0 的采样，只需检查 `argmax(p) == x_t`。若是则接受；否则输出 `argmax(p)` 并停止。

### 预期加速比

如果草稿模型的逐 token 接受率为 `α`，那么每次目标前向传播产生的预期 token 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = 草稿长度，α ∈ [0, 1]
```

当 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36`，即每次前向传播平均产生 3.36 个 token。一次单独的目标前向传播代价约为 `cost_q * K + cost_p`（K 次草稿步加一次目标验证）。如果 `cost_p >> cost_q * K`，则吞吐加速比为 `3.36× / 1 = 3.36×`。

唯一真正重要的参数是 `α`，而它完全取决于草稿模型与目标模型的对齐程度。一个好的草稿模型就是一切。

### 训练草稿模型：蒸馏

随机初始化的小模型作为草稿模型表现很差。标准做法是从目标模型蒸馏：

1. 选择一个小架构（70B 目标对应约 1B，7B 目标对应约 500M）。
2. 在大量文本语料上运行目标模型，存储其下一个 token 的分布。
3. 用 KL 散度（KL divergence）训练草稿模型去拟合目标模型的分布（而不是拟合真实 token）。

结果：`α` 在代码上通常为 0.6-0.8，在自然语言对话上为 0.7-0.85。生产中可获得 2-3 倍加速。

### EAGLE：树形草稿与特征复用

Li、Wei、Zhang、Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）指出了标准投机解码中的两个低效之处：

1. 草稿模型要进行 K 次串行步骤，每次都是完整堆栈。但草稿可以复用最近一次目标验证的特征（隐藏状态）——目标已经计算出了丰富的表示，而草稿却在从零重新推导。
2. 草稿输出的是线性链。如果草稿能输出一棵候选树（每个节点多个猜测），目标模型的一次前向传播就能通过树注意力掩码（tree attention mask）并行验证多条候选路径，并选择最长被接受的分支。

EAGLE-1 的改变：
- 草稿输入 = 位置 t 处目标的最终隐藏状态，而不是原始 token。
- 草稿架构 = 1 层 transformer 解码器（不是独立的小模型）。
- 输出 = 每深度 K = 4-8 个候选，深度 4-6 的树。

EAGLE-2（2024）增加了动态树拓扑：树在草稿不确定的地方变宽，在自信的地方保持窄。在不增加验证代价的前提下提高有效 `α`。

EAGLE-3（Li 等，2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）移除了固定的顶层特征依赖，并用新的“测试时模拟（test-time simulation）损失”训练草稿模型——草稿模型在训练时学习匹配目标模型在测试时的分布，而不是在教师强制（teacher forcing）训练分布下学习。接受率从 EAGLE-2 的 0.75 提升到 EAGLE-3 的 0.82，每次验证的平均 token 数从 3.0 提升到 4.5。

### 树注意力验证

当草稿输出一棵树时，目标模型使用**树注意力掩码**在单次前向传播中验证它——这是一种编码树拓扑结构的因果掩码，而非纯粹的线性序列。每个 token 只 attend 到树中的祖先节点。验证仍然只是一次前向传播、一次矩阵乘法；拓扑掩码只多花费少量 KV 缓存条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的第一个 token 候选，`c, d, e, f` 是第二个 token 候选，那么全部六个位置都在一次前向传播中完成验证。输出是任意被接受路径上的最长前缀。

### 何时有效，何时无效

**有效：**
- 可预测文本的聊天/补全任务（代码、常见英文、结构化输出）。`α` 很高。
- 解码阶段 GPU 计算未被占满的设置（内存受限阶段）。树形草稿可以利用闲置的 FLOPs。

**无效 / 没有收益：**
- 高度随机的输出（高温创意写作）。`α` 跌至 `1/|vocab|` 附近。
- 极高并发的批处理服务——批处理本身已经占满 FLOPs，留给树验证的空间很小。
- 目标模型很小，草稿模型没有明显更小。

生产环境中通常报告：聊天任务 2-3 倍实际加速，代码生成 3-5 倍，创意写作接近零加速。

## 动手实现

`code/main.py`：

- 一个参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确的拒绝规则，并通过经验 KL 散度（KL < 0.01）验证其保持目标分布。
- 一个 EAGLE 风格的树形草稿器，构建深度为 K、按 top-p 分支的树。
- 一个树注意力掩码构建器，为验证器生成正确的因果模式。
- 一个接受率测试框架，在一个极小的 LM 上运行以上两种方法（从 GPT-2-medium 目标蒸馏一个 GPT-2-small 草稿）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """一轮投机解码。返回被接受的 token 列表。"""
    # 1. 生成 K 个草稿 token
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标模型在每个草稿位置 + 1 个额外位置计算 p
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 从左到右接受/拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. 全部 K 个被接受 → 从目标模型采样一个奖励 token
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 实际使用

- **vLLM** 和 **SGLang** 都原生支持投机解码。参数：`--speculative_model`、`--num_speculative_tokens`。EAGLE-2/3 通过 `--spec_decoding_algorithm eagle` 参数支持。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 与 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 提供草稿）、`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 提供草稿）。
- **Medusa 头**（Cai 等，2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不引入独立草稿模型，而是在目标模型自身上添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付成果

本节课产出 `outputs/skill-speculative-tuning.md`——一项技能，用于分析目标模型的工作负载并选择：草稿模型、K（草稿长度）、树宽、温度，以及何时回退到普通解码。

## 练习题

1. 实现精确的拒绝规则并从经验上验证它。运行 `speculative_decode` 和普通目标采样各 1 万次；计算两个输出分布之间的总变差距离（TV distance）。应 < 0.01。

2. 计算加速公式。给定固定的 `α` 和 `K`，绘制每次目标前向传播的预期 token 数。找出 `α ∈ {0.5, 0.7, 0.9}` 时的最优 K。

3. 训练一个微型草稿模型。以 124M 的 GPT-2 为目标，在 1 亿 token 上用 KL 损失蒸馏一个 30M 的 GPT-2 草稿。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格的树形草稿。不再使用链式结构，而是让草稿在每个深度输出 top-3 分支。构建树注意力掩码，验证目标模型接受最长正确分支。

5. 测量失效模式。在 temperature=1.5（高随机性）下运行投机解码，展示 `α` 崩溃，且由于草稿开销算法比普通解码更慢。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Target model（目标模型） | “大模型” | 你想要采样结果的慢而高质量模型（p 分布） |
| Draft model（草稿模型） | “投机者” | 小而快的预测器（q 分布）；通常小 5-30 倍 |
| K / draft length（草稿长度） | “前瞻长度” | 每次验证轮次推测的 token 数 |
| α / acceptance rate（接受率） | “命中率” | 草稿提议逐 token 被接受的概率 |
| Exact rejection rule（精确拒绝规则） | “接受测试” | 比较 `r < p/q`，保持目标分布的测试 |
| Residual distribution（残差分布） | “修正后的 p-q” | `(p - q)+ / ||(p - q)+||_1`，拒绝时从中采样 |
| Tree drafting（树形草稿） | “分支式推测” | 草稿输出候选树，一次通过树结构注意力掩码验证 |
| Tree attention mask（树注意力掩码） | “拓扑掩码” | 编码树拓扑的因果掩码，每个节点只 attend 到祖先 |
| Medusa heads（Medusa 头） | “并行头” | 在目标模型自身上增加 K 个额外预测头；无独立草稿模型 |
| EAGLE feature reuse（EAGLE 特征复用） | “隐藏状态草稿” | 草稿输入是目标模型的最后一层隐藏状态，而非原始 token，从而压缩草稿 |
| Test-time simulation loss（测试时模拟损失） | “EAGLE-3 训练方法” | 让草稿模型学习匹配目标模型在测试时的分布，而非教师强制 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则与理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的同期投机采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 用并行头替代草稿模型
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用与树形草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时与测试时分布对齐
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi / 前瞻解码，一种无需投机模型的替代方案
