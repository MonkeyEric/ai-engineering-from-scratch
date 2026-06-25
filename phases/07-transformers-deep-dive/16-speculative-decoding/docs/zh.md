# 投机解码（Speculative Decoding）——起草、验证、重复

> 自回归解码（autoregressive decoding）是串行的：每个 token 都要等前一个生成完毕。投机解码打破了这条锁链：先由一个廉价模型一次性起草 N 个 token，再由大模型在一个前向传播（forward pass）中全部验证。当草稿正确时，你只花了一次大模型前向的代价就得到了 N 个生成 token。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 · 07（GPT 因果语言模型），Phase 7 · 12（KV 缓存与 Flash Attention）
**时长：** 约 60 分钟

## 问题背景

在 H100 上，一个 70B 的大语言模型采样一个 token 大约需要 30 毫秒；而一个 3B 的草稿模型只需约 3 毫秒。如果我们让 3B 模型提前起草 5 个 token，然后让 70B 模型*只运行一次*来验证这 5 个 token，总耗时为 `5×3 + 30 = 45 毫秒`，最多可接受 5 个 token；相比之下，直线生成需要 `5×30 = 150 毫秒。这就是投机解码的全部卖点：用少量额外的 GPU 显存（草稿模型）换取 2–4 倍的解码延迟降低。

这个技巧必须保持输出分布不变。由 Leviathan 等人（2023）与 Chen 等人同时提出的投机采样（speculative sampling）保证：输出序列与目标大模型独自采样时**同分布**。没有质量损失，只是更快。

到 2026 年，四大草稿-验证模型组合主导了推理部署：

1. **经典投机解码（Vanilla speculative，Leviathan 2023）。** 独立的草稿模型（如 Llama 3 1B）+ 验证模型（如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 在验证模型上增加多个解码头（decoding heads），并行预测位置 `t+1..t+k` 的 token。不需要单独的草稿模型。
3. **EAGLE 系列（Li 2024, 2025）。** 轻量草稿模型复用验证模型的隐藏状态（hidden states），接受率比经典方法更高；通常可达 3–4 倍加速。
4. **Lookahead 解码（Fu 2024）。** 采用 Jacobi 迭代，完全不需要草稿模型。属于自投机解码，应用场景较窄，但没有任何额外依赖。

到 2026 年，所有生产级推理栈都默认内置投机解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少都支持经典投机 + EAGLE-2。

## 核心概念

### 核心算法

给定验证模型 `M_q` 和更廉价的草稿模型 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀（prefix）。
2. **起草**：用 `M_p` 自回归地提出草稿 token `d_{k+1}, d_{k+2}, ..., d_{k+N}`，对应草稿概率为 `p_1..p_N`。
3. **并行验证**：将 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 一次性输入 `M_q`，得到位置 `k+1..k+N+1` 的验证概率 `q_1..q_{N+1}`。
4. **从左到右逐个接受/拒绝**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受该草稿 token。
5. 若首次在位置 `j` 被拒绝：从归一化后的“残差”分布 `(q_j - p_j)_+` 中采样 `t_j`，并丢弃 `j` 之后的所有草稿 token。
6. 若全部 `N` 个 token 都被接受：再从 `q_{N+1}` 中多采样一个 token（免费的奖励 token）。

残差分布（residual distribution）技巧是数学上的关键洞察，它保证输出分布与 `M_q` 从头开始采样完全一致。

### 什么决定了加速比

设 `α` 为每个草稿 token 的期望接受率，`c` 为草稿模型相对验证模型的成本比例。每步：

- 朴素生成：每个 token 调用一次大模型。
- 投机解码：当 `α` 较高时，每次大模型调用平均生成 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token。

经验法则：当 `α = 0.75`、`N = 5` 时，大模型调用次数减少约 3 倍；草稿成本仅为 1/5。总 wall-clock 时间下降约 2.5 倍。

**`α` 取决于：**

- 草稿模型对验证模型的逼近程度。同一家族、相同训练数据能显著提升 `α`。
- 解码策略。贪心（greedy）草稿对贪心验证：接受率高。温度采样（temperature sampling）更难匹配，接受率会下降。
- 任务类型。代码和结构化输出更容易被接受（可预测）；自由形式的创意写作接受率较低。

### Medusa——无需草稿模型的起草

Medusa 用验证模型上的额外输出头替换掉独立的草稿模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出自己的 logits。推理时从每个头采样得到候选序列，再用一次前向传播结合树形注意力（tree-attention）机制同时验证所有候选延续。

优点：不需要第二个模型。缺点：增加可训练参数；需要监督微调阶段（约 1B token）；接受率略低于使用优质草稿模型的经典投机解码。

### EAGLE——复用隐藏状态获得更好的草稿模型

EAGLE-1/2/3（Li et al., 2024–2025）把草稿模型做成一个极小的 transformer（通常 1 层），输入是验证模型最后一层的隐藏状态。由于草稿模型能看到验证模型的特征表示，其预测分布与验证模型的输出分布高度相关。接受率从经典方法的约 0.6 提升到 0.85 以上。

EAGLE-3（2025）进一步加入了候选延续的树搜索。vLLM 与 SGLang 已将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认投机路径。

### KV 缓存的进退舞步

验证阶段把 `N` 个草稿 token 一次性喂给验证模型，使其 KV 缓存（KV cache）延长 `N` 个条目。若某些草稿被拒绝，就必须把 KV 缓存回滚到已接受前缀的长度。

生产实现（如 vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）使用临时 KV 缓冲区来处理：先写入，接受后再提交。概念不难，但工程细节很繁琐。

## 动手实现

参见 `code/main.py`。我们实现了核心投机采样算法（拒绝步 + 残差分布），包括：

- 一个“大模型”，即对人工设定分布做确定性 softmax（这样可以解析地验证接受率数学）。
- 一个“草稿模型”，是对大模型分布的扰动。
- 一个接受/拒绝循环，保证输出边缘分布与直接采样一致。

### 步骤 1：拒绝判断

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数；`q_prob` 是验证模型对草稿 token 的概率；`p_prob` 是草稿模型对该 token 的概率。Leviathan 定理指出：这个 Bernoulli 决策加上拒绝时从残差分布中采样，能精确保持验证模型的分布。

### 步骤 2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

对 `q` 和 `p` 逐元素相减，负值截断为零后重新归一化。任何拒绝都从这个分布中采样。

### 步骤 3：一次投机步

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个全被接受 → 多采一个奖励 token → 一次验证调用产出六个 token。

### 步骤 4：测量接受率

运行 10,000 次投机步，改变草稿模型的质量水平。绘制接受率与草稿-验证分布间 KL 散度的关系图，应能看到清晰的单调关系。

### 步骤 5：验证分布等价性

经验验证：投机循环生成的 token 直方图应与直接从验证模型采样生成的直方图一致。这就是 Leviathan 定理的实际体现。卡方检验（chi-square test）会确认二者在采样误差范围内一致。

## 实际使用

生产环境：

```bash
# vLLM + EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM + 经典草稿模型
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中期，TensorRT-LLM 拥有最快的 Medusa 路径。`faster-whisper` 也为 Whisper-large 包装了投机解码，并配有一个小型草稿模型。

**选择草稿策略：**

| 策略 | 适用场景 | 加速比 |
|------|----------|--------|
| 经典草稿模型（1B/3B Llama 系列） | 快速原型，无需训练 | 1.8–2.3× |
| Medusa 多头 | 可以对验证模型微调 | 2–3× |
| EAGLE-2 / 3 | 生产环境，追求最大速度 | 3–4× |
| Lookahead | 无草稿、无训练、无额外参数 | 1.3–1.6× |

**不适合投机解码的情况：**

- 只生成 1–5 个 token 的单序列任务。开销占主导。
- 高度创意/高温采样（`α` 会下降）。
- 显存受限的部署（草稿模型增加 VRAM 占用）。

## 交付

参见 `outputs/skill-spec-decode-picker.md`。该技能会针对新的推理负载，选择投机解码策略（经典 / Medusa / EAGLE / lookahead）并调参（`N`、草稿温度等）。

## 练习

1. **简单。** 运行 `code/main.py`。确认在 50,000 个 token 上，投机采样的 token 分布与直接采样分布一致（卡方检验 p > 0.05）。
2. **中等。** 对 `α = 0.5, 0.7, 0.85`，绘制加速比（每次大模型前向产生的 token 数）随 `N` 变化的曲线，找出每个 `α` 下的最优 `N`。（提示：每次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个微型 Medusa：取第 14 课的大作业 GPT，添加 3 个额外 LM 头分别预测位置 t+2、t+3、t+4。用 tinyshakespeare 以联合多任务损失训练，比较与截断同一模型得到的经典草稿模型的接受率。
4. **困难。** 实现 KV 缓存回滚：从 10 个 token 前缀的 KV 缓存开始，喂入 5 个草稿 token，模拟在第 3 个位置发生拒绝。验证下一轮迭代中缓存读取结果与“前缀 + 前 2 个被接受的草稿”一致。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 草稿模型（draft model） | “便宜的那个” | 提出候选 token 的小模型；通常比验证模型便宜 10–50 倍。 |
| 验证模型（verifier） | “大的那个” | 我们要保持其分布的目标模型；每个投机步运行一次。 |
| 接受率（α） | “草稿多常猜对” | 验证模型接受该草稿 token 的每 token 概率。典型值 0.7–0.9。 |
| 残差分布（residual distribution） | “被拒绝时的兜底” | 归一化后的 `(q - p)_+`；从该分布采样可在拒绝时保持验证模型分布。 |
| 奖励 token（bonus token） | “免费送的那个” | 当全部 N 个草稿都被接受时，从验证模型的下一步分布中多采一个 token。 |
| Medusa | “无草稿模型的投机解码” | 在验证模型上增加多个 LM 头，并行预测位置 t+1..t+k。 |
| EAGLE | “基于隐藏状态的草稿” | 以验证模型最后一层隐藏状态为条件的微型 transformer 草稿模型。 |
| Lookahead 解码 | “Jacobi 迭代” | 用不动点迭代进行自投机解码；无需草稿模型。 |
| 树形注意力（tree attention） | “一次性验证多个候选” | 分支式验证，同时考虑若干草稿延续。 |
| KV 回滚（KV rollback） | “撤销被拒绝的草稿” | 临时 KV 缓冲区；接受则提交，拒绝则丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 核心算法与等价性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) —— 同时期提出；清晰的 Bernoulli 拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— Medusa 论文；树形注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— EAGLE-1；基于隐藏状态的草稿模型。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) —— EAGLE-2；动态草稿树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) —— EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) —— Lookahead，无草稿模型方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 生产级 canonical 参考，四种策略均已集成。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) —— EAGLE-1/2/3 的参考实现。
