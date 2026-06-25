# 投机解码（Speculative Decoding）与 EAGLE-3

> Phase 7 · Lesson 16 已经用数学证明了：Leviathan 拒绝规则能够精确保持验证器（verifier）的分布。本课从训练栈视角审视 2026 年生产环境下的投机解码。EAGLE-3 把草稿模型（draft model）从廉价的近似模型，改造成基于验证器自身隐藏状态训练的专用小型网络，并加入了训练时测试（training-time test）循环，使其训练分布与推理分布对齐。结果是：端到端加速比 3× 到 6.5×，对话场景下逐 token 接受率（acceptance rate）超过 0.9，且没有任何分布上的权衡。2026 年的每一款生产推理栈都默认搭载它。

**类型：** 实战构建
**语言：** Python（标准库）
**前置知识：** Phase 7 · 16（投机解码的数学原理）、Phase 10 · 12（推理优化）
**预计用时：** 约 75 分钟

## 学习目标

- 用一句话陈述 Leviathan 定理，并证明投机循环生成的样本与验证器（verifier）同分布。
- 梳理从原始投机解码（vanilla spec-decoding，Leviathan 2023）到 EAGLE、EAGLE-2 再到 EAGLE-3 的两年演进，并指出每一步消除了什么具体限制。
- 根据接受率 `α` 与草稿/验证器成本比 `c` 计算期望加速比，并为不同场景选择最优草稿长度（draft length）`N`。
- 从零实现完整的投机循环：起草、验证、从残差分布中拒绝采样、在拒绝时回滚 KV 缓存、在全部接受时发出奖励 token（bonus token）。

## 问题背景

70B 模型在 H100 上的自回归解码（autoregressive decoding）大约只有 35 token/秒。GPU 远未达到饱和。内存带宽是瓶颈：每个 token 都要从 HBM 加载 70B 权重，做一次运算，产出一个浮点数。计算单元大部分时间处于空闲状态。

投机解码把这个问题变成一个你真正能够解决的吞吐问题。一个廉价的草稿模型（draft model）用 `N` 次小的前向传播提出 `N` 个 token。验证器（verifier）在 prefix 加上全部 `N` 个草稿上只运行一次。如果验证器在第 `i` 个位置上的分布与草稿一致（我们将在统计意义上精确化这一点），则接受；否则拒绝，并从残差分布（residual distribution）中采样一个修正 token。一次大模型前向传播最多可产出 `N+1` 个被接受的 token，而非仅 1 个。

关键的定理来自 Leviathan、Kalman、Matias（ICML 2023）：输出分布与直接从验证器采样得到的分布完全相同。不是近似，而是完全相同。这正是投机解码能在生产环境中被接受的全部原因——它是一种纯粹的无质量损失的延迟优化。

Phase 7 · Lesson 16 给了你数学。本课给你的是训练栈。一个好的草稿模型带来的加速比，比廉价草稿高出约 2 倍。EAGLE、EAGLE-2 和 EAGLE-3（Li 等人，2024–2025）把“草稿 = 同模型的小版本”变成了一门精确的工程学科。2026 年的生产推理服务器默认采用 EAGLE-3。

## 核心概念

### 不变量：Leviathan 拒绝采样

设 `p(t)` 为给定某个前缀后草稿对下一个 token 的分布，`q(t)` 为验证器（verifier）的分布。从 `p` 中采样一个草稿 token `d ~ p`。以概率 `min(1, q(d) / p(d))` 接受它。若拒绝，则从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 中采样。最终得到的样本服从 `q`。无论 `p` 有多差都成立——它越差，拒绝越频繁，但输出依然精确。

将 `N` 个这样的调用串联起来，用一次验证器前向传播处理 `prefix + d_1 + ... + d_N`。验证器同时返回 `q_1, q_2, ..., q_{N+1}`。从左到右遍历。在第 `j` 个位置首次拒绝时，从 `residual(q_j, p_j)` 中采样并停止。若全部接受，则从 `q_{N+1}` 中采样一个奖励 token（bonus token）。

### 什么决定了加速比

设 `α` 为每个草稿 token 的期望接受率（acceptance rate）。设 `c = cost(draft) / cost(verifier)` 为成本比。每次验证器前向传播的期望接受 token 数为：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个被接受 token 的期望总墙上时间为 `(N * c + 1) / E[accepted]`。对 `N` 最小化该式即可得到甜点。当 `α = 0.8`、`c = 0.05` 时：最优 `N` 约为 5–7，加速比约 3.2×。当 `α = 0.95`、`c = 0.02` 时：最优 `N` 约为 8–10，加速比接近 5×。

最大的单一杠杆是 `α`。在固定 `N = 5` 的情况下，把 `α` 从 0.6（原始草稿）提升到 0.9（EAGLE-3），每次验证器前向传播的期望接受 token 数会从 2.2 增加到 4.1。在验证器不变的情况下，吞吐几乎翻倍。

### 两年演进

**原始投机解码（Vanilla Speculative，Leviathan，2023）。** 草稿模型是一个独立训练的、同系列的小型 LLM。易于接入，`α ≈ 0.6`，最好情况下加速比约 2×。

**EAGLE-1（Li 等人，2024）。** 草稿是一个极小的 Transformer——通常只有一到两层——以验证器最后一层的隐藏状态（hidden state）为输入，直接预测下一个 token。因为草稿能看到验证器的特征表示，所以它的分布更接近验证器。`α` 提升到 0.7–0.8。

**EAGLE-2（Li 等人，2024）。** 增加了动态草稿树（dynamic draft tree）：不再提出单个 `N` token 序列，而是提出一棵小型候选树，用一次验证器前向传播（树注意力，tree attention）为每个节点打分，然后选择概率最高的路径。每步的草稿长度变得自适应。每条被接受路径上的 token 接受率 `α` 攀升至 0.85 以上。

**EAGLE-3（Li 等人，2025，NeurIPS）。** 又有两处改动。首先，彻底舍弃特征预测损失（feature-prediction loss）——EAGLE-1/2 训练草稿去匹配验证器的隐藏状态，这限制了数据带来的收益上限。EAGLE-3 直接在 token 预测上训练。其次，训练时测试（Training-Time Test，TTT）：在草稿训练过程中，把草稿自己之前的预测结果作为后续多步的输入，与其在推理时的运行方式一致。这让训练与测试分布对齐，阻止误差累积。实测加速比：对话场景最高 6.5×，在 H100 上使用 SGLang、batch 64 时吞吐提升 38%。

### KV 缓存回滚

验证过程会在一次前向传播中把验证器的 KV 缓存（KV cache）扩展 `N` 个条目。如果在第 `j` 个位置发生拒绝，那么 `j-1` 之后的缓存内容就不再有效。两种常见实现：写入临时缓冲区并在接受后提交（vLLM、TensorRT-LLM），或者维护一份物理 KV 缓存加一个逻辑长度，在拒绝时截断。无论哪种方式，回滚成本都是每层每头若干字节，与前向传播成本相比可以忽略。

对于 EAGLE-2 的树搜索，验证器使用非因果掩码（non-causal mask）运行注意力，该掩码尊重树的拓扑结构。工程实现比较琐碎，但计算上只是一个带自定义掩码的标准 FlashAttention 调用。

### 2026 年的草稿架构

| 策略 | 草稿类型 | `α` | 加速比 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| 原始投机解码 | 独立的小型 LLM | 0.55-0.70 | 1.8-2.3× | 无（复用已有小模型） |
| Medusa | 验证器上的额外 LM 头 | 0.65-0.75 | 2-3× | 约 10 亿 SFT token |
| EAGLE-1 | 基于隐藏状态的一层 Transformer | 0.70-0.80 | 2.5-3× | 约 600 亿 token |
| EAGLE-2 | EAGLE-1 + 动态草稿树 | 0.80-0.88 | 3-4× | 约 600 亿 token |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | 约 600–2000 亿 token |
| Lookahead | 无草稿（Jacobi 迭代） | 不适用 | 1.3-1.6× | 无 |

在 2026 年的生产环境中：vLLM 和 SGLang 在有 EAGLE-3 时默认使用它，否则退回到 EAGLE-2。TensorRT-LLM 为 Meta 和 NVIDIA 的公开模型提供了最快的 Medusa 路径。llama.cpp 在 CPU 部署中提供原始投机解码。

## 动手构建

参见 `code/main.py`。这是完整的 Leviathan 投机循环，包含所有组件：N 个草稿、验证器并行传播、逐位置拒绝、残差采样、奖励 token、KV 缓存回滚，以及用经验验证输出分布与直接从 `q` 采样一致。

### 步骤 1：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 步骤 2：残差分布

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 步骤 3：完整的投机步

`spec_step` 函数从 `p` 起草 `N` 个 token，然后在一个并行的 `q` 评估中全部验证。对每个草稿 token 应用拒绝规则，首次拒绝时从残差中采样修正 token。如果全部被接受，则从 `q_{N+1}` 发出一个奖励 token（bonus token）。

### 步骤 4：KV 回滚的簿记

模拟器为每个 worker 维护一个逻辑 `kv_length`。当接受 `k` 个草稿时，`kv_length += k`。当在第 `j` 个位置拒绝时，缓存其实已经写到了 `j` 之后，但逻辑长度被设为 `prefix_length + j + 1`，即修正 token 之后一位。后续读取会截断到逻辑长度。

### 步骤 5：Leviathan 校验

运行 50,000 次投机步。统计被接受 token 的经验分布。与 50,000 次直接从 `q` 采样的结果进行比较。卡方统计量应远低于临界值。该定理在实践中成立。

### 步骤 6：加速比随 `α` 的变化

通过在不同幅度上把 `p` 从 `q` 扰动开来，扫描草稿质量。测量 `α`，然后绘制每次验证器调用的期望 token 数关于 `α` 和 `N` 的曲线。代码会打印一张表，展示 EAGLE-3 级别的草稿质量（`α ≈ 0.9`）如何让每次验证器调用解锁 4–5 个 token。

## 应用

使用 EAGLE-3 的生产级 `vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

根据 EAGLE-3 论文，在 H100 上使用 SGLang、batch 64 并开启 EAGLE-3，吞吐比 batch 64 的原始解码大约高 1.38 倍。

何时使用投机解码：

- 任何 p50 延迟比峰值吞吐更重要的交互式对话负载。
- 代码生成与结构化输出（JSON、SQL）。由于目标分布高度可预测，`α` 通常在 0.9 以上。
- 长文本生成（数千 token）。摊销后的加速收益持续累积。

何时不宜使用：

- 非常小的模型（< 3B）。草稿并不比验证器便宜多少。
- 极小的 batch-1 CPU 部署。草稿模型的内存开销可能得不偿失。
- 温度很高、创意性很强的采样场景，此时 `α` 会崩溃。

## 交付

本课产出 `outputs/skill-eagle3-tuner.md`。给定一个推理负载（模型、batch 大小、目标延迟、任务画像），它会推荐一种投机解码策略与调参（草稿家族、`N`、树深度、温度感知切换）。

## 练习

1. 运行 `code/main.py`。确认在 50,000 个样本上，Leviathan 分布校验的卡方统计量低于 95% 临界值。

2. 固定 `α = 0.9`、`c = 0.04`，让 `N` 从 1 扫描到 10。绘制每次验证器调用的期望 token 数，以及每个 token 的实际墙上时间。找出使墙上时间最小的 `N`，并解释曲线形状。

3. 修改代码以模拟 EAGLE-2 树搜索：每步草稿提出形状为 `[2, 2, 2]` 的树（八条候选路径）。验证器运行一次，概率最高的被接受路径胜出。计算每个叶节点的 `α` 与每次验证器调用的总 token 数，并与同等计算量下的线性链投机解码进行比较。

4. 为两个并发序列实现一个批处理 KV 缓存回滚模拟器。序列 A 全部草稿被接受；序列 B 在第 2 个位置拒绝。证明每个序列的 `kv_length` 都被正确更新，且没有浪费计算。

5. 阅读 EAGLE-3 论文的第 4 节（Training-Time Test）。用两句话解释：为什么缺少 TTT 的朴素草稿训练会遭受暴露偏差（exposure bias），以及为什么在训练中把草稿自己的预测喂给它可以修复该问题。将其与 seq2seq 中的计划采样（scheduled sampling）文献联系起来。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Leviathan 规则 | “min(1, q/p)” | 以概率 `min(1, q(d)/p(d))` 进行 Bernoulli 接受/拒绝；若拒绝时从残差分布采样，则精确保持验证器分布 |
| 残差分布 | “(q−p) 取正后归一化” | 将 `(q - p)_+` 在零处截断并重新归一化——拒绝时应采样的正确分布 |
| 接受率 α | “草稿有多准” | 在拒绝规则下每个 token 的 Bernoulli 成功概率期望；决定所有加速比计算 |
| EAGLE-1 | “隐藏状态草稿” | 以验证器最后一层隐藏状态为条件的小型 Transformer 草稿（Li 等人，2024） |
| EAGLE-2 | “动态草稿树” | EAGLE-1 加上一棵候选延续树，在一次验证器传播中用树注意力打分 |
| EAGLE-3 | “训练时测试” | 舍弃特征预测损失，直接在 token 预测上训练，并在训练时把草稿自己的输出喂给它 |
| 训练时测试（TTT） | “暴露偏差修复” | 在训练时自回归地运行草稿，使训练与测试输入分布匹配——与计划采样（scheduled sampling）直接对应 |
| KV 回滚 | “撤销被拒绝的草稿” | 在被拒绝后将验证器的 KV 缓存重置回已接受前缀长度的簿记操作 |
| 奖励 token | “免费送的那个” | 当全部 `N` 个草稿被接受时，从 `q_{N+1}` 免费多采样一个 token |
| 树注意力 | “一次性验证多个候选” | 使用尊重草稿树拓扑的非因果掩码的注意力；一次前向传播计算树中每个节点的 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 奠基性论文与等价定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — 同期独立提出，证明清晰
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1，基于隐藏状态的草稿
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 动态树搜索
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — 2026 年生产环境默认方案
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — 另一种无草稿方案
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 生产环境权威参考，已集成所有策略
