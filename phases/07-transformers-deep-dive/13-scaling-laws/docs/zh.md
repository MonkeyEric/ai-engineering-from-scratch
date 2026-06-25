# 缩放定律（Scaling Laws）

> 2020 年 Kaplan 论文说：模型越大，损失越低。2022 年 Hoffmann 论文说：你训练得不够。计算主要投入两个方向——参数量与 token 数——而如何分配并不显而易见。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer）、Phase 7 · 07（GPT）
**时间：** 约 45 分钟

## 问题

当你拥有 C FLOPs 的训练算力并希望获得最佳模型时，你面前有两个旋钮：

1. **参数量（N）多大？** 模型越大，容量越高。
2. **训练 token 数（D）多少？** 数据越多，容量利用越充分。

FLOPs 大致按 `6 × N × D` 缩放。你可以把 N 推高、D 压低，也可以把 D 推高、N 压低。哪种更好？

在 2022 年之前，答案是“拼命推 N”。GPT-3（2020）有 175B 参数，训练了约 300B token，比率约为每个参数 1.7 个 token。Kaplan 缩放定律支持这一做法。

Hoffmann 等人（2022）训练了一个名为 Chinchilla 的小模型族，发现了不同结论：最优比率接近 **每个参数 20 个 token**。GPT-3 的训练量少了 10 倍。Chinchilla（70B 参数、1.4T token）在每个基准测试上都击败了 GPT-3（175B 参数、300B token），而推理成本仅为其 2.5 分之一。

2026 年是 Chinchilla 的世界——但有一个重要转折。Llama 3 8B 在 15 万亿 token 上训练，比率达到每个参数 1875 个 token，是过去 Chinchilla 最优值的 94 倍。对于将大规模使用的模型，推理成本比训练成本更重要，因此过度训练（超过 Chinchilla 最优）以换取更小的可部署 footprint，已成为 2026 年的默认选择。

## 概念

![Chinchilla 曲线：不同 N/D 比率下损失随计算量的变化](../assets/scaling-laws.svg)

### Hoffmann 定律

根据 Chinchilla 论文，损失满足：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数数量（非嵌入参数）。
- `D` = 训练 token 数。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约损失上限。
- `A ≈ 406`，`B ≈ 411`。

随着规模变化，这两项彼此权衡。在固定计算量（C = 6ND）下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

计算最优：每个参数约 20 个 token。

### 为何仍要过度训练

Chinchilla 最优最小化的是每个训练 FLOP 对应的训练损失。但训练成本只付一次，推理成本却永远持续。

对于一个每月服务一万亿 token 的聊天机器人，推理 dominates 总成本。Llama 的做法是：模型更小、训练更久。8B 参数训练 15T token 是深度推理优化的：

- 可放入消费级 GPU。
- 延迟仅为 70B Chinchilla 最优模型的一小部分。
- 对大多数任务而言，质量足够接近。

DeepMind 2024 年的论文（“Over-training is the new optimal”）将这一做法形式化。对于推理主导的工作负载，正确比率更接近每个参数 100–500 个 token，具体取决于服务量。

### 涌现 vs 平滑

有人认为某些能力（算术、多步推理、思维链跟随）会在某个规模“突然涌现”。

Schaeffer 等人（2023）认为这是一种测量伪影：涌现指标使用不连续评分（精确匹配、阈值准确率），掩盖了底层 logits 的平滑提升。连续指标（交叉熵）则呈现平滑曲线。

2026 年的共识是：通过连续损失进行预测是可靠的。基准测试的跳跃往往是评分器伪影。请用连续指标来规划预算。

### 2026 年的图景

缩放定律仍然成立，但：

| 因素 | 变化方式 |
|------|----------|
| 数据质量 | 策划“优质”token（Phi 风格）可使有效算力提升 >2 倍 |
| MoE | 总参数量与激活 FLOP 解耦；按每激活 FLOP 应用缩放定律 |
| 后训练 | 某些能力（指令跟随、代码）更多受 SFT+RLHF 影响，而非预训练 |
| 多模态 | 图像与文本 token 共同缩放；每种模态有独立曲线 |
| 合成数据 | 模型自身生成训练数据；有效算力可复合增长 |

Muon 优化器（Kimi Moonlight，2024）在相同数据下相比 AdamW 实现了约 2 倍有效算力提升。2026 年的一些训练运行已默认使用 Muon。它改变了缩放定律中的绝对常数，而非形状。

## 动手实现

参见 `code/main.py`。我们实现 Chinchilla 损失方程，并在多个计算预算下求解计算最优的 `(N, D)`。

### 步骤 1：Chinchilla 损失

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 `C = 6ND` 下，将 `L` 作为 `(N, D)` 的等高线图绘制出来，找到最小值。

### 步骤 2：计算最优前沿

对于从 `1e17` 到 `1e25` FLOPs 的计算预算，在约束 `6ND = C` 下找到使损失最小的 `(N, D)`。验证比率 `D/N ≈ 20`。

### 步骤 3：过度训练成本

计算将模型缩小 10 倍（最优 N 的 1/10，最优 D 的 10 倍）所额外支付的损失。报告由此节省的推理 FLOP（与 N 成正比）。

### 步骤 4：与真实模型对比

填入 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（激活参数）等已知 `(N, D)` 组合，比较预测损失与实际报告损失。

## 应用

你不太可能亲自训练前沿模型。但缩放定律能告诉你：

1. **你的微调数据是否充足。** 如果任务特定数据低于基础模型每参数 20 个 token，预计损失会在某个下限处饱和。
2. **是否应选择更大的基础模型。** 如果你把所有预算都花在推理上，更倾向于选择更小、训练更久的模型。
3. **收益何时递减。** 超过 Chinchilla 最优 1000 倍后，对数损失变化将变成噪声。

**2026 年的研究轨迹：**

- **数据受限 regime。** 网络上高质量 token 有限（过滤后英文约 5–10 万亿）。前沿预训练正接近这一上限。合成数据、多语言、多模态以及大规模 RLHF 微调是下一个杠杆。
- **算力倍增技巧。** Muon 优化器、MoE、更好的数据策划——每一项都会移动绝对常数，而非渐近线。
- **强化学习的缩放定律。** 开放问题。早期证据表明 RL 样本上存在幂律，但指数与预训练非常不同。

## 交付

参见 `outputs/skill-training-budget-estimator.md`。该技能会根据计算预算、部署约束和目标损失，为新训练运行选择 `(N, D, 小时, GPU)`。

## 练习

1. **简单。** 运行 `code/main.py`。打印计算预算为 `1e20`、`1e22`、`1e24` 时的 Chinchilla 最优 `(N, D)`，并与真实模型表对比。
2. **中等。** 实现 Hoffmann 损失关于计算量的曲线。绘制计算最优前沿下损失随 `log10(C)` 的变化。找出该定律预测需要 `>10^28` FLOPs 才能使交叉熵再降低 0.1 的位置。
3. **困难。** 在 5 个微小模型（100K 到 10M 参数）上拟合你自己的缩放定律，这些模型在同一数据集上训练。估计 `α` 和 `E`。你的指数与已发表结果有多吻合？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 参数（Parameters，N） | “模型大小” | 非嵌入权重数量；决定容量。 |
| Token（D） | “训练数据” | 见过的训练 token 数量；决定参数被利用的程度。 |
| 计算量（Compute，C） | “花了多少 FLOP” | 标准 transformer 近似为 `6 × N × D`。 |
| Chinchilla 最优 | “D/N ≈ 20” | 使预训练每 FLOP 损失最小的比率。 |
| 过度训练（Over-training） | “超过 Chinchilla” | 花额外训练 FLOP 来节省推理 FLOP；D/N >> 20。 |
| 不可约损失（Irreducible loss） | “地板” | 缩放定律中的 `E` 项；数据本身的熵。 |
| 涌现能力（Emergent capability） | “规模达到后突然跃升” | 通常是评分器伪影；连续损失是平滑的。 |
| 有效算力（Effective compute） | “训练效率倍增器” | 更好的数据 / 优化器 / 架构能让每个 FLOP 走得更远。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) —— 第一篇缩放定律论文；训练不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) —— Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) —— 涌现作为测量伪影。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) —— 为何 Llama 的过度训练适合其工作负载。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) —— 2 倍算力倍增器。
