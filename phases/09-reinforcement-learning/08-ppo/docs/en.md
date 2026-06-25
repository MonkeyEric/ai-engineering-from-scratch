# 近端策略优化（Proximal Policy Optimization, PPO）

> A2C 每完成一次更新就把 rollout 丢弃。PPO 用带裁剪的重要性比率（importance ratio）包裹策略梯度（policy gradient），让你能在同一批数据上训练 10 多个 epoch，同时避免策略爆炸。Schulman 等（2017）。到 2026 年仍是默认的策略梯度算法。

**类型：** Build
**语言：** Python
**前置知识：** Phase 9 · 06（REINFORCE）、Phase 9 · 07（Actor-Critic）
**时间：** ~75 分钟

## 问题所在

A2C（第 07 课）是 on-policy 算法：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据必须从*当前*的 `π_θ` 采样。完成一次更新后，`π_θ` 发生变化，之前用过的数据就变成 off-policy。复用这些数据会让梯度产生偏差。

Rollout 的开销很大。在 Atari 上，8 个环境 × 128 步的一次 rollout = 1024 条转移，需要十几秒的环境时间。只训练一步就扔掉太浪费了。

信任区域策略优化（Trust Region Policy Optimization，TRPO，Schulman 2015）是第一个修复方案：约束每次更新，使新旧策略之间的 KL 散度保持在 `δ` 以下。理论优雅，但每一步都需要共轭梯度求解。2026 年已经没人跑 TRPO 了。

PPO（Schulman 等 2017）用一个简单的裁剪目标函数替代了硬信任区域约束。只多一行代码。每个 rollout 训练十个 epoch。无需共轭梯度。理论保证足够好。九年之后，它仍是从 MuJoCo 到 RLHF 的默认策略梯度算法。

## 核心概念

![PPO 裁剪代理目标：在 1 ± ε 处裁剪比率](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据时的旧策略的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取 `a_t` 的概率是旧策略的两倍。

**裁剪代理目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势（advantage）`A_t > 0` 且比率试图超过 `1 + ε`，裁剪会拉平梯度——不要把好动作的概率推到旧概率的 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图超过 `1 - ε`（意味着我们会增加坏动作的概率，而不是按裁剪后的方向降低它），裁剪会限制梯度——不要把坏动作推到 `-ε` 以下。

`min` 处理另一个方向：如果比率已经朝*有利*方向移动，你仍然会得到梯度（不会在你受损的那一侧被裁剪）。

典型值 `ε = 0.2`。把目标函数画成 `r_t` 的函数：在“好的一侧”有平顶屋顶，在“坏的一侧”有平底地板的分段线性函数。

**完整 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的演员-评论家（actor-critic）结构。三个系数，通常 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中各收集 `T` 步，共 `N × T` 条转移。
2. 计算优势（GAE），并把它们冻结为常数。
3. 冻结 `π_{θ_old}` 作为当前 `π_θ` 的快照。
4. 对于 `K` 个 epoch，对每个 minibatch `(s, a, A, V_target, log π_old(a|s))`：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度更新。
5. 丢弃 rollout，回到步骤 1。

`K = 10`、minibatch 大小为 64 是一组标准超参数。PPO 很鲁棒：具体数值在 ±50% 范围内通常影响不大。

**KL 惩罚变体。** 原始论文提出了另一种自适应 KL 惩罚方法：`L = L^{PG} - β · KL(π_θ || π_old)`，并根据观测到的 KL 调整 `β`。裁剪版本成为主流；KL 变体在 RLHF 中仍有使用（因为 KL 到参考模型本身就是始终需要的独立约束）。

## 动手实现

### 步骤 1：在 rollout 时记录 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照只拍一次，在 rollout 时完成。在后续更新 epoch 中不再改变。

### 步骤 2：计算 GAE 优势（第 07 课）

与 A2C 相同。对整个批次做归一化。

### 步骤 3：裁剪代理更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播 -surrogate，加上价值损失，减去熵
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 已裁剪
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“被裁剪 → 梯度为零” 的模式是 PPO 的核心。如果新策略已经朝有利方向漂移太远，更新就停止。

### 步骤 4：价值损失与熵

给评论家目标加上标准 MSE，给演员策略加上熵奖励，与 A2C 相同。

### 步骤 5：诊断指标

每次更新要关注三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **裁剪比例（clip fraction）**——比率落在 `[1-ε, 1+ε]` 之外的样本比例。应在 `~0.1-0.3`。如果 `~0`，说明裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`；如果 `~0.5+`，说明你在过拟合 rollout → 降低它们。
- **解释方差（explained variance）** `1 - Var(V_target - V_pred) / Var(V_target)`。评论家质量指标。随着评论家学习，应逐渐接近 1。

## 常见陷阱

- **裁剪系数调错。** `ε = 0.2` 是事实上的标准。`0.1` 会让更新过于保守；`0.3+` 容易导致不稳定。
- **epoch 过多。** `K > 20` 通常会让训练不稳定，因为策略会远离 `π_old`。尤其对大型网络，要限制 epoch 数。
- **没有奖励归一化。** 过大的奖励尺度会侵蚀裁剪范围。在计算优势前，对奖励做归一化（运行标准差）。
- **忘记优势归一化。** 每批次零均值/单位方差归一化是标准做法。跳过它会让 PPO 在大多数 benchmark 上表现糟糕。
- **学习率不衰减。** PPO 受益于线性衰减到零的学习率。恒定学习率通常更差。
- **重要性比率计算错误。** 为了数值稳定，始终用 `exp(log_new - log_old)`，而不是 `new / old`。
- **梯度符号弄反。** 最大化代理目标 = *最小化* `-L^{CLIP}`。符号翻转是最常见的 PPO bug。

## 应用场景

PPO 在 2026 年是众多出奇广泛的领域的默认 RL 算法：

| 使用场景 | PPO 变体 |
|----------|----------|
| MuJoCo / 机器人控制 | 高斯策略（Gaussian policy）PPO，GAE(0.95) |
| Atari / 离散游戏 | 类别策略（categorical policy）PPO，滚动 128 步 rollout |
| LLM 的 RLHF | 带参考模型 KL 惩罚的 PPO，在回复末尾从奖励模型（RM）获得奖励 |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理 LLM | GRPO（第 12 课）—— 无评论家的 PPO 变体 |
| 仅有偏好数据 | DPO —— 把 PPO+KL 折叠成闭式解，无需在线采样 |

PPO 的*损失形态*——裁剪代理 + 价值 + 熵——是 DPO、GRPO 以及几乎所有 RLHF 流水线的脚手架。

## 交付

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: 针对给定环境和训练预算，输出 PPO 训练配置和诊断计划。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定环境和训练预算，输出：

1. Rollout 大小。`N` 个环境 × `T` 步。
2. 更新计划。`K` 个 epoch、minibatch 大小、LR 计划。
3. 代理参数。`ε`（裁剪）、`c_v`、`c_e`、开启优势归一化。
4. 优势。GAE(`λ`)，明确写出 `γ` 和 `λ`。
5. 诊断计划。KL、裁剪比例、解释方差的阈值与告警。

拒绝 `K > 30` 或 `ε > 0.3`（不安全的信任区域）。拒绝任何没有优势归一化或没有 KL/裁剪监控的 PPO 运行。持续裁剪比例高于 0.4 标记为漂移。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，`ε=0.2, K=4`。在相同环境步数下与 A2C（每个 rollout 一个 epoch）比较样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报（return）与环境步数的关系，并跟踪每次更新的平均 KL。在这个任务上，`K` 多大时 KL 会爆炸？
3. **困难。** 把裁剪代理替换为自适应 KL 惩罚（若 `KL > 2·target` 则 `β` 翻倍，若 `KL < target/2` 则 `β` 减半）。比较最终回报、稳定性和无裁剪程度。

## 关键术语

| 术语 | 大家怎么叫 | 实际含义 |
|------|-----------|---------|
| 重要性比率（Importance ratio） | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`；偏离收集数据时的策略的程度。 |
| 裁剪代理目标（Clipped surrogate） | "PPO 的核心技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利侧超过裁剪后梯度变平。 |
| 信任区域（Trust region） | "TRPO / PPO 的意图" | 限制每次更新的 KL，以保证单调改进。 |
| KL 惩罚（KL penalty） | "软信任区域" | PPO 的另一种形式：`L - β · KL(π_θ || π_old)`。自适应 `β`。 |
| 裁剪比例（Clip fraction） | "裁剪触发的频率" | 诊断指标 —— 应为 0.1-0.3；超出说明调参不当。 |
| 多 epoch 训练（Multi-epoch training） | "数据复用" | 每个 rollout 训练 K 个 epoch；用方差代价换取样本效率。 |
| 类 on-policy（On-policy-ish） | "基本算 on-policy" | PPO 名义上是 on-policy，但 K>1 的 epoch 会安全地使用轻微 off-policy 数据。 |
| PPO-KL | "另一种 PPO" | KL 惩罚变体；用于 KL 到参考模型本身已是约束的 RLHF 场景。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) —— 原始论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) —— TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) —— 对每一个 PPO 超参数做了消融。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— InstructGPT；RLHF 中的 PPO 配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) —— 干净的现代 PyTorch 讲解。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) —— 被许多论文引用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) —— 语言模型 PPO 的生产配方；与第 09 课（RLHF）一起阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) —— “37 项代码级优化”论文；哪些 PPO 技巧是承重墙，哪些只是 folklore。
