# 时序差分 —— Q-Learning 与 SARSA

> 蒙特卡洛（Monte Carlo）必须等到一个回合（episode）结束才能更新。时序差分（Temporal Difference，TD）则通过自举（bootstrapping）下一个价值估计，在每一步之后更新。Q-learning 是离策略（off-policy）且乐观的；SARSA 是在策略（on-policy）且谨慎的。两者都只需一行代码。本阶段后续所有深度强化学习（deep-RL）方法都建立在这两者之上。

**类型：** 动手实践
**语言：** Python
**前置知识：** Phase 9 · 01（马尔可夫决策过程，MDP）、Phase 9 · 02（动态规划，Dynamic Programming）、Phase 9 · 03（蒙特卡洛，Monte Carlo）
**时间：** 约 75 分钟

## 问题所在

蒙特卡洛（Monte Carlo）可以工作，但它有两个昂贵的需求：它需要能够终止的回合，并且只有在最终回报（return）已知后才更新。如果你的回合长达 1,000 步，MC 就要等 1,000 步才更新任何内容。它是高方差、低偏差，并且在实践中很慢。

动态规划（Dynamic Programming，DP）则完全相反 —— 零方差的自举备份 —— 但它需要一个已知的模型。

时序差分（Temporal Difference，TD）学习介于两者之间。从单次转移 `(s, a, r, s')` 出发，构造单步目标 `r + γ V(s')`，并将 `V(s)` 向它靠近。无需模型。无需完整回合。代价是在等式右侧使用近似的 `V` 会带来偏差，但方差远低于 MC，并且从第一步起就能在线更新。

现代强化学习 —— DQN、A2C、PPO、SAC —— 都是围绕这个支点转动的。Phase 9 的其余内容就是在这个单步 TD 更新之上叠加函数近似和各种技巧，而你会在本课写下这个更新。

## 核心概念

![Q-learning 与 SARSA 对比：离策略 max 与在策略 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号中的量就是 TD 误差（TD error）`δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线对应物。收敛需要满足 Robbins-Monro 条件的学习率 `α`（`Σ α = ∞`，`Σ α² < ∞`），并且所有状态都被无限次访问。

**Q-learning（Q-learning）。** 一种离策略（off-policy）TD 控制方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

这里的 `max` 假设从 `s'` 开始将遵循*贪心（greedy）*策略，无论智能体实际采取什么动作。这种解耦使得 Q-learning 能够在通过 ε-贪心（ε-greedy）探索的同时学习 `Q*`。Mnih 等人（2015）将其转化为 Atari 上的深度 Q-learning（第 05 课）。

**SARSA。** 一种在策略（on-policy）TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

这个名字来自元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*采取的下一个动作 `a'`，而不是贪心的 `argmax`。它会收敛到当前运行的 ε-贪心策略 `π` 对应的 `Q^π`，而当 `ε → 0` 时，`Q^π` 就变成 `Q*`。

**悬崖行走（Cliff Walking）中的差异。** 在经典的悬崖行走任务中（掉下悬崖奖励 -100），Q-learning 会学习沿着悬崖边缘的最优路径，但在探索过程中偶尔会付出惩罚。SARSA 则会学习一条离悬崖一步远的安全路径，因为它把探索噪声纳入了 Q 值。经过训练，当 `ε → 0` 时两者都会达到最优。在实践中这很重要：当部署阶段确实存在探索时，SARSA 的行为更保守。

**期望 SARSA（Expected SARSA）。** 用 `π` 下 `Q(s', a')` 的期望值替代它：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

方差比 SARSA 更低（不需要对 `a'` 采样），目标仍然是在策略的。现代教科书中常常把它作为默认方法。

**n 步 TD 与 TD(λ)。** 通过在自举前等待 `n` 步，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 以几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 取平均。大多数深度 RL 使用 3 到 20 之间的 `n`。

## 动手实现

### 第 1 步：基于 ε-贪心策略的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。*唯一*与 Q-learning 不同的地方就是目标值那一行。

### 第 2 步：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号就是在策略与离策略之间的区别。

### 第 3 步：学习曲线

记录每 100 个回合的平均回报。Q-learning 在简单的确定性网格世界（GridWorld）上收敛更快；SARSA 在悬崖行走中更保守。在 `code/main.py` 中的 4×4 GridWorld 上，使用 `α=0.1, ε=0.1` 训练约 2,000 个回合后，两者都接近最优。

### 第 4 步：与 DP 真实值对比

运行值迭代（Value Iteration，第 02 课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格型 TD 智能体在 4×4 GridWorld 上训练 10,000 个回合后，误差通常在 `~0.5` 以内。

## 常见陷阱

- **初始 Q 值很重要。** 乐观初始化（对负奖励任务设 `Q = 0`）鼓励探索。悲观初始化可能让贪心策略永远被困住。
- **α 调度。** 对于非平稳问题，恒定 `α` 是可行的。理论上按 `α_n = 1/n` 衰减可以保证收敛，但实践中太慢 —— 把 `α` 固定在 `[0.05, 0.3]` 之间并监控学习曲线。
- **ε 调度。** 从高 `ε=1.0` 开始，衰减到 `ε=0.05`。"GLIE"（极限下贪婪且无限探索，Greedy in the Limit with Infinite Exploration）是收敛条件。
- **Q-learning 中的最大化偏差（maximization bias）。** 当 `Q` 有噪声时，`max` 算子会向上偏。这会导致过高估计 —— Hasselt 提出的 Double Q-learning（第 05 课的 DDQN 使用）通过维护两张 Q 表来解决这个问题。
- **非终止回合。** TD 可以在没有终止状态的情况下学习，但你需要限制步数，或在限制处正确处理自举。标准做法：把步数上限视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，请使用可哈希的键（用 tuple 而不是 list；用四舍五入后的浮点数元组，而不是原始值）。

## 如何应用

2026 年的 TD 方法概览：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 在策略安全关键场景 | SARSA / Expected SARSA | 探索过程中更保守。 |
| 高维状态 | DQN（Phase 9 · 05） | 带经验回放和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（Phase 9 · 07） | 在 Q 网络上做 TD 更新；策略网络输出动作。 |
| LLM 强化学习（基于奖励模型） | PPO / GRPO（Phase 9 · 08, 12） | 通过 GAE 得到类 TD 优势值的演员-评论家（actor-critic）方法。 |
| 离线强化学习 | CQL / IQL（Phase 9 · 08） | 带保守正则化的 Q-learning。 |

2026 年你在论文中读到的 "RL"，百分之九十都是 Q-learning 或 SARSA 的某种变体。在深入学习之前，先用手指记住这个表格型更新。

## 交付成果

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: 为表格型或小型特征型强化学习任务选择 Q-learning、SARSA、Expected SARSA。
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定一个表格型或小型特征型环境，输出：

1. 算法。Q-learning / SARSA / Expected SARSA / n-step 变体。一句话说明理由，关联在策略/离策略与方差。
2. 超参数。α、γ、ε 及其衰减调度。
3. 初始化。Q_0 取值（乐观 vs 零）及理由。
4. 收敛诊断。目标学习曲线，如果可做 DP 则检查 `|Q - Q*|`。
5. 部署注意事项。推理时探索会如何表现？是否需要 SARSA 的保守性？

拒绝将表格型 TD 应用于状态空间 > 10⁶ 的问题。拒绝在没有最大化偏差提示的情况下交付 Q-learning 智能体。标记任何整个训练过程保持 ε=1.0 的智能体（无利用阶段）。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制 2,000 个回合的学习曲线（每 100 回合平均回报）。谁收敛更快？
2. **中等。** 构建一个悬崖行走环境（4×12，最底一行是悬崖，奖励 -100 并重置回起点）。对比 Q-learning 和 SARSA 的最终策略。截图显示各自路径。谁更靠近悬崖？
3. **困难。** 实现 Double Q-learning。在一个有噪声奖励的 GridWorld 上（每步奖励加入 σ=5 的高斯噪声），展示 Q-learning 会明显高估 `V*(0,0)`，而 Double Q-learning 不会。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| TD error（时序差分误差） | "更新信号" | `δ = r + γ V(s') - V(s)`，自举后的残差。 |
| TD(0) | "单步时序差分" | 每次转移后只使用下一个状态的估计进行更新。 |
| Q-learning | "离策略强化学习入门" | 对下一状态动作取 `max` 的 TD 更新；无论行为策略如何都学习 `Q*`。 |
| SARSA | "在策略版 Q-learning" | 使用实际下一动作的 TD 更新；学习当前 ε-贪心策略 π 对应的 `Q^π`。 |
| Expected SARSA | "低方差 SARSA" | 用 π 下采样 `a'` 的期望替代采样值。 |
| GLIE | "正确的探索调度" | Greedy in the Limit with Infinite Exploration；Q-learning 收敛所需条件。 |
| Bootstrapping（自举） | "在目标中使用当前估计" | 区分 TD 与 MC 的关键。带来偏差但大幅降低方差。 |
| Maximization bias（最大化偏差） | "Q-learning 会高估" | 对噪声估计取 `max` 会向上偏；Double Q-learning 可修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) —— 原始论文与收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) —— TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) —— 最大化偏差的修复方案。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) —— Expected SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) —— 提出 SARSA 的论文（当时称为 "modified connectionist Q-learning"）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) —— 将 TD(0) 推广到 TD(n)，是从 Q-learning 到资格迹（eligibility traces）以及后来 PPO 中 GAE 的路径。
