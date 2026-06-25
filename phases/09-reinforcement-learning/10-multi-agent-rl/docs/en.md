# 多智能体强化学习（Multi-Agent RL）

> 单智能体强化学习假设环境是平稳的。当把两个学习中的智能体放进同一个世界时，这个假设就不成立了：每个智能体都是对方环境的一部分，而且双方都在不断变化。多智能体强化学习（multi-agent RL）就是当马尔可夫假设不再成立时，仍然让学习收敛的一套技巧。

**类型：** Build
**语言：** Python
**前置课程：** Phase 9 · 04（Q-learning）、Phase 9 · 06（REINFORCE）、Phase 9 · 07（Actor-Critic）
**时间：** 约 45 分钟

## 问题所在

机器人学习在房间里导航是单智能体强化学习问题。一支足球队不是。AlphaStar 对战《星际争霸》对手不是。竞价代理组成的市场不是。两辆车在十字路口协商通行顺序也不是。许多“多对多”的真实世界问题都不是。

在每一个多智能体场景中，从任意一个智能体的视角来看，其他智能体*就是*环境的一部分。当它们学习并改变自身行为时，环境就变得非平稳。马尔可夫性质——“下一状态只取决于当前状态和我的动作”——不再成立，因为下一状态还取决于*其他*智能体选择了什么动作，而它们的策略是不断移动的靶子。

这破坏了表格型收敛证明（Q-learning 的保证依赖于平稳环境）。也让朴素的深度强化学习失效：智能体互相追逐，陷入循环，永远无法收敛到稳定策略。因此需要多智能体专用技术：中心化训练 / 去中心化执行（centralized training / decentralized execution）、反事实基线（counterfactual baselines）、联盟训练（league play）、自我博弈（self-play）。

2026 年的应用场景包括：机器人集群、交通路由、自动驾驶车队、市场模拟器、多智能体大语言模型系统（Phase 16），以及任何有不止一个智能玩家的游戏。

## 核心概念

![四种 MARL 范式：独立式、中心化评论家、自我博弈、联盟训练](../assets/marl.svg)

**形式化：马尔可夫博弈（Markov Game）。** 马尔可夫决策过程（MDP）的推广：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移概率 `P(s' | s, a)`，以及每个智能体各自的奖励 `R_i(s, a, s')`。每个智能体 `i` 都在自身策略 `π_i` 下最大化自身回报。如果所有奖励相同，则称为**完全合作（fully cooperative）**；如果零和，则称为**对抗（adversarial）**；如果介于两者之间，则称为**一般和（general-sum）**。

**核心挑战：**

- **非平稳性（Non-stationarity）。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 依赖于 `π_{-i}`，而它在不断变化。
- **贡献分配（Credit assignment）。** 如果奖励是共享的，该归功于哪个智能体？
- **探索协调（Exploration coordination）。** 智能体必须探索互补策略，而不是重复探索相同的状态。
- **可扩展性（Scalability）。** 联合动作空间随智能体数量 `n` 指数增长。
- **部分可观测性（Partial observability）。** 每个智能体只能看到自己的观测，全局状态被隐藏。

**四种主流范式：**

**1. 独立 Q-learning / 独立 PPO（IQL、IPPO）。** 每个智能体学习自己的 Q 函数或策略，把其他智能体当作环境的一部分。简单，有时有效（尤其是经验回放可以起到平滑对手模型的作用）。理论收敛性：没有。实践中：适合松耦合任务，不适合紧耦合任务。

**2. 中心化训练，去中心化执行（CTDE）。** 现代最常见的范式。每个智能体拥有只基于局部观测 `o_i` 的*策略* `π_i`——部署时标准地去中心化执行。在*训练*阶段，中心化评论家 `Q(s, a_1, …, a_n)` 基于全局状态和联合动作进行估值。例如：
- **MADDPG**（Lowe et al. 2017）：为每个智能体配备中心化评论家的 DDPG。
- **COMA**（Foerster et al. 2017）：反事实基线——追问“如果我换成动作 `a'`，我的奖励会是多少？”——从而分离出我的贡献。
- **MAPPO** / 共享评论家的 **IPPO**（Yu et al. 2022）：使用中心化价值函数的 PPO。在 2026 年的合作型多智能体强化学习中占据主导地位。
- **QMIX**（Rashid et al. 2018）：值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，混合函数满足单调性约束。

**3. 自我博弈（Self-play）。** 同一个智能体的两个副本互相对战。对手的策略*就是*我过去某个快照的策略。AlphaGo / AlphaZero / MuZero、OpenAI Five 都采用了这一方法。最适合零和博弈；训练信号是对称的。

**4. 联盟训练（League play）。** 自我博弈在一般和 / 对抗环境中的扩展：维护一个由过去和当前策略组成的种群，从联盟中采样对手进行训练。联盟中还会加入“剥削者（exploiters）”（专门击败当前最强策略）和“主剥削者（main exploiters）”（专门击败剥削者）。AlphaStar（《星际争霸 II》）。当游戏存在“石头剪刀布”式的策略循环时是必需的。

**通信（Communication）。** 允许智能体之间发送学习得到的消息 `m_i`。在合作场景中有效。Foerster 等人（2016）证明了智能体间的可微通信可以端到端训练。如今基于大语言模型的多智能体系统（Phase 16）本质上就是用自然语言通信。

## 动手实现

本课使用一个 6×6 的 GridWorld，其中有两个合作智能体。它们从对角角落出发，必须到达同一个目标。共享奖励为：只要任一智能体还在移动，每步 `-1`；当两者都到达目标时 `+10`。参见 `code/main.py`。

### 步骤 1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 两个智能体

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间大小为 `|A|² = 16`。全局状态是两个智能体的位置。

### 步骤 2：独立 Q-learning

每个智能体维护自己的 Q 表，以联合状态为键。每步：两者都选择 ε-贪婪动作，收集联合转移，各自用共享奖励更新自己的 Q 值。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在这个任务上能跑通，因为奖励密集且一致。在紧耦合任务上会失败（例如，一个智能体必须*等待*另一个智能体）。

### 步骤 3：中心化 Q 与分解值更新

使用一个覆盖联合动作的 Q 函数 `Q(s, a_1, a_2)`，用共享奖励更新。执行时通过边缘化去中心化：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数级增长的联合动作空间换取*正确的*全局视角。

### 步骤 4：简单的自我博弈（对抗型双智能体）

同一个智能体，两个角色。训练智能体 A 对抗智能体 B；每过 `K` 个回合，把 A 的权重复制给 B。对称训练，稳定进步。这是 AlphaZero 配方的微缩版。

## 常见陷阱

- **非平稳的经验回放。** 对于独立智能体，经验回放比单智能体更差，因为旧的转移样本来自现在已经过时的对手。修复：重新标注或按新鲜度加权。
- **贡献分配模糊。** 长回合结束后才给出共享奖励，无法明确判断哪个智能体做出了贡献。修复：反事实基线（COMA）或为每个智能体设计奖励塑形（reward shaping）。
- **策略漂移 / 互相追逐。** 每个智能体的最优反应都会随着对方的更新而改变。修复：中心化评论家、较小的学习率，或一次只冻结一个智能体。
- **通过协调作弊。** 智能体会发现设计者没有预料到的协同漏洞。拍卖智能体可能收敛到出价为零。修复：谨慎设计奖励、加入行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：为每个智能体单独加熵奖励，或按角色进行条件化探索。
- **联盟循环。** 纯自我博弈可能陷入优势循环。修复：使用包含多样对手的联盟训练。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。可用函数近似、因子化动作空间（每个智能体一个策略输出头）来缓解。

## 如何应用

2026 年多智能体强化学习的应用地图：

| 领域 | 方法 | 说明 |
|--------|--------|-------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享评论家 + 去中心化执行器。 |
| 双人游戏（国际象棋、围棋、扑克） | 结合 MCTS 的自我博弈（AlphaZero） | 零和；对称训练。 |
| 复杂多人游戏（Dota、《星际争霸》） | 联盟训练 + 模仿预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / 带注意力的 PPO | 部分观测；可变队伍规模。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时使用平均场强化学习（mean-field RL）。 |
| 大语言模型多智能体系统（Phase 16） | 自然语言通信 + 角色条件化 | 在智能体规划层形成 RL 闭环。 |

在 2026 年，多智能体强化学习最大的增长领域是基于大语言模型的系统：语言模型智能体群体进行协商、辩论、构建软件。此时强化学习表现为对*轨迹级*输出的偏好优化，而非词元级优化（Phase 16 · 03）。

## 交付成果

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: 为给定任务选择合适的多智能体 RL 范式（IPPO、CTDE、self-play、league）。
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在双人合作 GridWorld 上训练独立 Q-learning。平均回报何时超过 0？绘制联合学习曲线。
2. **中等。** 增加一个“协调”任务：只有当两个智能体在同一回合踏上目标时才算达成。独立 Q-learning 还能收敛吗？哪里会出问题？
3. **困难。** 实现一个 MAPPO 风格的中心化评论家，并在协调任务上与独立 PPO 比较收敛速度。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Markov game | “多智能体 MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有各自的奖励。 |
| CTDE | “中心化训练，去中心化执行” | 训练时使用联合评论家；每个智能体的策略只使用局部观测。 |
| IPPO | “独立 PPO” | 每个智能体单独运行 PPO。简单基线；常被低估。 |
| MAPPO | “多智能体 PPO” | 价值函数以全局状态为条件的 PPO。 |
| QMIX | “单调值分解” | `Q_tot = f_monotone(Q_1, …, Q_n)`，允许去中心化的 argmax。 |
| COMA | “反事实多智能体” | 优势 = 我的 Q 值 - 对我的动作边缘化后的期望 Q 值。 |
| Self-play | “智能体对战过去的自己” | 同一个智能体，两个角色；零和游戏中的标准做法。 |
| League play | “种群训练” | 缓存历史策略，从池中采样对手；处理策略循环。 |

## 扩展阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 中心化评论家的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用于贡献分配的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 单调值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中出人意料地强。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联盟训练。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和游戏中的纯自我博弈。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包含教材对多智能体场景及 CTDE 旨在解决的非平稳性问题的简要讨论。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 涵盖合作、竞争和混合 MARL 以及收敛结果的综述。
