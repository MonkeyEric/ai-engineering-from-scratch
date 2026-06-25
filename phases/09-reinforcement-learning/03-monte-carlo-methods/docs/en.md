# 蒙特卡洛方法（Monte Carlo Methods）—— 从完整回合中学习

> 动态规划需要一个模型。蒙特卡洛只需要回合数据。执行策略、观察回报、取平均。这是强化学习中最简单的想法，也是开启后续一切内容的钥匙。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 9 · 01（马尔可夫决策过程），Phase 9 · 02（动态规划）
**时间：** 约 75 分钟

## 问题所在

动态规划很优雅，但它假设你可以对每个状态与动作查询 `P(s' | s, a)`。现实世界几乎不存在这种情况。机器人无法解析计算某个关节力矩后摄像头像素的分布；定价算法无法对所有可能的客户反应进行积分；大语言模型也无法枚举某个词元之后所有可能的续写。

你需要一种仅要求从环境中*采样*的方法。执行策略，得到一条轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它估计价值。这就是蒙特卡洛。

从 DP 到 MC 的转变在哲学上很重要：我们从*已知模型 + 精确备份*转向*采样 rollout + 平均回报*。方差会激增，但适用范围会爆炸式增长。本课之后的每个 RL 算法 —— TD、Q-learning、REINFORCE、PPO、GRPO —— 本质上都是蒙特卡洛估计器，有时在其之上叠加自举（bootstrapping）。

## 核心概念

![蒙特卡洛：rollout、计算回报、取平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**核心思想，一句话概括：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在策略 `π` 下访问状态 `s` 后观察到的回报。

**首次访问（first-visit）与每次访问（every-visit）MC。** 如果一个回合多次访问状态 `s`，首次访问 MC 只计算第一次访问之后的回报；每次访问 MC 则计算所有访问之后的回报。两者在极限下都是无偏的。首次访问更易于分析（独立同分布样本）。每次访问每回合使用更多数据，实践中通常收敛更快。

**增量均值。** 不需要保存所有回报，只需更新运行平均：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

整理后：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。把 `1/n` 换成常数步长 `α ∈ (0, 1)`，就得到了能够跟踪 `π` 变化的非平稳 MC 估计器。这一步就是从 MC 到 TD 再到所有现代 RL 算法的全部飞跃。

**探索现在成了问题。** DP 通过枚举接触到每个状态。MC 只能看到策略访问到的状态。如果 `π` 是确定性的，状态空间的大片区域永远不会被采样，其价值估计永远停留在零。历史上三种解决方案：

1. **探索性起点（exploring starts）。** 每个回合从随机的 (s, a) 对开始。能保证覆盖；但在实践中不现实（你无法把机器人“重置”到任意状态）。
2. **ε-贪心（ε-greedy）。** 相对于当前 Q 采取贪心动作，但以概率 `ε` 随机选择一个动作。所有状态-动作对都会被渐进采样。
3. **离策略（off-policy）MC。** 在行为策略 `μ` 下收集数据，通过重要性采样（importance sampling）学习目标策略 `π` 的价值。方差较高，但它是通向 DQN 等回放缓冲区方法的桥梁。

**蒙特卡洛控制（Monte Carlo Control）。** 评估 → 改进 → 评估，就像策略迭代一样，只不过评估是基于采样的：

1. 执行 `π`，得到一个回合。
2. 从观察到的回报更新 `Q(s, a)`。
3. 让 `π` 相对于 `Q` 是 ε-贪心的。
4. 重复。

在温和条件下（每对都被无限次访问，`α` 满足 Robbins-Monro），它以概率 1 收敛到 `Q*` 和 `π*`。

## 动手实现

### 步骤 1：rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

不需要模型，只有 `env.reset()` 和 `env.step(s, a)`。与 gym 环境接口相同，但已精简。

### 步骤 2：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次遍历，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免了重复求和。

### 步骤 3：首次访问 MC 策略评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

核心工作由三行完成：标记首次访问、增加计数、更新运行均值。

### 步骤 4：ε-贪心 MC 控制（on-policy）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 步骤 5：与 DP 黄金标准对比

你对 `V^π` 的 MC 估计应随着回合数 → ∞ 而趋近于第 02 课的 DP 结果。实践中：4×4 GridWorld 上运行 50,000 个回合，误差通常在 `~0.1` 以内。

## 常见陷阱

- **无限回合。** MC 要求回合*终止*。如果你的策略可能永远循环，请设置 `max_steps` 上限，并将该上限视为隐式失败。使用随机策略的 GridWorld 经常超时 —— 这很正常，只要正确计数即可。
- **方差。** MC 使用完整回报。在长回合中，方差很大 —— 末尾一次不幸的奖励会让 `V(s_0)` 同样幅度地偏移。TD 方法（第 04 课）通过自举来削减这一点。
- **状态覆盖。** 在全新的 Q 上使用贪心 MC 且存在平局时，只会尝试一个动作。你*必须*探索（ε-贪心、探索性起点、UCB）。
- **非平稳策略。** 如果 `π` 发生变化（如 MC 控制中），旧的回报来自不同策略。常数-α MC 能处理这一点；样本平均 MC 不能。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 沿整条轨迹相乘。方差随 horizon 爆炸。可通过每步加权 IS（per-decision weighted IS）进行限制，或改用 TD。

## 应用场景

蒙特卡洛方法在 2026 年的角色：

| 使用场景 | 为何使用 MC |
|----------|-------------|
| 短回合游戏（二十一点、扑克） | 回合自然终止；回报清晰。 |
| 日志策略的离线评估 | 对存储的轨迹取平均折扣回报。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶子节点进行 MC rollout 来指导选择。 |
| 大语言模型 RL 评估 | 对给定策略的采样续写计算平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC 的 `G_t`。 |
| 教授 RL | 真正有效的最简单算法 —— 去掉自举就能看到核心。 |

现代深度 RL 算法（PPO、SAC）通过 `n` 步回报或 GAE，在纯 MC（完整回报）与纯 TD（单步自举）之间插值。两个端点都是同一估计器的实例。

## 交付成果

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: 通过蒙特卡洛 rollout 评估策略，并在可用时生成与 DP 对比的收敛报告。
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

给定一个环境（回合制，具有 reset+step API）和一个策略，输出：

1. 方法。首次访问 vs 每次访问 MC。理由。
2. 回合预算。目标数量、方差诊断、期望标准误。
3. 探索方案。ε 调度（如需要）或探索性起点。
4. 黄金标准对比。表格情形下使用 DP 最优 V*；否则使用 Q-learning / PPO 基线的界限。
5. 终止检查。最大步数上限、超时、非终止轨迹的处理。

拒绝在没有有限 horizon 上限的非回合制任务上运行 MC。拒绝在表格任务每个状态少于 100 个回合时报告 V^π 估计。将任何零方差动作的策略标记为探索风险。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现均匀随机策略的首次访问 MC 评估。运行 10,000 个回合。绘制 `V(0,0)` 随回合数变化的曲线，并与 DP 答案对比。
2. **中等。** 实现 ε-贪心 MC 控制，`ε ∈ {0.01, 0.1, 0.3}`。比较 20,000 个回合后的平均回报。曲线长什么样？偏差-方差权衡在哪里？
3. **困难。** 实现带重要性采样的*离策略* MC：在均匀随机策略 `μ` 下收集数据，估计确定性最优策略 `π` 的 `V^π`。比较普通 IS、每步 IS 与加权 IS。哪种方差最低？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 蒙特卡洛（Monte Carlo） | “随机采样” | 通过对分布的独立同分布样本取平均来估计期望。 |
| 回报 `G_t` | “未来奖励” | 从步骤 `t` 到回合结束的折扣奖励之和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC（First-visit MC） | “每个状态只算一次” | 只有回合中的第一次访问会贡献价值估计。 |
| 每次访问 MC（Every-visit MC） | “使用所有访问” | 每次访问都贡献；略有偏差但样本效率更高。 |
| ε-贪心（ε-greedy） | “探索噪声” | 以概率 `1-ε` 选择贪心动作；以概率 `ε` 选择随机动作。 |
| 重要性采样（Importance sampling） | “修正从错误分布采样的偏差” | 通过 `π(a|s)/μ(a|s)` 的乘积对回报重新加权，从而用 `μ` 的数据估计 `V^π`。 |
| 同策略（On-policy） | “用自己的数据学习” | 目标策略 = 行为策略。普通 MC、PPO、SARSA。 |
| 离策略（Off-policy） | “用别人的数据学习” | 目标策略 ≠ 行为策略。重要性采样 MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 权威教材。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 首次访问与每次访问的分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — 离策略 MC 与方差控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS 估计器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — 首个大规模 MC/TD 自弈收敛到超人水平的实证演示；本阶段后半部分每一课的概念先驱。
