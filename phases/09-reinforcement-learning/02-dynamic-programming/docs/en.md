# 动态规划（Dynamic Programming）—— 策略迭代与价值迭代

> 动态规划（dynamic programming）就是“开了挂”的强化学习（reinforcement learning）。你已经知道状态转移和奖励函数，只需反复迭代 Bellman 方程，直到 `V` 或 `π` 不再变化。它是所有基于采样方法试图逼近的基准。

**类型：** Build  
**语言：** Python  
**前置知识：** Phase 9 · 01 (MDPs)  
**时间：** 约 75 分钟

## 问题背景

你面对的是一个已知模型的马尔可夫决策过程（Markov Decision Process，MDP）：对于任意状态-动作对，你都能直接查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存经理知道需求分布，棋类游戏有确定性的状态转移，网格世界（GridWorld）用四行 Python 就能描述。你拥有*模型*。

无模型强化学习（model-free RL，例如 Q 学习（Q-learning）、PPO、REINFORCE）是为了应对没有模型的情况——你只能对环境采样。但当你拥有模型时，就有更快、更好的方法：动态规划（dynamic programming）。Bellman 在 1957 年设计了这些方法，它们至今仍是“正确性”的定义：当人们说某个 MDP 的“最优策略”时，指的就是动态规划会返回的策略。

在 2026 年，你仍然需要掌握它们，原因有三。第一，强化学习研究中的每个表格环境（GridWorld、FrozenLake、CliffWalking）都用动态规划求解，得到黄金标准策略。第二，精确的值函数能帮你调试（debug）采样方法：如果 Q 学习对 `V*(s_0)` 的估计与动态规划答案相差 30%，那你的 Q 学习一定有 bug。第三，现代离线强化学习（offline RL）与规划方法（MCTS、AlphaZero 的搜索、Phase 9 · 10 中的基于模型的强化学习（model-based RL））都会在一个学习得到或已知的模型上迭代 Bellman 备份。

## 核心概念

![策略迭代与价值迭代对比](../assets/dp.svg)

**两种算法，都是对 Bellman 方程的不动点迭代。**

**策略迭代（policy iteration）。** 交替执行以下两个步骤，直到策略不再变化：

1. **评估（evaluation）**：给定策略 `π`，反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`，直到 `V^π` 收敛。
2. **改进（improvement）**：给定 `V^π`，让 `π` 关于 `V^π` 贪婪（greedy）：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛性有保证，因为（a）每次改进要么保持 `π` 不变，要么严格提高某些状态的 `V^π`；（b）确定性策略空间有限。即使状态空间很大，通常也只需大约 5–20 次外层迭代即可收敛。

**价值迭代（value iteration）。** 把评估和改进压缩成一次扫描。直接应用 Bellman *最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_new(s) - V(s)| < ε`。最后通过贪婪动作提取策略。每次迭代更快——没有内层评估循环——但通常需要更多迭代才能收敛。

**广义策略迭代（Generalized Policy Iteration，GPI）。** 统一的视角。价值函数和策略被锁定在一个双向改进循环中；任何把两者同时推向一致性的方法（异步价值迭代、改进型策略迭代、Q 学习、Actor-Critic、PPO）都是 GPI 的实例。

**为什么 `γ < 1` 很重要。** Bellman 算子（Bellman operator）在上确界范数（sup-norm）下是一个 `γ`-压缩映射：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩映射意味着唯一不动点和几何收敛。如果去掉 `γ < 1` 的条件，这一保证就失效了——你需要有限时间范围或吸收终止状态。

## 动手实现

### 步骤 1：构建 GridWorld MDP 模型

延用第 01 课的 4×4 网格世界（GridWorld）。我们加入随机变体：智能体以 `0.1` 的概率滑向一个随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回 `(s', r, p)` 的列表。这就是完整的模型。

### 步骤 2：策略评估

给定策略 `π(s) = {action: prob}`，迭代 Bellman 方程直到 `V` 不再变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 步骤 3：策略改进

将 `π` 替换为关于 `V` 的贪婪策略。如果 `π` 没有变化，说明已到达最优，直接返回。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 步骤 4：组合成策略迭代

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 任意初始策略
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

在 4×4 网格上通常 4–6 次外层迭代即可收敛。输出 `V*(0,0) ≈ -6`，以及一个能严格减少步数的策略。

### 步骤 5：价值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

同样的不动点，更少的代码行。

## 常见陷阱

- **忘记处理终止状态。** 如果把 Bellman 方程应用到吸收态，它仍会选出一个“最佳动作”，尽管什么都不会改变。用 `if s == terminal: V[s] = 0` 保护。
- **上确界范数（sup-norm）与 L2 收敛。** 使用 `max |V_new - V|`，而不是平均值。理论保证针对的是上确界范数。
- **原地（in-place）与同步（synchronous）更新。** 原地更新 `V[s]`（高斯-赛德尔（Gauss-Seidel）风格）比使用单独的 `V_new` 字典（雅可比（Jacobi）风格）收敛更快。生产代码通常使用原地更新。
- **策略平局。** 如果两个动作的 Q 值相同，`argmax` 每次可能按不同方式打破平局，导致“策略稳定”检查震荡。使用稳定的平局打破规则（按固定顺序取第一个动作）。
- **状态空间（state space）爆炸。** 动态规划每次扫描复杂度为 `O(|S| · |A|)`。大约能处理到 10⁷ 个状态。超过这个规模就需要函数近似（Phase 9 · 05 及以后）。

## 应用场景

在 2026 年，动态规划既是正确性基线，也是规划器的内层循环：

| 使用场景 | 方法 |
|----------|------|
| 精确求解小型表格 MDP | 价值迭代（更简单）或策略迭代（外层迭代更少） |
| 验证 Q 学习 / PPO 实现 | 与玩具环境上的动态规划最优 `V*` 对比 |
| 基于模型的强化学习（Phase 9 · 10） | 在学习得到的状态转移模型上做 Bellman 备份 |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索（Monte Carlo Tree Search，MCTS）= 异步 Bellman 备份 |
| 离线强化学习（CQL、IQL） | 保守 Q 迭代——对分布外（out-of-distribution，OOD）动作加惩罚的动态规划 |

每次有人说“最优价值函数”时，他们指的就是“动态规划的不动点”。当你在论文中看到 `V*` 或 `Q*` 时，脑海里应浮现这个循环。

## 交付

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行价值迭代，分别取 `γ ∈ {0.9, 0.99}`。多少次扫描后 `max |ΔV| < 1e-6`？把 `V*` 打印成 4×4 的网格。
2. **中等。** 在*随机* GridWorld（滑倒概率 `0.1`）上对比策略迭代与价值迭代。统计：扫描次数、实际运行时间、最终 `V*(0,0)`。哪种方法在迭代次数上收敛更快？在实际运行时间上呢？
3. **困难。** 实现改进型策略迭代（modified policy iteration）：在评估步骤中只做 `k` 次扫描，而不是收敛到精确 `V^π`。对 `k ∈ {1, 2, 5, 10, 50}` 绘制 `V*(0,0)` 误差随 `k` 变化的曲线。这条曲线说明了评估与改进之间的什么权衡？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 策略迭代（policy iteration） | “DP 算法” | 交替进行评估（`V^π`）和改进（关于 `V^π` 的贪婪 `π`），直到策略不再变化。 |
| 价值迭代（value iteration） | “更快的 DP” | 单次扫描中应用 Bellman 最优性备份；几何收敛到 `V*`。 |
| Bellman 算子（Bellman operator） | “那个递归” | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在上确界范数下是 `γ`-压缩映射。 |
| 压缩映射（contraction） | “DP 收敛的原因” | 满足 `||T x - T y|| ≤ γ ||x - y||` 的算子 `T` 具有唯一不动点。 |
| 广义策略迭代（GPI） | “万物皆 DP” | Generalized Policy Iteration：任何把 `V` 和 `π` 推向相互一致的方法。 |
| 同步更新（synchronous update） | “雅可比（Jacobi）风格” | 整个扫描中使用旧的 `V`；易于分析但较慢。 |
| 原地更新（in-place update） | “高斯-赛德尔（Gauss-Seidel）风格” | 边扫描边使用最新 `V`；实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) —— 策略迭代与价值迭代的经典讲解。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) —— 压缩映射论证的严格处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) —— 改进型策略迭代及其收敛分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) —— 原始策略迭代论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) —— 从动态规划到近似动态规划 / 深度强化学习的桥梁，后续每一课都会用到。
