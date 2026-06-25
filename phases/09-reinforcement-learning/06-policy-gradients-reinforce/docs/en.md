# 策略梯度（Policy Gradient）—— 从零实现 REINFORCE

> 别再估计价值。直接参数化策略，计算期望回报的梯度，然后沿梯度上升方向迈步。Williams（1992）用一个定理就把它写清楚了。这正是 PPO、GRPO 以及所有大语言模型强化学习循环存在的原因。

**类型：** Build
**语言：** Python
**前置知识：** Phase 3 · 03（反向传播），Phase 9 · 03（蒙特卡洛方法），Phase 9 · 04（时序差分学习）
**预计时间：** ~75 分钟

## 问题背景

Q-learning 和 DQN 参数化的是*价值*函数。你通过 `argmax Q` 来选择动作。对于离散动作和离散状态这没问题；但当动作是连续的（你要对 10 维力矩做 `argmax` 吗？）或者你想要随机策略时（`argmax` 本质上是确定性的），它就行不通了。

策略梯度直接参数化*策略*。`π_θ(a | s)` 是一个输出动作分布的神经网络。从该分布中采样即可执行动作。计算期望回报关于 `θ` 的梯度，沿梯度上升方向迈步。没有 `argmax`，没有贝尔曼递归，只是对 `J(θ) = E_{π_θ}[G]` 做梯度上升。

REINFORCE 定理（Williams，1992）告诉我们这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个回合，计算回报，将每一步的 `∇ log π_θ(a | s)` 与之相乘，取平均，梯度上升，完成。

2026 年的每一个大语言模型强化学习算法——PPO、DPO、GRPO——都是 REINFORCE 的改进版。亲手理解它，是学习本阶段后续内容以及 Phase 10 · 07（RLHF 实现）和 Phase 10 · 08（DPO）的先决条件。

## 核心概念

![策略梯度：softmax 策略、log-π 梯度、回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任意由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从第 `t` 步开始的折扣回报（discounted return）。期望是对从 `π_θ` 采样得到的完整轨迹 `τ` 取的。

**证明很短。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。利用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧，log-derivative trick）。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 与 θ 无关的环境项`。环境项消失。两行代数即可得到该定理。

**方差缩减技巧。** 原始 REINFORCE 的方差大得惊人——回报本身有噪声，`∇ log π` 也有噪声，二者相乘后噪声更大。两个标准修正方法：

1. **基线减法（baseline subtraction）。** 将 `G_t` 替换为 `G_t - b(s_t)`，其中 `b(s_t)` 是不依赖 `a_t` 的任意基线（baseline）。它是无偏的，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：`b(s_t) = V̂(s_t)` 由一个评论者（critic）学习 → 即演员-评论家（actor-critic）方法（第 07 课）。
2. **即时回报（reward-to-go）。** 将 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。对某个动作而言，只有未来回报才重要——过去奖励只会带来零均值噪声。

二者结合，得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE——A2C（第 07 课）和 PPO（第 08 课）的直接前身。

**Softmax 策略参数化。** 对于离散动作，标准选择是：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任意输出每个动作得分的神经网络。该梯度有一个简洁形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即所采取动作的得分减去该得分在策略下的期望值。

**连续动作的高斯（Gaussian）策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式解。这就是 Phase 9 · 07 的 SAC 所需的全部内容。

## 动手实现

### 第 1 步：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对于表格型环境，使用线性策略（每个动作一个权重向量）。对于 Atari，把网络换成 CNN，保留 softmax 输出头即可。

### 第 2 步：采样与对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 第 3 步：记录对数概率的回合 rollout

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 第 4 步：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（动作 `a` 的 one-hot 向量减去概率分布）是 softmax 策略梯度的核心。把它刻进肌肉记忆。

### 第 5 步：基线

对最近若干回合的 `G` 取滑动均值，就足以把方差降到让 4×4 GridWorld 跑起来；大约 500 个回合收敛。把基线升级为学习得到的 `V̂(s)`，你就得到了演员-评论家方法。

## 常见陷阱

- **梯度爆炸。** 回报可能非常大。在乘以 `∇ log π` 之前，务必将 `G` 在批次内归一化到 `~N(0, 1)`。
- **熵坍塌（entropy collapse）。** 策略过早收敛到近似确定性动作，停止探索，陷入局部最优。修复方法：在目标函数中加入熵奖励 `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要数千个回合。评论者基线（第 07 课）或 TRPO/PPO 的信任区域（第 08 课）是标准修复。
- **样本低效。** 同策略（on-policy）意味着每次更新后就把所有转移数据扔掉。通过重要性采样（importance sampling）做离线策略修正可以复用数据，但代价是方差增加（PPO 的比率就是一种截断的重要性采样权重）。
- **非平稳梯度。** 100 个回合前的梯度对应的是旧策略 `π`。因此同策略方法每隔几个 rollout 就更新一次。
- **信用分配。** 不使用 reward-to-go 时，过去奖励会变成噪声。永远使用 reward-to-go。

## 应用场景

2026 年，REINFORCE 本身已很少直接运行，但它的梯度公式无处不在：

| 应用场景 | 派生方法 |
|----------|---------------|
| 连续控制 | PPO / SAC 配合高斯策略 |
| 大语言模型 RLHF | 带 KL 惩罚（KL penalty）的 PPO，运行在 token 级策略上 |
| 大语言模型推理（DeepSeek） | GRPO——使用组相对基线、无评论者的 REINFORCE |
| 多智能体 | 中心化评论者的 REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅有偏好数据 | DPO——将 REINFORCE 重写为偏好似然损失（preference-likelihood loss），无需采样 |

当你在 2026 年的训练脚本里看到 `loss = -advantage * log_prob` 时，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是建立在这一行代码之上的方差缩减技巧。

## 交付产物

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: 为给定任务生成 REINFORCE / actor-critic / PPO 训练配置，并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定一个环境（离散 / 连续动作、horizon、奖励统计量），输出：

1. 策略头。Softmax（离散）或高斯（连续）及其参数量。
2. 基线。无（原始 REINFORCE）、滑动均值、学习得到的 `V̂(s)`，或 A2C 评论者。
3. 方差控制。默认启用 reward-to-go、回报归一化、梯度裁剪值。
4. 熵奖励。系数 β 及其衰减计划。
5. 批次大小。每次更新使用的回合数；同策略数据新鲜度约定。

当 horizon > 500 步时，拒绝使用无基线的 REINFORCE。拒绝给连续动作控制配 softmax 输出头。若某次运行 `β = 0` 且观测到的策略熵 < 0.1，则标记为熵坍塌。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用线性 softmax 策略实现 REINFORCE。不使用基线训练 1,000 个回合，绘制学习曲线并测量回报方差（std）。
2. **中等。** 加入滑动均值基线，再次训练。与原始版本比较样本效率和方差。基线将收敛所需步数减少了多少？
3. **困难。** 加入熵奖励 `β · H(π)`。对 `β ∈ {0, 0.01, 0.1, 1.0}` 做扫描，绘制最终回报和策略熵。在该任务上，最佳取值点在哪里？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 策略梯度（Policy gradient） | “直接训练策略” | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`；由对数导数技巧推导而来。 |
| REINFORCE | “原始 PG 算法” | Williams（1992）；将蒙特卡洛回报与对数策略梯度相乘。 |
| 对数导数技巧（Log-derivative trick） | “得分函数估计量” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；让期望的梯度可计算。 |
| 基线（Baseline） | “减小方差” | 从 `G` 中减去的任意 `b(s)`；无偏，因为 `E[b · ∇ log π] = 0`。 |
| 即时回报（Reward-to-go） | “只有未来回报才算数” | 用 `G_t^{from t}` 代替完整 `G_0`；正确且方差更低。 |
| 熵奖励（Entropy bonus） | “鼓励探索” | 目标函数中的 `+β · H(π(·|s))` 项，防止策略坍塌。 |
| 同策略（On-policy） | “用刚采样的数据训练” | 梯度期望是相对于当前策略取的——不能直接复用旧数据。 |
| 优势（Advantage） | “比平均好多少” | `A(s, a) = G(s, a) - V(s)`；带基线 REINFORCE 所乘的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) —— 原始 REINFORCE 论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) —— 带函数近似的现代策略梯度定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书式讲解。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) —— 清晰的教学阐述，含 PyTorch 代码。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) —— 方差缩减与自然梯度视角，将 REINFORCE 与信任区域家族（TRPO、PPO）联系起来。
