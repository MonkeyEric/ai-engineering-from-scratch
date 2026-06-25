# Actor-Critic — A2C 与 A3C

> REINFORCE 的噪声很大。引入一个学习 `V̂(s)` 的评论家（critic），从回报（return）中减去它，就得到一个具有相同期望但方差低得多的优势（advantage）。这就是 Actor-Critic。A2C 同步运行它，A3C 跨线程运行。二者都是所有现代深度强化学习（deep RL）方法的概念基础。

**类型：** 实战
**语言：** Python
**前置知识：** 第 9 阶段 · 04（时序差分学习，TD Learning），第 9 阶段 · 06（REINFORCE）
**时间：** 约 75 分钟

## 问题

原始 REINFORCE 有效，但方差很大。蒙特卡洛（Monte Carlo）回报 `G_t` 在不同回合之间可能波动 10 倍。将这种噪声乘以 `∇ log π` 再取平均，得到的梯度估计器需要数千个回合才能让策略移动与少量 DQN 更新相同的距离。

方差来自使用原始回报。如果减去基线（baseline）`b(s_t)`——任何状态函数，包括学习到的价值——期望不变而方差下降。最佳可处理的基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量就是*优势（advantage）*：

`A(s, a) = G - V̂(s)`

如果一个动作产生了高于平均水平的回报，它就是好的；低于平均水平则是差的。带有学习评论家的 REINFORCE 就是 *Actor-Critic*。评论家为演员（actor）提供了一个低方差的教师。这是 2015 年之后每种深度策略方法（A2C、A3C、PPO、SAC、IMPALA）的基础。

## 概念

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**两个网络，一个共享损失：**

- **演员（Actor）** `π_θ(a | s)`：策略（policy）。采样后用于执行动作。用策略梯度训练。
- **评论家（Critic）** `V_φ(s)`：估计从某个状态开始的期望回报。通过最小化 `(V_φ(s) - target)²` 训练。

**优势（Advantage）。** 两种标准形式：

- *蒙特卡洛优势（MC advantage）：* `A_t = G_t - V_φ(s_t)`。无偏，方差较高。
- *时序差分优势（TD advantage）：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用了 `V_φ`），方差低得多。也称为 *时序差分残差（TD residual）* `δ_t`。

**n 步优势（n-step advantage）。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现中，Atari 使用 `n = 5`，MuJoCo 上的 PPO 使用 `n = 2048`。

**广义优势估计（Generalized Advantage Estimation，GAE）。** Schulman 等人（2016）提出对所有 n 步优势做指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差、高偏差）。`λ = 1` 是 MC（高方差、无偏）。`λ = 0.95` 是 2026 年的默认设置——按需调节偏差/方差旋钮。

**A2C：同步优势 Actor-Critic（synchronous advantage actor-critic）。** 在 `N` 个并行环境中收集 `T` 步。为每一步计算优势。在合并后的批次上同时更新演员和评论家。重复。它是 A3C 更简单、更可扩展的兄弟版本。

**A3C：异步优势 Actor-Critic（asynchronous advantage actor-critic）。** Mnih 等人（2016）。启动 `N` 个工作线程，每个线程运行一个环境。每个工作线程基于自己的回合片段本地计算梯度，然后异步应用到共享参数服务器。不需要经验回放缓冲区——工作线程通过运行不同轨迹来去相关。A3C 证明了可以在 CPU 上大规模训练。到 2026 年，基于 GPU 的 A2C（批处理并行环境）占据主导，因为 GPU 需要大批次。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度（policy-gradient）损失、价值回归、熵（entropy）奖励。`c_v ~ 0.5`、`c_e ~ 0.01` 是经典的起点。

## 动手实现

### 步骤 1：一个评论家

线性评论家 `V_φ(s) = w · features(s)`，用均方误差（MSE）更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格型环境中，评论家在几百个回合内收敛。在 Atari 上，把线性评论家换成共享 CNN 主干 + 价值头（value head）。

### 步骤 2：n 步优势

给定长度为 `T` 的回合片段（rollout）和自举（bootstrapped）终值 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是评论家的目标。`advantages` 是乘以 `∇ log π` 的量。

### 步骤 3：组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

同策略（on-policy），每次更新使用一个回合片段，演员和评论家使用不同的学习率。

### 步骤 4：并行化（A3C 与 A2C）

- **A3C：** 启动 `N` 个线程。每个线程运行自己的环境并做前向传播。定期将梯度更新推送到共享主节点。主节点不加锁——竞争没关系，只会增加噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠成 `[N, obs_dim]` 批次，做批处理前向传播和批处理反向传播。GPU 利用率更高、确定性更强、更容易理解。2026 年的默认选择。

我们的玩具代码为清晰起见是单线程的；改写成批处理 A2C 只需三行 numpy。

## 陷阱

- **演员梯度前的评论家偏差。** 如果评论家还是随机的，它的基线就没有信息量，你实际上在用纯噪声训练。先预热（warm up）评论家几百步再开启策略梯度，或者使用较慢的演员学习率。
- **优势归一化（advantage normalization）。** 每批次将优势归一化为零均值/单位标准差。以接近零的成本大幅提升训练稳定性。
- **共享主干网络（shared trunk）。** 在图像输入上，演员和评论家使用共享特征提取器。分成独立头（heads）。共享特征同时蹭到两个损失的信号。
- **同策略约定（on-policy contract）。** A2C 每条数据只复用一次。更多次会让梯度有偏（重要性采样修正是 PPO 所做的事）。
- **熵崩塌（entropy collapse）。** 如果 `c_e = 0`，策略在几百次更新后就会变得接近确定性并停止探索。
- **奖励缩放。** 优势的幅度取决于奖励尺度。对奖励做归一化（例如用运行标准差除）以在不同任务间保持一致的梯度幅度。

## 应用

A2C/A3C 在 2026 年很少是最终选择，但它们是之后所有方法的架构基础：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 裁剪重要性比率，支持多轮更新 |
| IMPALA | A3C + V-trace 异策略修正 |
| SAC（第 9 阶段 · 07） | 带软价值评论家（soft-value critic）的异策略 A2C（下一课） |
| GRPO（第 9 阶段 · 12） | 没有评论家的 A2C —— 使用组相对优势 |
| DPO | 坍缩成偏好排序损失的 A2C，无需采样 |
| AlphaStar / OpenAI Five | A2C + 联盟训练（league training）+ 模仿预训练 |

如果你在 2026 年的论文中看到 "advantage"，就想到 Actor-Critic。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: 为给定环境生成 A2C / A3C / GAE 配置，包括优势估计和损失权重。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定环境和计算预算，输出：

1. 并行方式。A2C（GPU 批处理）还是 A3C（CPU 异步），以及 worker 数量。
2. 回合片段长度 T。每次更新每个环境走多少步。
3. 优势估计器。n 步或 GAE(λ)；指定 λ。
4. 损失权重。`c_v`（价值）、`c_e`（熵）、梯度裁剪。
5. 学习率。演员和评论家（如果使用，可分开设置）。

拒绝在视界（horizon）> 1000 的环境上使用单 worker A2C（太同策略、太慢）。拒绝在没有优势归一化的情况下交付。如果某次运行 `c_e = 0` 且观测熵 < 0.1，标记为熵崩塌（entropy-collapsed）。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用蒙特卡洛优势（`G_t - V(s_t)`）训练 Actor-Critic。与第 06 课中 REINFORCE 加运行均值基线的样本效率进行比较。
2. **中等。** 切换为时序差分残差优势（`r + γ V(s') - V(s)`）。测量优势批次的方差。它下降了多少？
3. **困难。** 实现 GAE(λ)。对 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}` 做扫描。绘制最终回报与样本效率的关系。在这个任务中，偏差/方差的甜区在哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Actor（演员） | "策略网络" | `π_θ(a|s)`，由策略梯度更新。 |
| Critic（评论家） | "价值网络" | `V_φ(s)`，通过 MSE 回归拟合回报 / TD 目标。 |
| Advantage（优势） | "比平均好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计量。`∇ log π` 的乘数。 |
| TD residual（TD 残差） | "δ" | `δ_t = r + γ V(s') - V(s)`；单步优势估计。 |
| GAE | "插值旋钮" | 以 `λ` 为参数对 n 步优势做指数加权和。 |
| A2C | "同步 Actor-Critic" | 跨环境批处理；每个回合片段一次梯度步。 |
| A3C | "异步 Actor-Critic" | 工作线程将梯度推送到共享参数服务器。原始论文；2026 年已较少见。 |
| Bootstrap（自举） | "用视界处的 V 截断" | 截断回合片段，加上 `γ^n V(s_{t+n})` 来闭合求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) —— A3C，原始异步 Actor-Critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) —— GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— 基础；当评论家是神经网络时，结合第 9 章函数逼近一起阅读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) —— 带 V-trace 异策略修正的可扩展分布式 Actor-Critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) —— 值得阅读的生产级 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) —— 双时间尺度 Actor-Critic 分解的基础收敛性结果。
