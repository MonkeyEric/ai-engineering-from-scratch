# 深度 Q 网络（DQN）

> 2013 年，Mnih 在原始像素上训练了一个 Q-learning 网络，在七款 Atari 游戏上击败了所有经典强化学习（RL）智能体。2015 年，扩展到 49 款游戏并发表于《Nature》，开启了深度强化学习（deep-RL）时代。DQN 就是 Q-learning 加上三个让函数逼近（function approximation）稳定下来的技巧。

**类型：** Build
**语言：** Python
**前置知识：** Phase 3 · 03（反向传播），Phase 9 · 04（Q-learning、SARSA）
**时间：** 约 75 分钟

## 问题所在

表格型 Q-learning 需要为每个（状态，动作）对保存一个 Q 值。棋盘上大约有 10⁴³ 个状态。一帧 Atari 画面是 210×160×3 = 100,800 个特征。表格型强化学习在几千个状态时就束手无策，更不用说数十亿个状态。

事后看来，修复方案显而易见：用神经网络 `Q(s, a; θ)` 替换 Q 表。但“事后看来显而易见”这一点却花了几十年才达成共识。在“致命三元组（deadly triad）”——函数逼近（function approximation）+ 自助法（bootstrapping）+ 离线策略学习（off-policy learning）——的作用下，朴素的 Q-learning 函数逼近会发散。Mnih 等人（2013、2015）提出了三个工程技巧来稳定学习：

1. **经验回放（experience replay）** 解除转移之间的相关性。
2. **目标网络（target network）** 固定自助目标。
3. **奖励裁剪（reward clipping）** 归一化梯度幅值。

DQN 在 Atari 上的成功，首次证明了单一架构、同一组超参数就能从原始像素解决数十个控制问题。此后所有“深度强化学习”成果——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都建立在这三项技巧的基础之上。

## 核心概念

![DQN 训练循环：环境、经验回放缓冲区、在线网络、目标网络、贝尔曼 TD 损失](../assets/dqn.svg)

**目标函数。** DQN 最小化神经网络 Q 函数上的一步时序差分（TD）损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络（online network），每步通过梯度下降（gradient descent）更新。`θ^-` = 目标网络（target network），定期从 `θ` 复制（约每 10,000 步）。`D` = 存储过往转移的经验回放缓冲区（replay buffer）。

**按重要性排列的三项技巧：**

**经验回放。** 容量约 `10⁶` 的环形缓冲区。每个训练步均匀随机采样一个小批量（minibatch）。这打破了时间相关性（连续帧几乎相同），让网络能多次从稀有的高奖励转移中学习，并消除连续梯度更新之间的相关性。没有它，神经网络上的同策 TD 在 Atari 上会发散。

**目标网络。** 在贝尔曼方程两边使用同一个网络 `Q(·; θ)` 会让目标在每次更新时都移动——“追逐自己的尾巴”。解决方法是：维护第二个权重冻结的网络 `Q(·; θ^-)`。每 `C` 步复制 `θ → θ^-`。这让回归目标在数千次梯度步内保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是一种更平滑的变体。

**奖励裁剪。** Atari 的奖励幅值从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可防止单个游戏主导梯度。在奖励幅值本身很重要时这是错误的；但在 Atari 这种只看符号的场景下没有问题。

**双重 DQN（Double DQN）。** Hasselt（2016）修正了最大化偏差（maximization bias）：用在线网络来*选择*动作，用目标网络来*评估*该动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

即插即用的替换，效果稳定更好。默认使用它。

**其他改进（Rainbow，2017）：** 优先回放（prioritized replay，多采样 TD 误差大的转移）、对决架构（dueling architecture，分离的 `V(s)` 和优势头）、噪声网络（noisy networks，可学习的探索）、n 步回报（n-step returns）、分布式 Q（distributional Q，C51/QR-DQN）、多步自助。每一项都能带来几个百分点的提升；增益大致可叠加。

## 动手实现

这里的代码仅使用标准库、不含 numpy——我们在一个极小的连续 GridWorld 上手写单隐藏层多层感知机（MLP），因此每个训练步只需微秒级时间。其算法与大规模 Atari DQN 完全一致。

### 第一步：经验回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 通常需要约 50,000 容量；对我们的玩具环境来说 5,000 就够了。

### 第二步：一个微型 Q 网络（手写的 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是整个网络。

### 第三步：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

其形式与第 04 课的 Q-learning 相同，只有两个区别：（a）我们通过可微的 `Q(·; θ)` 反向传播，而不是查表；（b）目标使用 `Q(·; θ^-)`。

### 第四步：外层循环

每个回合中，基于 `Q(·; θ)` 执行 ε-贪婪策略，将转移存入缓冲区，采样小批量，执行梯度步，定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们这个 16 维 one-hot 状态的微型 GridWorld 上，智能体大约在 500 个回合内学到接近最优的策略。在 Atari 上，把规模放大到 2 亿帧，并加上 CNN 特征提取器即可。

## 常见陷阱

- **致命三元组（deadly triad）。** 函数逼近 + 离线策略 + 自助可能导致发散。DQN 通过目标网络 + 经验回放来缓解；两者缺一不可。
- **探索（exploration）。** ε 必须衰减，通常在训练前 10% 内从 1.0 降到 0.01。早期探索不足会导致 Q 网络收敛到局部 basin。
- **高估（overestimation）。** 对有噪声的 Q 取 `max` 会产生向上偏差。生产环境务必使用 Double DQN。
- **奖励尺度（reward scale）。** 裁剪或归一化奖励；梯度幅值与奖励幅值成正比。
- **经验回放冷启动（coldstart）。** 缓冲区中至少有几千条转移后再开始训练。仅用约 20 个样本得到的早期梯度会过拟合。
- **目标网络同步频率。** 太频繁 ≈ 没有目标网络；太稀疏 ≈ 目标过时。Atari DQN 使用 10,000 个环境步。经验法则：约每训练总时长的 1/100 同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧来让状态具有马尔可夫性。任何包含速度信息的环境都需要帧堆叠或循环状态。

## 何时使用

到了 2026 年，DQN 已很少是最先进的算法，但仍是离线策略（off-policy）算法的基准参考：

| 任务 | 首选方法 | 为什么不用 DQN？ |
|------|----------|----------------|
| 离散动作 Atari 类任务 | Rainbow DQN 或 Muesli | 同一框架，更多技巧。 |
| 连续控制 | SAC / TD3（Phase 9 · 07） | DQN 没有策略网络。 |
| 同策 / 高吞吐 | PPO（Phase 9 · 08） | 无需经验回放；更容易扩展。 |
| 离线强化学习 | CQL / IQL / Decision Transformer | 保守的 Q 目标，避免自助爆炸。 |
| 大离散动作空间（推荐系统） | 带动作嵌入的 DQN，或 IMPALA | 可用；具体设计很重要。 |
| LLM 强化学习 | PPO / GRPO | 序列级别而非步级别；损失不同。 |

这些经验仍然适用。经验回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自对弈缓冲区以及每一种离线强化学习方法中。奖励裁剪在 PPO 中以优势归一化的形式延续。这套架构就是蓝图。

## 交付产物

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每个回合的回报曲线。运行均值超过 -10 需要多少个回合？
2. **中等。** 禁用目标网络（贝尔曼目标两边都使用在线网络）。测量训练不稳定性——回报会震荡还是发散？
3. **困难。** 添加 Double DQN：用在线网络选择 `argmax a'`，用目标网络评估。在带噪声奖励的 GridWorld 上，比较训练 1,000 回合后 `Q(s_0, best_a)` 与真实 `V*(s_0)` 的偏差，分别在有和没有 Double DQN 的情况下。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| DQN | “深度 Q-learning” | 带有神经网络 Q 函数、经验回放缓冲区和目标网络的 Q-learning。 |
| Experience replay | “打乱的转移” | 每个梯度步均匀采样的环形缓冲区；消除数据相关性。 |
| Target network | “冻结的自助目标” | 用于贝尔曼目标的 Q 网络周期复制；稳定训练。 |
| Deadly triad | “RL 为什么会发散” | 函数逼近 + 自助 + 离线策略 = 没有收敛保证。 |
| Double DQN | “最大化偏差的修正” | 在线网络选动作，目标网络评估动作。 |
| Dueling DQN | “V 和 A 头” | 将 Q 分解为 V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | “所有技巧合一” | DDQN + PER + dueling + n-step + noisy + distributional 的组合。 |
| PER | “优先回放” | 按 TD 误差幅值成比例采样转移。 |

## 延伸阅读

- [Mnih 等（2013）。Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) —— 2013 年 NeurIPS 研讨会论文，开启了深度强化学习。
- [Mnih 等（2015）。Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) ——《Nature》论文，49 款游戏的 DQN。
- [Hasselt、Guez、Silver（2016）。Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) —— DDQN。
- [Wang 等（2016）。Dueling Network Architectures](https://arxiv.org/abs/1511.06581) —— dueling DQN。
- [Hessel 等（2018）。Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) —— 技巧叠加的论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) —— 清晰的现代讲解。
- [Sutton & Barto（2018）。第 9 章 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书式讲解“致命三元组”（函数逼近 + 自助 + 离线策略），DQN 的目标网络和经验回放正是为驯服它而设计。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) —— 消融研究中使用的参考单文件 DQN；适合与本课从零实现的版本对照阅读。
