# 面向游戏的强化学习 —— AlphaZero、MuZero 与大语言模型推理时代

> 1992 年：TD-Gammon 仅凭时序差分（TD）就在西洋双陆棋上击败人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始统治国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明同样的配方，只是用 GRPO 取代 PPO，就能在推理任务上奏效。游戏是这一阶段每一次突破背后的基准。

**类型：** Build
**语言：** Python
**前置知识：** 第 9 阶段 · 第 05 课（DQN）、第 9 阶段 · 第 08 课（PPO）、第 9 阶段 · 第 09 课（RLHF）、第 9 阶段 · 第 10 课（MARL）
**时长：** 约 120 分钟

## 问题背景

游戏具备强化学习想要的一切：干净的奖励（胜/负）、无限回合（自弈重置）、完美模拟器（游戏本身就是模拟器）、离散或小型连续动作空间，以及迫使对抗鲁棒性的多智能体结构。

而且，游戏也是每一次重大强化学习突破的试金石。TD-Gammon（西洋双陆棋，1992）、Atari-DQN（2013）、AlphaGo（2016）、AlphaZero（2017）、OpenAI Five（Dota 2，2019）、AlphaStar（星际争霸 II，2019）、MuZero（学习模型，2019）、AlphaTensor（矩阵乘法，2022）、AlphaDev（排序算法，2023）、DeepSeek-R1（数学推理，2025）—— 最新的证据表明，游戏强化学习技术同样适用于文本。

本综合课程通过统一的视角 **自弈 + 搜索 + 策略改进** 来梳理三大里程碑架构：AlphaZero、MuZero 和 GRPO。每一种方法都是对前者的泛化；GRPO 尤其像是把 AlphaZero 的配方应用到了大语言模型（LLM）推理上，只不过动作变成了 token，胜利信号变成了数学验证。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO：同一循环，不同环境](../assets/rl-games.svg)

**统一循环。**

```python
while True:
    trajectory = self_play(current_policy, search)     # 智能体与自己对弈
    policy_target = search.improved_policy(trajectory) # 搜索改进原始策略
    policy_net.update(policy_target, value_target)     # 以搜索输出为监督目标
```

**AlphaZero（2017 年）。** Silver 等人提出。给定已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络（policy-value network）：一个共享塔 `f_θ(s) → (p, v)`。`p` 是合法动作的稀疏先验，`v` 是预期对局结果。
- 蒙特卡洛树搜索（Monte Carlo Tree Search，MCTS）：在每一步展开一棵可能续走树，用 `(p, v)` 作为先验与自举（bootstrap）信号，通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自弈（self-play）：智能体与自己对弈。在第 `t` 步，MCTS 的访问分布 `π_t` 成为策略的训练目标。
- 损失（loss）：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是真实对局结果（+1 / 0 / -1）。

零人类知识、零手工启发式。同一套方法在数千万盘自弈后掌握了国际象棋、将棋和围棋。

**MuZero（2019 年）。** Schrittwieser 等人提出。不再要求已知规则。

- 用一个*潜在动力学模型（latent dynamics model）* `(h, g, f)` 替代固定环境：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一潜在状态 + 即时奖励（reward）。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习到的潜在空间*中运行。搜索与训练循环保持不变。
- 适用于围棋、国际象棋、将棋 *以及* Atari —— 一种算法，无需规则知识。

**随机 MuZero（2022 年）。** 引入随机动力学与机会节点，可扩展到西洋双陆棋类游戏。

**Muesli、Gumbel MuZero（2022–2024 年）。** 在样本效率与确定性搜索上的改进。

**GRPO（2024–2025 年）。** DeepSeek-R1 配方。同样是 AlphaZero 形状的循环，但应用于语言模型推理：

- “游戏”：回答一个数学 / 编程 / 推理问题。“获胜” = 验证器（verifier，如测试用例通过、数值答案匹配）返回 1。
- 策略（policy）：LLM 本身。动作（action）：token。状态（state）：提示词（prompt）+ 已生成回复。
- 无需评论网络（critic，类似 PPO 中的 V_φ）。对每个提示词，从策略中采样 `G` 个补全（completion），计算每个补全的奖励（reward），再用 **组相对优势（group-relative advantage）** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格的更新信号。
- KL 惩罚（KL penalty）拉回参考策略（reference policy），防止偏移（与 RLHF 类似）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无需奖励模型（reward model）、无需评论网络、无需 MCTS。组相对基线（group-relative baseline）同时替代了这三者。在推理基准上达到或超过 PPO-RLHF 的效果，而计算量仅为其一小部分。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek，2025）在一篇论文中实际上包含两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基座模型出发。没有 SFT，直接应用 GRPO，奖励包含两个部分：*准确性奖励（accuracy reward）*（基于规则 —— 最终答案是否解析为正确数字 / 代码是否通过单元测试）和 *格式奖励（format reward）*（补全是否将思维链包裹在 `<think>…</think>` 标签中）。经过数千步训练，平均回复长度从约 100 个 token 增长到约 10,000 个，数学基准成绩接近 o1-preview。模型从零学会了推理。缺点：其思维链往往难以阅读、混合语言、缺乏风格打磨。
- **R1。** 用四阶段流程修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千条格式干净的长思维链示范，对基座模型进行监督微调。得到一个可读的起点。
  2. **面向推理的 GRPO。** 在准确性奖励 + 格式奖励基础上，增加 *语言一致性奖励（language-consistency reward）* 防止代码切换（code-switching）。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点采样约 60 万条推理轨迹，仅保留答案正确且思维链可读的样本，并与约 20 万条非推理 SFT 样本（写作、问答、自我认知）合并，再次微调基座模型。
  4. **全谱 GRPO。** 最后一轮 RL，同时覆盖推理任务（基于规则的奖励）和通用对齐任务（ helpfulness/harmlessness 偏好奖励）。

最终模型在 AIME 和 MATH-500 上达到 o1 水平，并以开放权重发布，且小到可以蒸馏。同一篇论文还发布了六个稠密蒸馏模型（Qwen-1.5B 到 Llama-70B），通过对 R1 的推理轨迹做 SFT 得到 —— 学生端无需 RL。对强大 RL 教师进行蒸馏，在学生规模上始终优于从头做 RL。

**为什么用 GRPO 而不是 PPO 做推理。** DeepSeekMath 论文（2024 年 2 月）给出三个原因：(1) 无需训练价值网络（value network），显存减半；(2) 组基线天然适合推理任务产生的稀疏端到端奖励；(3) 每个提示词单独归一化，使优势在不同难度的问题上可比，而 PPO 的单一评论网络做不到。

**无搜索 vs 基于搜索。** 游戏领域已经分化：

- *长时程完美信息博弈*（围棋、国际象棋）：仍基于搜索。AlphaZero / MuZero 占主导。
- *LLM 推理*：生产环境中尚未使用 MCTS；GRPO 在完整 rollout 上训练，推理时用 best-of-N 换取计算量。过程奖励模型（Process Reward Models，PRMs）暗示着逐步搜索可能会被重新引入。

## 动手实现

`code/main.py` 中的代码实现了**迷你版 GRPO** —— 一个带多组样本的多臂老虎机（bandit）。算法与 LLM 上的 GRPO 完全相同，只是策略（policy）和环境更简单。它用于教学 2025 年的核心创新：*损失函数*和*组相对优势*。

### 第 1 步：微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中，验证器运行单元测试或检查数学等式。

### 第 2 步：策略：每个提示词对 K 个答案 token 做 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于 LLM 在给定提示词下最后一层输出。

### 第 3 步：组采样与组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 惩罚：将 theta 拉回参考策略
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的关键技巧。无需评论网络。基线（baseline）就是组内均值，归一化使用组内标准差。

### 第 4 步：与无基线 REINFORCE 对比

相同设置、相同计算量、普通 REINFORCE。GRPO 收敛更快且更稳定。

### 第 5 步：观察熵与 KL

与 RLHF 使用相同的诊断指标：到参考策略的平均 KL、策略熵（entropy）、随时间变化的奖励。一旦这些指标稳定，训练即可结束。

## 常见陷阱

- **通过欺骗验证器实现奖励黑客（reward hacking）。** GRPO 继承了 RLHF 的风险：如果验证器有误或被利用，LLM 会找到漏洞。因此验证器必须鲁棒（多组测试用例、形式化证明）。
- **组大小过小。** 组基线的方差按 `1/√G` 缩放。`G < 4` 时优势信号嘈杂；通常选择 `G = 8` 到 `64`。
- **长度偏差（length bias）。** 不同长度的 LLM 补全具有不同的对数概率。应按 token 数量归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自弈循环。** AlphaZero 风格的训练在一般和博弈中可能陷入 dominance 循环。可通过多样化对手池缓解（联盟训练，见第 10 课）。
- **搜索-策略不匹配。** AlphaZero 训练策略网络拟合搜索输出。如果策略网络太小，无法表示搜索的分布，训练会停滞。
- **计算门槛。** MuZero / AlphaZero 需要大量计算。单个消融实验常常需要数百 GPU 小时。也有迷你演示（如 Connect Four 上的 AlphaZero）供学习使用。
- **验证器覆盖不足。** 如果单元测试对某个有 bug 的解法也通过了，就会强化这个 bug。应设计能捕捉边界情况的验证器。

## 应用指南

2026 年游戏强化学习各领域的主导方法：

| 领域 | 主流方法 |
|--------|-----------------|
| 双人零和棋类（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 非完美信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人在线策略游戏（Dota、星际争霸） | PPO + 自弈 + 联盟（OpenAI Five、AlphaStar） |
| LLM 数学 / 代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开源复现） |
| LLM 对齐 | DPO / RLHF-PPO（不是 GRPO；验证器是偏好而非可验证） |
| 机器人 | PPO + 域随机化（DR，不属于游戏 RL，但使用同样的策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

这一*配方* —— 自弈、搜索增强改进、策略蒸馏 —— 横跨文本、像素和物理控制。GRPO 是最年轻的实例；更多变体还在路上。

## 交付产物

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 老虎机。训练 2 个提示词 × 每个 4 个答案 token，在 `G=8` 时于 1,000 次更新内收敛。
2. **中等。** 接入 PPO（clipped）和普通 REINFORCE。在相同老虎机上比较样本效率与奖励方差。
3. **困难。** 扩展为长度 2 的“推理链”：智能体输出两个 token，验证器奖励成对结果。测量 GRPO 如何处理两步序列上的信用分配。（提示：按*完整序列*计算组优势，并传播到两个 token 位置。）

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| MCTS | “带学习网络的树搜索” | 蒙特卡洛树搜索；用学习到的 `(p, v)` 先验进行 UCB1/PUCT 选择。 |
| AlphaZero | “自弈 + MCTS” | 策略-价值网络，训练目标为匹配 MCTS 访问分布与对局结果。 |
| MuZero | “学习模型的 AlphaZero” | 相同循环，但在学习到的潜在空间中运行。 |
| GRPO | “无评论网络的 PPO” | 组相对策略优化（Group Relative Policy Optimization）；带组均值基线与 KL 惩罚的 REINFORCE。 |
| PUCT | “AlphaZero 的 UCB” | `Q + c · p · √N / (1 + N_a)` —— 平衡价值估计与先验。 |
| Self-play | “智能体与过去的自己对弈” | 零和博弈的标准做法；对称训练信号。 |
| League play | “基于群体的自弈” | 从历史、当前和专门克制者中采样对手。 |
| Verifier reward | “可验证 RL” | 奖励来自确定性检查器（测试通过、答案匹配）。 |
| Process reward | “PRM” | 对每个推理步骤打分，而非仅对最终答案打分。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) —— 提出 GRPO 与组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) —— 完整四阶段 R1 配方及 R1-Zero 消融。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) —— 大规模 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) —— 这一切的起点。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) —— 使用自定义奖励函数应用 GRPO 的生产级参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) —— R1 配方在多规模上的开源复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) —— 关于自弈、搜索与“设计奖励”的教科书框架，R1 在 LLM 规模上实现了它。
