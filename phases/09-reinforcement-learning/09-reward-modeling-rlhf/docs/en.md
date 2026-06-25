# 奖励建模与 RLHF

> 人类无法为“优秀的助手回复”写出一个奖励函数（reward function），但他们可以比较两条回复并挑出更好的那个。用这些偏好拟合一个奖励模型（reward model），再基于它用强化学习训练语言模型。Christiano 2017，InstructGPT 2022。正是这个配方把 GPT-3 变成了 ChatGPT。到了 2026 年，它大多已被 DPO 取代——但核心心智模型依然成立。

**类型：** 构建  
**语言：** Python  
**前置要求：** Phase 5 · 05（Sentiment），Phase 9 · 08（PPO）  
**时间：** 约 45 分钟

## 问题

你用下一个 token 预测目标训练了一个语言模型。它能写出语法正确的英文，但也会撒谎、胡扯，并且在该拒绝时不拒绝。继续预训练无法解决这些问题——网络文本就是病因，而不是解药。

你想要一个*标量奖励（scalar reward）*，用来表示“对于指令 X，回复 A 比回复 B 更好”。手工写出这样的奖励函数是不可能的。“有帮助性”并不是关于 token 的闭式表达式。但人类可以比较两个输出并标注偏好，而这种数据可以大规模低成本收集。

RLHF（Christiano et al. 2017；Ouyang et al. 2022）即“基于人类反馈的强化学习（Reinforcement Learning from Human Feedback）”，把偏好转换成奖励模型，再用 PPO 针对该奖励优化语言模型。三步走：SFT → RM → PPO。正是这个配方交付了 ChatGPT、Claude、Gemini 以及 2023–2025 年间所有对齐过的大语言模型。

到了 2026 年，PPO 这一步大部分已被 DPO（Phase 10 · 08）取代，因为它更便宜，在对齐微调上效果几乎一样好。但*奖励模型（reward model）*仍然是每个 Best-of-N 采样器、每个“可验证奖励强化学习”流程，以及每个使用过程奖励模型的推理模型的核心。理解了 RLHF，你就理解了整个对齐栈。

## 概念

![三阶段 RLHF：SFT、基于成对偏好的 RM 训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**第一阶段：监督微调（Supervised Fine-Tuning，SFT）。** 从一个预训练基座模型出发，用人工编写的高质量行为演示（遵循指令的回复、有帮助的回答等）进行微调。结果是模型 `π_SFT`，它*倾向于良好行为*，但动作空间仍然不受约束。

**第二阶段：奖励模型训练。**

- 针对提示 `x` 收集成对回复 `(y_+, y_-)`，人类标注“y_+ 优于 y_-”。
- 训练奖励模型 `R_φ(x, y)`，让 y_+ 获得更高分数。
- 损失函数是 **Bradley-Terry 成对逻辑斯蒂损失**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  其中 `σ` 是 sigmoid。奖励之差隐含着偏好的对数几率。Bradley-Terry 自 1952 年以来一直是标准选择，也是现代 RLHF 的主流目标函数。

- `R_φ` 通常由 SFT 模型初始化，顶部加一个标量输出头：同样的 transformer 骨干，单层线性层输出奖励。

**第三阶段：基于 RM 的 PPO，并加入 KL 惩罚。**

- 用 `π_SFT` 初始化可训练策略 `π_θ`，并冻结一个*参考模型* `π_ref = π_SFT`。
- 对完整回复 `y` 的总奖励为：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是一种*正则化项*，而不是硬约束的信任域。`β` 通常取 `0.01`–`0.05`。
- 用该奖励运行 PPO（Lesson 08）。优势（advantage）在 token 级别轨迹上计算，但 RM 只对整个回复打分。

**为什么需要 KL 惩罚？** 没有它，PPO 会很乐意找到奖励作弊（reward hacking）策略——RM 只在训练分布内的补全上训练过。分布外的回复可能比任何人类写的回复得分都高。KL 惩罚让 `π_θ` 停留在 RM 被训练过的数据流形附近。它是 RLHF 中最重要的超参数。

**2026 年的现状：**

- **DPO**（Rafailov 2023）：用闭式代数把第二、三阶段折叠成对偏好数据的单一监督损失。不需要 RM，也不需要 PPO。在对齐基准上质量相当，计算成本却低得多。将在 Phase 10 · 08 中介绍。
- **GRPO**（DeepSeek 2024–2025）：PPO 的变体，使用组相对基线（group-relative baseline）替代价值网络，奖励来自*验证器*（代码能运行 / 数学答案匹配）而非人类训练的 RM。在推理模型中占主导地位。将在 Phase 9 · 12 中介绍。
- **过程奖励模型（Process Reward Models，PRM）**：为部分解法（每个推理步骤）打分，用于 RLHF 与 GRPO 的推理变体。
- **Constitutional AI / RLAIF**：用已对齐的大语言模型生成偏好，而非人类。可放大偏好数据的规模。

## 动手构建

本课使用极小的合成“提示”和“回复”，表示为字符串。RM 是一个基于词袋（bag-of-tokens）表示的线性打分器。不使用真实的大语言模型——重要的是*流程形态*，而不是规模。参见 `code/main.py`。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，这部分由人工标注者完成。数据形态——`(prompt, preferred_response, rejected_response)`——完全相同。

### 步骤 2：Bradley-Terry 奖励模型

线性打分：`R(x, y) = w · bag(y)`。训练目标是最小化 BT 成对 log-loss：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

经过几百次更新后，`w` 会给“好词” token 赋予正权重，给“坏词” token 赋予负权重。

### 步骤 3：基于 RM 的类 PPO 策略

我们的玩具策略从词表中生成单个 token。我们在 RM 下对该 token 打分，计算 `log π_θ(token | prompt)`，加上相对参考模型的 KL 惩罚，并应用带裁剪的 PPO 替代目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # 对 theta 执行 PPO 风格更新，把 reward 视为 return
    ...
```

### 步骤 4：监控 KL

每次更新都跟踪平均 `KL(π_θ || π_ref)`。如果它缓慢超过 `~5-10`，说明策略已显著偏离 `π_SFT`——要么 `β` 太低，要么奖励作弊开始出现。这是真实 RLHF 中的头号诊断指标。

### 步骤 5：使用 TRL 的生产级配方

理解玩具流程后，下面是真实库用户编写的同样循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 对应第二阶段，`PPOTrainer`（内置 KL-to-reference）对应第三阶段。

```python
# 第二阶段：从成对偏好训练奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 数据集行格式：{"prompt", "chosen", "rejected"} —— Bradley-Terry 格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 第三阶段：带 KL 惩罚的 PPO，优化目标是 RM，参考模型为 SFT 检查点
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 冻结

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats 包括：mean_kl、clip_frac、value_loss —— 三项 PPO 诊断指标
```

库替你做了三件事。`adap_kl_ctrl=True` 实现了自适应 `β` 调度：如果观测到的 KL 超过 `target_kl`，`β` 翻倍；如果低于其一半，`β` 减半。参考模型按约定被冻结——你必须避免它与 `policy` 共享参数。价值头与策略共用同一骨干（`AutoModelForCausalLMWithValueHead` 会附加一个标量 MLP 头），因此 TRL 会分别报告 `policy/kl` 和 `value/loss`。

## 常见陷阱

- **过度优化 / 奖励作弊。** RM 并不完美；`π_θ` 会找到得分高但质量差的对抗性补全。症状：奖励持续上升，而人工评估分数停滞或下降。修复：早停、提高 `β`、扩充 RM 训练数据。
- **长度作弊。** 在有帮助性回复上训练的 RM 常常隐式地奖励长度，策略学会填充内容。补救：长度归一化奖励，或使用带有长度感知 RM 的 RLAIF。
- **RM 太小。** RM 至少需要与策略一样大。小 RM 无法忠实评估策略的输出。
- **KL 调参。** `β` 太低 → 漂移与奖励作弊；`β` 太高 → 策略几乎不更新。标准技巧是使用*自适应* `β`，使每步 KL 保持在固定目标。
- **偏好数据噪声。** 约 30% 的人工标注有噪声或含混。可通过只在标注一致的数据上训练 RM，或在 BT 上应用温度来校准。
- **离线策略问题。** 第一个 epoch 之后，PPO 数据会略带离线策略性质。像 Lesson 08 一样监控裁剪比例（clip fraction）。

## 应用场景

2026 年的 RLHF 是分层的：

| 层级 | 目标 | 方法 |
|------|------|------|
| 指令遵循、 helpfulness、harmlessness | 对齐 | DPO（Phase 10 · 08）优先于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 使用验证器奖励的 GRPO（Phase 9 · 12）。 |
| 长程多步任务 | 智能体 | 基于步骤过程奖励模型的 PPO / GRPO。 |
| 安全性 / 拒绝行为 | 安全 | 带独立安全 RM 的 RLHF-PPO，或 Constitutional AI。 |
| 推理时 Best-of-N | 快速对齐 | 在解码阶段使用 RM，无需训练策略。 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 上训练小型“奖励头”。 |

RLHF 曾是 2022–2024 年的*主*方法。到了 2026 年，生产级对齐流程以 DPO 为首选，只在需要 RM 密集或安全关键的步骤中保留 PPO。

## 交付

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: 为语言模型设计 RLHF / DPO / GRPO 对齐流程，包括 RM、KL 与数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定一个基座 LM、目标行为（对齐 / 推理 / 拒绝 / 智能体）以及偏好或验证器预算，输出：

1. 阶段。SFT？RM？DPO？GRPO？给出理由。
2. 偏好或验证器来源。人类、AI 反馈、基于规则的、单元测试通过，或奖励蒸馏。
3. KL 策略。固定 β、自适应 β，或 DPO（隐式 KL）。
4. 诊断指标。平均 KL、奖励稳定性、过度优化保护（留出的人工评估）。
5. 安全门。红队测试集、拒绝率、与 helpfulness RM 分离的安全 RM。

如果缺少 KL 监控，拒绝交付 RLHF-PPO。如果 RM 比目标策略小，拒绝使用。拒绝仅依赖长度的奖励。对任何未保留盲测人工评估集的流程，标记为缺乏过度优化保护。
```

## 练习

1. **简单。** 在 `code/main.py` 中训练 Bradley-Terry 奖励模型，使用 500 对合成偏好数据。在 100 对留出数据上测量成对准确率，应超过 90%。
2. **中等。** 在玩具 PPO-RLHF 循环中分别尝试 `β ∈ {0.0, 0.1, 1.0}`。每种情况绘制 RM 分数与相对参考模型 KL 随更新的变化曲线。哪种设置发生了奖励作弊？
3. **困难。** 在同一偏好数据上实现 DPO（闭式偏好似然损失），并与 RLHF-PPO 流程在计算消耗和最终 RM 分数上比较。

## 关键术语

| 术语 | 别人怎么说 | 实际含义 |
|------|-----------|---------|
| RLHF | “对齐 RL” | 三阶段 SFT + RM + PPO 流程（Christiano 2017, Ouyang 2022）。 |
| Reward Model（RM） | “打分网络” | 通过 Bradley-Terry 拟合成对偏好的学习标量函数。 |
| Bradley-Terry | “成对逻辑斯蒂损失” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 | “靠近参考模型” | 奖励中的 `β · KL(π_θ || π_ref)`；防止奖励作弊的正则化项。 |
| Reward hacking | “古德哈特定律” | 策略利用 RM 缺陷；症状：奖励上升，人工评估持平。 |
| RLAIF | “AI 标注的偏好” | 标签来自另一个语言模型而非人类的 RLHF。 |
| PRM | “过程奖励模型” | 为部分推理步骤打分；用于推理流程。 |
| Constitutional AI | “Anthropic 的方法” | 在显式规则指导下由 AI 生成偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) —— 开创 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) —— 早期用于摘要的 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) —— DPO；2026 年后 RLHF 的默认替代。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) —— RLAIF 与自我批评循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) —— HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) —— 生产级 `RewardTrainer` 与 `PPOTrainer`。阅读 trainer 源码以了解自适应 KL 与价值头细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf)，作者 Lambert、Castricato、von Werra、Havrilla —— 三阶段流程的经典图解教程。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) —— 库本身；`examples/` 包含 Llama、Mistral、Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) —— 奖励假设视角；思考奖励作弊的必读前提。
