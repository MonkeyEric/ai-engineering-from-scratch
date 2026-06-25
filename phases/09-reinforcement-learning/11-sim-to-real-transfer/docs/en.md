# 仿真到现实迁移（Sim-to-Real Transfer）

> 在仿真器里训练、在硬件上却失灵的策略，不过是一个背下了仿真器的策略。域随机化（domain randomization）、域自适应（domain adaptation）和系统辨识（system identification）是让学习得到的控制器跨越现实鸿沟（reality gap）的三大工具。

**类型：** Learn
**语言：** Python
**前置知识：** 第 9 阶段 · 第 08 课（PPO），第 2 阶段 · 第 10 课（偏差 / 方差）
**预计用时：** 约 45 分钟

## 问题背景

训练真实机器人缓慢、危险且昂贵。双足机器人需要数百万个训练回合才能学会行走；而真实双足机器人哪怕摔倒一次，也可能损坏硬件。仿真器（simulator）提供无限次重置、可复现的确定性结果、并行环境，并且不会造成物理损伤。

但仿真器并不完美。真实轴承的摩擦力高于 MuJoCo 模型；真实相机存在仿真器未建模的镜头畸变；真实电机存在延迟、背隙（backlash）和饱和（saturation）——99% 的仿真模型都会忽略这些。风、灰尘和多变的光照会摧毁只在“无菌”渲染环境中训练出的策略。现实鸿沟（reality gap），即仿真分布与真实分布之间的系统性差异，是强化学习（RL）机器人落地部署的核心难题。

你需要的是对**仿真到现实分布迁移具有鲁棒性**的策略。历史上主要有三类方法：随机化仿真器（域随机化）、用少量真实数据调整策略（域自适应 / 微调），或者辨识真实系统的参数并与之匹配（系统辨识）。到 2026 年，主流方案是三者结合，并辅以大规模并行仿真（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）。

## 核心概念

![三种仿真到现实迁移范式：域随机化、域自适应、系统辨识](../assets/sim-to-real.svg)

**域随机化（Domain Randomization，DR）。** Tobin 等人 2017，Peng 等人 2018。训练期间，对真实机器人上可能不同的每一个仿真参数进行随机化：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略会学习“今天处在哪个仿真中”的条件分布，从而在整个变化范围内泛化。只要真实机器人落在训练包络内，策略就能工作。

- **优点：** 不需要真实数据。一套方法，可复用到多种机器人。
- **缺点：** 过度随机化会训练出“通用”但过于保守的策略。噪声过大 ≈ 正则化过强。

**系统辨识（System Identification，SI）。** 在训练前，将仿真器参数拟合到真实数据。如果你能测量真实机械臂关节的摩擦力，就把它写进仿真器。然后训练一个期望这些值的策略。这种方法需要接触真实系统，但能直接缩小现实鸿沟。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残差模型误差（residual model error）对策略不可见；未被辨识到的小效应（例如电机死区，motor deadband）仍会导致部署失败。

**域自适应（Domain Adaptation）。** 先在仿真中训练，再用少量真实数据微调。有两种常见形式：

- **Real2Sim2Real：** 利用真实 rollout 学习残差仿真器 `f(s, a, z) - f_sim(s, a)`，然后在修正后的仿真器中训练。无需大量真实数据即可缩小差距。
- **观测自适应（observation adaptation）：** 训练一个策略，通过可学习的特征提取器（例如 GAN 像素到像素映射）将真实观测映射为类似仿真的观测。控制器仍运行在仿真域。

**特权学习 / 教师-学生（Privileged Learning / Teacher-Student）。** Miki 等人 2022（ANYmal 四足机器人）。在仿真中训练一个掌握**特权信息（privileged info）**的教师（teacher）——例如真实摩擦、地形高度、IMU 漂移。然后将知识蒸馏给只使用真实传感器观测的学生（student）。学生学会从历史观测中推断特权特征，从而在不同物理参数下保持鲁棒。

**大规模并行仿真。** 2024–2026 年。Isaac Lab、Mujoco MJX、Brax 都能在单张 GPU 上并行运行数千个机器人。PPO 配合 4,096 个并行人形机器人，几小时内就能收集相当于数年的经验。随着训练分布变宽，现实鸿沟随之缩小；当这 4,096 个环境各自拥有不同的随机化参数时，DR 的成本几乎为零。

**2026 年真实世界落地流程（以四足行走为例）：**

1. 在重力、摩擦、电机增益、负载（payload）均经过域随机化的大规模并行仿真中训练。
2. 用特权信息（地形图、真实 body 速度）训练教师策略。
3. 仅使用本体感知（proprioception，腿部关节编码器）将教师蒸馏为学生策略。
4. 可选：在真实 IMU 数据上通过自编码器（autoencoder）做观测自适应。
5. 部署。在 10 余种环境中零样本（zero-shot）运行。若失败，则使用带安全约束的 PPO 进行数分钟真实世界微调。

## 动手实现

本课代码是在带噪声转移的 GridWorld（网格世界）上对域随机化的小型演示。我们训练的策略会在“仿真”中经历随机化的打滑（slip）概率，并在训练时未见过的打滑“真实”水平上评估。其思想可直接映射到 MuJoCo 到真实硬件的迁移。

### 步骤 1：参数化仿真器

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是仿真器暴露的一个参数。在真实机器人中，它可以是摩擦、质量、电机增益——任何在仿真与真实之间会发生变化的量。

### 步骤 2：使用 DR 训练

每个回合开始时，从 `slip ~ Uniform[0.0, 0.4]` 中采样。然后用 PPO / Q 学习 / 任何你喜欢的方法训练。持续很多个回合。

### 步骤 3：在“真实”打滑上零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个位于训练支撑集内；`0.5` 和 `0.7` 在训练支撑集外。DR 训练的策略应在支撑集内保持接近最优，并在支撑集外 graceful 降级；而固定打滑训练的策略一旦离开训练打滑就会脆弱不堪。

### 步骤 4：与窄分布训练对比

用第二个策略，仅在 `slip = 0.0` 上训练。在同样的打滑扫描上评估。一旦真实打滑 > 0，回报应出现灾难性下降。

## 常见陷阱

- **随机化过度。** 在 `slip ∈ [0, 0.9]` 上训练，策略可能过于厌恶风险，连最优路径都不敢尝试。应匹配*预期*的真实世界分布，而不是“任何事情都可能发生”。
- **随机化不足。** 只在很窄的切片上训练，策略根本无法泛化。可使用自适应课程——自动域随机化（Automatic Domain Randomization，ADR）——在策略表现提升时逐步扩大分布范围。
- **参数空间选择错误。** 随机化错误的东西（例如相机色调，而真实差距是电机延迟），DR 就不会有帮助。先对真实机器人进行画像分析。
- **特权信息泄漏。** 教师若使用全局状态而非仅观测来做动作，学生可能无法追赶。确保教师的策略是学生仅凭观测历史就能实现的。
- **仿真到仿真迁移失败。** 如果你的策略对更难的仿真变体都不鲁棒，那它对真实世界也不会鲁棒。部署前务必先在留出（held-out）仿真变体上测试。
- **缺乏真实世界安全包络。** 一个在仿真中有效、“真实中也有效”的策略，若没有底层安全盾，仍可能损坏硬件。在非学习控制器中加入速率限制、力矩限制、关节限制。

## 实际应用

2026 年仿真到现实技术栈：

| 领域 | 技术栈 |
|------|--------|
| 腿式运动（ANYmal、Spot、人形机器人） | Isaac Lab + DR + 特权教师 / 学生 |
| 操作（灵巧手、抓取放置） | Isaac Lab + DR + 面向视觉的 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线自适应 |
| 手指 / 手内操作 | OpenAI Dactyl（前所未有规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + 少量真实微调 |

无论控制规模如何，流程都一致：尽可能拟合仿真器，对无法拟合的部分做随机化，训练超大规模策略，蒸馏，最后带着安全盾部署。

## 交付产物

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: 为指定机器人 + 任务规划仿真到现实迁移流程，覆盖 DR、SI 与安全。
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

给定机器人平台、任务以及真实硬件可用时间，输出：

1. 现实鸿沟清单。按预期影响排序的疑似来源（接触、感知、执行器延迟、视觉）。
2. DR 参数。精确列表、范围与分布。每个范围都需用真实测量值说明合理性。
3. SI 步骤。需要测量哪些参数；测量方法是什么。
4. 教师 / 学生拆分。教师使用哪些特权信息；学生使用哪些观测。
5. 安全包络。底层限制、急停、备用控制器。

若缺少以下三项，拒绝部署：(a) 零样本仿真变体测试，(b) 安全盾，(c) 回滚计划。若某 DR 范围超过真实 variability 的 3 倍，标记为可能过度随机化。
```

## 练习

1. **简单。** 在固定打滑 GridWorld（slip=0.0）上训练一个 Q 学习（Q-learning）智能体。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制回报随打滑变化的曲线。
2. **中等。** 训练一个 DR Q 学习智能体，从 `slip ~ Uniform[0, 0.3]` 中采样。在同样的扫描范围上评估。在分布外（out-of-distribution）的 slip=0.5 处，DR 能带来多大提升？
3. **困难。** 实现一个课程：从 slip=0.0 开始，当策略达到最优回报的 90% 时扩大 DR 范围。测量到达 slip=0.3 零样本性能所需的总环境步数，并与固定 DR 基线对比。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| 现实鸿沟（reality gap） | “仿真到真实的差异” | 训练与部署之间物理 / 感知的分布迁移。 |
| 域随机化（DR） | “在随机仿真中训练” | 训练期间随机化仿真参数，使策略具备泛化能力。 |
| 系统辨识（SI） | “测量真实并拟合仿真” | 估计真实物理参数，并将仿真器设置为与之匹配。 |
| 域自适应 | “在真实数据上微调” | 仿真训练后使用少量真实世界数据微调；可调整观测或动力学。 |
| 特权信息 | “给教师的真值” | 只有仿真器才有的信息；学生必须从观测历史中推断出来。 |
| 教师 / 学生 | “把特权蒸馏成可观测” | 教师借助捷径训练；学生学习如何在没有捷径的情况下模仿教师。 |
| ADR | “自动域随机化” | 随着策略提升逐步扩大 DR 范围的课程方法。 |
| Real2Sim | “用真实数据弥合差距” | 学习一个残差，使仿真器模仿真实 rollout。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) —— 最早的 DR 论文（面向机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) —— 面向动力学随机化的 DR，四足 locomotion。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) —— Dactyl，大规模 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) —— 面向 ANYmal 的教师-学生方法。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) —— 驱动 2025–2026 年部署的大规模并行仿真器。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) —— ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— Dyna 框架（用模型做规划与 rollout），是现代仿真到现实流程的底层思想。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) —— 仿真到现实方法的分类学与基准结果。
