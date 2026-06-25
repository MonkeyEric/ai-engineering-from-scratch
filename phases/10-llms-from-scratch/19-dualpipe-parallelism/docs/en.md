# DualPipe 流水线并行（DualPipe Parallelism）

> DeepSeek-V3 在 2,048 张 H800 GPU 上训练，混合专家（MoE）的专家被分散在不同节点之间。跨节点专家 all-to-all 通信的代价是：每消耗 1 GPU 小时的计算，就要消耗 1 GPU 小时的通信，GPU 有一半时间处于空闲状态。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，它将前向与反向计算同它们触发的 all-to-all 通信重叠执行。流水线气泡（bubble）被压缩，吞吐量提升；而在专家并行（Expert Parallelism，EP）已经把专家分散到不同 rank 的前提下，保存两份模型参数副本（也就是“dual”命名的由来）所带来的额外开销相对较小。本节课是一次 Learn 类型的 walkthrough，讲解 DualPipe 实际做了什么，以及为什么 Sea AI Lab 的 DualPipeV 改进版本会以略大一些的气泡为代价，消除 2 倍参数复制开销。

**类型：** Learn
**语言：** Python（stdlib，调度模拟器）
**前置知识：** Phase 10 · 05（分布式训练、FSDP、DeepSpeed），Phase 10 · 14（开源模型架构与 MoE）
**时间：** 约 60 分钟

## 学习目标

- 说出 DualPipe 前向-反向数据块（chunk）的四个组成部分，以及为什么每个部分都有独立的重叠窗口。
- 解释大规模下的流水线气泡问题，以及“无气泡（bubble-free）”在实践与宣传中分别意味着什么。
- 手动推导 P=8 个流水线并行（PP）rank、16 个微批次（micro-batches）下的 DualPipe 调度表，并确认前向与反向流如何填满彼此的闲置槽位。
- 说明 DualPipeV（Sea AI Lab，2025）所做的权衡：当专家并行（EP）不活跃时，以略大一些的气泡为代价，取消 2 倍参数复制。

## 问题背景

在 2,000 张 H800 GPU 上训练一个 671B 参数的 MoE 模型，会同时遇到三个相互叠加的瓶颈：

1. **显存压力。** 每张 GPU 只保存模型的一部分切片。在序列长度 8k、61 层、128 个注意力头的情况下，激活值（activation）显存非常巨大。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）会让 GPU 在等本阶段输入或梯度时处于空闲状态。即使在 1F1B 调度下，8 个阶段也会浪费约 12% 的 GPU 时间作为气泡。
3. **跨节点 all-to-all。** MoE 配合专家并行会把专家分散到不同节点。每次前向传播都会触发一次 all-to-all 把 token 派发给对应专家，以及一次 all-to-all 把专家输出聚合回来。在 2,000 张 GPU 的规模下，通信与计算的比率很容易达到 1:1。

这些问题各有单独解法：梯度检查点（gradient checkpointing）缓解显存，Zero Bubble（Sea AI Lab，2023）减少流水线气泡，专家并行通信内核优化 all-to-all。而 DualPipe 的作用是让它们协同工作：在单个前向-反向数据块内部把计算与通信重叠，同时从流水线两端注入微批次，并利用生成的调度表把 all-to-all 隐藏在计算窗口中。

论文报告的结果：流水线气泡几乎被消除，DeepSeek-V3 的 14.8T token 训练过程中平均 GPU 利用率超过 95%。

## 核心概念

### 流水线并行回顾

把一个 N 层模型切分到 P 个设备上。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个微批次前向流经设备 0 到 P-1，然后反向从 P-1 流回 0。每个设备只有在前一设备发送输出后才能开始本阶段前向，只有在后一设备发送上游梯度后才能开始本阶段反向。

GPipe（Huang 等，2019）一次只调度一个微批次，浪费了大部分 GPU 时间。1F1B（Narayanan 等，2021）交错多个微批次的前向与反向。Zero Bubble（Qi 等，2023）把反向传播拆成两部分——输入梯度（B）与权重梯度（W）——并调度它们以填充气泡。经过 Zero Bubble 之后，流水线已经非常紧凑。

DualPipe 是下一步。它在上面叠加了两个想法：

### 想法 1：数据块分解

每个前向数据块被拆成四个组成部分：

- **注意力（Attention）。** Q/K/V 投影、注意力计算、输出投影。
- **All-to-all 派发（dispatch）。** 跨节点通信，把 token 发送给对应的专家。
- **MLP。** MoE 专家计算。
- **All-to-all 聚合（combine）。** 跨节点通信，把专家输出带回来。

反向数据块则对每一部分求梯度。DualPipe 调度它们，使得 all-to-all 派发与下一个数据块的注意力计算并行，all-to-all 聚合与再下一个数据块的 MLP 计算并行。

### 想法 2：双向调度

大多数流水线调度只从阶段 0 注入微批次并向阶段 P-1 流动。DualPipe 则从**两端**同时注入微批次：阶段 0 会看到自己发起的正向微批次，阶段 P-1 也会看到自己发起的正向微批次。两股流在流水线中间相遇。

要做到这一点，设备 `i` 必须同时保存**靠前层** `i` 与**靠后层** `P - 1 - i`。这就是 DualPipe 中“dual”的含义：每个设备保存两份它所需模型层的副本（分别服务两个方向）。在 DeepSeek-V3 的规模下，这会带来 2 倍参数复制开销。但由于专家并行已经把 MoE 专家切得非常细，复制非专家层的开销相对而言只是“小意思”。

关键在于：一个方向的前向流与另一个方向的反向流，正好在单向调度会产生气泡的位置重叠。气泡因此消失。

### 一个手工推导的调度示例

考虑 P = 4 个 rank，8 个微批次，其中 4 个正向、4 个反向。时间从左到右推进，每行代表一个设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

解读 “F4/F5R” 这种记号：rank 1 在同一个时间槽里同时运行微批次 4 的正向（在流水线中从左到右）和微批次 5 的正向（从右到左）。这就是“双向”在操作层面的含义。

在 rank 2 处，两股流更早重叠；在 rank 0 和 P-1 处，重叠最晚。在调度表的稳定中间阶段，每个 rank 都同时运行某个方向的前向与另一个方向的反向。计算保持忙碌：前向的 all-to-all 派发隐藏在反向计算中，all-to-all 聚合隐藏在前向计算中。气泡被挤压出去。

### 气泡核算

标准 1F1B 流水线的气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进后有所下降，但未降到零。DualPipe 在稳定阶段可以做到零气泡，前提是微批次数量能被 2 倍流水线深度整除。在稳定阶段之外（预热与冷却阶段）仍有少量气泡，但它不会随微批次数量增加而增长——这是论文强调的关键性质。

宣传语境下：它被称为“无气泡”。技术语境下：气泡不随微批次数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）指出，只有当专家并行不是瓶颈时才能实现完全零气泡；在 EP 驱动的 all-to-all 场景下，总有一些调度上的折中。

### DualPipeV —— 改进版本

Sea AI Lab（2025）观察到，当 EP 通信重叠不再是关注重点时，2 倍参数复制是浪费的。他们的 DualPipeV 调度把双向注入折叠成一种“V 形”调度，只需单份参数副本。气泡比 DualPipe 略大，但显存节省非常可观。DeepSeek 在其开源 DualPipe 实现中把 DualPipeV 作为 EP 关闭模式采用。

权衡对比：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每设备参数副本数 | 2 | 1 | 1 | 1 |
| 气泡与微批次的关系 | 恒定 | 小幅增长 | 增长 | 增长 |
| 计算-通信重叠 | 完全 | 部分 | 最小 | 部分 |
| 适用场景 | 重度 EP 的 MoE | 稠密模型或轻量 EP | 基线 | 任意流水线 |

### 对 14.8T token 训练意味着什么

DeepSeek-V3 的预训练在 2,048 张 H800 GPU 上消耗了约 280 万 GPU 小时，处理 14.8T token。如果使用朴素的 1F1B，大约会损失 12–15% 的时间给流水线气泡——即 34–42 万 GPU 小时，足够训练一个完整的 70B 模型。DualPipe 回收了其中的大部分。没有内部日志很难直接量化，但论文宣称训练期间平均 GPU 利用率超过 95%。

对于较小规模（1,000 张 GPU 以下），DualPipe 有些杀鸡用牛刀——流水线气泡占总成本的比例较小，稠密模型训练也很少遇到 all-to-all 瓶颈。但对于数千 GPU 规模的前沿 MoE 训练，它几乎是必需的。

### 在软件栈中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 跨 rank 分片模型参数，DualPipe 跨 rank 调度计算，两者可以结合。
- 兼容 **ZeRO-3** 梯度分片。双副本复制的簿记工作需要与 ZeRO 的分片梯度协同。
- 需要针对特定集群拓扑调优的**自定义 all-to-all 内核**。DeepSeek 的开源内核是参考实现。

## 动手使用

`code/main.py` 是一个流水线调度模拟器。它接收 `(P, n_micro_batches, schedule)` 并打印 1F1B、Zero Bubble、DualPipe 与 DualPipeV 在稳定阶段的利用率。这是一个教学工具——其数值与论文中的定性结论一致，并不构成对生产实测加速比的断言。

模拟器的价值在于：用不同的 P 和微批次数量运行它，观察 1F1B 的气泡占比如何增长，而 DualPipe 不会。

真实训练运行中的集成注意事项：

- 选择能整除微批次数量的流水线并行深度。
- 确保专家并行网格支持双向 all-to-all。DeepSeek 的内核是参考实现。
- 首次调试该调度表时，预计要花费一周时间。簿记工作非常繁琐。
- 监控每个 rank 的 GPU 利用率，而不仅是聚合指标。DualPipe 的收益来自拉平最慢的 rank。

## 成果交付

本节课会产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑、互联带宽、模型结构），它会推荐流水线并行策略、应使用的调度算法，以及目标规模下预期的气泡占比。

## 练习题

1. 运行 `code/main.py`，分别使用 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)`。计算 GPU 利用率的差异，并以每百万 token 训练所回收的 GPU 小时数表达。

2. 手工绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。在每个时间槽中标记微批次 ID 与方向，并指出第一个没有气泡的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。识别 DualPipe 前向数据块中 all-to-all 派发的重叠窗口，并解释计算调度如何把它隐藏起来。

4. 计算 DualPipe 的 2 倍参数开销：一个 P=8 流水线阶段的 70B 稠密模型，以及一个 P=16 流水线阶段的 671B MoE 模型。说明为什么 MoE 场景下的开销比例更小（大部分参数是专家，被分片到很大的 EP 组中）。

5. 将 DualPipe 与 Chimera（2021 年的竞争双向调度器）进行对比。参考论文第 3.4 节，指出 DualPipe 新增而 Chimera 不具备的两个具体特性。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------------|------------------------|
| 流水线气泡（Pipeline bubble） | “每个 rank 的空闲时间” | 流水线阶段等待输入或梯度而浪费的 GPU 周期 |
| 1F1B | “默认流水线调度” | 一次前向 / 一次反向交错调度；DualPipe 击败的基线 |
| Zero Bubble | “Sea AI Lab 2023” | 把反向拆分为 B（输入梯度）和 W（权重梯度）；几乎完全拉紧流水线 |
| DualPipe | “DeepSeek-V3 调度” | 双向流水线 + 计算-通信重叠；气泡不随微批次数量增长 |
| DualPipeV | “Cut-in-half” | V 形改进版本，以略大一些的气泡为代价取消 2 倍参数复制 |
| 数据块（Chunk） | “流水线工作单位” | 一个微批次通过一个流水线阶段的一次前向或反向传播 |
| All-to-all 派发 | “把 token 发送给专家” | 把 token 路由到对应 MoE 专家的跨节点通信 |
| All-to-all 聚合 | “把专家输出带回来” | MLP 之后收集专家输出的跨节点通信 |
| 专家并行（Expert Parallelism，EP） | “专家跨 GPU 分布” | 把 MoE 专家分片到不同 rank，使不同 GPU 持有不同专家 |
| 流水线并行（Pipeline Parallelism，PP） | “层跨 GPU 分布” | 把模型层分片到不同 rank；DualPipe 所调度的维度 |
| 气泡占比（Bubble fraction） | “浪费的 GPU 时间” | （气泡时间 / 总时间）；DualPipe 致力于把它压到零 |

## 扩展阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) —— DualPipe 的主要参考文献
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) —— 开源参考实现，包含 DualPipeV（Cut-in-half）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) —— Zero Bubble 的前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) —— 影响 DeepSeek EP-off 模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) —— DualPipe 对标的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) —— 原始流水线并行论文与气泡问题
