# Jamba —— 混合 SSM-Transformer 架构

> 状态空间模型（State Space Model，SSM）与 Transformer 各有所长。Transformer 通过注意力（attention）换取质量，代价是二次方复杂度；SSM 通过递推（recurrence）实现线性时间推理与恒定显存，但质量稍逊。AI21 的 Jamba（2024 年 3 月）与 Jamba 1.5（2024 年 8 月）将二者置于同一模型：每 7 个 Mamba 层配 1 个 Transformer 层，每隔一个块使用混合专家（MoE），并在单张 80GB GPU 上支持 256k 上下文窗口。Mamba-3（ICLR 2026）则用复数值状态空间与多输入多输出（MIMO）投影进一步强化了 SSM 一侧。本课将完整解读这两种架构，并解释为何在长上下文领域，纯 SSM 与纯 Transformer 的尝试都未能走远，而混合方案却历经三年扩展考验依然成立。

**Type:** 学习
**Languages:** Python（标准库，层配比计算器）
**Prerequisites:** Phase 10 · 14（开放模型架构），Phase 10 · 17（原生稀疏注意力）
**Time:** 约 60 分钟

## 学习目标

- 说明 Jamba 块中的三种原语 —— Transformer 层、Mamba 层、MoE —— 以及 1:7:偶数 的交错配方。
- 概括 SSM 递推的基本形式，并解释它为何能实现恒定显存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV 缓存占用，并与纯 Transformer 模型的需求进行对比。
- 列举 Mamba-3 的三项创新（指数-梯形离散化、复数值状态更新、MIMO），以及每一项针对的问题。

## 问题背景

注意力在序列长度上是二次方复杂度。状态空间模型是线性复杂度。这一差异会被放大：在 256k token 时，Transformer 的注意力图每个头就有 650 亿个条目；而 SSM 的递推状态大小固定，与序列长度无关。

纯 SSM 模型（Mamba、Mamba-2）在小规模下能与 Transformer 的困惑度（perplexity）持平，但在状态跟踪任务上落后，并在某些上下文内检索（in-context retrieval）场景失败。直观理解：SSM 将历史压缩到固定状态中，当历史很长时，信息会泄漏。注意力精确记住一切，但付出二次方成本。

显而易见的解决方案是：两者都用。在需要精确回忆的地方放 Transformer 层，其余地方用 SSM 层，再调整比例。Jamba 是首个大规模应用这一混合配方并产品化的模型（总计 52B 参数、激活 12B、256k 上下文、单张 80GB GPU）。Jamba 1.5 将系列扩展到 398B 总计 / 94B 激活。Mamba-3（ICLR 2026）则是当前最优秀的纯 SSM 基线，下一代混合模型可围绕它重建。

本课通读这三篇论文，并建立“选择合适比例”的心智模型。

## 核心概念

### 一页纸讲清 SSM

状态空间模型通过一个固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步状态都通过线性动力学 `A` 演化，接收输入 `B x_t`，并输出 `C h_t`。`A`、`B`、`C` 都可以学习。注意关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要更早的 `x`。显存是恒定的。推理每个 token 的复杂度是 O(1)。

建模质量的关键在于 `A` 的结构。S4（Gu, 2021）使用高度结构化的矩阵，可在训练时作为长卷积高效计算。Mamba（Gu、Dao, 2023）将固定的 `A、B、C` 替换为数据依赖的参数（即“选择性”所在）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）则在特定位置重新引入复杂性。

关键性质：对于解码器 LLM，SSM 层可以作为注意力层的直接替代，用每层固定大小的状态取代不断增长的 KV 缓存。

### Jamba 块

Jamba 块按照两个数字交错层：

- `l`：注意力到 Mamba 的比例。Jamba 使用 `l = 8`，即每 7 个 Mamba 层配 1 个 Transformer 层（7 Mamba + 1 Attention = 每组 8 层）。
- `e`：MoE 频率。Jamba 使用 `e = 2`，即每隔一层应用 MoE。

块内的层序列如下：

```
M  M  M  M  M  M  M  A    （7 Mamba + 1 Attention）
|  M  |  M  |  M  |  M    （| 表示应用 MoE 的位置）
```

每个 Jamba 块共 8 层。深度为 4 个块（32 层总计）时，你会得到 28 个 Mamba 层和 4 个 Attention 层，其中 16 层使用 MoE。

### 为何是 1:7 的比例

AI21 进行了消融实验：怎样的注意力与 Mamba 比例能在困惑度和长上下文评估的上下文内召回上同时表现最佳？

- 注意力太多（1:1）：质量上升，但显存与速度下降。
- 注意力太少（1:15）：显存表现很好，但上下文内检索失败。
- 甜点区：1:7 或 1:8。

直观理解：Transformer 层负责精确回忆与状态跟踪；Mamba 层负责低成本地处理大量计算。

### 位置编码

Mamba 层本身通过递推具备位置感知能力。原始基于 Mamba 的混合模型中的注意力层不使用 RoPE —— SSM 层提供了位置信息。Jamba 1.5 在注意力层中加入 RoPE，以提升长上下文泛化能力，这是基于经验性长上下文评估的事后优化。

### 显存预算

以 Jamba-1 的规格为例（32 层：28 Mamba + 4 Attention，隐藏维度 4096，32 个注意力头）：

- KV 缓存（仅注意力层）：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`（BF16，256k 上下文）。只有 4 个注意力层贡献。
- SSM 状态：每个 token 前缀需要 `28 * hidden * state_size`，但它是每层固定大小，不随序列长度增长。典型 Mamba 状态为每特征 16，隐藏维度 4096：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

对比同规格纯 Transformer（32 层、相同隐藏维度、完整 MHA、32 头）：`2 * 32 * 32 * 128 * 256k * 2 = 128 GB`（BF16，256k 上下文）。KV 缓存减少 8 倍。即使与大多数 2024 模型使用的 GQA(8) 基线相比（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合方案也仅需约 16 GB，仍然小 2 倍。

这就是 AI21 所说的“单张 80GB GPU 跑 256k 上下文”。完整 MHA 纯 Transformer 的 KV 缓存根本放不下；即使是 GQA 基线，也没有空间留给权重和激活；而 Jamba 可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 一侧引入了三项创新：

1. **指数-梯形离散化（Exponential-trapezoidal discretization）。** 用更具表达力的递推取代 Mamba-2 中的欧拉法离散化。在核心递推内部对状态-输入执行类卷积操作，而非像过去那样在外部对 `x_t` 做卷积。

2. **复数值状态更新。** 之前的 Mamba 将状态矩阵从复数（S4）简化为实对角阵（Mamba），再简化为缩放单位阵（Mamba-2）。Mamba-3 重新引入复数值 —— 等价于在状态上应用数据依赖的旋转位置编码。这恢复了先前实数简化所牺牲的状态跟踪能力。

3. **多输入多输出（MIMO）投影。** 不再使用逐特征的标量投影，而是使用矩阵值投影。在不增加解码延迟的前提下，提升建模能力与推理时的硬件利用率。

在 1.5B 参数规模下，Mamba-3 的平均下游准确率比 Gated DeltaNet 高 0.6 分；MIMO 变体再提升 1.2 分，总计 1.8 分。在相同状态大小下，Mamba-3 仅用一半状态即可匹敌 Mamba-2。

Mamba-3 尚未大规模部署在产品级混合模型中 —— 但它是下一代 Jamba 级模型在 SSM 一侧的 obvious 候选。

### 何时选择混合架构

混合架构在以下场景占优：

- 上下文足够长，使得纯 Transformer 的 KV 缓存变得痛苦（64k 以上）。
- 任务同时包含短程结构（适合 SSM）与长程回忆（需要 Transformer）。
- 你希望部署在单 GPU 显存预算下，而 Transformer KV 缓存本身都放不下的场景。

混合架构在以下场景不占优：

- 上下文较短（16k 以下）。SSM 开销是浪费；纯 Transformer 足够。
- 任务需要处处到处的注意力（深度推理、多文档交叉引用）。混合架构中注意力层的稀疏性会受损。
- 你要扩展到万亿参数级别的前沿模型。目前纯 Transformer + MLA + MoE（DeepSeek-V3 风格）在能力竞赛中领先。

### 竞争格局

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | 线性时间、恒定显存 |
| Jamba | hybrid | 52B/12B | 单张 80GB 跑 256k |
| Jamba 1.5 Large | hybrid | 398B/94B | 企业级长上下文 |
| Mamba-3 | pure SSM | 1.5B（论文） | 恢复状态跟踪能力 |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | 前沿能力 |

2026 年的格局：纯 Transformer MoE 主导前沿，但混合架构占据 256k+ 长上下文细分赛道。Mamba-3 在状态跟踪上的优势，可能推动下一代混合模型的比例进一步降低（更多 SSM、更少注意力）。

## 动手实践

`code/main.py` 是一个混合架构显存计算器。给定 SSM-Transformer 比例与隐藏维度/层数配置，它可以计算：

- 目标上下文下的 KV 缓存。
- SSM 状态显存。
- 一系列模型规格在上下文 N 下的总显存。

计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格 1:7 混合架构。
- 纯 SSM（完全没有 KV 缓存）。

这些数字直接来自 Jamba-1 与 Jamba-1.5 论文的公开规格，并对假设变体进行了外推。

真实部署时的集成考量：

- 多数生产级推理服务（vLLM、SGLang）已支持 Jamba 与 Mamba，请确认具体版本。
- 在 256k 上下文下，Jamba 的显存优势体现在并发请求吞吐上。同样 VRAM 下，你能容纳的 Jamba 序列数多于 Transformer 序列数。
- Mamba-3 作为独立模型目前尚未产品化部署 —— 仅 1.5B 的研究预览版。

## 交付成果

本课将产出 `outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度分布、任务组合、显存预算），它将推荐选择纯 Transformer、Jamba 风格混合架构还是纯 SSM，并显式说明显存与质量之间的权衡。

## 练习题

1. 运行 `code/main.py`，计算 32 层纯 Transformer（隐藏维度 4096、32 头）与相同规格的 Jamba-1 混合架构在 256k 上下文下的 KV 缓存，验证 AI21 论文宣称的约 8 倍显存降低。

2. 修改计算器，建模 1:3 混合（4 Mamba : 1 Attention）与 1:15 混合（14 Mamba : 1 Attention）。绘制 KV 缓存随比例变化的曲线。在什么比例下 KV 缓存等于 SSM 状态显存？

3. 阅读 Jamba 论文第 3 节（arXiv:2403.19887）。解释为何 AI21 使用 Mamba-1 而非更快的 Mamba-2。提示：混合消融实验部分记录了原因。

4. 计算 Jamba 1.5 Large（398B 总计 / 94B 激活）中每隔一层使用 MoE 的参数开销。将其激活比例与 DeepSeek-V3（37B/671B）对比，并解释为何 Jamba 的架构推动了更高的激活比例。

5. 阅读 Mamba-3 论文第 3 节（arXiv:2603.15569）。用三句话解释为何复数值状态更新等价于数据依赖的旋转位置编码。结合 Phase 7 · Lesson 04 的 RoPE 推导。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| 状态空间模型（SSM） | “带固定状态的递推” | 具有学习递推 `h_t = A h_{t-1} + B x_t` 的层；每个 token 的显存恒定 |
| 选择性 SSM | “Mamba 的 trick” | 数据依赖的 A、B、C 参数，使模型在线性时间内获得类似门控的选择性 |
| Attention-to-Mamba 比例 | “用了多少注意力层” | 在 Jamba 中，`l = 8` 表示每 7 个 Mamba 层配 1 个注意力层 |
| Jamba 块 | “8 层一组” | 1 个注意力层 + 7 个 Mamba 层，MoE 按交替位置应用 |
| SSM 状态 | “隐藏缓冲区” | 每层固定大小的状态，替代 Mamba 层的 KV 缓存 |
| 256k 上下文 | “Jamba 的旗舰数字” | Jamba-1 在单张 80GB GPU 上支持的序列长度；同规格纯 Transformer 无法做到 |
| Mamba-3 | “2026 年的纯 SSM” | 当前最佳纯 SSM 架构，具有复数状态 + MIMO；混合模型重建的基线 |
| MIMO | “多输入多输出” | Mamba-3 的创新，使用矩阵值投影替代逐特征标量投影 |
| 指数-梯形离散化 | “Mamba-3 的递推” | 更具表达力的递推，涵盖 Mamba-2 的欧拉法离散化 |
| 混合架构 | “注意力与 SSM 混合” | 任何交错 Transformer 与 SSM 层的模型；Jamba 是产品化的典型代表 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) —— 原始 Jamba 论文，比例消融，256k 上下文声明
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) —— 扩展系列，398B/94B 与 12B/52B 公开版本
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) —— Jamba 所基于的选择性 SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) —— 简化的结构化状态空间继任者
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) —— 复数值状态、MIMO、2026 年纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) —— S4 论文，LLM 中 SSM 谱系的起点
