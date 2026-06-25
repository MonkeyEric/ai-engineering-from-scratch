# 梯度检查点（Gradient Checkpointing）与激活值重计算（Activation Recomputation）

> 反向传播（backpropagation）会保留每一个中间激活值（activation）。在 700 亿参数、128K 上下文的情况下，每个 rank 的激活值高达 3 TB。检查点用计算换内存：不保存，而是重新计算。问题在于丢弃哪些片段，而答案不是“全部丢弃”。

**类型：** 构建  
**语言：** Python（使用 numpy，可选 torch）  
**前置知识：** Phase 10 Lesson 04（Pre-Training Mini-GPT）、Phase 10 Lesson 05（Scaling & Distributed）  
**时长：** 约 70 分钟

## 问题背景

训练 transformer 时，每一层都会存储反向传播中每个可微操作的输入：注意力（attention）的输入、Q/K/V 投影、softmax 输出、前馈网络（FFN）输入、归一化（norm）输出以及残差流（residual stream）。对于隐藏维度为 `d`、序列长度为 `L`、批次大小为 `B` 的层，每层大约需要 `12 * B * L * d` 个浮点数。

当 `d=8192, L=8192, B=1` 时，BF16 下每层约 800 MB。一个 64 层模型的激活值就是 51 GB——这还没乘以微批次（microbatch）大小，没加上注意力 softmax 中间值（每头 `L^2`），也没考虑张量并行（tensor-parallel）的部分副本。

双重账单：BF16 权重加上优化器（optimizer）状态或许能塞进 80 GB，但激活值会让你超标。梯度检查点（gradient checkpointing，又称激活值重计算 activation recomputation）是标准解决方案。丢弃大部分激活值；在反向传播时重新执行前向计算以恢复它们。代价：额外的 FLOPs。收益：内存按检查点片段数与总层数的比例下降。

若简单地使用检查点，每步大约增加 33% 的前向 FLOPs。若做得好——按照 Korthikanti 等人提出的“智能选择”进行选择性检查点（selective checkpointing）——可以在增加不到 5% FLOP 开销的情况下节省 5 倍内存。而随着 FP8 矩阵乘法、FSDP 卸载（FSDP offload）以及专家并行 MoE（expert-parallel MoE）的出现，这一点尤为重要：你既负担不起内存，也负担不起浪费的计算。

## 核心概念

### 反向传播真正需要什么

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`。为了计算它们，它需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 一些激活导数中间值（ReLU/GELU/softmax 的导数取决于激活值本身）

前向传播会自动在自动求导图（autograd graph）中保存这些信息。每个 `tensor.retain_grad()` 以及每个需要输入的操作都会保留引用。

### 朴素全量检查点

将网络分成 `N` 个片段。在前向过程中，只保存每个片段的输入。当反向传播需要中间值时，重新运行该片段的前向计算以实例化它们，然后进行求导。

示例：32 层 transformer 被分成 32 个片段，每层一个。

- 内存：32 个层输入（很小）对比 32 ×（每层的激活体积）（巨大）。
- 额外计算：每个片段多一次前向，即总前向 FLOPs 增加约 33%（因为反向是前向的 2 倍，完整步骤从 `1 + 2 = 3` 个单位变成 `1 + 1 + 2 = 4` 个单位）。

这就是 Chen 等人 2016 年的原始方案：每 `sqrt(L)` 层设置一个检查点，以平衡内存与计算。当 `L=64` 时，就是 8 个检查点。

### 选择性检查点（Korthikanti 2022）

不是所有激活值的存储成本都一样。注意力 softmax 输出是 `B*L*L*heads`，随序列长度平方增长。FFN 隐藏层激活是 `B*L*4d`，线性增长。对于长序列，softmax 占主导。

选择性检查点保留存储成本低的激活值（线性投影、残差），只重新计算昂贵的激活值（注意力）。你只需付出少量重计算 FLOPs，却能节省 `O(L^2)` 的内存。

Megatron-Core 将其实现为“选择性”激活重计算。用于大多数 2024 年及以后的 frontier 训练运行。

### 激活值卸载（Offload）

重计算的替代方案：在前向与反向之间将激活值转移到 CPU 内存。需要 PCIe 带宽；当空闲带宽超过物化成本时才有益。混合策略很常见：某些层检查点，其他层卸载。

FSDP2 将卸载作为一等公民选项提供。当 GPU 受内存瓶颈限制但 CPU-GPU 传输仍有余量时，卸载效果显著。

### 重计算成本模型

每步 FLOPs，朴素检查点每 `k` 层设置一次，共 `L` 层：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 每个片段多一次前向
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点时，你只重计算注意力核，而非整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层的激活体积：`A`。`L` 层总激活内存：`L * A`。

全量检查点（片段大小为 1）：只保存 `L * input_volume`（标准 transformer 中约为 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层设置检查点：保存 `L/k * A`，加上活动片段内 `k-1` 层的激活。

当 `k = sqrt(L)` 时，内存与重计算成本都按 `sqrt(L)` 缩放——这是均匀成本层的最优权衡。

### 何时不应使用检查点

- 流水线阶段（pipeline stage）最内层已经正在执行。它们反正要完成。
- 如果首尾层占阶段计算主导地位（在 transformer 中很少见）。
- 注意力核已经在使用 FlashAttention —— FlashAttention 本身就会快速重计算 softmax，因此再叠加层级别的检查点收益甚微。

### 实现模式

1. **函数包装器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装一个片段。PyTorch 只保存输入，在反向时重新计算其他一切。
2. **基于装饰器：** 将层标记为可检查点；训练器在配置时决定哪些片段被包装。
3. **手动显式重计算：** 自己编写反向传播，调用自定义的 `recompute_forward`，用保存的输入复制前向计算。

三者在功能上结果相同。包装器是标准写法。

### 与 TP / PP / FP8 的交互

- **张量并行（Tensor parallel）：** 重计算时需要收集或重新分散检查点输入；需考虑通信成本。
- **流水线并行（Pipeline parallel）：** 典型模式是为每个流水线阶段的前向做检查点，以便逆序微批次可以复用激活内存。
- **FP8 重计算：** 重计算过程中更新的 amax 历史必须与原始前向一致，否则 FP8 缩放会漂移。大多数框架会对缩放做快照。

## 动手实现

### 步骤 1：带片段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤 2：需要全部激活值的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤 3：每 k 层检查点的内存方案

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤 4：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤 5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤 6：最优片段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤 7：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用现有工具

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch 中的标准包装器。包装一个函数；只保存输入，在反向时重计算。
- **Megatron-Core activation recomputation**：支持 `selective`、`full` 和 `block` 模式。2024 年及以后 frontier 训练的标准工具。
- **FSDP2 offload**：通过 `module.to_empty(device="cpu")` 与 FSDP2 中的 `offload_policy` 将激活值分片卸载到 CPU，而非重计算。
- **DeepSpeed ZeRO-Offload**：针对优化器状态和激活值的 CPU 卸载，与检查点互补。

## 交付成果

本课生成 `outputs/prompt-activation-recompute-policy.md` —— 一个接收模型配置（层数、隐藏维度、序列长度、批次大小）与可用 GPU 内存，并输出每层重计算策略（none / selective / full / offload）的提示词。

## 练习题

1. 验证正确性。对比运行 `model_forward` + `model_backward`（保存全部激活值）与 `model_forward_checkpointed` + `model_backward_checkpointed`（按片段）。参数梯度必须在机器精度下完全相同。
2. 将片段大小 `k` 从 1 扫到 `L`。绘制 FLOP 开销与内存曲线。找到拐点。
3. 实现选择性检查点：保存注意力模块的输入，但不保存其中间值。在序列长度 8192 的 32 层模型上，测量其 FLOP 开销相对于全层检查点的比例。
4. 添加卸载。将片段输入保存到模拟的“CPU 缓冲区”（一个独立列表）。将“PCIe 带宽”测量为字节/时间，找到卸载与重计算之间的盈亏平衡点。
5. 用真实 PyTorch transformer 对比使用与不使用 `torch.utils.checkpoint` 的情况。测量内存（通过 `torch.cuda.max_memory_allocated`）与每步时间。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 梯度检查点（gradient checkpointing） | “通过重做前向节省内存” | 只保存片段输入；在反向传播时重计算中间值，以获取支撑梯度计算的张量 |
| 激活值重计算（activation recomputation） | “与检查点相同” | 同一技术的 HPC 风格名称 |
| 片段大小（segment size，k） | “每个检查点包含多少层” | 一起丢弃并物化中间值的层数 |
| 选择性检查点（selective checkpointing） | “Korthikanti 的技巧” | 只重计算存储成本高的激活值（注意力 softmax），保留成本低的 |
| 全量检查点（full checkpointing） | “朴素版本” | 在每个片段中重计算每一层的中间值 |
| 块检查点（block checkpointing） | “粗粒度” | 对整个 transformer 块做检查点；粒度最大 |
| FLOP 开销（FLOP overhead） | “计算税” | 每步额外 FLOPs =（重计算 FLOPs）/（前向 + 反向 FLOPs）；朴素情况 33%，选择性 5% |
| 激活值卸载（activation offload） | “送到 CPU” | 在前向→反向之间将激活值移到 CPU 内存；重计算的替代方案 |
| sqrt-L 规则（sqrt-L rule） | “经典最优解” | 对于均匀成本层，最优检查点间隔为 `sqrt(L)` 层 |
| 注意力 softmax 体积（attention-softmax volume） | “O(L^2) 问题” | `L^2 * heads * batch` 个浮点数；在长上下文中主导激活内存 |

## 延伸阅读

- [Chen 等人，2016——《Training Deep Nets with Sublinear Memory Cost》](https://arxiv.org/abs/1604.06174) —— 形式化梯度检查点的开创性论文
- [Korthikanti 等人，2022——《Reducing Activation Recomputation in Large Transformer Models》](https://arxiv.org/abs/2205.05198) —— 选择性激活值重计算与形式化成本分析
- [Pudipeddi 等人，2020——《Training Large Neural Networks with Constant Memory using a New Execution Algorithm》](https://arxiv.org/abs/2002.05645) —— 通过反向模式重物化实现的替代性恒定内存方法
- [Ren 等人，2021——《ZeRO-Offload: Democratizing Billion-Scale Model Training》](https://arxiv.org/abs/2101.06840) —— 大规模激活值卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) —— 标准 API
- [Megatron-Core 激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) —— selective、full 与 block 模式
