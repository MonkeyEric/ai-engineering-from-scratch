# 混合专家模型（Mixture of Experts, MoE）

> 一个 70B 的稠密 Transformer 会为每个词元（token）激活全部参数。一个 671B 的 MoE 每个词元只激活 37B 参数，却在所有基准测试上都击败了前者。稀疏性（sparsity）是这十年最重要的扩展思想。

**类型：** 实战构建  
**语言：** Python  
**前置知识：** Phase 7 · 05（完整 Transformer）、Phase 7 · 07（GPT）  
**时长：** 约 45 分钟

## 问题背景

稠密 Transformer 在推理时的 FLOPs 等于其参数量（前向传播再乘以 2）。扩大稠密模型规模时，每个词元都要付出全部计算代价。到 2024 年，前沿模型撞上了算力墙：要想显著提升智能水平，每个词元所需的 FLOPs 呈指数级增长。

混合专家模型（Mixture of Experts, MoE）打破了这种绑定。把每个前馈网络（FFN）替换成 `E` 个独立专家（expert）加上一个为每个词元挑选 `k` 个专家的路由器（router）。总参数量 = `E × FFN_size`。每个词元激活的参数量 = `k × FFN_size`。2026 年的典型配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年的前沿模型几乎全是 MoE：DeepSeek-V3（总参数量 671B / 激活 37B）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前十名的开源模型都是 MoE。

## 核心概念

![MoE 层：路由器为每个词元从 E 个专家中选出 k 个](../assets/moe.svg)

### 替换 FFN

稠密 Transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 为每个词元从 E 个中选 k 个
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是独立的 FFN（通常是 SwiGLU）。路由器是一个单独的线性层。每个词元自行选择自己的 `k` 个专家，并以门控加权的方式混合它们的输出。

### 负载均衡问题

如果路由器把 90% 的词元都送进专家 3，其他专家就会闲置。人们尝试过三种解决方案：

1. **辅助负载均衡损失（auxiliary load-balancing loss）**（Switch Transformer、Mixtral）。添加一个与专家使用量方差成正比的惩罚项。有效，但引入了一个超参数和第二条梯度信号。
2. **专家容量 + 词元丢弃（expert capacity + token dropping）**（早期 Switch）。每个专家最多处理 `C × N/E` 个词元；溢出的词元跳过该层。会损害质量。
3. **无辅助损失均衡（auxiliary-loss-free balancing）**（DeepSeek-V3）。为每个专家添加一个可学习的偏置（bias），用于调整路由器的 top-k 选择。偏置在训练损失之外更新，主目标函数不受惩罚。这是 2024 年的重大突破。

DeepSeek-V3 的做法是：每步训练后，检查每个专家的使用量高于还是低于目标值。将偏置向 `±γ` 微调。选择时使用 `scores + bias`。而门控所用的专家概率保持原始 `scores` 不变。这样就把路由（routing）与表达（expression）解耦。

### 共享专家

DeepSeek-V2/V3 还将专家拆分为*共享（shared）*和*路由（routed）*两类。每个词元都会经过所有共享专家；路由专家则通过 top-k 选择。共享专家捕获通用知识，路由专家负责专门化。V3 使用 1 个共享专家，加上从 256 个路由专家中选出 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家和一个完整 FFN 一样宽。`E` 较小（8–64），`k` 较小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（为 FFN 尺寸的 1/8）。`E` 很大（256+），`k` 更大（8+）。总参数量相同，但组合数呈指数级增长。每个词元可能激活的“专家”组合有 `C(256, 8) = 400 万亿` 种。质量提升，延迟保持不变。

### 成本画像

每个词元、每一层：

| 配置 | 每词元激活参数量 | 总参数量 |
|------|------------------|----------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3 在几乎所有基准测试上都击败了 Llama 3 70B（稠密模型），同时每个词元的**激活 FLOPs 更少**。更多参数 = 更多知识。更多激活 FLOPs = 每个词元更多计算。MoE 将二者解耦。

### 代价：显存

无论哪些专家被激活，所有专家都要驻留在 GPU 上。一个 671B 模型以 fp16 权重存放需要约 1.3 TB 显存（VRAM）。前沿 MoE 部署需要专家并行（expert parallelism）——将专家分片到不同 GPU，通过网络路由词元。延迟主要由 all-to-all 通信决定，而非矩阵乘法。

## 动手实现

参见 `code/main.py`。一个用纯标准库实现的精简 MoE 层，包含：

- `n_experts=8` 个类 SwiGLU 专家（每个专家仅一个线性层，便于演示）
- top-k=2 路由
- 经 softmax 归一化的门控权重
- 通过每个专家的偏置实现无辅助损失均衡

### 步骤 1：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 对选中专家的原始分数做 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置在不干扰模型预测的前提下纠正负载不均。

### 步骤 2：让 100 个词元通过路由器

追踪每个专家的触发频率。没有偏置时，使用量会倾斜。加入偏置更新循环后（过度使用的专家 `-γ`，使用不足的专家 `+γ`），经过若干次迭代，使用量会收敛到均匀分布。

### 步骤 3：参数量对比

打印某个 MoE 配置的“等效稠密模型”参数量。以 DeepSeek-V3 为例：256 个路由专家 + 1 个共享专家，8 个激活，d_model=7168。总参数量令人瞠目，而激活参数量只有稠密 Llama 3 70B 的七分之一。

## 实际使用

使用 HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年的生产级推理：vLLM 原生支持 MoE 路由，SGLang 拥有最快的专家并行路径。二者都会自动处理 top-k 选择与专家并行。

**何时选择 MoE：**

- 你希望以更低的单词元推理成本获得前沿质量。
- 你拥有充足的显存（VRAM）/ 专家并行基础设施。
- 你的 workload 以词元密集型为主（聊天、代码），而非上下文密集型（长文档）。

**何时不选择 MoE：**

- 边缘部署——任何激活 FLOP 都要付出完整存储代价。
- 对延迟敏感的单用户服务——专家路由会带来额外开销。
- 小模型（<7B）——MoE 的质量优势只有在跨越算力阈值（约 6B 激活参数）后才会显现。

## 交付应用

参见 `outputs/skill-moe-configurator.md`。该技能会根据参数预算、训练词元数和部署目标，为新 MoE 选择 E、k 以及共享专家布局。

## 练习题

1. **简单。** 运行 `code/main.py`。观察无辅助损失均衡的偏置更新如何在 50 次迭代内让专家使用量趋于均匀。
2. **中等。** 将可学习路由器替换为基于哈希的路由器（确定性、无需学习）。比较质量与均衡性。为什么可学习路由器更优？
3. **困难。** 实现 GRPO 风格的“与推理一致的路由（rollout-matched routing）”（DeepSeek-V3.2 技巧）：记录推理时哪些专家被激活，并在梯度计算时强制使用相同路由。在一个小型策略梯度（policy-gradient）实验上测量其影响。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 专家（expert） | “众多 FFN 中的一个” | 一个独立的前馈网络；其参数专门负责 FFN 计算中某个稀疏切片。 |
| 路由器（router） | “门控” | 一个极小的线性层，为每个词元对每个专家打分；用于 top-k 选择。 |
| Top-k 路由（top-k routing） | “每个词元激活 k 个专家” | 每个词元的 FFN 计算恰好经过 k 个专家，按门控权重加权。 |
| 辅助损失（auxiliary loss） | “负载均衡惩罚” | 惩罚专家使用倾斜的额外损失项。 |
| 无辅助损失（auxiliary-loss-free） | “DeepSeek-V3 的技巧” | 仅通过对路由器选择加入每个专家的偏置来实现均衡；没有额外梯度。 |
| 共享专家（shared expert） | “始终在线” | 每个词元都会经过的额外专家；用于捕获通用知识。 |
| 专家并行（expert parallelism） | “按专家分片” | 将不同专家分布到不同 GPU；通过网络路由词元。 |
| 稀疏性（sparsity） | “激活参数 < 总参数” | 比例 `k × expert_size / (E × expert_size)`；DeepSeek-V3 为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer 等（2017）。Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) —— 原始思想。
- [Fedus、Zoph、Shazeer（2022）。Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) —— Switch，经典 MoE。
- [Jiang 等（2024）。Mixtral of Experts](https://arxiv.org/abs/2401.04088) —— Mixtral 8×7B。
- [DeepSeek-AI（2024）。DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) —— MLA + 无辅助损失 MoE + MTP。
- [Wang 等（2024）。Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) —— 基于偏置的均衡论文。
- [Dai 等（2024）。DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) —— 本课路由器所用的细粒度 + 共享专家拆分方法。
- [Kim 等（2022）。DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) —— 最早的共享专家论文。
