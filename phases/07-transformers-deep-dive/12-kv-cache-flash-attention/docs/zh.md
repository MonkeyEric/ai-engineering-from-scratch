# KV 缓存、Flash Attention 与推理优化

> 训练是并行的，受限于 FLOP。推理是串行的，受限于内存。瓶颈不同，技巧也不同。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 · 02（自注意力机制），Phase 7 · 05（完整 Transformer），Phase 7 · 07（GPT）
**时间：** ~75 分钟

## 问题背景

一个朴素的自回归解码器生成 `N` 个 token 需要 `O(N²)` 的计算量：每一步都要对完整前缀重新计算注意力。对于 4K token 的回复，那就是 1600 万次注意力操作，其中大部分是冗余的。每个前缀 token 的隐藏状态一旦计算出来就是确定的——你只需要用新 token 的查询（query）去与之前所有 token 缓存下来的键（key）和值（value）做注意力即可。

此外，注意力本身也会产生大量的数据搬运。标准注意力会物化一个 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM 的读写太多了。当 N≥2K 时，注意力会先成为内存瓶颈，再成为计算瓶颈。经典的注意力核函数在现代 GPU 上只能发挥 1/4 到 1/10 的性能。

两项都来自 Dao 等人的优化，将前沿推理从“慢”推向了“快”：

1. **KV 缓存（KV cache）。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力就是一次查询与缓存键的运算。推理的复杂度从 `O(N²)` 降低到每个生成步骤的 `O(N)`。
2. **Flash Attention。** 将注意力计算分块（tile），使完整的 N×N 矩阵永远不会写入 HBM。softmax 和矩阵乘法全部在 SRAM 中完成。在 A100 上可获得 2–4 倍的 wall-clock 加速；在 H100 上使用 FP8 可达 5–10 倍。

到 2026 年，这两者已成为标配。所有生产级推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认依赖它们。所有前沿模型都默认启用 Flash Attention。

## 核心概念

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV 缓存的数学

每个解码层、每个 token、每个头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于一个 7B 模型，32 层、32 个头、d_head=128、fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA 使用 8 个 KV 头）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这就是为什么 Llama 3 70B 在 128K 上下文、batch size 为 1 时，仅 KV 缓存就需要占用 40 GB A100 的大部分显存。

**GQA 是 KV 缓存的胜利。** 如果用 64 个头的 MHA，将会是 32 GB。MLA 还能进一步压缩。

### Flash Attention —— 分块技巧

标准注意力：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次 HBM 往返。在 H100 上，HBM 带宽为 3 TB/s，而 SRAM 为 30 TB/s。每次访问 HBM 相比全程在片内计算都会慢一个数量级。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个分块只访问一次 HBM。总内存占用从 `O(N²)` 降至 `O(N)`。反向传播会从正向传播中重新计算一些值，而不是存储它们——又省了一笔内存。

**数值技巧。** 运行中的 softmax 会跨分块维护 `(max, sum)`，因此最终归一化是精确的。这不是近似——Flash Attention 的输出与标准注意力在数学上是位等价的（忽略 fp16 非结合性的影响）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件上的加速 |
|------|------|----------|-----------------|
| Flash 1 | 2022 | 分片 SRAM 核函数 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性、因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步、FP8 | H100 上 1.5–2×（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 级流水线、软件 exp2 | 推理优先（初期仅支持正向） |

Flash 4 在发布时仅支持正向传播。训练仍使用 Flash 3。Flash 4 对 GQA 和变长序列的支持仍在推进中（预计 2026 年中）。

### 投机解码（Speculative Decoding）—— 另一种降低延迟的利器

小模型廉价地生成 N 个 token，大模型并行验证这 N 个 token。如果验证接受了 k 个 token，那么你只用一次大模型前向传播就换来了 k 次生成。在代码和散文上，典型的 k 值为 3–5。

2026 年的默认方案：
- **EAGLE 2 / Medusa。** 集成的草稿头，共享验证器的隐藏状态。无损质量下加速 2–3 倍。
- **基于草稿模型的投机解码。** 在消费级硬件上加速 2–4 倍。
- **Lookahead decoding。** 雅可比迭代；不需要草稿模型。比较小众但几乎是免费的。

### 连续批处理（Continuous Batching）

经典批处理推理：等待最慢的序列完成，再开始新的一批。短回复提前完成时会造成 GPU 空转。

连续批处理（最早由 Orca 提出，现已集成在 vLLM、TensorRT-LLM、SGLang 中）：一旦有旧请求完成，就立即把新请求换入批次。对于典型聊天工作负载，吞吐量可提升 5–10 倍。

### PagedAttention —— 把 KV 缓存当作虚拟内存

vLLM 的招牌功能。KV 缓存以 16 个 token 为一块进行分配；页表将逻辑位置映射到物理块。这允许在并行采样（beam search、parallel sampling）中共享 KV，支持提示缓存的前缀热替换，并能整理内存碎片。相比朴素的连续分配，吞吐量提升 4 倍。

## 动手实现

参见 `code/main.py`。我们实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个 `O(N)` 的 KV 缓存解码器。
3. 一个分块 softmax，模拟 Flash Attention 的运行最大值算法。

### 第一步：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

很简单：按层、按头保存每个 token 的 K、V 向量，并不断追加。

### 第二步：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

输出与一次性计算 `softmax(qK) V` 位等价，但任意时刻的工作集只是一个 `tile × d_head` 的分块，而不是完整的 `N × d_head`。

### 第三步：在 100 个 token 的生成上对比朴素与缓存解码

统计注意力操作次数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码会打印两者。

## 实际使用

```python
# HuggingFace transformers 在 decoder-only 的 generate() 中自动启用 KV 缓存。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 上使用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
```

vLLM 生产部署：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的一个大杀器——相同的系统提示、 few-shot 示例或长上下文文档可以在多次调用间复用 KV。对于带有重复工具提示的 Agent 工作负载，前缀缓存通常能带来 5 倍的吞吐量提升。

## 交付成果

参见 `outputs/skill-inference-optimizer.md`。该技能会为新的推理部署选择注意力实现、KV 缓存策略、量化方式和投机解码方案。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素解码器和缓存解码器输出相同；注意操作次数的差异。
2. **中等。** 实现前缀缓存：给定提示 P 和若干续写，先对 P 做一次前向传播填满 KV 缓存，然后为每个续写分支复用。对比每次都重新编码 P 的加速效果。
3. **困难。** 实现一个玩具版 PagedAttention：KV 缓存以固定的 16-token 块分配，并维护空闲列表。序列完成时，将其块归还池中。模拟 1000 条长度各异的聊天完成。对比连续分配的内存碎片情况。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| KV 缓存（KV cache） | “让解码变快的技巧” | 存储每个前缀 token 的 K 和 V；新查询直接与它们做注意力，无需重新计算。 |
| HBM | “GPU 主存” | 高带宽显存（High Bandwidth Memory）；H100 为 80 GB，B200 为 192 GB。带宽约 3 TB/s。 |
| SRAM | “片上内存” | 每个 SM 上的高速内存，H100 上每个 SM 约 256 KB。带宽约 30 TB/s。 |
| Flash Attention | “分块注意力核函数” | 不在 HBM 中物化 N×N 矩阵即可完成注意力计算。 |
| 连续批处理（Continuous batching） | “无等待批处理” | 完成的序列立即换出，新序列换入，无需清空整个批次。 |
| PagedAttention | “vLLM 的招牌” | KV 缓存以固定块分配并通过页表管理；消除碎片。 |
| 前缀缓存（Prefix caching） | “复用长提示” | 在多个请求间缓存共享前缀的 KV；对 Agent 场景大幅降本。 |
| 投机解码（Speculative decoding） | “草稿 + 验证” | 廉价草稿模型生成 token；大模型一次验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) —— Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) —— Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) —— Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) —— Blackwell 5 级流水线和软件 exp2 技巧；阅读仓库 README 了解本课提到的仅正向发布限制。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) —— vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 投机解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— EAGLE-1/2 论文，本课引用的集成草稿方法。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— 与 EAGLE 并列提及的 Medusa 方法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) —— 关于 16-token 块和页表设计的权威深度解读。
