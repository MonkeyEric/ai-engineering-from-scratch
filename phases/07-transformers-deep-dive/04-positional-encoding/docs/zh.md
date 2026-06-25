# 位置编码（Positional Encoding）—— 正弦、RoPE、ALiBi

> 注意力（attention）具有置换不变性。如果不加入位置信号，"The cat sat on the mat" 和 "mat the on sat cat the" 会产生相同的输出。三种算法解决了这个问题 —— 它们对“位置”含义的理解各不相同。

**类型：** 构建  
**语言：** Python  
**前置知识：** Phase 7 · 02（自注意力机制），Phase 7 · 03（多头注意力机制）  
**时间：** ~45 分钟

## 问题所在

缩放点积注意力对顺序不敏感。注意力矩阵 `softmax(Q K^T / √d) V` 由成对相似度计算得出。如果你打乱 `X` 的行，输出也会以同样方式被打乱。注意力内部没有任何机制关心位置。

在词袋（bag-of-words）模型中，这不算缺陷。但对于语言、代码、音频、视频 —— 任何顺序都承载意义的场景 —— 这是致命的。

解决思路是以某种方式将位置信息注入嵌入（embedding）。三个时代给出了三种答案：

1. **绝对正弦编码**（Vaswani 2017）。将位置的 `sin/cos` 函数加到嵌入上。简单、无需学习、但训练长度之外外推（extrapolation）效果差。
2. **RoPE — 旋转位置编码（Rotary Position Embeddings）**（Su 2021）。按与位置成正比的角度旋转 Q 和 K 向量。直接在点积（dot product）中编码*相对*位置。截至 2026 年占据主导地位。
3. **ALiBi — 线性偏置注意力（Attention with Linear Biases）**（Press 2022）。完全跳过嵌入技巧；根据距离对每个头（head）的注意力分数施加线性惩罚。长度外推能力出色。

截至 2026 年，几乎所有前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦编码已成为历史。

## 核心概念

![正弦绝对编码 vs RoPE 旋转 vs ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对正弦编码

预先计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前执行 `X' = X + PE[:N]`。每个维度都是不同频率的正弦波。模型学会从相位模式中读取位置。但在超过 `max_len` 后失效：如果模型只见过位置 0–2047，它就无法理解位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（而不是嵌入）。对于维度对 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base 默认为 10000
```

对位置为 `pos_k` 的键（key）应用同样的旋转。点积 `q'_m · k'_n` 会变成仅关于 `(m - n)` 的函数。也就是说：**注意力分数只取决于相对距离**，即便旋转是基于绝对位置进行的。这是一个巧妙的技巧。

RoPE 的扩展：`base` 可以进行缩放（NTK-aware、YaRN、LongRoPE），从而在无需重新训练的情况下外推到更长上下文。Llama 3 就这样把上下文从 8K 扩展到了 128K。

### ALiBi

跳过嵌入层面的技巧，直接偏置注意力分数：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是每个头特有的斜率（slope），例如 `1 / 2^(8·h/H)`。相近的 token 获得加分，遥远的 token 受到惩罚。训练时无额外开销。论文表明，其长度外推能力优于正弦编码，并在原始训练长度上媲美 RoPE。

### 2026 年如何选择

| 变体 | 外推能力 | 训练成本 | 代表模型 |
|---------|---------------|---------------|---------|
| 绝对正弦编码 | 差 | 无 | 原始 Transformer、早期 BERT |
| 可学习绝对编码 | 无 | 极小 | GPT-2、GPT-3 |
| RoPE | 经缩放后良好 | 无 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 极佳 | 需微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 极佳 | 无 | BLOOM、MPT、Baichuan |

RoPE 之所以胜出，是因为它无需改变注意力架构即可嵌入，编码相对位置，并且其 `base` 超参数为长上下文微调提供了一个清晰的旋钮。

## 动手实现

### 步骤 1：正弦编码

参见 `code/main.py`。核心只有 4 行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前，将结果加到嵌入矩阵上。

### 步骤 2：将 RoPE 应用于 Q、K

RoPE 直接作用于 Q 和 K。对每个维度对：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置为 `m` 的 Q 和位置为 `n` 的 K 应用同一函数。它们的点积会在每一对坐标上多出一个 `cos((m-n)·θ_i)` 因子。注意力无需额外代价即可学习相对位置。

### 步骤 3：ALiBi 斜率与偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads)，其中 h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 在 softmax 之前加到注意力分数上
```

将 `bias[h]` 加到第 `h` 个头的 `(seq_len, seq_len)` 注意力分数矩阵上，然后做 softmax。

### 步骤 4：验证 RoPE 的相对距离性质

取两个随机向量 `a, b`。分别按 `(pos_a, pos_b)` 和 `(pos_a + k, pos_b + k)` 旋转。两次点积在浮点误差范围内应相等。这就是 RoPE 的核心 —— 它对绝对偏移不变，只关心相对间隔。

## 实际使用

PyTorch 2.5+ 在 `torch.nn.functional` 中内置了 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力核函数内部完成。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 从 4K 扩展到 16K+ 时，将 `base` 缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值方法，能在长上下文上保持注意力熵。Llama 3.1 128K 使用了它。
- **LongRoPE。** 微软 2024 年的方法，使用进化搜索为每个维度挑选缩放因子。Phi-3-Long 使用了它。
- **位置插值 + 微调。** 仅将位置按扩展因子压缩，再用 1–5B token 微调。效果出奇地好。

## 交付物

参见 `outputs/skill-positional-encoding-picker.md`。该技能会根据目标上下文长度、外推需求和训练预算，为新模型选择一种编码策略。

## 练习

1. **简单。** 将正弦 `PE` 矩阵以热力图（heatmap）形式绘制出来，参数为 `max_len=512, d=128`。确认“随着维度索引增大，条纹逐渐变宽”的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。在长度为 256 的序列上训练一个小型语言模型，然后在长度 1024 上分别测试有缩放和无缩放的效果。测量困惑度（perplexity）。
3. **困难。** 在同一个注意力模块中实现 ALiBi 和 RoPE。在长度为 512 的复制任务上训练一个 4 层 Transformer。测试时外推到 2048。比较性能衰减。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 位置编码（positional encoding） | “让注意力知道顺序” | 任何添加到嵌入或注意力中以编码位置信息的信号。 |
| 正弦编码（sinusoidal） | “最原始的那种” | 将几何频率的 `sin/cos` 加到嵌入上；不能外推。 |
| RoPE | “旋转嵌入” | 按位置相关角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | “线性偏置技巧” | 在注意力分数上加 `-m·|i-j|`；无需嵌入，外推能力强。 |
| base | “RoPE 的旋钮” | RoPE 中的频率缩放因子；增大它可在推理时扩展上下文。 |
| NTK-aware | “一种 RoPE 缩放技巧” | 扩展上下文时重新缩放 `base`，避免高频维度被过度压缩。 |
| YaRN | “高级的那个” | 逐维度插值与外推，保持注意力熵。 |
| 外推（extrapolation） | “在训练长度之外也能工作” | 位置方案能否在超过训练时见过的 `max_len` 后仍给出正确输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦编码。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放方法。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 微软的方法，Phi-3-Long 使用，“使用它”一节亦有引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 生产级 RoPE 缩放方案实现（default、linear、dynamic、YaRN、LongRoPE、Llama-3）。
