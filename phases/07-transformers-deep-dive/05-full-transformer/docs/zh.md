# 完整 Transformer —— 编码器 + 解码器

> 注意力（attention）是主角。其余部分——残差连接（residual connection）、归一化（normalization）、前馈网络（feed-forward）、交叉注意力（cross-attention）——都是让注意力能够深层的脚手架。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 02（自注意力），Phase 7 · 03（多头注意力），Phase 7 · 04（位置编码）
**时间：** 约 75 分钟

## 问题

单层注意力只是一个特征提取器，还不是一个模型。每层一次矩阵乘法不足以承载语言所需的容量。你需要深度——但没有正确的“ plumbing（管道工程）”，深度会崩溃。

2017 年 Vaswani 等人的论文打包了六项设计决策，把单层注意力变成了可堆叠的块。此后每一个 Transformer——仅编码器（encoder-only，如 BERT）、仅解码器（decoder-only，如 GPT）、编码器-解码器（encoder-decoder，如 T5）——都继承了同一副骨架。到了 2026 年，各个块已被改进（RMSNorm、SwiGLU、前置归一化 pre-norm、RoPE），但骨架依旧相同。

本课讲的就是这副骨架。后续课程会分别专门化：06 讲编码器，07 讲解码器，08 讲编码器-解码器。

## 概念

![编码器与解码器块的内部连接](../assets/full-transformer.svg)

### 六个组成部分

1. **嵌入（embedding）+ 位置信号。** 词元（token）→ 向量。位置信息通过 RoPE（现代）或正弦位置编码（经典）注入。
2. **自注意力（self-attention）。** 每个位置都 attends 到其他所有位置。在解码器中会加掩码（masked）。
3. **前馈网络（Feed-Forward Network，FFN）。** 逐位置的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比率为 4×。
4. **残差连接（residual connection）。** `x + sublayer(x)`。没有它，梯度在约 6 层之后就会消失。
5. **层归一化（layer normalization）。** `LayerNorm` 或现代的 `RMSNorm`。它稳定残差流（residual stream）。
6. **交叉注意力（cross-attention，仅解码器）。** 查询（query）来自解码器，键（key）和值（value）来自编码器输出。

### 编码器块（用于 BERT、T5 encoder）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。没有掩码。所有位置都能看到所有位置。

### 解码器块（用于 GPT、T5 decoder）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个块有三个子层。中间那个——交叉注意力——是信息从编码器流向解码器的唯一通道。在纯解码器架构（GPT）中，交叉注意力被省略，只剩下掩码自注意力 + FFN。

### 前置归一化（pre-norm）与 后置归一化（post-norm）

原始论文：`x + sublayer(LN(x))` 对比 `LN(x + sublayer(x))`。后置归一化在 2019 年左右失宠——没有仔细的 warmup（学习率预热），深层训练更困难。前置归一化（子层*之前*做 `LN`）是 2026 年的默认选择：Llama、Qwen、GPT-3+、Mistral 都采用它。

### 2026 年的现代化块

Vaswani 2017 年发布时使用的是 LayerNorm + ReLU。现代堆栈把两者都替换了。生产环境中的块实际长这样：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| 归一化（Normalization） | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展比率 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量匹配） |
| 位置编码 | 正弦绝对位置 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值居中（少一次减法），既节省计算又经验上同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 的论文中，稳定地以约 0.5 的困惑度（perplexity，ppl）优势击败 ReLU/GELU FFN。

### 参数量

对于一个块，设 `d_model = d`，FFN 扩展比率为 `r`：

- MHA：`4 · d²`（Q、K、V、O 四个投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致 Llama 3 8B）时，总量为：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = 每层约 15 亿参数 × 32 ≈ 70 亿`（加上嵌入层和输出头）。这与公布的参数量一致。

## 动手构建

### 第一步：基础模块

使用第 03 课中的小型 `Matrix` 类（为独立起见已复制到本文件）：

- `layer_norm(x, eps=1e-5)` —— 减均值、除标准差。
- `rms_norm(x, eps=1e-6)` —— 除 RMS，不减均值。
- `gelu(x)` 以及 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整连线请见 `code/main.py`。

### 第二步：搭建两层编码器与两层解码器

把它们堆叠起来。将编码器输出传入每一个解码器的交叉注意力。在输出投影前加一层最终 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第三步：在玩具样例上跑前向传播

喂入一段 6 个词元的源序列和一段 5 个词元的目标序列。验证输出形状为 `(5, vocab)`。本课不讲训练，只关注架构，因此没有损失（loss）。

### 第四步：换成 RMSNorm + SwiGLU

把 LayerNorm 和 ReLU-FFN 替换成 RMSNorm 和 SwiGLU。确认形状仍然匹配。只需替换一个函数，就完成 2026 年的现代化升级。

## 如何使用

PyTorch/TensorFlow 的参考实现是：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年的生产代码会自己手写块，因为：

- Flash Attention 是在注意力内部调用，而不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考实现里。
- RoPE、RMSNorm、SwiGLU 也不是 PyTorch 默认配置。

HuggingFace `transformers` 有非常干净的参考块，值得阅读：`modeling_llama.py` 是 2026 年典型的仅解码器块典范。大约 500 行代码，值得完整走一遍。

**编码器 vs 解码器 vs 编码器-解码器——如何选择：**

| 需求 | 选择 | 示例 |
|------|------|---------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、对话、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

仅解码器在语言任务中胜出，因为它扩展最干净，同时能处理理解与生成。编码器-解码器在输入具有明确“源序列”身份时仍然最佳（翻译、语音识别、结构化任务）。

## 交付

参见 `outputs/skill-transformer-block-reviewer.md`。该技能会按照 2026 年默认值审查一个新的 Transformer 块实现，并标出缺失部分（前置归一化、RoPE、RMSNorm、GQA、FFN 扩展比率）。

## 练习

1. **简单。** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 时，统计 `encoder_block` 的参数量。通过实现该块并用 `sum(p.numel() for p in block.parameters())` 验证。
2. **中等。** 从后置归一化切换到前置归一化。分别初始化两种结构，在随机输入上经过 12 层堆叠后测量激活范数。后置归一化的激活应该会爆炸；前置归一化的激活应保持有界。
3. **困难。** 实现一个 4 层编码器-解码器，完成玩具复制任务（复制反转后的 `x`）。训练 100 步，报告损失（loss）。再换成 RMSNorm + SwiGLU + RoPE——损失会下降吗？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 块（Block） | “一个 Transformer 层” | 归一化 + 注意力 + 归一化 + FFN 的堆叠，外包残差连接。 |
| 残差连接（Residual） | “跳跃连接（skip connection）” | `x + f(x)` 的输出；让梯度能流过深层堆叠。 |
| 前置归一化（Pre-norm） | “先归一化，而不是后归一化” | 现代做法：`x + sublayer(LN(x))`。深层训练无需复杂的 warmup。 |
| RMSNorm | “去掉均值的 LayerNorm” | 除以 RMS；少一次操作，经验上同样稳定。 |
| SwiGLU | “大家都换用的 FFN” | `Swish(W1 x) ⊙ W3 x → W2`。在语言模型困惑度上击败 ReLU/GELU。 |
| 交叉注意力（Cross-attention） | “解码器如何看到编码器” | Q 来自解码器，K/V 来自编码器输出的 MHA。 |
| FFN 扩展比率（FFN expansion） | “中间 MLP 有多宽” | 隐藏层大小与 d_model 的比率，通常是 4（LayerNorm）或 2.6（SwiGLU）。 |
| 无偏置（Bias-free） | “去掉 +b 项” | 现代堆栈省略线性层的偏置；困惑度略有提升，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 原始块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) —— 为什么前置归一化在深层更优。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) —— RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) —— SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) —— 2026 年典型的仅解码器块典范。
