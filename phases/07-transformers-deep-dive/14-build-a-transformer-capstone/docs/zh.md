# 从零构建 Transformer —— 大作业

> 十三节课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**前置要求：** 第 7 阶段 · 01 到 13。不要跳过。
**时间：** ~120 分钟

## 问题

你读过了每一篇论文。你已经实现了注意力（attention）、多头（multi-head）拆分、位置编码（positional encoding）、编码器（encoder）和解码器（decoder）模块、BERT 和 GPT 的损失（loss）、MoE、KV 缓存（KV cache）。现在，让它们在真实任务上协同工作。

大作业：在字符级语言建模任务上端到端训练一个小的仅解码器（decoder-only）Transformer。它读莎士比亚。它生成新的莎士比亚。它小到可以在笔记本电脑上不到 10 分钟完成训练。它正确到只要换更大的数据集、延长训练时间，就能得到一个真正的语言模型（LM）。

这是本课程的“nanoGPT”。它并非原创 —— Karpathy 2023 年的 nanoGPT 教程是每个学生至少都要写一遍的参考实现。我们借鉴其结构，并围绕我们已学的内容重新组织。

## 概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

架构说明：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们提供的内容

- `GPTConfig` —— 配置所有超参数（hyperparameters）的单一入口。
- `MultiHeadAttention` —— 因果（causal）、批处理（batched），可选类 Flash 路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` —— 现代前馈网络（FFN）。
- `Block` —— 预归一化（pre-norm）、残差（residual）包裹的注意力 + FFN。
- `GPT` —— 嵌入（embedding）、堆叠的模块、语言模型头（LM head）、generate()。
- 使用 AdamW、余弦学习率（cosine LR）、梯度裁剪（gradient clipping）的训练循环。
- 基于莎士比亚文本的字符级分词器（tokenizer）。

### 我们不提供的内容

- RoPE —— 在第 04 课中已概念性实现。这里为了简洁使用可学习的位置嵌入（learned positional embeddings）。练习中要求你换用 RoPE。
- 生成时的 KV 缓存（KV cache）—— 每个生成步骤都会重新计算整个前缀（prefix）上的注意力。更慢但更简单。练习中要求你添加 KV 缓存。
- Flash Attention —— PyTorch 2.0+ 会在输入匹配时自动调度；我们使用 `F.scaled_dot_product_attention`。
- MoE —— 每个模块只有一个 FFN。你在第 11 课中见过 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，一个 4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失（training loss）从约 4.2（随机）收敛到约 1.5，耗时约 6 分钟。
- 采样输出看起来有莎士比亚的风格：古旧词汇、换行、像 “ROMEO:” 这样的专有名词会出现。
- 验证损失（val loss，文本最后 10% 的留出集）紧跟训练损失；在这个规模和预算下没有过拟合。

## 构建

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。参见 `code/main.py`。脚本负责：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 90/10 的训练/验证划分。
- 在支持硬件上使用 bf16 自动混合精度（autocast）的训练循环。
- 训练完成后进行采样。

### 第 1 步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。词汇量（vocabulary）极小。适合 4 字节的 vocab_size。没有 BPE，没有分词器带来的麻烦。

### 第 2 步：模型

参见 `code/main.py`。模块是第 05 课中的教科书结构 —— 预归一化、RMSNorm、SwiGLU、因果多头注意力（causal MHA）。4/4/128 配置下的参数量（parameter count）：约 800K。

### 第 3 步：训练循环

获取长度为 256 的随机 token 窗口批次。前向传播。错位一位的交叉熵（shift-by-one cross-entropy）。反向传播。AdamW 更新。记录日志。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第 4 步：采样

给定一个提示（prompt），重复前向传播，从 top-p 的 logits 中采样，追加，继续。500 个 token 后停止。

### 第 5 步：阅读输出

2,000 步之后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但有莎士比亚的样子。对于约 800K 参数和笔记本电脑上 6 分钟来说，这已经是明显的胜利。

## 使用

这个大作业是一个参考架构。三个扩展方向可让它变得真正实用：

1. **替换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词汇量从 65 跳到约 50,000。模型容量需要相应扩大。
2. **在更大的语料上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单张 A100 上处理 100 亿 token、训练 1.25 亿参数的 GPT 约需 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习会分别带你完成每一步。

最终你会得到一个 1.25 亿参数、能生成流利英语的 GPT。不是前沿模型。但同样的代码路径 —— 只是规模更大 —— 正是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用于训练研究检查点的路径。

## 交付

参见 `outputs/skill-transformer-review.md`。该技能会从头开始审查 Transformer 实现，覆盖之前全部 13 节课的正确性。

## 练习

1. **简单。** 运行 `code/main.py`。验证训练模型最后一步的验证损失低于 2.0。将 `max_steps` 从 2,000 改为 5,000 —— 验证损失是否继续下降？
2. **中等。** 将可学习位置嵌入替换为 RoPE。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少一样低。
3. **中等。** 在采样循环中实现 KV 缓存。分别用缓存和不用缓存生成 500 个 token。在笔记本电脑上，wall-clock 时间应提升 5–20 倍。
4. **困难。** 为模型添加第二个头，预测下一个之后的 token（MTP —— DeepSeek-V3 的多 token 预测）。联合训练。有帮助吗？
5. **困难。** 将每个模块的单个 FFN 替换为 4 专家的 MoE。路由器（Router）+ top-2 路由。观察在相同活跃参数量下验证损失如何变化。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| nanoGPT | "Karpathy 的教程仓库" | 最小的仅解码器 Transformer 训练代码，约 300 行；经典参考实现。 |
| tinyshakespeare | "标准玩具语料" | 约 1.1 MB 文本；自 2015 年以来每个字符级语言模型教程都用它。 |
| Tied embeddings | "共享输入/输出矩阵" | LM head 的权重等于 token 嵌入矩阵的转置；节省参数，提升质量。 |
| bf16 autocast | "训练精度技巧" | 前向/反向用 bf16，优化器状态用 fp32；自 2021 年起成为标准。 |
| Gradient clipping | "阻止梯度尖峰" | 将全局梯度范数限制在 1.0；防止训练崩溃。 |
| Cosine LR schedule | "2020 年后的默认选择" | 学习率先线性上升（warmup），再按余弦曲线衰减到最高值的 10%。 |
| MFU | "模型浮点运算利用率" | 实际达到的 FLOPs / 理论峰值；2026 年稠密模型 40%、MoE 30% 已算强劲。 |
| Val loss | "留出集损失" | 模型从未见过的数据上的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 经典带注释的实现。
