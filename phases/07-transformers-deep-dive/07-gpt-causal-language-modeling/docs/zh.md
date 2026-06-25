# GPT — 因果语言建模

> BERT 能看两边，GPT 只看过去。三角掩码（triangle mask）是现代 AI 中最具影响力的一行代码。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 02（自注意力 Self-Attention）、阶段 7 · 05（完整 Transformer）、阶段 7 · 06（BERT）
**时长：** 约 75 分钟

## 问题背景

语言模型回答一个问题：给定前 `t-1` 个词元（token），第 `t` 个词元的概率分布是什么？用“下一个词元预测”这一信号训练，你就能得到一个可以逐词元生成任意文本的模型。

为了在整段序列上端到端并行训练，你需要让每个位置的预测只依赖更早的位置。否则模型会直接偷看答案作弊。

因果掩码（causal mask）就干这件事。它是一个上三角矩阵，在 softmax 之前给注意力分数加上 `-inf`。softmax 之后，这些位置变成 0。每个位置只能关注自身及之前的位置。因为你只对整个序列应用一次，所以一次前向传播就能得到 N 个并行的下一个词元预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是仅解码器（decoder-only）的因果 Transformer，核心循环相同。只是规模更大、数据更好、RLHF 更优。

## 核心概念

![因果掩码生成三角注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构造一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前把 `M` 加到原始注意力分数上。`exp(-inf) = 0`，因此被掩码的位置权重为 0。注意力矩阵的每一行都仅是前面位置的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒级。对领域的影响：无远弗届。

### 并行训练，串行推理

训练：一次性前向传播整个 `(N, d_model)` 序列，计算 N 个交叉熵（cross-entropy）损失（每个位置一个），求和，反向传播。沿序列方向并行。这就是 GPT 训练可扩展的原因——一次 GPU 前向传播就能处理一个批次里的 100 万词元。

推理：逐个词元生成。输入 `[t1, t2, t3]` 得到 `t4`；输入 `[t1, t2, t3, t4]` 得到 `t5`；输入 `[t1, t2, t3, t4, t5]` 得到 `t6`。KV 缓存（第 12 课）保存 `t1…tn` 的隐藏状态，避免每步重复计算。但推理时的串行深度等于输出长度。这就是自回归代价，也是解码成为每个大语言模型（LLM）延迟瓶颈的原因。

### 损失函数——错位一位

给定词元 `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`，然后求和。这就是整段序列的交叉熵。

你听说过的每个 Transformer 语言模型都用这个损失训练。预训练、微调、SFT——同样的损失，不同的数据。

### 解码策略

训练完成后，采样策略的选择比多数人想的更重要。

| 方法 | 作用 | 适用场景 |
|------|------|----------|
| 贪心（Greedy） | 每步取 Argmax | 确定性任务、代码补全 |
| 温度（Temperature） | 将 logits 除以 T 后采样 | 创造性任务，T 越大多样性越高 |
| Top-k | 只从前 k 个词元中采样 | 截断低概率尾部 |
| Top-p（核采样 Nucleus） | 从累积概率 ≥ p 的最小集合中采样 | 2020 年后的默认策略；能根据分布形状自适应 |
| Min-p | 保留满足 `p > min_p * max_p` 的词元 | 2024 年后提出；比 top-p 更擅长剔除长尾 |
| 投机解码（Speculative decoding） | 小模型起草 N 个词元，大模型并行验证 | 在同等质量下降低 2–3 倍延迟 |

2026 年，min-p + 温度 0.7 是开放权重模型的一个合理默认配置。投机解码已是任何生产级推理栈的标配。

### “GPT 配方”何以奏效

1. **仅解码器（Decoder-only）。** 没有编码器开销。每层只做一次注意力 + FFN。
2. **规模化（Scaling）。** 124M → 1.5B → 175B → 万亿参数。Chinchilla 缩放定律（第 13 课）告诉你如何分配算力。
3. **上下文学习（In-context learning）。** 在 6B–13B 参数区间涌现。模型无需微调即可遵循少样本示例。
4. **RLHF。** 基于人类偏好的后训练把原始预训练文本模型变成了聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 支撑大规模稳定训练。

自 GPT-2 以来，核心架构变化不大。所有有趣的事情都发生在数据、规模和后训练上。

## 动手实现

### 步骤 1：因果掩码

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前把它加到注意力分数上。这就是全部机制。

### 步骤 2：一个两层 GPT 风格模型

堆叠两个解码器块（带掩码的自注意力 + FFN，无交叉注意力）。加上词元嵌入（token embedding）、位置编码（positional encoding）以及反嵌入层（unembedding，与词元嵌入矩阵共享权重——这是自 GPT-2 以来的标准技巧）。

### 步骤 3：端到端的下一个词元预测

在一个 20 词元的玩具词表上，为每个位置生成 logits。与错位一位的目标计算交叉熵损失。不求梯度——这是一次前向传播的 sanity check。

### 步骤 4：采样

实现贪心、温度、top-k、top-p、min-p。用固定提示分别运行并比较输出。一个采样函数只需 10 行左右。

## 实际使用

PyTorch，2026 年惯用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 运行前向传播，取出最后一个位置的 logits，采样下一个词元，追加到序列，然后重复。每个生产级 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都在以大量优化实现同一个循环——批处理预填充（batched prefill）、连续批处理（continuous batching）、KV 缓存分页（KV cache paging）、投机解码（speculative decoding）。

**GPT 与 BERT 一句话概括：** GPT 预测 `P(x_t | x_{<t})`，BERT 预测 `P(x_masked | x_unmasked)`。损失函数决定了模型能否生成文本。

## 交付产物

见 `outputs/skill-sampling-tuner.md`。该 skill 会为新的生成任务挑选采样参数，并在需要确定性解码时给出提示。

## 练习

1. **简单。** 运行 `code/main.py`，验证因果注意力矩阵在 softmax 后是下三角的。抽查：第 3 行应该只在第 0–3 列有权重。
2. **中等。** 实现束宽为 4 的束搜索（beam search）。在 10 个短提示上比较 beam-4 与贪心的困惑度（perplexity）。beam 总是更好吗？（提示：通常在翻译任务上有效，但在开放式聊天中不一定。）
3. **困难。** 实现投机解码：用一个极小的两层模型作为起草模型（draft），一个六层模型作为验证模型（verifier）。在 100 条长度为 64 的补全上测量实际 wall-clock 加速比。确认输出与验证模型贪心解码的结果一致。

## 关键术语

| 术语 | 俗称 | 实际含义 |
|------|------|----------|
| 因果掩码（Causal mask） | “三角掩码” | 加到注意力分数上的上三角 `-inf` 矩阵，使位置 `i` 只能看到 `≤ i` 的位置。 |
| 下一个词元预测（Next-token prediction） | “那个损失” | 每个位置上模型分布与真实下一个词元之间的交叉熵。 |
| 自回归（Autoregressive） | “一次生成一个” | 把输出反馈为输入；只有训练时能并行，生成时不能。 |
| Logits | “softmax 前的分数” | LM 头在 softmax 之前的原始输出；采样就在这些值上进行。 |
| 温度（Temperature） | “创造力旋钮” | 将 logits 除以 T；T→0 等价于贪心，T→∞ 接近均匀分布。 |
| Top-p | “核采样（Nucleus sampling）” | 把分布截断到累积概率 ≥ p 的最小集合，再从中采样。 |
| Min-p | “比 top-p 更好” | 保留满足 `p ≥ min_p × max_p` 的词元；根据分布尖锐程度自适应截断。 |
| 投机解码（Speculative decoding） | “起草 + 验证” | 廉价模型提议 N 个词元；大模型并行验证。 |
| 强制教学（Teacher forcing） | “训练技巧” | 训练时传入真实的上一个词元，而不是模型自己的预测。每个 seq2seq 语言模型的标准做法。 |

## 扩展阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) —— GPT-1 论文。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) —— GPT-2 论文。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) —— GPT-3 与上下文学习论文。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 投机解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) —— 因果语言模型的权威参考代码。
