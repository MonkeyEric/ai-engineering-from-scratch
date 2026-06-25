# Chameleon 与早期融合的纯 Token 多模态模型

> 到目前为止，我们见过的所有视觉语言模型（VLM）都把图像和文本分开处理。视觉 token 来自视觉编码器，经过投影器后，再在大语言模型（LLM）内部与文本相遇。视觉和文本词表从不重叠。Chameleon（Meta，2024 年 5 月）提出了一个问题：如果它们重叠会怎样？训练一个 VQ-VAE，把图像变成来自共享词表的离散 token 序列。每个多模态文档现在都是一个序列——文本 token 和图像 token 交错排列，使用单一的自回归损失。副作用是：模型可以生成混合模态输出——在单次推理调用中交替生成文本和图像 token。本节课阅读早期融合的核心论点，并从头构建一个玩具版本。

**Type:** Build
**Languages:** Python（stdlib，VQ-VAE tokenizer + interleaved decoder）
**Prerequisites:** Phase 12 · 05, Phase 8（Generative AI）
**Time:** ~180 分钟

## 学习目标

- 解释共享词表 + 单一损失如何改变模型的能力。
- 描述 VQ-VAE 如何将图像 token 化为与 transformer 的 next-token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 位置、LayerNorm 顺序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，并描述何时应该选择哪一种。

## 问题所在

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）把文本和图像当作两种不同的东西。文本 token 经过 `embed(text_token)`；图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，中途才合并。

三个后果：

1. LLM 只能消费图像，不能生成图像。输出只有文本。
2. 混合模态文档（像文章一样交替出现段落和图像）处理起来很别扭——你要么在模型外部解析多模态输入，要么串联多次生成。
3. 分布不匹配。视觉 token 和文本 token 位于隐藏空间的不同区域，造成微妙的对齐问题。

Chameleon 否定了这个前提：图像只是来自共享词表的离散 token 序列。在交错的文档上训练模型，一个损失，一个自回归解码器，你就能免费解锁混合模态生成。

## 核心概念

### 作为图像分词器的 VQ-VAE

这个分词器（tokenizer）是向量量化变分自编码器。架构如下：

- 编码器（Encoder）：CNN + ViT，将图像映射为空间特征图，例如 32×32、维度 256 的特征。
- 码本（Codebook）：K 个可学习向量（Chameleon 使用 8192），维度同样为 256。
- 量化（Quantization）：对每个空间特征，按 L2 距离查找最近的码本条目，用整数索引替换连续特征。
- 解码器（Decoder）：CNN，将量化后的特征还原为像素。

训练：VAE 重建损失 + commitment 损失 + 码本损失。码本索引构成图像的离散字母表。

对于 Chameleon：一张图像变成 32×32 = 1024 个 token，来自大小为 8192 的词表。再与文本 token（来自 LLM 的 BPE 词表，例如 32000）拼接。最终词表大小：40192。Transformer 看到一个序列，一个损失。

### 共享词表

Chameleon 的词表组合了文本 token、图像 token 和模态分隔符。每个 token 只有一个 ID。输入嵌入层将每个 ID 映射为 D 维隐藏向量。输出投影将隐藏向量映射回词表 logits。Softmax 选择下一个 token，无论什么模态。

分隔符很重要：`<image>` 和 `</image>` 标签包围图像 token 序列。在生成时，如果模型输出 `<image>`，下游软件就知道接下来 1024 个 token 是要送给解码器渲染像素的 VQ 索引。

### 混合模态生成

推理就是在共享词表上做 next-token 预测。示例提示词："画一只猫并描述它。"Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主决定顺序——可能先图像后文本，先文本后图像，或交错生成。同一个解码器，同一个损失。

与适配器 VLM 相比，后者只能生成文本。Chameleon 重新开启了模型输出模态的问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 顺序

早期融合训练在大规模上不稳定。Chameleon 的论文记录了三个技巧：

- QK-Norm。在注意力（attention）内部，对 query 和 key 投影做 LayerNorm，再做点积。防止深度上的 logit 幅度爆炸。2024 年后的多个大模型都在使用。
- Dropout 位置。在每次残差相加后都加 dropout，而不仅仅在注意力和 MLP 之后。当图像 token 的梯度可能占主导时，需要更强的正则化。
- LayerNorm 顺序。在残差分支上使用 Pre-LN（标准做法），并在最后一个块的跳跃连接上额外加一个 LN。稳定最终层的梯度流。

没有这些技巧，340 亿参数 Chameleon 的训练在多个检查点发散。有了这些技巧后，模型收敛。训练配方与架构本身一样重要。

### 分词器的重建上限

VQ-VAE 是有损的。在码本大小 8192、每 512×512 图像 1024 个 token 的情况下，重建 PSNR 上限约为 26–28 dB。这足以生成可识别的图像，但明显比连续空间扩散差（Stable Diffusion 3 达到 32+ dB）。

分词器是瓶颈。更好的分词器（MAGVIT-v2、IBQ、SBER-MoVQGAN）能提升上限。Emu3（第 12.12 课）仅通过更好的分词器就达到了 SDXL 级别的生成质量。

### Chameleon 与 BLIP-2 / LLaVA

Chameleon（早期融合，共享词表）：
- 一个损失，一个解码器。
- 可生成混合模态输出。
- 分词器决定质量上限。
- 成本高：推理路径上每生成一张图都要跑 VQ-VAE 解码器。

BLIP-2 / LLaVA（晚期融合，独立塔）：
- 视觉进，文本出。
- 复用预训练 LLM。
- 理解任务没有分词器瓶颈。
- 成本低：单次前向传播。

按任务选择。如果需要图像生成，选 Chameleon 家族。如果只需要理解，适配器 VLM 更简单，能复用更多预训练算力。

### Fuyu 与 AnyGPT

Fuyu（Adept，2023）是相关方法：完全跳过独立的视觉编码器，把原始图像 patch 通过 LLM 的输入投影当成 token 输入，无需分词器。比 Chameleon 简单，但失去了共享词表的输出生成能力。

AnyGPT（Zhan 等人，2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种模态都用同样的 VQ-VAE 技巧，共享 transformer。任意模态到任意模态生成。在第 12.16 课中有更多介绍。

## 动手实践

`code/main.py` 构建了一个端到端的玩具早期融合模型：

- 一个微型 VQ-VAE 风格的量化器，将 8×8 patch 映射为码本索引（K=16）。
- 共享词表包括（文本 ID 0..31）+（图像 ID 32..47）+（分隔符 48、49）。
- 一个玩具自回归解码器（bigram 表），在合成标题 + 图像 token 序列上训练。
- 给定提示词后，输出交替文本 + 图像 token 的采样循环。

代码故意让 transformer 保持极小（bigram），以便你能从头到尾追踪信号流。

## 交付成果

本节课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品需求（仅理解 vs 理解 + 生成、所需图像质量、成本预算），它在 Chameleon 家族（早期融合）和 LLaVA 家族（晚期融合）之间做出选择，并用定量的经验法则说明理由。

## 练习

1. Chameleon 使用 K=8192 个码本条目和每 512×512 图像 1024 个 token。与 24 位 RGB 图像相比，估计压缩比。它是有损的吗？有多有损？

2. 同样 VQ-VAE 密度下，一张 4K 图像（3840×2160）会产生多少图像 token？Chameleon 风格的模型能单次推理生成 4K 图像吗？什么会先崩溃——上下文长度、分词器质量，还是 KV cache？

3. 用纯 Python 实现 QK-Norm。给定 64 维 query 和 key，展示 LayerNorm 前后的点积。为什么在深度上控制幅度很重要？

4. 阅读 Chameleon 论文第 2.3 节关于训练稳定性的内容。描述在没有 QK-Norm 的情况下，340 亿参数模型观察到的确切失效模式。"norm 爆炸"的特征是什么？

5. 扩展玩具解码器，使其在仅文本提示词下输出混合模态响应。在训练数据分布为 60% 文本优先 / 40% 图像优先的情况下，测量模型选择图像优先的频率。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 早期融合（Early fusion） | "统一 token" | 图像被转换为与 transformer 词表从第一步就共享的离散 token |
| VQ-VAE | "图像分词器" | CNN + ViT + 码本，将图像映射为 transformer 可预测的整数索引 |
| 共享词表（Shared vocabulary） | "一个字典" | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | "注意力稳定器" | 在 query 和 key 做点积前对它们应用 LayerNorm，防止 norm 爆炸 |
| 混合模态生成（Mixed-modality generation） | "文本 + 图像输出" | 推理能自主地在一次前向传播中产生交错的文本和图像 token |
| 码本大小（Codebook size） | "K 个条目" | VQ-VAE 可量化到的离散向量数量；在压缩率和保真度之间权衡 |
| 分词器上限（Tokenizer ceiling） | "重建限制" | 解码 VQ token 能达到的最佳 PSNR；限制模型的图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
