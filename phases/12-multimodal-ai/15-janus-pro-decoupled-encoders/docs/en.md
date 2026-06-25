# Janus-Pro：统一多模态模型（unified multimodal models）的解耦编码器

> 统一多模态模型存在一种难以回避的张力。理解（understanding）需要语义特征——SigLIP 或 DINOv2 输出富含概念级信息的向量。生成（generation）需要利于重建的编码——VQ token 能够组合回清晰的像素。这两个目标无法由单个编码器（encoder）同时满足。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）提出的解决方案是：不再强求，将两个编码器解耦。在不同任务间共享 transformer 主体（transformer body），但将理解路径路由给 SigLIP，将生成路径路由给 VQ tokenizer。在 7B 规模下，Janus-Pro 在 GenEval 上超越 DALL-E 3，同时在 MMMU 上持平 LLaVA。本课将剖析为何“两个编码器”能在“一个编码器”失败的地方奏效。

**类型：** Build  
**语言：** Python（标准库，双编码器路由 + 共享主体信号）  
**先修：** Phase 12 · 13（Transfusion）、Phase 12 · 14（Show-o）  
**时长：** 约 120 分钟

## 学习目标

- 解释为何单个共享编码器会损害理解或生成质量。
- 描述 Janus-Pro 的路由（routing）机制：理解侧在输入端使用 SigLIP 特征，生成侧在输入和输出端均使用 VQ token。
- 追溯数据混合（data mix）扩展如何让 Janus-Pro 在 Janus 未能成功的地方取得突破。
- 比较解耦架构（Janus-Pro）、耦合连续架构（Transfusion）与耦合离散架构（Show-o）。

## 问题背景

统一模型在理解与生成之间共享 transformer 主体。此前的尝试（Chameleon、Show-o、Transfusion）都为两个方向使用同一个视觉 tokenizer。这个 tokenizer 本身就是一种折中：

- 为重建（生成）优化：VQ-VAE 能捕捉细粒度像素细节，但生成的 token 语义一致性较弱。
- 为语义（理解）优化：SigLIP 嵌入（embedding）能把“猫”的图像聚拢到“猫”的 token 附近，但无法很好地重建图像。

Show-o 和 Transfusion 为此付出了代价：至少有一个方向的质量明显下降。Janus-Pro 反问：当两个任务的需求不同时，为何非要一个 tokenizer？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构将两个编码器分离：

- **理解路径（Understanding path）。** 输入图像 → SigLIP-SO400m → 2 层 MLP → transformer 主体。
- **生成路径（Generation path）。** 输入图像（若基于已有图像进行条件生成）→ VQ tokenizer → token ID → transformer 主体。
- **输出生成。** transformer 预测的图像 token → VQ 解码器（decoder）→ 像素。

transformer 主体是共享的，而主体上下游的所有组件都是任务专属的。

输入通过提示格式消除歧义：`<understand>` 标签经 SigLIP 路由，`<generate>` 经 VQ 路由。也可以根据任务隐式路由。

### 为何有效

理解损失（understanding loss）获得 SigLIP 特征——这类特征经过 CLIP 风格的预训练，已针对语义相似性调优。模型在感知基准测试上的表现优于 Show-o / Transfusion，因为输入特征更适合该任务。

生成损失（generation loss）获得 VQ token——这类 token 经过 tokenizer 调优，利于重建。图像质量优于 Show-o，因为 VQ 编码能干净地组合回像素。

共享的 transformer 主体接触两种输入分布（SigLIP 和 VQ），并学会与两者协同。其核心论断是：只要有足够的数据和参数量，主体就能内化这种切换。

### 数据扩展——Janus 与 Janus-Pro 的对比

Janus（原始版本，arXiv 2410.13848）提出了解耦思想，但规模较小（1.3B 参数、数据有限）。Janus-Pro（arXiv 2501.17811）进行了扩展：

- 70 亿参数（对比 13 亿）。
- 第一阶段（对齐，alignment）使用 9000 万图文对，从 7200 万提升。
- 第二阶段（统一训练，unified）使用 7200 万，从 2600 万提升。
- 第三阶段新增 20 万张图像生成指令样本。

结果是：Janus-Pro-7B 在 MMMU 上持平 LLaVA（60.3 对比约 58），并在 GenEval 上击败 DALL-E 3（0.80 对比 0.67）。一个开放模型，在统一光谱的两端都具备竞争力。

### JanusFlow——整流流（rectified flow）变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为整流流生成路径（连续的）。分工变为 SigLIP 负责理解 + 整流流负责生成。质量上限进一步提升，但架构仍保持“解耦编码器 + 共享主体”。

### 共享主体（shared body）的职责

transformer 主体处理统一的序列，但面临两种输入分布。它的职责是：

- **理解任务：** 接收 SigLIP 特征 + 文本 token → 自回归地输出文本。
- **生成任务：** 接收文本 token +（可选的图像 VQ token）→ 自回归地输出图像 VQ token。

主体的每个块都没有模态专属权重。它就是你在 Qwen 或 Llama 中见到的那种文本风格 transformer，再加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主体可以用预训练大语言模型（LLM）初始化。Janus-Pro 确实基于 DeepSeek-MoE-7B 初始化。这一选择至关重要：LLM 带来的推理能力是纯从头训练的统一模型难以企及的。

### 与 InternVL-U 的对比

InternVL-U（第 12.10 课）是 2026 年的后续工作。它结合了：

- 原生多模态预训练（InternVL3 骨干网络，backbone）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头输出）。
- 统一的理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择纳入更大的框架。解耦编码器的思想如今已成为大规模统一模型的默认方案。

### 局限性

解耦编码器增加了架构复杂度：要训练两个 tokenizer、维护两条输入路径、面对两套失效模式。对于不需要生成的产品，Janus-Pro 属于过度设计——选择 LLaVA 系列的理解模型即可。

对于不需要理解的产品，Janus-Pro 则能力过剩——选择 Stable Diffusion 3 / Flux 模型即可。

对于同时需要两者的产品，Janus-Pro 已成为参考性的开放架构。

## 动手实践

`code/main.py` 模拟 Janus-Pro 的路由：

- 两个模拟编码器：类 SigLIP（生成 256 维语义向量）和类 VQ（生成整数编码）。
- 一个提示路由器（prompt router），根据任务标签选择编码器。
- 一个共享主体（stand-in），无论 token 序列来自哪个编码器，都统一处理。
- 一个从阶段 1（对齐）到阶段 3（指令微调，instruction tune）的加权采样调度切换。

打印三个示例的路由路径：图像问答（image QA）、文本到图像生成（T2I）、图像编辑。

## 交付成果

本课将产出 `outputs/skill-decoupled-encoder-picker.md`。针对一个希望以接近前沿质量同时实现统一生成与理解的产品，它会从 Janus-Pro、JanusFlow 或 InternVL-U 中做出选择，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败了 DALL-E 3。请解释为何一个 70 亿参数的开放模型能在生成任务上匹敌前沿闭源模型，却在理解任务上无法做到。
2. 实现一个路由函数：给定提示文本，将其分类为 `understand` 或 `generate`。对于“先描述再画草图”这类模糊提示，你如何处理？
3. JanusFlow 将 VQ 路径替换为整流流。transformer 主体现在输出什么？损失函数（loss）又有何变化？
4. 为 Janus-Pro 架构提议第四项可通过新增一个解耦编码器处理的任务。例如：图像分割（DINO 风格）、深度估计（MiDaS 风格）。
5. 阅读 Janus-Pro 论文第 4.2 节关于数据扩展的内容。相比 Janus，哪个数据阶段对文本到图像（T2I）质量提升贡献最大？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| 解耦编码（Decoupled encoding） | “两个视觉编码器” | 每个方向使用独立的 tokenizer 或编码器：理解用语义型，生成用重建型 |
| 共享主体（Shared body） | “一个 transformer” | 单个 transformer 处理任一编码器的输出；没有模态专属权重 |
| 用于理解的 SigLIP | “语义特征” | CLIP 家族视觉塔，提供丰富的概念特征，但重建能力差 |
| 用于生成的 VQ | “重建编码” | 向量量化（vector-quantized）token，能干净地解码回像素 |
| JanusFlow | “整流流变体” | 将 Janus-Pro 的 VQ 生成头替换为连续流匹配（flow-matching）生成头的版本 |
| 路由标签（Routing tag） | “任务标签” | 提示标记（`<understand>` / `<generate>`），用于选择输入编码器 |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
