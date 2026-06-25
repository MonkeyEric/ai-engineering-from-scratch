# LLaVA 与视觉指令微调（Visual Instruction Tuning）

> LLaVA（2023 年 4 月）是地球上被复现最多的多模态架构（multimodal architecture）。它用 2 层 MLP 替代了 BLIP-2 的 Q-Former，用朴素的令牌拼接（token concatenation）替代了 Flamingo 的门控交叉注意力（gated cross-attention），并用 GPT-4 从纯文本 caption 生成了 158k 条视觉指令轮次。2023 到 2026 年间，任何实践者只要构建过视觉语言模型（VLM），基本都会造出某种 LLaVA 变体。LLaVA-1.5 加入了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 将图像、多图和视频统一进一个配方。本课将拆解这个配方，实现投影器（projector），并解释为什么“更简单反而赢了”。

**Type:** Build
**Languages:** Python（stdlib，projector + instruction-template builder）
**Prerequisites:** Phase 12 · 02（CLIP），Phase 11（LLM Engineering — instruction tuning）
**Time:** ~180 分钟

## 学习目标

- 构建一个 2 层 MLP 投影器（projector），将 ViT 补丁嵌入（patch embeddings，维度 1024）映射到 LLM 的嵌入维度（维度 4096）。
- 掌握 LLaVA 两阶段配方：
  1. 在 558k caption 对上对齐投影器；
  2. 在 158k 条 GPT-4 生成的轮次上进行视觉指令微调（visual instruction tuning）。
- 构造 LLaVA 格式提示词（prompt）：包含图像令牌占位符（image token placeholder）、系统提示词（system prompt）以及用户/助手轮次。
- 解释为什么社区从 Q-Former 转向 MLP，尽管 Q-Former 在令牌预算（token budget）上占优。

## 问题背景

BLIP-2 的 Q-Former（第 12.03 课）把一张图像压缩成 32 个令牌。干净、高效、基准测试表现不错。但它有两个问题。

第一，Q-Former 虽然是可训练的，但它的损失（loss）并非最终任务。第 1 阶段训练 ITC+ITM+ITG，第 2 阶段训练 LM 损失。查询（queries）学到的是某种中间表示，LLM 还得再去解码。信息在瓶颈（bottleneck）中丢失了。

第二，Q-Former 占 188M 参数，在 LLaVA 所处的 2023 年规模下，你必须把它和目标 LLM 共同设计。换 LLM，就要重训 Q-Former；换视觉编码器，也要重训。每一种组合都是一项独立的研发项目。

LLaVA 的答案简单得令人尴尬：取出 ViT 的 576 个补丁令牌（patch tokens），每个通过一个 2 层 MLP（`1024 → 4096 → 4096`），然后把全部 576 个直接丢进 LLM 的输入序列。没有瓶颈，没有针对奇怪目标的第 1 阶段预训练，只用直接的 LM 损失来训练 MLP。

数据从何而来？LLaVA 的第二个洞察是：用仅文本的 GPT-4 生成指令数据。向 GPT-4 输入图像的 COCO caption 和边界框（bounding-box）数据，让它生成对话、详细描述和复杂推理问题。无需人工标注，就获得了 158k 条指令-回答轮次。

结果：一个 VLM，在 8 张 A100 上跑一天，在 MMMU 上击败 Flamingo，并发布了一个社区可扩展的开放检查点（checkpoint）。到 2023 年底，它已衍生出 50 多个分支。

## 核心概念

### 架构

LLaVA-1.5 13B：
- 视觉编码器：CLIP ViT-L/14 @ 336（第 1 阶段冻结，第 2 阶段可选择性解冻）。
- 投影器（Projector）：2 层 MLP，GELU 激活，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来也用到 Llama-3.1-8B）。

图像 + 文本提示词的前向传播：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占据 LLM 上下文的 576 个令牌。在 2048 上下文下，剩余 1472 个文本令牌；在 32k 上下文下，它只是零头。

### 第 1 阶段：投影器对齐

冻结 ViT，冻结 LLM，只训练 2 层 MLP。数据集：558k 图像-caption 对（LAION-CC-SBU）。损失：基于投影图像令牌条件下的 caption 语言建模。

单 epoch、batch 128 的情况下，几小时内即可完成。投影器学习将 ViT 空间映射到 LLM 空间，没有任务特定的监督。

### 第 2 阶段：视觉指令微调

解冻投影器（仍保持可训练），解冻 LLM（通常完全解冻，有时使用 LoRA），在 158k 视觉指令轮次上训练。

指令数据才是关键。Liu 等人通过以下步骤生成：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工 caption + 边界框列表）。
3. 用三个提示词模板发送给 GPT-4：
   - 对话："Generate a back-and-forth dialogue between a user and assistant about this image."
   - 详细描述："Give a rich, detailed description of the image."
   - 复杂推理："Ask a question that requires reasoning about the image, then answer it."
4. 将 GPT-4 的输出解析为 (instruction, response) 对。

这些步骤完全不接触真实图像——只使用文本描述。GPT-4 会产生幻觉（hallucinate）并补全看似合理的图像内容。存在一定噪声，但它有效：158k 轮次足以解锁对话能力。

### 为什么社区复制这个方案

- 无需调节第 1 阶段特定的损失，全程使用 LM 损失。
- 投影器几小时即可训完，而非几天。
- LLM 可以灵活替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重训投影器。
- 视觉指令数据流水线基于 GPT-4，为新领域重新生成的成本很低。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）增加了：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的系统提示词。
- 上下文从 2048 扩展到 32k。

LLaVA-NeXT（2024 年 1 月）增加了：
- AnyRes：将高分辨率图像切分为 2x2 或 1x3 的 336x336 网格 crop，再加一个全局低分辨率缩略图。每个 crop 变为 576 个令牌；每张图像总共约 2880 个视觉令牌。OCR 和图表任务性能大幅跃升。
- 更好的指令数据混合，使用 ShareGPT4V（高质量 GPT-4V caption）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 课深入讲解 OneVision。简而言之：使用同样的投影器，但通过课程学习（curriculum）同时覆盖单图、多图和视频，共享视觉令牌预算。

### 与 Q-Former 的对比

| | Q-Former（BLIP-2） | MLP（LLaVA） |
|---|---|---|
| 每张图像的视觉令牌 | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 188M + LM | 40M + LM |
| 第 1 阶段损失 | ITC+ITM+ITG | 仅 LM |
| LLM 即插即用 | 需要重训 | 少量重训即可替换 |
| 多图 | 不自然 | 自然（拼接） |
| 视频 | 不自然 | 自然（逐帧拼接） |
| 令牌预算 | 小 | 大 |

MLP 在简洁性和令牌灵活性上获胜。Q-Former 在令牌预算上获胜。到 2023 年底，令牌预算已不再是瓶颈（LLM 上下文扩展到 32k–128k+），简洁性因此占上风。

### 提示词格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位符令牌（placeholder）。在分词前，它被替换为 576 个视觉令牌（AnyRes 下为 2880）。分词器（tokenizer）看到的序列比训练时略长，但 LLM 能够处理这种新输入，因为第 1 阶段已经教会了它。

### 参数经济性

LLaVA-1.5-7B 的组成：
- CLIP ViT-L/14 @ 336：303M（第 1 阶段冻结，第 2 阶段常解冻）。
- 投影器（2 个线性层）：约 22M 可训练。
- Llama-7B：7B。
- 总计：7.3B 参数。第 2 阶段可训练：完整 7B + 22M 投影器。

第 2 阶段训练成本：约 20 小时，8xA100。这是关键数字——一天、一台机器、可复现。这就是 LLaVA 得以传播的原因。

## 使用它

`code/main.py` 实现了：

1. 2 层 MLP 投影器（玩具规模为 dim 16 → 32 → 32），用纯 Python 实现。
2. 提示词构建流水线：系统提示 + 将 `<image>` 替换为 N 个投影令牌 + 用户轮次 + 助手生成占位符。
3. 一个可视化工具，展示 576 令牌视觉块在 LLM 上下文中的占比（消耗 2k / 32k / 128k 上下文的百分比）。

## 交付它

本课生成 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 系列检查点，它运行 10 个提示词的 vibes-eval 套件（3 个 caption、3 个 VQA、2 个推理、2 个拒绝），并输出人类可读的记分卡（scorecard）。这不是基准测试，而是一次冒烟测试（smoke test），用于确认投影器与 LLM 连接良好。

## 练习

1. 计算 2 层 MLP 投影器 `1024 → 4096 → 4096` 的可训练参数数量。加上 GELU 和 bias，它占 LLaVA-13B 的多大比例？

2. 为“拒绝”场景构造一个 LLaVA 提示词——图像中包含一位私人个体。写出期望的助手回答。为什么 LLaVA 应该在该零样本（zero-shot）情况下拒绝？需要哪些训练数据来强化这种拒绝？

3. 阅读 LLaVA-NeXT 博客的 AnyRes 部分。计算 1344x672 图像在 AnyRes 下的视觉令牌数量，并与 336x336 下的基础 576 令牌进行比较。

4. LLaVA 第 1 阶段投影器使用 caption 上的 LM 损失进行训练。如果跳过第 1 阶段直接进入第 2 阶段（视觉指令微调）会怎样？引用 Prismatic VLMs 消融实验（arXiv:2402.07865）作答。

5. LLaVA-Instruct-150k 使用 GPT-4 和 COCO caption 生成指令。对于一个新领域（医学 X 光、卫星图像），描述生成领域指令的四步数据流水线。每一步可能出现什么问题？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Projector | "MLP bridge" | 带 GELU 的 2 层 MLP，将 ViT 维度映射到 LLM 维度 |
| Image token | "<image> placeholder" | 在推理前被替换为 N 个投影视觉令牌的提示词标记 |
| Visual instruction tuning | "LLaVA stage 2" | 在 GPT-4 生成的 (image, instruction, response) 三元组上训练 |
| Stage 1 alignment | "Projector pretraining" | 冻结 ViT 和 LLM，用 caption 上的 LM 损失训练投影器 |
| AnyRes | "Multi-crop tiling" | 将高分辨率图像切分为瓦片网格，并拼接每个瓦片的视觉令牌 |
| LLaVA-Instruct | "GPT-4-generated" | 从 COCO caption + GPT-4 合成的 158k 指令-回答对 |
| Vision encoder freeze | "Backbone locked" | CLIP 权重在第 1 阶段不更新，有时第 2 阶段也不更新 |
| ShareGPT4V | "Better captions" | 100 万条由 GPT-4V 生成的密集 caption，用于更高质量的对齐 |
| VQA | "Visual question answering" | 关于图像的自由形式问答任务 |
| Prismatic VLMs | "Design-space paper" | Karamcheti 2024 年系统测试投影器和数据选择的消融实验论文 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集 caption 数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融实验。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一的单图、多图、视频模型。
