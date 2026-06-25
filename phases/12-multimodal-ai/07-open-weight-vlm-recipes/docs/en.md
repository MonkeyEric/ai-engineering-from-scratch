# 开源权重视觉语言模型（VLM）实践配方：真正重要的是什么

> 2024-2026 年的开源权重 VLM 文献充斥着消融表。苹果的 MM1 测试了图像编码器（image encoder）、连接器（connector）与数据混合 13 种组合；艾伦人工智能研究所的 Molmo 证明，详细的人工描述优于 GPT-4V 蒸馏；Cambrian-1 对比了 20 余种编码器；Idefics2 形式化了五轴设计空间；Prismatic VLMs 在受控基准上比较了 27 种训练配方。在一片嘈杂中，有一小撮结论跨论文成立：图像编码器比连接器架构更重要，数据混合又比前两者都重要，而详细的人工描述优于蒸馏得到的合成数据。本节课替你读懂这些表格，省去你亲自翻阅之苦。

**类型：** 学习 + 实验
**语言：** Python（标准库，消融表解析器 + 配方选择器）
**前置条件：** 第 12 阶段 · 05（LLaVA 基线）
**时长：** 约 180 分钟

## 学习目标

- 说出 VLM 设计空间的五个轴：图像编码器、连接器、大语言模型（LLM）、数据混合、分辨率调度。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融表，并预测调节哪个旋钮会带动某个基准。
- 给定算力预算与任务组合，为新 VLM 挑选配方（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数下，详细的人工描述会击败 GPT-4V 蒸馏。

## 问题所在

开源权重的 VLM 有数百个。“好”与“最先进”之间的差距大多不在架构，而在数据、分辨率调度和编码器选择。知道模型表现不佳时该先拧哪个旋钮，能帮你避免五百万 GPU 小时的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）采用“图像-文本对预训练 + LLaVA-Instruct-150k”的范式。不错的基线，但 MMMU 大约在 35% 见顶。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）做了大量穷尽式消融。结论既出人意料又实用。

## 核心概念

### 五轴设计空间

Idefics2（Laurençon 等，2024）将设计空间归纳为五个轴：

1. **图像编码器。** CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。不同编码器的 patch 大小、分辨率与预训练目标各异。
2. **连接器。** MLP（2-4 层）、Q-Former（32 个查询 + 交叉注意力）、Perceiver Resampler（64 个查询）、C-Abstractor（卷积 + 双线性池化）。
3. **语言模型。** Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 尺寸是主要参数成本。
4. **训练数据。** 图像-文本对（CC3M、LAION）、交错数据（OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. **分辨率调度。** 固定 224/336/448、AnyRes、原生动态。训练期间逐步提升或保持不变。

每一款生产级 VLM 都要在这五个轴上做出选择。MMMU 分数的大部分差异可由轴 1、4、5 解释——与你选了哪种连接器关系不大。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节表明：把编码器从 CLIP ViT-L/14 换成 SigLIP SO400m/14，MMMU 提升 3 分以上；把连接器从 MLP 换成 Perceiver Resampler，提升不到 1 分。Idefics2 复现了同样结论：SigLIP > CLIP，Q-Former ≈ MLP ≈ Perceiver（相同 token 数下）。

Cambrian-1 的“Cambrian Vision Encoders Match-Up”（Tong 等，2024）在视觉中心型基准 CV-Bench 上测试了 20 余种编码器。榜首是 DINOv2 与 SigLIP 的混合；CLIP 居中；ImageBind 和 ViT-MAE 偏低。从 CLIP ViT-L 到 DINOv2 ViT-g/14，CV-Bench 差距约 5-7 分。

2026 年开源 VLM 的默认编码器是 SigLIP 2 SO400m/14，用于语义 + 密集特征；若需分割/定位，可与 DINOv2 ViT-g/14 特征拼接（Cambrian 的“Spatial Vision Aggregator”即采用此做法）。

### 轴 2：连接器设计影响不大

MM1、Idefics2、Prismatic、MM-Interleaved 得出一致结论：在固定视觉 token 数下，连接器架构几乎不影响结果。对 patch 做平均池化后接 2 层 MLP，与 32 查询的 Q-Former 在相同 token 预算下差距不到 1 分。

真正重要的是 token 数。更多视觉 token = 更多 LLM 计算 = 更好表现，到一定程度后边际递减。64 token/图 对 OCR 太少；576-1024 token 是大多数开源 VLM 的甜点；2048+ 只对文档和图表有帮助。

Q-Former 与 MLP 之争是成本问题，不是质量问题：Q-Former 把 token 固定在 32-64，不受图像分辨率影响；MLP 会输出全部 patch token。高分辨率输入时，Q-Former 节省 LLM 上下文；低分辨率时，差异只是噪声。

### 轴 3：LLM 尺寸决定上限

把 LLM 从 7B 翻倍到 13B，几乎每篇 VLM 论文的 MMMU 都能稳定提升 2-4 分。到 70B 时大多数基准趋于饱和。VLM 的多模态推理天花板就是 LLM 的文本推理天花板——视觉编码器只能喂料，不能替它推理。

这就是 Qwen2.5-VL-72B 和 Claude Opus 4.7 能在 MMMU-Pro 与 ScreenSpot-Pro 上碾压的原因：语言大脑足够大。7B VLM 无法通过精巧的连接器设计替代 70B VLM。

### 轴 4：数据——详细人工描述击败蒸馏

Molmo + PixMo（Deitke 等，2024）是 2024 年每个人都该读的一篇结果。艾伦人工智能研究所让标注员用 1-3 分钟对图像进行密集语音转写，得到 71.2 万张 densely-captioned 图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11/11 个基准上击败 Llama-3.2-90B-Vision。差距不在架构，而在描述质量。详细的人工描述每张图包含的信息量是短网页描述的 5-10 倍，且在 GPT-4V 蒸馏容易幻觉的地方保持事实 grounded。

ShareGPT4V（Chen 等，2023）和 Cauldron（Idefics2）也遵循了“人工 + GPT-4V 混合描述”的 playbook。趋势很清楚：在 2026 年的前沿，描述密度 > 描述数量 > 蒸馏便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融：384 → 448 提升 1-2 分；448 → 980 配合图像拆分（AnyRes）在 OCR 基准上再提升 3-5 分。固定分辨率训练在中等精度处见顶；分辨率 ramping（从 224 开始，到 448 或原生结束）训练更快、最终更高。

Cambrian-1 做了分辨率与 token 的权衡：固定算力下，可以选择低分辨率多 token，或高分辨率少 token。高分辨率在 OCR 上胜出；低分辨率多 token 在通用场景理解上胜出。

2026 年生产配方：Stage 1 固定 384，Stage 2 动态分辨率，最高 1280，面向 OCR 重任务。

### Prismatic 受控比较

Prismatic VLMs（Karamcheti 等，2024）是唯一同时控制所有轴的论文。相同的 13B LLM、相同的指令数据、相同的评测——每次只变一个轴。结果：

- 每图视觉 token 数解释约 60% 的方差。
- 编码器选择解释约 20%。
- 连接器架构解释约 5%。
- 其余（数据混合、调度器、学习率）解释约 15%。

这是一个粗略分解，但已是文献中对“我该先消融哪个轴”最干净的回答。

### 2026 年配方选择器

综合以上证据，2026 年新项目的默认开源 VLM 配方：

- **编码器：** SigLIP 2 SO400m/14，原生分辨率 + NaFlex；若需分割/定位，拼接 DINOv2 ViT-g/14 密集特征。
- **连接器：** 2 层 MLP 作用于 patch token。除非 token 受限，否则跳过 Q-Former。
- **LLM：** Qwen2.5 / Llama-3.1 / Gemma 2，7B 控制成本，70B 追求质量，按目标延迟选择。
- **数据：** PixMo + ShareGPT4V + Cauldron，再补充任务特定指令数据。
- **分辨率：** 动态（长边最小 256，最大 1280 像素）。
- **调度：** Stage 1 对齐（仅 projector），Stage 2 全模型微调，Stage 3 任务特定微调。

上述每一项默认值都可追溯到本节课末尾引用的论文中的实测消融。

## 动手使用

`code/main.py` 是一个消融表解析器与配方选择器。它编码了（精简后的）MM1 与 Idefics2 消融表，支持查询：

- “给定预算 X 与任务 Y，哪种配方胜出？”
- “在 7B Llama 上把 SigLIP 换成 CLIP，MMMU 预期差多少？”
- “想要 80% 置信度的答案，应该先消融哪个轴？”

输出是按排名的配方列表，附带预期基准差值和“先消融哪个轴”的建议。

## 交付成果

本节课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、算力预算和延迟目标，它会输出一份完整配方（编码器、连接器、LLM、数据混合、分辨率调度），并为每项选择附上支撑文献。避免工程师每次启动新 VLM 项目都重新发明 Idefics2 消融表。

## 练习题

1. 阅读 MM1 第 3.2 节。在固定 2B LLM、预算 5000 万张图时，哪种编码器胜出？如果换成 13B LLM，答案会反转吗？为什么？

2. Cambrian-1 发现，在视觉中心型基准上 DINOv2 + SigLIP 拼接优于任一单独使用，但在 MMMU 上没有增益。预测哪些基准会提升、哪些会持平。

3. 目标是在 2B LLM 上做一个移动端 UI 智能体。选择编码器、连接器、分辨率和数据混合，并用具体消融表为每项选择辩护。

4. Molmo 发布 4B 与 72B 模型。4B 可与闭源 7B VLM 竞争；72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。这对“LLM 尺寸瓶颈假说”说明什么？

5. 设计一张消融表，在 7B VLM 上隔离“数据混合质量”与“编码器质量”。最少需要几次训练运行？提出四个轴的设置。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 消融（ablation） | “拧一个旋钮” | 多次训练，只在某一个设计空间轴上不同，其余全部固定 |
| 连接器（connector） | “桥梁” / “投影器” | 将视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| 详细人工描述（detailed human caption） | “密集描述” | 多句人工撰写描述（通常 80-300 token），比网页 alt 文本更丰富 |
| 蒸馏（distillation） | “GPT-4V 描述” | 用更强的专有 VLM 生成的训练数据；方便但容易继承幻觉 |
| AnyRes / 动态分辨率 | “高分辨率路径” | 通过切片或 M-RoPE 喂入大于编码器原生分辨率的图像的策略 |
| 分辨率 ramp（resolution ramp） | “课程学习” | 从低分辨率开始、逐步提升的训练调度，加速对齐学习 |
| 视觉中心型基准（vision-centric bench） | “CV-Bench / BLINK” | 强调细粒度视觉感知而非偏重语言推理的评测 |
| PixMo | “Molmo 的数据” | 艾伦人工智能研究所 71.2 万张 densely-captioned 图像数据集；人工语音转写为密集描述 |

## 延伸阅读

- [McKinzie 等 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等 — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等 — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
