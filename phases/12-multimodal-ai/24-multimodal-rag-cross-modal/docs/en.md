# 多模态 RAG 与跨模态检索

> 原生视觉文档的 RAG 只是其中一隅。生产级多模态 RAG 的视野更广——跨文本、图像、音频和视频进行检索，用于旅行规划（"给我找一家安静、自然光充足的纯素早午餐店"）、医疗分诊（"哪些伤情与这张照片加这些记录匹配"）、电子商务（"找和我的自拍照类似且合身的穿搭"）以及现场服务（"根据这个发动机声音加上零件照片进行诊断"）等工作流。2025 年的三篇综述——Abootorabi 等、Mei 等、Zhao 等——将这些子问题系统化：跨模态检索（cross-modal retrieval）、检索融合（retrieval fusion）、生成 grounding、多模态评估。本节课阅读这些综述并设计一个生产级流水线。

**类型：** Build
**语言：** Python（标准库，跨模态检索器 + 融合 + grounded 生成器）
**先修：** Phase 12 · 23（ColPali），Phase 11（RAG 基础）
**时间：** 约 180 分钟

## 学习目标

- 设计跨模态检索：文本 → 图像、图像 → 文本、音频 → 视频等。
- 比较三种融合策略：分数融合（score fusion）、基于注意力的融合（attention-based fusion）、MoE 融合（MoE fusion）。
- 解释生成 grounding：当来源混合多种模态时，"引用来源"意味着什么。
- 说出 2025 年三篇经典多模态 RAG 综述及其子问题分类。

## 问题

单模态 RAG 是一个已解决的范式：嵌入查询、嵌入文本块、检索、塞进大语言模型（LLM）。多模态 RAG 需要：

1. 多个检索头（每种模态都需要在兼容空间中的嵌入）。
2. 跨模态检索结果的融合。
3. 能跨模态引用来源的生成 grounding。
4. 覆盖跨模态信号的评估指标。

2025 年的综述都指向了相同的分类法。

## 概念

### 跨模态检索

给定模态 A 的查询，检索模态 B 的文档。三种模式：

1. 共享嵌入空间（shared embedding space）。CLIP 和 CLAP 在共享空间中生成文本 + 图像 / 文本 + 音频的嵌入。跨模态的余弦相似度（cosine similarity）可直接使用。仅限于 CLIP 训练过的配对。

2. 每模态编码器 + 转换器。文本编码器 + 图像编码器 + 一个小的转换器模块，在模态空间之间映射。Gupta 等的 Sen2Sen 以及 2024 年的其他设计。灵活但增加了复杂度。

3. 视觉语言模型（VLM）作为编码器。使用 VLM 的隐藏状态作为检索表示。VLM 支持的任何模态都可用。质量更高，成本更高。

选择：文本 + 图像用 CLIP / SigLIP 2；文本 + 音频用 CLAP；追求前沿质量时用 VLM 隐藏状态做跨模态。

### 融合策略

你检索到 10 个结果：5 张图像、3 段文本、2 个音频片段。如何合并？

分数融合（score fusion，最廉价）。每种模态有自己的检索器，各自返回分数。在模态内归一化分数后求和。简单，通常有效。

基于注意力的融合（attention-based fusion）。将所有检索项拼接，让一个小型注意力网络为其赋权。需要训练。

MoE 融合（MoE fusion）。门控网络（gating network）路由到模态专属专家。不同类型的查询路由不同——视觉问题会给图像更高权重。

生产默认：分数融合，并对查询的主导模态稍加偏置。若 A/B 测试在你的领域显示明显收益，再升级到 MoE。

### 生成 grounding

LLM 应引用是哪一个检索项支撑了每个断言。对于多模态：

- 文本来源：标准引用 `[1]`。
- 图像来源：`[img 3]` 加简短标题。
- 音频：`[audio 2 at 0:34]`。

用带有 grounding 意识的数据训练生成器：训练目标中的每个断言都标注来源索引。推理时，模型会自然输出引用。

### 2025 年综述

Abootorabi 等（arXiv:2502.08826，"Ask in Any Modality"）：多模态 RAG 的分类法。涵盖检索、融合、生成。覆盖最广。

Mei 等（arXiv:2504.08748，"A Survey of Multimodal RAG"）：聚焦子任务基准和失效模式。对评估设计很有用。

Zhao 等（arXiv:2503.18016）：聚焦视觉的综述。在 ColPali 系列工作上很强。

阅读三篇，你就掌握了 2025 年春季的最新进展。大多数子问题仍然开放。

### MuRAG——奠基论文

MuRAG（Chen 等，2022）是第一项多模态 RAG 工作。从多模态知识库中检索图像 + 文本并生成答案。在 VLM 浪潮之前展示了可行性。现代系统（REACT、VisRAG、M3DocRAG）都建立在其基础上。

### 生产级旅行规划示例

查询："find me a quiet vegan brunch with natural light."

流水线：

1. 分解查询。"quiet" → 音频/评论关键词；"vegan brunch" → 菜单项；"natural light" → 图像特征。
2. 按模态检索：
   - 在评论上做文本检索："vegan brunch, quiet ambiance."
   - 在餐厅照片上做图像检索："natural light, airy."
   - 在环境声音片段上做音频检索："low decibel, no music."
3. 融合分数。每家餐厅得到一个综合分数。
4. Top-k 餐厅 → VLM 生成器，输入所有证据 → 带引用的答案。

这远远超出了文本 RAG 的范畴。每种模态都增加了文本 alone 无法捕捉的信号。

### 智能体多模态 RAG

多跳（multi-hop）：如果首轮检索没有返回高置信度答案，LLM 会重新改写并再次检索。Phase 14 中的智能体 RAG 模式在此适用。示例：

- 检索初始 top-10 → LLM 说"太吵了，过滤 <40 dB" → 重新检索。
- 检索图像 → LLM 发现其中一张有菜单 → 检索菜单文本 → 回答。

这增加了复杂度，但能处理单次检索无法应对的查询。

### 评估

跨模态评估仍不成熟。常见代理指标：

- 每模态的 Recall@k。
- 融合后的 top-k 准确率。
- 人工评判的端到端满意度。
- 任务特定指标（完成的预订、产生的购买）。

没有标准基准覆盖所有模态。大多数论文在领域特定任务上评估。

## 应用

`code/main.py`：

- 三个模拟检索器（文本、图像、音频），在共享的餐厅语料库上运行。
- 分数融合，用可配置权重组合各模态分数。
- 一个生成器存根，输出带引用的最终答案。
- 一个简单的智能体循环，当置信度低时改写查询。

## 交付

本节课产出 `outputs/skill-multimodal-rag-designer.md`。给定一份带多模态查询流的产品需求，设计检索器、融合、生成器和评估。

## 练习

1. 设计一个医疗分诊多模态 RAG：查询 = 受伤照片 + 文本症状。各模态从哪些知识库检索？

2. 分数融合是简单的加权和。它有什么失效模式是 MoE 融合能避免的？

3. 阅读 Abootorabi 等的分类法（第 3 节）。三个经典子问题是什么？它们如何映射到你选择的产品？

4. 为旅行规划多模态 RAG 设计评估规范。哪些指标覆盖图像召回、音频召回和综合正确性？

5. 智能体多跳 RAG 每轮往返都有延迟成本。在什么查询难度下，准确率提升才值得付出延迟？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 跨模态检索（cross-modal retrieval） | "Query one modality, retrieve another" | 文本查询检索图像；图像查询检索文本；需要共享空间或转换器 |
| 分数融合（score fusion） | "Combine scores" | 各模态检索分数的加权和；最简单的融合 |
| MoE 融合（MoE fusion） | "Modality-routed experts" | 门控网络决定每个查询更信任哪种模态的分数 |
| 有根据的生成（grounded generation） | "Cite your sources" | 答案中的每个断言都标注来源索引 |
| MuRAG | "First multimodal RAG" | 2022 年的论文，确立了多模态 RAG 范式 |
| 智能体多跳（agentic multi-hop） | "Reformulate and retry" | 当首轮检索置信度较低时，LLM 重新查询检索器 |

## 延伸阅读

- [Abootorabi 等 — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei 等 — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao 等 — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen 等 — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu 等 — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
