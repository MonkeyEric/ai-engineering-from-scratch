# 文档与图表理解

> 文档不是普通照片。PDF、科学论文、发票或手写表格都包含布局、表格、图表、脚注、页眉以及语义结构，这些都无法被单纯的图像理解所捕获。在视觉语言模型（VLM）出现之前，技术栈是一条流水线：Tesseract OCR + LayoutLMv3 + 表格抽取启发式规则。VLM 浪潮则用无需 OCR 的模型取而代之——Donut（2022）、Nougat（2023）、DocLLM（2023）——它们直接输出结构化标记语言（markup）。到 2026 年，最前沿的做法简单到“把页面原图以 2576px 原生分辨率喂给 Claude Opus 4.7”，结构化标记输出就自然得到了。本课回顾文档人工智能（document AI）的三个时代。

**类型：** 构建
**语言：** Python（标准库、布局感知文档解析器骨架）
**先修：** 第 12 阶段 · 05（LLaVA），第 5 阶段（NLP）
**时间：** 约 180 分钟

## 学习目标

- 解释文档 AI 的三个时代：OCR 流水线（OCR pipeline）、无需 OCR（OCR-free）与原生 VLM（VLM-native）。
- 描述 LayoutLMv3 的三路输入：文本、布局（layout，即边界框 bbox）与图像块（image patches），以及统一的掩码训练（unified masking）。
- 比较 Donut（无需 OCR，image → markup）、Nougat（科学论文 → LaTeX）、DocLLM（布局感知生成式模型）与 PaliGemma 2（原生 VLM）。
- 为新的文档任务（发票、科学论文、手写表格、中文小票）选择合适的文档模型。

## 问题背景

“理解这份 PDF”看似简单，实则困难。信息分布在：

- 文本内容（90% 的信息量）。
- 布局（layout，如页眉、脚注、边栏、双栏排版）。
- 表格（行、列、合并单元格）。
- 图表与示意图。
- 手写批注。
- 字体与排版（标题 vs 正文）。

原始 OCR 只会倾倒出文本并丢失其余信息。一个真正关心发票的系统需要知道 “Total: $1,245” 来自右下角，而非脚注。

## 核心概念

### 时代 1 —— OCR 流水线（OCR pipeline，2021 年前）

经典技术栈：

1. PDF → 每页一张图像。
2. Tesseract（或商业 OCR）提取文本，并给出每个词的边界框（bounding box）。
3. 布局分析器识别区块（页眉、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则表达式抽取字段。

这对清晰印刷文本有效。但在手写体、倾斜扫描、复杂表格、非英文文字上会失效。每种失败模式都需要一条自定义异常路径。

### TrOCR（2021）

TrOCR（Li 等，arXiv:2109.10282）用基于 Transformer 的编码器-解码器（encoder-decoder）取代了 Tesseract 经典的 CNN-CTC，并在合成与真实文本图像上训练。它在手写与多语言文本上取得显著优势。仍然是流水线（检测器 → TrOCR → 布局），但 OCR 这一步大幅提升。

### 时代 2 —— 无需 OCR（OCR-free，2022–2023）

最早的无需 OCR（OCR-free）模型提出：完全跳过检测，直接把图像像素映射到结构化输出。

Donut（Kim 等，arXiv:2111.15664）：

- 编码器-解码器（encoder-decoder）Transformer，编码器为 Swin-B。
- 输出为 JSON（用于表单理解，form understanding）、Markdown（用于摘要）或任何任务特定模式。
- 无需 OCR、无需布局、无需检测。

Nougat（Blecher 等，arXiv:2308.13418）：

- 专门在科学论文上训练。
- 输出为 LaTeX / Markdown。
- 可处理公式、多栏布局、图表。
- 如今每个 arXiv 解析器都会调用的模型。

这些都是专家模型，而非通才模型。Donut 处理科学论文会失败；Nougat 处理发票会失败。

### LayoutLMv3（2022）

这是另一条路线。LayoutLMv3（Huang 等，arXiv:2204.08387）保留 OCR，但增强了对布局的理解：

- 三路输入：OCR 文本词元（tokens）、每个词元的二维边界框（2D bounding boxes）、图像块（image patches）。
- 跨三种模态的掩码训练目标（masked text、masked patches、masked layout）。
- 下游任务：分类、实体抽取（entity extraction）、表格问答（table QA）。

LayoutLMv3 是基于 OCR 的文档理解的巅峰。在表单与发票上表现强劲。上游仍需要 OCR。在视觉语言模型出现之前，它在标准化文档基准上取得了最佳准确率。

### DocLLM（2023）

DocLLM（Wang 等，arXiv:2401.00908）是 LayoutLM 的生成式（generative）兄弟模型。它基于布局词元（layout tokens）生成自由形式答案。更擅长文档问答；但仍依赖 OCR 输入。

### 时代 3 —— 原生 VLM（VLM-native，2024 年后）

2024 年，视觉语言模型（VLM）已足够强大，可以完全取代流水线。将高分辨率整页图像喂给 VLM，直接提问，即可获得答案。

- LLaVA-NeXT 336-tile AnyRes 适用于小尺寸文档。
- Qwen2.5-VL 的动态分辨率可原生处理 2048 像素以上。
- Claude Opus 4.7 支持 2576px 的文档。
- PaliGemma 2（2025 年 4 月）专门针对文档与手写训练。

原生 VLM（VLM-native）与 OCR 流水线之间的差距迅速缩小。到 2026 年，原生 VLM 在以下方面占优：

- 场景文字（手写 + 印刷、混合文字）。
- 含合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带文字标注的图表。

OCR 流水线仍在以下方面占优：

- 超大规模纯扫描任务，且每页延迟（latency）至关重要。
- 流水线可靠性（确定性失败 vs VLM 幻觉，hallucinations）。
- 需要可审计 OCR 输出的监管环境。

### Claude 4.7 / GPT-5 的前沿水平

在 2576 像素原生输入下，前沿视觉语言模型（VLM）能以接近人类的准确率完成文档理解。2026 年初的基准数据如下：

- DocVQA：Claude 4.7 ~95.1、PaliGemma 2 ~88.4、Nougat ~77.3、流水线版 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2、GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

闭源模型的领先主要来自分辨率与基座大语言模型（LLM）的规模。70 亿参数的开源模型落后几分，但正在追赶。

### 数学公式与 LaTeX 输出

科学论文需要精确的 LaTeX 公式输出。Nougat 正是为此训练。以 LaTeX 为训练目标的 VLM（如 Qwen2.5-VL-Math、Nougat 衍生模型）能生成可用的 LaTeX。若无明确的 LaTeX 训练，VLM 只能生成可读但不精确的转录。

2026 年的科学论文流水线做法是：先用 Nougat 处理 PDF，再用 VLM 处理疑难页面。

### 手写文字

这仍然是最困难的子任务。印刷体与手写体混合（如医生笔记、填写表单）的场景，OCR 流水线在成本上仍优于 VLM。纯手写 VLM 正在进步（Claude 4.7、PaliGemma 2）。

### 2026 年选型指南

对于新的文档 AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + 规则，成本高效。
- 混合文档（科学论文 + 手写 + 表单）：原生 VLM（VLM-native，如 PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 收录：Nougat 处理数学，VLM 处理图表。
- 监管场景：OCR 流水线 + VLM 作为交叉校验器。

## 动手实践

`code/main.py`：

- 一个玩具级布局感知分词器：给定 (text, bbox) 对，生成 LayoutLMv3 风格的输入。
- 一个 Donut 风格的任务模式生成器：用于表单的 JSON 模板。
- 比较 OCR 流水线、Donut、Nougat 与原生 VLM（VLM-native）每页的 token 预算。

## 成果产出

本课将产出 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（领域、规模、质量、监管要求），在 OCR 流水线（OCR pipeline）、无需 OCR 的专用模型（OCR-free specialist）与原生 VLM（VLM-native）之间做出选择。

## 练习题

1. 你的项目每天处理 1000 万张发票。哪种技术栈能在不损失准确率的前提下最小化每页成本？
2. 为什么 LayoutLMv3 在表单问答（form QA）上优于纯 CLIP 视觉语言模型，但在场景文字上却表现较差？边界框（bbox）输入流牺牲了什么？
3. Nougat 生成 LaTeX。请设计一个测试用例，说明原生 VLM（VLM-native）在 LaTeX 保真度上胜过 Nougat，再设计一个 Nougat 胜出的用例。
4. 阅读 PaliGemma 2 论文（Google，2024）。相较于 PaliGemma 1，是哪些关键训练数据的加入提升了文档准确率？
5. 设计一种监管安全的混合方案：OCR 流水线为主，VLM 为二次交叉校验。当二者不一致时如何解决？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OCR pipeline | “Tesseract 风格” | 分阶段栈：检测 → OCR → 布局 → 规则；确定性强，但脆弱 |
| OCR-free | “Donut 风格” | 跳过显式 OCR 的 image-to-output Transformer；单模型 |
| Layout-aware | “LayoutLM” | 输入包含每个词元的 bbox 坐标；跨模态统一掩码 |
| VLM-native | “前沿 VLM” | 直接把页面图像以高分辨率喂给 Claude/GPT/Qwen 等 VLM；无需流水线 |
| DocVQA | “文档基准” | 文档视觉问答（Document VQA）标准；最常被引用的分数 |
| Markup output | “LaTeX / MD” | 结构化输出格式，而非自由文本；便于下游自动化 |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
