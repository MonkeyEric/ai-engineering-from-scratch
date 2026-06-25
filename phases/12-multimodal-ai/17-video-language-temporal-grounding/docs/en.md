# 视频-语言模型：时间 Token 与时序定位

> 视频不是照片的堆叠。一段 5 秒的片段包含因果顺序、动作动词和事件时序，这是图像模型无法表达的。Video-LLaMA（Zhang 等，2023 年 6 月）推出了首个具备音视频联合定位（audio-visual grounding）能力的开源视频大语言模型（video-LLM）。VideoChat 与 Video-LLaVA 扩展了这一范式。到 2025 年，Qwen2.5-VL 的 TMRoPE 已拉近与前沿闭源模型的差距。各系统以不同方式处理时间 token（temporal tokens）——每片段 Q-former（Q-former-per-clip）、每帧拼接池化（concat-pool per frame）、每 token TMRoPE（TMRoPE per token）。本节课解读这些范式，构建均匀采样与动态 FPS 采样的帧采样器，并在时序定位（temporal grounding）任务上评估。

**类型：** 构建  
**语言：** Python（标准库，帧采样器 + 时序定位评估器）  
**前置：** Phase 12 · 08（LLaVA-OneVision）  
**时间：** 约 180 分钟

## 学习目标

- 解释为什么时序位置编码（temporal positional encoding）能在视觉编码器（vision encoder）不变的情况下改变视频 VLM 的性能。
- 比较均匀采样（uniform sampling）、动态 FPS 采样（dynamic-FPS sampling）和事件驱动采样（event-driven sampling）在每秒 token 数（tokens-per-second）与定位准确率（grounding accuracy）上的权衡。
- 描述三种设计：每片段 Q-former（Video-LLaMA）、每帧池化（Video-LLaVA）和每 token M-RoPE（Qwen2.5-VL）。
- 说出四个视频基准测试（benchmarks）：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题背景

一段 1 分钟、30 FPS 的视频共有 1800 帧。若每帧产生 196 个视觉 token（ViT-B/224），总 token 数将达到 35.2 万，超过 2024 年任何大语言模型的上下文长度。

三种压缩策略应运而生：

1. 降低帧率采样（根据内容选择 1–8 FPS）。
2. 对每帧的图像块 token（patch tokens）进行激进池化（3×3 或 4×4 双线性池化）。
3. 通过 Q-former 压缩：输入 16 帧片段，输出 64 个 token。

三者的取舍各不相同：降采样损失时序细节，池化损失空间细节，Q-former 则各损失一点，但能显著节省 token。

另一条轴线是时序位置编码（temporal position encoding）：模型如何知道第 5 帧在第 6 帧之前？可选方案包括简单的 1D 时序 RoPE（Video-LLaMA）、可学习的时序嵌入（Video-LLaVA），以及 TMRoPE（Qwen2.5-VL，完整 3D）。

## 核心概念

### Video-LLaMA：每片段 Q-former + 音频分支

Video-LLaMA（2023）是首个开源视频大语言模型（video-LLM）。其架构如下：

- 以 2 FPS 截取 16 帧片段（即 8 秒）。
- 每帧 ViT 特征 -> 视频 Q-former，对所有 16 帧做交叉注意力（cross-attention）-> 32 个可学习查询（learned queries）-> 大语言模型。
- 并行的音频分支：波形 -> ImageBind 音频编码器 -> 音频 Q-former -> 32 个查询 -> 大语言模型。

优势：音视频联合推理（audio-visual joint reasoning）。劣势：片段长度固定，无法进行任意时间点的时序定位（time grounding）。

### VideoChat 与 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的思路，但去掉了音频并做了简化。Video-LLaVA（Lin 等，2023）在图像帧与视频帧上训练了同一个视觉编码器（“投影前先对齐”，Alignment Before Projection），得到统一的表征。二者都是冻结的 CLIP 编码器 + MLP + 大语言模型。

两者都无法处理长视频，都是 8–16 帧的系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 提出了 TMRoPE（Temporal-Modality Rotary Position Embedding，时序-模态旋转位置编码）。每个图像块 token（patch token）都携带 `(t, h, w)` 位置，其中 `t` 是真实时间戳，而非帧索引。

与简单时序嵌入（temporal embedding）的关键区别：

- 绝对时间，而非帧索引。模型看到的是“在 4.2 秒”，而不是“在第 15 帧”。
- 按 token 旋转，而非按片段。每个视觉 token 根据其时间戳独立旋转。
- 兼容动态 FPS。某处按 2 FPS 采样、另一处按 4 FPS 采样时，TMRoPE 原生处理不均匀间隔。

TMRoPE 支持“猫在第几秒起跳？”这类查询，模型可以输出“在 4.2 秒”。而 Video-LLaMA 只能说“在片段前半段”。

### 帧采样策略

**均匀采样（Uniform）：** 在视频时长内等距抽取 N 帧。简单，但会漏掉运动高峰。

**动态 FPS 采样（Dynamic FPS）：** 根据运动强度自适应采样。通过光流（optical flow）或帧差分（frame differencing）找出高运动片段并加密采样。Qwen2.5-VL 在训练时采用了这种方式。

**事件驱动采样（Event-driven）：** 运行轻量检测器，在动作发生处加密采样。VideoAgent 使用这种策略。

**关键帧 + 上下文：** 在镜头边界（shot boundaries）处采样，并补充相邻几帧。适用于电影级内容。

### 每帧池化

若按 1 FPS、每帧 576 token 计算，一段 5 分钟视频就是 172,800 token。Qwen2.5-VL-72B 的 128k 上下文可以塞下，但成本很高。

3×3 双线性池化可将每帧压缩到 64 token，5 分钟视频仅需 19,200 token，是大多数任务的甜蜜点。

在智能体工作流（agent workflows）等对空间细节（spatial detail）要求不高的场景，可进一步激进池化（6×6 -> 每帧 16 token）。

### 四个视频基准测试

- **VideoMME：** 综合视频理解，覆盖短、中、长视频。
- **TempCompass：** 细粒度时序推理，“之前/之后”类问题。
- **EgoSchema：** 长程第一人称视频（long-horizon first-person video）理解。
- **Video-MMMU：** 多模态多学科视频问答。

完整的视频 VLM 评估会覆盖全部四项。它们分别考察不同维度：TempCompass 侧重时序先后，EgoSchema 侧重 3 分钟以上长程推理，VideoMME 覆盖多种时长。

### 时序定位输出格式

时序定位（temporal grounding）的输出格式包括：

- **自由文本（Free text）：** "The cat jumps around the 4-second mark." 易于解析但不够精确。
- **结构化 JSON：** `{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 在训练中使用这种格式。
- **基于 token（Token-based）：** 在答案中插入特殊 token，如 `<time>4.1</time>`。这是 Qwen2.5-VL 的内部格式。

基于 token 的格式对下游使用（downstream use）最精确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026 年最佳实践

2026 年构建视频 VLM 的建议：

- **编码器（Encoder）：** SigLIP 2 配合 M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- **帧采样（Frame sampling）：** 动态 FPS（根据运动强度取 1–4 FPS），并设置最大帧数上限（max-frame cap）。
- **每帧池化（Per-frame pooling）：** 3×3 双线性池化。
- **输出：** 包含时间 + 事件字段的结构化 JSON。
- **基准测试（Benchmarks）：** 通用任务用 VideoMME + TempCompass；长程任务用 EgoSchema。

## 动手实现

`code/main.py` 包含：

- 均匀采样与动态 FPS 采样的帧采样器。
- 一个简易的时序定位评估器：给定时间 T 处的“真值”事件和模型输出，按容差评分。
- 三种方案的对比：Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）。

## 成果交付

本节课将产出 `outputs/skill-video-vlm-frame-planner.md`。针对具体视频任务（监控、动作识别、时序定位、视频摘要），它将选择合适的帧采样器、池化因子、输出格式以及预期精度等级（accuracy tier）。

## 练习题

1. 对于一段 3 分钟的烹饪演示，选择均匀采样还是动态 FPS？用 token 数量说明理由。
2. 相比简单的时序嵌入表（temporal embedding table），TMRoPE 具体增加了什么能力？
3. 为时序定位编写一个 VLM 可学习生成的 JSON 模式（JSON schema），并包含错误案例（error cases）。
4. 阅读 Video-LLaVA 第 3 节“Alignment Before Projection”。为什么这比分别训练图像和视频编码器更好？
5. 查看 VideoMME 排行榜（leaderboard），截至 2026 年，顶尖开源模型（open model）与顶尖闭源模型（proprietary model）之间差距多大？其中有多少可归因于时序编码，又有多少归因于基础大语言模型规模（base LLM scale）？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 时序定位（Temporal grounding） | “时间局部化答案” | VLM 输出事件发生时间戳范围的答案 |
| TMRoPE | “Time-Multimodal RoPE” | 使用绝对时间戳的 3D 旋转位置编码，应用于 Qwen2.5-VL |
| 动态 FPS（Dynamic FPS） | “运动感知采样” | 高运动片段加密采样，静态片段稀疏采样 |
| 帧池化（Frame pooling） | “每帧空间压缩” | 在进入大语言模型前，用双线性插值减少每帧的图像块数量 |
| 视频 Q-former（Video Q-former） | “片段压缩器” | 交叉注意力瓶颈，将 N 帧映射到 K 个可学习查询 |
| VideoMME | “视频基准测试” | 涵盖短/中/长视频的综合基准测试，包含 2500+ 样本 |

## 扩展阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
