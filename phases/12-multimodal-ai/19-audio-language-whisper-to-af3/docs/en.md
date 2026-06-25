# 音频-语言模型：从 Whisper 到 Audio Flamingo 3 的演进

> Whisper（Radford 等，2022 年 12 月）解决了语音识别问题——68 万小时弱监督多语言语音、一个简单的编码器-解码器 Transformer（encoder-decoder transformer）、以及一个让后续所有 ASR 版本都引用它的基准测试。但识别不等于推理。当被问到“这段录音里有哪些乐器”或“说话者表达了什么情绪”或“第 3 分钟发生了什么”时，需要的是音频理解，而不仅是转录。Qwen-Audio、SALMONN、LTU 以及 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步搭建起这一技术栈：保留 Whisper 级别的编码器，接入 Q-former，在音频-文本指令数据上训练，并引入思维链（chain-of-thought）推理。本课将沿着这条演进脉络展开。

**Type:** 动手构建
**Languages:** Python（标准库，对数梅尔频谱图 + 音频 Q-former 骨架）
**Prerequisites:** Phase 6（Speech and Audio）、Phase 12 · 03（Q-Former）
**Time:** 约 180 分钟

## 学习目标

- 从波形计算对数梅尔频谱图（log-Mel spectrogram）：加窗、FFT、滤波器组、对数变换。
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合编码器，以及各自适用的场景。
- 构建一个音频 Q-former：N 个可学习的查询向量（learnable queries）对频谱图块做交叉注意力（cross-attention）。
- 解释级联（Whisper 后接大语言模型）与端到端音频-大语言模型训练的区别：为什么端到端在推理任务上更具扩展性。

## 问题背景

Whisper 解决了语音识别。音频的“OCR”已成为商品能力。但“商品化”止步于转录。如果模型无法对所听到的内容进行推理——时间、说话人、情绪、音乐结构、环境音——那么仅靠转录无法驱动产品功能。

有三条显而易见的路线：

1. 级联（Cascade）：Whisper 先把音频转成文本，再由大语言模型（LLM）对文本进行推理。在纯语音场景下表现良好。对音乐、环境音、多人重叠、情绪等任务失效。
2. 端到端音频-大语言模型（End-to-end audio-LLM）：音频编码器直接把音频词元（audio tokens）输入 LLM，跳过转录。保留声学信息（情绪、说话人、环境）。需要新的训练数据。
3. 混合（Hybrid）：音频编码器 + 既能转录又能推理的文本解码器。Qwen-Audio 与 Audio Flamingo 选择了这条路线。

## 核心概念

### 对数梅尔频谱图：输入特征

每个音频编码器都从同一个特征开始：对数梅尔频谱图。

1. 重采样到 16 kHz。
2. 使用 25 ms 窗口、10 ms 帧移做短时傅里叶变换（STFT）。
3. 取 FFT 结果的幅度。
4. 应用梅尔滤波器组（通常 80 个，在 0–8000 Hz 之间按对数间隔分布），映射到感知频率。
5. 做对数压缩（`log(1 + x)`）以适配动态范围。

结果：一个形状为 (T, 80) 的二维数组，T 为时间帧数。一段 30 秒、帧率 100 Hz 的音频对应 (3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层类 ViT 的 Transformer，将对数梅尔频谱图作为时间帧序列处理。输出：每个时间帧一个隐藏状态向量（hidden-state vector）。

对于自动语音识别（ASR），Whisper 的解码器是一个基于交叉注意力的 Transformer，以编码器输出为条件生成文本词元。标准的编码器-解码器结构。

对于音频-大语言模型（ALM），你需要把编码器输出作为另一个 LLM 的输入。常见模式：Whisper 编码器冻结、Q-former 可训练、LLM 冻结或微调。

### BEATs 与面向音频的编码器

Whisper 主要在语音数据上训练，因此在音乐和环境音方面较弱。

BEATs（Chen 等，2022）是在 AudioSet 上训练的自监督（self-supervised）Transformer。在相同参数量（parameter count）下，它比 Whisper 更擅长捕捉音乐和环境音。

AF-Whisper（Audio Flamingo 3 的混合编码器）：把 Whisper 与 BEATs 的特征拼接起来作为音频输入。Whisper 承载语言信号，BEATs 承载声学信号。

### 音频 Q-former

与 BLIP-2 的视觉 Q-former 模式相同。固定数量的可学习查询向量（通常 32 或 64 个）对音频编码器输出的时间帧做交叉注意力。这些查询向量成为 LLM 消费的音频词元。

训练分为两个阶段：
- 对齐阶段（alignment stage）：只训练 Q-former，使用对比损失 + 字幕生成损失，在音频-文本对（AudioCaps、Clotho）上训练。
- 指令阶段（instruction stage）：端到端训练，解冻 LLM，在指令数据上训练。

### 演进脉络——SALMONN、Qwen-Audio、AF3

SALMONN（Tang 等，2023）：Whisper + BEATs + Q-former + LLaMA。首个具备真正推理能力的开源音频-大语言模型。在 MMAU 基准上的综合得分约 0.55。

Qwen-Audio（Chu 等，2023）：架构类似，但在更丰富的数据集上训练，并针对多轮对话做了优化。MMAU 约 0.60。

LTU — Listen, Think, Understand（Gong 等，2023）：使用显式推理数据，专注于在音频片段上做思维链推理。规模更小但方向更聚焦。

Audio Flamingo 3（Goel 等，2025 年 7 月）：当前开源最优（open SOTA）。80 亿参数 LLM 骨干（Qwen2 7B）、Whisper-large 编码器拼接 BEATs、64 查询 Q-former，在 100 万以上音频-文本指令对上训练。MMAU 0.72，在某些子任务上已接近闭源前沿模型。

AF3 还引入了音频按需思维链（on-demand chain-of-thought）：模型可以选择性地先输出思考词元（例如“让我先识别乐器：……”），再给出最终答案。在复杂推理任务上，启用思考后准确率可提升 3–5 个百分点。

### 级联 vs 端到端

级联流水线：

1. Whisper 把音频转录成文本。
2. LLM 对文本进行推理。

对“总结这期播客”这类任务非常完美。但对以下任务失效：
- “这首歌的情绪是什么？”——情绪在声音里，不在文字中。
- “谁在说话，Alice 还是 Bob？”——需要说话人识别。
- “爆炸发生在第几秒？”——文本无法保留时间定位信息。
- “这是真实音频还是生成的？”——深度伪造检测需要声学特征。

端到端模型保留声学信号。Qwen-Audio 与 AF3 原生支持音乐、环境音和情绪理解。

### 2026 年生产落地建议

对于新的音频理解产品：

- 如果目标是转录、没有音乐、不需要情绪推理：选择级联方案。
- 如果需要音乐、情绪、多人说话或复杂音频推理：选择 AF3 / Qwen-Audio 系列。

级联更便宜、更简单。端到端能力更强。

### MMAU——音频推理基准

MMAU（Massive Multimodal Audio Understanding）是 2024–2025 年的音频推理基准：

- 1 万个跨语音、音乐、环境音的音频-文本问答对。
- 涵盖分类、时间推理、因果推理、开放式问答。
- 专门测试级联流水线系统性地遗漏的能力。

开源最优（AF3）为 0.72；闭源前沿约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。这一差距小于 VideoMME 上开源与闭源的差距，说明音频-大语言模型正在成熟。

## 动手实践

`code/main.py`：

- 用标准库实现对数梅尔频谱图计算：加窗、朴素 DFT、梅尔滤波器组。
- 音频 Q-former 骨架：给定编码器输出的时间帧，计算 Q、K、V、注意力，并输出 N 个词元。
- 在一个玩具任务上对比级联与端到端。

## 产出成果

本课将产出 `outputs/skill-audio-llm-pipeline-picker.md`。针对具体音频任务（转录、音乐标签、情绪推理、多人说话人分割、环境音分类），文档会选择级联、端到端 AF3 或混合方案。

## 练习题

1. 计算一段 30 秒、16 kHz、25 ms 窗口、10 ms 帧移、80 个 Mel 频带的对数梅尔频谱图维度。若采样率变为 48 kHz，维度如何变化？
2. 为什么 Whisper 在音乐任务上表现不佳？BEATs 捕捉了哪些 Whisper 无法捕捉的音频特征？
3. 音频 Q-former 使用 64 个查询 vs 32 个查询：在什么任务复杂度下 64 个查询更有优势？32 个查询能节省哪些计算？
4. 阅读 AF3 论文第 4 节关于按需思考（on-demand thinking）的内容。提出三个最能从思维链中受益的音频任务。
5. 使用 AF3 的输出实现一个最简化的说话人分割（diarization）流水线。你如何标记说话人切换？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| 对数梅尔频谱图（log-Mel spectrogram） | “Mel 特征” | 经过梅尔滤波器组后得到的二维（时间，频率）对数幅度数组 |
| 音频 Q-former（Audio Q-former） | “音频感知器” | 从音频编码器输出到固定长度查询向量的交叉注意力瓶颈，查询向量再输入 LLM |
| 级联（Cascaded） | “ASR 后接 LLM” | Whisper 先转录，文本 LLM 再推理；会丢失声学信息 |
| 端到端（End-to-end） | “音频-大语言模型” | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | “AudioSet 音频编码器” | 在 AudioSet 上训练的自监督 Transformer；在音乐和环境音上表现强 |
| MMAU | “音频推理基准” | 1 万个跨语音、音乐、环境的问答对；2024 年的评估标准 |
| 按需思考（On-demand thinking） | “音频 CoT” | 模型可选择性地在最终答案前输出推理词元，准确率提升 3–5 个百分点 |

## 延伸阅读

- [Radford 等 — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu 等 — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel 等 — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang 等 — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong 等 — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
