# 神经音频编解码器 — EnCodec、SNAC、Mimi、DAC 与语义-声学拆分

> 2026 年的音频生成几乎完全基于词元（token）。EnCodec、SNAC、Mimi 与 DAC 将连续波形转换为 Transformer 可以预测的离散序列。语义词元与声学词元的拆分——第一个码本作为语义信息，其余作为声学信息——是自 Transformer 问世以来音频领域最重要的架构变革。

**类型：** Learn
**语言：** Python
**前置条件：** Phase 6 · 02（Spectrograms），Phase 10 · 11（Quantization），Phase 5 · 19（Subword Tokenization）
**时间：** 约 60 分钟

## 问题背景

语言模型（language model）处理的是离散词元（token）。音频是连续的。如果你想为语音/音乐构建类似 LLM 的模型——例如 MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——你首先需要一个**神经音频编解码器（neural audio codec）**：一个学习得到的编码器，将音频离散化为一个小型词元表（vocabulary），以及一个配套的解码器，用于重建波形。

由此形成了两大阵营：

1. **重建优先编解码器**——EnCodec、DAC。它们优化感知音频质量，词元是“声学（acoustic）”的，会捕捉所有信息，包括说话人身份、音色、背景噪声。
2. **语义优先编解码器**——Mimi（Kyutai）、SpeechTokenizer。它们强制第一个码本编码语言/语音内容（通常通过从 WavLM 蒸馏实现），后续码本则承载声学细节。

2024–2026 年的关键洞察是：**纯重建型编解码器在从文本生成语音时会得到模糊的语音。** 基于编解码器词元的 LLM 必须在同一个码本中同时学习语言结构和声学结构，这无法规模化。将它们拆分——码本 0 为语义，码本 1-N 为声学——正是 Moshi 和 Sesame CSM 能够工作的原因。

## 核心概念

![四种编解码器概览：EnCodec、DAC、SNAC（多尺度）、Mimi（语义+声学）](../assets/codec-comparison.svg)

### 核心技巧：残差向量量化（Residual Vector Quantization, RVQ）

现代音频编解码器不再使用一个需要数百万个码字才能达到好质量的巨型码本，而是都采用**RVQ**：由多个小型码本级联而成。第一个码本对编码器输出进行量化；第二个码本对残差进行量化；以此类推。每个码本含 1024 个码字，8 个码本的有效词元表大小为 1024^8 = 10^24。

在推理时，解码器将每一帧选中的所有码字相加，完成重建。

### 2026 年值得关注的四种编解码器

**EnCodec（Meta，2022）。** 基线模型。基于波形的编码器-解码器，RVQ 瓶颈。24 kHz，最多 32 个码本，默认 4 个码本 @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架构。MusicGen 使用它。

**DAC（Descript，2023）。** 采用 L2 归一化码本、周期性激活函数与改进损失的 RVQ。在所有开源编解码器中重建保真度最高——使用 12 个码本时有时与原语音难以区分。44.1 kHz 全频带。

**SNAC（Hubert Siuzdak，2024）。** 多尺度 RVQ——粗粒度码本以低于细粒度码本的帧率（frame rate）运行。它实际上以层次化方式建模音频：约 12 Hz 的粗略“草图”加上 50 Hz 的细节。Orpheus-3B 使用它，因为层次结构与基于 LM 的生成非常契合。

**Mimi（Kyutai，2024）。** 2026 年的变革者。12.5 Hz 帧率（极低），8 个码本 @ 4.4 kbps。码本 0 **从 WavLM 蒸馏**得到——训练目标是预测 WavLM 的语音内容特征。码本 1-7 为声学残差。这种拆分驱动了 Moshi（第 15 课）和 Sesame CSM。

### 帧率对语言建模至关重要

更低的帧率 = 更短的序列 = 更快的 LM。

| 编解码器 | 帧率 | 1 秒 = N 帧 | 适用场景 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM 高效处理 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

在 12.5 Hz 下，一段 10 秒的语音只有 125 个编解码器帧——Transformer 可以轻松预测。

### 语义词元 vs 声学词元

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义词元（Mimi 中的码本 0）。** 编码“说了什么”——音素、词语、内容。通过辅助预测损失从 WavLM 蒸馏得到。
- **声学词元（码本 1-7）。** 编码音色、说话人身份、韵律、背景噪声与细节。

自回归语言模型（AR LM）首先预测语义词元（以文本为条件），然后预测声学词元（以语义词元 + 说话人参考为条件）。正是这种分解使得现代文本转语音（TTS）能够实现零样本（zero-shot）克隆声音：语义模型负责内容，声学模型负责音色。

### 2026 年重建质量（比特率，越低越好）

| 编解码器 | 比特率 | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

像 Opus 这样的传统编解码器在感知质量上仍然以每比特更优胜出。神经编解码器的优势在于**离散词元**（Opus 不产生）和**生成模型质量**（LM 能用这些词元做什么）。

## 动手实现

### 步骤 1：使用 EnCodec 编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes 形状: (1, n_codebooks, n_frames), dtype=int64
```

在 6 kbps 下，`n_codebooks=8`。每个码字的取值范围为 0-1023（10 位）。

### 步骤 2：解码并测量重建误差

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 步骤 3：语义-声学拆分（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # 形状: (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

码本 0 是语义码本，并与 WavLM 对齐。你可以训练一个文本到语义（text-to-semantic）Transformer——词汇量比直接生成音频小得多。然后，一个独立的声学到波形（acoustic-to-waveform）解码器以说话人参考为条件生成波形。

### 步骤 4：为什么基于编解码器词元的 AR LM 有效

对于一段 10 秒的语音，使用 Mimi 的 12.5 Hz × 8 个码本：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 个词元对 Transformer 来说是微不足道的上下文。一个 256M 参数的 Transformer 在现代 GPU 上可以在数毫秒内生成 10 秒语音。

## 如何使用

按任务选择编解码器：

| 任务 | 编解码器 |
|------|-------|
| 通用音乐生成 | EnCodec-24k |
| 最高保真重建 | DAC-44.1k |
| 基于语音的 AR LM（TTS） | SNAC 或 Mimi |
| 流式全双工语音 | Mimi（12.5 Hz） |
| 带文本描述的音效库 | EnCodec + T5 条件 |
| 细粒度音频编辑 | DAC + 修复 |

经验法则：**如果你正在构建生成模型，先从 Mimi 或 SNAC 开始。如果你正在构建压缩流程，请使用 Opus。**

## 常见陷阱

- **码本过多。** 增加码本会线性提升保真度，但也会线性增加 LM 的序列长度。建议控制在 8–12 个。
- **帧率不匹配。** 在 12.5 Hz 的 Mimi 上训练 LM，然后在 50 Hz 的 EnCodec 上微调，会静默失败。
- **假设所有码本同等重要。** 在 Mimi 中，码本 0 承载内容；丢失它会破坏可懂度。而丢失码本 7 几乎听不出差别。
- **仅以重建质量作为唯一指标。** 一个编解码器重建效果再好，如果语义结构差，也可能对基于 LM 的生成毫无用处。

## 交付

保存为 `outputs/skill-codec-picker.md`。为给定的生成或压缩任务选择一种编解码器。

## 练习

1. **简单。** 运行 `code/main.py`。它实现了一个玩具标量 + 残差量化器，并在增加码本时测量重建误差。
2. **中等。** 安装 `encodec`，在留出的一段语音上比较 1、4、8、32 个码本。绘制 PESQ 或 MSE 随比特率变化的曲线。
3. **困难。** 加载 Mimi。编码一段音频。将码本 0 替换为随机整数后解码；再对码本 7 做同样操作。比较两种破坏效果——破坏码本 0 应使语音无法听懂，破坏码本 7 应几乎无变化。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| RVQ | 残差量化 | 多个小型码本级联；每个码本量化前一个码本的残差。 |
| 帧率 | 编解码器速度 | 每秒有多少个词元帧。越低 = LM 越快。 |
| 语义码本 | 码本 0（Mimi） | 从 SSL 特征蒸馏得到的码本；编码内容。 |
| 声学码本 | 其余码本 | 音色、韵律、噪声与细节。 |
| PESQ / ViSQOL | 感知质量 | 与 MOS 相关的客观指标。 |
| EnCodec | Meta 编解码器 | RVQ 基线；MusicGen 使用。 |
| Mimi | Kyutai 编解码器 | 12.5 Hz 帧率；语义-声学拆分；驱动 Moshi。 |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — the RVQ baseline.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — highest-fidelity open.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — multi-scale RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — semantic-acoustic split, WavLM distillation.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — the two-stage semantic/acoustic paradigm.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — the original streamable RVQ codec.
