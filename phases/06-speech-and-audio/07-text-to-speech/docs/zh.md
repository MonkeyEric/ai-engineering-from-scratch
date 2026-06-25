# 文本转语音（Text-to-Speech, TTS）——从 Tacotron 到 F5 与 Kokoro

> 自动语音识别（ASR）将语音转换为文本，而 TTS 将文本转换为语音。2026 年的技术栈分为三部分：文本 → 词元（tokens）、词元 → 梅尔谱（mel）、梅尔谱 → 波形（waveform）。每一部分都有一个可在笔记本电脑上运行的默认模型。

**类型：** 实战构建  
**语言：** Python  
**先修：** 第 6 阶段 · 02（频谱图与梅尔谱），第 5 阶段 · 09（Seq2Seq），第 7 阶段 · 05（完整 Transformer）  
**时长：** 约 75 分钟

## 问题

你有一段文本："Please remind me to water the plants at 6 pm."。你需要生成一段 3 秒长的自然音频：韵律（prosody）正确（停顿、重音），"plants" 的元音发音准确，并且能在 CPU 上 300 毫秒内完成，以支撑实时语音助手。此外，你还需要切换声音、处理语码混合（code-switched）输入（如 "remind me at 6 pm, daijoubu?"），并避免在人名上出糗。

现代 TTS 管线如下：

1. **文本前端。** 对文本进行归一化（日期、数字、邮箱），转换为音素（phonemes）或子词词元（subword tokens），并预测韵律特征。
2. **声学模型。** 文本 → 梅尔谱（mel spectrogram）。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器。** 梅尔谱 → 波形（waveform）。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022），以及 2024 年之后的神经编解码声码器。

到了 2026 年，随着端到端扩散模型（diffusion）和流匹配（flow-matching）模型的兴起，声学模型与声码器的边界变得模糊。但在调试时，“三部分”的心智模型依然有效。

## 核心概念

![Tacotron、FastSpeech、VITS、F5/Kokoro 横向对比](../assets/tts.svg)

**Tacotron 2（2017）。** 序列到序列（seq2seq）模型：字符嵌入（char-embedding）→ BiLSTM 编码器 → 位置敏感注意力（location-sensitive attention）→ 自回归（autoregressive）LSTM 解码器输出梅尔帧。速度较慢（AR），长文本稳定性差。仍作为基线被引用。

**FastSpeech 2（2020）。** 非自回归（non-autoregressive）模型。时长预测器（duration predictor）输出每个音素对应多少个梅尔帧。单次前向传播，比 Tacotron 快 10 倍。会损失一些自然度（单调对齐），但应用广泛。

**VITS（2021）。** 以变分推断（variational inference）端到端联合训练编码器、基于流的时长模型和 HiFi-GAN 声码器。质量高、单一模型。2022–2024 年间主导开源 TTS。变体包括：YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 基于流匹配的扩散 Transformer（diffusion transformer）。自然韵律，仅需 5 秒参考音频即可零样本声音克隆（zero-shot voice cloning）。位居 2026 年开源 TTS 排行榜前列。3.35 亿参数。

**Kokoro（2024）。** 小型模型（8200 万参数），可在 CPU 上运行，实时英语 TTS 中的佼佼者。仅支持闭词汇英语，Apache-2.0 协议。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业领域最先进的技术。ElevenLabs v2.5 的情绪标签（如 "[whispered]"、"[laughing]"）和角色声音在 2026 年的有声书制作中占主导地位。

### 声码器演进

| 时期 | 声码器 | 延迟 | 质量 |
|------|--------|------|------|
| 2016 | WaveNet | 仅离线 | 发布时最优 |
| 2018 | WaveRNN | 近实时 | 良好 |
| 2020 | HiFi-GAN | 100 倍实时 | 接近真人 |
| 2022 | BigVGAN | 50 倍实时 | 跨说话人/语言泛化 |
| 2024 | SNAC、DAC（神经编解码） | 与自回归模型集成 | 离散词元，比特高效 |

到 2026 年，大多数“TTS”模型已实现从文本到波形的端到端生成；梅尔谱更多是一种内部表示。

### 评估

- **MOS（Mean Opinion Score，平均意见分）。** 1–5 分，众包评分。仍是金标准；但获取成本极高。
- **CMOS（Comparative MOS，对比平均意见分）。** A/B 偏好测试。每次标注的置信区间更窄。
- **UTMOS、DNSMOS。** 无参考神经 MOS 预测器。常用于排行榜。
- **CER（Character Error Rate，字错误率）通过 ASR 评测。** 将 TTS 输出送入 Whisper，与输入文本计算 CER。作为可懂度的代理指标。
- **SECS（Speaker Embedding Cosine Similarity，说话人嵌入余弦相似度）。** 衡量声音克隆质量。

2026 年在 LibriTTS test-clean 上的数据：

| 模型 | UTMOS | CER（通过 Whisper） | 大小 |
|------|-------|---------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 动手实现

### 步骤 1：将输入音素化

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用的桥梁。不要向低于 VITS 水平的模型直接喂原始文本。

### 步骤 2：运行 Kokoro（2026 年 CPU 默认选择）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

可离线运行，单文件，8200 万参数。

### 步骤 3：使用 F5-TTS 进行声音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

提供一段 5 秒参考音频及其转录文本；F5 即可克隆韵律和音色。

### 步骤 4：从零实现 HiFi-GAN 声码器

完整实现太大，无法放入教程脚本，但结构如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练：对抗损失（短窗判别器）+ 梅尔谱重建损失 + 特征匹配损失。这已成标准化组件——可直接使用 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点。

### 步骤 5：完整管线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 如何选择

2026 年技术栈选型：

| 场景 | 选择 |
|------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 基于 5 秒参考的声音克隆 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书朗读 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 用 5–20 小时目标语言数据训练 VITS |
| 表现力 / 情绪标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

截至 2026 年的开源领先者：**F5-TTS 主打质量，Kokoro 主打效率**。除非你是历史研究者，否则不要选择 Tacotron。

## 常见陷阱

- **缺少文本归一化。** "Dr. Smith" 该读成 "Doctor Smith" 还是 "Drive Smith"？"2026" 该读成 "twenty twenty six" 还是 "two zero two six"？必须在音素化之前进行归一化。
- **OOV（Out-of-Vocabulary，词表外）专有名词。** "Ghumare" 读成 "ghyu-mair"？应为未知词元准备备用的字素到音素（grapheme-to-phoneme）模型。
- **削波。** 声码器输出很少削波，但推理时的梅尔缩放不匹配可能导致超出 ±1.0。务必执行 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；若下游管线期望 16 kHz，必须重采样，否则会产生混叠（aliasing）。

## 交付

保存为 `outputs/skill-tts-designer.md`。为指定的声音、延迟和语言目标设计一条 TTS 管线。

## 练习

1. **简单。** 运行 `code/main.py`。它基于一个玩具词汇表构建音素字典，估计每个音素的时长，并打印一个虚拟的 “mel” 时间表。
2. **中等。** 安装 Kokoro，分别使用 `af_bella` 和 `am_adam` 音色合成同一句子。比较音频时长和主观听感质量。
3. **困难。** 录制一段 5 秒的自己声音作为参考音频。使用 F5-TTS 进行克隆。报告参考音频与克隆输出之间的 SECS。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Phoneme | 声音单位 | 抽象的声音类别；英语中有 39 个（ARPABet）。 |
| Duration predictor | 每个音素持续多久 | 非自回归模型的输出；每个音素对应整数帧。 |
| Vocoder | Mel → waveform | 将梅尔谱映射为原始采样点的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于 GAN；2020–2024 年间占主导。 |
| MOS | 主观质量 | 人类评分员给出的 1–5 平均意见分。 |
| SECS | 声音克隆指标 | 目标与输出说话人嵌入之间的余弦相似度。 |
| F5-TTS | 2024 年开源最优 | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英语领先者 | 8200 万参数模型，Apache 2.0 协议。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — the seq2seq baseline.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end flow-based.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — current open-source SOTA.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — the vocoder that still ships in 2026.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 CPU-friendly English TTS.
