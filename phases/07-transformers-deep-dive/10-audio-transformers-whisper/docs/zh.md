# 音频 Transformer — Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个吃梅尔频谱图（mel spectrogram）、吐出文本的 ViT。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer）、Phase 7 · 08（编码器-解码器）、Phase 7 · 09（ViT）
**时长：** 约 45 分钟

## 问题背景

在 Whisper（OpenAI，Radford 等人，2022）之前，最先进的自动语音识别（ASR, automatic speech recognition）是 wav2vec 2.0 和 HuBERT —— 自监督特征提取器加一个微调头。质量高，但数据流水线昂贵，且容易受领域限制。多语言语音识别需要为每个语系单独训练模型。

Whisper 下了三个赌注：

1. **用所有数据训练。** 68 万小时弱标注音频，从互联网上抓取，覆盖 97 种语言。没有干净的学术语料，也没有音素标注。
2. **单一模型多任务。** 一个解码器联合训练转录、翻译、语音活动检测（VAD）、语言识别和时间戳预测，通过任务词元（task tokens）区分任务。
3. **标准编码器-解码器 Transformer。** 编码器消费对数梅尔频谱图（log-mel spectrogram），解码器自回归地生成文本词元。没有声码器，没有 CTC，没有 HMM。

结果：Whisper large-v3 在口音、噪声和零干净标注数据的语言上都表现稳健。到 2026 年，它已成为所有开源语音助手和大多数商业语音助手的默认语音前端。

## 核心概念

![Whisper 流水线：音频 → mel → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 分帧

音频采样率为 16 kHz。裁剪或填充到 30 秒。计算对数梅尔频谱图：80 个 mel 频段，10 ms 步幅 → 约 3,000 帧 × 80 维特征。这就是 Whisper 看到的“输入图像”。

### 步骤 2 — 卷积词干（convolutional stem）

两层 Conv1D，卷积核为 3、步幅为 2，将 3,000 帧降到 1,500。序列长度减半，参数量增加不多。

### 步骤 3 — 编码器

大型模型使用 24 层 Transformer 编码器，处理 1,500 个时间步。正弦位置编码（sinusoidal positional encoding）、自注意力（self-attention）、GELU 前馈网络（FFN）。输出 1,500 × 1,280 的隐藏状态。

### 步骤 4 — 解码器

24 层 Transformer 解码器。它自回归地从 BPE 词表中生成词元，该词表是 GPT-2 词表的超集，并增加了少量音频专用特殊词元。

### 步骤 5 — 任务词元

解码器提示（prompt）以控制词元开头，告诉模型该做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型就是按这一约定训练的。你通过前缀控制任务。这是 2026 年指令微调（instruction-tuning）在语音上的等价物。

### 步骤 6 — 输出

束搜索（beam search，宽度 5）配合对数概率阈值。当不存在 `<|notimestamps|>` 词元时，模型每 0.02 秒音频预测一次时间戳。

### Whisper 尺寸

| 模型 | 参数量 | 层数 | d_model | 注意力头 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4 层解码器) |

Large-v3-turbo（2024）将解码器从 32 层砍到 4 层。解码速度提升 8 倍，词错率（WER）仅下降不到 1 个百分点。正是这一解码速度突破，使 Whisper-turbo 成为 2026 年实时语音助手的默认选择。

### Whisper 不做什么

- 不做说话人分离（diarization，即判断谁在说话）。需要搭配 pyannote。
- 原生不支持实时流式 —— 30 秒窗口是固定的。现代封装（`faster-whisper`、`WhisperX`）通过 VAD + 重叠窗口强行实现流式。
- 不借助外部分块就无法处理超过 30 秒的长上下文。实际中效果尚可，因为人类语音的转录很少需要长距离上下文。

### 2026 年格局

| 任务 | 模型 | 备注 |
|------|-------|-------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘设备上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 ms 延迟目标 |
| 文本转语音（TTS） | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，但形似 Whisper |
| 音频 + 语言 | AudioLM、SeamlessM4T | 一个 Transformer 同时处理文本词元和音频词元 |

## 动手实现

参见 `code/main.py`。我们不训练 Whisper —— 而是构建对数梅尔频谱图流水线 + 任务词元提示格式化器。这些才是生产环境中你真正会改动的部分。

### 步骤 1：合成音频

生成一个 1 秒长的 440 Hz 正弦波，采样率 16 kHz。共 16,000 个样本。

### 步骤 2：对数梅尔频谱图（简化版）

完整 mel 频谱图需要 FFT。这里用简化的分帧 + 每帧能量版本来展示流水线，无需 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧长 25 ms，帧移 10 ms。与 Whisper 的窗口设置一致。每帧能量用于教学目的，代替 mel 频段。

### 步骤 3：填充到 30 秒

Whisper 始终处理 30 秒音频块。将频谱图填充（或裁剪）到 3,000 帧。

### 步骤 4：构建提示词元

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制接口。一个 4 词元前缀。

## 使用 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、兼容 OpenAI 的写法：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 需要一个模型覆盖多语言 ASR。
- 需要对噪声大、来源多样的音频进行稳健转录。
- 做研究 / 原型 ASR —— 最快的起点。

**何时选择其他方案：**

- 边缘端超低延迟流式 —— Moonshine 在同等质量下击败 Whisper。
- 需要 <200 ms 的实时对话 AI —— 用专用流式 ASR。
- 说话人分离 —— Whisper 不做这个；需要搭配 pyannote。

## 交付成果

参见 `outputs/skill-asr-configurator.md`。该技能会为新语音应用挑选 ASR 模型、解码参数和预处理流水线。

## 练习题

1. **简单。** 运行 `code/main.py`。确认 16 kHz、1 秒信号在 10 ms 帧移下约为 100 帧；30 秒则约为 3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整对数梅尔频谱图。验证 80 个 mel 频段与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差范围内一致。
3. **困难。** 实现流式推理：将音频切分为 10 秒窗口，重叠 2 秒，对每个窗口运行 Whisper，合并转录结果。在 5 分钟播客样本上测量与单次处理的词错率对比。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Mel 频谱图（Mel spectrogram） | “音频图像” | 二维表示：一个轴是频率频段，一个轴是时间帧；每个单元为对数缩放后的能量。 |
| 对数梅尔（Log-mel） | “Whisper 看到的东西” | 经过对数变换的 mel 频谱图，近似人类对响度的感知。 |
| 帧（Frame） | “一个时间片” | 25 ms 的样本窗口，以 10 ms 步幅重叠。 |
| 任务词元（Task token） | “语音的提示前缀” | 解码器提示中的特殊词元，如 `<|transcribe|>` / `<|translate|>`。 |
| 语音活动检测（VAD, voice activity detection） | “找到语音” | 在 ASR 前剔除静音的门控，能大幅降低成本。 |
| CTC | “连接时序分类（Connectionist Temporal Classification）” | 经典无对齐 ASR 损失；Whisper 不使用它。 |
| Whisper-turbo | “小解码器，完整编码器” | large-v3 编码器 + 4 层解码器；解码速度快 8 倍。 |
| Faster-whisper | “生产级封装” | 基于 CTranslate2 的重实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py`，约 400 行即可自上而下看完 Conv1D 词干 + 编码器 + 解码器。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 步骤 5–6 中描述的束搜索 + 任务词元逻辑就在这里；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 先驱工作；某些场景下仍是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产级封装，比参考实现快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年面向边缘设备的 ASR，形似 Whisper 但更小巧。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 微调标准配方，包括 mel 频谱图预处理器和词元-时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），与本课架构图一一对应。
