# 频谱图、梅尔尺度与音频特征

> 神经网络并不擅长直接处理原始波形。它们处理的是频谱图。更好地，它们处理的是梅尔频谱图。2026 年的每一个自动语音识别（ASR）、文本转语音（TTS）和音频分类器，成败都取决于这唯一的预处理选择。

**类型：** 实践
**语言：** Python
**前置：** 第 6 阶段 · 01（音频基础）
**时间：** 约 45 分钟

## 问题所在

取一段 10 秒、16 kHz 的音频片段。它包含 16 万个浮点数，全部落在 `[-1, 1]` 区间，与“狗叫”或“单词 cat”这些标签几乎完全不相关。原始波形虽然包含信息，但模型难以直接从中提取。两段相隔 100 毫秒说出的相同音素（phoneme），其原始采样值可能完全不同。

频谱图解决了这个问题。它压缩了人类感知忽略的时间细节（微秒级抖动），同时保留了感知关注处的结构：在约 10–25 毫秒的时间窗口内，哪些频率具有能量。

梅尔频谱图更进一步。人类对音高（pitch）的感知是对数式的：100 Hz 与 200 Hz 的差距，听起来与 1000 Hz 和 2000 Hz 的差距“相同”。梅尔尺度（mel scale）对频率轴进行相应的扭曲。从 2010 年到 2026 年，梅尔刻度频谱图一直是语音机器学习（speech ML）中最重要的特征。

## 核心概念

![从波形到 STFT、梅尔频谱图再到 MFCC 的流程](../assets/mel-features.svg)

**短时傅里叶变换（STFT）。** 将波形切分为相互重叠的帧（frame）（典型参数：25 ms 窗口、10 ms 跳跃，在 16 kHz 下对应 400 个采样点 / 160 个采样点）。将每一帧乘以一个窗函数（window function）（默认使用汉宁窗（Hann）；汉明窗（Hamming）则有略微不同的权衡）。对每一帧做快速傅里叶变换（FFT）。将幅度谱堆叠成形状为 `(n_frames, n_freq_bins)` 的矩阵，这就是你的频谱图。

**对数幅度（log-magnitude）。** 原始幅度跨越 5–6 个数量级。使用 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 压缩动态范围。每一条生产流水线都使用对数幅度，而不是原始幅度。

**梅尔尺度（mel scale）。** 频率 `f`（单位 Hz）映射到梅尔 `m` 的公式为 `m = 2595 * log10(1 + f / 700)`。该映射在 1 kHz 以下大致线性，在 1 kHz 以上大致对数。覆盖 0–8 kHz 的 80 个梅尔频带（mel bins）是 ASR 的标准输入。

**梅尔滤波器组（mel filterbank）。** 一组在梅尔尺度上等距排列的三角滤波器。每个滤波器都是相邻 FFT 频带的加权和。将 STFT 幅度与滤波器组矩阵相乘，即可通过一次矩阵乘法（matmul）得到梅尔频谱图。

**对数梅尔频谱图（log-mel spectrogram）。** `log(mel_spec + 1e-10)`。Whisper 的输入、Parakeet 的输入、SeamlessM4T 的输入。它是 2026 年通用的音频前端。

**梅尔频率倒谱系数（MFCC）。** 对数梅尔频谱图经过离散余弦变换（DCT，II 型）后，保留前 13 个系数。它使特征去相关，并进一步压缩。在 2015 年之前，MFCC 一直是主流特征；此后基于原始对数梅尔频谱图的卷积神经网络（CNN）/ Transformer 才逐渐追赶上。如今仍用于说话人识别（speaker recognition）（如 x-vectors、ECAPA）。

**分辨率权衡。** FFT 越大，频率分辨率越好，但时间分辨率越差。25 ms / 10 ms 是音频机器学习的默认参数；音乐常用 50 ms / 12.5 ms；瞬态检测（transient detection）（如鼓点、爆破音（plosives））则用 5 ms / 2 ms。

## 动手实现

### 步骤 1：对波形分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 10 秒、16 kHz 的音频片段，使用 `frame_len=400, hop=160` 分帧后，会得到 998 帧。

### 步骤 2：汉宁窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 之前逐元素相乘。可以消除因在非零端点处截断而引起的频谱泄漏（spectral leakage）。

### 步骤 3：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境中使用 `torch.stft` 或 `librosa.stft`（基于 FFT、向量化）。这里的循环仅用于教学；它适用于 `code/main.py` 中的短音频片段。

### 步骤 4：梅尔滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

覆盖 0–8 kHz 的 80 个梅尔频带，在 `n_fft=400` 下会得到一个 `(80, 201)` 矩阵。将 `(n_frames, 201)` 的 STFT 幅度与其转置相乘，即可得到 `(n_frames, 80)` 的梅尔频谱图。

### 步骤 5：对数梅尔

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案：`librosa.power_to_db`（参考归一化的 dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的裁剪 + 归一化流程（参见 Whisper 的 `log_mel_spectrogram`）。

### 步骤 6：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每一帧对数梅尔频谱应用 DCT，保留前 13 个系数。这就是你的 MFCC 矩阵。第一个系数通常会被丢弃（它编码了整体能量）。

## 实际应用

2026 年的技术栈：

| 任务 | 特征 |
|------|------|
| 自动语音识别 ASR（Whisper、Parakeet、SeamlessM4T） | 80 个对数梅尔频带，10 ms 跳跃，25 ms 窗口 |
| 文本转语音 TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 个梅尔频带，5–12 ms 跳跃，用于精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 个对数梅尔频带，10 ms 跳跃 |
| 说话人嵌入（speaker embedding）（ECAPA-TDNN、WavLM） | 80 个对数梅尔频带，或原始波形自监督学习（SSL） |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（非梅尔） |
| 关键词唤醒（keyword spotting） | 用于微型设备的 40 维 MFCC |

经验法则：**如果你不是在处理音乐，请从 80 个对数梅尔频带开始。** 任何偏离都需要由你承担举证责任。

## 2026 年仍会踩到的坑

- **梅尔频带数量不一致。** 训练时用 80 个梅尔频带，推理时用 128 个。这是静默失败（silent failure）。请在两端都记录特征形状。
- **上游采样率不一致。** 在 22.05 kHz 下计算的梅尔频谱图与在 16 kHz 下的看起来不同。请在特征提取*之前*先统一采样率（SR）。
- **dB 与 log 的区别。** Whisper 期望的是 log-mel，而不是 dB-mel。一些 Hugging Face 流水线会自动检测；你的自定义代码不会。
- **归一化漂移。** 训练时按每条语音（per-utterance）归一化，推理时却做全局归一化。这是会让词错误率（WER）翻倍的线上 bug。
- **填充导致的泄漏。** 在音频末尾补零会在尾部帧产生平坦频谱。请使用对称填充或复制填充。

## 交付

保存为 `outputs/skill-feature-extractor.md`。该技能会为给定的模型目标选择特征类型、梅尔频带数量、帧长/跳跃以及归一化方式。

## 练习

1. **入门。** 运行 `code/main.py`。它会合成一个啁啾信号（chirp，频率从 200 Hz 扫到 4000 Hz），并打印每一帧中 argmax 所在的梅尔频带。可绘图（可选），并确认结果与扫频一致。
2. **中等。** 使用 `n_mels` 取 `{40, 80, 128}`、`frame_len` 取 `{200, 400, 800}` 重新运行。测量时间轴上的尖峰带宽。哪种组合最能分辨这个啁啾信号？
3. **困难。** 实现 `power_to_db`，并在 AudioMNIST 上比较一个小型 CNN 分类器的 ASR 准确率，分别使用 (a) 原始 log-mel、(b) `ref=max` 的 dB-mel、(c) MFCC-13 + delta + delta-delta。报告 top-1 准确率。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| 帧（frame） | 一段切片 | 输入到一次 FFT 的 25 ms 波形片段。 |
| 跳跃（hop） | 步幅（stride） | 相邻帧之间的采样点数；10 ms 是 ASR 默认值。 |
| 窗（window） | Hann/Hamming 那个东西 | 逐点乘数，将帧两端逐渐收束到零。 |
| STFT | 频谱图生成器 | 分帧加窗后的 FFT；产生时间 × 频率矩阵。 |
| Mel | 扭曲的频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组（filterbank） | 那个矩阵 | 三角滤波器，将 STFT 投影到梅尔频带上。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年的标准。 |
| MFCC | 老派特征 | 对数梅尔频谱图的 DCT；13 个系数，去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始 mel 尺度论文。
- [OpenAI — Whisper 源码，log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa 特征提取文档](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 以及 hop/window 的参考。
- [NVIDIA NeMo — 音频预处理](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产级流水线。
