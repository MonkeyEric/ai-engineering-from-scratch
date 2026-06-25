# 音频基础——波形、采样与傅里叶变换

> 波形（waveform）是原始信号，频谱图（spectrogram）是中间表示，Mel 特征是适合机器学习使用的形式。现代所有自动语音识别（ASR）和文本转语音（TTS）流水线都要沿这条阶梯向上走，而第一级就是理解采样与傅里叶变换。

**类型：** 学习  
**语言：** Python  
**前置知识：** Phase 1 · 06（向量与矩阵），Phase 1 · 14（概率分布）  
**时长：** 约 45 分钟

## 问题

麦克风输出的是压力随时间变化的信号，而你的神经网络消费的是张量。两者之间堆叠着一系列约定；一旦违反，就会产生“静默 bug”：模型训练看起来很顺利，但词错误率（WER）翻倍；TTS 输出发出嘶嘶声；或者语音克隆系统记住了麦克风而不是说话人。

语音系统中的每个 bug 都可以追溯到以下三个问题之一：

1. 数据的采样率是多少？模型期望的采样率又是多少？
2. 信号是否发生了混叠（aliasing）？
3. 你处理的是原始样本，还是某种频率表示？

把这三个问题搞对，Phase 6 的其余内容就会变得可控；搞错了，即使是 Whisper-Large-v4 也会输出垃圾。

## 概念

![波形、采样、DFT 与频率槽的可视化](../assets/audio-fundamentals.svg)

**波形（Waveform）。** 一个位于 `[-1.0, 1.0]` 区间的一维浮点数组，按样本序号索引。要换算成秒，除以采样率：`t = n / sr`。一段 10 秒、16 kHz 的音频就是一个包含 160,000 个浮点数的数组。

**采样率（Sampling rate，sr）。** 每秒采集的样本数。2026 年的常见采样率：

| 采样率 | 用途 |
|------|-----|
| 8 kHz | 电话、传统 VOIP。奈奎斯特频率（Nyquist）为 4 kHz，会损失辅音信息。ASR 应避免使用。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都使用 16 kHz。 |
| 22.05 kHz | 旧模型的 TTS 声码器（vocoder）训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 影视、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农定理（Nyquist-Shannon）。** 采样率为 `sr` 时，可以无歧义地表示最高到 `sr/2` 的频率。`sr/2` 这条边界称为**奈奎斯特频率（Nyquist frequency）**。高于奈奎斯特频率的能量会发生**混叠（aliasing）**——被折叠回更低的频率并污染信号。因此降采样前务必先进行低通滤波（low-pass filter）。

**位深（Bit depth）。** 16 位 PCM（有符号 int16，范围 ±32,767）是通用的交换格式；24 位用于音乐，32 位浮点用于内部数字信号处理（DSP）。`soundfile` 等库读取 int16，但会暴露为 `[-1, 1]` 范围内的 float32 数组。

**傅里叶变换（Fourier Transform）。** 任何有限信号都可以表示为不同频率正弦波的叠加。离散傅里叶变换（DFT）对 `N` 个样本计算 `N` 个复数系数，每个频率槽（bin）一个。`bin k` 对应的频率为 `k · sr / N` Hz；模长是该频率的幅度，角度是该频率的相位。

**快速傅里叶变换（FFT）。** 当 `N` 为 2 的幂时，计算 DFT 的 `O(N log N)` 算法。每个音频库底层都用 FFT。在 16 kHz 下对 1024 个样本做 FFT，可得到 512 个可用频率槽，覆盖 0–8 kHz，分辨率为 15.6 Hz。

**分帧与加窗（Framing + window）。** 我们不会对整段音频直接做 FFT，而是将其切分为相互重叠的**帧**（frame，典型设置为 25 ms 帧长、10 ms 帧移），每帧乘以一个窗函数（Hann、Hamming）以消除边缘不连续，然后对每帧分别 FFT。这就是**短时傅里叶变换（STFT）**。第 02 课会在此基础上继续展开。

## 动手实现

### 步骤 1：读取音频片段并绘制波形

`code/main.py` 只使用标准库中的 `wave` 模块，以便演示零依赖。生产环境中你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步骤 2：从第一性原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

440 Hz 的正弦波（标准音 A）在 16 kHz 下持续 1 秒，会得到 16,000 个浮点数。使用 `wave.open(..., "wb")` 以 16 位 PCM 编码写入。

### 步骤 3：手写 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

复杂度为 `O(N²)`——对 `N=256` 验证正确性尚可，真实音频中毫无实用性。真实代码应调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤 4：寻找主频

幅度峰值索引 `k_star` 对应的频率为 `k_star * sr / N`。对上面的 440 Hz 正弦波运行此操作，应在 `440 * N / sr` 号槽出现峰值。

### 步骤 5：演示混叠

以 10 kHz 采样一个 7 kHz 正弦波（奈奎斯特频率为 5 kHz）。7 kHz 高于奈奎斯特频率，会折叠到 `10 − 7 = 3 kHz`，FFT 峰值出现在 3 kHz 处。这就是经典的混叠演示，也是每个 DAC/ADC 都配备砖墙式低通滤波器的原因。

## 实际应用

2026 年你真正会部署的栈：

| 任务 | 库 | 理由 |
|------|---------|-----|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、稳定、返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠处理。 |
| STFT / Mel | `torchaudio` 或 `librosa` | 支持 GPU；PyTorch 生态。 |
| 实时流式音频 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 查看文件信息 | `ffprobe` 或 `soxi` | 命令行工具，快速报告采样率/通道/编码。 |

决策规则：**先匹配采样率，再匹配其他任何东西**。Whisper 期望 16 kHz 单声道 float32。传入 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾输出。

## 交付

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否符合下游模型的预期，并在不符合时正确重采样。

## 练习

1. **简单。** 在 16 kHz 下合成一段 1 秒的 220 Hz + 440 Hz + 880 Hz 混合信号。运行 DFT，确认在预期槽位出现三个峰值。
2. **中等。** 在 48 kHz 下录制一段 3 秒的人声 WAV。先用 `torchaudio.transforms.Resample`（带抗混叠）降采样到 16 kHz，再用朴素抽取（每三个样本取一个）降到 16 kHz。对两者分别做 FFT，混叠出现在哪里？
3. **困难。** 仅使用 `math` 和步骤 3 中的 DFT，从零实现 STFT。帧大小 400，帧移 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅度，这就是第 02 课要讲的频谱图。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 采样率（Sample rate） | 每秒多少样本 | ADC 测量信号时的频率，单位为 Hz。 |
| 奈奎斯特频率（Nyquist） | 能表示的最高频率 | `sr/2`；高于它的能量会混叠回低频。 |
| 位深（Bit depth） | 每个样本的分辨率 | `int16` = 65,536 个等级；`float32` = `[-1, 1]` 范围内的 24 位精度。 |
| DFT | 序列的傅里叶变换 | `N` 个样本 → `N` 个复数频率系数。 |
| FFT | 快速 DFT | `O(N log N)` 算法，要求 `N` 为 2 的幂。 |
| 频率槽（Bin） | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层实现 | 随时间进行分帧加窗的 FFT。 |
| 混叠（Aliasing） | 奇怪的频率幽灵 | 高于奈奎斯特频率的能量镜像到更低的槽位。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) ——采样定理背后的原始论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) ——免费、经典的 DSP 教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) ——带代码的实用入门教程。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) ——参考现实世界音频为何不是干净正弦波。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) ——10 分钟理清频率槽直觉。
