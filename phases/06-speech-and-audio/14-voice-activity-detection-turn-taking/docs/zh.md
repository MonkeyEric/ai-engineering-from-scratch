# 语音活动检测与话轮转换 — Silero、Cobra 与 Flush 技巧

> 每个语音智能体的成败都取决于两个判断：用户现在是否在说话，以及他们是否说完了？语音活动检测（VAD）回答第一个问题；话轮检测（turn detection，即 VAD + 静音拖尾 + 语义端点模型）回答第二个。这两个判断中任意一个出错，你的助手要么会打断用户，要么滔滔不绝停不下来。

**类型：** 构建  
**语言：** Python  
**前置知识：** 第 6 阶段 · 11（实时音频），第 6 阶段 · 12（语音助手）  
**时间：** 约 45 分钟

## 问题背景

语音智能体在每个 20 毫秒的音频块上都要做出三个不同决策：

1. **当前帧是不是语音？** —— VAD。按帧进行二分类。
2. **用户是否开始了新的语句？** —— 起始检测（onset detection）。
3. **用户是否说完了？** —— 端点检测（end-pointing，即话轮结束）。

朴素的方案（能量阈值）在任何噪声环境下都会失效 —— 交通声、键盘声、人群嘈杂声都不行。2026 年的答案是：Silero VAD（开源、深度学习）+ 话轮检测模型（语义端点检测）+ 经 VAD 校准的静音拖尾（silence hangover）。

## 核心概念

![VAD 级联：能量门 → Silero → 话轮检测器 → flush 技巧](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第一层：能量门（energy gate）。** 成本最低。将 RMS 阈值设为 -40 dBFS。能滤除明显静音，但任何高于阈值的噪声都会触发。

**第二层：Silero VAD**（2020–2026，MIT 许可）。100 万参数。使用 6000 多种语言训练。在单 CPU 线程上每 30 毫秒音频块约 1 毫秒完成推理。在 5% 误报率（FPR）下真正率（TPR）为 87.7%。开源场景下的默认选择。

**第三层：语义话轮检测器（semantic turn detector）。** LiveKit 的话轮检测模型（2024–2026）或你自己的小型分类器。区分“句中停顿”和“说完了”。它使用语言上下文（语调 + 最近词汇），而不仅仅是静音。

### 关键参数及其默认值

- **阈值（Threshold）。** Silero 输出的是概率；默认以 &gt; 0.5 判定为语音，敏感模式可用 &gt; 0.3。阈值越低，越少截断首个词，但误报越多。
- **最短语音时长（Minimum speech duration）。** 拒绝短于 250 毫秒的语音 —— 通常是咳嗽声或椅子挪动声。
- **静音拖尾（Silence hangover，端点检测）。** VAD 返回 0 后，再等待 500–800 毫秒才宣布话轮结束。太短 → 打断用户；太长 → 感觉迟钝。
- **预卷缓冲（Pre-roll buffer）。** 在 VAD 触发前保留 300–500 毫秒的音频。防止“喂”这样的开头词被截断。

### Flush 技巧（Kyutai，2025）

流式语音转文本（STT）模型存在前视延迟（Kyutai STT-1B 为 500 毫秒，STT-2.6B 为 2.5 秒）。通常话音结束后还需等待这段时间才能获得转写结果。Flush 技巧：当 VAD 检测到话音结束时，**向 STT 发送一个 flush 信号**，强制立即输出。STT 以约 4 倍实时速度处理，因此 500 毫秒的缓冲约 125 毫秒即可完成。

端到端：125 毫秒 VAD + flush STT = 对话级延迟。

### 2026 年 VAD 对比

| VAD | 5% FPR 下的 TPR | 延迟 | 许可 |
|-----|----------------|------|------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020–2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业许可 |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是正确的默认选择。Cobra 是合规 / 精度升级方案。纯能量 VAD 在 2026 年的生产环境中没有立足之地。

## 动手实现

### 步骤 1：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：在 Python 中使用 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤 3：话轮结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤 4：flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能生效。Whisper 流式不支持 —— 它是基于块处理的，总是等待分块完成。

## 如何选择

| 场景 | VAD 选择 |
|-----------|-----------|
| 开放、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究 / 说话人分割 | pyannote segmentation |
| 零依赖兜底 | WebRTC VAD（遗留方案） |
| 需要话轮结束质量 | Silero + LiveKit turn-detector 分层 |

经验法则：除非真的别无选择，否则不要上线纯能量 VAD。

## 常见陷阱

- **固定阈值。** 安静环境有效，嘈杂环境失效。要么在设备上校准，要么切换到 Silero。
- **静音拖尾太短。** 智能体会打断用户说了一半的话。500–800 毫秒是对话语音的甜点。
- **静音拖尾太长。** 感觉迟钝。应与目标用户进行 A/B 测试。
- **没有预卷缓冲。** 会丢失用户音频开头的 200–300 毫秒。务必保留滚动预卷缓冲。
- **忽视语义端点检测。** “嗯，让我想想……”包含较长停顿。用户讨厌在思考中途被打断。应使用 LiveKit 的 turn-detector 或类似方案。

## 交付成果

保存为 `outputs/skill-vad-tuner.md`。针对你的工作负载，选择 VAD 模型、阈值、拖尾、预卷和话轮检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟了语音 + 静音 + 语音 + 咳嗽的序列，并测试三层 VAD。
2. **中等。** 安装 `silero-vad`，处理一段 5 分钟录音，调整阈值以尽量减少首词截断和误触发。报告精确率 / 召回率。
3. **困难。** 构建一个迷你话轮检测器：Silero VAD + 一个基于最近 10 个词嵌入（使用 sentence-transformers）的 3 层 MLP。在手工标注的话轮结束数据集上训练。以 F1 指标比纯 Silero 提升 10%。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| VAD（Voice Activity Detection，语音活动检测） | 人声检测器 | 按帧二分类：这是语音吗？ |
| Turn detection（话轮检测） | 端点检测 | VAD + 静音拖尾 + 语义端点。 |
| Silence hangover（静音拖尾） | 话音后等待 | 宣布话轮结束前需要等待的时间；500–800 毫秒。 |
| Pre-roll（预卷缓冲） | 语音前缓冲 | 在 VAD 触发前保留 300–500 毫秒音频。 |
| Flush trick（Flush 技巧） | Kyutai 技巧 | VAD → flush-STT → 从 500 毫秒延迟降到 125 毫秒。 |
| Semantic endpoint（语义端点） | “TA 是故意停下的吗？” | 看词语而不是只看静音的机器学习分类器。 |
| TPR @ FPR 5% | ROC 点 | VAD 标准评测点；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) —— 开源 VAD 的参考实现。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) —— 商用精度领先者。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) —— 低于 200 毫秒的工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) —— 生产级语义端点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) —— 遗留基线。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) —— 说话人分割级别分割。
