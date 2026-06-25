# 实时音频处理

> 批处理流水线处理的是文件，而实时流水线要在下一批 20 毫秒到达之前处理完当前这 20 毫秒。每一个对话式 AI、每一个演播室、每一个电话机器人，都是在这样的延迟预算里生存的。

**类型：** Build
**语言：** Python
**前置条件：** Phase 6 · 02（Spectrograms，频谱图）、Phase 6 · 04（ASR，自动语音识别）、Phase 6 · 07（TTS，文本转语音）
**时间：** ~75 分钟

## 问题背景

你想要一个“有生命感”的语音助手。人类对话轮转的延迟约为 230 毫秒（从静默到回应）。超过 500 毫秒会显得像机器人，超过 1500 毫秒则像坏了。2026 年，一个完整的**听 → 理解 → 回应 → 说**循环的延迟预算大致如下：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 ms |
| VAD（语音活动检测） | 10 ms |
| ASR（streaming，流式识别） | 150 ms |
| LLM（首 token） | 100 ms |
| TTS（首块音频） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai，2024）实现了 200 毫秒的全双工延迟；GPT-4o-realtime（2024）约为 320 毫秒。而 2022 年的级联流水线还在 2500 毫秒。10 倍提升来自三项技术：(1) 全链路流式处理、(2) 基于部分结果的异步流水线、(3) 可中断生成。

## 核心概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**帧 / 块 / 窗口（frame / chunk / window）。** 实时音频以固定大小的数据块流动。常见选择：20 毫秒（16 kHz 下为 320 个采样点）。下游所有模块都必须跟上这个节奏。

**环形缓冲区（ring buffer）。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。避免在热路径上分配内存。大小 ≈ 最大延迟 × 采样率；2 秒 16 kHz 的环形缓冲区为 32,000 个采样点。

**VAD（Voice Activity Detection，语音活动检测）。** 当没有说话时关闭下游工作，减少计算。Silero VAD 4.0（2024）在 CPU 上每 30 毫秒帧耗时不到 1 毫秒。`webrtcvad` 是更老的替代方案。

**流式 ASR（streaming ASR）。** 音频到达时即输出部分转录文本的模型。Parakeet-CTC-0.6B 的流式模式（NeMo，2024）在 320 毫秒延迟下词错误率（WER）为 2–5%。Whisper-Streaming（Macháček 等，2023）将 Whisper 分块，实现近流式识别，延迟约 2 秒。

**打断（interruption / barge-in）。** 当用户在助手说话时开口，你必须在 100 毫秒内：(a) 检测到插话、(b) 停止 TTS、(c) 丢弃剩余 LLM 输出，否则用户会觉得助手“听不见”。

**WebRTC Opus 传输。** 20 毫秒帧、48 kHz、自适应码率 8–128 kbps。浏览器和移动端的事实标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的常用协议栈。

**抖动缓冲区（jitter buffer）。** 网络包可能乱序或延迟到达。抖动缓冲区负责重排和平滑；太小会出现可闻断续，太大会增加延迟。典型值为 60–80 毫秒。

### 常见陷阱

- **线程竞争（Thread contention）。** Python 的 GIL 加上重型模型会饿死音频线程。应使用 C 回调式音频库（sounddevice、PortAudio），让 Python 远离热路径。
- **采样率转换延迟。** 流水线内部的重采样会增加 5–20 毫秒。要么在入口统一重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热（TTS priming）。** 即使是 Kokoro 这样的快速 TTS，首次请求也有 100–200 毫秒预热。应在首次真实对话前缓存模型并用一次虚拟运行预热。
- **回声消除（Echo cancellation）。** 没有 AEC 的话，TTS 输出会重新进入麦克风，并被 ASR 识别成机器人自己的话。WebRTC AEC3 是开源默认方案。

## 动手实现

### 第一步：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz 下 32,000 个采样点 = 2 秒。

### 第二步：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境请替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第三步：流式 ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第四步：打断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

关键依赖异步 I/O 和可取消的 TTS 流式播放。在 WebRTC 中，通常调用 peerconnection.stop() 来停止音频轨。

## 如何使用

2026 年的典型技术栈：

| 层次 | 选型 |
|-------|------|
| 传输层 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生方案 | OpenAI Realtime API 或 Moshi |

## 易错点

- **为了保险缓冲 500 毫秒。** 缓冲区的长度就是你延迟的下限。应尽量缩小它。
- **没有固定音频线程优先级。** 音频回调跑在比 UI 线程低的优先级上，负载高时会出现爆音。
- **TTS 块太小。** 低于 200 毫秒的块会让声码器伪影变得可闻。320 毫秒是甜点。
- **没有抖动缓冲区。** 真实网络有抖动；没有平滑处理就会出现爆音。
- **单次错误处理。** 音频流水线必须防崩溃。一个未捕获异常就会终结整个会话。

## 交付任务

保存为 `outputs/skill-realtime-designer.md`。设计一条实时音频流水线，并为每个阶段给出具体的延迟预算。

## 练习题

1. **简单。** 运行 `code/main.py`。它会模拟环形缓冲区 + 能量 VAD，并打印一个 10 秒假流的各阶段延迟。
2. **中等。** 使用 `sounddevice` 构建一个直通环路：以 20 毫秒帧处理麦克风输入，并在每帧打印 VAD 状态。
3. **困难。** 用 `aiortc` 做一个全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量端到端玻璃到玻璃延迟。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Ring buffer（环形缓冲区） | 循环队列 | 固定大小、无锁（或 SPSC 锁）的音频帧 FIFO。 |
| VAD（语音活动检测） | 静音门 | 区分语音与非语音的模型或启发式规则。 |
| Streaming ASR（流式 ASR） | 实时语音转文字 | 音频到达时输出部分文本；前视（lookahead）有界。 |
| Jitter buffer（抖动缓冲区） | 网络平滑器 | 对乱序包重排序并平滑；典型 60–80 毫秒。 |
| AEC（回声消除） | Echo cancellation | 消除扬声器到麦克风的反馈路径。 |
| Barge-in（插话/打断） | 用户中断 | 系统在 TTS 播放期间检测到用户说话，必须取消播放。 |
| Full duplex（全双工） | 双向同时 | 用户和机器人可同时说话；Moshi 即全双工。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 200 毫秒全双工延迟。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产级音频智能体编排。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 亚毫秒级 VAD，Apache 2.0 协议。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除。
