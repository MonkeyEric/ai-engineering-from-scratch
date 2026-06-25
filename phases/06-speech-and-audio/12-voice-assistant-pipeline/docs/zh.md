# 构建语音助手流水线 —— 第 6 阶段综合项目

> 把第 01–11 课的内容全部串起来。打造一个能听、会想、会回答的语音助手。在 2026 年，这已经是一个工程问题而不是研究问题——但集成细节决定它能否真正上线。

**类型：** 构建
**语言：** Python
**前置条件：** 第 6 阶段 · 04、05、06、07、11；第 11 阶段 · 09（函数调用）；第 14 阶段 · 01（智能体循环）
**时长：** ~120 分钟

## 问题

构建一个端到端助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的起止点。
3. 流式转写。
4. 把转写文本传给能调用工具的大语言模型（LLM）（计时器、天气、日历）。
5. 将 LLM 文本流式传给文本转语音（TTS）。
6. 把音频播放给用户。
7. 如果用户在助手回答时插话，就停下来。

延迟目标：在笔记本 CPU 上，用户说完话后 800 ms 内输出第一个 TTS 音频字节。质量目标：不丢词、静音时不出现幻觉字幕、不泄露声音克隆、不被提示注入攻击成功。

## 概念

![语音助手流水线：麦克风 → VAD → STT → LLM+工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频采集。** 麦克风 → 16 kHz 单声道 → 20 ms 分块。Python 中通常用 `sounddevice`，生产环境则使用原生 AudioUnit/ALSA/WASAPI。
2. **语音活动检测（VAD）（第 11 课）。** Silero VAD，阈值 0.5，最短语音 250 ms，静音挂起 500 ms。输出“开始”和“结束”信号。
3. **流式语音转文本（STT）（第 4–5 课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。输出部分转写和最终转写。
4. **带工具调用的大语言模型。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。用 JSON 模式定义工具。流式输出 token。
5. **流式文本转语音（TTS）（第 7 课）。** Kokoro-82M（最快的开源方案）或 Cartesia Sonic（商业方案）。在 LLM 输出 20 个 token 后开始 TTS。
6. **播放。** 扬声器输出； opus 编码用于低带宽网络。
7. **打断处理器。** 如果在 TTS 播放期间 VAD 触发，就停止播放、取消 LLM、重启 STT。

### 你一定会遇到的三种失败模式

1. **首词截断。** VAD 启动稍晚，用户的“嘿”被切掉。把起始阈值设为 0.3，而不是 0.5。
2. **中途打断混乱。** 用户打断后 LLM 仍在生成，助手会盖过用户说话。要把 VAD → 取消 LLM 连起来。
3. **静音幻觉。** Whisper 在静音预热帧上输出“Thanks for watching”。始终用 VAD 做门控。

### 2026 年生产参考栈

| 技术栈 | 延迟 | 许可证 | 说明 |
|--------|------|--------|------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350–500 ms | 商业 API | 2026 年行业默认方案 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500–800 ms | 基本开源 | 适合自己动手 |
| Moshi（全双工） | 200–300 ms | CC-BY 4.0 | 单模型；架构不同，见第 15 课 |
| Vapi / Retell（托管） | 300–500 ms | 商业 | 上线最快；可定制性有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私 / 边缘端 |

## 动手实现

### 第 1 步：带分块的麦克风采集（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 第 2 步：VAD 门控的回合采集

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 第 3 步：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 第 4 步：LLM 循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 第 5 步：打断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用它

参见 `code/main.py`，其中有一个可运行的模拟程序，用占位模型把七个组件连起来，即使没有硬件也能看到流水线全貌。要实现真实版本，把占位模块替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于 I/O

## 常见陷阱

- **永久记录 PII。** 完整回合音频在大多数司法管辖区都属于个人身份信息（PII）。保留 30 天，并静态加密。
- **不支持抢话。** 用户会打断。你的助手必须停下来。
- **阻塞式 TTS。** 同步 TTS 会阻塞事件循环。使用异步 TTS 或单独线程。
- **没有工具调用错误处理。** 工具会失败。LLM 必须收到错误信息并重试一次，然后优雅降级。
- **过度激进的幻觉过滤。** 过滤太严，助手会反复说“我无法帮你”。过滤太松，它会乱说。用留出集校准。
- **没有唤醒词选项。** 始终监听是隐私风险。加上唤醒词门控（Porcupine 或 openWakeWord）。

## 上线

另存为 `outputs/skill-voice-assistant-architect.md`。在给定预算、规模、语言和合规约束下，产出一份完整的技术栈规范。

## 练习

1. **简单。** 运行 `code/main.py`。它用占位模块模拟一个完整回合并打印各阶段延迟。
2. **中等。** 把 STT 占位模块换成真实的 Whisper 模型，在一段预录的 `.wav` 上运行。测量词错误率（WER）和端到端延迟。
3. **困难。** 加上工具调用：实现 `get_weather`（任意 API）和 `set_timer`。让 LLM 经过工具路由，并验证当用户说“set a 5 minute timer”时，正确的函数被触发，且语音回复确认这一点。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| 回合（Turn） | 用户 + 助手的一次往返 | 一次由 VAD 界定的用户语音 + 一段 LLM-TTS 回复。 |
| 抢话（Barge-in） | 插话 | 用户在助手说话时开口；助手停止。 |
| 唤醒词（Wake word） | “Hey assistant” | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| 端点检测（End-pointing） | 回合结束 | VAD + 最短静音判断用户已说完。 |
| 预卷（Pre-roll） | 语音前缓冲区 | 在 VAD 触发前保留 200–400 ms 音频，避免首词截断。 |
| 工具调用（Tool call） | 函数调用 | LLM 输出 JSON；运行时派发；结果回注到循环中。 |

## 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) —— 生产级参考。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) —— 适合 DIY 的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) —— 托管式原生语音路线。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) —— 全双工参考（第 15 课）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) —— 唤醒词门控。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) —— LLM 函数调用。
