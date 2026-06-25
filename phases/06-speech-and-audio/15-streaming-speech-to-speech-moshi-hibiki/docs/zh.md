# 流式语音到语音 —— Moshi、Hibiki 与全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 交付了一个单一模型，能够以 200 毫秒延迟同时听和说。Hibiki 则逐块进行语音到语音翻译。两者都抛弃了 ASR → LLM → TTS 流水线，转而采用基于 Mimi 编解码器 token 的统一全双工架构。这是新的参考设计。

**类型：** 学习
**语言：** Python
**前置知识：** 第 6 阶段 · 13（神经音频编解码器），第 6 阶段 · 11（实时音频），第 7 阶段 · 05（完整 Transformer）
**时间：** 约 75 分钟

## 问题所在

任何基于第 11 课和第 12 课构建的语音智能体都有一个约 300-500 毫秒的基本延迟下限：VAD 触发、STT 处理、LLM 推理、TTS 生成。每个阶段都有其自身的最小延迟。你可以优化和并行化，但流水线的形态限制了你。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果没有流水线会怎样？如果一个模型持续地直接接收音频并输出音频，将文本作为中间的“内心独白（inner monologue）”而非必需阶段，会怎样？

答案是**全双工语音到语音（full-duplex speech-to-speech）**。理论延迟为 160 毫秒（80 毫秒 Mimi 帧 + 80 毫秒声学延迟）。在单张 L4 GPU 上的实际延迟为 200 毫秒。这比一流的流水线语音智能体快了一倍。

## 概念

![Moshi 架构：两条并行的 Mimi 流 + 内心独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。**两条 Mimi 编解码器流，均为 12.5 Hz × 8 码本：

- 流 1：用户音频（经 Mimi 编码，持续到达）
- 流 2：Moshi 自身生成的音频（由 Moshi 生成）

**Transformer。**一个 70 亿参数的时间 Transformer 处理两条流和一个文本“内心独白”流。在每个 80 毫秒步，它：

1. 接收最新的用户 Mimi token（8 个码本）。
2. 接收最近的 Moshi Mimi token（8 个码本，按生成顺序）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 生成下一个 Moshi Mimi token（通过一个小型深度 Transformer 生成 8 个码本）。

三条流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话的同时听到用户；可以在用户打断时自我打断；可以在不中断主要话语的情况下进行反向通道反馈（“嗯嗯”）。

**深度 Transformer（Depth Transformer）。**在单个帧内，8 个码本不是并行预测的——它们之间存在码本间依赖关系。一个小型 2 层“深度 Transformer”在 80 毫秒内依次预测它们。这是自回归编解码器语言模型（AR codec LM）的标准分解方式（VALL-E、VibeVoice 也使用了类似方法）。

### 为什么内心独白文本有帮助

如果没有显式文本，模型必须在其声学流中隐式地建模语言。Moshi 的洞见是：强制模型在生成音频的同时输出文本 token。文本流本质上是 Moshi 所说内容的转录。这提高了语义一致性，使替换语言模型头部（language model head）更容易，并且可以免费获得转录文本。

### Hibiki：流式语音到语音翻译

架构相同，在翻译对上训练。源语言音频输入，目标语言音频持续输出。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——它使用句子级数据 + GRPO 强化学习来优化延迟。

最初支持四个语言对；可以用约 1000 小时数据适配到新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** —— 全双工对话（法语优先，英语支持良好）
- **Hibiki / Hibiki-Zero** —— 同声语音翻译
- **Kyutai STT** —— 流式 ASR（500 毫秒或 2.5 秒前瞻）
- **Kyutai Pocket TTS** —— 1 亿参数 TTS，可在 CPU 上运行（2026 年 1 月）
- **Unmute** —— 在公共服务器上整合上述组件的完整流水线

在 L40S GPU 上的吞吐量：64 个并发会话，速度为实时 3 倍。

### Sesame CSM —— 近亲

Sesame CSM（2025）使用了类似思路——Llama-3 主干 + Mimi 编解码器头部。但 CSM 是单向的（接收上下文 + 文本，生成语音），而非全双工。它是市场上最好的“语音存在感”TTS；与 Moshi 的全双工能力并不完全相同。

### 2026 年性能数据

| 模型 | 延迟 | 用例 | 许可证 |
|-------|---------|----------|---------|
| Moshi | 200 毫秒（L4）| 全双工英语 / 法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 同上 | 5 个语言对，无需对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 毫秒 TTFA | 上下文条件化 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 毫秒 | 闭源，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 毫秒 | 闭源，Google API | 商业 |

## 动手构建

### 步骤 1：接口

Moshi 暴露了一个 WebSocket 服务器，接收 80 毫秒的 Mimi 编码音频块，并返回 80 毫秒的 Mimi 编码音频块。双向、持续进行。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步骤 2：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### 步骤 3：训练目标（概念性）

对于每个 80 毫秒帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后是 `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内心独白）；音频在深度 Transformer 内按码本顺序预测。

### 步骤 4：Moshi 的优势与局限

Moshi 的优势：

- 在廉价硬件上实现低于 250 毫秒的端到端延迟。
- 自然的反向通道反馈和打断。
- 无需流水线胶水代码。

Moshi 的局限：

- 工具调用（未针对该任务训练；需要单独的 LLM 路径）。
- 长程推理（Moshi 是一个约 80 亿参数的对话模型，不是 Claude/GPT-4）。
- 小众主题的事实准确性。
- 大多数企业生产用例（2026 年仍在使用流水线）。

## 如何使用

| 场景 | 选择 |
|-----------|------|
| 最低延迟的语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示 / 研究 | Moshi、CSM |
| 带工具的企业智能体 | 流水线（第 12 课），而非 Moshi |
| 上下文中的定制语音 TTS | Sesame CSM |
| 任意语言语音到语音 | GPT-4o Realtime 或 Gemini 2.5 Live（商业）|

## 常见陷阱

- **有限的工具调用。**Moshi 是对话模型，不是智能体框架。如需工具，请与流水线结合使用。
- **特定音色条件化。**Moshi 使用单一训练好的角色；音色克隆需要单独的训练运行。
- **语言覆盖。**法语 + 英语表现优秀；其他语言有限。Hibiki-Zero 有帮助，但仍需训练数据。
- **资源成本。**一个完整的 Moshi 会话占用一个 GPU 槽位；这不是廉价的共享租户部署模式。

## 交付

保存为 `outputs/skill-duplex-pipeline.md`。为某个语音智能体工作负载选择流水线架构或全双工架构，并说明理由。

## 练习

1. **简单。**运行 `code/main.py`。它以符号方式模拟双流 + 内心独白架构。
2. **中等。**从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户语音结束到 Moshi 响应开始的挂钟延迟。
3. **困难。**拿你在第 12 课的流水线智能体，与 Moshi 在 20 条匹配的测试语句上比较 P50 延迟。写下流水线在架构上仍然获胜的情况。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| 全双工（Full-duplex）| 同时听和说 | 同一模型上两个音频流同时活跃。|
| 内心独白（Inner monologue）| 模型的文本流 | Moshi 在输出音频的同时 emit 文本 token。|
| 深度 Transformer（Depth transformer）| 码本间预测器 | 在单个 80 毫秒帧内预测 8 个码本的小型 Transformer。|
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 码本；语义 + 声学；为 Moshi 提供动力。|
| 流式 S2S（Streaming S2S）| 实时音频 → 音频 | 逐块翻译/对话，无流水线阶段。|
| 反向通道反馈（Back-channeling）| “嗯嗯”式回应 | Moshi 可以在不中断其回合的情况下发出简短确认。|

## 延伸阅读

- [Défossez 等人（2024）。Moshi —— 语音-文本基础模型](https://arxiv.org/html/2410.00037v2) —— 论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) —— 无需对齐数据的流式翻译。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) —— CSM 规格。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) —— 安装与服务器。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) —— 闭源商业竞品。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) —— 底层 STT/TTS 框架。
