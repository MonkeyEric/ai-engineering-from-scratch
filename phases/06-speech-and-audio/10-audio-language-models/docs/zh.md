# 音频-语言模型 —— Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频-语言模型已经能够对语音、环境音和音乐进行联合推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上追平了 GPT-4o Audio；Audio Flamingo Next 在 LongAudioBench 上超越了 Gemini 2.5 Pro。开源与闭源之间的差距已基本消失——但在多音频任务上，各家都接近随机水平。

**类型：** 学习  
**语言：** Python  
**先修知识：** Phase 6 · 04（自动语音识别 ASR）、Phase 12 · 03（视觉-语言模型）、Phase 7 · 10（音频 Transformer）  
**时长：** 约 45 分钟

## 问题背景

你有一段 5 秒的音频：先是狗叫，然后有人喊“stop！”，最后安静下来。围绕这段音频，有用的提问可以横跨多个维度：

- **转录（transcription）。** “说了什么？”——属于自动语音识别（ASR）的范畴。
- **语义推理（semantic reasoning）。** “这个人有危险吗？”——需要同时理解狗叫、喊叫和安静所共同表达的语境。
- **音乐推理（music reasoning）。** “演奏旋律的有哪些乐器？”
- **长音频检索（long-audio retrieval）。** “在这场 90 分钟的讲座里，讲师在哪里讲解了梯度下降？”

一个模型、一条提示就能回答以上所有问题，这就是**音频-语言模型（audio-language model，简称 LALM / ALM）**。它与纯 ASR 不同：LALM 生成自由格式的自然语言答案，而不只是文本转录。

## 核心概念

![音频-语言模型：音频编码器 + 投影器 + 大语言模型解码器](../assets/alm-architecture.svg)

### 三组件模板

2026 年的所有 LALM 都拥有相同的骨架：

1. **音频编码器（audio encoder）。** Whisper 编码器、BEATs、CLAP、WavLM，或各模型自研的编码器。
2. **投影器（projector）。** 线性层或 MLP，用于把音频编码器的特征映射到大语言模型（LLM）的词元嵌入（token embedding）空间。
3. **大语言模型（LLM）。** 基于 Llama / Qwen / Gemma 的解码器。它接收交错的文本 + 音频词元，输出文本。

训练通常分为三个阶段：

- **阶段 1。** 冻结编码器和 LLM，仅用 ASR / 音频描述（captioning）数据训练投影器。
- **阶段 2。** 在指令遵循型的音频任务（问答、推理、音乐理解）上进行全参数或 LoRA 微调（fine-tune）。
- **阶段 3（可选）。** 增加语音输入 / 语音输出（voice-in / voice-out）能力，引入语音解码器。Qwen2.5-Omni 和 AF3-Chat 都采用了这一做法。

### 2026 年模型速览

| 模型 | 骨干网络 | 音频编码器 | 输出模态 | 获取方式 |
|------|----------|------------|----------|----------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自研 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自研 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业许可 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业许可 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 2026 基准测试现实检验

**MMAU-Pro。** 1800 个问答对，覆盖语音、声音、音乐、混合音频，并包含多音频子集。

| 模型 | 整体 | 语音 | 声音 | 音乐 | 多音频 |
|------|------|------|------|------|--------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench 最前沿 | — | — | — | — |

**多音频列对所有人来说都是一记警钟。** 四选一随机猜测的准确率是 25%，而大多数模型就在这个水平附近。LALM 仍然难以完成两段音频的比较。

### 2026 年 LALM 的适用场景

- **呼叫中心录音合规审计。** “客服是否提到了必需的免责声明？”
- **无障碍辅助。** 为听障用户描述声音事件，而不仅仅是转录。
- **内容审核。** 同时检测暴力语言、威胁语气和背景环境。
- **播客 / 会议章节划分。** 基于语义摘要，而不仅仅是说话人转换。
- **音乐目录分析。** “找出所有 B 段转调的曲目。”

### 哪些场景（尚）不适用

- 细粒度音乐理论分析（低于和弦级别）。
- 长时间对话中的说话人归属推理（超过 10 分钟后性能下降）。
- 多音频比较（22–26% 仅略高于随机）。
- 实时流式推理（大多数模型仍是离线批量推理）。

## 动手实现

### 步骤 1：调用 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步骤 2：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就这么简单。投影器通常只有 1–3 层线性层。用 ASR 数据对（音频 → 转录文本）训练它，就是阶段 1 的预训练任务。

### 步骤 3：在 MMAU / LongAudioBench 上评测

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

请按类别（语音 / 声音 / 音乐 / 多音频）分别报告结果。聚合数字会掩盖模型的真实短板。

## 如何使用

| 任务 | 2026 年推荐 |
|------|-------------|
| 开放式音频问答（开源） | Qwen2.5-Omni-7B |
| 最佳开源长音频模型 | Audio Flamingo Next |
| 最佳闭源模型 | Gemini 2.5 Pro |
| 语音输入 / 语音输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（针对音乐优化的 AF-CLAP） |
| 呼叫中心审计 | 通过 API 使用 Gemini 2.5 Pro，并对政策文档做 RAG |

## 常见陷阱

- **过度信任多音频能力。** 如果你的任务需要判断“哪段音频包含 X”，那么随机水平附近的性能是真实存在的风险。
- **长音频性能衰减。** 超过 10 分钟后，大多数模型的说话人归属能力会崩。先做人声分离（第 6 课），再做摘要。
- **静音幻觉。** 使用 Whisper 编码器的 LALM 会继承 Whisper 的同类问题。建议用语音活动检测（VAD）进行门控。
- **基准测试结果挑选。** 厂商博客往往只展示表现最好的类别。请自己跑一遍 MMAU-Pro 的多音频子集。

## 交付物

保存为 `outputs/skill-alm-picker.md`。针对一个给定的音频理解任务，选择合适的 LALM、基准子集以及输出模态（文本还是语音）。

## 练习

1. **简单。** 运行 `code/main.py`，观察一个玩具级投影器模式，以及伪 LALM 如何将（音频嵌入，文本词元）路由为输出词元。
2. **中等。** 在 100 条 MMAU-Pro 语音样本上评测 Qwen2.5-Omni-7B，并与论文报告的数字对比。
3. **困难。** 构建一个最简音频描述基线：BEATs 编码器 + 2 层投影器 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影器，并与 SALMONN 在 Clotho-AQA 上对比。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| LALM | 音频版 ChatGPT | 音频编码器 + 投影器 + 大语言模型解码器。 |
| Projector | 适配器 | 把音频特征映射到 LLM 嵌入空间的小型 MLP。 |
| MMAU | 那个基准 | 覆盖语音、声音、音乐的 1 万个音频问答对。 |
| MMAU-Pro | 更难的 MMAU | 1800 道侧重多音频与推理的题目。 |
| LongAudioBench | 长音频评测 | 面向多分钟音频的语义查询评测。 |
| Voice-in / voice-out | 原生语音 | 模型直接接收语音并输出语音，不经过文本中转。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) —— 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) —— 语音输入 / 语音输出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) —— 开源长音频领先模型。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) —— LongAudioBench 最前沿。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) —— 双编码器先驱。
- [MMAU-Pro 排行榜](https://mmaubenchmark.github.io/) —— 2026 年实时排名。
