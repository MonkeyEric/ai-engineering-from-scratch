# Whisper — 架构与微调

> Whisper 是一个 30 秒窗口的 Transformer 编码器-解码器（transformer encoder-decoder），在 68 万小时多语言弱监督音频-文本配对数据上训练。一套架构、多种任务、覆盖 99 种语言的稳健性。2026 年的 ASR（自动语音识别，Automatic Speech Recognition）参考基准。

**Type:** 实战构建  
**Languages:** Python  
**Prerequisites:** 阶段 6 · 04（ASR）、阶段 5 · 10（注意力机制）、阶段 7 · 05（完整 Transformer）  
**Time:** 约 75 分钟

## 问题背景

Whisper 由 OpenAI 于 2022 年 9 月发布，是第一个以“商品化”形态交付的 ASR 模型：粘贴音频即可得到文本，支持 99 种语言，抗噪，还能在笔记本上运行。到 2024 年，OpenAI 已推出 Large-v3 和 Turbo 变体；到 2026 年，Whisper 已成为从播客转录、语音助手到 YouTube 字幕等所有场景的默认基线。

但你不能永远把 Whisper 当作黑箱流水线。领域迁移（domain shift）会毁掉它——技术术语、说话人口音、专有名词、短片段、静音。你需要知道：

1. 它内部究竟是什么。
2. 如何正确地给它分块、流式或长音频输入。
3. 何时以及如何微调（fine-tune）。

## 核心概念

![Whisper 编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构。** 标准的 Transformer 编码器-解码器（transformer encoder-decoder）。

- 输入：30 秒对数梅尔频谱图（log-mel spectrogram），80 个 mel 滤波器、10 ms 帧移 → 3000 帧。更短的片段补零填充，更长的片段切分。
- 编码器（Encoder）：卷积下采样（步幅 2）+ `N` 个 Transformer 块。Large-v3：32 层、1280 维、20 头。
- 解码器（Decoder）：`N` 个 Transformer 块，带因果自注意力（causal self-attention）和对编码器输出的交叉注意力（cross-attention）。尺寸与编码器相同。
- 输出：基于 51,865 词元词表（vocab）的 BPE 词元（BPE tokens）。

Large-v3 有 15.5 亿参数。Turbo 将解码器从 32 层减至 4 层，延迟降低 8 倍，词错误率（WER）仅上升不到 1%。

**提示格式（prompt format）。** Whisper 是一个多任务模型，通过解码器提示中的特殊词元（special tokens）进行控制：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签；决定是翻译还是转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 将任意语言输入翻译成英文输出，或逐字转录。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

正是这种提示格式让同一个模型能完成多种任务。把 `<|en|>` 改成 `<|fr|>`，它就会转录法语。

**30 秒窗口。** 所有输入都固定为 30 秒。长音频需要切分；短音频会被填充。原生不支持流式——这正是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数梅尔归一化（log-mel normalization）。** `(log_mel - mean) / std`，其中统计量来自 Whisper 自己的训练语料。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的模型变体

| 变体 | 参数量 | 延迟（A100） | 词错误率（LibriSpeech-clean） |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× 实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 流式 | 2.0% |

### 微调（Fine-tuning）

2026 年的标准流程：

1. 收集 10–100 小时目标领域音频及对齐转录文本。
2. 使用 `transformers.Seq2SeqTrainer` 并配合 `generate_with_loss` 回调。
3. 参数高效微调：在注意力层（attention layers）的 `q_proj`、`k_proj`、`v_proj` 上应用 LoRA（低秩自适应，Low-Rank Adaptation），可将 GPU 显存降低 4 倍，WER 成本小于 0.3%。
4. 如果数据少于 10 小时，冻结编码器，只微调解码器。
5. 始终使用 Whisper 自己的分词器（tokenizer）和提示格式；不要更换分词器。

社区结果：在 20 小时医学听写数据上微调 Medium 模型，医学词汇上的 WER 从 12% 降至 4.5%。在 4 小时冰岛语数据上微调 Turbo 模型，WER 从 18% 降至 6%。

## 动手实现

### 步骤 1：开箱即用运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防止失控重复
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终覆盖的关键默认值：`temperature=0.0`（采样默认会从 0.0 → 0.2 → 0.4 … 回退链），`condition_on_previous_text=False`（防止级联幻觉问题），以及 `no_speech_threshold=0.6`（静音检测）。

### 步骤 2：长音频分块

```python
# whisperx 是 2026 年带词级时间戳的长音频参考工具
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了（1）Silero VAD 门控，（2）通过 wav2vec 2.0 实现的词级对齐，（3）通过 `pyannote.audio` 实现的说话人分离（diarization）。2026 年生产级转录的主力工具。

### 步骤 3：使用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> 约 300 万可训练 / 8.09 亿总计
```

然后使用标准 Trainer 循环。每 1000 步保存检查点。在留出集上用 WER 评估。

### 步骤 4：检查每一层学到了什么

```python
# 在解码过程中提取交叉注意力权重，观察解码器关注何处。
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: 层 × 头 × 步 × 源长度
```

用热图（heatmap）可视化——你会看到对角线对齐，因为解码器步长在扫描编码器帧。这条对角线就是 Whisper 对词时间戳的理解。

## 如何使用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英语，离线 | 通过 `whisperx` 使用 Large-v3-turbo |
| 移动 / 边缘端 | Whisper-Tiny 量化（int8）或 Moonshine |
| 多语言长音频 | 通过 `whisperx` 使用 Large-v3 + 说话人分离 |
| 低资源语言 | 使用 LoRA 微调 Medium 或 Turbo |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐）|

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原版快 4 倍，输出一致。

## 2026 年仍会踩到的坑

- **静音上的幻觉文本。** Whisper 在带字幕的数据上训练，会生成“Thanks for watching!”、“Subscribe!”、歌词等内容。调用前务必先进行 VAD（语音活动检测，Voice Activity Detection）门控。
- **`condition_on_previous_text` 级联。** 一处幻觉会污染后续窗口。除非需要跨块流畅性，否则设为 `False`。
- **短片段填充。** 2 秒片段填充到 30 秒可能在后续静音中产生幻觉。使用 `pad=False` 或 VAD 门控。
- **错误的 mel 统计量。** 使用 librosa 的 mel 而非 Whisper 的会导致输出接近随机。请使用 `whisper.audio.log_mel_spectrogram`。

## 交付成果

保存为 `outputs/skill-whisper-tuner.md`。为指定领域设计一个 Whisper 微调或推理流水线。

## 练习题

1. **简单。** 运行 `code/main.py`。它会将 Whisper 风格的提示词元化，计算解码形状预算，并打印 10 分钟片段的分块计划。
2. **中等。** 安装 `faster-whisper`，转录一段 10 分钟播客，与人工转录对比 WER。尝试 `language="auto"` 与强制 `language="en"`。
3. **困难。** 使用 HF `datasets`，挑选一个 Whisper 表现不佳的语言（如乌尔都语），用 LoRA 在 2 小时数据上微调 Medium 模型 2 个 epoch，并报告 WER 变化。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| 30 秒窗口 | Whisper 的限制 | 硬输入上限；长音频需切分。 |
| SOT | 转录开始 | `<|startoftranscript|>` 启动解码器提示。 |
| 时间戳词元 | 时间对齐 | 每 0.02 秒偏移都是 51k 词表中的一个特殊词元。 |
| Turbo | 快速变体 | 4 层解码器，快 8 倍，WER 退化 <1%。 |
| WhisperX | 长音频封装 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效微调 | 在注意力上添加低秩适配器；仅训练约 0.3% 参数。 |
| 幻觉 | 静默失败 | Whisper 会从噪声/静音中生成流利的英文。 |

## 扩展阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 原始架构与训练配方。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8 倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长音频、词级对齐、说话人分离。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2，快 4 倍。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — LoRA / 全量微调的标准教程。
