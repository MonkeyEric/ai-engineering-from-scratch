# 语音识别（ASR）—— CTC、RNN-T、注意力

> 语音识别本质上是每个时间步的音频分类，再由一个懂得英语和静音的序列模型把它们拼接起来。CTC、RNN-T 和注意力是三种实现方式。选一种并理解为什么。

**类型：** Build
**语言：** Python
**前置知识：** 阶段 6 · 02（频谱图与 Mel 滤波）、阶段 5 · 08（用于文本的 CNN 与 RNN）、阶段 5 · 10（注意力机制）
**时间：** 约 45 分钟

## 问题定义

你有一段 10 秒、16 kHz 的音频片段，想要得到字符串："turn on the kitchen lights"。挑战在于结构性问题：音频帧与字符并非一一对应。单词 "okay" 可能耗时 200 毫秒，也可能 1200 毫秒。静音穿插在话语中。某些音素比其他音素更长。输出词元数量事先未知。

三种建模方式解决了这个问题：

1. **CTC（Connectionist Temporal Classification，连接时序分类）。** 每帧输出词元概率，包含一个特殊的 *空白符（blank）*。解码时折叠重复词元并移除空白符。非自回归（non-autoregressive），速度快。代表模型：wav2vec 2.0、MMS。
2. **RNN-T（Recurrent Neural Network Transducer，循环神经网络转导器）。** 联合网络（joint network）根据编码器帧与前面的词元预测下一个词元。可流式（streaming）推理。代表：Google 端侧 ASR、NVIDIA Parakeet。
3. **注意力编码器-解码器（attention encoder-decoder）。** 编码器将音频压缩为隐状态，解码器通过交叉注意力（cross-attention）自回归地生成词元。代表：Whisper、SeamlessM4T。

到 2026 年，LibriSpeech test-clean 上的 SOTA 词错误率（WER）为 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。数字差距很小，部署差异却很大。

## 核心概念

![三种 ASR 建模方式：CTC、RNN-T、注意力编码器-解码器](../assets/asr-formulations.svg)

**CTC 直观理解。** 让编码器输出 `T` 个帧级分布，覆盖 `V+1` 个词元（V 个字符 + 空白符）。对于长度为 `U < T` 的目标字符串 `y`，任何能折叠为 `y` 的帧对齐方式都计入。CTC 损失对所有这类对齐求和。推理：逐帧取最大概率，折叠重复，移除空白符。

优点：非自回归、可流式、零前瞻。缺点：*条件独立假设*——每一帧的预测相互独立，因此没有内部语言模型（language model，LM）。可通过束搜索（beam search）或浅层融合（shallow fusion）引入外部 LM 来弥补。

**RNN-T 直观理解。** 增加一个 *预测器（predictor）* 网络嵌入词元历史，以及一个 *联合器（joiner）* 把预测器状态与编码器帧结合成 `V+1` 上的联合分布（`+1` 代表空/不输出，null / no-emit）。显式建模了 CTC 忽略的条件依赖。可流式，因为每一步只依赖过去的帧和过去的词元。

优点：可流式 + 内部语言模型。缺点：训练更复杂、更耗显存（三维损失网格）；RNN-T 损失核函数本身就是一个独立的库类别。

**注意力编码器-解码器。** 编码器（6-32 层 Transformer）作用于 log-mel 帧。解码器（6-32 层 Transformer）通过交叉注意力到编码器输出，自回归生成词元。没有对齐约束——注意力可以看音频中的任意位置。除非限制注意力范围，否则无法流式（如 2024 年的分块 Whisper-Streaming）。

优点：离线 ASR 质量最高，可用标准 seq2seq 工具训练。缺点：自回归延迟与输出长度成正比；要流式需额外工程。

### WER：唯一指标

**词错误率（Word Error Rate，WER）** = `(S + D + I) / N`，其中 S=替换数，D=删除数，I=插入数，N=参考文本词数。它等价于词级别的莱文斯坦编辑距离（Levenshtein edit distance）。越低越好。WER 超过 20% 通常不可用；低于 5% 对朗读语音可接近人类水平。2026 年标准基准数据如下：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 规模 |
|------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些都是编码器-解码器或 RNN-T 架构。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8–2.1%。

## 动手实现

### 步骤 1：贪婪 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 每帧概率向量的列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复，丢弃空白符。例如：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：CTC 束搜索

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (词元序列, 对数概率)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境会使用带 LM 融合的前缀树束搜索（prefix tree beam search）；这里只是概念骨架。

### 步骤 3：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤 4：用 Whisper 推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强的通用 ASR 只需一行代码。在 24 GB GPU 上可达约 20 倍实时速度。

### 步骤 5：用 Parakeet 或 wav2vec 2.0 流式识别

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和状态传递；请使用支持流式的库（NeMo 用于 Parakeet，`transformers` pipeline 的 `chunk_length_s`）。

## 如何选择

2026 年选型表：

| 场景 | 选择 |
|-----------|------|
| 英语、离线、追求最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒性 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 端侧、移动端、<500 ms 延迟 | Whisper-Tiny 量化版或 Moonshine（2024） |
| 长音频 | 基于 VAD 分块的 Whisper（WhisperX） |
| 领域专用（医学、法律） | 微调（fine-tune）wav2vec 2.0 + 领域 LM 融合 |

## 2026 年仍在犯的陷阱

- **没有 VAD。** 对静音运行 Whisper 会产生幻觉（"Thanks for watching!"）。始终用语音活动检测（VAD）把关。
- **字符/词/子词 WER 混淆。** 应在文本规范化（normalization，小写、去标点）后报告词级别 WER。
- **语言识别（LID）漂移。** Whisper 的自动语言识别会把噪声片段错判为日语或威尔士语；确定语言时强制指定 `language="en"`。
- **长音频不分块。** Whisper 的窗口为 30 秒；更长音频请使用 `chunk_length_s=30, stride=5`。

## 交付

保存为 `outputs/skill-asr-picker.md`。为给定部署目标选择模型、解码策略、分块方式和 LM 融合方案。

## 练习

1. **简单。** 运行 `code/main.py`。它会对手工构造的 CTC 输出做贪婪解码，并相对参考文本计算 WER。
2. **中等。** 正确实现步骤 2 中的前缀树束搜索（正确处理空白合并规则）。在 10 个合成样例上与贪婪解码对比。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上用 `whisper-large-v3-turbo` 计算前 100 句的 WER，并与已发表数字对比。

## 关键术语

| 术语 | 大家的说法 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | 空白符损失 | 对所有帧到词元对齐求边缘概率；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一个词元预测器；处理词序依赖。 |
| 注意力编码器-解码器 | Whisper 风格 | 编码器 + 交叉注意力解码器；离线质量最佳。 |
| WER | 报告的那个数 | 词级别 `(S+D+I)/N`。 |
| Blank | 空 | CTC 中的特殊词元，表示“本帧不输出”。 |
| LM 融合 | 外部语言模型 | 在束搜索中加权加上语言模型的对数概率。 |
| VAD | 静音门 | 语音活动检测器；裁剪非语音部分。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) —— RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) —— 2022 年经典论文；v3-turbo 为 2024 年扩展。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) —— 2026 年 Open ASR Leaderboard 榜首。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) —— 25+ 模型的实时基准。
