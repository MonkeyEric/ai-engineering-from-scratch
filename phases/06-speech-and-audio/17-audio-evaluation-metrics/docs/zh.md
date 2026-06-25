# 音频评估 —— WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 无法度量，就无法交付。本课列出 2026 年各类音频任务的核心指标：自动语音识别（ASR）的 WER、CER、RTFx，文本转语音（TTS）的 MOS、UTMOS、SECS、ASR 回环 WER，音频语言模型（audio-language）的 MMAU、LongAudioBench，音乐生成的 FAD、CLAP，以及声纹验证的 EER。还会介绍用于横向对比的开放排行榜。

**类型：** 学习  
**语言：** Python  
**前置知识：** Phase 6 · 04、06、07、09、10；Phase 2 · 09（模型评估）  
**时间：** 约 60 分钟

## 问题背景

每个音频任务都有多个指标，分别衡量不同的维度。选错指标会导致模型在面板上很漂亮，上线后却很糟糕。2026 年的权威清单如下：

| 任务 | 主要指标 | 次要指标 |
|------|---------|---------|
| ASR | WER | CER · RTFx · 首个词元延迟（first-token latency） |
| TTS | MOS / UTMOS | SECS · ASR 回环 WER · CER · TTFA |
| 语音克隆（Voice cloning） | SECS（ECAPA 余弦） | MOS · CER |
| 声纹验证（Speaker verification） | EER | minDCF · 工作点上的 FAR / FRR |
| 说话人日志（Diarization） | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | 宏平均 F1（macro F1） · 每类召回率 |
| 音乐生成 | FAD | CLAP · 听测小组 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式语音到语音（Streaming S2S） | 延迟 P50/P95 | WER · MOS |

## 核心概念

![音频评估矩阵 —— 指标、任务与 2026 年排行榜](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错误率，Word Error Rate）。** `(S + D + I) / N`。评分前先转小写、去掉标点，并将数字归一化。可使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。&lt; 5% 即接近人类水平的朗读语音识别。

**CER（字符错误率，Character Error Rate）。** 公式与 WER 相同，但按字符计算。适用于汉语、粤语等声调语言，因为这类语言的词边界不明确。

**RTFx（实时因子倒数，inverse real-time factor）。** 每秒钟墙钟时间能处理的音频秒数。越高越好。Parakeet-TDT 可达 3380×，Whisper-large-v3 约 30×。

**首个词元延迟（First-token latency）。** 从音频输入到输出第一个转录词元的墙钟时间。对流式 ASR 至关重要。Deepgram Nova-3 约 150 ms。

### TTS 指标

**MOS（平均意见得分，Mean Opinion Score）。** 1-5 分的人工评分。黄金标准但速度慢。每个样本至少需要 20 名听众，每个模型至少需要 100 个样本。

**UTMOS（2022-2026）。** 基于学习的 MOS 预测器。在标准基准上与人工 MOS 的相关性约 0.9。F5-TTS：UTMOS 3.95；真实录音：4.08。

**SECS（说话人编码器余弦相似度，Speaker Encoder Cosine Similarity）。** 用于语音克隆。计算参考音频与克隆输出在 ECAPA 嵌入上的余弦相似度。&gt; 0.75 表示克隆出的说话人可辨识。

**ASR 回环 WER（WER-on-ASR-round-trip）。** 对 TTS 输出运行 Whisper，再与输入文本计算 WER。用于发现可懂度退化。2026 年 SOTA：CER &lt; 2%。

**TTFA（time-to-first-audio，首次输出音频时间）。** 墙钟延迟。Kokoro-82M 约 100 ms，F5-TTS 约 1 s。

### 语音克隆专属指标

**SECS + MOS + CER** 三位一体。SECS 高但 MOS 低意味着音色对了但不够自然；反过来则表示声音自然但说话人不对。

### 声纹验证

**EER（等错误率，Equal Error Rate）。** 错误接受率（FAR）等于错误拒绝率（FRR）时的阈值。ECAPA 在 VoxCeleb1-O 上为 0.87%。

**minDCF（最小检测代价，min Detection Cost）。** 在选定工作点（通常是 FAR=0.01）下的加权代价。比 EER 更贴近生产实际。

### 说话人日志

**DER（说话人日志错误率，Diarization Error Rate）。** `(FA + Miss + Confusion) / total_speaker_time`。漏检、误检和说话人混淆各占总说话人时长的比例。AMI 会议场景下 DER 10-20% 是现实水平；pyannote 3.1 + Precision-2 商用方案在录音质量良好的音频上可达 DER &lt; 10%。

**JER（Jaccard 错误率，Jaccard Error Rate）。** DER 的替代指标，对短片段偏置更鲁棒。

### 音频分类

多标签任务：**mAP（平均精度均值，mean Average Precision）** 跨所有类别计算。AudioSet 上 BEATs-iter3 为 0.548 mAP。

互斥多分类任务：**top-1、top-5 准确率**。Speech Commands v2 上 Audio-MAE 的 top-1 为 99.0%。

类别不平衡任务：**宏平均 F1（macro F1）** + **每类召回率**。要按类别报告——总体准确率会掩盖哪些类别失败。

### 音乐生成

**FAD（弗雷歇音频距离，Fréchet Audio Distance）。** 真实音频与生成音频在 VGGish 嵌入分布上的距离。MusicGen-small 在 MusicCaps 上为 4.5，MusicLM 为 4.0。越低越好。

**CLAP 分数（CLAP Score）。** 使用 CLAP 嵌入计算的文本-音频对齐分数。&gt; 0.3 表示对齐尚可。

**听测小组 MOS（Listening panel MOS）。** 消费级音乐质量的最终裁判。Suno v5 在 TTS Arena（来自成对人类偏好投票）上的 ELO 为 1293。

### 音频语言模型基准

**MMAU（Massive Multi-Audio Understanding，大规模多音频理解）。** 1 万条音频问答对。

**MMAU-Pro。** 1800 道难题，分四类：语音 / 声音 / 音乐 / 多音频。四选一的随机概率为 25%。Gemini 2.5 Pro 总体约 60%；多音频类别所有模型约 22%。

**LongAudioBench。** 包含数分钟音频片段和语义查询。Audio Flamingo Next 超过 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 音频描述基准。使用 SPICE、CIDEr、FENSE 等指标。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户说完到助手首次发出可听响应的墙钟时间。Moshi 为 200 ms，GPT-4o Realtime 为 300 ms。

**输出端的 WER / MOS。**

**打断响应速度（Barge-in responsiveness）。** 从用户打断到助手静音的时间。目标 &lt; 150 ms。

### 2026 年排行榜

| 排行榜 | 跟踪内容 | 网址 |
|------------|--------|-----|
| Open ASR Leaderboard（HF） | 英语 + 多语言 + 长音频 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，成对投票 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | 大音频语言模型（LALM）推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 声纹识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐大音频语言模型 | （MMAU 内） |
| HEAR benchmark | 自监督音频表示 | `hearbenchmark.com` |

## 动手实现

### 第 1 步：带归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# 约 0.17
```

### 第 2 步：TTS 回环 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 第 3 步：语音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 第 4 步：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 第 5 步：声纹验证的 EER（与第 6 课代码相同）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用方法

每次部署都要搭配固定的评估 harness，并在每次模型更新时运行。三条基本原则：

1. **评分前先归一化。** 转小写、去标点、展开数字。并在报告中写明所用的归一化规则。
2. **报告分布，而不是只报平均值。** 延迟报 P50/P95/P99，分类报每类召回率，MMAU 报每类准确率。
3. **至少跑一个权威公开基准。** 即便生产数据不同，在 Open ASR、TTS Arena 或 MMAU 上报告结果，也能让 reviewers 做同类比较。

## 常见陷阱

- **UTMOS 外推。** 它在 VCTK 风格的干净语音上训练，对带噪、克隆或情感语音打分偏差较大。
- **MOS 听测偏置。** 20 名 Amazon Mechanical Turk 工人 ≠ 20 名目标用户。若 stakes 高，应付费招募领域对口的面板。
- **FAD 依赖参考集。** 不同模型之间必须使用同一参考分布进行对比。
- **总体 WER 掩盖问题。** 整体 5% 的 WER 可能掩盖某些口音群体上 30% 的 WER。应按人口统计学切片报告。
- **公开基准饱和。** 前沿模型在标准基准上已接近天花板。应建立能反映真实业务流量的内部留存测试集。

## 交付物

保存为 `outputs/skill-audio-evaluator.md`。为任意音频模型发布挑选指标、基准和报告格式。

## 练习题

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / 简化版 FAD / 简化版 MMAU。
2. **中等。** 搭建一个 TTS 回环 WER harness。将 Kokoro 或 F5-TTS 的输出通过 Whisper 跑一遍，对 50 条提示词计算 WER，并标记 WER &gt; 10% 的提示词。
3. **困难。** 在 MMAU-Pro 的语音子集和多音频子集上（各 50 题）给第 10 课选定的音频语言模型打分。报告每类准确率，并与公开数字对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|---------|
| WER | ASR 分数 | 归一化后的词级 `(S+D+I)/N`。 |
| CER | 字符级 WER | 用于声调语言或字符级系统。 |
| MOS | 人工评分 | 1-5 分；20+ 名听众 × 100 个样本。 |
| UTMOS | 机器 MOS 预测 | 学习模型；与人工 MOS 相关性约 0.9。 |
| SECS | 语音克隆相似度 | 参考音频与克隆音频的 ECAPA 余弦相似度。 |
| EER | 声纹验证分数 | FAR 等于 FRR 时的阈值。 |
| DER | 说话人日志分数 | `(FA + Miss + Confusion) / total`。 |
| FAD | 音乐生成质量 | VGGish 嵌入上的弗雷歇距离。 |
| RTFx | 吞吐率 | 每秒钟墙钟时间处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) —— 支持归一化工具的 WER/CER 库。
- [UTMOS（Saeki 等，2022）](https://arxiv.org/abs/2204.02152) —— 基于学习的 MOS 预测器。
- [Fréchet Audio Distance（Kilgour 等，2019）](https://arxiv.org/abs/1812.08466) —— 音乐生成的事实标准。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) —— 2026 年实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) —— 人工投票的 TTS 排行榜。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) —— 音频语言模型推理排行榜。
- [HEAR benchmark](https://hearbenchmark.com/) —— 音频自监督学习基准。
