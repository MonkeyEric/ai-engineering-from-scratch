# 语音克隆与语音转换

> 语音克隆（voice cloning）能朗读文本并模仿他人音色。语音转换（voice conversion）则把你的声音改写为他人的声音，同时保留原内容。两者都依赖同一个基础：将说话人身份与内容分离。

**类型：** 构建  
**语言：** Python  
**前置：** Phase 6 · 06（说话人识别），Phase 6 · 07（文本转语音（TTS））  
**时间：** 约 75 分钟

## 问题背景

2026 年，只需 5 秒的音频片段，就能在消费级 GPU 上生成高质量的任意人声音克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 等产品都提供零样本（zero-shot）或少样本（few-shot）克隆。这项技术既是福音（无障碍 TTS、配音、辅助语音），也是武器（诈骗电话、政治深度伪造、知识产权盗窃）。

两个密切相关的任务：

- **语音克隆（TTS 侧）：** 文本 + 5 秒参考语音 → 该音色对应的音频。
- **语音转换（语音侧）：** 源音频（A 说 X）+ B 的参考语音 → B 说 X 的音频。

两者都将波形分解为（内容、说话人、韵律（prosody）），然后把一个来源的内容与另一个来源的说话人重新组合。

你如今在 2026 年发布时必须遵守的关键约束：**水印（watermark）和同意门控（consent gates）在欧盟（AI 法案，2026 年 8 月生效）和加利福尼亚州（AB 2905，2025 年生效）已属法律强制要求**。你的管线必须在输出中嵌入不可听水印，并拒绝非授权克隆。

## 核心概念

![语音克隆与转换：分解、替换说话人、重新组合](../assets/voice-cloning.svg)

**零样本克隆（zero-shot cloning）。** 将 5 秒片段输入一个已在数千名说话人上训练过的模型。说话人编码器（speaker encoder）将片段映射为说话人嵌入（speaker embedding）；文本转语音（TTS）解码器根据该嵌入和文本进行条件生成。

应用于：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调（few-shot fine-tuning）。** 录制 5–30 分钟目标语音。用 LoRA 对基础模型微调约一小时。质量从“还可以”跃升到“难以区分”。Coqui 与 ElevenLabs 都支持此模式；社区也将其用于 F5-TTS。

**语音转换（VC）。** 两大流派：

- **识别-合成。** 运行类似自动语音识别（ASR）的模型提取内容表征（如软音素后验、PPG），再用目标说话人嵌入重新合成。对语言和口音鲁棒。应用于 KNN-VC（2023）、Diff-HierVC（2023）。
- **解耦（disentanglement）。** 训练一个自编码器（autoencoder），在瓶颈隐空间（latent space）中分离内容、说话人和韵律（prosody）。推理时替换说话人嵌入。质量较低但速度更快。应用于 AutoVC（2019）、VITS-VC 变体。

**基于神经编解码器（neural codec）的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox —— 将音频视为 SoundStream / EnCodec 的离散 token，并在这些编解码器 token 上训练大型自回归（autoregressive）或流匹配（flow-matching）模型。短提示下的质量可与 ElevenLabs 媲美。

### 伦理部分，不是事后补丁

**水印（watermarking）。** PerTh（Perth）和 SilentCipher（2024）可在音频中不可感知地嵌入约 16–32 位 ID，能经受重编码、流媒体传输和常见编辑。已是可用于生产的开源方案。

**同意门控（consent gates）。** 必须为每次克隆输出配对可验证的同意记录。“本人 Rohit，于 2026-04-22 授权将本声音用于 X 目的。”存储在防篡改日志中。

**检测（detection）。** AASIST、RawNet2 与 Wav2Vec2-AASIST 可作为检测器使用。ASVspoof 2025 挑战赛公布了针对 ElevenLabs、VALL-E 2 和 Bark 输出的最先进检测器等错误率（EER）为 0.8–2.3%。

### 数据（2026）

| 模型 | 零样本？ | SECS（与目标相似度） | WER（可懂度） | 参数量 |
|------|----------|----------------------|---------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

SECS > 0.70 对大多数听众而言通常已难以与目标区分。

## 动手实现

### 步骤 1：用识别-合成方式分解（代码演示见 main.py）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上很简单；实现工作量主要在 `tts_model` 和说话人编码器（speaker encoder）中。

### 步骤 2：用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录文本必须与音频完全一致；不匹配会破坏对齐。

### 步骤 3：用 KNN-VC 进行语音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 使用 WavLM 为源语音和目标语音池提取逐帧嵌入（embedding），然后用目标池中的最近邻替换每一帧源嵌入。非参数化方法，只需约一分钟目标语音即可工作。

### 步骤 4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约 32 位有效载荷，经过 MP3 重编码和轻度噪声后仍可检测。

### 步骤 5：同意门控

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 应用场景

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业化生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（改写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + speaker adapter |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 常见陷阱

- **参考转录文本不一致。** F5-TTS 等要求参考文本与参考音频完全一致，包括标点。
- **参考音频混响严重。** 回声会毁掉克隆。请录制干燥、近距离拾音的音频。
- **情绪不匹配。** 训练参考是“欢快”的，就会把所有克隆都生成得欢快。参考情绪要与目标用途匹配。
- **语言泄漏。** 克隆英语说话人后让模型说法语，往往仍带有英语口音；请使用跨语言模型（XTTS、VALL-E X）。
- **没有水印。** 2026 年 8 月起在欧盟将不具备上线合法性。

## 交付

保存为 `outputs/skill-voice-cloner.md`。设计一个带同意门控、水印和质量目标的克隆或转换管线。

## 练习

1. **简单。** 运行 `code/main.py`。该演示通过计算替换前后的余弦相似度，展示说话人嵌入（speaker embedding）的交换效果。
2. **中等。** 使用 OpenVoice v2 克隆你自己的声音。测量参考与克隆之间的 SECS，并通过 Whisper 测量 CER。
3. **困难。** 对 20 条克隆音频应用 SilentCipher 水印，再经过 128 kbps MP3 编解码，检测有效载荷。报告比特准确率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 零样本克隆 | 5 秒就够了 | 预训练模型 + 说话人嵌入（speaker embedding）；无需训练。 |
| PPG | 音素后验图（phonetic posteriorgram） | 逐帧 ASR 后验，用作与语言无关的内容表征。 |
| KNN-VC | 最近邻转换 | 用目标池中的最近邻替换每一帧源语音。 |
| 神经编解码器 TTS | VALL-E 风格 | 在 EnCodec/SoundStream token 上的自回归模型。 |
| 水印 | 不可听的签名 | 嵌入音频的比特，能经受重编码。 |
| SECS | 克隆保真度 | 目标与克隆说话人嵌入之间的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) —— 开源的最先进零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 与 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) —— 神经编解码器 TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) —— 基于解耦的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) —— 基于检索的语音转换。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) —— 可用于生产的 32 位音频水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/) —— 检测器与合成器的军备竞赛，2026 年更新。
