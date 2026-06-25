# 语音反欺骗与音频水印 —— ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆的发展速度超过了防御手段。2026 年的生产级语音系统需要两样东西：一个能够区分真实语音与伪造语音的检测器（AASIST、RawNet2），以及一种能够经受压缩和编辑的水印（AudioSeal）。两者都要部署，否则就不要上线语音克隆功能。

**类型：** Build
**语言：** Python
**前置知识：** Phase 6 · 06（说话人识别）、Phase 6 · 08（语音克隆）
**时长：** ~75 分钟

## 问题背景

三种相关的防御技术：

1. **反欺骗 / 深度伪造检测（Anti-spoofing / deepfake detection）。** 给定一段音频，判断它是合成的还是真实的？ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是该领域的黄金标准。
2. **音频水印（Audio watermarking）。** 在生成的音频中嵌入人耳不可感知的信号，后续可由检测器提取。AudioSeal（Meta）和 WavMark 是目前的开源选择。
3. **可信溯源（Authenticated provenance）。** 对音频文件 + 元数据进行加密签名。C2PA / 内容真实性倡议（Content Authenticity Initiative）。

检测技术用于应对不合作的攻击者。水印技术用于合规 —— AI 生成的音频应当可被识别。2026 年，两者缺一不可。

## 核心概念

![反欺骗、水印与溯源 —— 三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 —— 2024-2025 年的基准

相比前几版最大的变化：

- **众包数据（crowdsourced data）**（而非录音棚纯净数据）—— 更贴近真实场景。
- **约 2000 名说话人**（此前约 100 名）。
- **32 种攻击算法。** TTS + 语音转换（voice conversion）+ 对抗扰动。
- **两个赛道（track）。** 独立 countermeasure（CM）检测；面向生物识别系统的 spoofing-robust ASV（SASV）。

ASVspoof 5 上的最先进结果：~7.23% EER。在较早的 ASVspoof 2019 LA 上：0.42% EER。真实世界部署：在野外音频上预计为 5-10% EER。

### AASIST 与 RawNet2 —— 检测模型家族

**AASIST**（2021 年提出，持续更新至 2026 年）。基于图注意力（graph-attention）的频谱特征建模。目前在 ASVspoof 5 countermeasure 任务上处于 SOTA。

**RawNet2。** 原始波形的卷积前端 + TDNN 骨干网络。更简单的基线；经过微调后仍具竞争力。

**NeXt-TDNN + SSL 特征。** 2025 年变体：ECAPA 风格 + WavLM 特征 + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal —— 2024 年的水印默认选择

Meta 的 **AudioSeal**（2024 年 1 月发布，v0.2 于 2024 年 12 月）。关键设计：

- **局部化（Localized）。** 在 16 kHz 采样率下逐帧检测水印（分辨率为 1/16000 秒）。
- **生成器与检测器联合训练。** 生成器学习嵌入不可听信号；检测器学习在各种增强后找到它。
- **鲁棒性。** 可经受 MP3 / AAC 压缩、均衡器（EQ）、±10% 变速、+10 dB SNR 噪声混合。
- **快速。** 检测器运行速度为实时 485 倍；比 WavMark 快 1000 倍。
- **容量。** 16 位有效载荷（payload）（可编码模型 ID、生成时间戳、用户 ID），可嵌入每条语音中。

### WavMark

AudioSeal 之前的开源基线。可逆神经网络，32 bits/sec。问题：

- 同步暴力搜索很慢。
- 可被高斯噪声或 MP3 压缩移除。
- 不适合实时场景。

### WaveVerify（2025 年 7 月）

针对 AudioSeal 的弱点 —— 尤其是时域篡改（反转、变速）。使用基于 FiLM 的生成器 + 混合专家（Mixture-of-Experts）检测器。在标准攻击下与 AudioSeal 相当；可应对时域编辑。

### 攻击者利用的缺口

来自 AudioMarkBench：“在变调（pitch shift）下，所有水印的位恢复准确率（Bit Recovery Accuracy）均低于 0.6，接近完全移除。” **变调是通用攻击。** 2026 年的水印都无法完全抵御激进的变调修改。这就是为什么需要 alongside 水印部署检测器（AASIST）。

### C2PA / 内容真实性倡议

这不是 ML 技术 —— 而是一种清单（manifest）格式。音频文件携带关于创作工具、作者、日期的加密签名元数据。Audobox / Seamless 在使用它。对溯源很好；但如果恶意行为者重新编码并剥离元数据，则完全失效。

## 动手实现

### 步骤 1：简单的频谱特征检测器（玩具示例）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音的高频能量通常异常平坦。生产级检测器使用 AASIST，而非此方法。但直觉成立。

### 步骤 2：AudioSeal 嵌入与检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1] 范围内的浮点数 —— 水印存在的概率
# decoded_payload: 16 位；与嵌入的 payload 比对
```

### 步骤 3：评估 —— EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 步骤 4：生产级集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都应附带：(1) 水印，(2) 签名清单，(3) 符合保留策略的审计日志。

## 应用场景

| 使用场景 | 防御手段 |
|----------|----------|
| 上线 TTS / 语音克隆 | 每个输出都嵌入 AudioSeal（不可妥协） |
| 生物识别语音解锁 | AASIST + ECAPA 集成；活体挑战 |
| 呼叫中心欺诈检测 | 对 20% 的来电采样运行 AASIST |
| 播客真实性验证 | 上传时 C2PA 签名，AI 生成内容加 AudioSeal |
| 研究 / 训练检测器 | ASVspoof 5 训练/开发/评估集 |

## 常见陷阱

- **只加水印却从不用检测器。** 毫无意义。把检测器也部署到 CI 中。
- **检测器未做校准（calibration）。** 在 ASVspoof LA 上训练的 AASIST 会过拟合；真实场景准确率下降。需针对你的领域校准。
- **变调缺口。** 激进变调会移除大多数水印。要准备检测回退方案。
- **元数据剥离后重新托管。** C2PA 可通过重新编码轻易绕过。必须同时部署加密 + 感知（水印）防御。
- **把活体检测当作反欺骗。** 让用户读随机短语。能防止重放攻击，但无法防御实时克隆。

## 上线交付

保存为 `outputs/skill-spoof-defender.md`。为你的语音生成部署选择检测模型、水印、溯源清单以及运维手册。

## 练习

1. **简单。** 运行 `code/main.py`。在合成音频上测试玩具检测器 + 玩具水印嵌入/检测。
2. **中等。** 安装 `audioseal`，在一段 TTS 输出中嵌入 16 位 payload，再解码。用噪声破坏音频并测量位恢复准确率（Bit Recovery Accuracy）。
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在一批 F5-TTS 生成的样本上测试 —— 观察 OOD 检测如何退化。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| ASVspoof | 基准 | 两年一度的挑战赛；2024 年为 ASVspoof 5。 |
| CM（countermeasure） | 检测器 | 分类器：真实语音 vs 合成/转换语音。 |
| SASV | 说话人验证 + CM | 集成的生物识别 + 欺骗检测。 |
| AudioSeal | Meta 水印 | 局部化、16 位 payload、比 WavMark 快 485 倍。 |
| Bit Recovery Accuracy | 水印生存能力 | 攻击后恢复出的 payload 比特比例。 |
| C2PA | 溯源清单 | 关于创作/作者身份的加密元数据。 |
| AASIST | 检测器家族 | 基于图注意力的反欺骗 SOTA。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) —— 当前基准。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) —— 默认水印方案。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) —— 面向时域攻击的 MoE 检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) —— SOTA 检测骨干。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) —— 鲁棒性评估。
- [C2PA 规范](https://c2pa.org/specifications/specifications/) —— 溯源清单格式。
