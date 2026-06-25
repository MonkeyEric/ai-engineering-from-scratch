# 音乐生成 —— MusicGen、Stable Audio、Suno 与授权地震

> 2026 年的音乐生成：Suno v5 和 Udio v4 主导商业领域；MusicGen、Stable Audio Open 和 ACE-Step 引领开源方向。技术问题已基本解决。法律问题（华纳音乐 5 亿美元和解、环球音乐集团和解）在 2025–2026 年重塑了整个领域。

**类型：** 动手实践
**语言：** Python
**先修知识：** 第 6 阶段 · 第 02 课（频谱图），第 4 阶段 · 第 10 课（扩散模型）
**时长：** 约 75 分钟

## 问题定义

文本 → 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **器乐生成。** 例如提示词“lo-fi hip-hop drums with warm keys”→ 生成音频。代表模型：MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（含人声 + 歌词）。** 例如提示词“Country song about rainy Texas nights”→ 完整歌曲。代表模型：Suno、Udio、YuE、ACE-Step。
3. **条件化 / 可控生成。** 扩展已有片段、重新生成桥段、切换风格、分轨（stem separation），或进行修复（inpaint）。Udio 的修复 + 分轨功能是 2026 年需要追赶的特性。

## 核心概念

![音乐生成：神经编解码器 token 的语言模型 vs 扩散模型，2026 模型地图](../assets/music-generation.svg)

### 基于神经编解码器 token 的自回归语言模型

Meta 的 **MusicGen**（2023，MIT 协议）及众多衍生模型：以文本/旋律嵌入（embedding）为条件，自回归地预测 EnCodec token（32 kHz，4 个码本），再用 EnCodec 解码。参数量 3 亿到 33 亿。是一个强基线；但超过 30 秒后效果会下降。

**ACE-Step**（开源，40 亿参数的 XL 版本于 2026 年 4 月发布）将这一路线扩展到基于歌词的完整歌曲生成。它是开源社区最接近 Suno 的模型。

### 基于梅尔频谱或隐变量的扩散模型

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：在压缩音频上进行隐空间扩散（latent diffusion）。擅长生成循环乐段、声音设计和氛围纹理。结构化完整歌曲能力较弱。

**AudioLDM / AudioLDM2**：通过类似文生图（T2I）的隐空间扩散实现文本到音频生成，泛化到音乐、音效和语音。

### 混合方案（商用）—— Suno、Udio、Lyria

闭源权重。很可能是自回归编解码器语言模型 + 基于扩散的声码器（vocoder），并配备专门的人声 / 鼓点 / 旋律头。Suno v5（2026）在 ELO 1293 的质量评分上领先。Udio v4 新增修复（inpainting）+ 分轨功能（可分别下载 bass、drums、vocals）。

### 评估指标

- **FAD（Fréchet Audio Distance，弗雷歇音频距离）。** 使用 VGGish 或 PANNs 特征，度量生成音频与真实音频分布在嵌入空间中的距离。越低越好。MusicGen small 在 MusicCaps 上约为 4.5 FAD；SOTA 约 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 以 ELO 1293 领先。
- **文本-音频对齐度。** 提示词与输出之间的 CLAP 分数。
- **音乐性瑕疵。** 节拍错位过渡、人声短语漂移、超过 30 秒后结构散架。

## 2026 模型地图

| 模型 | 参数量 | 长度 | 人声 | 许可协议 |
|------|--------|------|------|----------|
| MusicGen-large | 3.3B | 30 s | 无 | MIT |
| Stable Audio Open | 1.2B | 47 s | 无 | Stability 非商业 |
| ACE-Step XL（2026 年 4 月） | 4B | > 2 min | 有 | Apache-2.0 |
| YuE | 7B | > 2 min | 有，多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4 min | 有，ELO 1293 | 商业 |
| Udio v4（闭源） | ? | 4 min | 有 + 分轨 | 商业 |
| Google Lyria 3（闭源） | ? | 实时 | 有 | 商业 |
| MiniMax Music 2.5 | ? | 4 min | 有 | 商业 API |

## 法律环境（2025–2026）

- **华纳音乐诉 Suno 和解案。** 5 亿美元。华纳音乐现在对 Suno 上的 AI  likeness、音乐版权和用户生成曲目拥有监督权。环球音乐集团也与 Udio 达成了类似和解。
- **欧盟 AI 法案** + **加州 SB 942**：AI 生成的音乐必须披露。
- **Riffusion / MusicGen** 采用 MIT 协议，没有合规包袱，但也没有商业人声能力。

可安全交付的模式：

1. 仅生成器乐（MusicGen、Stable Audio Open、MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music）并遵守每次生成的许可。
3. 基于自有或已授权曲库训练（大多数企业最终走这条路）。
4. 为生成内容添加水印 + 元数据标签。

## 动手实践

### 步骤 1：使用 MusicGen 生成音乐

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种尺寸：`small`（3 亿参数，快）、`medium`（15 亿）、`large`（33 亿）。用于验证“这个想法是否可行”时，small 已足够。

### 步骤 2：旋律条件控制

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收色度图（chromagram），在保留曲调的同时替换音色。适合“把这段旋律变成弦乐四重奏”这类需求。

### 步骤 3：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算基于 VGGish 嵌入的距离。适合用作流派级别的回归测试；但不能替代人类听感。

### 步骤 4：接入 LLM-音乐工作流

结合第 7–8 课的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 应用场景

| 目标 | 技术栈 |
|------|--------|
| 器乐声音设计 | Stable Audio Open |
| 游戏 / 自适应音乐 | Google Lyria RealTime（闭源） |
| 完整歌曲含人声（商业） | Suno v5 或 Udio v4，并明确许可 |
| 完整歌曲含人声（开源） | ACE-Step XL 或 YuE |
| 短广告 jingle | MusicGen 基于哼唱参考进行旋律条件生成 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍会踩的坑

- **版权洗白提示词。** “Song in the style of Taylor Swift”——商业 Suno/Udio 现在会过滤这类提示，开源模型不会。需要自行添加过滤列表。
- **重复 / 超过 30 秒后的漂移。** 自回归模型会循环。可对多段生成结果做交叉淡化（crossfade），或使用 ACE-Step 保证结构连贯性。
- **速度漂移。** 模型会偏离设定 BPM。在提示词中加入 BPM 标签，并用 librosa 的 `beat_track` 做后筛选。
- **人声可懂度。** Suno 表现优秀；开源模型在歌词上常常含糊。如果歌词重要，使用商业 API 或进行微调。
- **单声道输出。** 开源模型生成单声道或伪立体声。可用真正的立体声重建方法升级（如 ezst、Cartesia 的立体声扩散）。

## 交付

将方案保存为 `outputs/skill-music-designer.md`。需选定模型、许可策略、长度 / 结构计划，以及音乐生成部署的披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它生成一段“生成式”和弦进行 + 鼓点模式，以 ASCII 符号表示——一幅音乐生成的卡通图。如需可导入任意 MIDI 播放器播放。
2. **中等。** 安装 `audiocraft`，用 MusicGen-small 基于 4 个不同流派提示生成 10 秒片段，并针对参考流派集合测量 FAD。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody），用不同音色提示生成同一曲调的三种变体。计算与提示词的 CLAP 相似度，验证对齐程度。

## 关键术语

| 术语 | 业界说法 | 实际含义 |
|------|----------|----------|
| FAD | Audio FID | 真实音频与生成音频嵌入分布之间的弗雷歇距离。 |
| Chromagram（色度图） | 旋律即音高 | 每帧 12 维向量；旋律条件化的输入。 |
| Stems（分轨） | 乐器轨道 | 分离出的 bass / drums / vocals / melody，以 WAV 形式提供。 |
| Inpainting（修复） | 重生成某一段 | 遮蔽一段时间窗口；模型只重生成该部分。 |
| CLAP | 文本-音频版 CLIP | 对比式音频-文本嵌入；用于评估文本-音频对齐度。 |
| EnCodec | 音乐编解码器 | Meta 的神经编解码器，MusicGen 使用；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) —— 开源自回归基线。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) —— 声音设计默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) —— 40 亿参数开源完整歌曲生成器，2026 年 4 月发布。
- [Suno v5 platform docs](https://suno.com) —— 商业质量领导者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) —— 面向音乐 + 音效的隐空间扩散。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) —— 2025 年 11 月的先例。
