# 音频生成

> 音频是一维信号，采样率为 16-48 kHz。一段 5 秒的音频就有 8–24 万个采样点。没有Transformer能直接处理这么长的序列。到 2026 年，所有生产级音频模型都采用了同一套方案：神经编解码器（neural codec，如 Encodec、SoundStream、DAC）先把音频压缩成 50-75 Hz 的离散token，再由Transformer或扩散模型生成这些token。

**类型：** 构建
**语言：** Python
**前置知识：** 第 6 阶段 · 02（音频特征），第 6 阶段 · 04（自动语音识别，ASR），第 8 阶段 · 06（去噪扩散概率模型，DDPM）
**时间：** 约 45 分钟

## 问题背景

音频生成有三种典型任务：

1. **文本转语音（text-to-speech，TTS）。** 给定文本生成语音。干净语音频带窄、音素结构强，用“token上的Transformer”就能很好解决。代表：VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示（文本、旋律、和弦进行、风格）生成音乐。分布更广。代表：MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效 / 声音设计。** 给定提示生成环境音或拟音（Foley）。代表：AudioGen、AudioLDM 2、Stable Audio Open。

这三者都基于相同的技术底座：神经音频编解码器（neural audio codec）+ 基于token的自回归（token-AR）或扩散（diffusion）生成器。

## 核心概念

![音频生成：编解码器token + Transformer或扩散模型](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器把波形压缩成逐时间步的向量；残差向量量化（residual vector quantization，RVQ）把每个向量变成 K 个码本的级联索引。解码器再还原成音频。例如：24 kHz 音频以 2 kbps 压缩，使用 8 层 RVQ 码本、75 Hz 帧率，相当于每秒 600 个token。

```
波形（16000 采样点/秒）
    └─ 编码器卷积 ─┐
                   ├─ RVQ 第 1 层 → 75 Hz 索引
                   ├─ RVQ 第 2 层 → 75 Hz 索引
                   ├─ …
                   └─ RVQ 第 8 层
```

### 其上的两种生成范式

**基于token的自回归（token-autoregressive）。** 把 RVQ token展平成序列，用仅解码器的Transformer自回归生成。MusicGen 使用“延迟并行（delayed parallel）”策略，同时生成 K 路码本流并通过每路偏移来缩短等效序列长度。VALL-E 则根据文本提示 + 3 秒语音样本来生成语音token。

**潜在扩散（latent diffusion）。** 把编解码器token当作连续潜变量，或用分类扩散来建模。Stable Audio 2.5 在连续音频潜变量上做流匹配（flow matching）。AudioLDM 2 采用文本 → Mel频谱 → 音频的扩散链路。

2024-2026 年的趋势：音乐生成领域流匹配正在占据上风（推理更快、样本更干净），而语音领域仍是token自回归的天下，因为它天然因果、适合流式输出。

## 产业现状

| 系统 | 任务 | 骨干网络 | 延迟 |
|------|------|----------|------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 | 首token 约 300ms |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态自回归 | 约 200ms |
| NaturalSpeech 3 | TTS | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / 音效 | DiT + 音频潜变量流匹配 | 1 分钟片段约 10s |
| Suno v4 | 完整歌曲 | 未公开；疑似 token-AR | 每首歌约 30s |
| Udio v1.5 | 完整歌曲 | 未公开 | 每首歌约 30s |
| MusicGen 3.3B | 音乐 | 32kHz Encodec 上的 Token-AR | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | 5s 片段约 5s |
| Riffusion v2 | 音乐 | 频谱图扩散 | 约 10s |

## 动手实现

`code/main.py` 演示了核心思想：用两种不同“风格”的合成“音频token”序列（风格 A 为高低token交替，风格 B 为单调递增）训练一个极简的 next-token Transformer；然后按风格条件采样。

### 步骤 1：合成音频token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # “类语音”：交替
        return [i % vocab_size for i in range(length)]
    # “类音乐”：递增
    return [(i * 3) % vocab_size for i in range(length)]
```

### 步骤 2：训练一个极简token预测器

一个以风格为条件的 bigram 风格预测器。重点在于模式：编解码器token → 交叉熵训练 → 自回归采样。

### 步骤 3：条件采样

给定风格token和起始token，从预测分布中采样下一个token，并继续生成 20–40 个token。

## 常见陷阱

- **编解码器质量决定输出上限。** 如果编解码器无法忠实还原某种声音，生成器再好也没用。目前开源最强的是 DAC。
- **RVQ 误差累积。** 每一层 RVQ 建模的是前一层的残差；第 1 层的误差会向下传播。对高层采样时使用 temperature 0 有帮助。
- **音乐长程结构。** 30 秒音频在 75 Hz 下超过 2 万个token，对Transformer压力很大。MusicGen 使用滑动窗口 + 提示延续；Stable Audio 使用更短片段 + 交叉淡化（crossfade）。
- **边界伪影。** 不同生成片段之间做交叉淡化需要精细的重叠相加。
- **对干净数据的巨大需求。** 音乐生成器需要数万小时的授权音乐。2024 年 Suno / Udio 被 RIAA 起诉的事件让这个问题浮出水面。
- **语音克隆伦理。** 3 秒语音样本 + 文本提示就足以让 VALL-E / XTTS / ElevenLabs 克隆声音。每个生产模型都需要滥用检测 + 退出名单。

## 应用场景

| 任务 | 2026 年技术栈 |
|------|--------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 语音克隆（需验证同意） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐，快速生成 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音助手 | GPT-4o realtime 或 Gemini Live |
| 开源音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付成果

保存 `outputs/skill-audio-brief.md`。Skill 接收一份音频需求简报（任务、时长、风格、声音、许可证），输出：模型 + 托管方案、提示格式（流派标签、风格描述、结构标记）、编解码器 + 生成器 + 声码器链路、随机种子协议，以及评估计划（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成序列是否符合该风格的模式。
2. **中等。** 添加延迟并行解码：模拟两路必须偏移 1 步的token流，并训练一个联合预测器。
3. **困难。** 使用 HuggingFace transformers 本地运行 MusicGen-small。用三个不同提示各生成一段 10 秒音频；对风格一致性做 A/B 评测。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------|
| 编解码器（codec） | “神经压缩” | 音频的编码器 / 解码器；典型输出是 50-75 Hz 的token。 |
| 残差向量量化（RVQ） | “Residual VQ” | K 个量化器级联；每个建模前一个量化器的残差。 |
| Token | “一个编解码器符号” | 码本中的离散索引；常见 1024 或 2048。 |
| 延迟并行（delayed parallel） | “偏移码本” | 以交错偏移同时生成 K 路token流，从而缩短序列长度。 |
| 流匹配（flow matching） | “2024 年音频领域的赢家” | 扩散采样的更直路径替代方案；采样更快。 |
| 语音提示（voice prompt） | “3 秒样本” | 说话人嵌入或token前缀，用于引导克隆出的声音。 |
| Mel频谱图（mel spectrogram） | “可视化图” | 对数幅度的感知频谱图；许多 TTS 系统使用它。 |
| 声码器（vocoder） | “Mel 转波形” | 把 Mel频谱图还原成音频的神经组件。 |

## 生产注意：音频是流式问题

音频是唯一一种用户期待“边生成边播放”的输出模态，而不是一次性全部给出。从生产角度看，这意味着输出token时间（Time Per Output Token，TPOT）至关重要，因为用户的听觉速度就是目标吞吐量，而不是阅读速度。对于 16 kHz 音频、按约 75 token/秒（Encodec）分词，服务器必须对每个用户每秒生成 ≥75 个token，才能保证播放不卡顿。

这带来两个架构层面的影响：

- **流匹配音频模型不容易直接流式化。** Stable Audio 2.5 和 AudioCraft 2 都是一次渲染固定长度的片段。要流式输出，必须对片段分块并重叠边界——类似滑动窗口扩散——相比编解码器自回归模型会增加 100-300ms 的延迟开销。

如果产品是“实时语音聊天”或“实时音乐续写”，选择编解码器自回归路线。如果是“提交后渲染一段 30 秒片段”，流匹配在质量和总延迟上更有优势。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) —— 编解码器的标杆之作。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) —— 首个广泛应用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) —— DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) —— VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) —— MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) —— AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) —— 2025 年的文本到音乐流匹配模型。
