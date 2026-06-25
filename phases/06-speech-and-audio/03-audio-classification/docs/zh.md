# 音频分类 — 从基于 MFCC 的 k-NN 到 AST 与 BEATs

> 从“狗叫 vs 警笛”到“这是哪种语言”，一切都属于音频分类。特征是梅尔（mel）特征，架构每十年都在更替，评估始终是 AUC、F1 与逐类别召回率（per-class recall）。

**类型：** 实战构建  
**语言：** Python  
**前置知识：** Phase 6 · 02（Spectrograms & Mel），Phase 3 · 06（CNNs），Phase 5 · 08（CNNs & RNNs for Text）  
**时间：** 约 75 分钟

## 问题定义

你拿到一段 10 秒的音频，想知道：“这是什么？”城市声音（警笛、电钻、狗叫）、语音命令（yes/no/stop）、语言识别（en/es/ar）、说话人情绪（愤怒/中性）、环境声（室内/室外、嘈杂人声）——这些都属于**音频分类（audio classification）**。到了 2026 年，基线架构已经非常成熟：对数梅尔（log-mel）→ CNN 或 Transformer → softmax。

核心难点不在网络，而在数据。音频数据集往往存在严重的类别不平衡、强烈的域偏移（domain shift，干净 vs 噪声），以及标签噪声（label noise，谁来判断“urban babble”和“restaurant noise”？）。80% 的工作是数据整理、数据增强和评估，而不是把 CNN 换成 Transformer。

## 核心概念

![音频分类演进阶梯：从 MFCC 上的 k-NN 到 AST 再到 BEATs](../assets/audio-classification.svg)

**基于 MFCC 的 k-NN（1990 年代基线）。** 将每个音频片段的梅尔频率倒谱系数（MFCC）展平，计算与标注样本库的余弦相似度，返回前 K 个样本的多数投票。在干净的小数据集（如 Speech Commands、ESC-50）上表现惊人，且无需 GPU。

**基于对数梅尔的二维 CNN（2015–2019）。** 把形状为 `(T, n_mels)` 的对数梅尔谱图当作图像处理，使用 ResNet-18 或 VGG 风格的网络，对时间轴做全局平均池化，再经 softmax 输出类别。到 2026 年，它仍是大多数 Kaggle 竞赛的基线。

**音频谱图 Transformer（Audio Spectrogram Transformer, AST，2021–2024）。** 把对数梅尔谱图切分成小块（patch，例如 16×16），加入位置嵌入（position embeddings），输入到视觉 Transformer（ViT）中。在 AudioSet 的监督学习任务上曾达到最优水平（mAP 0.485）。

**BEATs 与 WavLM-base（2024–2026）。** 在数百万小时音频上进行自监督预训练（self-supervised pretraining），然后在你的任务上微调，所需的有标签数据仅为监督学习的 1%–10%。到 2026 年，这已成为非语音音频任务的默认起点。BEATs-iter3 在 AudioSet 上比 AST 高出 1–2 个 mAP，而计算量仅为后者的 1/4。

**Whisper 编码器作为冻结骨干网络（2024）。** 取出 Whisper 的编码器，去掉解码器，接上一个线性分类器。在语言识别和简单事件分类上接近 SOTA，且无需任何音频增强——这是一个“免费午餐”式的基线。

### 类别不平衡才是真正的挑战

ESC-50：50 个类别，每类 40 段音频——平衡且简单。UrbanSound8K：10 个类别， imbalance 达 10:1。AudioSet：632 个类别，长尾比例高达 100,000:1。行之有效的技巧包括：

- 训练时使用平衡采样（balanced sampling），评估时不用。
- Mixup：将两个片段（及其标签）线性插值，作为数据增强。
- SpecAugment：随机遮挡时间和频率 band。简单，但至关重要。

### 评估方式

- 互斥多分类（Speech Commands）：top-1 准确率、top-5 准确率。
- 多标签多分类（AudioSet、UrbanSound 风格）：平均精度均值（mean average precision, mAP）。
- 严重不平衡：逐类别召回率（per-class recall）+ 宏平均 F1（macro F1）。

2026 年需要知道的数字：

| 基准 | 基线 | 2026 SOTA | 来源 |
|------|------|-----------|------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs 论文（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR 2026 排行榜 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2 结果 |

## 动手实现

### 步骤 1：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步骤 2：固定长度汇总

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但有效：沿时间轴取均值 + 方差，能把 13 维 MFCC 转换成 26 维固定嵌入。运行极快，甚至在 2017 年仍能击败 ESC-50 上的部分前沿神经网络基线。

### 步骤 3：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步骤 4：升级到基于对数梅尔的 CNN

在 PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # 输入 x 形状: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万参数，单张 RTX 4090 在 ESC-50 上约 10 分钟即可完成训练，准确率达到 80% 以上。

### 步骤 5：2026 年的默认方案 —— 微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，可通过 `beats` 库加载 `microsoft/BEATs-base`；transformers API 的结构与之相同。

## 如何使用

2026 年的选型栈：

| 场景 | 首选方案 |
|------|----------|
| 极小数据集（<1000 段） | 基于 MFCC 均值的 k-NN（你的基线）+ 音频增强 |
| 中等数据集（1K–100K） | 微调 BEATs 或 AST |
| 大数据集（>100K） | 从头训练，或微调 Whisper 编码器 |
| 实时 / 边缘设备 | 40-MFCC CNN，量化为 int8（关键词唤醒风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE 损失 + Mixup + SpecAugment |
| 语言识别 | MMS-LID、SpeechBrain VoxLingua107 基线 |

决策规则：**优先使用冻结的骨干网络，而不是从零训练**。只微调 BEATs 的分类头，就能在数小时内达到 SOTA 的 95%，无需数周。

## 交付物

保存为 `outputs/skill-classifier-designer.md`。为指定的音频分类任务选择架构、数据增强、类别平衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它会在一个 4 类合成数据集（不同音高的纯音）上训练基于 MFCC 的 k-NN 基线。输出混淆矩阵。
2. **中等。** 把 `summarize` 替换为 [均值、方差、偏度、峰度]。四阶矩池化是否在相同合成数据集上超过“均值+方差”？
3. **困难。** 使用 `torchaudio`，在 ESC-50 的第 1 折（fold 1）上训练一个二维 CNN。报告 5 折交叉验证准确率。再加入 SpecAugment（时间掩码=20，频率掩码=10）并报告提升幅度。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| AudioSet | 音频界的 ImageNet | Google 的 200 万段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类 × 40 段环境声。 |
| AST | Audio Spectrogram Transformer | 对对数梅尔 patch 应用 ViT；2021 年的 SOTA。 |
| BEATs | 自监督音频模型 | 微软的模型，截至 2026 年 iter3 在 AudioSet 上领先。 |
| Mixup | 成对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩码的增强 | 将频谱图中随机的时间和频率 band 置零。 |
| mAP | 主要多标签指标 | 跨类别和阈值的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021–2024 年的标杆架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024 年以后的默认选择。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主流的音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 沿用至今的 50 类基准。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632 类 YouTube 分类体系；仍是黄金标准。
