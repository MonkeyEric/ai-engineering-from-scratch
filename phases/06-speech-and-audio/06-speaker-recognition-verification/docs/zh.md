# 说话人识别与声纹验证

> ASR 问的是“他说了什么？”，而说话人识别问的是“谁在说话？”。背后的数学看起来一样——嵌入（embedding）加余弦（cosine）相似度——但每一个生产环境决策都取决于一个 EER 数字。

**类型：** 构建  
**语言：** Python  
**前置知识：** Phase 6 · 02（Spectrograms & Mel），Phase 5 · 22（Embedding Models）  
**时长：** 约 45 分钟

## 问题背景

用户说一段口令。你想知道：这是否是他/她声称的那个人（*验证（verification）*，1:1），还是注册库中的某一个人（*识别（identification）*，1:N）？或者都不是——这是否是一个未注册说话人（*开集（open-set）*）？

2018 年之前：GMM-UBM + i-vector。EER 尚可，但对信道迁移（channel shift，如手机 vs 笔记本）和情绪变化很敏感。2018–2022 年：x-vector（以角边距（angular margin）训练的 TDNN 主干网络）。2022 年以后：ECAPA-TDNN 与 WavLM-large 嵌入（embedding）。到 2026 年，该领域已被三种模型和一个指标主导。

该指标就是 **EER（等错误率，Equal Error Rate）**。设定决策阈值（threshold），使错误接受率（False Accept Rate）等于错误拒绝率（False Reject Rate），二者的交点就是 EER。每一篇论文、每一个排行榜、每一次采购评估都会用到它。

## 核心概念

![注册与验证流程：嵌入 + 余弦相似度 + EER](../assets/speaker-verification.svg)

**流程。** 注册（enrollment）：录制目标说话人 5–30 秒的音频；计算固定维度的嵌入（embedding）（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证（verification）：获取测试语音的嵌入；计算余弦相似度（cosine similarity）；再与阈值比较。

**ECAPA-TDNN（2020 年提出，2026 年仍占主导）。** 全称 Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。使用 1D 卷积块（挤压激励（squeeze-excitation））、多头注意力池化（multi-head attention pooling），再经线性层映射到 192 维。在 VoxCeleb 1+2（2,700 名说话人，110 万条语音）上以加性角边距损失（Additive Angular Margin loss，AAM-softmax）训练。

**WavLM-SV（2022 年以后）。** 使用 AAM 损失（AAM loss）微调预训练的 WavLM-large 自监督学习（SSL）主干网络。质量更高但更慢——模型体积 300+ MB，而 ECAPA-TDNN 仅 15 MB。

**x-vector（基线）。** TDNN + 统计池化（statistics pooling）。经典架构；在 CPU / 边缘设备上仍有用。

**AAM-softmax。** 在角空间（angular space）中对标准 softmax 加入边距 `m`：对正确类别使用 `cos(θ + m)`。它强制增大类间角间距。典型参数为 `m=0.2`，缩放因子 `s=30`。

### 打分方式

- **余弦相似度（cosine）。** 计算注册与测试嵌入之间的相似度，再基于阈值做判断。
- **PLDA（概率线性判别分析，Probabilistic LDA）。** 将嵌入投影到一个潜在空间（latent space），使同说话人与不同说话人的样本具有闭式似然比（likelihood ratio）。在余弦相似度基础上叠加 PLDA，可降低 10–20% 的 EER。2020 年前的标准方案；现在主要用于闭集（closed-set）场景。
- **分数归一化（score normalization）。** `S-norm` 或 `AS-norm`：用一组冒名顶替者（imposter）的均值与标准差对每个分数做归一化。跨域评估时必不可少。

### 2026 年需要了解的关键数字

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐（A100）|
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### 说话人分割聚类（Diarization）

多说话人音频中“谁在什么时候说话”。流程：VAD（语音活动检测）→ 切分 → 对每个片段提取嵌入 → 聚类（层次聚类（agglomerative）或谱聚类（spectral））→ 平滑边界。现代栈：`pyannote.audio` 3.1，它将说话人分割、嵌入与聚类封装在一个调用中。2026 年在 AMI 上的 SOTA（state-of-the-art）DER 约为 15%（2022 年为 23%）。

## 动手实现

### 步骤 1：用 MFCC 统计量构造极简嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

这与 SOTA 相差甚远——仅供教学。`code/main.py` 用它对合成说话人数据做概念验证。

### 步骤 2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤 3：由相似度对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 `(eer, threshold_at_eer)`。两个值都要报告。

### 步骤 4：用 SpeechBrain 部署生产级模型

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 注册：对 3-5 条干净样本的嵌入取平均
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# 验证
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA 典型阈值；请在你的数据上微调
```

### 步骤 5：用 pyannote 进行说话人分割聚类

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 如何使用

2026 年的技术选型：

| 场景 | 选择 |
|-----------|------|
| 闭集（closed-set）1:1 验证，边缘端 | ECAPA-TDNN + 余弦阈值 |
| 开集（open-set）验证，云端 | WavLM-SV + AS-norm |
| 说话人分割聚类（Diarization，会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（Anti-spoofing，重放 / 深度伪造检测） | AASIST 或 RawNet2 |
| 超小型嵌入式（KWS + 注册） | Titanet-Small（NeMo）|

## 常见陷阱

- **信道不匹配（Channel mismatch）。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。必须在目标信道上评估。
- **语音过短（Short utterances）。** 测试音频低于 3 秒时，EER 会急剧变差。
- **注册样本带噪（Enrollment with noise）。** 一条含噪注册样本就会污染锚点（anchor）。使用 ≥3 条干净样本并取平均。
- **跨场景固定阈值（Fixed threshold across conditions）。** 必须在目标域的留出开发集（held-out dev set）上调整阈值。
- **对未归一化嵌入计算余弦（Cosine on non-normalized embeddings）。** 先进行 L2 归一化；否则模长会主导结果。

## 交付

保存为 `outputs/skill-speaker-verifier.md`。选定模型、注册协议、阈值调优方案以及防欺诈措施。

## 练习

1. **简单。** 运行 `code/main.py`。它会构建合成的“说话人”（不同音调特征），完成注册，并在 100 对测试列表上计算 EER。
2. **中等。** 在 30 条 VoxCeleb1 语音（5 位说话人 × 每人 6 条）上使用 SpeechBrain ECAPA。分别用余弦相似度和 PLDA 计算 EER。
3. **困难。** 使用 `pyannote.audio` 构建完整的注册 → 分割聚类 → 验证流程。在 AMI 开发集上评估 DER。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| EER |  headline 指标 | 错误接受率等于错误拒绝率时的阈值。 |
| 验证（Verification） | 1:1 | “这是 Alice 吗？” |
| 识别（Identification） | 1:N | “谁在说话？” |
| 开集（Open-set） | 可能存在未知说话人 | 测试集可能包含未注册说话人。 |
| 注册（Enrollment） | 登记 | 计算说话人的参考嵌入。 |
| AAM-softmax | 损失函数 | 带加性角边距的 softmax；强制类簇分离。 |
| PLDA | 经典打分 | 概率 LDA；在嵌入之上进行似然比打分。 |
| DER | 分割聚类指标 | Diarization Error Rate——漏检 + 误检 + 混淆。 |

## 延伸阅读

- [Snyder 等（2018）。X-Vectors：面向说话人识别的稳健深度神经网络嵌入](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) —— 经典的深度嵌入论文。
- [Desplanques 等（2020）。ECAPA-TDNN](https://arxiv.org/abs/2005.07143) —— 2020–2026 年的主流架构。
- [Chen 等（2022）。WavLM：面向全栈语音处理的大规模自监督预训练](https://arxiv.org/abs/2110.13900) —— SV 与分割聚类的 SSL 主干网络。
- [Bredin 等（2023）。pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) —— 生产级分割聚类 + 嵌入栈。
- [VoxCeleb 排行榜（2026 年更新）](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) —— 各模型当前 EER 排名。
