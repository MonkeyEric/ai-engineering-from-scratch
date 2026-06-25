# OCR 与文档理解

> OCR 是一个三阶段流水线——检测文本框、识别字符、重组版式。每个现代 OCR 系统都会重新排列或合并这些阶段。

**类型：** 学习 + 使用
**语言：** Python
**先修知识：** Phase 4 Lesson 06（检测）、Phase 7 Lesson 02（自注意力）
**时长：** 约 45 分钟

## 学习目标

- 梳理经典 OCR 流水线（检测 → 识别 → 版式）以及现代端到端替代方案（Donut、Qwen-VL-OCR）
- 实现 CTC（Connectionist Temporal Classification，连接时序分类）损失，用于序列到序列的 OCR 训练
- 使用 PaddleOCR 或 EasyOCR 进行无需训练的生产级文档解析
- 区分 OCR、版式解析与文档理解，并为每个任务选择合适的工具

## 问题背景

满屏文字的图片随处可见：收据、发票、身份证、扫描书籍、表单、白板、路牌、截图。从中提取结构化数据——不仅仅是字符，而是“这是总金额”——是应用视觉领域价值最高的任务之一。

该领域可分为三个技能层级：

1. **OCR 本身**：把像素变成文本。
2. **版式解析**：将 OCR 输出分组为区域（标题、正文、表格、页眉）。
3. **文档理解**：从版式中提取结构化字段（“invoice_total = $42.50”）。

每一层都有经典方法和现代方法，而“我想从图片中拿到文字”与“我需要从这张收据中提取总金额”之间的差距，往往比大多数团队意识到的更大。

## 核心概念

### 经典流水线

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**：生成每行或每个单词的四边形框。
- **文本识别**：将每个区域裁剪为固定高度，运行 CNN + BiLSTM + CTC，生成字符序列。
- **版式重组**：恢复阅读顺序（拉丁语系为从上到下、从左到右；阿拉伯语、日语等则不同）。

### 一段话讲清 CTC

OCR 识别需要从固定长度的特征图中生成可变长度的序列。CTC（Graves 等，2006）允许你在没有字符级对齐的情况下训练这样的模型。模型在每个时间步输出一个在（词表 + 空白）上的分布；CTC 损失会对所有经过“合并重复并删除空白”后等于目标文本的对齐路径进行边缘化求和。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 2015 年 CRNN 能够奏效的原因，也是 2026 年大多数生产级 OCR 模型仍在使用的训练方式。

### 现代端到端模型

- **Donut**（Kim 等，2022）——ViT 编码器 + 文本解码器；直接读取图像并输出 JSON。无需文本检测器，也无需版式模块。
- **TrOCR**——ViT + Transformer 解码器，用于行级 OCR。
- **Qwen-VL-OCR / InternVL**——面向 OCR 任务微调的大型视觉语言模型；在复杂文档上 2026 年表现最佳。
- **PaddleOCR**——成熟的开源生产包，采用经典 DB + CRNN 流水线；仍是开源主力军。

端到端模型需要更多的数据和算力，但避免了多阶段流水线的误差累积。

### 版式解析

对于结构化文档，运行版式检测器（LayoutLMv3、DocLayNet），为每个区域打上标签：Title、Paragraph、Figure、Table、Footnote。阅读顺序于是变成“按版式顺序遍历区域并拼接”。

对于表单，使用**键值提取**模型（富视觉文档用 Donut，普通扫描件用 LayoutLMv3）。它们以图像 + 检测文本 + 位置为输入，预测结构化键值对。

### 评估指标

- **CER（Character Error Rate，字符错误率）**——Levenshtein 距离 / 参考文本长度。越低越好。干净扫描件的生产目标：< 2%。
- **WER（Word Error Rate，词错误率）**——词粒度上的同类指标。
- **结构化字段 F1**——用于键值任务；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON 编辑距离**——用于端到端文档解析；Donut 论文提出了归一化树编辑距离。

## 动手实现

### 步骤 1：CTC 损失 + 贪婪解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C)，词表包含空白符（下标为 0）的 log-softmax
    targets:        (N, S) int，目标序列（不含空白符）
    input_lengths:  (N,) 每个样本实际使用的时间步数
    target_lengths: (N,) 每个样本目标序列长度
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: 索引序列列表（已删除空白符、合并重复）
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时会调用高效的 CuDNN 实现。贪婪解码器比束搜索简单，通常 CER 仅比束搜索差约 1%。

### 步骤 2：微型 CRNN 识别器

用于行级 OCR 的最小化 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

输入高度固定（CNN 通过池化将高度压为 1）。宽度是 CTC 的时间维度。

### 步骤 3：合成 OCR 数据

生成黑底白字的数字串，用于端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实 OCR 数据集会加入字体、噪声、旋转、模糊和颜色。但上述流水线完全一致。

### 步骤 4：训练草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个非常简单的合成数据上，损失应在 200 步内从约 3 降到约 0.2。

## 实际使用

三条生产级路径：

- **PaddleOCR**——成熟、快速、多语言。一行调用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR**——原生 Python、多语言、PyTorch 后端。
- **Tesseract**——经典工具；当现代模型在老旧的扫描文档上表现不佳时仍然有用。

对于端到端文档解析，使用 Donut 或视觉语言模型：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于结构可重复（重复出现相同版式）的收据、发票和表单，微调 Donut。对于任意文档或需要推理的 OCR，视觉语言模型如 Qwen-VL-OCR 是目前的首选。

## 交付成果

本节课产出：

- `outputs/prompt-ocr-stack-picker.md`——一个根据文档类型、语言和结构选择 Tesseract / PaddleOCR / Donut / VLM-OCR 的提示词。
- `outputs/skill-ctc-decoder.md`——一个从零实现贪婪解码与束搜索 CTC 解码器（含长度归一化）的技能文档。

## 练习题

1. **（简单）** 在 5 位随机数字串上训练 TinyCRNN 500 步。在留出测试集上报告 CER。
2. **（中等）** 将贪婪解码替换为束搜索（beam_width=5）。报告 CER 差异。束搜索在哪些输入上表现更好？
3. **（困难）** 在 20 张收据上使用 PaddleOCR 提取行项目，并针对 `{item_name, price}` 对计算与人工标注真值的 F1。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OCR | “从像素中提取文字” | 将图像区域转换为字符序列 |
| CTC | “无需对齐的损失” | 不需要逐时间步标签即可训练序列模型的损失；对所有可能的对齐路径边缘化 |
| CRNN | “经典 OCR 模型” | CNN 特征提取器 + BiLSTM + CTC；2015 年的基线，至今仍在生产中使用 |
| Donut | “端到端 OCR” | ViT 编码器 + 文本解码器；直接从图像输出 JSON |
| Layout parsing | “寻找区域” | 检测并标注文档中的 Title/Table/Figure/Paragraph 等区域 |
| Reading order | “文本序列” | 将识别到的区域按可读顺序排列；拉丁语系简单，复杂版式则不然 |
| CER / WER | “错误率” | Levenshtein 距离 / 参考长度，分别在字符或词粒度上计算 |
| VLM-OCR | “会阅读的 LLM” | 面向 OCR 任务训练或提示的视觉语言模型；在复杂文档上是当前 SOTA |

## 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) —— 最初的 CNN+RNN+CTC 架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— CTC 原始论文，算法思想密集
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) —— 无需 OCR 的文档理解 Transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) —— 开源生产级 OCR 套件
