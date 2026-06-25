# Qwen-VL 系列与动态 FPS 视频

> Qwen-VL 系列——包括 Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025）——是 2026 年最具影响力的开源视觉语言模型（VLM）家族。每一代都押注了一项决定性的架构选择，而其余开源生态在十二个月内纷纷跟进：通过 M-RoPE（多模态旋转位置编码）实现原生动态分辨率、带绝对时间对齐的动态 FPS 采样、ViT（视觉 Transformer）中的窗口注意力（window attention），以及结构化智能体（agent）输出格式。到 Qwen3-VL，这套配方已经稳定：一个支持原生宽高比输入的 2D-RoPE-ViT 编码器、一个将特征投影到大型 Qwen3 语言底座的 MLP 投影器（MLP projector），以及把 OCR（光学字符识别）、定位（grounding）和智能体（agent）行为作为一级目标的训练阶段。本课按时间线梳理这一家族，帮助你理解每个设计参数的来龙去脉。

**类型：** 学习
**语言：** Python（标准库，M-RoPE 编码器 + 动态 FPS 采样器）
**先修要求：** Phase 12 · 06（patch-n'-pack）
**时长：** ~120 分钟

## 学习目标

- 计算 M-RoPE 的三轴旋转（时间、高度、宽度），并解释为什么三者都不可或缺。
- 为一段视频选择动态 FPS 采样策略，并能权衡每秒 token 数与事件检测准确率。
- 按顺序说出 Qwen-VL 四代的升级点，以及每一代解锁了什么能力。
- 编写 Qwen2.5-VL 风格的 JSON 智能体输出格式，并从 VLM 响应中解析结构化工具调用。

## 问题背景

Qwen-VL 于 2023 年 8 月发布，直接回应 LLaVA-1.5 和 BLIP-2。Qwen 团队瞄准的差距有三个：分辨率、视频和结构化输出。

分辨率：LLaVA-1.5 跑在 336x336。对照片够用，对中文发票或密集表格截图则完全不行。Qwen-VL 的首项创新是 448x448 与定位（grounding）边界框输出，让模型能够“指”出东西。

视频：Video-LLaMA 把逐帧编码器堆叠后喂给大语言模型（LLM）。短片段有效，但对时间轴才是信号的多分钟视频无能为力。Qwen 团队希望单个编码器能理解时间。

结构化输出：LLaVA 输出自由文本。智能体（agent）需要 JSON。Qwen-VL 显式训练了 JSON 输出格式，包括以文本 token 形式输出的边界框（bounding box）坐标。

每一代 Qwen-VL 都在延伸这三条轴线之一。

## 核心概念

### Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为编码器（25 亿参数），与 LLama 兼容的 Q-Former（单步，256 个查询），Qwen-7B 底座。贡献：

- 448x448 分辨率（当时开源 VLM 的最先进水平）。
- 定位（grounding）：在带有显式坐标 token 输出的图文对上训练。例如“猫在 <box>(112, 204), (280, 344)</box>”。
- 中英双语训练从一开始就纳入。

当时基准：英文与 GPT-4V 相当，中文占据优势。真正的亮点是 grounding 监督。

### Qwen2-VL（2024 年 9 月）——M-RoPE 与原生分辨率

Qwen2-VL 用原生动态分辨率 ViT 编码器取代了固定分辨率 + Q-Former 堆栈。关键变化：

- 原生动态分辨率。ViT 接受任意可被 28 整除的 HxW（patch 为 14，空间合并 2x）。一张 1120x672 的图像（40x24 个合并 patch）产生 960 个视觉 token。无需 resize、无需切块、无需缩略图。
- M-RoPE（Multimodal RoPE，多模态旋转位置编码）。每个 token 携带三维位置 (t, h, w) 而非一维。图像 t=0，视频 t = frame_index。RoPE 按每个轴的频率旋转查询/键（query/key）向量。没有位置嵌入（position embedding）表。
- MLP 投影器（MLP projector）。放弃 Q-Former，在合并后的 patch token 上使用两层 MLP。
- 视频的动态 FPS。默认以 1-2 FPS 采样视频，但模型接受任意帧数。

结果：Qwen2-VL-7B 在多个多模态基准上与 GPT-4o 持平，并在 DocVQA 上击败它（94.5 vs 88.4）。架构变革是关键一招。

### Qwen2.5-VL（2025 年 2 月）——动态 FPS + 绝对时间

Qwen2.5-VL 的重大转向是视频。动态 FPS 不只是“需要时多采样几帧”。论文将其形式化为：

- 绝对时间 token（absolute time token）。不使用位置索引（第 0、1、2 帧……），而使用真实时间戳。例如“在 0:04，猫跳了起来。”模型看到的是与帧 token 交错的 `<time>0.04</time>` token。
- 动态 FPS。慢镜头用 1 FPS，动作场景用 4+ FPS。用户或训练器决定；M-RoPE 自适应。
- ViT 中的窗口注意力（window attention）。空间注意力被窗口化（在局部块内）以提升吞吐；每隔几层加一次全局注意力。
- 显式 JSON 输出格式。在工具调用数据上训练：`"{\"tool\": \"click\", \"coords\": [380, 220]}"`。开箱即用，适合智能体。
- MRoPE-v2 缩放。位置随最大输入尺寸缩放，因此 10 分钟视频不会超出频率范围。

基准：Qwen2.5-VL-72B 在多数视频基准上击败 GPT-4o，在文档上与 Gemini 2.0 持平，并在 GUI 定位（ScreenSpot：准确率 84%，GPT-4o 仅 38%）上创下开源模型最先进水平。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是一次整合式升级，而非重新发明：更大的 LLM 主干（Qwen3-72B）、更多训练数据、更强的 OCR、通过 Qwen3 “思考模式”获得更强推理能力。ViT 和 M-RoPE 保持不变。论文重点放在数据与训练改进，而非架构。

家族启示：到 2025 年，Qwen-VL 架构已经稳定。后续代际扩展的是算力和数据，而非基础组件。

### M-RoPE 的数学原理

经典旋转位置编码（RoPE）使用成对坐标，按位置 `m` 旋转维度为 `d` 的查询 `q`：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 将隐藏维度（hidden dimension）分成三个频段。假设 `d = 96`。32 维给时间，32 维给高度，32 维给宽度。每个频段按各自轴的位置旋转。位于 (t=5, h=10, w=20) 的 patch 会在三个频段上分别应用旋转 `R_t(5)`、`R_h(10)`、`R_w(20)`。

文本 token 使用 `t = text_index, h = 0, w = 0`（或某种归一化选择），保持兼容性。视频帧使用 `t = frame_time, h = row, w = col`。单张图像使用 `t = 0`。

好处：一种位置编码即可处理文本、图像和视频，无需分支代码或不同的位置表。

### 动态 FPS 采样逻辑

给定一段时长为 `T` 秒的视频和 token 预算 `B`：

1. 计算能承受的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选取满足 `fps <= fps_max` 的目标 FPS。
3. 若运动程度高（光流启发式或用户显式要求），选更高 FPS；若运动程度低，选更低 FPS。
4. 按所选 FPS 均匀采样；在帧之间插入 `<time>t</time>` token。

Qwen2.5-VL 隐式地训练了这一逻辑；推理时用户通过 `fps` 参数控制。一段 60 秒的动作序列以 4 FPS、每帧 81 token 计算，共 19440 token，在 32k 上下文中可以承受。

### 结构化智能体输出

Qwen2.5-VL 的智能体训练明确针对结构化工具调用：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出执行 `JSON.parse`。相比之下，自由形式的“click at (1024, 512)”需要正则和消歧。正是这种转变，让 Qwen2.5-VL 的 ScreenSpot 得分从 Qwen2-VL 的 55% 跃升至 84%。

## 动手实践

`code/main.py` 实现了：

- 对混合文本、图像 patch 和视频帧的打包序列进行 M-RoPE 位置计算。
- 动态 FPS 采样器：给定（时长、预算、运动级别），选择 FPS 并输出帧时间戳。
- 一个玩具级 Qwen2.5-VL JSON 输出解析器，处理带坐标字段的工具调用响应。

运行它，然后感受在 5 分钟视频上把固定 FPS 换成动态 FPS 的差异。

## 产出交付

本课产出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定一个视频任务（监控、智能体、动作识别、无障碍），它会输出 Qwen2.5-VL 配置（帧预算、FPS 策略、窗口注意力开关、智能体输出模式）和延迟估计。每当你把 Qwen-VL 家族模型部署到视频产品时，都可使用它。

## 练习题

1. 计算一个位于 (t=3, h=5, w=7)、隐藏维度 48（每频段 16，基 theta 10000）的 patch 的 M-RoPE 旋转。展示每个频段前三对坐标的旋转角。
2. 一段 10 分钟安防摄像头录像以 1 FPS 采样会产生多少帧？在 384 分辨率、3x 池化下，总 token 数是多少？Qwen2.5-VL 默认 32k 上下文能否装下？
3. 为 30 秒网球对拉、30 秒食谱演示、30 秒 UI 智能体录屏分别选择 FPS。用动态 FPS 逻辑为每个选择辩护。
4. Qwen2.5-VL 完全放弃了 Q-Former。为什么简单的 MLP 在 2025 年可行，2023 年却不行？（提示：数据规模和编码器质量。）
5. 把三个 Qwen2.5-VL JSON 工具调用输出解析成 Python 字典。畸形 JSON 会怎么失败，Qwen 官方 cookbook 推荐什么恢复策略？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| M-RoPE | “Multimodal RoPE” | 隐藏维度中带有时间、高度、宽度三个频段的 3D 旋转位置编码 |
| Dynamic FPS | “智能采样” | 根据运动程度、时长和 token 预算为每段视频选择帧采样率 |
| Absolute time token | “时间戳 token” | 与帧 token 交错的 `<time>t</time>`，让模型看到真实秒数而非帧索引 |
| Window attention | “局部注意力” | 为提速把空间自注意力限制在小窗口内；周期性加入全局注意力 |
| Structured agent output | “JSON 模式” | 训练数据监督，教会 VLM 输出可解析的 JSON，包含坐标和工具名 |
| min_pixels / max_pixels | “分辨率边界” | Qwen2.5-VL 的每请求控制项，限制总像素数从而控制 token 数 |
| Grounding | “指出来” | 以文本 token 形式输出边界框坐标；自 Qwen-VL v1 起使用 |

## 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
