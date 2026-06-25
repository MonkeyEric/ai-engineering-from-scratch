# 评估指标 — FID、CLIP Score 与人类偏好

> 每个生成模型排行榜都会引用 FID、CLIP score 和人类偏好竞技场的胜率。但每个数字都有可被“有心人”利用的失效模式。如果你不了解这些失效模式，就无法区分真正的改进与刷分跑分。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 8 · 01（Taxonomy），Phase 2 · 04（Evaluation Metrics）
**时间：** 约 45 分钟

## 问题背景

生成模型的评判标准是*样本质量（sample quality）*与*条件遵循度（conditioning adherence）*，但两者都没有闭合形式的度量方法。你的模型要生成 10,000 张图像；必须有什么东西给它们打分；你还要在不同模型家族、不同分辨率、不同架构之间相信这些数字。在 2014–2026 年的大浪淘沙中，有三个指标存活了下来：

- **FID（Fréchet Inception Distance）。** 在 Inception 网络特征空间中，真实分布与生成分布之间的距离。数值越低越好。
- **CLIP score。** 生成图像的 CLIP 图像嵌入（embedding）与提示文本的 CLIP 文本嵌入之间的余弦相似度。数值越高越好，衡量提示遵循度。
- **人类偏好（human preference）。** 让两个模型在同一提示下正面交锋，由人类（或 GPT-4 级别的模型）挑选更好的一张，汇总成 Elo 分数。

你还会见到：IS（Inception Score，已基本退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一个都是为了修正前一个指标的某种失效模式而生。

## 核心概念

![FID、CLIP 与偏好：三个维度，各自的失效模式](../assets/evaluation.svg)

### FID — 样本质量

Heusel 等人（2017）。步骤如下：

1. 为 N 张真实图像和 N 张生成图像提取 Inception-v3 特征（2048 维）。
2. 对每个池拟合一个高斯分布：计算均值 `μ_r, μ_g` 与协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解读：特征空间中两个多元高斯分布之间的 Fréchet 距离。数值越低 = 分布越相似。

失效模式：
- **小样本 N 有偏。** FID 是特征分布上的均方量 —— 小 N 会低估协方差，给出虚假的低 FID。务必使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 在 ImageNet 上训练。与 ImageNet 差异大的领域（人脸、艺术、文字图像）会产生无意义的 FID。应使用领域特定的特征提取器。
- **可被刷分。** 过拟合到 Inception 先验可以在视觉质量没有提升的情况下获得低 FID。用 CMMD（见下文）来对抗。

### CLIP score — 提示遵循度

Radford 等人（2021）。对于一张生成图像 + 一条提示：

```python
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 30k 生成图像上取平均，得到一个可在模型间比较的标量。

失效模式：
- **CLIP 自身的盲区。** CLIP 的组合推理能力较弱（“一个红色立方体放在蓝色球体上”经常失败）。模型可能在 CLIP score 上排名很高，却并未真正遵循复杂提示。
- **短提示偏差。** 短提示在野外拥有更多 CLIP-图像匹配，长提示的 CLIP score 机械性地更低。
- **提示刷分。** 在提示中加入 “high quality, 4k, masterpiece” 会抬高 CLIP score，但并不改善图文绑定。

CMMD（Jayasumana 等人，2024）修正了其中一些问题：它使用 CLIP 特征替代 Inception，用最大均值差异（maximum-mean discrepancy）替代 Fréchet 距离，对细微质量差异更敏感。

### 人类偏好 — 黄金标准

准备一组提示池。用模型 A 和模型 B 分别生成。将成对结果展示给人类（或强 LLM 裁判），把胜负汇总为 Elo 或 Bradley-Terry 分数。常见基准：

- **PartiPrompts（Google）**：1,600 条多样化提示，12 个类别。
- **HPSv2**：107k 人类标注，广泛用作自动代理指标。
- **ImageReward**：137k 提示-图像偏好对，MIT 许可证。
- **PickScore**：在 Pick-a-Pic 的 260 万偏好对上训练。
- **Chatbot Arena 风格的图像竞技场**：https://imagearena.ai/ 等。

失效模式：
- **裁判方差。** 非专家与专家的偏好不同。两者都要用。
- **提示分布。** 精心挑选的提示会偏向某个模型家族。必须记录提示来源。
- **LLM 裁判奖励 hack。** GPT-4 裁判会被“好看但错误”的输出欺骗。需用人类结果三角验证。

## 组合使用

一份生产级评估报告应包含：

1. 在 10–30k 样本上与留出真实分布对比的 FID（样本质量）。
2. 相同样本与其提示之间的 CLIP score / CMMD（提示遵循度）。
3. 与上一代模型在盲测竞技场中的胜率（整体偏好）。
4. 失效模式分析：随机抽取 50 个输出，标记已知问题（手部解剖、文字渲染、物体数量一致性）。

任何单一指标都是谎言。三个相互印证的指标 + 定性评审才构成一个“主张”。

## 动手实现

`code/main.py` 在合成“特征向量”上实现了 FID、类 CLIP score 与 Elo 聚合（我们用 4 维向量代替 Inception 特征）。你将看到：

- 小 N 与大 N 下的 FID 计算 —— 展示偏差。
- “CLIP score”作为特征池之间的余弦相似度。
- 来自合成偏好流的 Elo 更新规则。

### 第一步：四行实现 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 第二步：类 CLIP 的余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 第三步：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 常见陷阱

- **N=1000 的 FID。** 经验法则：N 低于 10k 不可靠。报告低 N FID 的论文大多在刷分。
- **跨分辨率比较 FID。** Inception 的 299×299  resize 会改变特征分布。只应在相同分辨率下比较。
- **只报告一个种子。** 至少运行 3 个种子，报告标准差。
- **通过负提示抬高 CLIP score。** 某些流水线通过过度拟合提示来提升 CLIP，需检查视觉是否过饱和。
- **Elo 的提示重叠偏差。** 如果两个模型在训练时都见过某条基准提示，Elo 就失去意义。请使用留出的提示集。
- **人类评估的付费众包偏差。** Prolific、MTurk 标注者往往更年轻、更偏向科技圈。应混合招募艺术/设计专家。

## 实际应用

2026 年的生产级评估协议：

| 支柱 | 最低要求 | 推荐方案 |
|------|---------|-------------|
| 样本质量 | 10k FID vs 留出真实分布 | + 5k CMMD + 按类别子集 FID |
| 提示遵循度 | 30k CLIP score | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好 | 200 对盲测 vs 基线 | + 2000 对人类 + LLM 裁判 + Chatbot Arena |
| 失效分析 | 50 条人工标记 | 500 条人工标记 + 自动安全分类器 |

一份报告同时包含四根支柱 = 主张。只有一根 = 营销。

## 交付成果

保存 `outputs/skill-eval-report.md`。该技能接收一个新模型检查点 + 基线，输出完整评估计划：样本量、指标、失效模式探针、签署标准。

## 练习题

1. **简单。** 运行 `code/main.py`。在同一合成分布上比较 N=100 与 N=1000 的 FID。报告偏差幅度。
2. **中等。** 用合成 CLIP 风格特征实现 CMMD（公式见 Jayasumana 等人，2024）。比较其对质量差异的敏感度与 FID。
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 子集中取 1000 对图像-提示，在偏好上微调一个小型 CLIP 打分器，并测量其与留出集的一致性。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| FID | “Fréchet Inception Distance” | 真实与生成 Inception 特征高斯拟合之间的 Fréchet 距离。 |
| CLIP score | “图文相似度” | CLIP 图像与文本嵌入之间的余弦相似度。 |
| CMMD | “FID 的替代品” | 基于 CLIP 特征的 MMD；偏差更小，无需高斯假设。 |
| IS | “Inception Score” | Exp KL(p(y\|x) \|\| p(y))；与现代模型相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | “学习型偏好代理” | 在人类偏好上训练的小模型；用作自动裁判。 |
| Elo | “国际象棋等级分” | 对成对胜负进行 Bradley-Terry 聚合。 |
| PartiPrompts | “基准提示集” | Google 策划的 1,600 条提示，涵盖 12 个类别。 |
| FD-DINO | “自监督替代方案” | 使用 DINOv2 特征的 FD；更适合 ImageNet 外领域。 |

## 生产提示：评估本身也是一种推理负载

在 10k 样本上跑 FID 意味着要生成 10k 张图像。对单张 L4 上的 50 步 SDXL base、1024² 分辨率而言，这大约是 11 小时的单请求推理。评估预算是真实存在的，其本质正是离线推理场景（最大化吞吐，忽略 TTFT）：

- **尽量增大批次，忘记延迟。** 离线评估 = 在显存允许范围内使用最大静态批次。在 80GB H100 上用 `pipe(...).images` 配合 `num_images_per_prompt=8`，墙钟时间比单请求快 4–6 倍。
- **缓存真实特征。** 对真实参考集提取 Inception（FID）或 CLIP（CLIP score、CMMD）特征只需*一次*，存为 `.npz`。不要每次评估都重新计算。

对于 CI / 回归门禁：每个 PR 在 500 样本子集上跑 FID + CLIP score（约 30 分钟）；每晚跑完整 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失效模式综述。
