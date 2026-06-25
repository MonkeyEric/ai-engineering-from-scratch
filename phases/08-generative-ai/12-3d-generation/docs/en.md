# 三维生成

> 三维（3D）是二维到三维杠杆效应最强的模态。2023 年的突破是三维高斯泼溅（3D Gaussian Splatting）。2024–2026 年的生成式推进在此基础上叠加了多视图扩散（multi-view diffusion）+ 三维重建，从而仅凭一段提示词或一张照片就能生成物体与场景。

**类型：** 学习  
**语言：** Python  
**前置条件：** Phase 4（视觉）、Phase 8 · 07（潜空间扩散 / Latent Diffusion）  
**时间：** ~45 分钟

## 问题背景

三维内容制作很痛苦：

- **表示方式。** 网格（meshes）、点云（point clouds）、体素网格（voxel grids）、有向距离场（signed distance fields, SDFs）、神经辐射场（Neural Radiance Fields, NeRFs）、三维高斯（3D Gaussians），各有取舍。
- **数据稀缺。** ImageNet 有 1400 万张图像；最大的干净三维数据集 Objaverse-XL（2023）约有 1000 万个物体，且大多质量低下。
- **内存。** 一个 512³ 的体素网格就有 1.28 亿个体素；一个有用的场景 NeRF 每条光线需要 100 万次采样。生成比重建更难。
- **监督信号。** 二维图像有像素真值；三维通常只有少量二维视角，必须从中“提升”出三维。

2026 年的技术栈把问题拆成两步。首先，用扩散模型生成*二维多视图图像*；其次，将*三维表示*（通常是高斯泼溅）拟合到这些图像上。

## 核心概念

![三维生成：多视图扩散 + 三维重建](../assets/3d-generation.svg)

### 表示方式：三维高斯泼溅（Kerbl 等，2023）

将场景表示为约 100 万个三维高斯的集合。每个高斯有 59 个参数：位置（3）、协方差（6，或四元数 4 + 缩放 3）、不透明度（1）、球谐颜色（degree 3 时 48，degree 0 时 3）。

渲染 = 投影 + alpha 合成。速度很快（在 4090 上 1080p 约 100 fps）。可微。通过梯度下降拟合真实照片。一个场景在消费级 GPU 上 5–30 分钟即可拟合完成。

2023–2024 年的两项创新：
- **生成式高斯泼溅。** LGM、LRM、InstantMesh 等模型直接通过一张或少量图像预测高斯云。
- **四维高斯泼溅（4D Gaussian Splatting）。** 高斯带每帧偏移，用于动态场景。

### 多视图扩散

在预训练图像扩散模型基础上进行微调，使其根据文本提示词或单张图像生成同一物体的多个一致视角。代表工作包括 Zero123（Liu 等，2023）、MVDream（Shi 等，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出 4–16 个环绕视角，再通过高斯泼溅或 NeRF 提升到三维。

### 文本到三维（Text-to-3D）流水线

| 模型 | 输入 | 输出 | 耗时 |
|------|------|------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 小时/资产 |
| Magic3D | text | mesh + texture | ~40 分钟 |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 分钟 |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 分钟 |
| LRM (Meta, 2023) | image | triplane | ~5 秒 |
| InstantMesh (2024) | image | mesh | ~10 秒 |
| SV3D (Stability, 2024) | image | novel views | ~2 分钟 |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 分钟 |
| TripoSR (2024) | image | mesh | ~1 秒 |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 秒 |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 秒 |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 秒 |

2025–2026 方向：直接文本到网格（text-to-mesh）模型，输出带 PBR 材质、可直接进游戏引擎的网格。对于通用物体，多视图扩散中间步骤仍是表现最好的方案。

### NeRF（背景知识）

神经辐射场（Neural Radiance Field，Mildenhall 等，2020）。一个微型 MLP 接收 `(x, y, z, view direction)`，输出 `(color, density)`。通过沿光线积分来渲染。在新视角合成质量上击败基于网格的方法，但渲染速度慢 100–1000 倍。在大多数实时应用中已被高斯泼溅取代，但在研究中仍占主导。

## 动手实现

`code/main.py` 实现了一个玩具版的二维“高斯泼溅”拟合：用若干二维高斯泼溅之和表示一张合成目标图像（平滑渐变）。通过梯度下降优化位置、颜色和协方差来匹配目标。你可以看到两个核心操作：前向渲染（泼溅 + alpha 合成）与梯度下降拟合。

### 步骤 1：二维高斯泼溅

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤 2：求和渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实三维高斯泼溅会按深度排序高斯并依次 alpha 合成。我们的二维玩具版本直接求和。

### 步骤 3：梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 常见陷阱

- **视角不一致。** 如果独立生成 4 个视角，它们在物体结构上存在分歧，三维拟合结果会模糊。解决办法：使用共享注意力的多视图扩散。
- **背面幻觉。** 单张图像 → 三维必须“脑补”不可见的背面。质量差异很大。
- **高斯泼溅爆炸。** 无约束训练会增长到 1000 万个 splat 并过拟合。致密化（densification）+ 剪枝启发式策略（来自 3D-GS 原始论文）至关重要。
- **拓扑问题。** 来自隐式场（SDFs）的网格常有孔洞或自相交。发布前需运行网格重划分（remesher，例如 Blender 的 voxel remesh）。
- **训练数据许可证。** Objaverse 许可证混杂；商业用途需按模型确认。

## 实际应用

| 任务 | 2026 年推荐 |
|------|-------------|
| 从照片重建场景 | 高斯泼溅（3DGS、Gsplat、Scaniverse） |
| 游戏的文本到三维物体 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到三维 | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 少量图像的新视角合成 | CAT3D、SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| 化身 / 着装人体 | Gaussian Avatar、HUGS |
| 研究 / SOTA | 上周刚发布的新模型 |

要在游戏或电商流水线中投产三维：Meshy 4 或 Rodin Gen-1.5 输出 PBR 网格，可直接导入 Unity / Unreal。

## 交付任务

保存 `outputs/skill-3d-pipeline.md`。该技能接收一个三维需求（输入：文本 / 单图 / 多图；输出：mesh / splat / NeRF；用途：渲染 / 游戏 / VR），并输出：流水线（多视图扩散 + 拟合，或直接网格模型）、基座模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 用 4、16、64 个高斯运行 `code/main.py`。报告最终与目标图像的 MSE。
2. **中等。** 扩展到彩色高斯（RGB）。验证重建结果匹配目标颜色模式。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片重建真实物体。报告拟合时间和在留出视角上的最终 SSIM。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------|
| 3D Gaussian Splatting | “3DGS” | 将场景表示为三维高斯云；可微 alpha 合成渲染。 |
| NeRF | “Neural radiance field” | MLP 输出三维点的颜色 + 密度；通过光线积分渲染。 |
| Triplane | “Three 2-D planes” | 将三维分解为三个轴对齐的二维特征平面；比体素更便宜。 |
| SDS | “Score distillation sampling” | 用二维扩散分数作为伪梯度来训练三维模型。 |
| Multi-view diffusion | “Many views at once” | 一次性输出一批一致相机视角的扩散模型。 |
| PBR | “Physically-based rendering” | 包含反照率、粗糙度、金属度、法线等通道的材质。 |
| Densification | “Grow splats” | 3DGS 训练启发式：在高梯度区域拆分/克隆 splat。 |

## 生产注记：三维尚未形成统一基础设施

与图像（潜空间扩散 + DiT）和视频（时空 DiT）不同，截至 2026 年三维还没有单一主导运行时。生产决策取决于表示形式：

- **NeRF / triplane。** 推理是光线步进 + 每个采样一次 MLP 前向。512² 渲染需要数百万次 MLP 前向。需积极批量化光线采样；可应用 SDPA / xformers。
- **多视图扩散 + LRM 重建。** 两阶段流水线。阶段 1（多视图 DiT）与第 07 课的扩散服务器相同。阶段 2（LRM transformer）是对各视角的一次性前向。整体延迟画像为“扩散 + 一次性”——按阶段选择对应的 serving 原语。
- **SDS / DreamFusion。** 这是每资产的优化，不是推理。应构建为离线任务，而非请求处理器。

对大多数 2026 年的产品而言，正确答案是“按需运行多视图扩散模型，异步重建为 3DGS，再实时 serving 该 3DGS”。这样能把工作负载清晰拆分为 GPU 推理服务器（快）和离线优化器（慢）。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视图扩散。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
