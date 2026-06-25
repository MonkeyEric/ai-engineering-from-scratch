# StyleGAN

> 大多数生成器会同时把 `z` 灌入每一层。StyleGAN 把它拆开：先把 `z` 映射到一个中间表示 `w`，再通过 AdaIN 在每一种分辨率层级*注入* `w`。仅此一处改动就解耦了潜在空间，并让逼真的人脸生成在七年时间里成了“已解决的问题”。

**类型：** 实践构建
**语言：** Python
**前置课程：** Phase 8 · 03（GANs）、Phase 4 · 08（Normalization）、Phase 3 · 07（CNNs）
**时长：** 约 45 分钟

## 问题背景

DCGAN 通过一系列转置卷积把 `z` 映射成图像。问题在于：`z` 同时控制姿态、光照、身份、背景——所有因素纠缠（entangle）在一起。沿着 `z` 的某一轴移动，这四项都会变。你无法让模型做到“同一个人、换种姿势”，因为它的表示方式并非如此分解。

Karras 等人（2019，NVIDIA）提出：别再直接把 `z` 喂进卷积层。改用一个固定的 `4×4×512` 张量作为网络输入；学习一个 8 层 MLP，把 `z ∈ Z` 映射到 `w ∈ W`；再通过*自适应实例归一化*（adaptive instance normalization，AdaIN）在每一分辨率注入 `w`：先对每个卷积特征图做归一化，再用 `w` 的仿射投影去缩放和平移。最后为每层加入噪声，以产生随机细节（皮肤毛孔、发丝）。

结果是：`W` 空间中“高层风格”（姿态、身份）与“精细风格”（光照、颜色）的轴向大致正交。你可以把图像 A 的 `w` 用于低分辨率层、图像 B 的 `w` 用于高分辨率层，从而在两张图之间交换风格。这解锁了图像编辑、跨域风格化以及整个“StyleGAN 反演（StyleGAN inversion）”研究方向。

## 核心概念

![StyleGAN：映射网络 + AdaIN + 逐层噪声](../assets/stylegan.svg)

**映射网络（mapping network）。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制约束为高斯分布——它会学习出适配数据分布的形状。

**合成网络（synthesis network）。**从一个可学习的常量 `4×4×512` 开始。每个分辨率块：`上采样 → 卷积 → AdaIN(w_i) → 噪声 → 卷积 → AdaIN(w_i) → 噪声`。分辨率逐级翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。对每个特征图做归一化后再重新赋予风格。这里的“风格”指的是特征图的一阶和二阶统计量。

**逐层噪声（per-layer noise）。**向每个特征图加入单通道高斯噪声，并按每个通道的可学习因子缩放。它在不影响全局结构的前提下控制随机细节。

**截断技巧（truncation trick）。**推理时采样 `z`，计算 `w = mapping(z)`，然后做 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是大量样本的 `w` 均值。`ψ < 1` 用多样性换取质量。几乎所有 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 主要创新 |
|---------|------|------------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进式增长（progressive growing）。 |
| StyleGAN2 | 2020 | 权重解调（weight demodulation）取代 AdaIN（修复水滴状伪影）；跳跃/残差架构；路径长度正则化（path-length regularization）。 |
| StyleGAN3 | 2021 | 无混叠卷积（alias-free convolution）+ 等变核（equivariant kernels）；消除纹理粘附到像素网格。 |
| StyleGAN-XL | 2022 | 类别条件（class-conditional）、1024²、ImageNet。 |
| R3GAN | 2024 | 以更强的正则化重新包装；在 FFHQ-1024 上缩小与扩散模型（diffusion）的差距，参数量仅为其 1/20。 |

到 2026 年，StyleGAN3 仍是以下场景的首选：（a）高帧率窄域照片级真实感生成，（b）小样本域适应（用 100 张图像在新数据集上训练并冻结映射网络），（c）基于反演的编辑（找到能重建真实照片的 `w`，再编辑该 `w`）。而在开放域文本到图像任务中，它不是合适的工具——该用扩散模型（diffusion）。

## 动手实现

`code/main.py` 实现了一个一维的玩具版“StyleGAN 简化版”：一个映射 MLP、一个接受可学习常量向量并用来自 `w` 的缩放/偏置进行调制的合成函数，以及逐层噪声。它表明：通过仿射调制注入 `w`，效果与把 `z` 拼接到生成器输入相当或更好。

### 步骤 1：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 步骤 2：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每个特征图的缩放与偏置通过线性投影从 `w` 得到。

### 步骤 3：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每个通道的 sigma 都是可学习的。

## 常见陷阱

- **水滴状伪影（droplet artifacts）。**StyleGAN 1 的特征图中会出现块状水滴，因为 AdaIN 把均值归零。StyleGAN 2 的权重解调通过改为缩放卷积权重来修复。
- **纹理粘附（texture sticking）。**StyleGAN 1 和 2 的纹理跟随像素坐标而非物体坐标（插值时可见）。StyleGAN 3 的无混叠卷积通过加窗 sinc 滤波器修复。
- **模式覆盖（mode coverage）。**截断 `ψ < 0.7` 看起来干净，但只从一个狭窄的锥形区域采样；若需要多样性，应使用 `ψ = 1.0`。
- **反演（inversion）是有损的。**把真实照片反演到 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。结果在多次迭代后会漂移。

## 应用场景

| 应用场景 | 方案 |
|----------|----------|
| 照片级真实人脸（动漫、产品、窄域） | StyleGAN3 FFHQ / 自定义微调 |
| 基于照片的人脸编辑 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 换脸 / 表情重演 | StyleGAN + 编码器 + 融合 |
| 头像流水线 | StyleGAN3 配合 ADA 进行低数据微调 |
| 小样本域适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 不要用它——用扩散模型 |

对于那些答案就是“一个人脸照片”的产品级演示，StyleGAN 在推理成本（单次前向传播，在 4090 上小于 10ms）和同等质量下的清晰度方面都优于扩散模型。

## 交付成果

保存 `outputs/skill-stylegan-inversion.md`。该技能接收一张真实照片并输出：反演方法（e4e / ReStyle / HyperStyle）、预期的潜在空间损失（latent loss）、编辑预算（在 `W` 中还能走多远才会出现伪影），以及一组已知可用的编辑方向（年龄、表情、姿态）。

## 练习

1. **简单。**分别用 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较固定潜在变量与扰动潜在变量下输出的分散程度。
2. **中等。**实现混合正则化：在一个训练批次中计算 `w_a`、`w_b`，前半段合成使用 `w_a`，后半段使用 `w_b`。解码器能否学到解耦的风格？
3. **困难。**加载预训练的 StyleGAN3 FFHQ 模型（`ffhq-1024.pkl`）。在带标签样本上训练一个 SVM，找到控制“微笑”的 `w` 方向；报告在身份开始漂移前你能把它推多远。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Mapping network | “那个 MLP” | `f: Z → W`，8 层，把潜在空间几何与数据统计解耦。 |
| W space | “风格空间” | 映射网络的输出；大致已解耦。 |
| AdaIN | “自适应实例归一化” | 对特征图归一化，再按 `w` 投影缩放 + 平移。 |
| Truncation trick | “Psi” | `w = mean + ψ·(w - mean)`，ψ<1 用多样性换质量。 |
| Path-length regularization | “PL 正则” | 惩罚 `w` 单位变化导致图像变化过大；让 `W` 更平滑。 |
| Weight demodulation | “StyleGAN2 的修复” | 归一化卷积权重而非激活值；消除水滴状伪影。 |
| Alias-free | “StyleGAN3 的 trick” | 加窗 sinc 滤波器；消除纹理粘附到像素网格。 |
| Inversion | “给真实图像找 w” | 优化或编码 `x → w`，使得 `G(w) ≈ x`。 |

## 生产环境注：为何 2026 年仍在部署 StyleGAN

StyleGAN3 在 4090 上生成一张 1024² 的 FFHQ 人脸不到 10 毫秒——`num_steps = 1`，没有 VAE 解码，也没有交叉注意力（cross-attention） pass。在生产环境里，这是所有图像生成器的延迟下限。同样分辨率下，50 步 SDXL + VAE 解码流水线约需 3 秒。这有 **300 倍**的差距，对于窄域产品（头像服务、证件照流水线、库存人脸生成），它在总拥有成本（TCO）上更具优势。

两个运维层面的结果：

- **无需调度器，也无需批处理调度器（batcher）。**在目标占用率下使用静态批次（static batch）即为最优。连续批处理（continuous batching，对 LLM 和扩散模型至关重要）在这里没有收益，因为每个请求的 FLOPs 完全相同。
- **截断 `ψ` 就是安全旋钮。**`ψ < 0.7` 时从映射网络范围的一个狭窄锥形中采样。这是服务层控制样本方差的唯一杠杆。高峰期降低 `ψ`，为付费用户提高 `ψ`。

## 延伸阅读

- [Karras 等人（2019）。基于风格的 GAN 生成器架构](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras 等人（2020）。分析并改进 StyleGAN 的图像质量](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras 等人（2021）。无混叠生成对抗网络](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov 等人（2021）。为 StyleGAN 图像操控设计编码器](https://arxiv.org/abs/2102.02766) — e4e 反演。
- [Sauer 等人（2022）。StyleGAN-XL：将 StyleGAN 扩展到大规模多样化数据集](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang 等人（2024）。R3GAN：GAN 已死，GAN 万岁！](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 方案。
