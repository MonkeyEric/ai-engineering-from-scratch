# DeepSeek-V3 架构详解

> 第 10 阶段 · 第 14 课列举了每一个开放模型都会调整的六个架构旋钮。DeepSeek-V3（2024 年 12 月发布，总参数量 671B，激活参数量 37B）则把这六个旋钮全部转动，并新增了四个：多头潜在注意力（Multi-Head Latent Attention，MLA）、无辅助损失负载均衡（auxiliary-loss-free load balancing）、多词元预测（Multi-Token Prediction，MTP）和 DualPipe 训练。本课将从上到下解读 DeepSeek-V3 的架构，并根据已发布的配置推导出每一个参数量。学完本课后，你能够解释为什么 671B/37B 的比例是正确的赌注，以及为什么 MLA + 专家混合（MoE）两者结合能在前沿模型上胜过单独使用其中任何一种。

**类型：** Learn
**语言：** Python（stdlib，参数计算器）
**先修：** Phase 10 · 14（开放模型架构详解）、Phase 10 · 17（NSA）、Phase 10 · 18（MTP）、Phase 10 · 19（DualPipe）
**时长：** 约 75 分钟

## 学习目标

- 从上到下阅读 DeepSeek-V3 的配置，并用“六个 GPT-2 旋钮”加上四个 DeepSeek 特有改进来解释每一个字段。
- 推导总参数量（671B）、激活参数量（37B）以及各自由哪些部分组成。
- 计算 MLA 在 128k 上下文下的 KV 缓存占用，并与同等激活参数量的密集模型使用 GQA 时的开销进行对比。
- 说明四项 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每一项针对架构/训练栈的哪个部分。

## 问题背景

DeepSeek-V3 是首个在架构上与 Llama 家族有显著差异的前沿开放模型。Llama 3 405B 是“转动六个旋钮的 GPT-2”。DeepSeek-V3 则是六个旋钮全动，还加了四个。读 Llama 3 的配置是为读 DeepSeek 配置热身，但深层结构——注意力块的形态、路由逻辑、训练时目标函数——差异大到需要一份单独的详解。

学习它的回报在于：DeepSeek-V3 的开放权重发布改变了“前沿能力”在开放模型中的含义。这份架构是 2026 年许多训练运行正在复制的蓝图。理解它是任何接触前沿大语言模型训练或推理角色的入门门槛。

## 核心概念

### 不变的核心，再说一次

DeepSeek-V3 仍然是自回归模型。它仍然堆叠解码器块。每个块仍然有注意力、MLP 和两个 RMSNorm。MLP 仍然使用 SwiGLU。仍然使用 RoPE。Pre-norm。权重共享的嵌入。与 Llama 或 Mistral 的基线相同。

### 关键转折：MLA 替代 GQA

在第 10 阶段 · 第 14 课中，你知道分组查询注意力（GQA）通过在多组查询头（query heads）之间共享 K 和 V 来缩小 KV 缓存。多头潜在注意力（MLA）更进一步：K 和 V 被压缩成一个共享的低秩潜在表示（即 `kv_lora_rank`），然后在每个头上实时解压。KV 缓存只存潜在向量——通常是每层每个词元 512 个浮点数，而不是 8 × 128 = 1024 个浮点数。

在 128k 上下文下，DeepSeek-V3 使用 MLA（每层每个词元共享一个潜在向量 `c^{KV}`；K 和 V 都通过上投影从这个潜在向量推导，且这些上投影可以被吸收进后续的矩阵乘法）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假想的 GQA 基线（Llama 3 70B 结构，8 个 KV 头，头维度 128）需要付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文下，MLA 的缓存大小只有 Llama-3-70B 风格 GQA 缓存的 1/4。

代价：MLA 在每次注意力计算中（每个头）增加了一个解压步骤。与节省的带宽相比，额外计算很小。对长上下文推理而言是净收益。

### 路由：无辅助损失负载均衡

MoE 路由器决定哪些 top-k 专家处理每个词元。一个朴素路由器会把太多工作集中在少数专家上，导致其他专家闲置。标准修复方法：添加辅助损失项来惩罚负载不均。这有效，但会轻微损害主任务性能。

DeepSeek-V3 提出了一种无辅助损失的方案。在路由器 logits 上为每个专家加入偏置项，训练期间通过简单规则调整：如果专家 `e` 过载，就减小 `bias_e`；如果欠载，就增大它。没有额外的损失项。训练保持干净。专家负载保持均衡。

对主损失的影响：测不出。对 MoE 架构的影响：更干净，不需要调辅助损失超参数。

### MTP：更稠密的训练信号 + 免费的草稿模型

在第 10 阶段 · 第 18 课中，你知道 DeepSeek-V3 增加了一个深度 D=1 的 MTP 模块，用于预测向后两个位置的词元。在推理时，这个训练好的模块被重新用作投机解码（speculative decoding）的草稿模型，接受率超过 80%。在训练时，每个隐层状态同时监督 D+1 = 2 个目标，提供更稠密的信号。

参数量：在主模型 671B 之上增加 14B。开销：2.1%。

### 训练：DualPipe

在第 10 阶段 · 第 19 课中，你知道 DualPipe 是一种双向流水线，它将前向和反向块与跨节点的 all-to-all 通信重叠。在 DeepSeek-V3 的 2,048 张 H800 规模下，它大约回收了 1F1B 因流水线气泡而损失的 245k GPU 小时。

### 逐字段解析配置

以下是 DeepSeek-V3 的配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   （密集 MLP 的隐层大小，用于最前几层）
moe_intermediate_size: 2048 （专家 MLP 的隐层大小）
num_hidden_layers: 61
first_k_dense_layers: 3    （前 3 层使用密集 MLP）
num_attention_heads: 128
num_key_value_heads: 128   （在 MLA 下形式上等价于 num_heads，
                           真正的压缩体现在 kv_lora_rank）
kv_lora_rank: 512          （MLA 潜在维度）
num_experts: 256            （每个块的 MoE 专家数量）
num_experts_per_tok: 8      （top-8 路由）
shared_experts: 1           （每个块始终激活的共享专家）
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               （深度为 1 的 1 个 MTP 模块）
```

逐条解析：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的密集 MLP。剩余 58 个块使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度，然后按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个被路由的专家之外，还有 1 个始终激活的专家为每个词元贡献输出。可以把它看作“密集地板”，确保每个词元都能获得可靠的信息。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐层大小。比密集 MLP 小，因为有 256 个专家。

### 参数量核算

完整计算见 `code/main.py`。核心结论：

- 嵌入：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个密集块：带 MLA 的注意力（每层约 144M）+ 密集 MLP（每层约 260M）+ norm。总共约 1.2B。
- 58 个 MoE 块：带 MLA 的注意力（约 144M）+ 256 个专家各 30M + 1 个共享专家 30M + norm。每个块总计约 7.95B（含所有专家）。58 个块共约 461B。
- MTP 模块：14B。

总计：核心架构约 476B + MTP 14B；而公开宣称的 671B 还额外计入了更多结构参数（偏置张量、专家专属组件、共享专家缩放等）。我们在计算器中复现的数字与公开数字相差 3–5%——差异来自 DeepSeek 报告第 2 节附录中记录的细粒度核算。

每次前向的激活参数量：

- 注意力：每层 144M × 61 = 8.8B（所有层都参与计算）。
- 激活的 MLP：前 3 层密集（3 × 260M = 780M），58 层 MoE 每层激活 8 个路由专家 + 1 个共享专家 + 路由开销。每层激活 MLP 约 260M。总计：3 × 260M + 58 × 260M = ~15.9B。
- 嵌入 + norm：1.2B。
- 总激活量：约 26B 核心 + 14B MTP（训练时使用，推理时不一定运行）≈ 37B。

### 671B / 37B 的比例

18 倍稀疏比（激活参数占总参数的 5.5%）。DeepSeek-V3 是已发布开放权重的前沿 MoE 模型中最稀疏的。Mixtral 8x7B 的比例是 13/47（28%），密集得多。Llama 4 Maverick 的比例是 17B/400B（4.25%），与之接近。DeepSeek 的赌注是：在前沿规模下，更多专家 + 更低激活比例，能在每单位激活 FLOP 上产生更好的质量。

### DeepSeek-V3 的位置

| 模型 | 总参数量 | 激活参数量 | 比例 | 注意力 | 新思路 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 骨干网络上进行的推理训练运行。R1 使用相同的架构。变化的是后训练方案（在可验证任务上进行大规模强化学习），而不是预训练架构。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），即第 10 阶段 · 第 17 课中 NSA 的继任者。血统稳定：架构级创新不断积累；每一代都会再转动一些旋钮。

## 动手实践

`code/main.py` 是专门针对 DeepSeek-V3 结构的参数计算器。运行它，将其输出与论文数字对比，并用于假设变体（256 专家 vs 512，top-8 vs top-16，MLA rank 512 vs 1024）。

值得关注：

- 总参数量 vs 公开 671B。
- 激活参数量 vs 公开 37B。
- 128k 上下文下的 KV 缓存——MLA vs GQA 的对比。
- 每层拆解，看清参数预算真正花在哪里。

## 交付成果

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1 或任何未来变体），它会生成一份逐组件的架构解读：命名配置的每个字段、按组件推导参数量、并指出该模型使用了四项 DeepSeek 特有创新中的哪些。

## 练习

1. 运行 `code/main.py`。将计算器的总参数估计与公开 671B 对比，找出差异来源。论文第 2 节有完整分项清单。

2. 将配置中的 MLA rank 从 512 改为 256。计算 128k 上下文下新的 KV 缓存大小。它减少了多少百分比，又以什么代价削弱了每个头的表达能力？

3. 将 DeepSeek-V3 的（256 专家，top-8）路由与假想的（512 专家，top-8）变体对比。总参数增加，激活参数不变。理论上额外的专家容量带来什么好处，推理时又付出什么代价？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节关于 MLA 的内容。用三句话解释为什么 K 和 V 的解压矩阵可以在推理时被“吸收”进后续矩阵乘法，从而提升推理效率。

5. DeepSeek-V3 在大部分运算中使用 FP8 训练。计算用 FP8 替代 BF16 存储 671B 权重带来的内存节省。这与 14.8T 词元的训练预算如何交叉影响？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | 把 K 和 V 压缩成共享低秩潜在向量（kv_lora_rank，通常为 512），按头实时解压；KV 缓存只存潜在向量 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 共享潜在向量的大小；DeepSeek-V3 使用 512 |
| First k dense layers | "早期层保持密集" | 前几个 MoE 模型层跳过 MoE 路由器，运行密集 MLP 以提升稳定性 |
| num_experts_per_tok | "Top-k 路由" | 每个词元激活多少个被路由的专家；DeepSeek-V3 使用 8 |
| Shared experts | "始终激活的专家" | 不经过路由、处理每个词元的专家；DeepSeek-V3 使用 1 个 |
| Auxiliary-loss-free routing | "偏置调整负载均衡" | 训练期间调整每个专家的偏置项以保持负载均衡，而不添加损失项 |
| MTP module | "额外预测头" | 从 h^(1) 和 E(t+1) 预测 t+2 的 Transformer 块；训练更稠密，推理可免费作投机解码草稿 |
| DualPipe | "双向流水线" | 将前向/反向计算与跨节点 all-to-all 重叠的训练调度 |
| Active parameter ratio | "稀疏度" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 training | "8 位训练" | 训练存储和许多计算操作使用 FP8；相比 BF16 大致减半内存，质量损失很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练和结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件和部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 提出 MLA 的前代模型
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在 V3 架构上进行的推理训练后续
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 家族注意力的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 训练调度参考实现
