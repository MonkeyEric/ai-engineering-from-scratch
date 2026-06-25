# 学习率调度与预热

> 学习率是最重要的超参数。不是架构。不是数据集大小。不是激活函数。是学习率。如果你只调一个参数，就调这个。

**类型：** 构建  
**语言：** Python  
**前置条件：** 第03.06课（优化器），第03.08课（权重初始化）  
**时间：** 约90分钟

## 学习目标

- 从零实现常数、阶梯衰减、余弦退火、预热+余弦和1cycle学习率调度器
- 演示学习率选择的三种失效模式：发散（太高）、停滞（太低）和振荡（无衰减）
- 解释为什么基于Adam的优化器需要预热，以及它如何稳定早期训练
- 在同一任务上比较五种调度器的收敛速度，并为给定的训练预算选择合适的调度器

## 问题

将学习率设为0.1。训练发散——损失在3步内跳到无穷大。设为0.0001。训练爬行——100轮后，模型几乎没从随机状态移动。设为0.01。训练前50轮有效，然后损失在某个最小值附近振荡，永远无法到达，因为步长太大了。

最优学习率不是常数。它在训练过程中变化。早期，你需要大步子快速覆盖空间。训练后期，你需要小步子沉淀到尖锐的最小值。90%准确率的模型和95%准确率的模型之间的差异，往往就是调度。

过去三年发布的每一个主要模型都使用学习率调度。Llama 3使用峰值lr=3e-4，2000步预热，余弦衰减到3e-5。GPT-3使用lr=6e-4，预热375M token。这些不是随意选择的。它们是耗费数百万美元的广泛超参数搜索的结果。

你需要理解调度，因为默认值对你的问题不会有效。当你微调预训练模型时，正确的调度与从头训练不同。当你增加批次大小时，预热期需要改变。当训练在第10,000步崩溃时，你需要知道是调度问题还是其他问题。

## 概念

### 常数学习率

最简单的方法。选一个数，每一步都用这个数。

```
lr(t) = lr_0
```

很少是最优的。对于训练末期来说要么太高（在最小值附近振荡），要么对于训练初期来说太低（在小步子上浪费算力）。对小模型和调试来说还行。对于训练时间超过一小时的任何东西来说，这是一个糟糕的选择。

### 阶梯衰减

ResNet时代的传统方法。在固定的轮次将学习率乘以一个因子（通常是10倍）。

```
lr(t) = lr_0 * gamma^(floor(epoch / step_size))
```

其中gamma=0.1且step_size=30意味着：每30轮学习率下降10倍。ResNet-50使用这个——lr=0.1，在第30、60和90轮下降10倍。

问题：最优衰减点取决于数据集和架构。换一个问题，你就需要重新调整何时衰减。过渡是突变的——当学习率突然变化时，损失可能会飙升。

### 余弦退火

从最大学习率到最小学习率的平滑衰减，遵循余弦曲线：

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

其中t是当前步数，T是总步数。

在t=0时，余弦项为1，所以lr = lr_max。在t=T时，余弦项为-1，所以lr = lr_min。衰减在开始时温和，中间加速，接近结束时再次变得温和。

这是大多数现代训练运行的默认选择。除了lr_max和lr_min之外没有需要调整的超参数。余弦形状匹配了大多数学习发生在训练中期的经验观察——你希望在那个关键时期有合理的步长。

### 预热：为什么从小开始

Adam和其他自适应优化器维护梯度均值和方差的运行估计。在第0步，这些估计被初始化为零。最初的几次梯度更新基于垃圾统计。如果在此期间你的学习率很大，模型会迈出巨大的、方向不明的步长。

预热解决了这个问题。从一个小学习率开始（通常是lr_max / warmup_steps甚至为零），在前N步线性上升到lr_max。当你达到完整学习率时，Adam的统计量已经稳定下来。

```
lr(t) = lr_max * (t / warmup_steps)     for t < warmup_steps
```

典型预热：总训练步数的1-5%。Llama 3训练了约1.8万亿token，预热了2000步。GPT-3预热了3.75亿token。

### 线性预热+余弦衰减

现代默认。线性上升，然后余弦衰减：

```
if t < warmup_steps:
    lr(t) = lr_max * (t / warmup_steps)
else:
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
```

这就是Llama、GPT、PaLM和大多数现代Transformer使用的。预热防止早期不稳定。余弦衰减将模型沉淀到一个好的最小值。

### 1cycle策略

Leslie Smith的发现（2018）：在训练的前半段将学习率从低值升高到高值，然后在后半段再降回来。违反直觉——为什么要在训练中途*增加*学习率？

理论：高学习率通过向优化轨迹添加噪声来充当正则化。模型在上升阶段探索更多损失曲面，找到更好的盆地。下降阶段则在找到的最佳盆地内精炼。

```
阶段1（0到T/2）：    lr从lr_max/25上升到lr_max
阶段2（T/2到T）：    lr从lr_max下降到lr_max/10000
```

在固定的算力预算下，1cycle通常比余弦退火训练得更快。权衡：你必须提前知道总步数。

### 调度形状

```mermaid
graph LR
    subgraph "常数"
        C1["lr"] --- C2["lr"] --- C3["lr"]
    end

    subgraph "阶梯衰减"
        S1["0.1"] --- S2["0.1"] --- S3["0.01"] --- S4["0.001"]
    end

    subgraph "余弦退火"
        CS1["lr_max"] --> CS2["逐渐"] --> CS3["陡峭"] --> CS4["lr_min"]
    end

    subgraph "预热+余弦"
        WC1["0"] --> WC2["lr_max"] --> WC3["余弦"] --> WC4["lr_min"]
    end
```

### 决策流程图

```mermaid
flowchart TD
    Start["选择LR调度"] --> Know{"知道总<br/>训练步数？"}

    Know -->|"是"| Budget{"算力预算？"}
    Know -->|"否"| Constant["使用常数LR<br/>加手动衰减"]

    Budget -->|"大（数天/数周）"| WarmCos["预热+余弦衰减<br/>（Llama/GPT默认）"]
    Budget -->|"小（数小时）"| OneCycle["1cycle策略<br/>（收敛最快）"]
    Budget -->|"中等"| Cosine["余弦退火<br/>（安全默认）"]

    WarmCos --> Warmup["预热 = 步数的1-5%"]
    OneCycle --> FindLR["用LR范围测试找lr_max"]
    Cosine --> MinLR["设置lr_min = lr_max / 10"]
```

### 已发表模型的实际数字

```mermaid
graph TD
    subgraph "已发表的LR配置"
        L3["Llama 3（405B）<br/>峰值: 3e-4<br/>预热: 2000步<br/>调度: 余弦到3e-5"]
        G3["GPT-3（175B）<br/>峰值: 6e-4<br/>预热: 375M token<br/>调度: 余弦到0"]
        R50["ResNet-50<br/>峰值: 0.1<br/>预热: 无<br/>调度: 30,60,90轮阶梯衰减x0.1"]
        B["BERT（340M）<br/>峰值: 1e-4<br/>预热: 10K步<br/>调度: 线性衰减"]
    end
```

## 动手实现

### 第1步：调度函数

每个函数接受当前步数并返回该步的学习率。

```python
import math


def constant_schedule(step, lr=0.01, **kwargs):
    return lr


def step_decay_schedule(step, lr=0.1, step_size=100, gamma=0.1, **kwargs):
    return lr * (gamma ** (step // step_size))


def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=1e-5, **kwargs):
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def warmup_cosine_schedule(step, lr=0.01, total_steps=1000, warmup_steps=100, lr_min=1e-5, **kwargs):
    if total_steps <= warmup_steps:
        return lr * (step / max(warmup_steps, 1))
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def one_cycle_schedule(step, lr=0.01, total_steps=1000, **kwargs):
    mid = max(total_steps // 2, 1)
    if step < mid:
        return (lr / 25) + (lr - lr / 25) * step / mid
    else:
        progress = (step - mid) / max(total_steps - mid, 1)
        return lr * (1 - progress) + (lr / 10000) * progress
```

### 第2步：可视化所有调度

打印一个基于文本的图表，显示每种调度在训练过程中的演变。

```python
def visualize_schedule(name, schedule_fn, total_steps=500, **kwargs):
    steps = list(range(0, total_steps, total_steps // 20))
    if total_steps - 1 not in steps:
        steps.append(total_steps - 1)

    lrs = [schedule_fn(s, total_steps=total_steps, **kwargs) for s in steps]
    max_lr = max(lrs) if max(lrs) > 0 else 1.0

    print(f"\n{name}:")
    for s, lr_val in zip(steps, lrs):
        bar_len = int(lr_val / max_lr * 40)
        bar = "#" * bar_len
        print(f"  第 {s:4d} 步: lr={lr_val:.6f} {bar}")
```

### 第3步：训练网络

一个在圆形数据集上的简单两层网络，与之前的课程相同，但现在我们改变调度。

```python
import random


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    return max(0.0, x)


def relu_deriv(x):
    return 1.0 if x > 0 else 0.0


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


def train_with_schedule(schedule_fn, schedule_name, data, epochs=300, base_lr=0.05, **kwargs):
    random.seed(0)
    hidden_size = 8
    total_steps = epochs * len(data)

    std = math.sqrt(2.0 / 2)
    w1 = [[random.gauss(0, std) for _ in range(2)] for _ in range(hidden_size)]
    b1 = [0.0] * hidden_size
    w2 = [random.gauss(0, std) for _ in range(hidden_size)]
    b2 = 0.0

    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for x, target in data:
            lr = schedule_fn(step, lr=base_lr, total_steps=total_steps, **kwargs)

            z1 = []
            h = []
            for i in range(hidden_size):
                z = w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
                z1.append(z)
                h.append(relu(z))

            z2 = sum(w2[i] * h[i] for i in range(hidden_size)) + b2
            out = sigmoid(z2)

            error = out - target
            d_out = error * out * (1 - out)

            for i in range(hidden_size):
                d_h = d_out * w2[i] * relu_deriv(z1[i])
                w2[i] -= lr * d_out * h[i]
                for j in range(2):
                    w1[i][j] -= lr * d_h * x[j]
                b1[i] -= lr * d_h
            b2 -= lr * d_out

            total_loss += (out - target) ** 2
            if (out >= 0.5) == (target >= 0.5):
                correct += 1
            step += 1

        avg_loss = total_loss / len(data)
        accuracy = correct / len(data) * 100
        epoch_losses.append(avg_loss)

    return epoch_losses
```

### 第4步：比较所有调度

使用每种调度训练相同的网络，比较最终损失和收敛行为。

```python
def compare_schedules(data):
    configs = [
        ("常数", constant_schedule, {}),
        ("阶梯衰减", step_decay_schedule, {"step_size": 15000, "gamma": 0.1}),
        ("余弦", cosine_schedule, {"lr_min": 1e-5}),
        ("预热+余弦", warmup_cosine_schedule, {"warmup_steps": 3000, "lr_min": 1e-5}),
        ("1cycle", one_cycle_schedule, {}),
    ]

    print(f"\n{'调度':<20} {'起始损失':>12} {'中期损失':>12} {'最终损失':>12} {'最佳损失':>12}")
    print("-" * 70)

    for name, schedule_fn, extra_kwargs in configs:
        losses = train_with_schedule(schedule_fn, name, data, epochs=300, base_lr=0.05, **extra_kwargs)
        mid_idx = len(losses) // 2
        best = min(losses)
        print(f"{name:<20} {losses[0]:>12.6f} {losses[mid_idx]:>12.6f} {losses[-1]:>12.6f} {best:>12.6f}")
```

### 第5步：LR太高 vs 太低

演示三种失效模式：太高（发散）、太低（爬行）和恰到好处。

```python
def lr_sensitivity(data):
    learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]

    print("\nLR敏感性（常数调度，100轮）：")
    print(f"  {'LR':>10} {'起始损失':>12} {'最终损失':>12} {'状态':>15}")
    print("  " + "-" * 52)

    for lr in learning_rates:
        losses = train_with_schedule(constant_schedule, f"lr={lr}", data, epochs=100, base_lr=lr)
        start = losses[0]
        end = losses[-1]

        if end > start or math.isnan(end) or end > 1.0:
            status = "发散"
        elif end > start * 0.9:
            status = "几乎没动"
        elif end < 0.15:
            status = "已收敛"
        else:
            status = "学习中"

        end_str = f"{end:.6f}" if not math.isnan(end) else "NaN"
        print(f"  {lr:>10.4f} {start:>12.6f} {end_str:>12} {status:>15}")
```

## 如何使用（PyTorch）

PyTorch 在 `torch.optim.lr_scheduler` 中提供了调度器：

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

for step in range(1000):
    loss = train_step(model, optimizer)
    scheduler.step()
```

对于预热+余弦，使用 lambda 调度器或 HuggingFace 的 `get_cosine_schedule_with_warmup`：

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000,
)
```

HuggingFace 函数是大多数 Llama 和 GPT 微调脚本使用的。当不确定时，使用预热+余弦，预热 = 总步数的 3-5%。它几乎对一切都有效。

## 交付内容

本课程产出：
- `outputs/prompt-lr-schedule-advisor.md` —— 一个为你的训练设置推荐正确学习率调度和超参数的提示

## 练习

1. 实现指数衰减：lr(t) = lr_0 * gamma^t，其中 gamma = 0.999。在圆形数据集上与余弦退火进行比较。

2. 实现学习率范围测试（Leslie Smith）：训练几百步，同时将 LR 从 1e-7 指数增加到 1。绘制损失 vs LR 的图表。最优最大 LR 就在损失开始增加之前。

3. 使用预热+余弦训练，但改变预热长度：总步数的 0%、1%、5%、10%、20%。找到训练最稳定的最佳点。

4. 实现带热重启的余弦退火（SGDR）：每 T 步将学习率重置为 lr_max 并再次衰减。在更长的训练运行中与标准余弦进行比较。

5. 构建一个“调度外科医生”，监控训练损失，当损失稳定时自动从预热切换到余弦，当损失停滞太久时降低 lr。

## 关键术语

| 术语 | 人们常说的话 | 实际含义 |
|------|----------------|----------------------|
| 学习率 | “模型学得有多快” | 乘以梯度以确定参数更新大小的标量 |
| 调度 | “随时间改变LR” | 将训练步数映射到学习率的函数，旨在优化收敛 |
| 预热 | “从小LR开始” | 在前N步将LR从接近零线性上升到目标值，以稳定优化器统计量 |
| 余弦退火 | “平滑LR衰减” | 在训练过程中按照余弦曲线从lr_max到lr_min降低LR |
| 阶梯衰减 | “在里程碑处降低LR” | 在固定的轮次间隔将LR乘以一个因子（通常为0.1） |
| 1cycle策略 | “先升后降” | Leslie Smith的方法，在单个周期内先升后降LR以实现更快收敛 |
| LR范围测试 | “找到最佳学习率” | 短暂训练同时增加LR，以找到损失开始发散的值 |
| 带热重启的余弦 | “重置并重复” | 周期性地将LR重置为lr_max并再次衰减（SGDR） |
| Eta min | “LR的底线” | 调度衰减到的最小学习率 |
| 峰值学习率 | “最大LR” | 训练期间达到的最高LR，通常在预热之后 |

## 进一步阅读

- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017) —— 引入了余弦退火和热重启
- Smith, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (2018) —— 1cycle策略论文
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023) —— 记录了大规模使用的预热+余弦调度
- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017) —— 大批量训练的线性缩放规则和预热