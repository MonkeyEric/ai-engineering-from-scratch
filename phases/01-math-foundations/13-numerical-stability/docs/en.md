# 数值稳定性

> 浮点数是一个有漏洞的抽象。它会在训练过程中咬你一口，而你根本不会预见到它的到来。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段1，第01-04课
**时间：** 约120分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的溢出、下溢和灾难性抵消
- 使用中心有限差分法验证解析梯度与数值梯度的一致性
- 解释为什么 bfloat16 比 float16 更适合训练，以及损失缩放如何防止梯度下溢

## 问题描述

你的模型训练了三个小时，然后损失变成了 NaN。你加了一个打印语句。在第 9,000 步时 logits 还正常。到第 9,001 步它们变成了 `inf`。到第 9,002 步，每个梯度都是 `nan`，训练宣告死亡。

或者：你的模型训练完成了，但准确率比论文声称的低 2%。你检查了所有东西。架构匹配。超参数匹配。数据匹配。问题是论文用了 float32，而你用了 float16 却没有正确的缩放。32 位累积的舍入误差悄悄地吞噬了你的准确率。

或者：你从头实现了交叉熵损失。它在小 logits 上没问题。当 logits 超过 100 时，它返回 `inf`。softmax 溢出了，因为 `exp(100)` 已经超出了 float32 的表示范围。每个机器学习框架都用一行两行的技巧处理了这个问题。而你根本不知道这个技巧存在。

数值稳定性不是一个理论问题。它决定了一次训练运行是成功还是静默失败。你最终会调试的每一个严重的机器学习 bug，归根结底都会落到浮点数上。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点数值。一个浮点数由三部分组成：符号位、指数和尾数（有效数字）。

```
Float32 布局（共 32 位）：
[1 符号位] [8 指数位] [23 尾数位]

数值 = (-1)^符号 * 2^(指数 - 127) * 1.尾数
```

尾数决定了精度（多少位有效数字）。指数决定了范围（数值可以多大或多小）。

```
格式      位数   指数   尾数   十进制有效数字  范围（约）
float64    64     11     52       ~15-16        +/- 1.8e308
float32    32     8      23       ~7-8          +/- 3.4e38
float16    16     5      10       ~3-4          +/- 65,504
bfloat16   16     8      7        ~2-3          +/- 3.4e38
```

float32 给你大约 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但不能区分 1.00000001 和 1.00000002。超过 7 位之后，一切都是舍入噪声。

float16 给你大约 3 位精度。它能表示的最大数值是 65,504。对于机器学习来说，这个值小得令人不安，因为 logits、梯度和激活值经常会超过这个范围。

bfloat16 是谷歌针对 float16 的范围问题给出的答案。它拥有与 float32 相同的 8 位指数（相同的范围，最高 3.4e38），但只有 7 位尾数（精度比 float16 还低）。对于训练神经网络来说，范围比精度更重要，所以 bfloat16 通常胜出。

### 为什么 0.1 + 0.2 != 0.3

0.1 这个数字无法在二进制浮点数中精确表示。在二进制下，它是一个无限循环小数：

```
0.1 的二进制 = 0.0001100110011001100110011... （无限循环）
```

float32 将其截断为 23 位尾数。存储的值大约是 0.100000001490116。类似地，0.2 存储为大约 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对机器学习很重要，因为：

1.  像 `if loss < threshold` 这样的损失比较可能给出错误答案
2.  累积许多小数值（数千步的梯度更新）会偏离真实的和
3.  如果你用 `==` 比较浮点数，校验和与可重复性测试会失败

解决方法：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你减去两个几乎相等的浮点数时，有效数字会抵消，剩下的舍入噪声被提升到高位。

```
a = 1.0000001    （在 float32 中存储为 1.00000011920929）
b = 1.0000000    （在 float32 中存储为 1.00000000000000）

真实差值：  0.0000001
计算值：    0.00000011920929

相对误差： 19.2%
```

仅仅一次减法就产生了 19% 的相对误差。在机器学习中，这种情况发生在：

- 计算具有大均值的数据的方差：当 E[x] 很大时，`E[x^2] - E[x]^2`
- 减去几乎相等的对数概率
- 使用过小的 epsilon 计算有限差分梯度

解决方法：重新排列公式以避免相减两个很大的、几乎相等的数。对于方差，使用 Welford 算法或先对数据中心化。对于对数概率，始终在对数空间中计算。

### 溢出和下溢

溢出发生在结果太大而无法表示时。下溢发生在结果太小（比最小的可表示正数更接近零）时。

```
Float32 边界：
  最大值：         3.4028235e+38
  最小正数（常规）：1.175e-38
  最小正数（非规约）：1.401e-45
  溢出：           任何 > 3.4e38 的数变为 inf
  下溢：           任何 < 1.4e-45 的数变为 0.0
```

`exp()` 函数是机器学习中溢出的主要来源：

```
exp(88.7)  = 3.40e+38   （勉强能放入 float32）
exp(89.0)  = inf         （溢出）
exp(-87.3) = 1.18e-38   （刚好高于下溢阈值）
exp(-104)  = 0.0         （下溢到 0）
```

`log()` 函数则是另一方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （正常）
log(1e-46) = -inf        （输入已下溢为 0，然后 log(0) = -inf）
```

在机器学习中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。如果没有正确的技巧，`log(exp(x))` 的组合就是一个雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何一个 `x_i` 很大，`exp(x_i)` 就会溢出。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都会下溢为 0，然后 `log(0)` 就是 `-inf`。

技巧：在取指数之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 之后，最大的指数项是 `exp(0) = 1`。不可能发生溢出。求和中的至少一项是 1，所以总和至少为 1，`log(1) = 0`。不可能下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    （加 c 再减 c）
= log(sum(exp(x_i - c) * exp(c)))               （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))               （提取公因子 exp(c)）
= c + log(sum(exp(x_i - c)))                    （log(a*b) = log(a) + log(b)）
```

令 `c = max(x)`，溢出就被消除了。

这个技巧在机器学习中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有这个技巧，logits 如 [100, 101, 102] 会导致溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
总和      = 2.99e44

这些数会溢出 float32 吗（最大值 ~3.4e38）？等等，2.69e43 比 3.4e38 大吗？实际上：
exp(88.7) 已经达到 float32 的极限了。
exp(100) 在 float32 中就是 inf。
```

使用技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
总和 = 1.503

softmax = [0.090, 0.245, 0.665]
```

这些概率完全相同。计算是安全的。这不是一个优化技巧，而是保证正确性的必要条件。

### NaN 和 Inf：检测与预防

`nan`（非数字）和 `inf`（无穷大）会像病毒一样在计算中传播。梯度更新中的一个 `nan` 会使得权重变成 `nan`，然后每个后续输出都是 `nan`。训练在一两步之内就死亡了。

`inf` 如何出现：
- 对大正数取 `exp()`
- 除以零：`1.0 / 0.0`
- `float32` 累加溢出

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数取 `sqrt()`
- 对负数取 `log()`
- 任何涉及已有 `nan` 的算术运算

检测：

```python
import math

math.isnan(x)       # 如果 x 是 nan 返回 True
math.isinf(x)       # 如果 x 是 +inf 或 -inf 返回 True
math.isfinite(x)    # 如果 x 既不是 nan 也不是 inf 返回 True
```

预防策略：

1.  钳制 `exp()` 的输入：`exp(clamp(x, -80, 80))`
2.  分母加上 epsilon：`x / (y + 1e-8)`
3.  在 `log()` 内部加上 epsilon：`log(x + 1e-8)`
4.  使用稳定实现（log-sum-exp，稳定 softmax）
5.  梯度裁剪以防止权重爆炸
6.  调试期间在每个前向传播后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自反向传播）可能有 bug。数值梯度检查通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度，远优于前向差分的 `(f(x+h) - f(x)) / h`（只有 O(h)）。

选择 h：太大则近似不准确。太小则灾难性抵消会破坏结果。`h = 1e-5` 到 `1e-7` 是典型值。

检查：计算解析梯度和数值梯度之间的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

在实现新层或损失函数时，总是检查梯度。PyTorch 为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPU 拥有专门的硬件（Tensor Cores），可以以比 float32 快 2-8 倍的速度计算 float16 矩阵乘法。混合精度训练利用了这一点：

```
1. 维护 float32 格式的主权重副本
2. 前向传播使用 float16（快速）
3. 损失计算使用 float32（防止溢出）
4. 反向传播使用 float16（快速）
5. 将梯度缩放到 float32
6. 更新 float32 主权重
```

纯 float16 训练的问题：梯度通常非常小（1e-8 甚至更小）。float16 会将任何低于约 6e-8 的数下溢为 0。你的模型会停止学习，因为所有的梯度更新都变成了零。

解决方案是损失缩放：

```
1. 将损失乘以一个大的缩放因子（例如 1024）
2. 反向传播计算的是 (loss * 1024) 的梯度
3. 所有梯度都变大了 1024 倍（被推到了 float16 下溢阈值之上）
4. 更新权重前，将梯度除以 1024
5. 净效果：相同的更新，但没有下溢
```

动态损失缩放会自动调整缩放因子。从一个较大的值（如 65536）开始。如果梯度溢出为 `inf`，则将其减半。如果连续 N 步没有溢出，则将其加倍。

### bfloat16 vs float16：为什么 bfloat16 在训练中胜出

```
float16:   [1 符号位] [5 指数位]  [10 尾数位]
bfloat16:  [1 符号位] [8 指数位]  [7 尾数位]
```

float16 精度更高（10 尾数位 vs 7），但范围有限（最大值约 65,504）。bfloat16 精度较低，但具有与 float32 相同的范围（最大值约 3.4e38）。

对于训练神经网络：

- 在训练峰值期间，激活值和 logits 经常会超过 65,504。float16 会溢出；bfloat16 可以处理。
- float16 训练需要损失缩放，但 bfloat16 通常不需要，因为它的范围覆盖了梯度量级的整个频谱。
- bfloat16 是 float32 的简单截断：丢弃尾数的低 16 位。转换是简单的，且指数部分无损。

float16 更适合推理，因为推理时数值范围是受限的，精度更重要。bfloat16 更适合训练，因为训练时范围更重要。这就是为什么 TPU 和现代 NVIDIA GPU（A100, H100）拥有原生的 bfloat16 支持。

### 梯度裁剪

梯度爆炸发生在梯度经过多层指数增长时（常见于 RNN、深度网络和 Transformer）。一个巨大的梯度可以在一步之内破坏所有权重。

两种裁剪类型：

**按值裁剪：** 独立地钳制每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留了梯度的方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的事情。这是标准选择。

典型值：Transformer 用 `max_norm=1.0`，强化学习用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

梯度裁剪不是一个 hack。它是一个安全机制。没有它，一个离群的批次就可能产生足够大的梯度，毁掉数周的训练。

### 归一化层作为数值稳定器

批归一化、层归一化和 RMS 归一化通常被描述为帮助训练收敛的正则化器。它们同时也是数值稳定器。

没有归一化，激活值可能在层间指数级增长或收缩：

```
第 1 层：值在 [0, 1] 范围
第 5 层：值在 [0, 100] 范围
第 10 层：值在 [0, 10,000] 范围
第 50 层：值在 [0, inf] 范围
```

归一化在每一层重新中心化和重新缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为 1e-5）防止当所有激活值相同时出现除零错误。可学习参数 `gamma` 和 `beta` 让网络可以恢复它所需要的任何尺度。

这使数值在整个网络中保持在一个安全的范围内，既防止了前向传播中的溢出，也防止了反向传播中的梯度爆炸。

### 常见的机器学习数值 Bug

**Bug：** 训练几个 epoch 后损失变为 NaN。
**原因：** logits 变得太大，softmax 溢出。或者学习率太高，权重发散。
**修复：** 使用稳定的 softmax（最大值减法），降低学习率，添加梯度裁剪。

**Bug：** 损失卡在 log(num_classes)。
**原因：** 模型输出接近均匀概率。通常意味着梯度在消失，或者模型根本没有在学习。
**修复：** 检查数据标签是否正确，验证损失函数，检查是否有死亡的 ReLU 神经元。

**Bug：** 验证准确率比预期低 1-3%。
**原因：** 混合精度没有使用正确的损失缩放。梯度下溢静默地将小的更新置为零。
**修复：** 启用动态损失缩放，或者改用 bfloat16。

**Bug：** 某些层的梯度范数为 0.0。
**原因：** 死亡的 ReLU 神经元（所有输入为负），或者 float16 下溢。
**修复：** 使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：** 模型在一块 GPU 上工作，但在另一块上给出不同的结果。
**原因：** 非确定性的浮点累加顺序。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
**修复：** 接受小的差异（1e-6），或者设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：** 损失计算中的 `exp()` 返回 `inf`。
**原因：** 原始 logits 没有经过最大值减法技巧就直接传入 `exp()`。
**修复：** 使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug：** 从 float32 切换到 float16 后训练发散。
**原因：** float16 无法表示低于 6e-8 的梯度幅度或高于 65,504 的激活值。
**修复：** 使用带损失缩放的混合精度（AMP），或者改用 bfloat16。

## 动手实现

### 步骤1：演示浮点精度限制

```python
print("=== 浮点精度 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差值: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤2：实现朴素版 vs 稳定版 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"朴素版:  {softmax_naive(safe_logits)}")
print(f"稳定版: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"稳定版: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 步骤3：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"朴素版:  {logsumexp_naive(safe):.6f}")
print(f"稳定版: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"稳定版: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 步骤4：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"朴素版:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"稳定版: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "失败"
        print(f"  参数 {i}: 解析={a:.8f} 数值={n:.8f} "
              f"相对误差={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用它

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"原始范数: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"裁剪后范数:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向保留: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现及所有边界情况演示请参见 `code/numerical.py`。

## 交付成果

本课程产出：
- `code/numerical.py`，包含稳定的 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现将在阶段3构建训练循环时以及阶段4实现注意力机制时再次出现。

## 练习

1.  **灾难性抵消。** 在 float32 中使用朴素公式 `E[x^2] - E[x]^2` 计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 在线算法计算。将误差与真实方差 (0.6667) 进行比较。

2.  **精度探索。** 在 Python 中找到最小的正 float32 数值 `x`，使得 `1.0 + x == 1.0`。这就是机器精度 epsilon。验证它与 `numpy.finfo(numpy.float32).eps` 匹配。

3.  **Log-sum-exp 边界情况。** 使用以下情况测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值都非常负 (-1000)。验证在朴素版本失败的地方，它能给出正确结果。

4.  **神经网络层的梯度检查。** 实现一个简单的线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证一个 3x2 权重矩阵的正确性。

5.  **损失缩放实验。** 模拟 float16 训练：生成范围在 [1e-9, 1e-3] 的随机梯度，转换为 float16，并测量其中有多少变为零。然后应用损失缩放（乘以 1024），转换为 float16，缩放回来，再次测量变为零的比例。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| IEEE 754 | "浮点数标准" | 定义二进制浮点格式、舍入规则和特殊值（inf, nan）的国际标准。每个现代 CPU 和 GPU 都实现它。 |
| 机器精度 epsilon | "精度极限" | 在给定的浮点格式下，使得 1.0 + e != 1.0 的最小值 e。对于 float32，约为 1.19e-7。 |
| 灾难性抵消 | "减法导致精度丢失" | 当相减两个几乎相等的浮点数时，有效数字抵消，舍入噪声主导了结果。 |
| 溢出 | "数值太大" | 结果超过最大可表示值，变成 inf。exp(89) 会使 float32 溢出。 |
| 下溢 | "数值太小" | 结果比最小可表示正数更接近零，变成 0.0。exp(-104) 会使 float32 下溢。 |
| Log-sum-exp 技巧 | "先减去最大值" | 通过提取公因子 exp(max(x)) 来计算 log(sum(exp(x)))，以防止溢出和下溢。用于 softmax、交叉熵和对数概率计算。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在取指数之前减去 max(logits)。数值结果相同，但不可能溢出。 |
| 梯度检查 | "验证你的反向传播" | 将来自反向传播的解析梯度与来自有限差分的数值梯度进行比较，以捕获实现错误。 |
| 混合精度 | "Float16 前向，Float32 反向" | 对速度关键的操作使用低精度浮点数，对数值敏感的操作使用高精度浮点数。典型加速为 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播之前将损失乘以一个大常数，使梯度保持在 float16 的可表示范围内，然后在权重更新之前除以同一个常数。 |
| bfloat16 | "Brain 浮点格式" | 谷歌的 16 位格式，具有 8 位指数（与 float32 范围相同）和 7 位尾数（精度低于 float16）。更适合训练。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量，使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "非数字" | 来自未定义运算（0/0, inf-inf, sqrt(-1)）的特殊浮点值。会传播到所有后续算术运算中。 |
| Inf | "无穷大" | 来自溢出或除以零的特殊浮点值。可能组合产生 NaN（inf - inf, inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。速度慢但可靠，用于验证。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) —— 权威参考，内容密集但全面
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) —— NVIDIA 引入用于 float16 训练的损失缩放的论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) —— PyTorch 中混合精度的实用指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) —— 为什么谷歌为 TPU 选择这种格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) —— 减少浮点求和舍入误差的算法