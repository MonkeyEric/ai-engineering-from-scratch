# 张量运算

> 张量是数据与深度学习之间的共同语言。每一张图像、每一个句子、每一个梯度都流经它们。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段1，第01课（线性代数直觉）、第02课（向量、矩阵与运算）
**时间：** 约90分钟

## 学习目标

- 从零实现一个包含形状、步长、重塑、转置和逐元素运算的张量类
- 应用广播规则对不同形状的张量进行运算，且不复制数据
- 为点积、矩阵乘法、外积和批量运算编写einsum表达式
- 追踪多头注意力中每一步的确切张量形状

## 问题描述

你在构建一个Transformer。前向传播看起来很清晰。运行后却得到：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`。你盯着这些形状。尝试了转置。现在它显示 `Expected 4D input (got 3D input)`。你加了一个unsqueeze。然后又出了别的问题。

形状错误是深度学习代码中最常见的bug。它们在概念上并不难——每个操作都有形状契约——但错误会迅速蔓延。一个Transformer有几十个重塑、转置和广播操作串联在一起。弄错一个轴，错误就会级联放大。更糟糕的是，有些形状错误根本不会抛出异常。它们会通过错误的维度广播或对错误的轴求和，静默地产生垃圾结果。

矩阵处理两个集合之间的成对关系。真实数据并不局限于两个维度。一批32张224x224的RGB图像是一个4D张量：`(32, 3, 224, 224)`。具有12个头的自注意力也是4D的：`(batch, heads, seq_len, head_dim)`。你需要一种能推广到任意维度数的数据结构，并且其操作能在所有维度上干净地组合。这种结构就是张量。掌握它的运算之后，形状错误就会变得极易调试。

## 核心概念

### 什么是张量

张量是一个具有统一数据类型的多维数组。维度的数量称为**秩**（或**阶**）。每个维度称为一个**轴**。**形状**是一个元组，列出了每个轴上的大小。

```mermaid
graph LR
    S["标量<br/>秩0<br/>形状: ()"] --> V["向量<br/>秩1<br/>形状: (3,)"]
    V --> M["矩阵<br/>秩2<br/>形状: (2,3)"]
    M --> T3["3D张量<br/>秩3<br/>形状: (2,2,2)"]
    T3 --> T4["4D张量<br/>秩4<br/>形状: (B,C,H,W)"]
```

总元素数 = 所有大小的乘积。形状 `(2, 3, 4)` 包含 `2 * 3 * 4 = 24` 个元素。

### 深度学习中的张量形状

按照惯例，不同的数据类型映射到特定的张量形状。

```mermaid
graph TD
    subgraph 视觉
        V1["(B, C, H, W)<br/>32, 3, 224, 224"]
    end
    subgraph 自然语言处理
        N1["(B, T, D)<br/>16, 128, 768"]
    end
    subgraph 注意力
        A1["(B, H, T, D)<br/>16, 12, 128, 64"]
    end
    subgraph 权重
        W1["线性层: (out, in)<br/>Conv2D: (out_c, in_c, kH, kW)<br/>嵌入层: (vocab, dim)"]
    end
```

PyTorch 使用 NCHW（通道优先）。TensorFlow 默认使用 NHWC（通道最后）。布局不匹配会导致静默的性能下降或错误。

### 内存布局如何工作

内存中的二维数组是一个一维的字节序列。**步长**告诉你：要沿着某个轴移动一步，需要跳过多少个元素。

```mermaid
graph LR
    subgraph "行优先（C顺序）"
        R["a b c d e f<br/>步长: (3, 1)"]
    end
    subgraph "列优先（F顺序）"
        C["a d b e c f<br/>步长: (1, 2)"]
    end
```

转置不会移动数据。它交换步长，使得张量变成**非连续的**——此时一行的元素在内存中不再相邻。

### 广播规则

广播让你可以在不复制数据的情况下对不同形状的张量进行运算。从右向左对齐形状。当两个维度相等或其中一个为1时，它们是兼容的。维度较少的张量会在左侧用1补齐。

```
张量 A:     (8, 1, 6, 1)
张量 B:        (7, 1, 5)
补齐后 B:   (1, 7, 1, 5)
结果:       (8, 7, 6, 5)
```

### Einsum：通用的张量运算

爱因斯坦求和约定用字母标记每个轴。出现在输入但未出现在输出中的轴会被求和。同时出现在两个输入中的轴会被保留。

```mermaid
graph LR
    subgraph "矩阵乘法: ik,kj -> ij"
        A["A(I,K)"] --> |"对 k 求和"| C["C(I,J)"]
        B["B(K,J)"] --> |"对 k 求和"| C
    end
```

关键模式：`i,i->`（点积）、`i,j->ij`（外积）、`ii->`（迹）、`ij->ji`（转置）、`bij,bjk->bik`（批量矩阵乘法）、`bhtd,bhsd->bhts`（注意力分数）。

## 动手实现

代码位于 `code/tensors.py`。每个步骤都引用了其中的实现。

### 步骤1：张量存储与步长

张量存储一个扁平的数值列表以及形状元数据。步长告诉索引逻辑如何将多维索引映射到扁平位置。

```python
class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
```

对于形状 `(3, 4)`，步长是 `(4, 1)` —— 前进一行需要跳过4个元素，前进一列需要跳过1个元素。

### 步骤2：重塑、挤压、解挤压

重塑在不改变元素顺序的前提下改变形状。总元素数必须保持不变。使用 `-1` 可以让该维度的大小自动推断。

```python
t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))
```

`squeeze` 删除大小为1的轴。`unsqueeze` 插入一个大小为1的轴。解挤压对于广播至关重要——一个形状为 `(D,)` 的偏置向量要加到形状为 `(B, T, D)` 的批量数据上，需要先解挤压成 `(1, 1, D)`。

```python
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)
```

### 步骤3：转置与重排

`transpose` 交换两个轴。`permute` 重新排列所有轴。这就是在 NCHW 和 NHWC 之间转换的方法。

```python
mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.permute((0, 2, 3, 1))
```

转置或重排之后，张量在内存中是非连续的。在 PyTorch 中，`view` 对非连续张量会失败——此时应使用 `reshape`，或者先调用 `.contiguous()`。

### 步骤4：逐元素运算与规约

逐元素运算（加法、乘法、减法）独立应用于每个元素，并保持形状不变。规约操作（求和、平均、最大值）会折叠一个或多个轴。

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])
c = a + b
d = a * 2
s = a.sum(axis=0)
```

CNN 中的全局平均池化：`(B, C, H, W).mean(axis=[2, 3])` 产生 `(B, C)`。NLP 中的序列平均池化：`(B, T, D).mean(axis=1)` 产生 `(B, D)`。

### 步骤5：使用 NumPy 演示广播

`tensors.py` 中的 `demo_broadcasting_numpy()` 函数展示了核心模式。

```python
activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
result = activations + bias

images = np.random.randn(2, 3, 4, 4)
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
result = images * scale

a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
outer = a * b
```

通过广播计算成对距离：将 `(M, 2)` 重塑为 `(M, 1, 2)`，将 `(N, 2)` 重塑为 `(1, N, 2)`，相减、平方、沿最后一维求和、再开平方根。结果形状为 `(M, N)`。

### 步骤6：Einsum 运算

`demo_einsum()` 和 `demo_einsum_gallery()` 函数演示了每一种常见模式。

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dot = np.einsum("i,i->", a, b)

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
matmul = np.einsum("ik,kj->ij", A, B)

batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)
batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)
```

一次收缩运算的计算代价是所有指标大小（包括保留的和求和的）的乘积。对于 `bij,bjk->bik`，若 B=32, I=128, J=64, K=128，则计算量为 `32 * 128 * 64 * 128 = 33,554,432` 次乘加运算。

### 步骤7：通过 einsum 实现注意力机制

`demo_attention_einsum()` 函数端到端地实现了多头注意力。

```python
B, H, T, D = 2, 4, 8, 16
E = H * D

X = np.random.randn(B, T, E)
W_q = np.random.randn(E, E) * 0.02

Q = np.einsum("bte,ek->btk", X, W_q)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)

scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
weights = softmax(scores, axis=-1)
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
output = np.einsum("bte,ek->btk", concat, W_o)
```

每一步都是一个张量运算：投影（通过 einsum 实现矩阵乘法）、头部分割（重塑 + 转置）、注意力分数（通过 einsum 实现批量矩阵乘法）、加权求和（通过 einsum 实现批量矩阵乘法）、头部合并（转置 + 重塑）、输出投影（通过 einsum 实现矩阵乘法）。

## 使用它

### 自实现 vs NumPy

| 操作 | 自实现（Tensor 类） | NumPy |
|---|---|---|
| 创建 | `Tensor([[1,2],[3,4]])` | `np.array([[1,2],[3,4]])` |
| 重塑 | `t.reshape((3,4))` | `a.reshape(3,4)` |
| 转置 | `t.transpose(0,1)` | `a.T` 或 `a.transpose(0,1)` |
| 挤压 | `t.squeeze(0)` | `np.squeeze(a, 0)` |
| 求和 | `t.sum(axis=0)` | `a.sum(axis=0)` |
| Einsum | 不适用 | `np.einsum("ij,jk->ik", a, b)` |

### 自实现 vs PyTorch

```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
t.shape
t.stride()
t.is_contiguous()

t.reshape(3, 2)
t.unsqueeze(0)
t.transpose(0, 1)
t.transpose(0, 1).contiguous()

torch.einsum("ik,kj->ij", A, B)
```

PyTorch 增加了自动求导、GPU 支持和优化的 BLAS 内核。形状语义是相同的。如果你理解了自实现版本，那么 PyTorch 的形状错误就变得可读了。

### 每个神经网络层都是一种张量运算

| 运算 | 张量形式 | Einsum |
|---|---|---|
| 线性层 | `Y = X @ W.T + b` | `"bd,od->bo"` + 偏置 |
| 注意力 QKV | `Q = X @ W_q` | `"btd,dh->bth"` |
| 注意力分数 | `Q @ K.T / sqrt(d)` | `"bhtd,bhsd->bhts"` |
| 注意力输出 | `softmax(scores) @ V` | `"bhts,bhsd->bhtd"` |
| 批归一化 | `(X - mu) / sigma * gamma` | 逐元素 + 广播 |
| Softmax | `exp(x) / sum(exp(x))` | 逐元素 + 规约 |

## 交付成果

本课程产出两个可复用的提示词模板：

1. **`outputs/prompt-tensor-shapes.md`** —— 一个用于调试张量形状不匹配的系统性提示词模板。包含每种常见操作（矩阵乘法、广播、拼接、Linear、Conv2d、BatchNorm、softmax）的决策表以及一个修复查找表。

2. **`outputs/prompt-tensor-debugger.md`** —— 一个分步调试提示词模板。当形状错误阻塞你时，将其粘贴到任何AI助手中，输入错误信息和张量形状，就能得到确切的修复方案。

## 练习

1. **简单 —— 重塑往返。** 取一个形状为 `(2, 3, 4)` 的张量。将其重塑为 `(6, 4)`，再重塑为 `(24,)`，最后回到 `(2, 3, 4)`。通过打印扁平数据验证每一步的元素顺序都被保持。

2. **中等 —— 实现广播。** 扩展 `Tensor` 类，添加一个 `broadcast_to(shape)` 方法，将大小为1的维度扩展到目标形状。然后修改 `_elementwise_op`，使其在运算前自动进行广播。用形状 `(3, 1)` 和 `(1, 4)` 测试，期望产生 `(3, 4)` 的结果。

3. **困难 —— 从零构建 einsum。** 实现一个基本的 `einsum(subscripts, *tensors)` 函数，至少能处理：点积（`i,i->`）、矩阵乘法（`ij,jk->ik`）、外积（`i,j->ij`）和转置（`ij->ji`）。解析下标字符串，识别被收缩的指标，并循环遍历所有指标组合。将你的结果与 `np.einsum` 进行比较。

4. **困难 —— 注意力形状追踪器。** 编写一个函数，接收 `batch_size`、`seq_len`、`embed_dim` 和 `num_heads` 作为输入，并打印多头注意力中每一步的确切形状：输入、Q/K/V 投影、头部分割、注意力分数、softmax 权重、加权求和、头部合并、输出投影。与 `demo_attention_einsum()` 的输出进行验证。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 张量 | "矩阵，但维度更多" | 一个具有统一类型、确定的形状、步长和运算的多维数组 |
| 秩 | "维度的数量" | 轴的数量。一个矩阵的秩是2，而不是它的矩阵秩 |
| 形状 | "张量的大小" | 一个元组，列出了每个轴上的大小。`(2, 3)` 表示2行3列 |
| 步长 | "内存如何布局" | 沿着每个轴移动一个位置所需要跳过的元素个数 |
| 广播 | "形状不同时它也能工作" | 一套严格的规则：从右对齐，维度要么相等，要么其中一个为1 |
| 连续性 | "这个张量是正常的" | 元素按顺序存储在内存中，与逻辑布局相比没有间隙或重排 |
| Einsum | "一种写矩阵乘法的花哨方式" | 一种通用记法，可以用一行表达式表示任何张量收缩、外积、迹或转置 |
| View | "和reshape一样" | 共享相同内存缓冲区但具有不同形状/步长元数据的张量。对非连续数据会失败 |
| 收缩 | "对某个指标求和" | 一种通用操作：张量之间共享的指标被相乘并求和，产生一个较低秩的结果 |
| NCHW / NHWC | "PyTorch vs TensorFlow 格式" | 图像张量的内存布局约定。NCHW 将通道放在空间维度之前，NHWC 将通道放在之后 |

## 延伸阅读

- [NumPy 广播](https://numpy.org/doc/stable/user/basics.broadcasting.html) —— 带有可视化示例的规范规则
- [PyTorch 张量视图](https://pytorch.org/docs/stable/tensor_view.html) —— 视图何时有效，何时会复制
- [einops](https://github.com/arogozhnikov/einops) —— 一个让张量重塑变得可读且安全的库
- [图解 Transformer](https://jalammar.github.io/illustrated-transformer/) —— 可视化流经注意力的张量形状
- [NumPy 中的爱因斯坦求和](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) —— 完整的 einsum 文档及示例