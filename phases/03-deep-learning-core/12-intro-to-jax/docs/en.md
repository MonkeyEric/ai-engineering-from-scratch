# JAX 简介

> PyTorch 会原地修改张量。TensorFlow 会构建计算图。JAX 则编译纯函数。这最后一点会改变你对深度学习的思考方式。

**类型：** 构建  
**语言：** Python  
**前置条件：** 第三阶段第 01–10 课，基础 NumPy  
**时间：** 约 90 分钟

## 学习目标

- 使用 JAX 的函数式 API（`jax.numpy`、`jax.grad`、`jax.jit`、`jax.vmap`）编写纯函数式的神经网络代码
- 解释 PyTorch 的即时可变模型与 JAX 的函数式编译模型之间的关键设计差异
- 应用 JIT 编译和 vmap 向量化，与纯 Python 相比加速训练循环
- 在 JAX 中训练一个简单网络，并将显式状态管理与 PyTorch 的面向对象方法进行对比

## 问题

你已经知道如何在 PyTorch 中构建神经网络。你定义一个 `nn.Module`，调用 `.backward()`，然后执行优化器的步进。这是可行的。有数百万人在使用它。

但 PyTorch 的 DNA 中固有⼀个约束：它在 Python 中急切地、⼀次⼀个操作地追踪执行。每一个 `tensor + tensor` 都是⼀次独立的内核启动。每⼀个训练步骤都会重新解释同样的 Python 代码。这在大多数情况下没问题，但当你需要在 2,048 块 TPU 上训练⼀个 5400 亿参数的模型时，这种开销就会要你的命。

Google DeepMind 在 JAX 上训练 Gemini。Anthropic 在 JAX 上训练 Claude。这些都不是小规模操作——它们是地球上最大规模的神经网络训练运行。他们选择 JAX，是因为 JAX 将你的训练循环视为⼀个可编译的程序，而不是⼀系列 Python 调用。

JAX 是拥有三项超能力的 NumPy：自动微分、JIT 编译到 XLA，以及自动向量化。你编写⼀个处理单个样本的函数。JAX 会给你⼀个能处理批次、计算梯度、编译成机器码，并在多个设备上运行的函数。所有这些都无需修改原始函数。

## 概念

### JAX 的设计哲学

JAX 是⼀个函数式框架。没有类，没有可变状态，没有 `.backward()` 方法。相比之下：

| PyTorch | JAX |
|---------|-----|
| 带状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 通过 XLA 进行 JIT 编译 |
| 手动 `for x in batch:` 循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组树（pytree） |

这不是风格偏好问题。这是编译器的约束。JIT 编译要求纯函数——相同的输入总是产生相同的输出，没有副作用。正是这个限制让 100 倍的加速成为可能。

### jax.numpy：熟悉的界面

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

函数名相同。广播规则相同。切片语义相同。但数组驻留在 GPU/TPU 上，并且每个操作都可被编译器追踪。

一个关键区别：JAX 数组是不可变的。不能用 `a[0] = 5`。而要用：`a = a.at[0].set(5)`。这在前一周会感觉别扭，之后就会明白——不可变性正是使 `grad`、`jit` 和 `vmap` 等变换可组合的原因。

### jax.grad：函数式自动微分

PyTorch 将梯度附着在张量上（`.grad`）。JAX 将梯度附着在函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接受一个函数并返回一个新函数，该新函数计算梯度。没有 `.backward()` 调用。没有存储在张量上的计算图。梯度只是另一个你可以调用、组合或 JIT 编译的函数。

这可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。雅可比矩阵。海森矩阵。全部通过组合 `grad` 实现。PyTorch 也能做到（`torch.autograd.functional.hessian`），但那是附加的。在 JAX 中，这是基础。

约束：`grad` 只对纯函数有效。内部不能有 print 语句（它们会在追踪期间运行，而不是执行期间）。不能修改外部状态。没有显式的随机数密钥管理就不能使用随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

在第一次调用时，JAX 会追踪该函数——它记录发生了什么操作，但不执行它们。然后它将这个追踪交给 XLA（加速线性代数），这是 Google 为 TPU 和 GPU 打造的编译器。XLA 会融合操作、消除冗余内存复制，并生成优化的机器码。

后续调用完全跳过 Python。编译后的代码在加速器上以 C++ 速度运行。

JIT 何时有益：
- 训练步骤（相同的计算重复数千次）
- 推理（相同的模型，不同的输入）
- 任何被调用多次且输入形状类似的函数

JIT 何时有害：
- 包含依赖于值的 Python 控制流的函数（`if x > 0`，其中 x 是被追踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪会隐藏实际执行过程）

控制流限制是真实存在的。`jax.lax.cond` 替代 `if/else`。`jax.lax.scan` 替代 `for` 循环。这些不是可选的——这是编译的代价。

### vmap：自动向量化

你编写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将其提升为处理批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 表示：不对 `params` 进行批处理（共享），对 `x` 的第 0 维进行批处理。不需要手写 `for` 循环。不需要重塑。不需要处理批次维度。JAX 会自动推断批次维度并向量化整个计算。

这不仅仅是语法糖。`vmap` 会生成融合的向量化代码，比 Python 循环快 10-100 倍。而且它可以与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度。一行代码。这在 PyTorch 中几乎不可能实现，除非使用黑科技。

### pmap：跨设备的数椐并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 会将函数复制到所有可用设备（GPU/TPU）上，并拆分批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 会跨设备同步梯度。

Google 使用 `pmap`（及其后继者 `shard_map`）在数千块 TPU v5e 芯片上训练 Gemini。编程模型是这样的：编写单设备版本，用 `pmap` 包装，完成。

### Pytrees：通用数据结构

JAX 操作“pytree”——列表、元组、字典和数组的嵌套组合。你的模型参数就是⼀个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 将 `f` 应用于每个叶子节点。这就是优化器一次更新所有参数的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法。没有参数注册。树结构就是模型。

### 函数式 vs 面向对象

PyTorch 将状态存储在对象内部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带有显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数被传入。没有存储。没有修改。这使得每个函数都是可测试、可组合和可编译的。这也意味着你要自己管理参数——或者使用像 Flax 或 Equinox 这样的库。

### JAX 生态系统

JAX 提供原语。库提供易用性：

| 库 | 角色 | 风格 |
|---------|------|------|
| **Flax**（Google） | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 Pytree，Pythonic |
| **Optax**（DeepMind） | 优化器 + LR 调度 | 可组合的梯度变换 |
| **Orbax**（Google） | 检查点 | 保存/恢复 pytree |
| **CLU**（Google） | 指标 + 日志 | 训练循环工具 |

Optax 是标准的优化器库。它将梯度变换（Adam、SGD、裁剪）与参数更新分离，使组合变得非常简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时使用 JAX vs PyTorch

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU 支持 | 一等公民（Google 同时构建了二者） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 支持 CUDA） | 最佳（原生 CUDA） |
| 调试 | 困难（追踪 + 编译） | 容易（即时执行，逐行调试） |
| 生态系统 | 研究导向（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（无处不在） |
| 大规模训练 | 卓越（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| 原型开发速度 | 较慢（函数式开销） | 更快（可变、即改即用） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 使用者 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

诚实的答案是：除非你有特定的理由，否则使用 PyTorch。这些理由包括——使用 TPU、需要逐样本梯度、超大规模多设备训练，或者在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每个随机操作都需要一个显式的 PRNG 密钥：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

这起初很烦人。但它保证了跨设备和编译的可重现性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 设置中无法保证的特性。

## 动手实现

### 第 1 步：安装与数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。784 个输入，两个隐藏层分别为 256 和 128 个神经元，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第 2 步：初始化参数

没有类。只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 初始化，手动完成。从⼀个种子拆出三个 PRNG 密钥。每个权重都是嵌套字典中的不可变数组。

### 第 3 步：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。传入参数，得到预测。没有 `self`，没有存储状态。`loss_fn` 从头计算交叉熵——softmax、对数、负均值。

### 第 4 步：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次传递中同时返回损失值和梯度。`@jax.jit` 装饰器会将两个函数编译为 XLA。在第一次调用之后，每个训练步骤的运行都不再触及 Python。

### 第 5 步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 轮。约 97% 的测试准确率。第一轮较慢（JIT 编译）。第 2-10 轮很快。

注意缺少什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合的函数调用。梯度被计算、由 Adam 变换、并应用于参数——所有这些都在 `train_step` 内部完成。

## 如何使用（JAX）

### Flax：Google 标准

Flax 是最常见的 JAX 神经网络库。它重新引入了 `nn.Module`，但带有显式的状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与模型分离。`model.init()` 创建参数。`model.apply(params, x)` 运行前向传播。模型对象本身没有状态。

### Equinox：Pythonic 替代方案

Equinox（由 Patrick Kidger 开发）将模型表示为 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一个 pytree。不需要 `.apply()`。参数就是模型的叶子节点。这更接近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 将梯度变换与更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——都作为变换链组合在一起。每个变换都看到梯度，修改它，并传递给下一个。没有单一的优化器类。

## 交付内容

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次 JIT 调用较慢（编译）。在基准测试前先预热。
- 在 JIT 内部避免对 JAX 数组进行 Python 循环。请使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可在 JIT 内部工作。常规 `print()` 不行。
- 使用 `jax.profiler` 或 TensorBoard 进行性能分析。XLA 编译可能隐藏瓶颈。
- JAX 默认会预分配 75% 的 GPU 内存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可禁用。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课程产出：**
- `outputs/prompt-jax-optimizer.md` —— 一个用于选择正确 JAX 优化器配置的提示
- `outputs/skill-jax-patterns.md` —— 一份涵盖 JAX 函数式模式的技能文档

## 练习

1. 为 MLP 添加 Dropout。在 JAX 中，Dropout 需要 PRNG 密钥——在前向传播中传递一个密钥，并为每个 Dropout 层拆分它。比较有和没有 Dropout 时的测试准确率。

2. 使用 `jax.vmap` 为一批 32 张 MNIST 图像计算逐样本梯度。计算每个样本的梯度范数。哪些样本的梯度最大，为什么？

3. 将手写的前向函数替换为一个通用的 `mlp_forward(params, x)`，它能适用于任意数量的层。使用 `jax.tree.leaves` 自动确定深度。

4. 对训练步骤进行有无 `@jax.jit` 的基准测试。各计时 100 步。在你的硬件上加速比有多大？第一次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。在有无裁剪的情况下训练。绘制训练过程中的梯度范数以观察效果。

## 关键术语

| 术语 | 人们常说的话 | 实际含义 |
|------|----------------|----------------------|
| XLA | “让 JAX 快起来的东西” | 加速线性代数——一个编译器，从计算图中融合操作并生成优化的 GPU/TPU 内核 |
| JIT | “即时编译” | JAX 在第一次调用时追踪函数，编译为 XLA，然后在后续调用中运行编译后的版本 |
| 纯函数 | “无副作用” | 输出仅取决于输入的函数——没有全局状态、没有修改、没有不带显式密钥的随机数 |
| vmap | “自动批处理” | 将处理单个样本的函数变换为处理批次的函数，无需重写 |
| pmap | “自动并行” | 将函数复制到多个设备上并拆分输入批次 |
| Pytree | “嵌套数组字典” | 任何由列表、元组、字典和数组组成的嵌套结构，JAX 可以遍历和变换 |
| 追踪 | “记录计算” | JAX 使用抽象值执行函数以构建计算图，而不计算实际结果 |
| 函数式自动微分 | “对函数求梯度” | 通过变换函数来计算导数，而不是将梯度存储附着在张量上 |
| Optax | “JAX 的优化器库” | 一个可组合的梯度变换库——Adam、SGD、裁剪、调度——可以串联使用 |
| Flax | “JAX 的 nn.Module” | Google 为 JAX 提供的神经网络库，在保持显式状态的同时添加了层抽象 |

## 进一步阅读

- JAX 文档：https://jax.readthedocs.io/ —— 官方文档，包含关于 grad、jit 和 vmap 的优秀教程
- "JAX: composable transformations of Python+NumPy programs"（Bradbury 等，2018）—— 解释设计理念的原始论文
- Flax 文档：https://flax.readthedocs.io/ —— Google 为 JAX 提供的神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations"（2021）—— 比 Flax 更 Pythonic 的替代方案
- DeepMind, "Optax: composable gradient transformation and optimisation" —— 标准优化器库
- "You Don't Know JAX"（Colin Raffel，2020）—— 一份关于 JAX 陷阱和模式的实用指南，出自 T5 作者之一