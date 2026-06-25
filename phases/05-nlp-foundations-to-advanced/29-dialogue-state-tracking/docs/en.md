# 对话状态跟踪

> "我想要一家北区便宜的餐厅……其实改成中档……再加意大利菜。" 三轮对话，三次状态更新。DST 负责让槽值字典保持同步，这样预订才不会出错。

**类型：** 构建  
**语言：** Python  
**先修知识：** Phase 5 · 17（Chatbots）, Phase 5 · 20（Structured Outputs）  
**时间：** 约 75 分钟

## 问题所在

在任务导向对话系统中，用户目标被编码为一组槽值对：`{cuisine: italian, area: north, price: moderate}`。每一轮用户发言都可能添加、修改或删除某个槽。系统必须读取整段对话，并正确输出当前状态。

只要有一个槽出错，系统就会订错餐厅、排错航班或刷错卡。DST 是用户话语与后端执行之间的关键枢纽。

尽管 LLM 大行其道，DST 在 2026 年仍然重要，原因在于：

- 合规敏感领域（银行、医疗、航空订票）需要确定性的槽值，而不是自由文本生成。
- 使用工具的智能体在调用 API 之前仍然需要先解析槽位。
- 多轮修正视乎简单实则困难："其实不，改成周四。"

现代流程：经典 DST 概念 + LLM 抽取器 + 结构化输出护栏。

## 核心概念

![DST：对话历史 → 槽值状态](../assets/dst.svg)

**任务结构。** 模式（schema）定义领域（restaurant、hotel、taxi）及其槽（cuisine、area、price、people）。每个槽可以是空的、用封闭集合中的值填充（`price: {cheap, moderate, expensive}`），或是自由文本值（`name: "The Copper Kettle"`）。

**两种 DST 建模方式。**

- **分类。** 对每个（slot, candidate_value）组合预测是/否。适用于封闭词表槽。2020 年前的标准做法。
- **生成。** 给定对话，以自由文本形式生成槽值。适用于开放词表槽。现代默认方法。

**评估指标。** 联合目标准确率（JGA）——所有槽都正确的轮次占比。全有或全无。2026 年 MultiWOZ 2.4 排行榜最高约 83%。

**架构。**

1. **基于规则（槽正则 + 关键词）。** 窄域的强基线。可调试。
2. **TripPy / BERT-DST。** 基于 BERT 编码的复制式生成。LLM 出现前的标准方法。
3. **LDST（LLaMA + LoRA）。** 经过指令微调的 LLM，配合领域-槽提示。在 MultiWOZ 2.4 上达到 ChatGPT 级别水平。
4. **无本体（2024–26）。** 跳过模式，直接生成槽名和槽值。可处理开放领域。
5. **提示 + 结构化输出（2024–26）。** LLM + Pydantic 模式 + 约束解码。5 行代码即可投入生产。

### 经典失败模式

- **跨轮共指。** "我们就选第一个选项吧。" 需要解析指的是哪个选项。
- **覆盖还是追加。** 用户说 "add Italian"。你要替换 cuisine 还是追加？
- **隐式确认。** "OK cool"——这是否接受了推荐的预订？
- **修正。** "Actually make it 7 pm." 必须更新时间，而不能清空其他槽。
- **指向前一轮系统话语。** "Yes, that one." 哪个 "that"？

## 动手实现

### 第 1 步：基于规则的槽抽取器

参见 `code/main.py`。正则表达式 + 同义词词典可以覆盖窄域中 70% 的标准说法：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在标准词表之外很脆弱。适用于确定性的槽确认。

### 第 2 步：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三条不变原则：

- 不要重置用户没有触碰过的槽。
- 显式否定（"never mind the cuisine"）必须清空。
- 用户修正（"actually..."）必须覆盖，而不是追加。

### 第 3 步：基于 LLM 的结构化输出 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证返回有效的状态对象。无需正则、不会出现模式不匹配、也不会产生幻觉槽。

### 第 4 步：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统有多少轮能把**所有**槽都弄对？对于 MultiWOZ 2.4，2026 年顶尖系统为 80–83%。在你的窄词表上，领域内系统应当超过这一数字，否则不如直接用 LLM 基线。

### 第 5 步：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到修正时，覆盖最近更新的槽，而不是追加。没有 LLM 帮助时很难做对。现代做法：始终让 LLM 根据完整对话历史重新生成整个状态，而不是增量更新——这自然就能处理修正。

## 常见陷阱

- **全历史重新生成的成本。** 每轮都让 LLM 重新生成状态的累计 token 复杂度为 O(n²)。限制历史长度或对较早轮次做摘要。
- **模式漂移。** 事后新增槽会破坏旧的训练数据。给模式加版本号。
- **大小写敏感。** "Italian"、"italian"、"ITALIAN"——到处都要归一化。
- **隐式继承。** 如果用户之前说过 "for 4 people"，新的不同时间请求不应清空人数。始终传入完整历史。
- **自由文本 vs 封闭集合。** 姓名、时间、地址需要自由文本槽；菜系、区域是封闭的。在模式中混合使用两者。

## 应用场景

2026 年的技术栈：

| 场景 | 方案 |
|-----------|----------|
| 窄域（一两个意图） | 基于规则 + 正则 |
| 广域，有标注数据 | LDST（在 MultiWOZ 风格数据上用 LLaMA + LoRA） |
| 广域，无标注，可生产 | LLM + Instructor + Pydantic 模式 |
| 语音 / 口语 | ASR + 归一化 + LLM-DST |
| 多领域预订流程 | 用按领域划分的 Pydantic 模型引导 LLM |
| 合规敏感 | 以规则为主，LLM 兜底并带确认流程 |

## 交付产物

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: 设计一个对话状态跟踪器——模式、抽取器、更新策略、评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

给定一个用例（领域、语言、词表开放度、合规需求），输出：

1. 模式。领域列表、每个领域的槽、每个槽是开放词表还是封闭词表。
2. 抽取器。基于规则 / seq2seq / LLM-with-Pydantic。给出理由。
3. 更新策略。重新生成整个状态 / 增量更新；修正处理；否定处理。
4. 评估。在留出对话集上的联合目标准确率，槽级精确率/召回率，最难槽的混淆情况。
5. 确认流程。何时明确请用户确认（破坏性操作、低置信度抽取）。

对于合规敏感槽，如果没有基于规则的二次校验，则拒绝纯 LLM 的 DST。对于无法根据用户修正回滚槽位的 DST，一概拒绝。给没有版本标签的模式打上标记。
```

## 练习

1. **简单。** 在 `code/main.py` 中为 3 个槽（cuisine、area、price）构建基于规则的状态跟踪器。在 10 段手工编写的对话上测试。测量 JGA。
2. **中等。** 在同一数据集上用 Instructor + Pydantic + 一个小型 LLM。对比 JGA。检查最难的轮次。
3. **困难。** 同时实现两种方法并做路由：以基于规则为主，当规则方法以置信度输出少于 2 个槽时使用 LLM 兜底。测量组合后的 JGA 和每轮推理成本。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态跟踪 | 在对话轮次之间维护槽值字典。 |
| Slot | 用户意图单元 | 后端需要的命名参数（cuisine、date）。 |
| Domain | 任务领域 | Restaurant、hotel、taxi——槽的集合。 |
| JGA | 联合目标准确率 | 所有槽都正确的轮次占比。全有或全无。 |
| MultiWOZ | 基准测试 | 多领域 Wizard-of-Oz 数据集；DST 标准评估集。 |
| Ontology-free DST | 无模式 | 直接生成槽名和槽值，无需固定列表。 |
| Correction | "Actually..." | 覆盖之前已填充槽位的一轮发言。 |

## 延伸阅读

- [Budzianowski 等人（2018）。MultiWOZ——大规模多领域 Wizard-of-Oz 数据集](https://arxiv.org/abs/1810.00278) —— 经典基准。
- [Feng 等人（2023）。迈向由 LLM 驱动的对话状态跟踪（LDST）](https://arxiv.org/abs/2310.14970) —— 用于 DST 的 LLaMA + LoRA 指令微调。
- [Heck 等人（2020）。TripPy——面向值无关神经对话状态跟踪的三重复制策略](https://arxiv.org/abs/2005.02877) —— 基于复制的 DST 主力方法。
- [King、Flanigan（2024）。基于 LLM 的无监督端到端任务导向对话](https://arxiv.org/abs/2404.10753) —— 基于 EM 的无监督 TOD。
- [MultiWOZ 排行榜](https://github.com/budzianowski/multiwoz) —— 经典 DST 结果。
