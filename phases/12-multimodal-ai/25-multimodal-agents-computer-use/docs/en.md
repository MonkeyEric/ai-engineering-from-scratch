# 多模态智能体与计算机使用（综合项目）

> 2026 年的前沿产品是一种多模态智能体（multimodal agent）：它能读取屏幕截图（screenshot）、点击按钮、浏览网页界面、填写表单，并端到端完成工作流。SeeClick 和 CogAgent（2024）证明了 GUI 定位（GUI grounding）这一原语的可行性。Ferret-UI 将其扩展到移动端。ChartAgent 引入了针对图表的视觉工具使用（visual tool use）。VisualWebArena 和 AgentVista（2026）是前沿模型竞相追逐的基准测试（benchmark）——即便是 Gemini 3 Pro 和 Claude Opus 4.7，在 AgentVista 的困难任务上也只能拿到约 30% 的分数。本综合项目（capstone）将串起第 12 阶段的所有主线：感知（高分辨率视觉语言模型 VLM）、推理（带工具使用的大语言模型 LLM）、定位（坐标输出）、长程记忆（long-horizon memory）和评估（evaluation）。

**类型：** 综合项目
**语言：** Python（标准库、动作模式 + 智能体循环骨架）
**前置知识：** 第 12 阶段 · 05（LLaVA）、第 12 阶段 · 09（Qwen-VL JSON）、第 14 阶段（智能体工程）
**时间：** 约 240 分钟

## 学习目标

- 设计一个多模态智能体循环：感知 → 推理 → 执行 → 观察 → 重复。
- 构建一个 GUI 定位输出模式（点击坐标、输入文本、滚动、拖拽），使视觉语言模型（VLM）能以 JSON 形式输出。
- 比较仅屏幕截图智能体、可访问性树（accessibility tree）智能体与混合（hybrid）智能体。
- 在 VisualWebArena 的一个小型切片上搭建多模态智能体基准测试（benchmark）评估。

## 问题

一个订票网站工作流：“给我找一张 4 月 15 日飞往东京的航班，靠过道座位，票价 800 美元以下，预订它。”

一个多模态智能体需要：

1. 截取浏览器屏幕截图。
2. 将屏幕截图、URL 与目标解析为计划。
3. 输出结构化动作：点击（在 x,y 处）、输入“Tokyo”（在元素 E 处）、向下滚动、选择（单选按钮）。
4. 将动作应用到浏览器。
5. 观察新状态（下一张屏幕截图）。
6. 重复直到任务完成。

每一步都是一次多模态视觉语言模型（VLM）调用。VLM 的输出必须是可解析的 JSON。错误会在多步之间累积，因此恢复机制至关重要。

## 概念

### GUI 定位——基础原语

GUI 定位（GUI grounding）是指：给定一张屏幕截图和一条自然语言指令，输出要点击（或执行其他动作）的 (x, y) 坐标。

SeeClick（arXiv:2401.10935）是首个达到规模的公开成果：在合成与真实 GUI 数据上微调视觉语言模型（VLM），以纯文本 token 形式输出坐标。效果不错。

CogAgent（arXiv:2312.08914）为密集界面增加了 1120×1120 高分辨率编码。在网页导航任务上得分约 84%。

Ferret-UI（arXiv:2404.05719）专注于移动界面，并与 iOS 可访问性（accessibility）数据集成。

输出格式通常为 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 有助于恢复：如果坐标在不同屏幕截图之间发生漂移，语义提示可以让系统重新定位。

### 动作模式

一个典型的动作模式（action schema）包含 6–10 种动作类型：

- `click`：点击 (x, y)
- `type`：输入文本 (text, x?, y?)
- `scroll`：滚动 (direction, amount)
- `drag`：拖拽 (x0, y0, x1, y1)
- `select`：选择 (option_index)
- `hover`：悬停 (x, y)
- `navigate`：导航 (url)
- `wait`：等待 (ms)
- `done`：完成 (success, explanation)

智能体每一步输出一个动作。浏览器封装层执行该动作并返回新状态。

### 仅屏幕截图 vs 可访问性树

两种输入模式：

- 仅屏幕截图（Screenshot-only）：完整图像，无结构信息。最通用；适用于任何应用。
- 可访问性树（Accessibility tree）：结构化 DOM / iOS 可访问性信息。对定位（grounding）更可靠；在可获取树信息的场景适用。
- 混合（Hybrid）：两者结合，以树作为原子动作的可靠定位依据，以屏幕截图提供语义上下文。

生产级智能体在可能时采用混合模式。浏览器自动化（Selenium + accessibility）始终拥有树信息；桌面应用有时也有。

### 长程记忆

一个 20 步的工作流会产生 20 张屏幕截图。视觉语言模型（VLM）的上下文很快就会填满。三种压缩策略：

- 摘要链（Summary-chain）：每 5 步总结一次已发生的事，丢弃旧屏幕截图。
- 跳帧（Skip-frame）：保留第一张、最后一张以及每第 3 张屏幕截图。
- 工具记录日志（Tool-recorded log）：执行动作，保留文本形式的操作日志；不再回看旧屏幕截图。

Claude 的计算机使用（computer-use）API 采用日志模式。更简单，也更可靠。

### 视觉工具使用

ChartAgent（arXiv:2510.04514）引入了针对图表理解的视觉工具使用（visual tool use）：裁剪、缩放、OCR、调用外部检测。智能体可以输出“裁剪到区域 (100, 200, 300, 400) 然后调用 OCR”这样的工具调用。工具返回文本，视觉语言模型（VLM）继续推理。

这一模式具有通用性：标记集合提示（set-of-mark prompting）、区域标注（region annotation）和外部检测工具都遵循同一种“输出工具调用，接收结构化响应”的模式。

### 2026 年基准测试

- ScreenSpot-Pro。在约 1k 张网页屏幕截图上进行 GUI 定位（GUI grounding）。公开最佳水平（SOTA）Qwen2.5-VL-72B 约 85%。前沿模型约 90%。
- VisualWebArena。端到端网页任务（购物、论坛、分类信息）。公开最佳水平（SOTA）约 20%。Gemini 3 Pro 约 27%。
- AgentVista（arXiv:2602.23166）。2026 年最困难的基准测试（benchmark）。覆盖 12 个领域的真实工作流。前沿模型得分 27–40%；开源模型 10–20%。
- WebArena / WebShop。较老的基准测试；已被前沿模型接近饱和。

### 为什么仍然困难

智能体性能瓶颈：

1. 精细尺度视觉定位（visual grounding）。在移动分辨率下，“点击那个小 X”经常失败。
2. 长程规划（long-horizon planning）。执行 10 个动作后，智能体会偏离目标。
3. 错误恢复（error recovery）。当一次点击失败（点错按钮），检测并恢复的场景很少出现在训练数据中。
4. 跨页面上下文。在标签页之间跳转或处理长表单时会丢失状态。

研究方向：记忆架构、显式重新规划（replanning）、多模态验证（multimodal verification）（通过屏幕截图匹配判断动作是否成功）。

### 综合项目动手实现

本综合项目（capstone）任务：构建一个计算机使用（computer-use）智能体，它需要：

1. 读取订票网站模拟页面的 HTML + 屏幕截图。
2. 规划多步序列：搜索 → 选择 → 填写表单 → 提交。
3. 输出符合动作模式（action schema）的 JSON 动作。
4. 在一个固定的 10 任务切片上评估。

本课程提供了可轻松扩展到真实浏览器的脚手架代码。

## 使用它

`code/main.py` 是综合项目（capstone）的脚手架：

- 动作模式（action schema）JSON 定义（10 个动作）。
- 以字典表示的模拟浏览器状态。
- 智能体循环骨架：接收状态、输出动作、应用、循环。
- 10 任务迷你基准测试（合成页面），用于衡量端到端成功率。
- 动作失败时的错误恢复钩子。

## 交付它

本课程将产出 `outputs/skill-multimodal-agent-designer.md`。给定一个计算机使用（computer-use）产品（领域、动作集合、评估目标），设计完整的智能体循环、记忆策略、定位模式与预期基准测试（benchmark）分数。

## 练习

1. 用一个 `screenshot_region` 工具（裁剪 + 缩放）扩展动作模式（action schema）。哪些任务会受益？

2. 阅读 AgentVista（arXiv:2602.23166）。描述最困难的任务类别，以及为什么前沿模型仍会失败。

3. 长程记忆（long-horizon memory）压缩：设计一个摘要链（summary-chain），最多保留 4 张实时屏幕截图，其余任意数量仅记录日志。

4. 构建一个错误恢复（error recovery）钩子：当动作失败（未找到按钮）时，智能体下一步做什么？

5. 在 10 个网页任务上，比较仅屏幕截图（screenshot-only）的 Claude 4.7 与混合屏幕截图 + 可访问性树（accessibility tree）的 Qwen2.5-VL。在哪些任务上谁更胜一筹？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| GUI 定位 | “点击坐标” | 模型输出指令目标在屏幕截图上的 (x,y) 坐标 |
| 动作模式 | “工具定义” | 有效动作（点击、输入、滚动、拖拽）的 JSON 描述 |
| 可访问性树 | “结构化 DOM” | 来自浏览器 / iOS API 的机器可读界面层级 |
| 混合智能体 | “屏幕截图 + 树” | 同时使用图像与结构化信息；比单独使用任一种更可靠 |
| 视觉工具使用 | “缩放/裁剪/检测” | 智能体在规划过程中调用外部视觉工具（OCR、检测） |
| 摘要链 | “记忆压缩” | 定期文本摘要替代冗长的屏幕截图历史 |
| VisualWebArena | “端到端网页基准” | 2024 年端到端网页任务基准测试 |
| AgentVista | “2026 年困难基准” | 覆盖 12 个领域的真实工作流；即便 Gemini 3 Pro 也仅约 30% |

## 延伸阅读

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
