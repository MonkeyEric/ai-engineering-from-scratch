# 结构化输出（Structured Output）—— JSON Schema、Pydantic、Zod 与约束解码

> 即使是前沿模型，“礼貌地请求模型返回 JSON”也有 5% 到 15% 的失败率。结构化输出通过约束解码（constrained decoding）弥合这一差距：模型在字面上被禁止生成任何会违反模式的词元（token）。OpenAI 的严格模式（strict mode）、Anthropic 的模式化工具调用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 以及 Zod 的 `.parse` 都是同一理念的五种表面形式。本节课将构建一个模式验证器以及一个严格模式契约，学员可在所有生产级抽取流程中使用它。

**类型：** 构建  
**语言：** Python（标准库，JSON Schema 2020-12 子集）  
**前置：** Phase 13 · 02（函数调用深入解析）  
**时间：** 约 75 分钟

## 学习目标

- 使用正确的约束（`enum`、`min/max`、`required`、`pattern`）为抽取目标编写 JSON Schema 2020-12。
- 解释为什么严格模式与约束解码提供的保证不同于“生成后再验证”。
- 区分三种失败模式：解析错误（parse error）、模式违规（schema violation）、模型拒绝（refusal）。
- 交付一个具备类型化修复（typed repair）与类型化拒绝处理的抽取流程。

## 问题

一个正在阅读采购订单邮件的代理（agent）需要把自由文本转换成 `{customer, line_items, total_usd}`。有三种方法。

**方法一：在提示词中要求 JSON。** “请以 JSON 格式回复，字段包括 customer、line_items、total_usd。” 在前沿模型上成功率约为 85% 到 95%。会以六种方式失败：缺少花括号、尾部逗号、类型错误、幻觉字段、达到词元限制被截断、以及泄露的散文式文字，例如 “Here is your JSON:”。

**方法二：生成后再验证。** 自由生成，解析，对照模式验证，失败则重试。可靠但昂贵——每次重试都要付费，截断类错误每次出现都要多花费一轮对话。

**方法三：约束解码。** 提供方在解码阶段强制执行模式。无效词元会被从采样分布中掩码掉。输出保证可解析且保证符合模式。失败收敛为一种模式：拒绝（refusal）——模型判定输入不符合模式。

到 2026 年，所有前沿提供方都以某种形式提供了方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`；如果模型拒绝，响应中还会包含 `refusal`。
- **Anthropic。** 对 `tool_use` 输入执行模式强制；`stop_reason: "refusal"` 并不存在，但 `end_turn` 且未进行工具调用就是信号。
- **Gemini。** 在请求级别使用 `responseSchema`；到 2026 年，Gemini 已针对部分类型提供词元级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 会输出一个类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，将提供方输出与 Zod 模式进行比对；可与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同主线：一次性声明模式，端到端强制执行。

## 概念

### JSON Schema 2020-12 —— 通用语言

每个提供方都接受 JSON Schema 2020-12。你最常用的构造包括：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子模式的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许的值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子模式。
- `additionalProperties`：`false` 表示禁止额外字段（默认值随模式而异）。

OpenAI 严格模式增加了三项要求：每个属性都必须列在 `required` 中；到处都要设置 `additionalProperties: false`；不能存在未解析的 `$ref`。违反这些，API 会在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从类似数据类的模型生成 JSON Schema。Pydantic AI 在此基础上封装，于是你可以这样写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

代理框架会将该模式在边缘侧翻译成 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出以类型化的 `Invoice` 实例返回。验证错误会抛出 `ValidationError`，并带有类型化的错误路径。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 的等价物。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，它会转换为 API 的 JSON Schema 载荷。

### 拒绝（Refusal）

严格模式无法强迫模型作答。如果输入无法适配模式（“这封邮件是一首诗，不是发票”），模型会输出一个 `refusal` 字段说明原因。你的代码必须将拒绝作为一等结果处理，而不是当作失败。拒绝本身也是有价值的安全信号：当模型被要求从受保护内容的邮件中提取信用卡号时，它会返回带有安全原因的拒绝。

### 开放生态中的约束解码

开放权重实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从模式构建确定性有限自动机；在每一步，掩码掉会违反该 FSM 的 logits。
2. **结合 JSON 解析器的 logit 掩码**：与模型同步运行流式 JSON 解析器；在每一步计算合法下一词元集合。
3. **带验证器的投机解码**（speculative decoding）：便宜的草稿模型提议词元，验证器强制执行模式。

商业提供方在幕后选择其中一种。2026 年的最先进技术在短结构化输出上比纯生成更快，长输出则大致相当。

### 三种失败模式

1. **解析错误（Parse error）。** 输出不是合法 JSON。在严格模式下不可能发生。在非严格提供方上仍可能发生。
2. **模式违规（Schema violation）。** 输出可解析但违反模式。在严格模式下不可能发生。在严格模式外很常见。
3. **拒绝（Refusal）。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你处于严格模式之外时（Anthropic 工具调用、非严格 OpenAI、旧版 Gemini），恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常一次重试就够了。三次重试能捕获弱模型的抖动。超过三次说明模式本身有问题：模型无法满足某些输入，需要修改提示词或模式。

### 小模型支持

约束解码在小模型上也有效。一个带语法强制的 30 亿参数开放模型，在结构化任务上优于未经 prompting 的 700 亿参数模型。这正是结构化输出对生产至关重要的主要原因：它将可靠性从模型规模中解耦出来。

## 使用

`code/main.py` 提供了一个仅使用标准库的最小 JSON Schema 2020-12 验证器（涵盖类型、必填、`enum`、min/max、`pattern`、`items`、`additionalProperties`）。它包装了一个 `Invoice` 模式，并对一段模拟 LLM 输出运行验证器，演示了解析错误、模式违规和拒绝三条路径。在生产环境中，把模拟输出换成任意提供方的真实响应即可。

值得关注的地方：

- 验证器返回类型化的 `[ValidationError]` 列表，包含路径和消息。这正是你希望暴露给重试提示词的形状。
- 拒绝分支不会重试。它会记录并返回一个类型化的拒绝。Phase 14 · 09 会将拒绝作为安全信号使用。
- `additionalProperties: false` 检查会在对抗性测试输入上触发，展示严格模式为何能堵住幻觉字段的大门。

## 交付

本节课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本抽取目标（发票、支持工单、简历等），该技能会生成一个兼容严格模式的 JSON Schema 2020-12，以及一个与之镜像的 Pydantic 模型，并预置类型化拒绝与重试处理桩代码。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，将其 `total_usd` 设为负数。确认验证器通过 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带 `discriminator` 的 `oneOf`。常见场景：`line_item` 要么是产品要么是服务，由 `kind` 标记。严格模式在此处规则微妙；请参考 OpenAI 的结构化输出指南。

3. 将同一个 Invoice 模式写成 Pydantic `BaseModel`，并比较 `model_json_schema()` 输出与你手写的模式。找出 Pydantic 默认设置、但手写版本省略的那一个字段。

4. 测量拒绝率。构造十条不应被抽取的输入（一首歌词、一个数学证明、一封空白邮件），用真实提供方的严格模式运行。统计拒绝次数与幻觉输出次数。这将构成你拒绝感知重试的基准真值。

5. 通读 OpenAI 的结构化输出指南。找出它在严格模式下明确禁止、但普通 JSON Schema 允许的那一种构造。然后设计一个非必需使用该禁止构造的模式，并将其重构为严格模式兼容版本。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| JSON Schema 2020-12 | “模式规范” | 每个现代提供方都支持的 IETF 草案模式方言 |
| 严格模式（Strict mode） | “保证符合模式” | OpenAI 通过约束解码强制执行模式的开关 |
| 约束解码（Constrained decoding） | “Logit 掩码” | 在解码时掩码非法下一词元的强制执行机制 |
| 拒绝（Refusal） | “模型拒绝” | 输入无法适配模式时的类型化结果 |
| 解析错误（Parse error） | “非法 JSON” | 输出未能解析为 JSON；在严格模式下不可能发生 |
| 模式违规（Schema violation） | “形状错误” | 已解析但违反类型 / 必填 / 枚举 / 范围 |
| `additionalProperties: false` | “不允许额外字段” | 禁止未知字段；OpenAI 严格模式要求设置 |
| Pydantic BaseModel | “类型化输出” | 可生成并验证 JSON Schema 的 Python 类 |
| Zod schema | “TypeScript 输出类型” | 用于验证提供方输出的 TypeScript 运行时模式 |
| 语法强制（Grammar enforcement） | “开放权重约束解码” | 基于 FSM 的 logit 掩码，如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) —— 严格模式、拒绝与模式要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) —— 2024 年 8 月发布文章，解释解码保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) —— 序列化到各提供方的类型化 `output_type` 绑定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) —— 规范原文
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) —— 企业部署笔记与严格模式注意事项
