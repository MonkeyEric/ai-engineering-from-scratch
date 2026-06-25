# 根目录（Roots）与用户征询（Elicitation）—— 作用范围与运行中的用户输入

> 硬编码路径会在用户打开不同项目时立即失效。预填充的工具参数会在用户描述不足时失效。根目录（roots）将服务器限定在一组由用户控制的 URI 内；用户征询（elicitation）则在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。两种客户端原语，分别修复两种常见的 MCP 故障模式。SEP-1036（URL 模式征询，2025-11-25）在 2026 年上半年之前均为实验性 —— 在依赖它之前请检查 SDK 版本。

**类型：** 构建  
**语言：** Python（标准库，根目录与用户征询演示）  
**前置条件：** Phase 13 · 07（MCP 服务器）  
**时间：** 约 45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在已声明的根目录集合内。
- 使用 `elicitation/create` 在工具调用中途向用户请求确认或结构化输入。
- 在表单模式与 URL 模式之间做出选择（后者为实验性；存在漂移风险）。

## 问题背景

一个笔记 MCP 服务器在生产环境中会遇到两个具体故障。

**硬编码路径假设。** 服务器基于 `~/notes` 编写。在另一台机器上将笔记存放在 `~/Documents/Notes` 的用户会收到静默失败的工具调用（找不到文件），更糟的是会写入错误的位置。

**用户知道但缺失的参数。** 用户要求“删除旧的 TPS 报告笔记”。模型调用 `notes_delete(title: "TPS report")`，但存在 2023、2024、2025 年三个匹配的笔记。工具无法猜测。以“存在歧义”失败令人恼火；对三个笔记全部执行则是灾难。

根目录（roots）修复第一种：客户端在 `initialize` 时声明服务器可以访问的 URI 集合。用户征询（elicitation）修复第二种：服务器暂停工具调用并发送 `elicitation/create`，让用户选择其中一个。

## 核心概念

### 根目录

客户端在 `initialize` 时声明根目录列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器随后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将根目录视为边界：任何在根目录集合之外的文件读取或写入都应被拒绝。客户端不会强制执行这一点（服务器仍是用户信任的代码），但符合规范的服务器会遵守它。

当用户添加或删除根目录时，客户端会发送 `notifications/roots/list_changed`。服务器会重新调用 `roots/list` 并更新其边界。

### 为什么根目录是客户端原语

根目录由客户端声明，因为它们代表用户的授权模型。用户告诉 Claude Desktop“让此笔记服务器访问这两个目录”。服务器无法扩大该范围。

### 用户征询：默认表单模式

`elicitation/create` 接收一个表单模式（schema）和一段自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户回答，然后返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的动作：`accept`（用户已填写）、`decline`（用户关闭表单）、`cancel`（用户中止整个工具调用）。

表单模式是扁平的 —— v1 不支持嵌套对象。SDK 通常会拒绝超过单层的复杂结构。

### 用户征询：URL 模式（SEP-1036，实验性）

新增于 2025-11-25。服务器可以发送一个 URL 而不是模式：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开该 URL，等待完成，并在用户返回后继续。适用于 OAuth 流程、支付授权以及表单无法满足的文档签名等场景。

漂移风险说明：SEP-1036 的响应格式仍在确定中；部分 SDK 返回回调 URL，另一些返回完成令牌。在正式环境使用 URL 模式前，请阅读所用 SDK 的发布说明。

### 何时适合使用用户征询

- 破坏性操作前请求用户确认（破坏性提示 + 用户征询）。
- 消除歧义（从 N 个匹配项中选择一个）。
- 首次运行设置（API 密钥、目录、偏好设置）。
- OAuth 类流程（URL 模式）。

### 何时不应使用用户征询

- 填充模型本可以用普通文本询问的必填参数。使用常规重新提示，而非征询对话框。
- 高频调用。用户征询会打断对话；不要在循环中触发它。
- 任何服务器可以在事后验证的事项。请直接验证、返回错误，并让模型用文本向用户说明。

### 人在回路桥梁

用户征询与采样（sampling）共同支撑 MCP 的“人在回路（human-in-the-loop）”模型。服务器代理循环可以为用户输入（用户征询）或模型推理（采样）而暂停。Phase 13 · 11 已介绍采样；本课介绍用户征询。将二者结合，即可实现完整的循环内控制。

## 动手实践

`code/main.py` 扩展了笔记服务器，包含：

- 响应 `roots/list` 的处理器，并在根目录列表变更通知后重新查询。
- 当多个笔记匹配时使用 `elicitation/create` 进行消除歧义的 `notes_delete` 工具。
- 使用 URL 模式用户征询打开首次运行配置页面的 `notes_setup` 工具（模拟）。
- 拒绝针对已声明根目录之外 URI 操作的边界检查。

演示运行三个场景：正常路径（一个匹配项）、消除歧义（三个匹配项，触发征询）、越界写入（被拒绝）。

## 交付成果

本课生成 `outputs/skill-elicitation-form-designer.md`。对于可能需要用户确认或消除歧义的工具，该技能会设计征询表单模式和消息模板。

## 练习

1. 运行 `code/main.py`。触发消除歧义路径；确认模拟的用户回答被路由回工具。

2. 新增一个每次都需要用户征询确认的 `notes_archive` 工具（破坏性提示）。体验如何：这与模型用文本再次询问相比有什么区别？

3. 为首次运行 OAuth 流程实现 URL 模式用户征询。注意漂移风险并添加 SDK 版本守卫。

4. 扩展 `roots/list` 处理：当通知到达时，服务器应原子性地重新读取并重新扫描可能已超出作用范围的打开文件句柄。

5. 阅读 GitHub 上 SEP-1036 的问题讨论串。找出一个影响服务器应如何处理 URL 模式回调的未决问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 根目录（Root） | “同意边界” | 客户端允许服务器访问的 URI |
| `roots/list` | “服务器请求作用范围” | 客户端返回当前根目录集合 |
| `notifications/roots/list_changed` | “用户更改了作用范围” | 客户端发出根目录集合已变更的信号 |
| 用户征询（Elicitation） | “在调用中途询问用户” | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | “该方法” | 用户征询请求的 JSON-RPC 方法 |
| 表单模式（Form mode） | “由模式驱动的表单” | 在客户端 UI 中渲染的扁平 JSON Schema 表单 |
| URL 模式（URL mode） | “浏览器重定向” | SEP-1036 实验性；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | “用户响应结果” | 服务器需要处理的三种分支 |
| 消除歧义（Disambiguation） | “选择一个” | 工具存在 N 个候选项时常见的用户征询用例 |
| 扁平表单（Flat form） | “仅顶层属性” | 用户征询模式不支持嵌套 |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — 根目录规范参考
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 用户征询规范参考
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增功能概览
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式用户征询提案（实验性，存在漂移风险）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — 用户体验概览
