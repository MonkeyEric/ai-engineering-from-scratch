# 构建 MCP 服务器 —— Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio 的 hello-world。真正的服务器会暴露工具、资源和提示，处理能力协商，发出结构化错误，并且跨 SDK 表现一致。本节课从头构建一个 notes 服务器：stdlib stdio 传输、JSON-RPC 分发、三种服务器原语，以及一种纯函数风格，让你在进阶时可以无缝迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型：** 构建  
**语言：** Python（标准库，stdio MCP 服务器）  
**前置知识：** Phase 13 · 06（MCP 基础）  
**时间：** 约 75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环，从 stdin 读取 JSON-RPC 消息并将响应写入 stdout。
- 按照 JSON-RPC 2.0 规范和 MCP 的附加错误码发出结构化错误响应。
- 将标准库实现迁移到 FastMCP（Python SDK）或 TypeScript SDK，而无需重写工具逻辑。

## 问题背景

在你使用远程传输（Phase 13 · 09）或认证层（Phase 13 · 16）之前，你需要一个干净的本地服务器。本地即 stdio：服务器由客户端作为子进程启动，消息通过换行分隔的 stdin/stdout 流动。

2025-11-25 规范规定，stdio 消息编码为显式以 `\n` 分隔的 JSON 对象。这里不使用 SSE；SSE 是旧的远程模式，并将在 2026 年中期被移除（Atlassian 的 Rovo MCP 服务器已于 2026 年 6 月 30 日弃用；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON 对象就是全部的线格式。

notes 服务器是一个良好的示例，因为它锻炼了全部三种服务器原语。工具执行变更（`notes_create`）。资源暴露数据（`notes://{id}`）。提示提供模板（`review_note`）。本节课的结构可推广到任何领域。

## 核心概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC 信封的内容。调试日志应输出到 stderr。
- 每个请求都必须以携带相同 `id` 的响应匹配。
- 通知不得响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的能力。客户端依赖能力集合来控制功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目包含 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块（content block）是带类型的。最常见的：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形态。协议级错误（未知方法、参数错误）属于 JSON-RPC 错误。工具级错误（调用有效但工具执行失败）以 `{content: [...], isError: true}` 返回。这让模型能在上下文中看到失败。

### 实现资源

资源按设计是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义方案如 `notes://`。

当你将数据作为资源暴露而非工具时：

- 模型不会“调用”它；客户端可在用户请求时将其注入上下文。
- 订阅（subscription）让服务器在资源变化时推送更新（Phase 13 · 10）。
- Phase 13 · 14 用 `ui://` 扩展为交互式资源。

### 实现提示

提示（prompt）是带命名参数的模板。宿主将其展示为斜杠命令。`review_note` 提示可能接收 `note_id` 参数，并生成一个多消息提示模板，客户端将其喂给自己的模型。

### Stdio 传输细节

- 换行分隔的 JSON。无长度前缀分帧。
- 不要缓冲。每次写入后执行 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### 注解

每个工具可携带描述安全属性的 annotations：

- `readOnlyHint: true` —— 纯读取，可安全重试。
- `destructiveHint: true` —— 不可逆副作用；客户端应确认。
- `idempotentHint: true` —— 相同输入产生相同输出。
- `openWorldHint: true` —— 与外部系统交互。

客户端用这些来决定 UX（确认对话框、状态指示器）和路由（Phase 13 · 17）。

### 进阶路径

`code/main.py` 中的标准库服务器约 180 行。FastMCP（Python）将相同逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等价的形态。进阶路径是即插即用的，当你准备好时即可迁移；概念（能力、分发、内容块）相同。

## 使用它

`code/main.py` 是一个基于 stdio 的完整 notes MCP 服务器，仅使用标准库。它处理 `initialize`、`tools/list`、`tools/call` 三种工具（`notes_list`、`notes_search`、`notes_create`）、每条笔记的 `resources/list` 和 `resources/read`，以及 `review_note` 提示。你可以通过管道输送 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

需要关注：

- 分发器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器都返回内容块列表，而不是裸字符串。
- 当执行器抛出异常时设置 `isError: true`。

## 交付

本节课生成 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（notes、tickets、files、database），该技能会搭建一个 MCP 服务器，正确划分工具 / 资源 / 提示，并提供 SDK 迁移路径。

## 练习

1. 运行 `code/main.py`，并用手动构造的 JSON-RPC 消息驱动它。练习 `notes_create`，然后用 `resources/read` 检索新笔记。

2. 添加 `notes_delete` 工具，带 `annotations: {destructiveHint: true}`。验证客户端会显示确认对话框（这需要真实宿主；Claude Desktop 可用）。

3. 实现 `resources/subscribe`，使得笔记被修改时服务器推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器迁移到 FastMCP。Python 文件应缩减到 80 行以内。线级行为必须一致；用相同的 JSON-RPC 测试套件验证。

5. 阅读规范中的 `server/tools` 章节，找出本节课服务器未实现的工具定义字段。（提示：有多个；选一个并添加。）

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| MCP server | "暴露工具的东西" | 通过 stdio 或 HTTP 使用 MCP JSON-RPC 通信的进程 |
| stdio transport | "子进程模型" | 由客户端启动服务器；通过 stdin/stdout 通信 |
| Dispatcher | "方法路由器" | JSON-RPC 方法名到处理函数的映射 |
| Content block | "工具结果片段" | 工具响应 `content` 数组中的带类型元素 |
| `isError` | "工具级失败" | 表示工具失败；与 JSON-RPC 错误区分开 |
| Annotations | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 基于 MCP 协议的装饰器式高级框架 |
| Resource URI | "可寻址数据" | `file://`、`db://` 或自定义方案，用于标识资源 |
| Prompt template | "斜杠命令简介" | 服务器提供的带参数槽的模板，供宿主 UI 使用 |
| Capability declaration | "功能开关" | 在 `initialize` 中按原语声明的标志 |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) —— 参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) —— 对等的 TypeScript 实现
- [FastMCP — server framework](https://gofastmcp.com/) —— MCP 服务器的装饰器式 Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) —— 使用任一 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) —— `tools/*` 消息的完整参考
