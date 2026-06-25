# 模型上下文协议（Model Context Protocol, MCP）

> 2025 年之前构建的每个大语言模型（LLM）应用都有自己的工具模式。随后 Anthropic 推出了 MCP，Claude 采用了它，OpenAI 采用了它，到 2026 年它已成为把任意 LLM 连接到任意工具、数据源或智能体的默认传输格式。写一次 MCP 服务器，所有宿主都能与它通信。

**类型：** 构建
**语言：** Python
**前置知识：** 第 11 阶段 · 09（函数调用），第 11 阶段 · 03（结构化输出）
**时长：** 约 75 分钟

## 问题背景

你发布了一个聊天机器人，需要三个工具：数据库查询、日历 API 和文件读取器。你为 Claude 写了三份 JSON Schema。然后销售团队希望在 ChatGPT 中使用同样的工具——你又为 OpenAI 的 `tools` 参数重写了一遍。接着你又接入 Cursor、Zed 和 Claude Code——三次重写，每一种的 JSON 约定都略有不同。一周后 Anthropic 新增了一个字段；你得更新六份模式。

这就是 2025 年之前的现实。每个宿主（host，运行 LLM 的实体）和每个服务器（server，暴露工具与数据的实体）都使用自定义协议。规模化意味着 N×M 的集成矩阵。

模型上下文协议（MCP）将这个矩阵折叠为一条规范：一个基于 JSON-RPC 的规范。一个服务器暴露工具（tools）、资源（resources）和提示词（prompts）。任何兼容的宿主——Claude Desktop、ChatGPT、Cursor、Claude Code、Zed 以及一长串智能体框架——都能无需定制胶水代码地发现并调用它们。

到 2026 年初，MCP 已成为三大厂商（Anthropic、OpenAI、Google）以及所有主流智能体框架的默认工具与上下文协议。

## 核心概念

![MCP：一个宿主、一个服务器、三种能力](../assets/mcp-architecture.svg)

**三种原语。** MCP 服务器只暴露三类东西。

1. **工具（tools）**——模型可以调用的函数。对应 OpenAI 的 `tools` 或 Anthropic 的 `tool_use`。每个工具有名称、描述、JSON Schema 输入以及处理函数。
2. **资源（resources）**——模型或用户可以请求的只读内容（文件、数据库行、API 响应）。通过 URI 寻址。
3. **提示词（prompts）**——用户可作为快捷方式调用的可复用提示模板。

**传输格式。** JSON-RPC 2.0，基于 stdio、WebSocket 或可流式 HTTP。每条消息都是 `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。发现方法有 `tools/list`、`resources/list`、`prompts/list`；调用方法有 `tools/call`、`resources/read`、`prompts/get`。

**宿主、客户端与服务器。** 宿主是 LLM 应用（如 Claude Desktop）。客户端（client）是宿主内部的一个子组件，只与一个服务器通信。服务器是你的代码。一个宿主可以同时挂载多个服务器。

### 握手

每个会话以 `initialize` 开始。客户端发送协议版本及其能力。服务器回应其版本、名称和支持的能力集（`tools`、`resources`、`prompts`、`logging`、`roots`）。之后的所有交互都基于这些能力进行协商。

### MCP 不是什么

- 它不是检索 API。RAG（第 11 阶段 · 06）仍然决定拉取什么；MCP 只是将检索结果作为资源暴露出来的传输层。
- 它不是智能体框架。MCP 是管道；LangGraph、PydanticAI 和 OpenAI Agents SDK 等框架位于其上。
- 它不绑定 Anthropic。该规范及参考实现以开源形式托管在 `modelcontextprotocol` 组织下。

## 动手构建

### 第 1 步：最小 MCP 服务器

官方 Python SDK 是 `mcp`（曾用名 `mcp-python`）。高级封装 `FastMCP` 可为处理函数添加装饰器。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个整数相加。"""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """返回应用当前的 JSON 配置。"""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """检查代码的正确性与风格。"""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

三个装饰器分别注册了三类原语。类型提示会生成宿主看到的 JSON Schema。在 Claude Desktop 或 Claude Code 中运行，将服务器入口指向该文件即可。

### 第 2 步：从宿主调用 MCP 服务器

官方 Python 客户端使用 JSON-RPC。把它和 Anthropic SDK 配对只需十几行代码。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` 返回的就是 LLM 将看到的模式。生产宿主会在每次对话中注入这些模式，使模型能够生成 `tool_use` 块，再由客户端转发给服务器。

### 第 3 步：可流式 HTTP 传输

Stdio 适合本地开发。对于远程工具，使用可流式 HTTP——每个请求一个 POST，可选 Server-Sent Events 发送进度；自 2025-06-18 规范修订版起已支持。

```python
# 在服务器入口中
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

宿主配置（Claude Desktop 的 `mcp.json` 或 Claude Code 的 `~/.mcp.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

服务器使用同样的装饰器；只有传输层发生变化。

### 第 4 步：作用域与安全

MCP 工具是在他人信任边界上运行的任意代码。三个必备模式。

- **能力白名单。** 宿主暴露 `roots` 能力，使服务器只能看到允许的路径。在工具处理函数中强制执行；不要信任模型提供的路径。
- **人工确认再变更。** 只读工具可自动执行。写入/删除工具必须要求确认——当服务器在工具元数据中设置 `destructiveHint: true` 时，宿主会弹出审批 UI。
- **工具投毒防御。** 恶意资源可能包含隐藏的提示注入指令（例如“总结时还要调用 `exfil`”）。将资源内容视为不可信数据；永远不要让它进入系统消息领域。参见第 11 阶段 · 12（护栏）。

详见 `code/main.py`，其中包含可运行的服务器 + 客户端示例，演示了上述全部内容。

## 2026 年仍会踩的坑

- **模式漂移。** 模型在第 1 轮看到了 `tools/list`。第 5 轮工具集合发生变化。模型调用了已不存在的工具。宿主应在收到 `notifications/tools/list_changed` 时重新列出工具。
- **过大的资源块。** 把 2MB 文件作为资源倾倒会浪费上下文。在服务器端分页或总结。
- **服务器过多。** 挂载 50 个 MCP 服务器会耗尽工具预算（第 11 阶段 · 05）。大多数前沿模型在超过约 40 个工具后性能下降。
- **版本错位。** 规范修订版（2024-11、2025-03、2025-06、2025-12）会引入破坏性字段。在 CI 中锁定协议版本。
- **Stdio 死锁。** 向 stdout 输出日志的服务器会破坏 JSON-RPC 流。只记录到 stderr。

## 使用建议

2026 年的 MCP 技术栈：

| 场景 | 选择 |
|------|------|
| 本地开发、单用户工具 | Python `FastMCP`，stdio 传输 |
| 远程团队工具 / SaaS 集成 | 可流式 HTTP，OAuth 2.1 认证 |
| TypeScript 宿主（VS Code 扩展、Web 应用） | `@modelcontextprotocol/sdk` |
| 高吞吐量服务器、强类型访问 | 官方 Rust SDK（`modelcontextprotocol/rust-sdk`） |
| 探索生态服务器 | `modelcontextprotocol/servers` 单体仓库（Filesystem、GitHub、Postgres、Slack、Puppeteer） |

经验法则：如果工具是只读、可缓存，并且会被两个或以上宿主调用，就将其作为 MCP 服务器发布。如果只是一次性内联逻辑，则保留为本地函数（第 11 阶段 · 09）。

## 交付成果

保存 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: 设计并搭建一个包含工具、资源及安全默认值的 MCP 服务器。
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

给定一个领域（内部 API、数据库、文件源）以及将挂载该服务器的宿主，输出：

1. 原语映射。哪些能力应作为 `tools`（动作），哪些作为 `resources`（只读数据），哪些作为 `prompts`（用户调用的模板）。每个原语一行。
2. 认证方案。Stdio（受信任的本地环境）、带 API 密钥的可流式 HTTP，或带 PKCE 的 OAuth 2.1。选择并说明理由。
3. 模式草稿。每个工具参数的 JSON Schema，`description` 字段针对模型工具选择调优（而非 API 文档）。
4. 破坏性操作清单。每个会改变状态的工具；要求设置 `destructiveHint: true` 并人工审批。
5. 测试计划。每个工具：一份仅验证模式的契约测试、一次通过 MCP 客户端的往返测试、一个红队提示注入用例。

拒绝交付未经审批路径就向磁盘写入或调用外部 API 的服务器。拒绝在单个服务器上暴露超过 20 个工具；应拆分为按领域划分的服务器。
```

## 练习

1. **简单。** 为 `demo-server` 添加一个 `subtract` 工具。从 Claude Desktop 连接。通过发送 `tools/list_changed` 通知，确认宿主无需重启即可识别新工具。
2. **中等。** 添加一个暴露 `/var/log/app.log` 最近 100 行的 `resource`。强制执行 roots 白名单，即使模型请求 `../etc/passwd` 也会将其阻止。
3. **困难。** 构建一个 MCP 代理，将三个上游服务器（Filesystem、GitHub、Postgres）多路复用为一个聚合面。处理名称冲突，并干净地转发 `notifications/tools/list_changed`。

## 关键术语

| 术语 | 大家的说法 | 实际含义 |
|------|-----------|---------|
| MCP | “LLM 的工具协议” | 向任意 LLM 宿主暴露工具、资源和提示词的 JSON-RPC 2.0 规范。 |
| Host | “Claude Desktop” | LLM 应用——拥有模型和用户 UI，挂载一个或多个客户端。 |
| Client | “连接” | 宿主内部每个服务器一个的连接，使用 JSON-RPC 与单个服务器通信。 |
| Server | “有工具的那个东西” | 你的代码；宣告工具/资源/提示词并处理其调用。 |
| Tool | “函数调用” | 模型可调用的动作，具有 JSON Schema 输入和文本/JSON 结果。 |
| Resource | “只读数据” | 通过 URI 寻址的内容（文件、行、API 响应），宿主可以请求。 |
| Prompt | “保存的提示词” | 用户可调用的模板（常带参数），以斜杠命令形式呈现。 |
| Stdio transport | “本地开发模式” | 宿主将服务器作为子进程启动；JSON-RPC 通过 stdin/stdout 传输。 |
| Streamable HTTP | “2025-06 的远程传输” | 请求用 POST，可选 SSE 发送服务器主动消息；取代了旧的仅 SSE 传输。 |

## 延伸阅读

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) —— 规范参考，按日期版本化。
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) —— Filesystem、GitHub、Postgres、Slack、Puppeteer 参考服务器。
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) —— 发布文章，含设计原理。
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) —— 本课使用的官方 SDK。
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) —— roots、destructive hints、tool poisoning。
- [Google A2A specification](https://google.github.io/A2A/) —— Agent2Agent 协议；与 MCP 的“智能体到工具”范围互补的“智能体到智能体”通信兄弟标准。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— MCP 在更广泛智能体设计模式库（增强型 LLM、工作流、自主智能体）中的位置。
