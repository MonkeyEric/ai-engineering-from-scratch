# 异步任务（Async Tasks，SEP-1686）——长时间工作的“立即调用、稍后获取”

> 真实的智能体工作往往耗时数分钟到数小时：CI 运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞 UI。SEP-1686 于 2025-11-25 合并，引入了任务（Tasks）原语：任何请求都可以被增强为任务，结果可以稍后获取，或通过状态通知以流式接收。漂移风险说明：截至 2026 年上半年，Tasks 仍处于实验阶段；SDK 接口仍在围绕该规范进行设计。

**类型：** 构建
**语言：** Python（标准库、异步任务状态机）
**前置条件：** Phase 13 · 07（MCP server），Phase 13 · 09（transports）
**时间：** ~75 分钟

## 学习目标

- 判断何时将工具从同步调用升级为任务增强型调用（服务器端运行超过 30 秒）。
- 掌握任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，使崩溃不会丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题

`generate_report` 工具会运行一个耗时数分钟的数据提取流水线。在同步模型下，可选方案如下：

1. 保持连接打开三分钟。远程传输会断开它；客户端超时；UI 冻结。
2. 立即返回占位符；要求客户端轮询自定义端点。这会破坏 MCP 的统一性。
3. 即发即弃；没有结果。

这些方案都不理想。SEP-1686 增加了第四种方案：任务增强（task augmentation）。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器立即返回任务 id。客户端轮询 `tasks/status`，完成后获取 `tasks/result`。服务器端状态在重启后仍然保留。

## 概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定），一个请求即可变为任务。服务器会立即返回：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器承诺保留状态的时间；超过 ttl 后，任务结果将被丢弃。

### 按工具选择加入

工具注解（tool annotations）可以声明任务支持：

- `taskSupport: "forbidden"` — 该工具始终以同步方式运行。适用于快速工具。
- `taskSupport: "optional"` — 客户端可以请求任务增强。
- `taskSupport: "required"` — 客户端必须使用任务增强。

`generate_report` 工具应为 `required`。`notes_search` 工具应为 `forbidden`。

### 状态

```
working  -> input_required -> working  （通过请求补充信息循环）
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是只追加（append-only）的：一旦进入 `completed`、`failed` 或 `cancelled`，任务即进入终止状态。

### 方法

- `tasks/status {taskId}` — 返回当前状态和可选的进度提示。
- `tasks/result {taskId}` — 阻塞返回；若尚未完成则返回 404。
- `tasks/cancel {taskId}` — 幂等；终止状态会忽略该请求。
- `tasks/list` — 可选；枚举活动和最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

采用流式接收而非轮询能带来更好的用户体验。轮询始终作为最小接口被支持。

### 持久化状态

规范要求声明支持任务的服务器必须持久化状态。崩溃不应导致 ttl 内已完成的结果丢失。存储可以是 SQLite、Redis 或文件系统。本课 13 的实验装置使用文件系统。

### 取消语义

`tasks/cancel` 是幂等（idempotent）的。如果任务正在执行，服务器会尝试停止它（检查执行器协作式取消）。如果已是终止状态，该请求即为无操作。

### 崩溃恢复

服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将因进程死亡而遗留的 `working` 任务标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled`。

### 异步任务与采样

任务本身可以调用 `sampling/createMessage`。这就是长时间运行研究任务的工作方式：服务器的任务线程按需向客户端模型发起采样（sampling），而客户端 UI 将任务显示为 `working`，并附带定期进度更新。

### 为何仍处于实验阶段

SEP-1686 已于 2025-11-25 发布，但更广泛的路线图列出了三个开放问题：持久订阅原语（durable subscription primitives）、子任务（subtasks，父子任务关系）以及结果 ttl 标准化。预计该规范将在 2026 年演进。生产代码应仅将 Tasks 视为常见场景下的稳定功能，并对子任务可能带来的未来 SDK 变更做好防护。

## 使用它

`code/main.py` 实现了一个持久化任务存储（文件系统后端）和一个在后台线程运行的 `generate_report` 工具。客户端调用该工具后会立即获得任务 id，在 worker 更新进度期间轮询 `tasks/status`，完成后获取 `tasks/result`。取消有效；通过杀死 worker 线程并重新加载状态来模拟崩溃恢复。

需要关注：

- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- Worker 线程更新 `progress` 字段；轮询会显示它不断推进。
- 客户端发起的取消会设置一个事件；worker 检查后会提前退出。
- “崩溃”后的状态重载会将进行中的任务标记为 `failed`，错误为 `CRASH_RECOVERY`。

## 交付它

本课会生成 `outputs/skill-task-store-designer.md`。针对一个长时间运行的工具（研究、构建、导出），该 skill 会设计任务存储（状态结构、ttl、持久化）、选择合适的 taskSupport 标志，并草拟进度通知。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。
2. 在运行过程中调用 `tasks/cancel`。验证 worker 响应并且状态变为 `cancelled`。
3. 模拟崩溃恢复：杀死 worker 线程，重新启动加载器，并观察 `CRASH_RECOVERY` 失败模式。
4. 将存储扩展到 SQLite。持久化收益相同；查询能力增强（例如列出会话 X 的所有任务）。
5. 阅读 2026 年的 MCP 路线图文章。找出最有可能在未来一年影响 SDK API 设计的与 Tasks 相关的开放问题。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 任务（Task） | “长时间运行的工具调用” | 通过 `_meta.task` 增强以异步执行的请求 |
| SEP-1686 | “Tasks 规范” | 于 2025-11-25 加入 Tasks 的规范演进提案（Spec Evolution Proposal） |
| `_meta.task` | “任务信封” | 包含 id、state、ttl 的每次请求元数据 |
| taskSupport | “工具标志” | 每个工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | “轮询方法” | 获取当前状态和可选的进度提示 |
| `tasks/result` | “获取结果” | 返回已完成负载，若未完成则返回 404 |
| `tasks/cancel` | “停止它” | 幂等的取消请求 |
| `ttl` | “保留预算” | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | “状态推送” | 服务器主动发起的状态变更事件 |
| 持久化存储（Durable store） | “崩溃安全状态” | 文件系统 / SQLite / Redis 持久层 |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案与完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 设计详解与原理
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 机制与状态机
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK 级任务实现模式
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 开放问题与 2026 年优先事项，包括子任务
