# MCP Apps —— 通过 `ui://` 提供交互式 UI 资源

> 纯文本的工具输出限制了智能体（agent）能展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日正式发布）允许工具返回经过沙箱隔离的交互式 HTML，并直接在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中内嵌渲染。仪表盘、表单、地图、3D 场景，都可以通过这一个扩展实现。本节课将介绍 `ui://` 资源方案、`text/html;profile=mcp-app` MIME 类型、iframe 沙箱 postMessage 协议，以及允许服务器渲染 HTML 所带来的安全面。

**Type:** Build
**Languages:** Python（标准库，UI 资源发射器），HTML（示例应用）
**Prerequisites:** Phase 13 · 07（MCP server），Phase 13 · 10（resources）
**Time:** ~75 分钟

## 学习目标

- 从工具调用返回 `ui://` 资源，并设置正确的 MIME 类型和元数据。
- 通过 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现用于 UI 与宿主通信的 iframe 沙箱 postMessage JSON-RPC。
- 应用 CSP 和权限策略（permissions policy）默认值，以防御源自 UI 的攻击。

## 问题背景

2025 年风格的 `visualize_timeline` 工具可能会返回“以下是按时间顺序排列的 14 条笔记：……”。这只是一段文字。用户真正想要的是交互式时间线。在 MCP Apps 出现之前，选择有限：客户端专用的小部件 API（Claude artifacts、OpenAI Custom GPT HTML），或者根本没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）将该契约标准化。工具结果包含一个 `resource`，其 URI 为 `ui://...`，MIME 类型为 `text/html;profile=mcp-app`。宿主在沙箱化的 iframe 中渲染它，iframe 具有受限的 CSP 且默认没有网络访问，除非显式授予。UI 通过一种小型的 postMessage JSON-RPC 方言向宿主发送消息。

每个兼容的客户端（Claude Desktop、ChatGPT、Goose、VS Code）都会以相同方式渲染同一个 `ui://` 资源。一个服务器，一个 HTML 包，通用 UI。

## 核心概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

宿主随后调用 `resources/read` 获取 `ui://notes/timeline` URI，并返回：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

宿主在沙箱化的 `<iframe>` 中渲染 HTML，具备：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明的更严格策略）
- 通过响应头应用服务器声明的 CSP。
- 没有 cookie，也没有来自宿主源的 localStorage。
- 网络访问仅限于 CSP 中的 `connectSrc`。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信。一种小型的 JSON-RPC 2.0 方言：

务必始终将 `targetOrigin` 固定为对等方的确切源，并在接收端将 `event.origin` 与允许列表进行校验，然后再处理任何载荷。切勿在该通道的任何一侧使用 `"*"`——消息体中携带工具调用和资源读取。

```js
// iframe 到宿主（固定为宿主源）
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 宿主到 iframe（固定为 iframe 源）
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 两侧的接收器
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // 此时可以安全处理 event.data
});
```

UI 可调用的宿主端方法：

- `host.callTool(name, arguments)` —— 调用服务器工具。
- `host.readResource(uri)` —— 读取 MCP 资源。
- `host.getPrompt(name, arguments)` —— 获取提示模板。
- `host.close()` —— 关闭 UI。

每次调用仍然经过 MCP 协议并继承服务器权限。

### 权限

`_meta.ui.permissions` 列表请求额外的能力：

- `camera` —— 访问用户摄像头（用于扫描文档的 UI）。
- `microphone` —— 语音输入。
- `geolocation` —— 地理位置。
- `network:*` —— 比 `connectSrc` 更广泛的网络访问。

每项权限都会作为提示在用户看到 UI 之前展示。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新的攻击面：

- **通过 UI 的提示注入（prompt injection）。** 恶意服务器 UI 可以显示看起来像系统消息的文本并欺骗用户。宿主渲染应明显区分服务器 UI 与宿主 UI。
- **通过 `connectSrc` 的数据外泄。** 如果 CSP 允许 `connect-src: *`，UI 可以将数据发送到任意位置。默认值应严格限制。
- **点击劫持（clickjacking）。** UI 覆盖宿主界面。宿主必须防止 z-index 操作并强制执行透明度规则。
- **窃取焦点。** UI 占据键盘焦点并捕获下一条消息。宿主必须拦截。

Phase 13 · 15 将把这些作为 MCP 安全的一部分深入讲解；本节课仅作介绍。

### `ui/initialize` 握手

iframe 加载完成后，会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主响应能力列表和会话令牌。UI 在后续每次宿主调用中都使用该会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露了两个便捷原语：

- `AppRenderer`（服务器端）—— 包装 React / Vue / Solid 组件，并发射带有正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）—— 接收资源，挂载 iframe，并调解 postMessage。

你可以使用它们，也可以手写 HTML 和 JSON-RPC。

### 生态现状

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持情况：

- **Claude Desktop。** 自 2026 年 1 月起完全支持。
- **ChatGPT。** 通过 Apps SDK 完全支持（底层使用相同的 MCP Apps 协议）。
- **Cursor。** 测试中；通过设置启用。
- **VS Code。** 仅 Insider 版本。
- **Goose。** 完全支持。
- **Zed、Windsurf。** 已纳入路线图。

生产环境中的服务器：仪表盘、地图可视化、数据表格、图表构建器、沙箱 IDE 预览。

## 动手实践

`code/main.py` 扩展了笔记服务器，添加了一个 `visualize_timeline` 工具，该工具返回 `ui://notes/timeline` 资源，并处理针对该 URI 的 `resources/read`，返回一个虽小但完整的 HTML 包，其中包含 SVG 时间线。HTML 使用标准库模板渲染——无需构建系统。postMessage 在 JS 注释中勾勒出来，因为标准库无法驱动浏览器。

值得关注的点：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 在没有网络访问的情况下渲染；所有数据均内联。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（已文档化，但在此标准库示例中处于非活动状态）。

## 交付成果

本节课产出 `outputs/skill-mcp-apps-spec.md`。针对一个能从交互式 UI 中受益的工具，该技能产物将包含完整的 MCP Apps 契约：`ui://` URI、CSP、权限、postMessage 入口点以及安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查生成的 HTML。直接在浏览器中打开该 HTML；验证 SVG 是否渲染。然后勾勒出 UI 调用 `host.callTool("notes_update", ...)` 时将使用的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码需要做出哪些改动？

3. 添加第二个 UI 资源 `ui://notes/editor`，包含一个用于就地编辑笔记的表单。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可能在哪些地方注入内容？iframe 沙箱能防御什么、不能防御什么？

5. 阅读 SEP-1724 规范，找出 MCP Apps SDK 中的一项本玩具实现未使用的能力。（提示：组件级状态同步。）

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| MCP Apps | "交互式 UI 资源" | 2026-01-26 发布的 SEP-1724 扩展 |
| `ui://` | "应用 URI 方案" | 用于 UI 包的资源方案 |
| `text/html;profile=mcp-app` | "该 MIME" | MCP App HTML 的内容类型 |
| Iframe sandbox | "渲染容器" | 使用 CSP 和权限对 UI 进行浏览器沙箱隔离 |
| postMessage JSON-RPC | "UI 到宿主的通信线" | 用于宿主调用的小型 JSON-RPC-over-postMessage 方言 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果链接到 UI 资源的元数据 |
| CSP | "Content-Security-Policy" | 声明脚本、网络、样式等允许来源 |
| AppRenderer | "服务器 SDK 原语" | 将框架组件转换为 `ui://` 资源 |
| AppFrame | "客户端 SDK 原语" | 挂载 iframe 并调解 postMessage 的助手 |
| `ui/initialize` | "握手" | UI 到宿主的第一个 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现与 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — 高层文档
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布文章
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格 SDK 参考
