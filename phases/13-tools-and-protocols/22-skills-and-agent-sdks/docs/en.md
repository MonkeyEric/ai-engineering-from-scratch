# 技能（Skills）与智能体 SDK（Agent SDKs）—— Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 回答“有哪些工具”，技能（Skills）回答“如何完成一项任务”。2026 年的技术栈将两者分层叠加。Anthropic 的 Agent Skills（2025 年 12 月发布的开放标准）以 SKILL.md 形式提供，并支持渐进式披露（progressive disclosure）。OpenAI 的 Apps SDK 则是 MCP 加上小部件元数据（widget metadata）。AGENTS.md（目前已有 60,000 多个仓库采用）位于仓库根目录，作为项目级智能体上下文。本节课介绍三者的覆盖范围，并构建一个可在不同智能体之间迁移的最小 SKILL.md + AGENTS.md 组合包。

**类型：** 学习
**语言：** Python（标准库，SKILL.md 解析器与加载器）
**前置要求：** Phase 13 · 07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 区分三个层级：AGENTS.md（项目上下文）、SKILL.md（可复用知识）、MCP（工具）。
- 编写带有 YAML 前置元数据（frontmatter）和渐进式披露（progressive disclosure）的 SKILL.md。
- 以文件系统方式将技能加载到智能体运行时（agent runtime）中。
- 将一个技能与 MCP 服务器及 AGENTS.md 组合，使一个包能在 Claude Code、Cursor 和 Codex 中工作。

# 问题

一名工程师将“撰写发布说明”的工作流提炼成一个多步提示词：“读取最近合并的 PR，按区域分组，逐条总结，按照团队风格撰写变更日志条目，发布到 Slack 草稿。”他把这份流程放在团队的 Notion 文档中。

现在他希望在 Claude Code、Cursor 和 Codex CLI 中使用这套工作流。每个智能体加载指令的方式都不同：Claude Code 斜杠命令、Cursor rules、Codex `.codex.md`。这名工程师只能把工作流复制三份，并分别维护。

AGENTS.md 和 SKILL.md 共同解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的智能体在会话启动时都会读取它。它回答：“这个项目如何运作？约定是什么？运行测试用哪些命令？”
- **SKILL.md** 是一个可移植的包：YAML 前置元数据（name、description）+ Markdown 正文 + 可选资源。支持技能的智能体可以按需按名称加载它们。
- **MCP**（第 13 阶段 · 06-14）负责技能需要调用的工具。

三个层级，一个可移植的产物。

## 概念

### AGENTS.md（agents.md）

2025 年底发布，截至 2026 年 4 月已有 60,000 多个仓库采用。仓库根目录下的一个文件。格式如下：

```markdown
# 项目：my-service

## 约定
- TypeScript 开启严格模式。
- Python 端使用 Pydantic 定义模型。
- 测试使用 `pnpm test` 运行。

## 构建与运行
- `pnpm dev` 启动本地开发服务器。
- `pnpm build` 构建生产包。
```

智能体在会话启动时读取该文件，并据此校准在该项目中的行为。2026 年的主流编程智能体都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: 按照本项目风格，为最近合并的 PR 撰写变更日志条目。
---

# 发布说明撰写器

被调用时，执行以下步骤：

1. 列出自上一个标签以来合并的 PR。使用 `gh pr list --base main --state merged`。
2. 按标签分组：feature、fix、chore、docs。
3. 每组中的每个 PR 写一行：`- <title> (#<num>)`。
4. 起草发布说明并暂存到 CHANGELOG.md。

如果用户说 "ship"，则运行 `git tag vX.Y.Z` 和 `gh release create`。

## 注意事项

- 不要包含没有 PR 的提交。
- 公开变更日志中跳过 "chore" 条目。
```

前置元数据（frontmatter）声明技能的身份。正文是技能加载时展示给模型的提示词。

### 渐进式披露（Progressive Disclosure）

技能可以引用子资源（sub-resources），智能体仅在需要时才获取它们。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 中说“详见 style-guide.md 中的风格规则”。智能体仅在技能正在运行时才会拉取 style-guide.md。这样可以避免用模型可能不需要的细节填充提示词。

### 文件系统发现（Filesystem Discovery）

智能体运行时会扫描已知目录以查找 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载依据文件夹名称和前置元数据中的 `name` 字段。Claude Code、Anthropic Claude Agent SDK 以及 SkillKit（跨智能体）都遵循这一模式。

### Anthropic Claude Agent SDK

TypeScript 的 `@anthropic-ai/claude-agent-sdk` 与 Python 的 `claude-agent-sdk` 在会话启动时加载技能，并在运行时将其暴露为可调用的“智能体（agents）”。当用户调用时，智能体循环（agent loop）会将请求分派给对应技能。

### OpenAI Apps SDK

2025 年 10 月发布，直接基于 MCP 构建。它将 OpenAI 此前的 Connectors 与 Custom GPT Actions 统一到一个开发者界面下。一个 Apps SDK 应用包含：

- 一个 MCP 服务器（工具、资源、提示词）。
- 用于 ChatGPT UI 的小部件元数据（widget metadata）。
- 可选的 MCP Apps `ui://` 资源，用于交互式界面。

同一协议，更丰富的用户体验。

### 通过 SkillKit 实现跨智能体可移植性

SkillKit 等跨智能体分发层会将单个 SKILL.md 翻译成 32 种以上 AI 智能体（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的本地格式。一份事实来源，多处消费。

### 三层栈

| 层级 | 文件 | 加载时机 | 用途 |
|------|------|----------|------|
| AGENTS.md | 仓库根目录 | 会话启动 | 项目级约定 |
| SKILL.md | skills 目录 | 技能被调用 | 可复用工作流 |
| MCP server | 外部进程 | 需要工具时 | 可调用动作 |

三者组合工作：智能体会话启动时读取 AGENTS.md，用户调用技能，技能指令中包含 MCP 工具调用，智能体通过 MCP 客户端进行分派。

## 使用它

`code/main.py` 提供了一个基于标准库的 SKILL.md 解析器与加载器。它发现 `./skills/` 下的技能，解析 YAML 前置元数据与 Markdown 正文，并生成以技能名称为键的字典。随后它模拟一个智能体循环，按名称调用 `release-notes-writer`。

关注要点：

- 使用极简的标准库解析器解析 YAML 前置元数据（不依赖 `pyyaml`）。
- 技能正文原样存储；调用时由智能体将其追加到系统提示词前。
- 通过 `read_subresource` 函数按需拉取引用文件，演示渐进式披露。

## 交付它

本节课生成 `outputs/skill-agent-bundle.md`。给定一个工作流，该技能会生成组合在一起的 SKILL.md + AGENTS.md + MCP 服务器蓝图（MCP-server-blueprint）包，可在不同智能体间移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能，并确认加载器能识别它。

2. 为本课程仓库编写一份 AGENTS.md。包含测试命令、风格约定以及第 13 阶段的心智模型。

3. 将团队内部文档中的多步骤工作流移植到一份 SKILL.md 中，并在 Claude Code 中验证它能被加载。

4. 手动将该技能翻译成 Cursor 和 Codex 的本地规则格式。统计格式之间的差异——这就是 SkillKit 自动化的翻译面。

5. 阅读 Anthropic Agent Skills 博客文章。找出 Claude Agent SDK 中的一项本节课加载器未覆盖的功能。（提示：智能体子调用（agent sub-invocation）。）

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| SKILL.md | “技能文件” | YAML 前置元数据加 Markdown 正文，由智能体运行时加载 |
| AGENTS.md | “仓库根目录智能体上下文” | 会话启动时读取的项目级约定文件 |
| Progressive disclosure | “懒加载子资源” | 技能正文引用文件，仅在需要时才拉取 |
| Frontmatter | “顶部的 YAML 块” | `---` 分隔符中的元数据（name、description） |
| Claude Agent SDK | “Anthropic 的技能运行时” | `@anthropic-ai/claude-agent-sdk`，加载技能并路由 |
| OpenAI Apps SDK | “MCP + 小部件元数据” | OpenAI 基于 MCP 构建的开发者界面，附加 ChatGPT UI 钩子 |
| Skill discovery | “文件系统扫描” | 遍历已知目录查找 SKILL.md，并按名称索引 |
| Cross-agent portability | “一个技能多处使用” | 通过 SkillKit 类工具将一个 SKILL.md 翻译成 32 种以上智能体格式 |
| Agent Skill | “可移植知识” | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | “MCP 加 ChatGPT UI” | Connectors 与 Custom GPTs 在 MCP 上的统一 |

## 延伸阅读

- [Anthropic — Agent Skills 公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式与采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例
