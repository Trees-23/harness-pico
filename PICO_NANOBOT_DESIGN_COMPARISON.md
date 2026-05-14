# pico 与 nanobot 设计差异详解

本文基于以下两个源码目录对比：

- pico: `/home/kdm/MyWorkSpace/AgentLearnSpace/AllProjects/pico`
- nanobot: `/home/kdm/MyWorkSpace/AgentLearnSpace/AllProjects/nanobot-main`

核心结论：pico 的核心链路 `loop -> runner -> tool -> context -> skill` 已按 nanobot 的主要接口语义完成对齐：主链路使用 provider-native messages、Nanobot 工具名、`PicoRunSpec.initial_messages` / `PicoRunResult` 合同、SkillsLoader 注入、runner 临时上下文治理、tool result 规范化、持久化清洗和 `_snip_history()`。仍保留的差异主要在产品层：消息总线、多渠道会话、web/message/my 工具、完整 MCP 连接链路、常驻 idle scan、全异步执行模型等。

---

## 1. Tool 板块设计对比

### 1.1 nanobot 的 tool 设计

nanobot 的工具体系在 `nanobot/agent/tools/` 下，核心抽象是：

- `Tool`: 所有工具的抽象基类，定义 `name`、`description`、`parameters`、`read_only`、`exclusive`、`concurrency_safe`、`execute()`。
- `Schema`: JSON Schema 片段校验器，用于参数类型、枚举、范围、数组、对象等验证。
- `tool_parameters`: 装饰器，把 JSON Schema 固定到工具类上。
- `ToolRegistry`: 工具注册、注销、schema 生成、参数 cast/validate、执行入口。

工具定义最终都会输出 OpenAI function/tool schema：

```python
{
  "type": "function",
  "function": {
    "name": tool.name,
    "description": tool.description,
    "parameters": tool.parameters,
  },
}
```

注册方式集中在 `AgentLoop._register_default_tools()`：

- 文件类：`read_file`、`write_file`、`edit_file`、`list_dir`
- 搜索类：`glob`、`grep`
- notebook：`notebook_edit`
- shell：`exec`，受 `exec_config.enable` 控制
- web：`web_search`、`web_fetch`，受 `web_config.enable` 控制
- 消息发送：`message`
- 子代理：`spawn`
- 定时任务：`cron`，仅当传入 `cron_service` 时注册
- 自我工具：`my`，受 tools config 控制
- MCP：通过 `connect_mcp_servers()` 动态注册 `mcp_*` 工具、资源、prompt wrapper

nanobot 的工具调用路径是：

1. 模型返回结构化 `tool_calls`。
2. `AgentRunner._execute_tools()` 将多个 tool call 按并发安全性分 batch。
3. 每个 call 进入 `_run_tool()`。
4. `_run_tool()` 先做 repeated external lookup 防护，再调用 `ToolRegistry.prepare_call()`。
5. `prepare_call()` 负责找工具、参数 cast、参数校验。
6. 校验通过后执行 `tool.execute(**params)`。
7. 结果被 `_normalize_tool_result()` 处理为空结果补位、大结果落盘、过长截断。
8. runner 将结果写成标准 tool message:

```python
{
  "role": "tool",
  "tool_call_id": tool_call.id,
  "name": tool_call.name,
  "content": normalized_result,
}
```

nanobot 工具实现的完整度较高：

- `read_file` 支持 UTF-8 文本、图片 content blocks、PDF 文本抽取、分页、设备路径黑名单、二进制拒绝、重复读取去重。
- `edit_file` 有多级匹配策略：精确匹配、trim 匹配、引号归一、近似诊断、缩进/引号风格保留。
- `glob`/`grep` 支持分页、类型过滤、上下文、固定字符串、输出模式等。
- `exec` 有 deny patterns、sandbox wrapper、workspace 限制、环境变量 allowlist、超时、输出截断、跨平台 shell。
- `web_fetch` 有 URL/SSRF 校验、重定向限制、外部内容 untrusted banner。
- `message` 是多渠道发送文件/消息的正式出口。
- `spawn` 与 `SubagentManager` 集成，可以把子代理结果回流到主会话。

### 1.2 pico 的 tool 设计

pico 的新工具体系在 `pico/tooling/` 下，结构明显来自 nanobot：

- `pico/tooling/base.py`: `Tool`、`Schema`、`tool_parameters`
- `pico/tooling/registry.py`: `ToolRegistry`
- `pico/tooling/pico_tools.py`: 标准工具 adapter
- `pico/tooling/capabilities.py`: 具体能力实现
- `pico/tooling/executor.py`: 工具执行网关
- `pico/tooling/ecosystem.py`: cron/spawn/notebook 以及少量 MCP tool wrapper/helper 等生态工具

pico 主链路模型可见基础工具：

- `list_dir`
- `read_file`
- `grep`
- `glob`
- `exec`
- `write_file`
- `edit_file`
- `delegate`
- `spawn`
- `cron`
- `notebook_edit`
- `mcp_*` tool wrapper/helper 有雏形，但当前 pico 没有 nanobot 那种 `connect_mcp_servers()` 完整连接链路，也没有 MCP resource/prompt wrapper 的完整动态注册体系

旧 Pico 名称 `list_files`、`search`、`run_shell`、`patch_file` 现在只作为 `ToolRegistry` alias 保留，用于 legacy `<tool>` 解析、评测脚本和旧测试兼容；它们不再出现在 `get_definitions()` 暴露给模型的 tool schema 中。

注册入口是 `build_standard_tool_registry(agent)`：

1. 注册基础文件/shell 工具。
2. 如果 `agent.depth < agent.max_depth`，注册 `delegate` 和 `spawn`。
3. 注册 `cron`。
4. 注册 `notebook_edit`。

pico 的调用路径：

1. `PicoRunner` 从模型响应里解析出工具调用。
2. provider-native messages/tool call 优先走 `complete_messages_with_tools()`；旧 string provider 通过 adapter 退化到 `complete_with_tools()` / `complete()`。legacy `<tool>...</tool>` 或 XML 风格工具语法仅作为兼容路径保留。
3. runner 调用 `agent.run_tool(name, args)`。
4. `PicoLoop.run_tool()` 调用 `ToolExecutor.execute()`。
5. `ToolExecutor` 调用 `ToolRegistry.prepare_call()` 做参数 cast/validate。
6. 执行前做 preflight security、重复调用防护、审批、防写冲突锁、workspace snapshot。
7. 执行 `tool.execute(**prepared_args)`，再更新 memory、trace metadata、process note。
8. runner 将结果写成 tool message，并进入下一轮模型调用。

pico 的 `ToolExecutor` 是它相对 nanobot 的一个本地增强点：它集中处理审批、路径安全、重复工具调用、写工具锁、workspace diff、memory 更新、trace metadata。这些职责在 nanobot 中分散在具体工具、runner 和 loop 周边。

### 1.3 tool 板块主要差距

pico 已经有标准 Tool/Registry/Executor 抽象，但未达到 nanobot 的地方包括：

1. 产品工具覆盖面仍不足  
   核心文件/搜索/shell 工具名和主能力已经对齐到 `edit_file`、`glob`、`grep`、`exec`、`list_dir`。pico 仍缺 nanobot 的 `web_search`、`web_fetch`、`message`、`my`、完整 MCP tool/resource/prompt wrapper 和连接链路。

2. 文件工具能力基本对齐核心路径  
   pico `read_file` 已支持文本分页、图片 content blocks、二进制拒绝、设备路径防护和 builtin skill 目录读取。PDF/更完整的 provider token 语义仍可继续补齐。

3. 编辑工具核心能力已对齐  
   pico 主工具改为 `edit_file`，支持精确匹配、trim 匹配、引号归一和近似诊断；`patch_file` 仅为旧名 alias。

4. 搜索工具核心命名已对齐  
   pico 主工具改为 `grep`/`glob`，支持 workspace 限制、分页/结果上限和 `output_mode=count` 等核心选项；高级上下文/类型过滤能力仍比 nanobot 简化。

5. shell 安全模型部分对齐  
   pico 主工具改为 `exec`，已有 deny patterns、workspace cwd、env allowlist、审批、内部 URL preflight、超时到 600s 和输出截断。nanobot 的 sandbox wrapper 和内部状态文件写保护仍更完整。

6. 多渠道工具缺失  
   nanobot 的 `message` 工具是“把文件/媒体真正发给用户”的工具。pico 没有等价消息总线/渠道工具。

7. 异步生态不完整  
   nanobot runner/tool 全链路 async。pico 工具接口是 async，但 runner 中大量地方通过同步 `agent.run_tool()` 和 `asyncio.run()` 桥接，整体仍是同步主循环。

---

## 2. loop -> runner -> context -> tool 核心链路

### 2.1 nanobot 核心链路

nanobot 的链路分层清晰：

```text
MessageBus
  -> AgentLoop.run()
  -> AgentLoop._dispatch()
  -> AgentLoop._process_message()
  -> ContextBuilder.build_messages()
  -> AgentLoop._run_agent_loop()
  -> AgentRunner.run()
  -> provider.chat_with_retry()/chat_stream_with_retry()
  -> AgentRunner._execute_tools()
  -> ToolRegistry / Tool.execute()
  -> tool messages append
  -> next runner iteration
  -> final assistant message
  -> AgentLoop._save_turn()
```

各环节职责：

- `AgentLoop.run()`  
  常驻消费 `MessageBus` 入站消息。定期调用 `auto_compact.check_expired()`，并处理 priority command。

- `_dispatch()`  
  对同一 session 加锁串行，对不同 session 并发。给当前 session 注册 pending queue，使同一会话中途发来的新消息进入 injection，而不是另开竞争任务。

- `_process_message()`  
  是产品层主入口。负责 session 读取、崩溃恢复、auto compact、slash command、token consolidation、工具上下文设置、用户消息提前持久化、调用 `_run_agent_loop()`、保存新增消息、清 checkpoint。

- `ContextBuilder.build_messages()`  
  生成 provider 需要的 chat messages。它不是把所有东西拼成一个字符串，而是返回:

```python
[
  {"role": "system", "content": system_prompt},
  *history,
  {"role": "user", "content": runtime_context + current_user_message},
]
```

- `AgentRunner.run()`  
  是纯 agent 执行循环。它不直接管理 session 文件，不关心 CLI/IM/Web UI。它拿 `AgentRunSpec` 执行模型调用、工具调用、checkpoint callback、injection callback、上下文临时治理。

- `ToolRegistry` / `Tool.execute()`  
  runner 通过 registry 拿 schema 给模型，也通过 registry 校验并执行工具。

数据流转重点：

1. session 里的持久历史是 `session.messages`。
2. `session.get_history(max_messages=0)` 返回从 `last_consolidated` 之后的未归档历史，并修正开头边界。
3. `ContextBuilder` 把 system prompt、历史、当前 user message 合成 `initial_messages`。
4. runner 每一轮从 `messages` 派生 `messages_for_model`，做临时治理后发给 provider。
5. 工具结果 append 到 `messages`，但 `messages_for_model` 的临时修补/压缩不反写 session。
6. runner 返回 `AgentRunResult.messages`。
7. loop 用 `_save_turn(session, all_msgs, skip)` 保存本轮新增消息，并移除 runtime context、多模态 data URL 等不该长期保存的内容。

### 2.2 pico 核心链路

pico 的主链路是：

```text
PicoLoop.ask(user_message)
  -> _restore_interrupted_turn_if_needed()
  -> prepare_session_for_turn()
  -> record_user_turn()
  -> start_task_run_for_turn()
  -> runner.run(PicoRunSpec)
  -> PicoRunner.ask()
  -> run_history_compaction_for_turn()
  -> govern_messages_for_model()
  -> build_model_prompt()
  -> model_client.complete_with_tools() or complete()
  -> parse response
  -> run_tool()
  -> ToolExecutor.execute()
  -> ToolRegistry / Tool.execute()
  -> record tool message
  -> next iteration
  -> final assistant message
```

各环节职责：

- `PicoLoop.ask()`  
  pico 当前的一轮总调度入口。它先恢复 checkpoint，再准备 session，再记录用户消息，再创建 task/run state，最后调用 runner。

- `PicoRunner.run()` / `PicoRunner.ask()`  
  pico 已定义 `PicoRunSpec` / `PicoRunResult`，主输入以 `initial_messages` 为准，runner 返回 `final_content`、`messages`、`tools_used`、`stop_reason`、`error`、`usage`、`metadata`。runner 主模型请求不再调用 `agent.build_model_prompt()`，而是使用 `messages_for_model` 局部变量和 provider-native messages adapter。仍保留少量 loop lifecycle adapter 方法，例如 record/checkpoint/trace/report 写入，这是 pico 同步产品层与 nanobot 纯 runner 的主要剩余差异。

- `PicoRunner.govern_messages_for_model()`  
  在每次 provider 请求前临时治理 `messages_for_model`：丢孤儿 tool result、补缺失 tool result、microcompact 旧工具结果、工具结果落盘/截断、`_snip_history()` 全局裁剪，并在裁剪后再次修复 tool message 合法性。

- `PicoLoop.build_context_messages_for_turn()` / `prompt_metadata_for_messages_for_runner()`  
  loop 在 runner 前构建 provider-native `initial_messages`，并把 runtime context、session summary、skills、history、当前用户消息交给 `ContextBuilder.build_messages()`。字符串 prompt 只作为旧 provider adapter 的渲染形式。

- `ContextBuilder.build()`  
  仅作为兼容 bridge，把 chat messages 转成字符串：

```text
SYSTEM:
...

Transcript:
USER:
...
TOOL:
...
```

- `ToolExecutor.execute()`  
  统一执行工具前后的产品语义：安全检查、审批、重复调用防护、snapshot diff、memory 更新、process note。

pico 的关键剩余差异是：runner 合同和主模型请求已经对齐，但同步产品层仍通过 adapter 提供 record/checkpoint/trace/report/tool execution 等副作用；nanobot 的 runner/loop 边界更纯，且有消息总线和 injection callback。

### 2.3 调用顺序对比

nanobot 正常用户消息处理顺序：

1. `_process_message()` 读取/创建 session。
2. 恢复 runtime checkpoint。
3. 恢复 pending user turn。
4. `auto_compact.prepare_session()`。
5. slash command dispatch。
6. `consolidator.maybe_consolidate_by_tokens()`。
7. 设置 message/spawn/cron/my 工具上下文。
8. `history = session.get_history(max_messages=0)`。
9. `initial_messages = context.build_messages(...)`。
10. 提前保存用户消息并标记 pending user turn。
11. `_run_agent_loop()` -> `AgentRunner.run()`。
12. runner 内部循环：临时上下文治理 -> provider -> tool -> checkpoint -> provider。
13. `_save_turn()` 保存新增消息。
14. 清 pending/runtime checkpoint。
15. 异步调度一次后台 token consolidation。

pico 当前正常用户消息处理顺序：

1. `PicoLoop.ask()` 恢复 runtime/pending checkpoint。
2. `prepare_session_for_turn()`，内部调用 auto compact。
3. `record_user_turn()`，写 user message、pending user turn、用户事实记忆。
4. `start_task_run_for_turn()`。
5. `run_history_compaction_for_turn()` 在 context build 前触发。
6. `build_run_spec()` 写入 `PicoRunSpec.initial_messages`。
7. `runner.run()`。
8. 每次模型调用前 `govern_messages_for_model()` 治理局部 `messages_for_model`。
9. `model_client.complete_messages_with_tools()`，旧 provider 退到 string adapter。
10. 工具调用 -> `ToolExecutor.execute()`。
11. 记录 tool message、checkpoint、trace。
12. final 后记录 assistant、清 checkpoint、写 report。
13. loop 在回合后再次触发一次 token consolidation。

---

## 3. context 上下文板块细节

### 3.1 nanobot 最终注入 prompt/messages 的内容

nanobot 最终给 provider 的是 chat messages，不是单个字符串。核心由 `ContextBuilder.build_messages()` 生成。

#### system message 内容

`build_system_prompt()` 按顺序拼接：

1. identity 模板 `templates/agent/identity.md`
   - runtime: 操作系统、架构、Python 版本
   - workspace 路径
   - long-term memory 路径：`{workspace}/memory/MEMORY.md`
   - history log 路径：`{workspace}/memory/history.jsonl`
   - custom skills 路径：`{workspace}/skills/{skill-name}/SKILL.md`
   - platform policy
   - channel format hint
   - 搜索建议：优先 `grep`/`glob`
   - untrusted content 片段
   - message 工具发送文件的硬性说明

2. workspace bootstrap files  
   从 workspace 根目录读取：
   - `AGENTS.md`
   - `SOUL.md`
   - `USER.md`
   - `TOOLS.md`

3. long-term memory  
   从 `memory/MEMORY.md` 读取。如果内容不是内置模板，就注入：

```text
# Memory

## Long-term Memory
...
```

4. always skills  
   `SkillsLoader.get_always_skills()` 返回的技能会被完整加载进 system prompt。

5. skills summary  
   其他可用 skills 以摘要形式通过 `agent/skills_section.md` 注入，提示模型需要时用 `read_file` 读取 `SKILL.md`。

6. recent history  
   从 `memory/history.jsonl` 读取 dream 尚未处理的 entries，即 cursor 大于 `.dream_cursor` 的历史摘要，最多 50 条，注入：

```text
# Recent History

- [timestamp] content
```

#### history 内容

`history = session.get_history(max_messages=0)`。

它只返回 `session.messages[session.last_consolidated:]` 之后的未归档消息，并做边界修复：

- 尽量从 user message 开始；
- 避免从孤立 tool result 开始；
- 只保留 provider 需要的字段：`role`、`content`、`tool_calls`、`tool_call_id`、`name`、`reasoning_content`。

#### current user message 内容

当前用户消息不会直接裸放，而是和 runtime context 合并成同一个 user message：

```text
[Runtime Context — metadata only, not instructions]
Current Time: ...
Channel: ...
Chat ID: ...

[Resumed Session]
Inactive for N minutes.
Previous conversation summary: ...
[/Runtime Context]

用户原始输入
```

如果带 media：

- `_process_message()` 先用 `extract_documents()` 从媒体中抽文档文本；
- `ContextBuilder._build_user_content()` 对图片生成 base64 `image_url` content blocks；
- runtime context 作为第一个 text block；
- session 落盘时 `_save_turn()` 会移除 runtime context，图片 data URL 会变成占位文本，避免持久化巨大载荷。

#### session summary 注入

`session_summary` 来自 `auto_compact.prepare_session()` 返回的 `pending`：

- 若某个 session 因 idle 被 auto compact 归档，摘要会存到 `_summaries` 或 session metadata `_last_summary`。
- 下一次处理该 session 时，`prepare_session()` 返回格式化摘要：

```text
Inactive for N minutes.
Previous conversation summary: ...
```

它只注入 runtime context 的 `[Resumed Session]` 区块，不作为独立 system instruction。

#### token consolidation 摘要注入

`Consolidator.maybe_consolidate_by_tokens()` 将旧 session messages 摘要写入 `memory/history.jsonl`，并在 session metadata 写 `_last_summary`。下一次 `auto_compact.prepare_session()` 会把它转成 runtime context 的 resumed summary。与此同时，`memory/history.jsonl` 中 dream 未处理的 entries 也会通过 system prompt 的 `# Recent History` 注入。

### 3.2 pico 最终注入 prompt/messages 的内容

pico 主链路现在给 runner 的是 provider-native chat messages。`PicoLoop.build_run_spec()` 设置 `PicoRunSpec.initial_messages = ContextBuilder.build_messages(...)`，runner 每轮基于局部 `messages_for_model` 调 provider。`ContextBuilder.build()` 仍存在，但只作为 string provider/旧测试兼容 adapter，不再是主链路入口。

兼容字符串 prompt 现在渲染为：

```text
SYSTEM:
{system_prompt}

Transcript:
{messages_to_prompt(messages[1:])}
```

#### system prompt 内容

pico 的 `build_system_prompt()` 顺序：

1. identity 模板 `pico/templates/agent/identity.md`
   - runtime
   - workspace 路径
   - long-term memory 路径：`{workspace}/.pico/memory/MEMORY.md`
   - history log 路径：`{workspace}/.pico/memory/history.jsonl`
   - custom skills 路径
   - platform policy
   - channel format hint
   - 搜索建议：优先 `grep`/`glob`/`list_dir`
   - untrusted content 片段

2. bootstrap files  
   pico 读取的是 `.pico` 目录下：
   - `.pico/AGENTS.md`
   - `.pico/SOUL.md`
   - `.pico/USER.md`
   - `.pico/TOOLS.md`

3. long-term memory  
   `.pico/memory/MEMORY.md` 如果不是模板内容，则注入。

4. always skills  
   `SkillsLoader` 会把 always skill 的 `SKILL.md` 正文完整注入 system prompt。

5. skills summary  
   其他可用 skills 以摘要形式注入，提示模型需要时用 `read_file` 读取对应 `SKILL.md`。workspace skills 优先于 builtin skills，支持 frontmatter requirements。

6. recent history  
   `.pico/memory/history.jsonl` 中 dream cursor 之后的 entries，最多 50 条。

#### Relevant memory 状态

pico 旧版额外有一层 in-session memory retrieval：

- 用户消息进入 `record_user_turn()` 后，会用 `remember_user_facts()` 抽取“我喜欢/我用/我的...”这类用户事实，写入 `session["memory"]["episodic_notes"]`。
- 工具读文件后会记录 recent files、file summaries、episodic notes。
- 旧 `ContextBuilder.build()` 曾根据当前 user message 调用 `agent.memory.retrieval_candidates(user_message, limit=3)`，把最多 3 条相关 note 放入 `Relevant memory`。

这层不是 nanobot 原版主链路设计，已从 pico 主 prompt 和兼容 string adapter 中移除。`retrieval_candidates()` / `retrieval_view()` 仍作为 memory API 保留，Dream、long-term memory 和 session memory 不删除；只是不会再把 `Relevant memory:` 段注入模型主上下文。

#### transcript 内容

pico 的 transcript 来自：

```python
history = agent.session_manager.live_history(agent.session, max_messages=0)
history, reductions = _reduce_history_for_prompt(history)
messages = build_messages(history, current_message=user_message, ...)
transcript = messages_to_prompt(messages[1:])
```

默认 `_reduce_history_for_prompt()` 只保留最近 6 条 history message，超过则在 metadata 的 `budget_reductions` 里记录：

```python
{"section": "history", "strategy": "keep_recent_messages", "dropped_messages": N}
```

注意：pico 在 `PicoLoop.ask()` 中先 `record_user_turn(user_message)`，因此当前 user message 已经在 session history 末尾。`build_messages()` 看到最后一条也是 user，会把 runtime context + 当前 user content 合并进最后一条 user message，而不是再 append 一条 user。

#### runtime context

pico runtime context 格式为：

```text
[Runtime Context - metadata only, not instructions]
Current Time: ...
Channel: cli
Chat ID: {session id}

[Resumed Session]
{session_summary}
[/Runtime Context]
```

`session_summary` 来自：

- `agent.archived_history_summary()`，优先 `_prepared_session_summary`
- 否则 `session["history_archive"]["latest_summary"]`

这里的 `runtime context` 容易和 pico 删除掉的旧 runtime 状态混淆。两者不是一回事：

- 已删除/迁移掉的是 session payload 里的旧 runtime 状态字段，例如 `runtime_identity`、`resume_state`、旧 `checkpoints` 等。这类字段属于持久运行状态。
- 这里的 runtime context 是每次组 prompt 时临时生成的“运行元信息文本块”，只告诉模型当前时间、渠道、chat id、恢复摘要等。更准确的名字应该是 `prompt runtime metadata block`。

所以“runtime 被删了”不代表 prompt 里不再有 runtime metadata。前者是持久状态字段，后者是临时提示词片段。

#### media 支持

pico `ContextBuilder._build_user_content()` 支持图片转 base64 `image_url` blocks，但当前 `PicoLoop.ask()`/`build_run_spec()` 没有把 media 参数纳入主链路。因此它是 context builder 层具备能力，但产品入口未完整打通。

### 3.3 context 板块差距

1. provider-native chat messages 主链路已对齐。  
   pico 仍保留字符串 prompt adapter 给旧 provider 使用；多模态产品入口和 thinking/reasoning blocks 的 provider 细节仍比 nanobot 简化。

2. SkillsLoader 主链路已对齐。  
   pico 已支持 workspace/builtin skills、frontmatter、requirements、always skills 完整注入和 skills summary；生态深度仍比 nanobot 简化。

3. nanobot bootstrap 在 workspace 根目录；pico bootstrap 在 `.pico/`。  
   这是有意迁移差异，不是纯缺陷。pico 的 MemoryStore 会把旧根目录文件迁移到 `.pico/`。

4. nanobot 会在保存 session 时清理 runtime context 和多模态 data URL；pico 已在 `_save_turn()` / `record()` 持久化路径清理 runtime context、图片 data URL 和过长 tool result，核心语义已对齐。

5. nanobot 的 context token 预算仍更精确：pico 已有 `_snip_history()`，会估算 `messages + tool definitions` 并保留 system 与最近合法窗口，但当前估算仍是轻量字符/token 近似，不是 provider tokenizer 级别。

---

## 4. nanobot 三套上下文清洗/压缩机制

用户重点疑问是：runner 的“五层上下文清洗”、`memory.py` 的会话压缩、`autocompact.py` 的会话压缩都作用于 session 上下文，它们何时触发、顺序如何、优先级如何。

先给结论：

- runner 五层清洗只作用于“本次模型调用前的临时 messages_for_model”，不反写 session。
- `memory.py` 的 `Consolidator.maybe_consolidate_by_tokens()` 作用于“持久 session 的 last_consolidated 游标”和 `memory/history.jsonl`，按 token 预算触发。它也会做摘要压缩，但通常不删除原始 `session.messages`，更像逻辑压缩 / 游标压缩。
- `autocompact.py` 的 `AutoCompact` 作用于“长时间 idle 的 session 文件”，会把 session 的旧消息摘要归档，只保留最近合法尾部。它会真实改写 `session.messages`，更像物理压缩 / 裁剪 session 活跃历史。
- 正常用户消息进来时，顺序是：恢复 checkpoint -> auto compact prepare -> slash command -> token consolidation -> build context -> runner 五层清洗。
- 回合结束后，还会后台调度一次 token consolidation。
- idle 时，在 `AgentLoop.run()` 的 1 秒 timeout 分支里，auto compact 会扫描过期 session 并后台归档。

先把三个对象分清楚：

```text
session.messages
  持久历史，磁盘里的长期会话记录。

history = session.get_history()
  从 session.messages 取出来准备组 prompt 的历史；会受 last_consolidated 影响。

messages_for_model
  runner 每次真正发给模型前的临时版本；只为这一次请求合法、短、完整。
```

三套机制的作用对象不同：

```text
AutoCompact
  改 session.messages，主要处理 idle 很久的会话。

Consolidator
  改 session.last_consolidated，并把旧消息摘要写进 memory/history.jsonl。

Runner 五层清洗
  不改 session，只改本次 messages_for_model。
```

### 4.1 runner 的“五层上下文清洗”

nanobot runner 中每次调用模型前都会从真实 `messages` 生成临时 `messages_for_model`：

```python
messages_for_model = self._drop_orphan_tool_results(messages)
messages_for_model = self._backfill_missing_tool_results(messages_for_model)
messages_for_model = self._microcompact(messages_for_model)
messages_for_model = self._apply_tool_result_budget(spec, messages_for_model)
messages_for_model = self._snip_history(spec, messages_for_model)
messages_for_model = self._drop_orphan_tool_results(messages_for_model)
messages_for_model = self._backfill_missing_tool_results(messages_for_model)
```

所谓“五层”通常指前五个主步骤；后两步是 `_snip_history()` 之后的二次修复。

#### 第 1 层：drop orphan tool results

函数：`_drop_orphan_tool_results()`

作用：删除没有对应 assistant `tool_calls` 声明的 tool result。

原因：OpenAI/Anthropic 等 provider 对 tool message 的上下文结构有要求。tool message 必须能关联到前面的 assistant tool call，否则 provider 可能拒绝请求。

触发：每次 runner iteration 调模型前。

是否持久化：不持久化，只影响 `messages_for_model`。

#### 第 2 层：backfill missing tool results

函数：`_backfill_missing_tool_results()`

作用：如果历史里存在 assistant 发起的 tool_call，但缺少对应 tool result，就插入一条合成 tool message：

```text
[Tool result unavailable — call was interrupted or lost]
```

原因：崩溃、取消、旧 session 数据不完整时，assistant tool call 后没有 tool result，会破坏 provider 消息协议。backfill 可以把“不完整上下文”修成“合法但带错误结果的上下文”。

触发：每次 runner iteration 调模型前；`_snip_history()` 后也会再做一次。

是否持久化：不持久化。

#### 第 3 层：microcompact

函数：`_microcompact()`

作用：对较早且较长的工具结果做轻量压缩。可压缩工具包括：

- `read_file`
- `exec`
- `grep`
- `glob`
- `web_search`
- `web_fetch`
- `list_dir`

规则：

- 保留最近 10 个 compactable tool result。
- 更早的 compactable tool result 如果 content 是字符串且长度 >= 500，则替换为：

```text
[tool_name result omitted from context]
```

触发：每次 runner iteration 调模型前。

是否持久化：不持久化。

例子：一次长任务里模型多次搜索和读文件，历史中出现 18 个较长工具结果：

```text
tool(grep): 12000 字
tool(read_file): 8000 字
tool(exec): 9000 字
...
最近 10 个 tool result
```

runner 默认只完整保留最近 10 个 compactable tool result。更早的长结果在本次 `messages_for_model` 里会变成：

```text
[grep result omitted from context]
```

这不代表真实 session 里的工具结果被删了，也不代表 `memory/history.jsonl` 被改了。它只是在“这一次发给模型的临时上下文”里告诉模型：这里曾经有一个 grep 结果，但为了省上下文省略了正文。

#### 第 4 层：apply tool result budget

函数：`_apply_tool_result_budget()`

作用：对历史里的每条 tool message 调 `_normalize_tool_result()`：

- 空结果补位；
- 超大结果写入 workspace 下的 tool-result 文件；
- 超过 `max_tool_result_chars` 的字符串截断。

触发：每次 runner iteration 调模型前。

是否持久化：不直接持久化。当前轮新产生的工具结果在 append 时也会 `_normalize_tool_result()`，那部分会进入 runner 返回的 messages，再由 loop 保存。

例子：某次 `read_file` 读取了一个 300KB 文件，而 `max_tool_result_chars=16000`。即使这是最近工具结果，不能被 microcompact 省略，也不能把 300KB 原样塞给模型。`_apply_tool_result_budget()` 会把它落盘或截断，模型看到类似：

```text
[Tool result too large; full content saved to ...]
```

所以它和 microcompact 的区别是：

```text
microcompact:
  针对“老工具结果太多”，省略较早结果。

apply_tool_result_budget:
  针对“单条工具结果自己太大”，落盘或截断。
```

#### 第 5 层：snip history

函数：`_snip_history()`

作用：如果整段 prompt 估算 token 超过预算，则保留：

- 全部 system messages；
- 最近的非 system messages；
- 尽量从合法 user 起点开始；
- 避免 provider 拒绝 system -> assistant 这种非法序列；
- 必要时宁可略超预算，也要保留最近 user 起点。

预算计算：

```text
budget = context_window_tokens - max_output_tokens - 1024 safety buffer
```

如果配置了 `context_block_limit`，则直接用它。

估算会考虑：

- messages
- tool definitions
- provider/model 的 token 估算函数

触发：每次 runner iteration 调模型前，且 `spec.context_window_tokens` 有值时。

是否持久化：不持久化。

为什么已经经过 AutoCompact 和 Consolidator 后还需要 `_snip_history()`？因为前两者是会话级的提前减负，`_snip_history()` 是 provider 请求级的最后兜底。总量仍可能超，常见原因包括：

- AutoCompact 只处理 idle 过期会话；用户连续聊天、TTL 未开或未过期时不会触发。
- Consolidator 只能压 session history 中“可安全归档的旧前缀”，不能随便压当前任务刚产生的 assistant/tool 片段。
- system prompt、bootstrap md、skills、tool definitions、当前用户输入也占 token，这些不是 Consolidator 的主要压缩对象。
- 本轮新产生的大工具结果是在 Consolidator 之后才出现的，只能由 runner 在下一次模型调用前治理。
- 后台 consolidation 可能还没跑完，runner 不能假设前面一定已经压好。

例子：

```text
模型窗口: 128k
system + md + skills: 25k
tool definitions: 12k
session history 压缩后: 20k
当前用户问题: 5k
刚刚 read_file 返回: 90k

合计: 152k，仍然超窗口。
```

这时即使旧 session history 已经压过，最终 provider request 仍会超，必须靠 runner 的 `microcompact`、`apply_tool_result_budget` 和 `_snip_history()` 做最后处理。

#### snip 后二次修复

`_snip_history()` 可能裁掉 assistant/tool 对的一部分，导致新的孤儿 tool result 或缺失 tool result。因此 runner 会再次调用：

1. `_drop_orphan_tool_results()`
2. `_backfill_missing_tool_results()`

这不是新的压缩层，而是对裁剪结果的协议修复。

### 4.2 memory.py 中的 session 压缩

文件：`nanobot/agent/memory.py`

核心类：`Consolidator`

核心方法：`maybe_consolidate_by_tokens(session, session_summary=None)`

它解决的问题不是“本次 provider 请求临时太长”，而是“持久 session 历史越来越长，需要把旧消息归档成摘要”。

触发位置有三个：

1. 普通用户消息 `_process_message()` 中，`auto_compact.prepare_session()` 之后、`build_messages()` 之前：

```python
await self.consolidator.maybe_consolidate_by_tokens(session, session_summary=pending)
```

2. system message 分支中同样会调用。

3. 回合结束保存 session 后，后台调度：

```python
self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
```

触发条件：

1. `session.messages` 非空。
2. `context_window_tokens > 0`。
3. 估算当前 session prompt tokens：
   - 用 `session.get_history(max_messages=0)` 拿未归档历史；
   - 调 `context.build_messages()` 加上 system prompt、runtime context、token probe；
   - 加上 tool definitions；
   - 用 provider/model token estimator 估算。
4. 如果 `estimated < budget`，不压缩。
5. 如果 `estimated >= budget`，开始压缩。

预算：

```text
budget = context_window_tokens - max_completion_tokens - 1024
target = budget // 2
```

压缩过程：

1. 从 `session.last_consolidated` 开始找可归档边界。
2. 边界必须尽量落在 user turn 上，避免切断对话结构。
3. 每轮最多归档 60 条消息。
4. 最多 5 轮。
5. 对 chunk 调 `archive()`。
6. `archive()` 用 LLM 调 `agent/consolidator_archive.md`，让模型提取关键事实：
   - user facts
   - decisions
   - solutions
   - events
   - preferences
7. 摘要写入 `memory/history.jsonl`。
8. `session.last_consolidated = end_idx`，并保存 session。
9. 最后一条 summary 写入 `session.metadata["_last_summary"]`，等待下一次 `AutoCompact.prepare_session()` 注入 runtime context。

注意：`Consolidator` 不删除 session.messages，它通过 `last_consolidated` 游标让 `session.get_history()` 跳过已归档消息。也就是说，原始消息还在 session 文件中，但默认 prompt history 不再带它们。

这里要特别注意 tool result 的位置：tool 结果当然也是 session history，通常长这样：

```python
{"role": "tool", "tool_call_id": "...", "name": "grep", "content": "..."}
```

但 `Consolidator` 能压的是 session history 中“可安全归档的旧前缀”，不是任意压所有 tool result。它会尽量按 user turn 边界切，避免把当前正在进行的 assistant/tool 片段切碎。比如：

```text
1 user: 帮我分析项目
2 assistant: 我要 grep
3 tool: grep 返回 80k
4 assistant: 我要 read_file
5 tool: read_file 返回 60k
```

这时还没有新的 user 边界，3 和 5 虽然是 tool result，也属于当前任务马上要用的证据，Consolidator 很难立刻把它们归档。它更适合压这种旧完整回合：

```text
1 user: 上周任务 A
2 assistant: ...
3 tool: ...
4 assistant final: ...
5 user: 今天任务 B
```

此时可以把 1-4 摘要掉，让今天任务 B 仍能看到最近上下文。

### 4.3 autocompact.py 中的 session 压缩

文件：`nanobot/agent/autocompact.py`

核心类：`AutoCompact`

它解决的问题是：session 长时间 idle 后，主动把未归档旧消息摘要掉，只保留最近合法尾部，降低下次唤醒成本。

#### idle 扫描触发

`AgentLoop.run()` 每次等入站消息 timeout 1 秒后，会调用：

```python
self.auto_compact.check_expired(
    self._schedule_background,
    active_session_keys=self._pending_queues.keys(),
)
```

`check_expired()` 遍历 `sessions.list_sessions()`：

- session key 为空则跳过；
- 正在 `_archiving` 则跳过；
- 当前有 active pending queue 则跳过；
- `_is_expired(updated_at)` 为真才归档；
- `_ttl <= 0` 时永不过期。

过期判断：

```text
now - session.updated_at >= session_ttl_minutes * 60
```

`updated_at` 来自 session 自身：追加消息、保存 session、压缩归档等操作都会更新它。AutoCompact 并不是靠模型判断“很久没说话”，而是靠这个时间戳和配置的 `session_ttl_minutes` 做机械判断。如果 TTL 没配置或小于等于 0，就不会因为空闲触发后台归档。

#### 归档过程

`_archive(key)`：

1. invalidate session cache。
2. 重新 load session。
3. `_split_unconsolidated(session)`。
4. `archive_msgs` 是可归档前缀。
5. `kept_msgs` 是最近合法尾部，默认 8 条左右，但会扩展到合法 user 起点。
6. 若有 `archive_msgs`，调用 `consolidator.archive(archive_msgs)` 生成摘要并写入 `memory/history.jsonl`。
7. 摘要放入内存 `_summaries[key]`，也写入 `session.metadata["_last_summary"]`。
8. `session.messages = kept_msgs`。
9. `session.last_consolidated = 0`。
10. 保存 session。

和 `Consolidator.maybe_consolidate_by_tokens()` 的区别：

- token consolidation 通过 `last_consolidated` 跳过旧消息，不删除原始 session messages。
- auto compact 会真的把 session.messages 缩成最近尾部，旧消息只剩 `memory/history.jsonl` 摘要。

可以这样记：

```text
Consolidator:
  看到 prompt token 可能超预算 -> 摘要旧前缀 -> 推进 last_consolidated。
  原始 session.messages 通常还在。

AutoCompact:
  看到 session idle 过久 -> 摘要旧前缀 -> session.messages 只保留最近合法尾部。
  活跃 session 历史真的变短。
```

#### prepare_session 触发

普通请求进来时，`_process_message()` 也会调用：

```python
session, pending = self.auto_compact.prepare_session(session, key)
```

`prepare_session()` 做两件事：

1. 如果当前 session 正在归档或已过期，就 reload session。
2. 如果有 `_summaries[key]` 或 `session.metadata["_last_summary"]`，返回格式化 summary：

```text
Inactive for N minutes.
Previous conversation summary: ...
```

这个 `pending` 会传给：

- `consolidator.maybe_consolidate_by_tokens(session, session_summary=pending)`
- `context.build_messages(..., session_summary=pending)`

最终出现在 runtime context 的 `[Resumed Session]` 中。

### 4.4 三者触发时机和优先级

普通用户消息的顺序：

```text
_process_message()
  1. restore runtime checkpoint
  2. restore pending user turn
  3. auto_compact.prepare_session()
  4. slash command dispatch
  5. consolidator.maybe_consolidate_by_tokens()
  6. context.build_messages()
  7. user message early persist
  8. AgentRunner.run()
       每次 provider 请求前：
       a. drop orphan tool results
       b. backfill missing tool results
       c. microcompact old tool results
       d. apply tool result budget
       e. snip history by token budget
       f. drop/backfill again after snip
  9. _save_turn()
  10. clear checkpoint
  11. background consolidator.maybe_consolidate_by_tokens()
```

idle 后台的顺序：

```text
AgentLoop.run() timeout
  -> auto_compact.check_expired()
  -> schedule AutoCompact._archive()
  -> consolidator.archive()
  -> memory/history.jsonl
  -> session.messages = recent legal suffix
```

优先级理解：

1. checkpoint 恢复优先级最高  
   因为它要先把中断的 assistant/tool 状态实体化进 session，否则后面压缩可能看不到真实上下文。

2. auto compact prepare 早于 token consolidation  
   因为 idle summary 要作为 `session_summary` 参与 token 估算和 prompt 注入。比如用户隔了 3 小时回来，AutoCompact 之前生成了“上一段对话摘要”，`prepare_session()` 需要先把这个摘要取出来，后面的 `consolidator.maybe_consolidate_by_tokens()` 才能把这段摘要算进 token，`context.build_messages()` 也才能把它放进 `[Resumed Session]`。

3. token consolidation 早于 context build  
   因为它会推进 `last_consolidated`，影响 `session.get_history()` 返回哪些消息。比如 session 有 100 条消息，consolidator 发现太长后把 1-70 摘要并设置 `last_consolidated=70`。随后 `session.get_history()` 只返回 71-100。如果先 build context，再 consolidation，就会先把 1-100 都塞进 prompt，压缩就晚了。

4. runner 五层清洗最后发生  
   因为它只处理本次 provider request 的临时视图，应该在 system prompt、history、current user message 都组装完成之后再做。

5. 回合结束后再后台 token consolidation  
   用于把本轮新增消息也纳入预算治理，但不阻塞用户响应。

---

## 5. pico 对应机制与差距

### 5.1 pico runner 的上下文治理

pico `PicoRunner.govern_messages_for_model()` 有类似 nanobot runner 的临时治理：

1. `drop_orphan_tool_results()`
2. `backfill_missing_tool_results()`
3. `microcompact()`
4. 对 tool message 调 `normalize_tool_result()`，大结果落盘/截断

差距：

- 已有 `_snip_history()` 请求级全局裁剪，但 token 估算仍是轻量近似，不如 nanobot 结合 provider/model/tool schema 的估算完整。
- 可压缩工具集合已改为 Nanobot 核心名：`read_file`、`exec`、`grep`、`glob`、`web_search`、`web_fetch`、`list_dir`。
- 治理链路已改为局部 `messages_for_model`，不再 monkey patch `agent.session_manager.live_history`。

这里的剩余差距不再是“有没有 `_snip_history()`”，而是估算精度和产品配置深度：pico 已估算“system + history + current user + tool definitions”的整体 token，并保留 system 与最近合法窗口；但还没有 nanobot 那种 provider tokenizer/模型配置级别的预算计算。

### 5.2 pico 的 token consolidation

pico 有 `pico/consolidator.py`，逻辑是 nanobot `memory.py::Consolidator` 的同步/简化版：

- 用字符数粗估 token。
- `CONTEXT_WINDOW_TOKENS = 1800`，`SAFETY_BUFFER_TOKENS = 256`。
- 从 `last_consolidated` 开始找 user 边界。
- 每轮最多 60 条、最多 5 轮。
- 通过 `model_client.complete(prompt, max_tokens)` 摘要。
- 摘要写入 `.pico/memory/history.jsonl`。
- 更新 `session["last_consolidated"]` 和 `session["history_archive"]`。

触发位置：

- `PicoLoop.ask()` 在 `build_run_spec()` 前调用 `run_history_compaction_for_turn()`。
- runner 返回后，`PicoLoop.ask()` 再同步触发一次 `run_history_compaction_for_turn()`。
- `compact_history_if_needed()` 调 `history_consolidator.maybe_consolidate_by_tokens()`。

差距：

- nanobot 在 context build 前和回合结束后都会触发；pico 现在也在 context build 前和 runner 返回后触发。
- nanobot 估算时会用 provider/tokenizer/tool definitions；pico 主要按 prompt 字符数粗估。
- nanobot Consolidator 是 async 并有 per-session asyncio lock；pico 用同步 RLock。

### 5.3 pico 的 AutoCompact

pico 有 `pico/auto_compact.py`，基本移植 nanobot idle compact：

- `_RECENT_SUFFIX_MESSAGES = 8`
- `_split_unconsolidated()` 保留最近合法尾部。
- `_archive_session()` 摘要旧消息，更新 `history_archive` 和 `_last_summary`。
- `prepare_session()` 如果 session 过期，会同步归档；然后返回 resumed summary。

触发位置：

- `PicoLoop.ask()` 开头调用 `prepare_session_for_turn()`。
- `prepare_session_for_turn()` 调 `auto_compactor.prepare_session()`。
- `PicoLoop` 初始化了 `CronService` 和 dream cron，但没有 nanobot 那种常驻 MessageBus loop 每秒 idle scan 的同等主循环逻辑。

差距：

- nanobot 的 auto compact 有 `AgentLoop.run()` 常驻 idle 扫描；pico 当前主要在 ask 前 prepare 时触发，`check_expired()` 虽存在但缺少同等常驻消息总线驱动。
- nanobot 能跳过 active session keys；pico 也有参数，但产品层活跃队列模型不完整。

### 5.4 pico context 注入状态

pico 主链路已经使用 provider-native messages：

- system prompt 是第一条 `system` message。
- history 以原生 role messages 传递。
- runtime context 与当前 user message 合并到同一条当前 role message。
- string prompt 只作为旧 provider adapter，不再包含 `Relevant memory` 补偿段。

剩余缺口：

- media 入口未完整接入。
- prompt budget 主要靠固定最近 6 条 history 和工具结果压缩，不如 nanobot 的 token-aware snip。

---

## 6. 简明对照表

| 模块 | nanobot | pico | pico 未达到处 |
|---|---|---|---|
| Runner 纯度 | `AgentRunner` 基本只依赖 `AgentRunSpec` | `PicoRunner` 主合同已是 `PicoRunSpec.initial_messages -> PicoRunResult` | 仍有同步产品层 adapter 副作用 |
| 模型输入 | provider-native chat messages | 主链路 provider-native messages，string adapter 兼容旧 provider | media 产品入口仍未完整打通 |
| 工具注册 | default + config + MCP + channel + web | standard registry + 部分 ecosystem | web/message/my/MCP 完整生态不足 |
| 文件读取 | 文本/图片/PDF/分页/二进制防护 | 文本/图片/分页/二进制防护/设备路径防护 | PDF 等高级能力仍可补 |
| 编辑工具 | `edit_file` 多级匹配和诊断 | `edit_file` 多级匹配和诊断，`patch_file` 为 alias | 缩进/复杂格式保持仍可增强 |
| 搜索工具 | `glob`/`grep` 丰富参数 | `glob`/`grep` 核心参数 | 类型过滤/上下文等高级参数较简化 |
| Shell | `exec` deny/sandbox/workspace/env | `exec` deny/workspace/env/审批/截断 | sandbox wrapper、内部状态写保护仍弱 |
| Context bootstrap | workspace 根目录 md | `.pico/` md | 属于设计迁移差异 |
| Skills | 完整 SkillsLoader 注入 | 已接入 workspace/builtin SkillsLoader、always skills、summary | requirements/生态深度较简化 |
| Session 压缩 | token consolidation + idle auto compact + runner 临时治理 | 三者都有简化移植，含 `_snip_history()` | token 估算、常驻 idle scan 不足 |
| 持久化清洗 | `_save_turn()` 清 runtime/media/tool 结果 | 已清 runtime/media/tool 结果 | 产品入口覆盖面仍较窄 |
| 中途注入 | pending queue + runner injection callback | 无同等级消息总线模型 | 多消息并发/插话能力不足 |

---

## 7. 建议补齐顺序

如果目标是让 pico 更接近 nanobot，建议按以下顺序补：

1. 若要继续提高 runner 纯度，把 record/checkpoint/trace/report 等同步产品层 adapter 再下沉到 loop 或 callback。
2. 用 provider/model tokenizer 替换 `_snip_history()` 当前轻量 token 估算。
3. 补齐 PDF、grep 上下文/类型过滤、exec sandbox wrapper、内部状态写保护等高级工具能力。
4. 若 pico 要做常驻 agent，再补 MessageBus/pending queue/injection/idle auto compact scan；如果只做 CLI 单轮工具，保留同步简化也可以。
5. 再考虑 web/message/my/完整 MCP 连接链路等产品工具生态。
