# Agent Harness 项目面试拷问 100 题

定位：面试官会刻意追问“你到底做了什么、为什么这么做、边界在哪里、失败怎么办”。下面每题都给出参考回答，回答时要结合源码机制，不要只背概念。

## 一、项目总览与架构

### 1. 你说搭建本地代码 Agent Harness，Harness 到底是什么，不就是调了个 LLM API 吗？

答：不是单纯调 API。Harness 是把模型接入、上下文构建、工具注册、工具调用、权限控制、执行状态、记忆系统和审计日志串成一个可运行闭环。LLM API 只负责推理，Harness 负责把用户输入变成结构化上下文，把工具能力以 schema 暴露给模型，执行模型返回的 tool calls，并把结果、状态和历史可恢复地管理起来。

### 2. 这套链路从用户输入到最终回答怎么走？

答：用户消息进入消息队列或 CLI 入口后，先进入会话管理层，加载短期 session history；随后上下文构建器组装 system、memory、skills、runtime metadata 和当前消息；runner 调用 provider，并把工具 JSON Schema 作为 tools 参数传入；模型返回 tool_calls 后，工具注册表做参数校验和调度；工具结果写回 tool 消息，再进入下一轮模型调用；直到模型输出最终文本，最后记录审计日志、保存状态并返回用户。

### 3. 你这个项目解决的核心问题是什么？

答：主要是四类问题：多轮任务导致上下文膨胀，工具调用参数和时机不可控，文件和 shell 工具权限失控，长任务失败后无法复现或恢复。对应做了上下文治理、工具 schema 校验和执行调度、路径隔离/沙箱/命令黑名单、checkpoint 与 JSONL 审计。

### 4. 你怎么证明这是 Agent 系统，不是聊天机器人？

答：核心区别是具备可执行工具闭环。模型不只是输出文本，而是可以基于 schema 调用 read/write/edit/exec/search 等工具；runner 会执行工具并把结果回填给模型，多轮迭代完成任务。同时有权限治理、状态恢复和结果审计，这些是代码 Agent 必须具备的运行层能力。

### 5. 这个系统里最关键的抽象是什么？

答：我认为是三类抽象：Tool 抽象统一工具名称、描述、参数 schema 和 execute；Provider 抽象统一不同模型接口返回的 content/tool_calls/finish_reason；Session/Memory 抽象区分短期会话、归档摘要和长期记忆。runner 把这三类抽象串起来。

### 6. 如果让我看一条主链路，你会让我看哪些模块？

答：入口看 loop/session manager，prompt 拼装看 context manager，模型-工具循环看 runner，工具注册和参数校验看 tooling/tools registry，具体权限看 filesystem/shell/security/sandbox，记忆治理看 consolidator/auto_compact/dream，审计看 run_store/metrics/checkpoint。

### 7. 你在项目里最大的工程取舍是什么？

答：没有追求“全部上下文永远完整塞给模型”，而是把上下文分为热数据、归档摘要和长期记忆。这样牺牲部分原文上下文，但换来 token 成本稳定、请求可控和长任务可持续运行。

### 8. 本地代码 Agent 和云端通用 Agent 的权限模型差异是什么？

答：本地代码 Agent 的工具直接触达用户文件系统和 shell，风险更高。权限不能只靠提示词，需要在工具执行层做硬约束，比如路径 resolve 后校验、环境变量白名单、命令黑名单、沙箱挂载和结果审计。

### 9. 如果模型胡乱调用不存在的工具怎么办？

答：工具注册表会检查工具名是否存在，不存在直接返回错误结果给模型，不会进入任何执行逻辑。错误会作为 tool result 或执行事件记录，模型可以基于错误自我修正。

### 10. 如果模型调用工具参数类型错了怎么办？

答：执行前会按 JSON Schema 做 cast 和 validate。比如字符串数字可安全转 int，布尔字符串可转 bool；必填缺失、类型不符、enum 越界、范围越界都会返回参数错误，不执行工具。

## 二、上下文构建与治理

### 11. 你说冷热数据分层，热数据和冷数据分别是什么？

答：热数据是当前 session 的近期消息、当前用户输入和本轮工具结果，要求强时效、强关联；冷数据是归档到 history.jsonl 的摘要、长期 MEMORY/SOUL/USER 文件和技能说明，要求长期复用但不一定逐字保留。

### 12. 为什么要冷热分层，不直接全量历史塞模型？

答：全量历史会快速超过上下文窗口，且大量旧工具结果对当前任务价值低。冷热分层可以保留近期细节，同时把旧内容压缩成摘要或长期事实，降低 token 开销和模型注意力干扰。

### 13. ContextBuilder 通常会拼哪些东西？

答：一般包括系统身份、工作区规则、长期记忆、技能摘要、recent history、session history、runtime metadata、当前用户消息和可选媒体内容。工具 schema 不是靠文本拼进去，而是通过 provider 请求的 tools 参数传入。

### 14. 你说工具 schema 不在 prompt 里，那 LLM 怎么知道工具？

答：runner 调用 provider 时传入 tools 参数，里面是多个 JSON Schema function definition，包括工具 name、description、parameters。模型的 tool calling 能力会直接读取这些结构化定义。

### 15. Recent History 和 session history 有什么区别？

答：session history 是当前会话里的原始 user/assistant/tool 消息；Recent History 是从归档历史 history.jsonl 中取出的摘要记录，通常用于跨会话或长期背景补充。两者来源和粒度不同。

### 16. token-probe 是什么？

答：Consolidator 在真正构建本轮 prompt 前，用一个占位 current_message，比如 `[token-probe]`，调用 ContextBuilder 组装一次近似完整 prompt，再估算 token。它不是发给模型推理，只是估算预算。

### 17. 为什么要在真正 build_messages 前做 token-probe？

答：如果等真正请求模型时才发现超预算，就只能失败或临时强裁。提前 probe 可以先把旧 session 消息归档，降低真实请求超限概率。

### 18. 你说阈值动态压缩，阈值是什么？

答：主要是上下文窗口预算。预算大致等于 `context_window_tokens - max_output_tokens - safety_buffer`。如果估算 prompt 超过预算，Consolidator 会归档旧消息；runner 内还有单条工具结果字符阈值和历史窗口裁剪。

### 19. 你说五层上下文治理流水线，五层是什么？

答：典型包括：清理孤儿 tool result、补齐缺失 tool result、microcompact 旧工具结果、对超大工具结果落盘/截断、按上下文预算 snip history。最后还会再次清理/补齐，保证消息序列合法。

### 20. 什么是孤儿 tool result？

答：role=tool 的消息必须对应前面 assistant 发起的 tool_call_id。如果历史裁剪或中断导致 tool result 前面没有对应 tool call，这条 tool result 就是孤儿，provider 可能拒绝请求，需要删除。

### 21. 为什么要补齐缺失 tool result？

答：如果 assistant 历史里有 tool_call，但对应 tool result 丢了，OpenAI/Anthropic 这类接口会认为消息序列不完整。补一条合成错误 tool result，可以让历史结构合法，并让模型知道上次工具被中断。

### 22. microcompact 压缩什么？

答：它只处理较早的、可压缩工具结果，比如 read_file、exec、grep、web_fetch 等。通常保留最近若干条工具结果，旧的大结果替换成占位文本，例如 `[read_file result omitted from context]`。

### 23. microcompact 是摘要吗？

答：不是智能摘要。它是上下文级别的占位替换，用非常短的文本告诉模型旧工具结果被省略了，目的是降低 token，而不是保留语义细节。

### 24. 单条工具结果过大怎么处理？

答：超过阈值时，把完整结果写入工作区的工具结果目录，再把 tool message 中的 content 替换成引用信息，包括保存路径、原始长度和 preview。这样模型能看到摘要和路径，必要时可再 read_file。

### 25. 这里的 16000 是 token 吗？

答：不是，是字符数。token 预算用于整个 prompt 的估算，工具结果预算通常用字符数控制单条内容大小。

### 26. snip_history 做什么？

答：它在整个 prompt 超预算时，保留 system 消息和最近的合法 user/assistant/tool 交互，从旧消息开始裁掉，并尽量从合法 user 边界开始，避免 provider 拒绝非法消息序列。

### 27. snip_history 会改 session 吗？

答：不会。它处理的是本次发给模型的临时 messages_for_model，不直接修改持久 session.messages。

### 28. 为什么既有 Consolidator 又有 snip_history？

答：Consolidator 是提前把旧消息摘要进长期归档，降低信息损失；snip_history 是请求前兜底裁剪，保证本次请求一定尽量落在上下文窗口内。一个偏长期治理，一个偏本次请求安全。

### 29. 为什么 runner 内每轮都要做上下文治理？

答：工具循环中每轮都会新增 assistant tool_call 和 tool result，工具结果可能很大，也可能因中断产生结构洞。因此每轮模型请求前都要重新治理临时上下文。

### 30. 你怎么估算 token？

答：优先使用 provider 的 token counter，如果没有则用 tiktoken 或字符近似。估算内容包括 messages 和 tools schema，因为工具 schema 也占上下文。

### 31. 如果 token 估算不准怎么办？

答：保留 safety buffer，比如 1024 token；同时 runner 有最终 snip_history 兜底，provider 报 length 时也有续写/恢复策略。估算不追求绝对精确，而是工程上保守。

### 32. 为什么要保留 system 消息？

答：system 消息包含身份、规则、工具使用约束和长期上下文入口，是模型行为的高优先级控制信息。裁剪时优先保留 system，避免模型失去运行约束。

### 33. 为什么裁剪时要找 user 边界？

答：很多 provider 要求对话序列以 user 开始或保持 tool_call/tool_result 配对。直接从中间 assistant/tool 截断会造成非法历史，所以要找到合法起点。

### 34. runtime metadata 为什么要标为非指令？

答：Current Time、Channel、Chat ID 这类信息只是运行环境，不应覆盖系统指令。显式标注为 metadata only 可以减少 prompt injection 风险。

### 35. 如果用户上传图片或文件，怎么进上下文？

答：ContextBuilder 会把图片编码成多模态 block，文档类内容先抽取文本。持久化 session 时会把 base64 图片替换成 placeholder，避免会话文件爆炸。

## 三、检查点与恢复

### 36. 你说 checkpoint，具体记录哪些阶段？

答：主要记录 awaiting_tools、tools_completed、final_response。awaiting_tools 表示 assistant 已经发起工具调用；tools_completed 表示工具结果已回来；final_response 表示最终回答已生成。

### 37. checkpoint 解决什么问题？

答：解决工具执行中断、进程被杀、用户 stop 后历史不完整的问题。恢复时可以把已经发生的 assistant tool_call 和 tool results 写回 session，避免下一轮上下文断裂。

### 38. 如果工具调用发起了但结果没回来，恢复时怎么办？

答：恢复逻辑会为 pending tool calls 补一条合成 tool result，内容类似“任务在工具完成前中断”。这样 provider 看到的是完整 tool_call/tool_result 结构。

### 39. checkpoint 是不是每一步都存？

答：不是所有细粒度操作都存，而是在关键状态边界存。这样控制写入成本，同时覆盖最容易导致历史不一致的阶段。

### 40. checkpoint 和审计日志有什么区别？

答：checkpoint 是为了恢复运行状态，强调当前任务可继续；审计日志是为了事后追踪，记录模型/工具交互、耗时、错误、指标等，强调可复盘。

### 41. 原子写状态快照为什么重要？

答：状态文件如果写到一半进程崩溃，会导致 JSON 损坏。原子写通常先写临时文件，再 replace，保证读到的要么是旧完整状态，要么是新完整状态。

### 42. 如果恢复时 checkpoint 和 session 已有历史重复怎么办？

答：恢复逻辑应做 overlap/dedup，根据消息 role、content、tool_call_id、tool_calls 等字段判断尾部重叠，避免重复追加同一条 assistant/tool 消息。

### 43. 用户中途发新消息怎么办？

答：同一 session 已有活跃任务时，新消息进入 pending queue，通过 injection callback 注入当前 runner，而不是开第二个并发任务抢同一段 session history。

### 44. 为什么同一 session 要串行？

答：同一会话共享 history、checkpoint、tool context。如果并发处理两个用户消息，容易出现历史乱序、工具结果错配、状态覆盖。串行锁可以避免这类竞争。

### 45. 不同 session 能并发吗？

答：可以，不同 session 之间状态隔离。系统还可以用全局 semaphore 限制最大并发请求数，避免资源耗尽。

## 四、长效记忆管理

### 46. 三层隔离存储具体是哪三层？

答：短期会话层保存当前 session 的原始对话和工具结果；归档摘要层用 history.jsonl 保存压缩后的对话摘要；长期记忆文件层用 MEMORY.md、SOUL.md、USER.md 保存长期事实、人设和用户偏好。

### 47. 为什么要把人设、用户偏好、事实依据分开？

答：它们更新频率和语义不同。人设相对稳定，用户偏好需要跨会话复用，事实依据需要可增删和过期治理。分开存储能降低互相污染，也便于 Dream 定向编辑。

### 48. Consolidator 做什么？

答：在活跃会话 prompt 超预算时，把 session 中尚未归档的旧消息按 user-turn 边界切块，调用模型总结后写入 history.jsonl，并推进 last_consolidated，避免重复归档。

### 49. Consolidator 会删除 session.messages 吗？

答：通常不会。它只是归档摘要并推进 last_consolidated。真实 session 原文仍保留，短期连续对话还可以使用。

### 50. 如果 Consolidator 不删消息，怎么降低 token？

答：它本身不直接让 get_history 变短，而是先为旧内容提供摘要备份；真正本次请求降 token 由 runner 的 snip_history 完成。Consolidator 降低的是后续裁剪时的信息损失。

### 51. AutoCompact 做什么？

答：AutoCompact 是空闲会话瘦身器。session 空闲超过 TTL 后，它把尚未归档的旧尾巴摘要进 history.jsonl，只保留最近合法后缀，并真正修改 session.messages。

### 52. AutoCompact 会重复压缩 Consolidator 已处理内容吗？

答：不会。AutoCompact 切分时从 `session.messages[session.last_consolidated:]` 开始，只处理未归档尾巴。之前已 consolidated 的部分不会重复摘要。

### 53. AutoCompact 默认多久触发？

答：默认 session_ttl_minutes 为 0，表示关闭。配置大于 0 后，主循环空闲时会检查 session 是否超过该空闲时间。

### 54. AutoCompact 是异步还是同步？

答：扫描是同步发生在主循环空闲 timeout 分支里；真正归档 `_archive()` 是通过后台任务异步调度执行。

### 55. AutoCompact prepare_session 做什么？

答：它不做归档，只处理归档后的结果：必要时重新加载 session，取出 `_last_summary`，作为 session_summary 注入下一轮 runtime context。

### 56. Dream 做什么？

答：Dream 是定时长期记忆维护任务。它读取 history.jsonl 中 Dream 尚未处理的摘要记录，结合当前 MEMORY/SOUL/USER 文件，先分析哪些记忆需要更新，再通过工具化编辑修改长期记忆或技能。

### 57. Dream 默认多久跑一次？

答：默认每 2 小时一次，通过 cron system job 注册。也可以用配置覆盖成 cron 表达式。

### 58. Dream 是不是跑两次？

答：不是两个独立任务，而是一次 Dream.run 内部有两个 phase。Phase 1 不带工具做分析，Phase 2 启动 runner 带文件工具做修改。

### 59. Dream 怎么知道哪些历史没处理？

答：history.jsonl 每条有 cursor，Dream 维护 `.dream_cursor`。每次只读取 cursor 大于 last_dream_cursor 的记录，处理后推进 cursor。

### 60. Recent History 为什么最多取 50 条？

答：这是给 prompt 的近期待处理归档摘要窗口，避免 history.jsonl 全量进入 system prompt。它只取 Dream 未处理 entries 的最近 50 条。

### 61. Dream 每次处理多少条？

答：默认 max_batch_size 是 20。也就是说 prompt 里 recent history 可能最多看 50 条，但 Dream 一次真正消化通常取前 20 条未处理记录。

### 62. Dream 怎么判断 MEMORY.md 哪些内容过期？

答：它通过 git line age 给 MEMORY.md 每行标注距离上次修改的天数。超过阈值，比如 14 天，会在 Phase 1 prompt 里加 `← Nd`，让模型判断是否陈旧。

### 63. 过期标注会自动删除记忆吗？

答：不会。标注只是信号，最终是否更新、合并、删除由 Dream Phase 2 根据分析结果通过 edit_file 等工具执行。

### 64. 为什么不用向量数据库做记忆？

答：本地代码 Agent 的记忆规模和可解释性需求更适合文件化存储。Markdown/JSONL 易审计、易版本控制、易被工具编辑。向量检索适合大规模语义召回，但不是这个项目的必要前提。

### 65. history.jsonl 会无限增长吗？

答：有 compact_history 机制限制最大 entries，默认保留一定数量的最近记录。另外 Dream 推进 cursor 后，旧记录可被压缩或裁剪，避免无限膨胀。

### 66. 长期记忆被 LLM 乱改怎么办？

答：Dream Phase 2 只开放受限文件工具，写入范围受 allowed_dir 控制；edit_file 要求精确 old_text，失败会返回 diff 提示；还可以通过 gitstore 或审计记录追踪修改。

### 67. AutoCompact 之后用户问“刚才说什么”怎么办？

答：AutoCompact 会保留最近合法尾巴，并把被归档部分的摘要作为 session_summary 注入下一轮 runtime context。模型能看到摘要而不是完全失忆。

### 68. 三层记忆之间会不会重复？

答：会有一定重叠，但语义不同。session 是原始短期事实，history.jsonl 是压缩摘要，MEMORY/SOUL/USER 是长期稳定知识。重复可以接受，关键是通过 cursor 和归档边界避免重复处理。

### 69. 如何避免 Dream 反复处理同一段历史？

答：依靠 `.dream_cursor`。处理完 batch 后推进到 batch 最后一条 cursor，下次只取更大的 cursor。

### 70. 为什么 Dream Phase 2 要用工具，而不是直接让模型输出完整新文件？

答：工具化编辑可以局部修改，减少覆盖风险；read_file/edit_file 能校验当前内容，old_text 不匹配会失败并提示，避免模型凭空重写整个 memory。

## 五、工具安全与运行治理

### 71. 工具安全第一层是什么？

答：注册层。没注册的工具不会出现在 tools schema 里，模型正常看不到也调用不到。比如 exec/web/MCP/my 都可以通过配置决定是否注册或注册哪些能力。

### 72. 第二层是什么？

答：参数 schema 层。工具执行前通过 ToolRegistry 做名称解析、参数类型转换和 JSON Schema 校验，不合格参数直接返回错误，不进入 execute。

### 73. read_only、exclusive 是权限吗？

答：严格说不是权限审批，而是运行治理。read_only 表示副作用较低，可参与并发；exclusive 表示即使并发开启也要独占运行，比如 exec，避免副作用冲突。

### 74. 文件工具包括哪些？

答：read_file、write_file、edit_file、list_dir、grep、glob、notebook_edit 都属于文件或文件搜索工具。其中 write/edit 是写工具，但仍走文件路径解析和 allowed_dir 限制。

### 75. 写文件和编辑文件进沙箱吗？

答：不进。它们是 Python 文件操作，不启动外部 shell。主要依靠 `_resolve_path()` 做路径隔离。沙箱主要用于 exec 这种任意命令执行工具。

### 76. 如果不开 restrict_to_workspace，文件工具安全吗？

答：安全边界会弱很多。allowed_dir 为空时，文件工具可能访问 workspace 外路径。因此生产或面试回答里要强调开启 restrict_to_workspace 或 sandbox 场景下路径才被强约束。

### 77. 路径隔离怎么防止 `../`？

答：不是只看字符串，而是先 expanduser/resolve 得到真实路径，再判断是否 relative_to allowed_dir。这样 `../` 和大部分符号链接逃逸都会被 resolve 后拦截。

### 78. exec 为什么风险最高？

答：exec 接受自由文本命令，可以读写文件、启动进程、联网、删除数据、读取环境变量。它比结构化文件工具的能力面大得多，所以需要命令 guard、环境隔离、路径检查和沙箱。

### 79. 命令黑名单在哪里体现？

答：ExecTool 的 deny_patterns。它匹配 rm -rf、dd、mkfs、shutdown、fork bomb、写内部 history/cursor 文件等危险模式，命中直接拒绝。

### 80. 命令白名单是什么？

答：allow_patterns。如果配置了 allowlist，则命令必须匹配其中至少一个模式，否则拒绝。它适合生产环境只允许有限命令集。

### 81. 除了命令黑白名单，还有什么白名单？

答：路径 allowed_dir 是目录白名单；allowed_env_keys 是环境变量白名单；MCP enabled_tools 是远端工具白名单；SSRF whitelist 是网络 CIDR 白名单。

### 82. 环境变量为什么要白名单？

答：子进程如果继承全量环境，可能泄露 API key、token、secret。默认只传 HOME/LANG/TERM 等最小变量，需要的变量必须显式 allowed_env_keys。

### 83. SSRF 防护做在哪里？

答：WebFetch 会校验 URL scheme、hostname 解析结果和 redirect 后地址，阻断 private/internal IP；ExecTool 也会扫描命令文本里的 URL，发现内部地址则拒绝。

### 84. WebSearch 也做 SSRF 吗？

答：WebFetch 的 SSRF 更完整。WebSearch 主要调用配置的搜索 provider；如果自定义 SearXNG base_url，应额外注意配置可信，因为它不是用户每次任意 fetch 的同一条校验路径。

### 85. MCP 工具有什么风险？

答：MCP 是远端能力注入，如果 enabled_tools 放太宽，模型能调用过多外部工具。风险包括数据外泄、不可控副作用和 schema 不可信。需要显式配置 server 和工具 allowlist。

### 86. my 自省工具怎么防止越权？

答：它有 BLOCKED、READ_ONLY、DENIED_ATTRS、SENSITIVE_NAMES 等限制，阻止访问核心对象、敏感字段、Python 魔术属性和反射逃逸路径。set 还可通过 allow_set 关闭。

### 87. 沙箱一般做什么？

答：限制进程看到的文件系统、可写目录、环境变量、进程空间和临时目录。更强的还会限制网络、CPU、内存和系统调用。

### 88. 项目里的沙箱怎么做？

答：exec 可选 bwrap。它用 bubblewrap 新建隔离环境，把系统目录只读挂载，把 workspace 读写挂载，把 media 只读挂载，给新的 /tmp，并用 tmpfs 遮住 workspace 父目录。

### 89. workspace 在沙箱里是复制的吗？

答：不是复制，是 bind mount。workspace 在沙箱内读写会直接作用到宿主机真实 workspace 文件。

### 90. tmpfs 遮住 workspace 父目录是什么意思？

答：把 workspace.parent 在沙箱视角替换成一个空的内存文件系统，隐藏 parent 下的 config、token 等文件；然后再单独把真实 workspace 挂回来。

### 91. 这个沙箱是否完全禁网？

答：不一定。代码里主要是文件系统隔离，并保留 resolv.conf 等运行能力。网络风险仍要依赖 URL guard、WebFetch SSRF 和部署层网络策略。

### 92. 为什么工具错误不直接抛异常？

答：很多工具错误是模型可恢复的，比如参数错、路径不存在、old_text 不匹配。把错误作为 tool result 回填，模型可以尝试修正。只有特定模式如子代理 fail_on_tool_error 才作为致命错误。

### 93. 工具结果为什么要落盘？

答：大结果直接进上下文会爆 token。落盘保留完整结果，同时给模型一个路径和 preview，必要时可再读取局部内容。

### 94. 落盘文件会永久保留吗？

答：不是永久。工具结果目录懒清理，按 bucket 过期时间和最大 bucket 数控制，比如清理 7 天前的旧 bucket，并限制最多保留一定数量。

### 95. 为什么 exec 要 exclusive？

答：shell 命令可能修改文件、启动进程或消耗资源，并发执行容易互相影响。exclusive 可以避免和其他工具并发造成不可预测副作用。

## 六、审计与可观测性

### 96. JSONL 事件流记录什么？

答：每轮模型请求、工具调用、参数摘要、工具结果状态、错误、耗时、token 使用、最终状态等。JSONL 方便流式追加、增量读取和故障复盘。

### 97. 为什么不用普通日志代替审计？

答：普通日志偏人读，结构不稳定；JSONL 审计是机器可解析事件流，可以按 run/session/tool 维度统计指标、复盘链路和生成报告。

### 98. 最终报告沉淀哪些关键指标？

答：可以包括总轮次、工具调用次数、成功/失败工具、token 输入输出、耗时、是否触发上下文压缩、是否恢复 checkpoint、最终状态和错误原因。

### 99. 怎么定位工具误调用？

答：看审计里的模型输出 tool_calls、工具 schema、参数、prepare_call 校验结果和工具事件。如果工具不该出现，查注册层；如果参数错，查 schema；如果越权，查工具实现权限。

### 100. 如果让我质疑你这项目只是包装已有框架，你怎么反驳？

答：我会强调自己做的是 Agent runtime 的关键工程层：上下文预算治理、工具 schema 网关、权限边界、checkpoint 恢复、长期记忆分层和审计复盘。这些不是简单接 API，而是让本地代码 Agent 能长期、安全、可追踪地执行复杂任务的基础设施。

