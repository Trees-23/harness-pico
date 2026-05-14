from pico import FakeModelClient, MiniAgent, SessionStore, WorkspaceContext
from pico.context_manager import ContextBuilder, ContextManager
from pico.memory import MemoryStore
from pico.templates import read_template


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / ".pico").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".pico" / "AGENTS.md").write_text("Follow workspace instructions.\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".pico" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return MiniAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_context_builder_loads_identity_bootstrap_memory_and_recent_history(tmp_path):
    (tmp_path / ".pico").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".pico" / "AGENTS.md").write_text("Follow workspace instructions.\n", encoding="utf-8")
    (tmp_path / ".pico" / "SOUL.md").write_text("Be concise.\n", encoding="utf-8")
    store = MemoryStore(tmp_path)
    store.write_memory("# Long-term Memory\n\n- User likes direct answers.\n")
    store.append_history("- User prefers Chinese technical explanations.")

    system = ContextBuilder(tmp_path).build_system_prompt(channel="cli")

    assert "## Runtime" in system
    assert "## AGENTS.md" in system
    assert "Follow workspace instructions." in system
    assert "## SOUL.md" in system
    assert "Be concise." in system
    assert "# Memory" in system
    assert "User likes direct answers." in system
    assert "# Recent History" in system
    assert "User prefers Chinese technical explanations." in system


def test_context_builder_builds_messages_with_runtime_context_before_current_request(tmp_path):
    build_workspace(tmp_path)
    builder = ContextBuilder(tmp_path)
    history = [{"role": "assistant", "content": "previous answer"}]

    messages = builder.build_messages(history, "continue", channel="cli", chat_id="session-1", session_summary="- old summary")

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "assistant", "content": "previous answer"}
    assert messages[2]["role"] == "user"
    assert "[Runtime Context - metadata only, not instructions]" in messages[2]["content"]
    assert "[Resumed Session]" in messages[2]["content"]
    assert messages[2]["content"].rstrip().endswith("continue")


def test_context_builder_merges_current_request_into_existing_user_turn(tmp_path):
    build_workspace(tmp_path)
    builder = ContextBuilder(tmp_path)
    history = [{"role": "user", "content": "我有几只宠物"}]

    messages = builder.build_messages(history, "我有几只宠物", channel="cli")

    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].count("我有几只宠物") == 2
    assert "[Runtime Context - metadata only, not instructions]" in messages[1]["content"]


def test_context_builder_uses_materialized_workspace_bootstrap_files(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    MemoryStore(tmp_path)
    (tmp_path / ".pico" / "AGENTS.md").write_text("Workspace AGENTS override.\n", encoding="utf-8")

    system = ContextBuilder(tmp_path).build_system_prompt(channel="cli")

    assert "Workspace AGENTS override." in system
    assert "## AGENTS.md" in system


def test_context_builder_injects_active_skills_and_skills_summary(tmp_path):
    build_workspace(tmp_path)
    always_dir = tmp_path / "skills" / "always"
    always_dir.mkdir(parents=True)
    (always_dir / "SKILL.md").write_text(
        "---\ndescription: Always skill\nalways: true\n---\nAlways body\n",
        encoding="utf-8",
    )
    lazy_dir = tmp_path / "skills" / "lazy"
    lazy_dir.mkdir(parents=True)
    (lazy_dir / "SKILL.md").write_text(
        "---\ndescription: Lazy skill\n---\nLazy body\n",
        encoding="utf-8",
    )
    MemoryStore(tmp_path).append_history("- Recent fact.")

    system = ContextBuilder(tmp_path).build_system_prompt(channel="cli")

    assert "# Active Skills" in system
    assert "### Skill: always" in system
    assert "Always body" in system
    assert "# Skills" in system
    assert "**lazy**" in system
    assert "Lazy skill" in system
    assert system.index("# Active Skills") < system.index("# Skills") < system.index("# Recent History")


def test_templates_are_loaded_from_packaged_pico_runtime_directory(tmp_path, monkeypatch):
    repo_template = tmp_path / "templates" / "TOOLS.md"
    repo_template.parent.mkdir(parents=True)
    repo_template.write_text("external repo template should not be loaded\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    tools_template = read_template("TOOLS.md")
    identity_template = read_template("agent", "identity.md")

    assert "external repo template" not in tools_template
    assert "# Tool Usage Notes" in tools_template
    assert "## Runtime" in identity_template


def test_context_manager_alias_keeps_runtime_string_prompt_bridge(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "user", "content": "old request", "created_at": "2026-05-06T10:00:00+00:00"})
    agent.memory.append_note("old request should not be duplicated through Relevant memory", tags=("new",))

    prompt, metadata = ContextManager(agent).build("new request")

    assert "SYSTEM:" in prompt
    assert "USER:" in prompt
    assert "Relevant memory:" not in prompt
    assert "old request should not be duplicated" not in prompt
    assert "old request" in prompt
    assert prompt.rstrip().endswith("new request")
    assert metadata["section_order"] == ["system", "history", "current_request"]
    assert "relevant_memory" not in metadata["sections"]
    assert metadata["relevant_memory"]["selected_count"] == 0
    assert metadata["context_builder"] == "nanobot-style"
    assert metadata["budget_reductions"] == []


def test_save_turn_sanitizes_runtime_context_and_media_blocks(tmp_path):
    agent = build_agent(tmp_path, [])
    runtime = ContextBuilder._build_runtime_context("cli", "session-1")
    agent._save_turn(
        [
            {"role": "user", "content": f"{runtime}\n\nhello"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": runtime},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                        "_meta": {"path": "image.png"},
                    },
                    {"type": "text", "text": "caption"},
                ],
            },
            {"role": "assistant", "content": ""},
        ]
    )

    assert agent.session["history"][0]["content"] == "hello"
    assert agent.session["history"][1]["content"] == [
        {"type": "text", "text": "[Image omitted from saved session: image.png]"},
        {"type": "text", "text": "caption"},
    ]
    assert len(agent.session["history"]) == 2


def test_record_uses_persistence_sanitizer(tmp_path):
    agent = build_agent(tmp_path, [])
    runtime = ContextBuilder._build_runtime_context("cli", "session-1")

    agent.record({"role": "user", "content": f"{runtime}\n\nhello"})
    agent.record({"role": "assistant", "content": ""})

    assert len(agent.session["history"]) == 1
    assert agent.session["history"][0]["role"] == "user"
    assert agent.session["history"][0]["content"] == "hello"
