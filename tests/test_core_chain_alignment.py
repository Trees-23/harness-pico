from pico import FakeModelClient, MiniAgent, ModelResponse, SessionStore, ToolCallRequest, WorkspaceContext
from pico.runner import PicoRunResult


def build_agent(tmp_path, outputs=None):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    return MiniAgent(
        model_client=FakeModelClient(outputs or []),
        workspace=workspace,
        session_store=SessionStore(tmp_path / ".pico" / "sessions"),
        approval_policy="auto",
    )


def test_core_chain_spec_context_tool_and_result_contracts_are_nanobot_style(tmp_path):
    skill_dir = tmp_path / "skills" / "always"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: Always skill\nalways: true\n---\nUse precise file tools.\n",
        encoding="utf-8",
    )
    agent = build_agent(
        tmp_path,
        [
            ModelResponse(
                "",
                [ToolCallRequest("call_1", "read_file", {"path": "README.md", "start": 1, "end": 1})],
            ),
            "<final>Done.</final>",
        ],
    )
    agent.memory.append_note("legacy relevant memory should not be injected", tags=("inspect",))

    spec = agent.build_run_spec("Inspect README")
    tool_definition_names = {item["function"]["name"] for item in spec.tool_registry.get_definitions()}

    assert spec.initial_messages
    assert spec.messages == []
    assert spec.initial_messages[0]["role"] == "system"
    assert "# Active Skills" in spec.initial_messages[0]["content"]
    assert "Use precise file tools." in spec.initial_messages[0]["content"]
    assert spec.initial_messages[-1]["role"] == "user"
    assert "[Runtime Context - metadata only, not instructions]" in spec.initial_messages[-1]["content"]
    assert "Relevant memory:" not in str(spec.initial_messages)
    assert {"list_dir", "grep", "glob", "exec", "edit_file"}.issubset(tool_definition_names)
    assert {"list_files", "search", "run_shell", "patch_file"}.isdisjoint(tool_definition_names)

    result = agent.runner.run(spec)

    assert isinstance(result, PicoRunResult)
    assert result.final_content == "Done."
    assert result.stop_reason == "completed"
    assert agent.model_client.messages
    assert agent.model_client.messages[0][0]["role"] == "system"
    assert "Relevant memory:" not in str(agent.model_client.messages[0])

    tool_message = next(message for message in result.messages if message["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["name"] == "read_file"
    assert "# README.md" in tool_message["content"]
    assert result.messages[-1]["role"] == "assistant"
    assert result.messages[-1]["content"] == "Done."
    assert result.tools_used == ["read_file"]


def test_core_chain_legacy_tool_names_are_execution_aliases_only(tmp_path):
    agent = build_agent(tmp_path)
    registry = agent.tool_registry
    definition_names = {item["function"]["name"] for item in registry.get_definitions()}

    assert {"list_files", "search", "run_shell", "patch_file"}.isdisjoint(definition_names)
    assert registry.prepare_call("list_files", {"path": "."})[0].name == "list_dir"
    assert registry.prepare_call("search", {"pattern": "demo", "path": "."})[0].name == "grep"
    assert registry.prepare_call("run_shell", {"command": "echo ok", "timeout": 20})[0].name == "exec"

    (tmp_path / "alias.txt").write_text("old\n", encoding="utf-8")
    assert agent.run_tool("patch_file", {"path": "alias.txt", "old_text": "old", "new_text": "new"}) == "patched alias.txt"
