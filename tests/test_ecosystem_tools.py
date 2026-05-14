import json
from types import SimpleNamespace

from pico import FakeModelClient, MiniAgent, SessionStore, WorkspaceContext
from pico.tooling import normalize_mcp_schema, register_mcp_tool


def build_agent(tmp_path, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return MiniAgent(
        model_client=FakeModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".pico" / "sessions"),
        approval_policy=kwargs.pop("approval_policy", "auto"),
        **kwargs,
    )


def test_ecosystem_tools_are_registered_with_required_attributes(tmp_path):
    agent = build_agent(tmp_path)

    cron = agent.tool_registry.get("cron")
    spawn = agent.tool_registry.get("spawn")
    notebook = agent.tool_registry.get("notebook_edit")

    assert (cron.read_only, cron.exclusive, cron.concurrency_safe) == (False, False, False)
    assert (spawn.read_only, spawn.exclusive, spawn.concurrency_safe) == (True, False, False)
    assert (notebook.read_only, notebook.exclusive, notebook.concurrency_safe) == (False, False, False)


def test_spawn_is_hidden_when_delegate_depth_is_exhausted(tmp_path):
    agent = build_agent(tmp_path, depth=1, max_depth=1)

    assert "delegate" not in agent.tool_registry
    assert "spawn" not in agent.tool_registry


def test_cron_tool_goes_through_executor_approval(tmp_path):
    agent = build_agent(tmp_path, approval_policy="never")

    result = agent.run_tool("cron", {"action": "add", "message": "later", "every_seconds": 60})

    assert result == "error: approval denied for cron"
    assert agent._last_tool_result_metadata["tool_status"] == "rejected"
    assert agent._last_tool_result_metadata["risk_level"] == "high"


def test_notebook_edit_updates_workspace_via_executor(tmp_path):
    agent = build_agent(tmp_path)

    result = agent.run_tool(
        "notebook_edit",
        {
            "path": "demo.ipynb",
            "cell_index": 0,
            "new_source": "print('hi')",
            "edit_mode": "insert",
            "cell_type": "code",
        },
    )

    notebook = json.loads((tmp_path / "demo.ipynb").read_text(encoding="utf-8"))
    assert result == "notebook created demo.ipynb"
    assert notebook["cells"][0]["source"] == "print('hi')"
    assert agent._last_tool_result_metadata["workspace_changed"] is True


def test_mcp_schema_normalize_and_dynamic_registration_are_standard_tool_path(tmp_path):
    agent = build_agent(tmp_path)
    tool_def = SimpleNamespace(
        name="lookup",
        description="Lookup",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["query"],
        },
    )

    class Session:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(text=f"{name}:{arguments['query']}")])

    normalized = normalize_mcp_schema(tool_def.inputSchema)
    wrapper = register_mcp_tool(agent.tool_registry, Session(), "remote", tool_def, read_only=True)
    result = agent.run_tool("mcp_remote_lookup", {"query": 123})
    names = [item["function"]["name"] for item in agent.tool_registry.get_definitions()]

    assert normalized["properties"]["query"]["nullable"] is True
    assert wrapper.read_only is True
    assert result == "lookup:123"
    assert names[-1] == "mcp_remote_lookup"
