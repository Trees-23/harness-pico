import asyncio

import pytest

import pico.tools as legacy_tools
from pico.tooling import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    ObjectSchema,
    StringSchema,
    Tool,
    ToolRegistry,
    build_standard_tool_registry,
    tool_parameters,
    tool_parameters_schema,
)
from pico import FakeModelClient, MiniAgent, SessionStore, WorkspaceContext


@tool_parameters(
    tool_parameters_schema(
        required=["path", "count"],
        path=StringSchema("path", min_length=1),
        count=IntegerSchema(description="count", minimum=1, maximum=5),
        enabled=BooleanSchema(description="enabled"),
        tags=ArraySchema(StringSchema("tag"), min_items=1),
        nested=ObjectSchema(
            {"level": IntegerSchema(minimum=0)},
            required=["level"],
        ),
    )
)
class DemoTool(Tool):
    @property
    def name(self):
        return "demo"

    @property
    def description(self):
        return "Demo tool."

    @property
    def read_only(self):
        return True

    async def execute(self, **kwargs):
        return kwargs


@tool_parameters(tool_parameters_schema(required=["value"], value=StringSchema("value")))
class NamedTool(Tool):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return f"{self._name} tool"

    async def execute(self, **kwargs):
        return f"{self._name}:{kwargs['value']}"


@tool_parameters(tool_parameters_schema())
class FailingTool(Tool):
    @property
    def name(self):
        return "fail"

    @property
    def description(self):
        return "Failing tool."

    async def execute(self, **kwargs):
        raise RuntimeError("boom")


def test_tool_parameters_returns_deep_copied_schema():
    tool = DemoTool()

    first = tool.parameters
    first["properties"]["path"]["minLength"] = 99
    second = tool.parameters

    assert second["properties"]["path"]["minLength"] == 1
    assert tool.read_only is True
    assert tool.exclusive is False
    assert tool.concurrency_safe is True


def test_registry_prepare_call_casts_recursively_and_validates():
    registry = ToolRegistry()
    registry.register(DemoTool())

    tool, params, error = registry.prepare_call(
        "demo",
        {
            "path": 123,
            "count": "3",
            "enabled": "true",
            "tags": [1, "two"],
            "nested": {"level": "2"},
        },
    )

    assert error is None
    assert tool is not None
    assert params == {
        "path": "123",
        "count": 3,
        "enabled": True,
        "tags": ["1", "two"],
        "nested": {"level": 2},
    }


def test_registry_prepare_call_formats_unknown_and_invalid_errors():
    registry = ToolRegistry()
    registry.register(DemoTool())

    _, _, unknown = registry.prepare_call("missing", {})
    _, _, not_object = registry.prepare_call("demo", [])
    _, params, invalid = registry.prepare_call("demo", {"path": "", "count": "9", "tags": []})

    assert "Tool 'missing' not found" in unknown
    assert "parameters must be a JSON object" in not_object
    assert "Invalid parameters for tool 'demo'" in invalid
    assert "path must be at least 1 chars" in invalid
    assert "count must be <= 5" in invalid
    assert "tags must have at least 1 items" in invalid
    assert params["count"] == 9


def test_registry_definitions_are_stably_sorted_with_mcp_last():
    registry = ToolRegistry()
    registry.register(NamedTool("zeta"))
    registry.register(NamedTool("mcp_remote_alpha"))
    registry.register(NamedTool("alpha"))

    names = [item["function"]["name"] for item in registry.get_definitions()]

    assert names == ["alpha", "zeta", "mcp_remote_alpha"]

    registry.register(NamedTool("beta"))
    names = [item["function"]["name"] for item in registry.get_definitions()]
    assert names == ["alpha", "beta", "zeta", "mcp_remote_alpha"]


def test_registry_execute_returns_tool_result_or_model_consumable_error():
    registry = ToolRegistry()
    registry.register(NamedTool("echo"))
    registry.register(FailingTool())

    assert asyncio.run(registry.execute("echo", {"value": "ok"})) == "echo:ok"
    failure = asyncio.run(registry.execute("fail", {}))
    invalid = asyncio.run(registry.execute("echo", {"value": 3}))

    assert "Error executing fail: boom" in failure
    assert "[Analyze the error above" in failure
    assert invalid == "echo:3"


def build_agent(tmp_path, *, depth=0, max_depth=1):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    return MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=SessionStore(tmp_path / ".pico" / "sessions"),
        approval_policy="auto",
        depth=depth,
        max_depth=max_depth,
    )


def test_standard_pico_tool_registry_maps_required_tool_attributes(tmp_path):
    registry = build_standard_tool_registry(build_agent(tmp_path))

    expected = {
        "list_dir": (True, False, True),
        "read_file": (True, False, True),
        "grep": (True, False, True),
        "glob": (True, False, True),
        "exec": (False, True, False),
        "write_file": (False, False, False),
        "edit_file": (False, False, False),
        "delegate": (True, False, False),
        "spawn": (True, False, False),
        "cron": (False, False, False),
        "notebook_edit": (False, False, False),
    }

    assert set(registry.tool_names) == set(expected)
    for name, attributes in expected.items():
        tool = registry.get(name)
        assert tool is not None
        assert (tool.read_only, tool.exclusive, tool.concurrency_safe) == attributes


def test_standard_pico_tool_registry_definitions_are_stable_and_include_schemas(tmp_path):
    registry = build_standard_tool_registry(build_agent(tmp_path))

    names = [item["function"]["name"] for item in registry.get_definitions()]
    read_file_schema = registry.get("read_file").parameters
    exec_schema = registry.get("exec").parameters

    assert names == [
        "cron",
        "delegate",
        "edit_file",
        "exec",
        "glob",
        "grep",
        "list_dir",
        "notebook_edit",
        "read_file",
        "spawn",
        "write_file",
    ]
    assert read_file_schema["required"] == ["path"]
    assert read_file_schema["properties"]["start"]["type"] == "integer"
    assert exec_schema["properties"]["timeout"]["maximum"] == 600


def test_standard_tool_registry_includes_nanobot_core_names(tmp_path):
    registry = build_standard_tool_registry(build_agent(tmp_path))
    names = {item["function"]["name"] for item in registry.get_definitions()}

    assert {"list_dir", "grep", "glob", "exec", "edit_file"}.issubset(names)
    assert registry.get("exec").exclusive is True
    assert registry.get("edit_file").concurrency_safe is False
    assert registry.get("grep").read_only is True


def test_standard_pico_tool_registry_prepare_call_casts_text_protocol_args(tmp_path):
    registry = build_standard_tool_registry(build_agent(tmp_path))

    _, read_params, read_error = registry.prepare_call(
        "read_file",
        {"path": "README.md", "start": "1", "end": "2"},
    )
    _, shell_params, shell_error = registry.prepare_call(
        "exec",
        {"command": "echo hi", "timeout": "20"},
    )
    _, invalid_params, invalid_error = registry.prepare_call(
        "read_file",
        {"path": "README.md", "start": "3", "end": "1"},
    )

    assert read_error is None
    assert read_params["start"] == 1
    assert read_params["end"] == 2
    assert shell_error is None
    assert shell_params["timeout"] == 20
    assert "Invalid parameters for tool 'read_file'" in invalid_error
    assert "end must be >= start" in invalid_error
    assert invalid_params["start"] == 3


def test_read_file_supports_text_images_and_rejects_binary(tmp_path):
    agent = build_agent(tmp_path)
    (tmp_path / "binary.bin").write_bytes(b"\xff\x00\xff")
    (tmp_path / "image.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    )

    text = agent.run_tool("read_file", {"path": "README.md", "start": 1, "end": 1})
    image = agent.run_tool("read_file", {"path": "image.png", "start": 1, "end": 1})
    binary = agent.run_tool("read_file", {"path": "binary.bin", "start": 1, "end": 1})

    assert "# README.md" in text
    assert isinstance(image, list)
    assert image[0]["type"] == "image_url"
    assert image[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "cannot read binary file" in binary


def test_edit_file_supports_trim_and_quote_normalization(tmp_path):
    agent = build_agent(tmp_path)
    (tmp_path / "notes.txt").write_text("alpha\n  beta  \nquote: “hello”\n", encoding="utf-8")

    trim_result = agent.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": "beta", "new_text": "gamma"},
    )
    quote_result = agent.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": 'quote: "hello"', "new_text": "quote: ok"},
    )

    content = (tmp_path / "notes.txt").read_text(encoding="utf-8")
    assert trim_result == "patched notes.txt"
    assert quote_result == "patched notes.txt"
    assert "gamma" in content
    assert "quote: ok" in content


def test_grep_and_glob_nanobot_aliases_support_core_options(tmp_path):
    agent = build_agent(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("needle\nneedle\n", encoding="utf-8")
    (tmp_path / "src" / "b.txt").write_text("needle\n", encoding="utf-8")

    globbed = agent.run_tool("glob", {"path": "src", "pattern": "*.py"})
    counted = agent.run_tool("grep", {"path": "src", "pattern": "needle", "output_mode": "count"})

    assert globbed == "src/a.py"
    assert "a.py" in counted


def test_exec_rejects_dangerous_commands_and_allows_longer_timeout(tmp_path):
    agent = build_agent(tmp_path)

    rejected = agent.run_tool("exec", {"command": "rm -rf /", "timeout": 20})
    _, params, error = agent.tool_registry.prepare_call("exec", {"command": "echo ok", "timeout": "600"})

    assert "command rejected by safety policy" in rejected
    assert error is None
    assert params["timeout"] == 600


def test_standard_pico_tool_registry_hides_delegate_when_depth_is_exhausted(tmp_path):
    registry = build_standard_tool_registry(build_agent(tmp_path, depth=1, max_depth=1))

    assert "delegate" not in registry
    assert "delegate" not in [item["function"]["name"] for item in registry.get_definitions()]


def test_runtime_legacy_prompt_specs_are_derived_from_standard_registry(tmp_path):
    agent = build_agent(tmp_path)

    assert set(agent.tools) == set(agent.tool_registry.tool_names)
    assert agent.tools["exec"]["risky"] is True
    assert agent.tools["read_file"]["risky"] is False
    assert agent.tools["read_file"]["schema"] == agent.tool_registry.get("read_file").parameters


def test_legacy_tools_registry_is_deprecated_shim_derived_from_standard_registry(tmp_path):
    agent = build_agent(tmp_path)

    with pytest.warns(DeprecationWarning, match="legacy spec/validator APIs are deprecated"):
        tools = legacy_tools.build_tool_registry(agent)

    assert set(tools) == set(agent.tool_registry.tool_names)
    assert tools["read_file"]["schema"] == agent.tool_registry.get("read_file").parameters
    assert tools["exec"]["risky"] is True
    assert tools["read_file"]["risky"] is False
    assert tools["read_file"]["run"]({"path": "README.md", "start": "1", "end": "1"}).startswith("# README.md")


def test_legacy_validate_tool_is_deprecated_shim_for_prepare_call(tmp_path):
    agent = build_agent(tmp_path)

    with pytest.warns(DeprecationWarning, match="legacy spec/validator APIs are deprecated"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            calls = []
            original = agent.tool_registry.prepare_call

            def spy_prepare_call(name, args):
                calls.append((name, args))
                return original(name, args)

            monkeypatch.setattr(agent.tool_registry, "prepare_call", spy_prepare_call)
            legacy_tools.validate_tool(agent, "read_file", {"path": "README.md", "start": "1", "end": "1"})

    assert calls == [("read_file", {"path": "README.md", "start": "1", "end": "1"})]


def test_legacy_tool_names_are_aliases_not_model_visible_definitions(tmp_path):
    agent = build_agent(tmp_path)
    registry = build_standard_tool_registry(agent)
    definition_names = {item["function"]["name"] for item in registry.get_definitions()}

    assert {"list_files", "search", "run_shell", "patch_file"}.isdisjoint(definition_names)
    assert registry.prepare_call("list_files", {"path": "."})[0].name == "list_dir"
    assert registry.prepare_call("search", {"pattern": "demo", "path": "."})[0].name == "grep"
    assert registry.prepare_call("run_shell", {"command": "echo ok", "timeout": 20})[0].name == "exec"

    (tmp_path / "legacy.txt").write_text("old\n", encoding="utf-8")
    result = agent.run_tool("patch_file", {"path": "legacy.txt", "old_text": "old", "new_text": "new"})
    assert result == "patched legacy.txt"
