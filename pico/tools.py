"""Deprecated compatibility shims for Pico tools."""

import warnings
from functools import partial

from .tooling.capabilities import (
    tool_delegate,
    tool_list_files,
    tool_patch_file,
    tool_read_file,
    tool_run_shell,
    tool_search,
    tool_write_file,
)

_DEPRECATED_COMPAT_API = (
    "pico.tools legacy spec/validator APIs are deprecated; use "
    "pico.tooling.ToolRegistry and ToolRegistry.prepare_call() instead."
)

TOOL_EXAMPLES = {
    "list_dir": '<tool>{"name":"list_dir","args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "grep": '<tool>{"name":"grep","args":{"pattern":"binary_search","path":"."}}</tool>',
    "glob": '<tool>{"name":"glob","args":{"pattern":"**/*.py","path":"."}}</tool>',
    "exec": '<tool>{"name":"exec","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "edit_file": '<tool name="edit_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
}


def build_tool_registry(agent):
    warnings.warn(_DEPRECATED_COMPAT_API, DeprecationWarning, stacklevel=2)
    registry = build_standard_tool_registry(agent)
    tools = {}
    for definition in registry.get_definitions():
        function = definition.get("function", {})
        name = function.get("name")
        if not name:
            continue
        tool = registry.get(name)
        if tool is None:
            continue
        tools[name] = {
            "schema": function.get("parameters", {}),
            "risky": not tool.read_only,
            "description": function.get("description", ""),
            "run": partial(agent.run_tool, name),
        }
    return tools


def build_standard_tool_registry(agent):
    from .tooling.pico_tools import build_standard_tool_registry as _build_standard_tool_registry

    return _build_standard_tool_registry(agent)


def tool_example(name):
    return TOOL_EXAMPLES.get(name, "")


def validate_tool(agent, name, args):
    warnings.warn(_DEPRECATED_COMPAT_API, DeprecationWarning, stacklevel=2)
    registry = getattr(agent, "tool_registry", None) or build_standard_tool_registry(agent)
    _, _, error = registry.prepare_call(name, args or {})
    if error:
        if name == "delegate" and "not found" in error and agent.depth >= agent.max_depth:
            raise ValueError("delegate depth exceeded")
        raise ValueError(error)


