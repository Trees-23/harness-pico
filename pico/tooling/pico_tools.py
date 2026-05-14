"""Standard Tool adapters for Pico capabilities."""

from __future__ import annotations

from typing import Any

from .base import Tool, tool_parameters
from . import capabilities
from .ecosystem import CronTool, NotebookEditTool, SpawnTool
from .registry import ToolRegistry
from .schema import IntegerSchema, StringSchema, tool_parameters_schema


class PicoTool(Tool):
    """Adapter base for standard Pico tool capabilities."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def concurrency_safe(self) -> bool:
        return self.read_only and not self.exclusive

    def _execute_legacy(self, args: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def execute(self, **kwargs: Any) -> Any:
        return self._execute_legacy(kwargs)


@tool_parameters(
    tool_parameters_schema(
        path={"type": "string", "description": "Directory path relative to the workspace.", "default": "."},
    )
)
class ListDirTool(PicoTool):
    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List files in the workspace."

    @property
    def read_only(self) -> bool:
        return True

    def _execute_legacy(self, args: dict[str, Any]) -> Any:
        return capabilities.tool_list_files(self.agent, args)


@tool_parameters(
    tool_parameters_schema(
        required=["path"],
        path=StringSchema("UTF-8 file path relative to the workspace.", min_length=1),
        start={"type": "integer", "description": "1-based first line to read.", "minimum": 1, "default": 1},
        end={"type": "integer", "description": "1-based last line to read.", "minimum": 1, "default": 200},
    )
)
class ReadFileTool(PicoTool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a UTF-8 file by line range."

    @property
    def read_only(self) -> bool:
        return True

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        errors = super().validate_params(params)
        if not errors and "start" in params and "end" in params and params["end"] < params["start"]:
            errors.append("end must be >= start")
        return errors

    def _execute_legacy(self, args: dict[str, Any]) -> Any:
        return capabilities.tool_read_file(self.agent, args)


@tool_parameters(
    tool_parameters_schema(
        required=["pattern"],
        pattern=StringSchema("Search pattern.", min_length=1),
        path={"type": "string", "description": "File or directory path relative to the workspace.", "default": "."},
    )
)
class GrepTool(PicoTool):
    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "Search the workspace with rg or a simple fallback."

    @property
    def read_only(self) -> bool:
        return True

    def _execute_legacy(self, args: dict[str, Any]) -> Any:
        return capabilities.tool_search(self.agent, args)


@tool_parameters(
    tool_parameters_schema(
        pattern={"type": "string", "description": "Glob pattern.", "default": "**/*"},
        path={"type": "string", "description": "Directory path relative to the workspace.", "default": "."},
    )
)
class GlobTool(ListDirTool):
    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return "List matching files in the workspace."

    def _execute_legacy(self, args: dict[str, Any]) -> Any:
        return capabilities.tool_glob(self.agent, args)


@tool_parameters(
    tool_parameters_schema(
        required=["command"],
        command=StringSchema("Shell command to run in the workspace.", min_length=1),
        timeout={"type": "integer", "description": "Timeout in seconds.", "minimum": 1, "maximum": 600, "default": 20},
    )
)
class ExecTool(PicoTool):
    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Run a shell command in the repo root."

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def concurrency_safe(self) -> bool:
        return False

    def _execute_legacy(self, args: dict[str, Any]) -> Any:
        return capabilities.tool_run_shell(self.agent, args)


@tool_parameters(
    tool_parameters_schema(
        required=["path", "content"],
        path=StringSchema("File path relative to the workspace.", min_length=1),
        content=StringSchema("Text content to write."),
    )
)
class WriteFileTool(PicoTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write a text file."

    @property
    def concurrency_safe(self) -> bool:
        return False

    def _execute_legacy(self, args: dict[str, Any]) -> Any:
        return capabilities.tool_write_file(self.agent, args)


@tool_parameters(
    tool_parameters_schema(
        required=["path", "old_text", "new_text"],
        path=StringSchema("File path relative to the workspace.", min_length=1),
        old_text=StringSchema("Exact text block to replace.", min_length=1),
        new_text=StringSchema("Replacement text."),
    )
)
class EditFileTool(PicoTool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Replace one exact text block in a file."

    @property
    def concurrency_safe(self) -> bool:
        return False

    def _execute_legacy(self, args: dict[str, Any]) -> Any:
        return capabilities.tool_patch_file(self.agent, args)


PatchFileTool = EditFileTool


@tool_parameters(
    tool_parameters_schema(
        required=["task"],
        task=StringSchema("Bounded investigation task for the child agent.", min_length=1),
        max_steps=IntegerSchema(description="Maximum child-agent steps.", minimum=1, maximum=20),
    )
)
class DelegateTool(PicoTool):
    @property
    def name(self) -> str:
        return "delegate"

    @property
    def description(self) -> str:
        return "Ask a bounded read-only child agent to investigate."

    @property
    def read_only(self) -> bool:
        return True

    @property
    def concurrency_safe(self) -> bool:
        return False

    def _execute_legacy(self, args: dict[str, Any]) -> Any:
        return capabilities.tool_delegate(self.agent, args)


BASE_TOOL_CLASSES: tuple[type[PicoTool], ...] = (
    ListDirTool,
    ReadFileTool,
    GrepTool,
    GlobTool,
    ExecTool,
    WriteFileTool,
    EditFileTool,
)

LEGACY_TOOL_ALIASES: dict[str, str] = {
    "list_files": "list_dir",
    "search": "grep",
    "run_shell": "exec",
    "patch_file": "edit_file",
}


def build_standard_tool_registry(agent: Any) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_class in BASE_TOOL_CLASSES:
        registry.register(tool_class(agent))
    for alias, target in LEGACY_TOOL_ALIASES.items():
        registry.register_alias(alias, target)
    if agent.depth < agent.max_depth:
        registry.register(DelegateTool(agent))
        registry.register(SpawnTool(agent))
    registry.register(CronTool(agent))
    registry.register(NotebookEditTool(agent))
    return registry
