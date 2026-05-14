"""Tool registry and prepare-call gate for Pico."""

from __future__ import annotations

from typing import Any

from .base import Tool

_RETRY_HINT = "\n\n[Analyze the error above and try a different approach.]"


class ToolRegistry:
    """Registry for standard Tool objects."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._aliases: dict[str, str] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def register_alias(self, alias: str, target: str) -> None:
        self._aliases[str(alias)] = str(target)

    def unregister(self, name: str) -> None:
        key = str(name)
        self._tools.pop(key, None)
        self._aliases.pop(key, None)
        self._cached_definitions = None

    def resolve_name(self, name: str) -> str:
        return self._aliases.get(str(name), str(name))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(self.resolve_name(name))

    def has(self, name: str) -> bool:
        return self.resolve_name(name) in self._tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        function = schema.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        if self._cached_definitions is not None:
            return self._cached_definitions
        definitions = [tool.to_schema() for tool in self._tools.values()]
        builtins: list[dict[str, Any]] = []
        dynamic: list[dict[str, Any]] = []
        for schema in definitions:
            name = self._schema_name(schema)
            if name.startswith("mcp_"):
                dynamic.append(schema)
            else:
                builtins.append(schema)
        builtins.sort(key=self._schema_name)
        dynamic.sort(key=self._schema_name)
        self._cached_definitions = builtins + dynamic
        return self._cached_definitions

    def prepare_call(self, name: str, params: dict[str, Any]) -> tuple[Tool | None, dict[str, Any], str | None]:
        requested_name = str(name)
        tool = self._tools.get(self.resolve_name(requested_name))
        if not tool:
            available = ", ".join(self.tool_names)
            return None, params, f"Error: Tool '{name}' not found. Available: {available}"
        if not isinstance(params, dict):
            return tool, params, (
                f"Error: Tool '{name}' parameters must be a JSON object, got {type(params).__name__}."
            )
        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return tool, cast_params, f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
        return tool, cast_params, None

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        tool, prepared_params, error = self.prepare_call(name, params)
        if error:
            return error + _RETRY_HINT
        try:
            assert tool is not None
            result = await tool.execute(**prepared_params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _RETRY_HINT
            return result
        except Exception as exc:
            return f"Error executing {name}: {exc}" + _RETRY_HINT

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return self.has(str(name))
