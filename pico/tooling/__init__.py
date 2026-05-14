"""Pico tool foundation layer.

This package is the stage-1 Tool/Schema/Registry base. It intentionally lives
under ``pico.tooling`` while legacy ``pico.tools`` remains a module; once the
legacy module is reduced to a shim, these definitions can move to the final
``pico.tools`` package without changing the contracts.
"""

from .base import Schema, Tool, tool_parameters
from .executor import ToolExecutor
from .ecosystem import CronTool, MCPToolWrapper, NotebookEditTool, SpawnTool, normalize_mcp_schema, register_mcp_tool
from .locks import FileLockManager
from .pico_tools import build_standard_tool_registry
from .registry import ToolRegistry
from .schema import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    NumberSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)

__all__ = [
    "ArraySchema",
    "BooleanSchema",
    "CronTool",
    "IntegerSchema",
    "MCPToolWrapper",
    "NotebookEditTool",
    "NumberSchema",
    "ObjectSchema",
    "Schema",
    "StringSchema",
    "SpawnTool",
    "Tool",
    "ToolExecutor",
    "ToolRegistry",
    "FileLockManager",
    "build_standard_tool_registry",
    "normalize_mcp_schema",
    "register_mcp_tool",
    "tool_parameters",
    "tool_parameters_schema",
]
