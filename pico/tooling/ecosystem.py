"""Nanobot-aligned ecosystem tools for Pico."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..cron import CronSchedule, now_ms_from_datetime
from .base import Tool, tool_parameters
from .registry import ToolRegistry
from .schema import BooleanSchema, IntegerSchema, StringSchema, tool_parameters_schema


def normalize_mcp_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}
    normalized = dict(schema)
    raw_type = normalized.get("type")
    if isinstance(raw_type, list) and "null" in raw_type:
        non_null = [item for item in raw_type if item != "null"]
        if len(non_null) == 1:
            normalized["type"] = non_null[0]
            normalized["nullable"] = True
    for key in ("oneOf", "anyOf"):
        options = normalized.get(key)
        if not isinstance(options, list):
            continue
        non_null = [item for item in options if isinstance(item, dict) and item.get("type") != "null"]
        has_null = any(isinstance(item, dict) and item.get("type") == "null" for item in options)
        if has_null and len(non_null) == 1:
            merged = {item_key: item_value for item_key, item_value in normalized.items() if item_key != key}
            merged.update(non_null[0])
            merged["nullable"] = True
            normalized = merged
            break
    if isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {
            key: normalize_mcp_schema(value) if isinstance(value, dict) else value
            for key, value in normalized["properties"].items()
        }
    if isinstance(normalized.get("items"), dict):
        normalized["items"] = normalize_mcp_schema(normalized["items"])
    if normalized.get("type") == "object":
        normalized.setdefault("properties", {})
        normalized.setdefault("required", [])
    return normalized


class MCPToolWrapper(Tool):
    def __init__(
        self,
        session: Any,
        server_name: str,
        tool_def: Any,
        *,
        read_only: bool = False,
        tool_timeout: int = 30,
    ) -> None:
        self._session = session
        self._original_name = str(getattr(tool_def, "name", "")).strip()
        self._name = f"mcp_{server_name}_{self._original_name}"
        self._description = str(getattr(tool_def, "description", "") or self._original_name)
        self._parameters = normalize_mcp_schema(getattr(tool_def, "inputSchema", None))
        self._read_only = bool(read_only)
        self._tool_timeout = int(tool_timeout)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._parameters)

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def concurrency_safe(self) -> bool:
        return self.read_only and not self.exclusive

    async def execute(self, **kwargs: Any) -> str:
        result = await self._session.call_tool(self._original_name, arguments=kwargs)
        blocks = getattr(result, "content", [])
        if not blocks:
            return "(no output)"
        parts = []
        for block in blocks:
            text = getattr(block, "text", None)
            parts.append(text if isinstance(text, str) else str(block))
        return "\n".join(parts) or "(no output)"


def register_mcp_tool(
    registry: ToolRegistry,
    session: Any,
    server_name: str,
    tool_def: Any,
    *,
    read_only: bool = False,
    tool_timeout: int = 30,
) -> MCPToolWrapper:
    wrapper = MCPToolWrapper(session, server_name, tool_def, read_only=read_only, tool_timeout=tool_timeout)
    registry.register(wrapper)
    return wrapper


@tool_parameters(
    tool_parameters_schema(
        required=["action"],
        action=StringSchema("Action: list, add, or remove.", enum=["list", "add", "remove"]),
        name=StringSchema("Optional job name."),
        message=StringSchema("Message or task for action='add'."),
        every_seconds=IntegerSchema(description="Repeat interval in seconds.", minimum=1),
        at=StringSchema("ISO datetime for one-shot jobs."),
        job_id=StringSchema("Job id for action='remove'."),
        deliver=BooleanSchema(description="Whether the cron callback should deliver output."),
    )
)
class CronTool(Tool):
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return "List, add, or remove scheduled Pico jobs."

    @property
    def read_only(self) -> bool:
        return False

    @property
    def concurrency_safe(self) -> bool:
        return False

    async def execute(
        self,
        action: str,
        name: str = "",
        message: str = "",
        every_seconds: int | None = None,
        at: str | None = None,
        job_id: str = "",
        deliver: bool = False,
        **kwargs: Any,
    ) -> str:
        if action == "list":
            jobs = self.agent.cron.list_jobs(include_disabled=True)
            if not jobs:
                return "No cron jobs."
            return "\n".join(
                f"{job.id} {job.name} enabled={job.enabled} next={job.state.next_run_at_ms or 'none'}"
                for job in jobs
            )
        if action == "remove":
            if not job_id:
                return "Error: job_id is required for remove"
            return f"cron remove {job_id}: {self.agent.cron.remove_job(job_id)}"
        if action != "add":
            return f"Error: unknown cron action '{action}'"
        if not str(message).strip():
            return "Error: message is required for add"
        if at:
            schedule = CronSchedule(kind="at", at_ms=now_ms_from_datetime(datetime.fromisoformat(at)))
            delete_after_run = True
        elif every_seconds:
            schedule = CronSchedule(kind="every", every_ms=int(every_seconds) * 1000)
            delete_after_run = False
        else:
            return "Error: either every_seconds or at is required"
        job = self.agent.cron.add_job(
            name=name or message[:30],
            schedule=schedule,
            message=message,
            deliver=deliver,
            delete_after_run=delete_after_run,
        )
        return f"cron added {job.id}: {job.name}"


@tool_parameters(
    tool_parameters_schema(
        required=["task"],
        task=StringSchema("Task for a bounded read-only child agent.", min_length=1),
        label=StringSchema("Optional label."),
        max_steps=IntegerSchema(description="Maximum child-agent steps.", minimum=1, maximum=20),
    )
)
class SpawnTool(Tool):
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return "Spawn a bounded read-only child agent task."

    @property
    def read_only(self) -> bool:
        return True

    @property
    def concurrency_safe(self) -> bool:
        return False

    async def execute(self, task: str, label: str = "", max_steps: int = 3, **kwargs: Any) -> str:
        del label
        delegate = self.agent.tool_registry.get("delegate")
        if delegate is None:
            return "Error: delegate depth exceeded"
        return await delegate.execute(task=task, max_steps=max_steps)


def _new_cell(source: str, cell_type: str = "code") -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type, "source": source, "metadata": {}}
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


@tool_parameters(
    tool_parameters_schema(
        required=["path", "cell_index"],
        path=StringSchema("Path to the .ipynb notebook file.", min_length=1),
        cell_index=IntegerSchema(description="0-based cell index.", minimum=0),
        new_source=StringSchema("New source content."),
        cell_type=StringSchema("Cell type.", enum=["code", "markdown"]),
        edit_mode=StringSchema("Edit mode.", enum=["replace", "insert", "delete"]),
    )
)
class NotebookEditTool(Tool):
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def name(self) -> str:
        return "notebook_edit"

    @property
    def description(self) -> str:
        return "Edit Jupyter notebook cells."

    @property
    def concurrency_safe(self) -> bool:
        return False

    async def execute(
        self,
        path: str,
        cell_index: int,
        new_source: str = "",
        cell_type: str = "code",
        edit_mode: str = "replace",
        **kwargs: Any,
    ) -> str:
        if not path.endswith(".ipynb"):
            return "Error: notebook_edit only works on .ipynb files"
        fp = self.agent.path(path)
        if not fp.exists():
            if edit_mode != "insert":
                return f"Error: File not found: {path}"
            notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [_new_cell(new_source, cell_type)]}
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
            return f"notebook created {path}"
        notebook = json.loads(fp.read_text(encoding="utf-8"))
        cells = list(notebook.get("cells", []))
        if edit_mode == "delete":
            if cell_index >= len(cells):
                return f"Error: cell_index {cell_index} out of range"
            cells.pop(cell_index)
        elif edit_mode == "insert":
            cells.insert(min(cell_index + 1, len(cells)), _new_cell(new_source, cell_type))
        else:
            if cell_index >= len(cells):
                return f"Error: cell_index {cell_index} out of range"
            cells[cell_index]["source"] = new_source
            cells[cell_index]["cell_type"] = cell_type
            if cell_type == "markdown":
                cells[cell_index].pop("outputs", None)
                cells[cell_index].pop("execution_count", None)
            else:
                cells[cell_index].setdefault("outputs", [])
                cells[cell_index].setdefault("execution_count", None)
        notebook["cells"] = cells
        fp.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        return f"notebook edited {path}"

