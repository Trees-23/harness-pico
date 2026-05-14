"""Pico product-semantics gateway for standard tool execution."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from ..security.network import contains_internal_url
from ..skills import BUILTIN_SKILLS_DIR
from ..workspace import clip
from .base import Tool
from .locks import FileLockManager
from .registry import ToolRegistry

_LEGACY_TOOL_NAMES = {
    "list_files": "list_dir",
    "search": "grep",
    "run_shell": "exec",
    "patch_file": "edit_file",
}


class ToolExecutor:
    """Thin gateway for approval, auditing, snapshots, memory, and trace metadata."""

    def __init__(self, agent: Any, registry: ToolRegistry) -> None:
        self.agent = agent
        self.registry = registry
        self.file_locks = FileLockManager()

    @staticmethod
    def _run_coroutine(coro: Any) -> Any:
        return asyncio.run(coro)

    @staticmethod
    def _risk_level(tool: Tool | None) -> str:
        return "high" if tool is None or not tool.read_only else "low"

    @staticmethod
    def _read_only(tool: Tool | None) -> bool:
        return bool(tool is not None and tool.read_only)

    def _set_metadata(
        self,
        *,
        tool: Tool | None,
        status: str,
        error_code: str = "",
        security_event_type: str = "",
        affected_paths: list[str] | None = None,
        workspace_changed: bool = False,
        diff_summary: list[str] | None = None,
        workspace_fingerprint: bool = False,
    ) -> None:
        metadata: dict[str, Any] = {
            "tool_status": status,
            "tool_error_code": error_code,
            "security_event_type": security_event_type,
            "risk_level": self._risk_level(tool),
            "read_only": self._read_only(tool),
            "affected_paths": list(affected_paths or []),
            "workspace_changed": bool(workspace_changed),
            "diff_summary": list(diff_summary or []),
        }
        if workspace_fingerprint:
            metadata["workspace_fingerprint"] = self.agent.workspace.fingerprint()
        self.agent._last_tool_result_metadata = metadata

    def _invalid_arguments_result(self, name: str, error: str) -> str:
        message = f"error: invalid arguments for {name}: {self._strip_registry_error(error)}"
        example = self.agent.tool_example(name)
        if example:
            message += f"\nexample: {example}"
        return message

    @staticmethod
    def _strip_registry_error(error: str) -> str:
        text = str(error or "")
        prefix = "Error: "
        if text.startswith(prefix):
            text = text[len(prefix) :]
        return text

    @staticmethod
    def _canonical_name(name: str) -> str:
        return _LEGACY_TOOL_NAMES.get(str(name), str(name))

    def _preflight_security(self, name: str, args: dict[str, Any]) -> str | None:
        canonical_name = self._canonical_name(name)
        try:
            if canonical_name == "list_dir":
                path = self.agent.path(args.get("path", "."))
                if not path.is_dir():
                    return "path is not a directory"
            elif canonical_name == "read_file":
                raw_path = str(args["path"])
                if raw_path.startswith("/"):
                    try:
                        candidate = Path(raw_path).resolve()
                        candidate.relative_to(BUILTIN_SKILLS_DIR.resolve())
                    except Exception:
                        path = self.agent.path(raw_path)
                    else:
                        path = candidate
                else:
                    path = self.agent.path(raw_path)
                if not path.is_file():
                    return "path is not a file"
            elif canonical_name in {"grep", "glob"}:
                self.agent.path(args.get("path", "."))
            elif canonical_name == "exec":
                if contains_internal_url(str(args.get("command", ""))):
                    return "internal/private URL detected"
            elif canonical_name == "write_file":
                path = self.agent.path(args["path"])
                if path.exists() and path.is_dir():
                    return "path is a directory"
            elif canonical_name == "edit_file":
                path = self.agent.path(args["path"])
                if not path.is_file():
                    return "path is not a file"
            elif canonical_name == "notebook_edit":
                path = self.agent.path(args["path"])
                if path.exists() and path.is_dir():
                    return "path is a directory"
        except Exception as exc:
            return str(exc)
        return None

    def _write_lock_key(self, name: str, args: dict[str, Any]) -> str | None:
        if self._canonical_name(name) not in {"write_file", "edit_file", "notebook_edit"}:
            return None
        try:
            return str(self.agent.path(args["path"]))
        except Exception:
            return None

    def _execute_tool(self, tool: Tool, prepared_args: dict[str, Any]) -> Any:
        return self._run_coroutine(tool.execute(**prepared_args))

    def execute(self, name: str, args: dict[str, Any] | None) -> str:
        tool, prepared_args, error = self.registry.prepare_call(name, args or {})
        if tool is None:
            self._set_metadata(
                tool=None,
                status="rejected",
                error_code="unknown_tool",
                affected_paths=[],
                workspace_changed=False,
            )
            return f"error: unknown tool '{name}'"

        if error:
            security_event_type = "path_escape" if "path escapes workspace" in error else ""
            self._set_metadata(
                tool=tool,
                status="rejected",
                error_code="invalid_arguments",
                security_event_type=security_event_type,
            )
            return self._invalid_arguments_result(name, error)

        preflight_error = self._preflight_security(name, prepared_args)
        if preflight_error:
            security_event_type = ""
            if "path escapes workspace" in preflight_error:
                security_event_type = "path_escape"
            elif "internal/private URL" in preflight_error:
                security_event_type = "internal_url_block"
            self._set_metadata(
                tool=tool,
                status="rejected",
                error_code="invalid_arguments",
                security_event_type=security_event_type,
            )
            return self._invalid_arguments_result(name, preflight_error)

        if self.agent.repeated_tool_call(name, prepared_args):
            self._set_metadata(tool=tool, status="rejected", error_code="repeated_identical_call")
            if name == "read_file":
                return (
                    "error: repeated read_file call for an overlapping already-read file range; "
                    "use the previous tool result to answer or request a non-overlapping range"
                )
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"

        if not tool.read_only and not self.agent.approve(name, prepared_args):
            self._set_metadata(
                tool=tool,
                status="rejected",
                error_code="approval_denied",
                security_event_type="read_only_block" if self.agent.read_only else "approval_denied",
            )
            return f"error: approval denied for {name}"

        before_snapshot = self.agent.capture_workspace_snapshot() if not tool.read_only else {}
        try:
            lock_key = self._write_lock_key(name, prepared_args)
            if lock_key is None:
                raw_result = self._execute_tool(tool, prepared_args)
            else:
                with self.file_locks.lock(lock_key):
                    raw_result = self._execute_tool(tool, prepared_args)
            result = clip(raw_result) if isinstance(raw_result, str) else raw_result
            after_snapshot = self.agent.capture_workspace_snapshot() if not tool.read_only else before_snapshot
            affected_paths, diff_summary = self.agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            tool_status = "ok"
            tool_error_code = ""
            if self._canonical_name(name) == "exec":
                match = re.search(r"exit_code:\s*(-?\d+)", result)
                exit_code = int(match.group(1)) if match else 0
                if exit_code != 0 and workspace_changed:
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                elif exit_code != 0:
                    tool_status = "error"
                    tool_error_code = "tool_failed"
            self.agent.update_memory_after_tool(name, prepared_args, result)
            self._set_metadata(
                tool=tool,
                status=tool_status,
                error_code=tool_error_code,
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                diff_summary=diff_summary,
                workspace_fingerprint=True,
            )
            self.agent.record_process_note_for_tool(name, self.agent._last_tool_result_metadata)
            return result
        except Exception as exc:
            after_snapshot = self.agent.capture_workspace_snapshot() if not tool.read_only else before_snapshot
            affected_paths, diff_summary = self.agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            self._set_metadata(
                tool=tool,
                status="partial_success" if workspace_changed else "error",
                error_code="tool_partial_success" if workspace_changed else "tool_failed",
                security_event_type=security_event_type,
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                diff_summary=diff_summary,
                workspace_fingerprint=True,
            )
            self.agent.record_process_note_for_tool(name, self.agent._last_tool_result_metadata)
            return f"error: tool {name} failed: {exc}"
