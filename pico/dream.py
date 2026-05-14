"""Nanobot-style long-term memory processor for pico."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .memory import MemoryStore
from .templates import render_template
from .tooling.pico_tools import EditFileTool, ReadFileTool, WriteFileTool
from .tooling.registry import ToolRegistry
from .workspace import clip

DREAM_MAX_BATCH_SIZE = 20
DREAM_MAX_ITERATIONS = 8
DREAM_MAX_NEW_TOKENS = 512
DREAM_INTERVAL_SECONDS = 300
DREAM_STALE_THRESHOLD_DAYS = 14

_ALLOWED_DREAM_PATHS = (
    ".pico/memory/MEMORY.md",
    ".pico/USER.md",
    ".pico/SOUL.md",
)


@dataclass
class DreamResult:
    processed_entries: int
    new_cursor: int
    completed: bool
    tool_events: list[dict[str, Any]]
    analysis: str


class _DreamToolAgent:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._allowed = {(self.root / relative).resolve() for relative in _ALLOWED_DREAM_PATHS}

    def path(self, raw_path: str) -> Path:
        path = Path(str(raw_path))
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if resolved not in self._allowed:
            raise ValueError(f"path escapes dream memory boundary: {raw_path}")
        return resolved


def _parse_action(raw: str) -> tuple[str, Any]:
    raw = str(raw or "")
    if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
        body = _extract(raw, "tool")
        try:
            payload = json.loads(body)
        except Exception:
            return "retry", _retry_notice("model returned malformed tool JSON")
        if not isinstance(payload, dict) or not str(payload.get("name", "")).strip():
            return "retry", _retry_notice("tool payload is missing a tool name")
        args = payload.get("args", {})
        if args is None:
            payload["args"] = {}
        elif not isinstance(args, dict):
            return "retry", _retry_notice("tool payload args must be a JSON object")
        return "tool", payload
    if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
        payload = _parse_xml_tool(raw)
        if payload is not None:
            return "tool", payload
        return "retry", _retry_notice()
    if "<final>" in raw:
        final = _extract(raw, "final").strip()
        return ("final", final) if final else ("retry", _retry_notice("model returned an empty <final> answer"))
    stripped = raw.strip()
    return ("final", stripped) if stripped else ("retry", _retry_notice("model returned an empty response"))


def _retry_notice(problem: str | None = None) -> str:
    prefix = "Dream runtime notice"
    if problem:
        prefix += f": {problem}"
    return (
        f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
        'For multi-line files, prefer <tool name="write_file" path=".pico/memory/MEMORY.md"><content>...</content></tool>.'
    )


def _extract(text: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return text
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


def _extract_raw(text: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return text
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return text[start:]
    return text[start:end]


def _parse_attrs(text: str) -> dict[str, str]:
    attrs = {}
    for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
        attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
    return attrs


def _parse_xml_tool(raw: str) -> dict[str, Any] | None:
    match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
    if not match:
        return None
    attrs = _parse_attrs(match.group("attrs"))
    name = str(attrs.pop("name", "")).strip()
    if not name:
        return None
    body = match.group("body")
    args = dict(attrs)
    for key in ("content", "old_text", "new_text", "path"):
        if f"<{key}>" in body:
            args[key] = _extract_raw(body, key)
    body_text = body.strip("\n")
    if name == "write_file" and "content" not in args and body_text:
        args["content"] = body_text
    return {"name": name, "args": args}


class Dream:
    """Processes archived summaries into durable memory files."""

    def __init__(
        self,
        store: MemoryStore,
        model_client: Any,
        *,
        interval_seconds: int = DREAM_INTERVAL_SECONDS,
        max_batch_size: int = DREAM_MAX_BATCH_SIZE,
        max_iterations: int = DREAM_MAX_ITERATIONS,
        max_new_tokens: int = DREAM_MAX_NEW_TOKENS,
        stale_threshold_days: int = DREAM_STALE_THRESHOLD_DAYS,
    ):
        self.store = store
        self.model_client = model_client
        self.interval_seconds = max(0, int(interval_seconds))
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_iterations = max(1, int(max_iterations))
        self.max_new_tokens = max(64, int(max_new_tokens))
        self.stale_threshold_days = max(1, int(stale_threshold_days))
        self._last_run_at = 0.0
        self._tool_agent = _DreamToolAgent(self.store.workspace_root)
        self.tool_registry = self._build_tool_registry()

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(ReadFileTool(self._tool_agent))
        registry.register(WriteFileTool(self._tool_agent))
        registry.register(EditFileTool(self._tool_agent))
        registry.register_alias("patch_file", "edit_file")
        return registry

    def _list_existing_skills(self) -> list[str]:
        entries: list[str] = []
        skills_dir = self.store.workspace_root / "skills"
        if not skills_dir.exists():
            return entries
        for item in sorted(skills_dir.iterdir(), key=lambda path: path.name.lower()):
            if not item.is_dir():
                continue
            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue
            description = "(no description)"
            try:
                for line in skill_md.read_text(encoding="utf-8").splitlines():
                    if line.lower().startswith("description:"):
                        description = line.split(":", 1)[1].strip() or description
                        break
            except Exception:
                pass
            entries.append(f"{item.name} - {description}")
        return entries

    def _annotate_with_ages(self, content: str) -> str:
        ages = self.store.git.line_ages(".pico/memory/MEMORY.md")
        if not ages:
            return content
        had_trailing = content.endswith("\n")
        lines = content.splitlines()
        if len(lines) != len(ages):
            return content
        annotated: list[str] = []
        for line, age in zip(lines, ages):
            if not line.strip() or age.age_days <= self.stale_threshold_days:
                annotated.append(line)
                continue
            annotated.append(f"{line}  <- {age.age_days}d")
        result = "\n".join(annotated)
        if had_trailing:
            result += "\n"
        return result

    def run(self) -> DreamResult | None:
        self._last_run_at = time.monotonic()
        last_cursor = self.store.get_last_dream_cursor()
        entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return None

        batch = entries[: self.max_batch_size]
        analysis = self._phase1_analysis(batch)
        completed, tool_events = self._phase2_apply(batch, analysis)
        new_cursor = int(batch[-1]["cursor"])
        self.store.set_last_dream_cursor(new_cursor)
        self.store.compact_history()
        return DreamResult(
            processed_entries=len(batch),
            new_cursor=new_cursor,
            completed=completed,
            tool_events=tool_events,
            analysis=analysis,
        )

    def _phase1_analysis(self, batch: list[dict[str, Any]]) -> str:
        prompt = self._build_phase1_prompt(batch)
        return str(self.model_client.complete(prompt, min(self.max_new_tokens, 400)) or "").strip()

    def _phase2_apply(self, batch: list[dict[str, Any]], analysis: str) -> tuple[bool, list[dict[str, Any]]]:
        system = render_template("agent/dream_phase2.md", strip=True)
        user = self._build_phase2_user_prompt(batch, analysis)
        transcript = [{"role": "user", "content": user}]
        tool_events: list[dict[str, Any]] = []

        for _ in range(self.max_iterations):
            prompt = self._conversation_prompt(system, transcript)
            raw = str(self.model_client.complete(prompt, min(self.max_new_tokens, 256)) or "")
            kind, payload = _parse_action(raw)
            if kind == "final":
                return True, tool_events
            if kind == "retry":
                transcript.append({"role": "assistant", "content": raw})
                transcript.append({"role": "user", "content": str(payload)})
                continue

            name = str(payload.get("name", "")).strip()
            args = dict(payload.get("args", {}) or {})
            result = self._run_tool(name, args)
            tool_events.append(
                {
                    "name": name,
                    "args": args,
                    "status": "ok" if not str(result).startswith("error:") else "error",
                    "detail": clip(str(result), 240),
                }
            )
            transcript.append({"role": "assistant", "content": raw})
            transcript.append({"role": "tool", "name": name, "args": args, "content": str(result)})

        return False, tool_events

    def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.tool_registry:
            return f"error: dream cannot use tool '{name}'"
        tool, prepared_args, error = self.tool_registry.prepare_call(name, args)
        if error:
            return f"error: invalid arguments for {name}: {error}"
        try:
            assert tool is not None
            execute_legacy = getattr(tool, "_execute_legacy", None)
            if callable(execute_legacy):
                return str(execute_legacy(prepared_args))
            return f"error: dream tool '{name}' is not sync-compatible"
        except Exception as exc:
            return f"error: tool {name} failed: {exc}"

    def _build_phase1_prompt(self, batch: list[dict[str, Any]]) -> str:
        history_text = "\n".join(
            f"[{entry.get('timestamp', '')}] {entry.get('content', '')}".strip()
            for entry in batch
        )
        annotated_memory = self._annotate_with_ages(self.store.read_memory() or "(empty)")
        sections = [
            render_template("agent/dream_phase1.md", strip=True, stale_threshold_days=self.stale_threshold_days),
            "## Archived Summaries",
            history_text or "(empty)",
            "## Current MEMORY.md",
            annotated_memory,
            "## Current USER.md",
            self.store.read_user() or "(empty)",
            "## Current SOUL.md",
            self.store.read_soul() or "(empty)",
        ]
        return "\n\n".join(sections)

    def _build_phase2_user_prompt(self, batch: list[dict[str, Any]], analysis: str) -> str:
        history_text = "\n".join(
            f"[{entry.get('timestamp', '')}] {entry.get('content', '')}".strip()
            for entry in batch
        )
        skills = self._list_existing_skills()
        return "\n\n".join(
            [
                "## Analysis",
                analysis or "(empty)",
                "## Allowed Paths",
                "\n".join(f"- {path}" for path in _ALLOWED_DREAM_PATHS),
                "## Current MEMORY.md",
                self.store.read_memory() or "(empty)",
                "## Current USER.md",
                self.store.read_user() or "(empty)",
                "## Current SOUL.md",
                self.store.read_soul() or "(empty)",
                "## Existing Skills",
                "\n".join(f"- {item}" for item in skills) or "(none)",
                "## Archived Summaries",
                history_text or "(empty)",
            ]
        )

    @staticmethod
    def _conversation_prompt(system: str, transcript: list[dict[str, Any]]) -> str:
        lines = [system]
        for item in transcript:
            role = item.get("role", "user")
            if role == "tool":
                lines.append(
                    "TOOL RESULT:\n"
                    f"name: {item.get('name', '')}\n"
                    f"args: {json.dumps(item.get('args', {}), ensure_ascii=False, sort_keys=True)}\n"
                    f"content:\n{item.get('content', '')}"
                )
            else:
                lines.append(f"{str(role).upper()}:\n{item.get('content', '')}")
        return "\n\n".join(lines)
