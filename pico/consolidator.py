"""Nanobot-style session consolidation for pico."""

from __future__ import annotations

import json
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable

from .session_manager import Session
from .templates import render_template
from .workspace import clip

SESSION_HISTORY_RECENT_WINDOW = 8
SESSION_SUMMARY_CHAR_LIMIT = 1600
CONTEXT_WINDOW_TOKENS = 1800
SAFETY_BUFFER_TOKENS = 256
MAX_CONSOLIDATION_ROUNDS = 5
MAX_CONSOLIDATION_MESSAGES = 60


def format_history_entries(history: list[dict[str, Any]]) -> str:
    lines = []
    for item in history:
        role = str(item.get("role", "")).strip()
        if role == "tool":
            name = str(item.get("name", "")).strip() or "tool"
            args = json.dumps(item.get("args", {}), ensure_ascii=False, sort_keys=True)
            content = clip(str(item.get("content", "")).strip(), 900)
            lines.append(f"[tool:{name}] {args}")
            if content:
                lines.append(content)
            continue
        content = clip(str(item.get("content", "")).strip(), 900)
        if content:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    return max(1, len(format_history_entries([message])) // 4)


def split_history_for_compaction(
    history: list[dict[str, Any]], keep_recent: int = SESSION_HISTORY_RECENT_WINDOW
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history = list(history or [])
    if keep_recent < 0:
        keep_recent = 0
    if len(history) <= keep_recent:
        return [], history
    return history[:-keep_recent], history[-keep_recent:]


def build_consolidation_prompt(history: list[dict[str, Any]], existing_summary: str = "", task_summary: str = "") -> str:
    transcript = format_history_entries(history)
    existing_summary = clip(str(existing_summary or "").strip(), SESSION_SUMMARY_CHAR_LIMIT)
    task_summary = clip(str(task_summary or "").strip(), 300)
    sections = [
        render_template("agent/consolidator_archive.md", strip=True),
    ]
    if task_summary:
        sections.append(f"Current task:\n{task_summary}")
    if existing_summary:
        sections.append(f"Existing session summary:\n{existing_summary}")
    sections.append(f"New transcript to merge:\n{transcript or '(empty)'}")
    return "\n\n".join(sections)


def normalize_session_summary(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if text.startswith("<final>") and text.endswith("</final>"):
        text = text[len("<final>") : -len("</final>")].strip()
    lines = []
    for raw in text.splitlines():
        line = str(raw).strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">") and " " not in line:
            continue
        lines.append(line)
    return clip("\n".join(lines), SESSION_SUMMARY_CHAR_LIMIT)


@dataclass
class ConsolidationResult:
    cursor: int
    summary: str
    archived_messages: int
    last_consolidated: int


class _SessionLock:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __enter__(self) -> "_SessionLock":
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._lock.release()
        return False


class Consolidator:
    _MAX_CONSOLIDATION_ROUNDS = MAX_CONSOLIDATION_ROUNDS
    _MAX_CHUNK_MESSAGES = MAX_CONSOLIDATION_MESSAGES
    _SAFETY_BUFFER = SAFETY_BUFFER_TOKENS

    def __init__(
        self,
        store: Any,
        summarize_fn: Callable[[str, int], str],
        *,
        context_window_tokens: int = CONTEXT_WINDOW_TOKENS,
        max_completion_tokens: int = 256,
        build_messages: Callable[..., list[dict[str, Any]]] | None = None,
        get_tool_definitions: Callable[[], list[dict[str, Any]]] | None = None,
    ):
        self.store = store
        self.summarize_fn = summarize_fn
        self.context_window_tokens = int(context_window_tokens)
        self.max_completion_tokens = int(max_completion_tokens)
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, _SessionLock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> _SessionLock:
        key = str(session_key or "").strip() or "__default__"
        lock = self._locks.get(key)
        if lock is None:
            lock = _SessionLock()
            self._locks[key] = lock
        return lock

    @staticmethod
    def _session_key(session_payload: dict[str, Any]) -> str:
        for field in ("id", "session_id", "key"):
            value = str(session_payload.get(field, "")).strip()
            if value:
                return value
        return "session"

    @staticmethod
    def _session_channel(session_payload: dict[str, Any]) -> str | None:
        metadata = dict(session_payload.get("session_metadata", {}) or {})
        value = str(metadata.get("channel", "")).strip()
        return value or None

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "")).upper() or "UNKNOWN"
            content = message.get("content", "")
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        parts.append(str(block))
                        continue
                    text = block.get("text")
                    image = block.get("image_url")
                    parts.append(str(text or image or block))
                content = "\n".join(parts)
            lines.append(f"{role}:\n{content}")
        return "\n\n".join(lines)

    def _estimate_tool_tokens(self) -> int:
        if self._get_tool_definitions is None:
            return 0
        try:
            payload = self._get_tool_definitions() or []
        except Exception:
            return 0
        if not payload:
            return 0
        return max(1, len(json.dumps(payload, ensure_ascii=False, sort_keys=True)) // 4)

    def pick_consolidation_boundary(
        self, session_payload: dict[str, Any], tokens_to_remove: int
    ) -> tuple[int, int] | None:
        history = list(session_payload.get("history", []))
        start = max(0, int(session_payload.get("last_consolidated", 0) or 0))
        if start >= len(history) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for index in range(start, len(history)):
            message = history[index]
            if index > start and message.get("role") == "user":
                last_boundary = (index, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)
        return last_boundary

    def _cap_consolidation_boundary(self, session_payload: dict[str, Any], end_idx: int) -> int | None:
        history = list(session_payload.get("history", []))
        start = max(0, int(session_payload.get("last_consolidated", 0) or 0))
        if end_idx - start <= self._MAX_CHUNK_MESSAGES:
            return end_idx
        capped_end = start + self._MAX_CHUNK_MESSAGES
        for index in range(capped_end, start, -1):
            if index < len(history) and history[index].get("role") == "user":
                return index
        return None

    def estimate_session_prompt_tokens(self, session_payload: dict[str, Any], session_summary: str | None = None) -> int:
        live_history = Session.from_payload(session_payload).get_history(max_messages=0)
        if self._build_messages is not None:
            try:
                prompt_messages = self._build_messages(
                    history=live_history,
                    current_message="[token-probe]",
                    channel=self._session_channel(session_payload),
                    chat_id=self._session_key(session_payload),
                    session_summary=session_summary,
                )
                prompt = self._messages_to_prompt(prompt_messages)
                return max(1, len(prompt) // 4) + self._estimate_tool_tokens()
            except Exception:
                pass
        history_tokens = sum(estimate_message_tokens(message) for message in live_history)
        summary_tokens = max(0, len(str(session_summary or "").strip()) // 4)
        return history_tokens + summary_tokens + 256

    def archive(self, messages: list[dict[str, Any]], *, existing_summary: str = "", task_summary: str = "") -> tuple[int, str | None]:
        if not messages:
            return 0, None
        prompt = build_consolidation_prompt(
            messages,
            existing_summary=existing_summary,
            task_summary=task_summary,
        )
        try:
            raw_summary = self.summarize_fn(prompt, self.max_completion_tokens)
            summary = normalize_session_summary(raw_summary)
            if not summary:
                return 0, None
            cursor = int(self.store.append_history(summary))
            return cursor, summary
        except Exception:
            cursor = int(self.store.raw_archive(messages))
            return cursor, None

    def maybe_consolidate_by_tokens(
        self,
        session_payload: dict[str, Any],
        *,
        session_summary: str | None = None,
        task_summary: str = "",
    ) -> ConsolidationResult | None:
        history = list(session_payload.get("history", []))
        if not history or self.context_window_tokens <= 0:
            return None

        session_key = self._session_key(session_payload)
        with self.get_lock(session_key):
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            if budget <= 0:
                return None
            target = max(1, budget // 2)
            estimated = self.estimate_session_prompt_tokens(session_payload, session_summary=session_summary)
            if estimated <= 0 or estimated < budget:
                return None

            last_result: ConsolidationResult | None = None
            last_summary = str(session_summary or "").strip()
            total_archived_messages = 0
            metadata = session_payload.setdefault("session_metadata", {})
            for _ in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    break
                boundary = self.pick_consolidation_boundary(session_payload, max(1, estimated - target))
                if boundary is None:
                    break
                end_idx = self._cap_consolidation_boundary(session_payload, boundary[0])
                if end_idx is None:
                    break
                start = max(0, int(session_payload.get("last_consolidated", 0) or 0))
                chunk = history[start:end_idx]
                if not chunk:
                    break
                cursor, summary = self.archive(
                    chunk,
                    existing_summary=last_summary,
                    task_summary=task_summary,
                )
                session_payload["last_consolidated"] = end_idx
                archived_messages = len(chunk)
                total_archived_messages += archived_messages
                effective_summary = last_summary
                if summary:
                    effective_summary = summary
                    last_summary = summary
                    metadata["_last_summary"] = {
                        "text": summary,
                        "last_active": str(session_payload.get("updated_at", "")).strip(),
                    }
                last_result = ConsolidationResult(
                    cursor=int(cursor),
                    summary=effective_summary,
                    archived_messages=total_archived_messages,
                    last_consolidated=end_idx,
                )
                estimated = self.estimate_session_prompt_tokens(session_payload, session_summary=last_summary)
                if not summary:
                    break

            return last_result
