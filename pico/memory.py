"""Nanobot-style memory store plus pico working-memory facade."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .consolidator import normalize_session_summary
from .gitstore import GitStore
from .templates import read_template
from .workspace import clip, now

WORKING_FILE_LIMIT = 8
EPISODIC_NOTE_LIMIT = 12
FILE_SUMMARY_LIMIT = 6
MEMORY_DIR_NAME = "memory"
RUNTIME_DIR_NAME = ".pico"
MEMORY_FILE_NAME = "MEMORY.md"
HISTORY_FILE_NAME = "history.jsonl"
SOUL_FILE_NAME = "SOUL.md"
USER_FILE_NAME = "USER.md"
AGENTS_FILE_NAME = "AGENTS.md"
TOOLS_FILE_NAME = "TOOLS.md"
HEARTBEAT_FILE_NAME = "HEARTBEAT.md"
CURSOR_FILE_NAME = ".cursor"
DREAM_CURSOR_FILE_NAME = ".dream_cursor"
LEGACY_HISTORY_FILE_NAME = "HISTORY.md"
MAX_HISTORY_ENTRIES = 1000


DEFAULT_MEMORY_SECTIONS = (
    "User Information",
    "Preferences",
    "Project Context",
    "Important Notes",
)


def default_memory_state() -> dict[str, Any]:
    return {
        "working": {
            "task_summary": "",
            "recent_files": [],
        },
        "episodic_notes": [],
        "file_summaries": {},
        "task": "",
        "files": [],
        "notes": [],
        "next_note_index": 0,
    }


def default_history_archive_state() -> dict[str, Any]:
    return {
        "latest_summary": "",
        "latest_cursor": 0,
        "compaction_count": 0,
        "archived_messages": 0,
        "last_compacted_at": "",
    }


class MemoryStore:
    """Pure file I/O for pico memory files, aligned to nanobot layout."""

    def __init__(self, workspace_root: str | Path, max_history_entries: int = MAX_HISTORY_ENTRIES):
        self.workspace_root = Path(workspace_root)
        self.max_history_entries = int(max_history_entries)
        self.runtime_dir = self.workspace_root / RUNTIME_DIR_NAME
        self.memory_dir = self.runtime_dir / MEMORY_DIR_NAME
        self.legacy_memory_dir = self.workspace_root / MEMORY_DIR_NAME
        self.memory_file = self.memory_dir / MEMORY_FILE_NAME
        self.history_file = self.memory_dir / HISTORY_FILE_NAME
        self.legacy_history_file = self.memory_dir / LEGACY_HISTORY_FILE_NAME
        self.agents_file = self.runtime_dir / AGENTS_FILE_NAME
        self.tools_file = self.runtime_dir / TOOLS_FILE_NAME
        self.soul_file = self.runtime_dir / SOUL_FILE_NAME
        self.user_file = self.runtime_dir / USER_FILE_NAME
        self.heartbeat_file = self.runtime_dir / HEARTBEAT_FILE_NAME
        self.cursor_file = self.memory_dir / CURSOR_FILE_NAME
        self.dream_cursor_file = self.memory_dir / DREAM_CURSOR_FILE_NAME
        self._git = GitStore(
            self.workspace_root,
            tracked_files=[
                ".pico/AGENTS.md",
                ".pico/TOOLS.md",
                ".pico/SOUL.md",
                ".pico/USER.md",
                ".pico/HEARTBEAT.md",
                ".pico/memory/MEMORY.md",
            ],
        )
        self.sync()
        self._maybe_migrate_legacy_history()

    @property
    def git(self) -> GitStore:
        return self._git

    def _migrate_legacy_file(self, legacy_path: Path, new_path: Path, *, template_text: str | None = None) -> bool:
        if not legacy_path.exists():
            return False
        legacy_text = legacy_path.read_text(encoding="utf-8")
        if new_path.exists():
            if template_text is None:
                return False
            current_text = new_path.read_text(encoding="utf-8")
            if current_text.strip() != str(template_text).strip():
                return False
            if legacy_text.strip() == str(template_text).strip():
                return False
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(legacy_text, encoding="utf-8")
        return True

    def sync(self) -> list[Path]:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []
        agents_template = read_template("AGENTS.md")
        tools_template = read_template("TOOLS.md")
        soul_template = read_template("SOUL.md")
        user_template = read_template("USER.md")
        heartbeat_template = read_template("HEARTBEAT.md")
        memory_template = read_template("memory", "MEMORY.md")

        self._migrate_legacy_file(self.workspace_root / AGENTS_FILE_NAME, self.agents_file, template_text=agents_template)
        self._migrate_legacy_file(self.workspace_root / TOOLS_FILE_NAME, self.tools_file, template_text=tools_template)
        self._migrate_legacy_file(self.workspace_root / SOUL_FILE_NAME, self.soul_file, template_text=soul_template)
        self._migrate_legacy_file(self.workspace_root / USER_FILE_NAME, self.user_file, template_text=user_template)
        self._migrate_legacy_file(
            self.workspace_root / HEARTBEAT_FILE_NAME,
            self.heartbeat_file,
            template_text=heartbeat_template,
        )
        self._migrate_legacy_file(self.legacy_memory_dir / MEMORY_FILE_NAME, self.memory_file, template_text=memory_template)
        self._migrate_legacy_file(self.legacy_memory_dir / HISTORY_FILE_NAME, self.history_file, template_text="")
        self._migrate_legacy_file(self.legacy_memory_dir / CURSOR_FILE_NAME, self.cursor_file, template_text="0")
        self._migrate_legacy_file(self.legacy_memory_dir / DREAM_CURSOR_FILE_NAME, self.dream_cursor_file, template_text="0")

        for path, content in (
            (self.agents_file, agents_template),
            (self.tools_file, tools_template),
            (self.soul_file, soul_template),
            (self.user_file, user_template),
            (self.heartbeat_file, heartbeat_template),
            (self.memory_file, memory_template),
            (self.history_file, ""),
        ):
            if path.exists():
                continue
            path.write_text(content, encoding="utf-8")
            created.append(path)
        for path, content in ((self.cursor_file, "0"), (self.dream_cursor_file, "0")):
            if path.exists():
                continue
            path.write_text(content, encoding="utf-8")
        return created

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(content, encoding="utf-8")

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    def bootstrap_context(self) -> str:
        parts = []
        soul = self.read_soul().strip()
        user = self.read_user().strip()
        if soul:
            parts.append(f"## SOUL.md\n\n{soul}")
        if user:
            parts.append(f"## USER.md\n\n{user}")
        return "\n\n".join(parts)

    def get_memory_context(self) -> str:
        text = self.read_memory().strip()
        if not text:
            return ""
        template = read_template("memory", "MEMORY.md")
        if str(text).strip() == str(template).strip():
            return ""
        return f"## Long-term Memory\n{text}"

    def append_history(self, entry: str) -> int:
        cursor = self._next_cursor()
        record = {
            "cursor": cursor,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "content": str(entry or "").rstrip(),
        }
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with self.history_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    def _next_cursor(self) -> int:
        if self.cursor_file.exists():
            try:
                return int(self.cursor_file.read_text(encoding="utf-8").strip()) + 1
            except Exception:
                pass
        last = self._read_last_entry() or {}
        cursor = last.get("cursor")
        return int(cursor) + 1 if isinstance(cursor, int) else 1

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self.history_file.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in self.history_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        entries = self._read_entries()
        return entries[-1] if entries else None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with self.history_file.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        return [
            entry
            for entry in self._read_entries()
            if isinstance(entry.get("cursor"), int) and int(entry["cursor"]) > int(since_cursor)
        ]

    def compact_history(self) -> None:
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        self._write_entries(entries[-self.max_history_entries :])

    def get_last_dream_cursor(self) -> int:
        if self.dream_cursor_file.exists():
            try:
                return int(self.dream_cursor_file.read_text(encoding="utf-8").strip())
            except Exception:
                return 0
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self.dream_cursor_file.write_text(str(int(cursor)), encoding="utf-8")

    @staticmethod
    def format_messages(messages: list[dict[str, Any]]) -> str:
        lines = []
        for message in messages:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            stamp = str(message.get("created_at", message.get("timestamp", "?")))[:16]
            role = str(message.get("role", "")).upper() or "UNKNOWN"
            lines.append(f"[{stamp}] {role}: {content}")
        return "\n".join(lines)

    def raw_archive(self, messages: list[dict[str, Any]]) -> int:
        return self.append_history(f"[RAW] {len(messages)} messages\n{self.format_messages(messages)}")

    def _maybe_migrate_legacy_history(self) -> None:
        if not self.legacy_history_file.exists():
            return
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return
        legacy_text = self.legacy_history_file.read_text(encoding="utf-8", errors="replace")
        chunks = [chunk.strip() for chunk in legacy_text.split("\n\n") if chunk.strip()]
        entries = [
            {
                "cursor": index,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "content": chunk,
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        if entries:
            self._write_entries(entries)
            self.cursor_file.write_text(str(entries[-1]["cursor"]), encoding="utf-8")
            self.dream_cursor_file.write_text(str(entries[-1]["cursor"]), encoding="utf-8")
        backup_path = self.memory_dir / "HISTORY.md.bak"
        self.legacy_history_file.replace(backup_path)


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_workspace_path(raw_path: str | Path, workspace_root: str | Path | None = None) -> Path | None:
    path = Path(str(raw_path))
    if workspace_root is None:
        return path

    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path: str | Path, workspace_root: str | Path | None = None) -> str:
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    root = Path(workspace_root).resolve()
    return resolved.relative_to(root).as_posix()


def file_freshness(raw_path: str | Path, workspace_root: str | Path | None = None) -> str | None:
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _tokenize(text: str) -> set[str]:
    text = str(text)
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text)}
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        run = run.strip()
        if not run:
            continue
        tokens.add(run)
        chars = list(run)
        for size in (2, 3):
            if len(chars) < size:
                continue
            for index in range(len(chars) - size + 1):
                tokens.add("".join(chars[index : index + size]))
    return tokens


def _parse_timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def _looks_like_question(text: str) -> bool:
    text = str(text).strip()
    if not text:
        return False
    if any(marker in text for marker in ("?", "？", "吗", "么", "嘛")):
        return True
    lowered = text.lower()
    english_markers = ("what", "why", "how", "when", "where", "which", "who", "can you", "could you", "do you")
    if any(marker in lowered for marker in english_markers):
        return True
    chinese_markers = ("什么", "为什么", "为何", "如何", "怎么", "多少", "几个", "哪", "是否")
    return any(marker in text for marker in chinese_markers)


def _looks_like_user_fact(text: str) -> bool:
    text = str(text).strip()
    if not text or _looks_like_question(text):
        return False
    zh_patterns = (
        r"^我(有|是|叫|在|住|来自|用|使用|喜欢|偏好|需要|想要|养|会|做|负责|主要做)",
        r"^我的",
    )
    en_patterns = (
        r"^(?:i am|i'm|i have|i use|i like|i prefer|i need|i work|my )",
    )
    return any(re.search(pattern, text, re.I) for pattern in (*zh_patterns, *en_patterns))


def _fact_tags(text: str) -> list[str]:
    text = str(text).strip()
    lowered = text.lower()
    tags = set(_tokenize(text))
    pet_markers = ("猫", "狗", "宠物", "cat", "dog", "pet", "pets")
    if any(marker in text or marker in lowered for marker in pet_markers):
        tags.update({"宠物", "pet"})
    preference_markers = ("喜欢", "偏好", "prefer", "like")
    if any(marker in text or marker in lowered for marker in preference_markers):
        tags.update({"偏好", "preference"})
    tool_markers = ("python", "python3", "java", "golang", "go", "rust", "ide", "vscode", "vim")
    for marker in tool_markers:
        if marker in lowered:
            tags.add(marker)
    return sorted(tag for tag in tags if str(tag).strip())[:16]


def extract_user_fact_lines(text: str) -> list[dict[str, Any]]:
    lines = []
    for raw in str(text or "").splitlines():
        line = clip(str(raw).strip(), 500)
        if not line or not _looks_like_user_fact(line):
            continue
        lines.append(
            {
                "text": line,
                "tags": _fact_tags(line),
                "source": "user",
                "kind": "user_fact",
            }
        )
    return lines


def _normalize_note(note: Any, index: int) -> dict[str, Any]:
    if isinstance(note, dict):
        text = clip(str(note.get("text", "")).strip(), 500)
        tags = [str(tag).strip() for tag in _ensure_list(note.get("tags", [])) if str(tag).strip()]
        source = str(note.get("source", "")).strip()
        created_at = str(note.get("created_at", "")).strip() or now()
        note_index = int(note.get("note_index", index))
        kind = str(note.get("kind", "episodic")).strip() or "episodic"
        return {
            "text": text,
            "tags": _dedupe_preserve_order(tags),
            "source": source,
            "created_at": created_at,
            "note_index": note_index,
            "kind": kind,
        }
    text = clip(str(note).strip(), 500)
    return {
        "text": text,
        "tags": [],
        "source": "",
        "created_at": now(),
        "note_index": index,
        "kind": "episodic",
    }


def normalize_memory_state(state: dict[str, Any] | None, workspace_root: str | Path | None = None) -> dict[str, Any]:
    if state is None:
        state = default_memory_state()
    elif not isinstance(state, dict):
        raise TypeError("memory state must be a mapping")

    working = state.get("working")
    if not isinstance(working, dict):
        working = {}
    working.setdefault("task_summary", "")
    working.setdefault("recent_files", [])
    working["task_summary"] = clip(str(working.get("task_summary", "")).strip(), 300)
    working["recent_files"] = _dedupe_preserve_order(
        [
            canonicalize_path(path, workspace_root)
            for path in _ensure_list(working.get("recent_files", []))
            if str(path).strip()
        ]
    )[-WORKING_FILE_LIMIT:]
    state["working"] = working

    if not str(working["task_summary"]).strip() and state.get("task"):
        working["task_summary"] = clip(str(state.get("task", "")).strip(), 300)
    if not working["recent_files"] and state.get("files"):
        working["recent_files"] = _dedupe_preserve_order(
            [
                canonicalize_path(path, workspace_root)
                for path in _ensure_list(state.get("files", []))
                if str(path).strip()
            ]
        )[-WORKING_FILE_LIMIT:]

    episodic_notes = state.get("episodic_notes")
    if not isinstance(episodic_notes, list):
        episodic_notes = []
    if not episodic_notes and state.get("notes"):
        episodic_notes = [
            _normalize_note(note, index)
            for index, note in enumerate(_ensure_list(state.get("notes", [])))
            if str(note).strip()
        ]
    else:
        episodic_notes = [
            _normalize_note(note, index)
            for index, note in enumerate(episodic_notes)
            if not (isinstance(note, str) and not str(note).strip())
        ]
    episodic_notes = episodic_notes[-EPISODIC_NOTE_LIMIT:]
    state["episodic_notes"] = episodic_notes

    file_summaries = state.get("file_summaries")
    if not isinstance(file_summaries, dict):
        file_summaries = {}
    normalized_file_summaries = {}
    for path, summary in file_summaries.items():
        path = canonicalize_path(path, workspace_root)
        if isinstance(summary, dict):
            text = clip(str(summary.get("summary", "")).strip(), 500)
            created_at = str(summary.get("created_at", "")).strip() or now()
            freshness = summary.get("freshness")
            freshness = None if freshness in (None, "") else str(freshness).strip() or None
        else:
            text = clip(str(summary).strip(), 500)
            created_at = now()
            freshness = None
        if path and text:
            normalized_file_summaries[path] = {
                "summary": text,
                "created_at": created_at,
                "freshness": freshness,
            }
    state["file_summaries"] = normalized_file_summaries

    next_note_index = state.get("next_note_index")
    if not isinstance(next_note_index, int) or next_note_index < 0:
        next_note_index = 0
    max_index = max((note["note_index"] for note in episodic_notes), default=-1)
    state["next_note_index"] = max(next_note_index, max_index + 1)

    state["task"] = working["task_summary"]
    state["files"] = list(working["recent_files"])
    state["notes"] = [note["text"] for note in episodic_notes]
    return state


def set_task_summary(state: dict[str, Any], summary: str, workspace_root: str | Path | None = None) -> dict[str, Any]:
    state = normalize_memory_state(state, workspace_root)
    state["working"]["task_summary"] = clip(str(summary).strip(), 300)
    state["task"] = state["working"]["task_summary"]
    return state


def remember_file(state: dict[str, Any], path: str, workspace_root: str | Path | None = None) -> dict[str, Any]:
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    files = [item for item in state["working"]["recent_files"] if item != path]
    files.append(path)
    state["working"]["recent_files"] = files[-WORKING_FILE_LIMIT:]
    state["files"] = list(state["working"]["recent_files"])
    return state


def append_note(
    state: dict[str, Any],
    text: str,
    tags: tuple[str, ...] | list[str] = (),
    source: str = "",
    created_at: str | None = None,
    workspace_root: str | Path | None = None,
    kind: str = "episodic",
) -> dict[str, Any]:
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(text).strip(), 500)
    if not text:
        return state
    normalized_tags = _dedupe_preserve_order(
        [str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()]
    )
    note = {
        "text": text,
        "tags": normalized_tags,
        "source": str(source).strip(),
        "created_at": str(created_at).strip() if created_at else now(),
        "note_index": int(state.get("next_note_index", 0)),
        "kind": str(kind).strip() or "episodic",
    }
    state["next_note_index"] = note["note_index"] + 1
    notes = [item for item in state["episodic_notes"] if item["text"] != note["text"]]
    notes.append(note)
    state["episodic_notes"] = notes[-EPISODIC_NOTE_LIMIT:]
    state["notes"] = [item["text"] for item in state["episodic_notes"]]
    return state


def set_file_summary(state: dict[str, Any], path: str, summary: str, workspace_root: str | Path | None = None) -> dict[str, Any]:
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    summary = clip(str(summary).strip(), 500)
    if not path or not summary:
        return state
    state["file_summaries"][path] = {
        "summary": summary,
        "created_at": now(),
        "freshness": file_freshness(path, workspace_root),
    }
    return state


def invalidate_file_summary(state: dict[str, Any], path: str, workspace_root: str | Path | None = None) -> dict[str, Any]:
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if path:
        state["file_summaries"].pop(path, None)
    return state


def invalidate_stale_file_summaries(
    state: dict[str, Any], workspace_root: str | Path | None = None
) -> tuple[dict[str, Any], list[str]]:
    state = normalize_memory_state(state, workspace_root)
    invalidated = []
    for path, summary in list(state["file_summaries"].items()):
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("freshness") == current_freshness:
            continue
        invalidated.append(path)
        state["file_summaries"].pop(path, None)
    return state, invalidated


def summarize_read_result(result: str, limit: int = 180) -> str:
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    if lines[0].startswith("# "):
        lines = lines[1:]
    if not lines:
        return "(empty)"
    return clip(" | ".join(lines[:3]), limit)


def retrieval_candidates(
    state: dict[str, Any], query: str, limit: int = 3, workspace_root: str | Path | None = None
) -> list[dict[str, Any]]:
    state = normalize_memory_state(state, workspace_root)
    query_tokens = _tokenize(query)
    ranked = []
    kind_priority = {
        "user_fact": 3,
        "episodic": 1,
        "process": 0,
    }
    for note in state["episodic_notes"]:
        if str(note.get("source", "")).strip() == "assistant":
            continue
        note_tags = {tag.lower() for tag in note.get("tags", [])}
        note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
        exact_tag_match = int(bool(query_tokens & note_tags))
        keyword_overlap = len(query_tokens & note_tokens)
        if exact_tag_match == 0 and keyword_overlap == 0:
            continue
        recency = _parse_timestamp(note.get("created_at"))
        note_index = int(note.get("note_index", 0))
        kind = str(note.get("kind", "episodic")).strip() or "episodic"
        ranked.append(((exact_tag_match, keyword_overlap, kind_priority.get(kind, 1), recency, note_index), note))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in ranked[:limit]]


def retrieval_view(state: dict[str, Any], query: str, limit: int = 3, workspace_root: str | Path | None = None) -> str:
    candidates = retrieval_candidates(state, query, limit=limit, workspace_root=workspace_root)
    lines = ["Relevant memory:"]
    if not candidates:
        lines.append("- none")
        return "\n".join(lines)
    for note in candidates:
        lines.append(f"- {note['text']}")
    return "\n".join(lines)


def render_memory_text(state: dict[str, Any], workspace_root: str | Path | None = None) -> str:
    state = normalize_memory_state(state, workspace_root)
    bootstrap = MemoryStore(workspace_root).get_memory_context() if workspace_root is not None else ""
    lines = [
        "Memory:",
        f"- task: {state['working']['task_summary'] or '-'}",
        f"- recent_files: {', '.join(state['working']['recent_files']) or '-'}",
    ]
    summaries = []
    for path in state["working"]["recent_files"][:FILE_SUMMARY_LIMIT]:
        summary = state["file_summaries"].get(path, {})
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("summary") and summary.get("freshness") == current_freshness:
            summaries.append(f"- {path}: {summary['summary']}")
    if summaries:
        lines.append("- file_summaries:")
        lines.extend(f"  {line}" for line in summaries)
    else:
        lines.append("- file_summaries: -")
    lines.append(f"- episodic_notes: {len(state['episodic_notes'])}")
    if bootstrap:
        lines.append("- long_term_memory:")
        lines.extend(f"  {line}" for line in bootstrap.splitlines())
    return "\n".join(lines)


def is_effectively_empty(state: dict[str, Any], workspace_root: str | Path | None = None) -> bool:
    state = normalize_memory_state(state, workspace_root)
    return (
        not str(state["working"]["task_summary"]).strip()
        and not state["working"]["recent_files"]
        and not state["episodic_notes"]
        and not state["file_summaries"]
    )


class MemoryManager:
    def __init__(self, state: dict[str, Any] | None = None, workspace_root: str | Path | None = None):
        self.workspace_root = workspace_root
        self.state = normalize_memory_state(state, workspace_root)
        self.store = MemoryStore(workspace_root) if workspace_root is not None else None

    def to_dict(self) -> dict[str, Any]:
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    def canonical_path(self, path: str) -> str:
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary: str) -> "MemoryManager":
        self.state = set_task_summary(self.state, summary, self.workspace_root)
        return self

    def remember_file(self, path: str) -> "MemoryManager":
        self.state = remember_file(self.state, path, self.workspace_root)
        return self

    def append_note(
        self,
        text: str,
        tags: tuple[str, ...] | list[str] = (),
        source: str = "",
        created_at: str | None = None,
        kind: str = "episodic",
    ) -> "MemoryManager":
        self.state = append_note(
            self.state,
            text,
            tags=tags,
            source=source,
            created_at=created_at,
            workspace_root=self.workspace_root,
            kind=kind,
        )
        return self

    def remember_user_facts(self, text: str) -> list[str]:
        remembered = []
        for note in extract_user_fact_lines(text):
            self.append_note(
                note["text"],
                tags=tuple(note["tags"]),
                source=note["source"],
                kind=note["kind"],
            )
            remembered.append(note["text"])
        return remembered

    def set_file_summary(self, path: str, summary: str) -> "MemoryManager":
        self.state = set_file_summary(self.state, path, summary, self.workspace_root)
        return self

    def invalidate_file_summary(self, path: str) -> "MemoryManager":
        self.state = invalidate_file_summary(self.state, path, self.workspace_root)
        return self

    def invalidate_stale_file_summaries(self) -> list[str]:
        self.state, invalidated = invalidate_stale_file_summaries(self.state, self.workspace_root)
        return invalidated

    def retrieval_candidates(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        return retrieval_candidates(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def retrieval_view(self, query: str, limit: int = 3) -> str:
        return retrieval_view(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def render_memory_text(self) -> str:
        return render_memory_text(self.state, self.workspace_root)

    def bootstrap_text(self) -> str:
        if self.store is None:
            return ""
        return self.store.bootstrap_context()


LayeredMemory = MemoryManager
HistoryArchiveStore = MemoryStore
