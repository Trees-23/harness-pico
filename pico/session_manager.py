"""Nanobot-style JSONL session persistence for pico."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .workspace import now

SESSION_HISTORY_FILE_NAME = "cli_direct.jsonl"
LEGACY_SESSION_HISTORY_FILE_NAME = "history.jsonl"


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now()
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return datetime.now()


def _find_legal_message_start(messages: list[dict[str, Any]]) -> int:
    for index, message in enumerate(messages):
        if message.get("role") in {"user", "assistant"}:
            return index
    return 0


@dataclass
class Session:
    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Session":
        session_id = str(payload.get("id", "")).strip()
        history = list(payload.get("history", []))
        metadata = dict(payload.get("session_metadata", {}) or {})
        if "_last_summary" in payload:
            metadata["_last_summary"] = payload["_last_summary"]
        return cls(
            key=session_id,
            messages=history,
            created_at=_parse_datetime(payload.get("created_at")),
            updated_at=_parse_datetime(payload.get("updated_at")),
            metadata=metadata,
            last_consolidated=max(0, int(payload.get("last_consolidated", 0) or 0)),
        )

    def apply_to_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["history"] = list(self.messages)
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        payload["last_consolidated"] = max(0, int(self.last_consolidated))
        payload["session_metadata"] = dict(self.metadata)
        if "_last_summary" in self.metadata:
            payload["_last_summary"] = dict(self.metadata["_last_summary"])
        else:
            payload.pop("_last_summary", None)
        return payload

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        message = {
            "role": role,
            "content": content,
            "created_at": kwargs.pop("created_at", now()),
            **kwargs,
        }
        self.messages.append(message)
        self.updated_at = datetime.now()

    def clear(self) -> None:
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        unconsolidated = list(self.messages[self.last_consolidated :])
        if max_messages and max_messages > 0:
            sliced = unconsolidated[-max_messages:]
        else:
            sliced = unconsolidated

        for index, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[index:]
                break

        start = _find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]
        return sliced

    def retain_recent_legal_suffix(self, max_messages: int) -> None:
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return

        start_idx = max(0, len(self.messages) - max_messages)
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        retained = list(self.messages[start_idx:])
        start = _find_legal_message_start(retained)
        if start:
            retained = retained[start:]

        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.last_consolidated = max(0, self.last_consolidated - dropped)
        self.updated_at = datetime.now()


class SessionManager:
    HISTORY_FILE_NAME = SESSION_HISTORY_FILE_NAME
    LEGACY_HISTORY_FILE_NAME = LEGACY_SESSION_HISTORY_FILE_NAME

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    def session_dir(self, session_id: str) -> Path:
        return self.root / str(session_id)

    def history_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / self.HISTORY_FILE_NAME

    def _get_session_path(self, key: str) -> Path:
        return self.history_path(key)

    def legacy_history_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / self.LEGACY_HISTORY_FILE_NAME

    def save(self, session_payload: dict[str, Any]) -> Path:
        session_id = str(session_payload["id"])
        record = Session.from_payload(session_payload)
        self.save_record(record)

        legacy_history_path = self.legacy_history_path(session_id)
        if legacy_history_path.exists() and legacy_history_path != self.history_path(session_id):
            try:
                legacy_history_path.unlink()
            except OSError:
                pass
        return self.history_path(session_id)

    def save_record(self, session: Session) -> Path:
        path = self.history_path(session.key)
        self._write_jsonl_atomic(path, session)
        self._cache[session.key] = session
        return path

    def load(self, session_id: str) -> dict[str, Any]:
        record = self._load_record(session_id)
        if record is not None:
            return record.apply_to_payload({"id": str(session_id)})
        raise FileNotFoundError(f"session not found: {session_id}")

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]
        session = self._load_record(key) or Session(key=key)
        self._cache[key] = session
        return session

    def invalidate(self, key: str) -> None:
        self._cache.pop(str(key), None)

    def delete_session(self, key: str) -> bool:
        self.invalidate(key)
        removed = False
        for path in (self.history_path(key), self.legacy_history_path(key)):
            if not path.exists():
                continue
            try:
                path.unlink()
                removed = True
            except OSError:
                pass
        return removed

    def latest(self) -> str | None:
        candidates = list(self.root.glob(f"*/{self.HISTORY_FILE_NAME}"))
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime)
        latest = candidates[-1]
        return latest.parent.name

    def live_history(self, session_payload: dict[str, Any], max_messages: int = 0) -> list[dict[str, Any]]:
        record = Session.from_payload(session_payload)
        return record.get_history(max_messages=max_messages)

    def append_message(self, session_payload: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        record = Session.from_payload(session_payload)
        message = dict(item)
        if not str(message.get("created_at", "")).strip():
            message["created_at"] = now()
        record.messages.append(message)
        record.updated_at = datetime.now()
        record.apply_to_payload(session_payload)
        return session_payload

    @staticmethod
    def _write_jsonl_atomic(path: Path, record: Session) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "_type": "metadata",
            "key": record.key,
            "session_id": record.key,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "last_consolidated": int(record.last_consolidated),
            "metadata": dict(record.metadata),
        }
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            for row in record.messages:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            temp_name = handle.name
        os.replace(temp_name, path)

    def _load_record(self, session_id: str) -> Session | None:
        history_path = self.history_path(session_id)
        if not history_path.exists():
            history_path = self.legacy_history_path(session_id)
        if not history_path.exists():
            return None

        messages: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}
        created_at: datetime | None = None
        updated_at: datetime | None = None
        last_consolidated = 0

        try:
            with history_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if payload.get("_type") == "metadata":
                        metadata = dict(payload.get("metadata", {}) or {})
                        created_at = _parse_datetime(payload.get("created_at"))
                        updated_at = _parse_datetime(payload.get("updated_at"))
                        last_consolidated = max(0, int(payload.get("last_consolidated", 0) or 0))
                        continue
                    messages.append(payload)
        except Exception:
            return self._repair(session_id)

        return Session(
            key=str(session_id),
            messages=messages,
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=metadata,
            last_consolidated=last_consolidated,
        )

    def _repair(self, session_id: str) -> Session | None:
        history_path = self.history_path(session_id)
        if not history_path.exists():
            history_path = self.legacy_history_path(session_id)
        if not history_path.exists():
            return None

        messages: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}
        created_at: datetime | None = None
        updated_at: datetime | None = None
        last_consolidated = 0
        try:
            with history_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("_type") == "metadata":
                        metadata = dict(payload.get("metadata", {}) or {})
                        created_at = _parse_datetime(payload.get("created_at"))
                        updated_at = _parse_datetime(payload.get("updated_at"))
                        try:
                            last_consolidated = max(0, int(payload.get("last_consolidated", 0) or 0))
                        except Exception:
                            last_consolidated = 0
                        continue
                    messages.append(payload)
        except OSError:
            return None
        if not messages and not metadata:
            return None
        return Session(
            key=str(session_id),
            messages=messages,
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=metadata,
            last_consolidated=last_consolidated,
        )

    @staticmethod
    def _session_payload(session: Session) -> dict[str, Any]:
        return {
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": dict(session.metadata),
            "messages": list(session.messages),
            "last_consolidated": int(session.last_consolidated),
        }

    def read_session_file(self, key: str) -> dict[str, Any] | None:
        record = self._load_record(key)
        return self._session_payload(record) if record is not None else None

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self.root.glob(f"*/{self.HISTORY_FILE_NAME}"):
            key = path.parent.name
            record = self._load_record(key)
            if record is None:
                continue
            sessions.append(
                {
                    "key": record.key,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                    "path": str(path),
                }
            )
        return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)


SessionStore = SessionManager
