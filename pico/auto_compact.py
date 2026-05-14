"""Nanobot-style idle auto compaction for pico."""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from datetime import datetime
from typing import Any

from .session_manager import Session


class AutoCompact:
    _RECENT_SUFFIX_MESSAGES = 8

    def __init__(self, sessions, consolidator, session_ttl_minutes: int = 0):
        self.sessions = sessions
        self.consolidator = consolidator
        self._ttl = int(session_ttl_minutes)
        self._archiving: set[str] = set()
        self._summaries: dict[str, tuple[str, datetime]] = {}

    def _is_expired(self, ts: datetime | str | None, now: datetime | None = None) -> bool:
        if self._ttl <= 0 or not ts:
            return False
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        current = now
        if current is None:
            current = datetime.now(ts.tzinfo) if ts.tzinfo is not None else datetime.now()
        elif ts.tzinfo is not None and current.tzinfo is None:
            current = current.replace(tzinfo=ts.tzinfo)
        return (current - ts).total_seconds() >= self._ttl * 60

    @staticmethod
    def _format_summary(text: str, last_active: datetime) -> str:
        current = datetime.now(last_active.tzinfo) if last_active.tzinfo is not None else datetime.now()
        idle_minutes = int((current - last_active).total_seconds() / 60)
        return f"Inactive for {idle_minutes} minutes.\nPrevious conversation summary: {text}"

    def _split_unconsolidated(self, session_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        record = Session.from_payload(session_payload)
        tail = list(record.messages[record.last_consolidated :])
        if not tail:
            return [], []
        probe = Session(
            key=record.key,
            messages=tail.copy(),
            created_at=record.created_at,
            updated_at=record.updated_at,
            metadata={},
            last_consolidated=0,
        )
        probe.retain_recent_legal_suffix(self._RECENT_SUFFIX_MESSAGES)
        kept = probe.messages
        cut = len(tail) - len(kept)
        return tail[:cut], kept

    def check_expired(self, schedule_background, active_session_keys: Collection[str] = ()) -> None:
        now = datetime.now()
        for info in self.sessions.list_sessions():
            key = str(info.get("key", "")).strip()
            if not key or key in self._archiving or key in active_session_keys:
                continue
            if not self._is_expired(info.get("updated_at"), now):
                continue
            self._archiving.add(key)
            schedule_background(self._archive(key))

    async def _archive(self, key: str) -> None:
        try:
            self.sessions.invalidate(key)
            session = self.sessions.get_or_create(key)
            payload = {
                "id": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "history": list(session.messages),
                "last_consolidated": int(session.last_consolidated),
                "session_metadata": dict(session.metadata),
                "history_archive": {},
            }
            self._archive_session(payload, key)
            session.messages = list(payload.get("history", []))
            session.last_consolidated = int(payload.get("last_consolidated", 0) or 0)
            session.updated_at = datetime.fromisoformat(str(payload.get("updated_at") or datetime.now().isoformat()))
            session.metadata = dict(payload.get("session_metadata", {}) or {})
            self.sessions.save_record(session)
        finally:
            self._archiving.discard(key)

    def _archive_session(self, session_payload: dict[str, Any], key: str) -> str | None:
        archive_msgs, kept_msgs = self._split_unconsolidated(session_payload)
        if not archive_msgs and not kept_msgs:
            session_payload["updated_at"] = datetime.now().isoformat()
            return None

        last_active = datetime.fromisoformat(str(session_payload.get("updated_at") or datetime.now().isoformat()))
        summary = ""
        if archive_msgs:
            _, archived_summary = self.consolidator.archive(
                archive_msgs,
                existing_summary=str(session_payload.get("history_archive", {}).get("latest_summary", "")).strip(),
            )
            summary = archived_summary or ""
        if summary and summary != "(nothing)":
            self._summaries[key] = (summary, last_active)
            session_payload.setdefault("session_metadata", {})["_last_summary"] = {
                "text": summary,
                "last_active": last_active.isoformat(),
            }
            archive_state = session_payload.setdefault("history_archive", {})
            archive_state["latest_summary"] = summary
            archive_state["compaction_count"] = int(archive_state.get("compaction_count", 0)) + 1
            archive_state["archived_messages"] = int(archive_state.get("archived_messages", 0)) + len(archive_msgs)
            archive_state["last_compacted_at"] = datetime.now().isoformat()
        session_payload["history"] = kept_msgs
        session_payload["last_consolidated"] = 0
        session_payload["updated_at"] = datetime.now().isoformat()
        return summary or None

    def prepare_session(self, session_payload: dict[str, Any], key: str) -> tuple[dict[str, Any], str | None]:
        if not key:
            return session_payload, None
        if key not in self._archiving and self._is_expired(session_payload.get("updated_at")):
            self._archiving.add(key)
            try:
                self._archive_session(session_payload, key)
            finally:
                self._archiving.discard(key)

        entry = self._summaries.pop(key, None)
        if entry:
            session_payload.setdefault("session_metadata", {}).pop("_last_summary", None)
            return session_payload, self._format_summary(entry[0], entry[1])

        metadata = session_payload.setdefault("session_metadata", {})
        if "_last_summary" in metadata:
            meta = dict(metadata.pop("_last_summary"))
            last_active = datetime.fromisoformat(str(meta.get("last_active") or datetime.now().isoformat()))
            return session_payload, self._format_summary(str(meta.get("text", "")).strip(), last_active)
        return session_payload, None
