"""Runtime checkpoint helpers for interrupted pico turns."""

from __future__ import annotations

import uuid
from typing import Any

from .workspace import now

CHECKPOINT_NONE_STATUS = "no-checkpoint"
RUNTIME_CHECKPOINT_RESTORED_STATUS = "runtime-checkpoint-restored"
PENDING_USER_TURN_RESTORED_STATUS = "pending-user-turn-restored"
RUNTIME_CHECKPOINT_METADATA_KEY = "runtime_checkpoint"
PENDING_USER_TURN_METADATA_KEY = "pending_user_turn"
PENDING_TOOL_RESULT_CONTENT = "Error: Task interrupted before this tool finished."
PENDING_USER_TURN_CONTENT = "Error: Task interrupted before a response was generated."


def new_checkpoint_id() -> str:
    return "ckpt_" + uuid.uuid4().hex[:8]


def set_runtime_checkpoint(session: dict[str, Any], payload: dict[str, object]) -> str:
    checkpoint_id = new_checkpoint_id()
    metadata = session.setdefault("session_metadata", {})
    metadata[RUNTIME_CHECKPOINT_METADATA_KEY] = {
        "checkpoint_id": checkpoint_id,
        **dict(payload or {}),
    }
    return checkpoint_id


def mark_pending_user_turn(session: dict[str, Any]) -> None:
    session.setdefault("session_metadata", {})[PENDING_USER_TURN_METADATA_KEY] = True


def clear_pending_user_turn(session: dict[str, Any]) -> bool:
    metadata = session.setdefault("session_metadata", {})
    if PENDING_USER_TURN_METADATA_KEY not in metadata:
        return False
    metadata.pop(PENDING_USER_TURN_METADATA_KEY, None)
    return True


def clear_runtime_checkpoint(session: dict[str, Any]) -> bool:
    metadata = session.setdefault("session_metadata", {})
    if RUNTIME_CHECKPOINT_METADATA_KEY not in metadata:
        return False
    metadata.pop(RUNTIME_CHECKPOINT_METADATA_KEY, None)
    return True


def checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
    return (
        message.get("role"),
        message.get("content"),
        message.get("tool_call_id"),
        message.get("name"),
        message.get("args"),
    )


def materialize_runtime_checkpoint(session: dict[str, Any]) -> bool:
    metadata = session.setdefault("session_metadata", {})
    checkpoint = metadata.get(RUNTIME_CHECKPOINT_METADATA_KEY)
    if not isinstance(checkpoint, dict):
        return False

    restored_messages: list[dict[str, Any]] = []
    assistant_message = checkpoint.get("assistant_message")
    if isinstance(assistant_message, dict):
        restored = dict(assistant_message)
        restored.setdefault("created_at", now())
        restored_messages.append(restored)

    completed_tool_results = checkpoint.get("completed_tool_results") or []
    for message in completed_tool_results:
        if isinstance(message, dict):
            restored = dict(message)
            restored.setdefault("created_at", now())
            restored_messages.append(restored)

    pending_tool_calls = checkpoint.get("pending_tool_calls") or []
    for tool_call in pending_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_id = tool_call.get("id")
        function = dict(tool_call.get("function", {}) or {})
        restored_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": str(function.get("name", "")).strip() or "tool",
                "args": function.get("arguments", {}),
                "content": PENDING_TOOL_RESULT_CONTENT,
                "created_at": now(),
            }
        )

    history = session.setdefault("history", [])
    overlap = 0
    max_overlap = min(len(history), len(restored_messages))
    for size in range(max_overlap, 0, -1):
        existing = history[-size:]
        restored = restored_messages[:size]
        if all(
            checkpoint_message_key(left) == checkpoint_message_key(right)
            for left, right in zip(existing, restored)
        ):
            overlap = size
            break

    if restored_messages[overlap:]:
        history.extend(restored_messages[overlap:])
        session["updated_at"] = now()

    clear_pending_user_turn(session)
    clear_runtime_checkpoint(session)
    return True


def materialize_pending_user_turn(session: dict[str, Any]) -> bool:
    metadata = session.setdefault("session_metadata", {})
    if not metadata.get(PENDING_USER_TURN_METADATA_KEY):
        return False

    history = session.setdefault("history", [])
    if history and history[-1].get("role") == "user":
        history.append(
            {
                "role": "assistant",
                "content": PENDING_USER_TURN_CONTENT,
                "created_at": now(),
            }
        )
        session["updated_at"] = now()

    clear_pending_user_turn(session)
    return True


def restore_interrupted_turn(session: dict[str, Any]) -> dict[str, str]:
    if materialize_runtime_checkpoint(session):
        return {"status": RUNTIME_CHECKPOINT_RESTORED_STATUS}
    if materialize_pending_user_turn(session):
        return {"status": PENDING_USER_TURN_RESTORED_STATUS}
    return {"status": CHECKPOINT_NONE_STATUS}


def legacy_checkpoint(session: dict[str, Any]) -> dict[str, Any] | None:
    checkpoints = session.get("checkpoints")
    if not isinstance(checkpoints, dict):
        return None
    current_id = str(checkpoints.get("current_id", "")).strip()
    items = checkpoints.get("items")
    if not current_id or not isinstance(items, dict):
        return None
    item = items.get(current_id)
    return dict(item) if isinstance(item, dict) else None
