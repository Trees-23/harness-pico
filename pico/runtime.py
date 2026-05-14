"""Compatibility shim for the Pico runtime public API."""

from __future__ import annotations

from . import checkpoint as checkpointlib
from .loop import (
    DEFAULT_FEATURE_FLAGS,
    DEFAULT_SHELL_ENV_ALLOWLIST,
    REDACTED_VALUE,
    SENSITIVE_ENV_NAME_MARKERS,
    PicoLoop,
)
from .session_manager import SessionManager, SessionStore

CHECKPOINT_NONE_STATUS = checkpointlib.CHECKPOINT_NONE_STATUS
RUNTIME_CHECKPOINT_RESTORED_STATUS = checkpointlib.RUNTIME_CHECKPOINT_RESTORED_STATUS
PENDING_USER_TURN_RESTORED_STATUS = checkpointlib.PENDING_USER_TURN_RESTORED_STATUS
RUNTIME_CHECKPOINT_METADATA_KEY = checkpointlib.RUNTIME_CHECKPOINT_METADATA_KEY
PENDING_USER_TURN_METADATA_KEY = checkpointlib.PENDING_USER_TURN_METADATA_KEY

Pico = PicoLoop
MiniAgent = PicoLoop

for _name in (
    "tool_list_files",
    "tool_read_file",
    "tool_search",
    "tool_run_shell",
    "tool_write_file",
    "tool_patch_file",
    "tool_delegate",
):
    getattr(PicoLoop, _name).__module__ = __name__

__all__ = [
    "CHECKPOINT_NONE_STATUS",
    "DEFAULT_FEATURE_FLAGS",
    "DEFAULT_SHELL_ENV_ALLOWLIST",
    "MiniAgent",
    "PENDING_USER_TURN_METADATA_KEY",
    "PENDING_USER_TURN_RESTORED_STATUS",
    "Pico",
    "PicoLoop",
    "REDACTED_VALUE",
    "RUNTIME_CHECKPOINT_METADATA_KEY",
    "RUNTIME_CHECKPOINT_RESTORED_STATUS",
    "SENSITIVE_ENV_NAME_MARKERS",
    "SessionManager",
    "SessionStore",
]
