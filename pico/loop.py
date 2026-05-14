"""Agent 运行时核心逻辑。

Pico 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import json
import os
import re
import time
import uuid
import hashlib
from datetime import datetime
from pathlib import Path

from . import auto_compact as auto_compactlib
from . import checkpoint as checkpointlib
from . import dream as dreamlib
from . import memory as memorylib
from . import consolidator as consolidatorlib
from .context_manager import ContextBuilder
from .cron import CronJob, CronPayload, CronSchedule, CronService
from .runner import PicoRunner, PicoRunSpec
from .run_store import RunStore
from .task_state import TaskState
from .session_manager import SessionManager, SessionStore
from . import tools as toolkit
from .tooling import capabilities as tool_capabilities
from .tooling import ToolExecutor, build_standard_tool_registry
from .workspace import IGNORED_PATH_NAMES, clip, now

SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
REDACTED_VALUE = "<redacted>"
DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "relevant_memory": False,
    "context_reduction": True,
    "prompt_cache": True,
    "history_compaction": True,
}
CHECKPOINT_NONE_STATUS = checkpointlib.CHECKPOINT_NONE_STATUS
RUNTIME_CHECKPOINT_RESTORED_STATUS = checkpointlib.RUNTIME_CHECKPOINT_RESTORED_STATUS
PENDING_USER_TURN_RESTORED_STATUS = checkpointlib.PENDING_USER_TURN_RESTORED_STATUS
RUNTIME_CHECKPOINT_METADATA_KEY = checkpointlib.RUNTIME_CHECKPOINT_METADATA_KEY
PENDING_USER_TURN_METADATA_KEY = checkpointlib.PENDING_USER_TURN_METADATA_KEY
SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")


def _image_placeholder_text(path: str) -> str:
    path = str(path or "").strip()
    return f"[Image omitted from saved session: {path}]" if path else "[Image omitted from saved session]"


class PicoLoop:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
        progress_callback=None,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_manager = (
            session_store
            if isinstance(session_store, SessionManager)
            else SessionManager(session_store.root)
        )
        self.session_store = self.session_manager
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.progress_callback = progress_callback
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / ".pico" / "runs")
        self.memory_store = memorylib.MemoryStore(Path(workspace.repo_root))
        context_window_tokens = consolidatorlib.CONTEXT_WINDOW_TOKENS
        self.history_consolidator = consolidatorlib.Consolidator(
            self.memory_store,
            lambda prompt, max_new_tokens: self.model_client.complete(prompt, max_new_tokens),
            context_window_tokens=context_window_tokens,
            max_completion_tokens=min(self.max_new_tokens, 256),
        )
        self.auto_compactor = auto_compactlib.AutoCompact(self.session_manager, self.history_consolidator)
        self.dream = dreamlib.Dream(
            self.memory_store,
            self.model_client,
            max_new_tokens=min(self.max_new_tokens, 512),
        )
        self.cron = CronService(Path(workspace.repo_root) / ".pico" / "cron" / "jobs.json", on_job=self._on_cron_job)
        self._register_dream_cron_job()
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "updated_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "last_consolidated": 0,
            "session_metadata": {},
            "memory": memorylib.default_memory_state(),
            "history_archive": memorylib.default_history_archive_state(),
        }
        self._legacy_checkpoint = checkpointlib.legacy_checkpoint(self.session)
        self._ensure_session_shape()
        self.memory = memorylib.MemoryManager(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.tool_registry = build_standard_tool_registry(self)
        self.tools = self.build_tools()
        self.tool_executor = ToolExecutor(self, self.tool_registry)
        self.runner = PicoRunner(self)
        self.context_builder = ContextBuilder(self)
        self.context_manager = self.context_builder
        self.resume_state = {"status": CHECKPOINT_NONE_STATUS}
        self.session_path = self.session_store.save(self.session)
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_completion_metadata = {}
        self._prepared_session_summary = ""
        self._last_tool_result_metadata = {}
        self._runtime_identity_mismatch_fields = []
        self._stale_checkpoint_paths = []

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _ensure_session_shape(self):
        self.session.setdefault("history", [])
        self.session.setdefault("updated_at", self.session.get("created_at", now()))
        self.session.setdefault("last_consolidated", 0)
        session_metadata = self.session.setdefault("session_metadata", {})
        if not isinstance(session_metadata, dict):
            session_metadata = {}
            self.session["session_metadata"] = session_metadata
        self.session.setdefault("memory", memorylib.default_memory_state())
        history_archive = self.session.setdefault("history_archive", memorylib.default_history_archive_state())
        if not isinstance(history_archive, dict):
            history_archive = memorylib.default_history_archive_state()
            self.session["history_archive"] = history_archive
        for key, value in memorylib.default_history_archive_state().items():
            history_archive.setdefault(key, value)
        self.session.pop("checkpoints", None)
        self.session.pop("runtime_identity", None)
        self.session.pop("resume_state", None)

    def history_archive_state(self):
        self._ensure_session_shape()
        return self.session["history_archive"]

    def archived_history_summary(self):
        prepared = str(getattr(self, "_prepared_session_summary", "")).strip()
        if prepared:
            return prepared
        return str(self.history_archive_state().get("latest_summary", "")).strip()

    def compact_history_if_needed(self, user_message=""):
        if not self.feature_enabled("history_compaction"):
            return None
        result = self.history_consolidator.maybe_consolidate_by_tokens(
            self.session,
            session_summary=str(self.history_archive_state().get("latest_summary", "")).strip(),
            task_summary=self.memory.to_dict().get("working", {}).get("task_summary", "") or user_message,
        )
        if result:
            archive_state = self.history_archive_state()
            archive_state["latest_summary"] = result.summary
            archive_state["latest_cursor"] = int(result.cursor)
            archive_state["compaction_count"] = int(archive_state.get("compaction_count", 0)) + 1
            archive_state["archived_messages"] = int(archive_state.get("archived_messages", 0)) + int(result.archived_messages)
            archive_state["last_compacted_at"] = now()
            if result.summary:
                self.session.setdefault("session_metadata", {})["_last_summary"] = {
                    "text": result.summary,
                    "last_active": str(self.session.get("updated_at", "")).strip() or now(),
                }
            self._prepared_session_summary = ""
            self.session_path = self.session_store.save(self.session)
        return result

    def prepare_session_for_turn(self) -> str:
        self.session, prepared_summary = self.auto_compactor.prepare_session(
            self.session,
            str(self.session.get("id", "")).strip(),
        )
        self._prepared_session_summary = prepared_summary or ""
        return self._prepared_session_summary

    def record_user_turn(self, user_message) -> None:
        self.memory.set_task_summary(user_message)
        self._evaluate_legacy_checkpoint()
        self.record({"role": "user", "content": user_message, "created_at": now()})
        self._mark_pending_user_turn()
        self.remember_user_message(user_message)

    def clear_pending_user_turn_for_runner(self) -> None:
        self._clear_pending_user_turn()

    def clear_runtime_checkpoint_for_runner(self) -> None:
        self._clear_runtime_checkpoint()

    def create_checkpoint_for_runner(self, task_state, user_message, trigger, **kwargs):
        return self.create_checkpoint(task_state, user_message, trigger, **kwargs)

    def write_task_state_for_runner(self, task_state) -> None:
        self.run_store.write_task_state(task_state)

    def emit_trace_for_runner(self, task_state, event, payload=None):
        return self.emit_trace(task_state, event, payload)

    def emit_progress_for_runner(self, event, payload=None) -> None:
        self.emit_progress(event, payload)

    def write_report_for_runner(self, task_state) -> None:
        self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))

    @staticmethod
    def _should_finish_after_repeated_tool(name: str, user_message: str) -> bool:
        if name in {"write_file", "edit_file", "patch_file", "notebook_edit"}:
            return True
        if name != "read_file":
            return False
        text = str(user_message or "").lower()
        read_markers = ("读", "读取", "查看", "打开", "read", "show", "display", "cat")
        recovery_markers = ("recover", "recovery", "修复", "恢复", "still finish", "finish the")
        return any(marker in text for marker in read_markers) and not any(
            marker in text for marker in recovery_markers
        )

    def is_repeated_tool_result_for_runner(self, name: str, result, user_message: str) -> bool:
        return str(result or "").startswith("error: repeated ") and self._should_finish_after_repeated_tool(
            name,
            user_message,
        )

    @staticmethod
    def _successful_tool_summary(tool_message: dict[str, object]) -> str:
        name = str(tool_message.get("name") or "tool")
        args = tool_message.get("args") if isinstance(tool_message.get("args"), dict) else {}
        content = str(tool_message.get("content") or "").strip()
        path = str(args.get("path") or "").strip()
        if name == "write_file":
            return f"Done. Wrote {path}." if path else "Done. Wrote the requested file."
        if name in {"edit_file", "patch_file"}:
            return f"Done. Patched {path}." if path else "Done. Patched the requested file."
        if name == "read_file":
            return content or "Done. Read the requested file."
        return content or f"Done. Completed {name}."

    def _last_successful_tool_message(self) -> dict[str, object] | None:
        for message in reversed(self.session.get("history", [])):
            if message.get("role") != "tool":
                continue
            content = str(message.get("content", ""))
            if content.startswith("error:"):
                continue
            return message
        return None

    def finish_with_repeated_tool_fallback_for_runner(
        self,
        task_state,
        *,
        run_started_at: float,
        repeated_result: str,
    ) -> str | None:
        previous = self._last_successful_tool_message()
        if previous is None:
            return None
        final = self._successful_tool_summary(previous)
        self.record({"role": "assistant", "content": final, "created_at": now()})
        self.clear_pending_user_turn_for_runner()
        self.clear_runtime_checkpoint_for_runner()
        task_state.finish_success(final)
        self.write_task_state_for_runner(task_state)
        self.emit_trace_for_runner(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "fallback_reason": "repeated_tool_call",
                "repeated_tool_result": clip(repeated_result, 500),
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        self.write_report_for_runner(task_state)
        self._prepared_session_summary = ""
        return final

    def start_task_run_for_turn(self, user_message):
        task_state = TaskState.create(
            run_id=self.new_run_id(),
            task_id=self.new_task_id(),
            user_request=user_message,
        )
        task_state.resume_status = self.resume_state.get("status", "no-checkpoint")
        self.current_task_state = task_state
        self.current_run_dir = self.run_store.start_run(task_state)
        self.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )
        self._emit_legacy_checkpoint_events(task_state)
        return task_state

    def run_history_compaction_for_turn(self, task_state, user_message):
        compaction = None
        try:
            compaction = self.compact_history_if_needed(user_message)
        except Exception as exc:
            self.emit_trace(
                task_state,
                "history_compaction_failed",
                {
                    "error": clip(str(exc), 300),
                },
            )
            return None
        if compaction:
            self.emit_trace(
                task_state,
                "history_compacted",
                {
                    "cursor": compaction.cursor,
                    "archived_messages": compaction.archived_messages,
                    "last_consolidated": compaction.last_consolidated,
                    "summary": clip(compaction.summary, 300),
                },
            )
        return compaction

    async def _on_cron_job(self, job: CronJob) -> str | None:
        if job.name == "dream":
            if not self.feature_enabled("memory") or not bool(self.feature_flags.get("dream", True)):
                return "dream skipped"
            self.dream.run()
            return "dream completed"
        return None

    def _register_dream_cron_job(self) -> None:
        self.cron.register_system_job(
            CronJob(
                id="dream",
                name="dream",
                enabled=bool(self.feature_flags.get("dream", True)),
                schedule=CronSchedule(kind="every", every_ms=max(60_000, int(self.dream.interval_seconds) * 1000)),
                payload=CronPayload(kind="system_event", message="dream"),
            )
        )

    def invalidate_stale_memory(self):
        invalidated = self.memory.invalidate_stale_file_summaries()
        self.session["memory"] = self.memory.to_dict()
        return invalidated

    def _set_runtime_checkpoint(self, payload: dict[str, object]) -> str:
        checkpoint_id = checkpointlib.set_runtime_checkpoint(self.session, payload)
        self.session_path = self.session_store.save(self.session)
        return checkpoint_id

    def _mark_pending_user_turn(self) -> None:
        checkpointlib.mark_pending_user_turn(self.session)
        self.session_path = self.session_store.save(self.session)

    def _clear_pending_user_turn(self) -> None:
        if checkpointlib.clear_pending_user_turn(self.session):
            self.session_path = self.session_store.save(self.session)

    def _clear_runtime_checkpoint(self) -> None:
        if checkpointlib.clear_runtime_checkpoint(self.session):
            self.session_path = self.session_store.save(self.session)

    @staticmethod
    def _checkpoint_message_key(message):
        return checkpointlib.checkpoint_message_key(message)

    def _restore_runtime_checkpoint(self) -> bool:
        if not checkpointlib.materialize_runtime_checkpoint(self.session):
            return False
        self.resume_state = {"status": RUNTIME_CHECKPOINT_RESTORED_STATUS}
        return True

    def _restore_pending_user_turn(self) -> bool:
        if not checkpointlib.materialize_pending_user_turn(self.session):
            return False
        self.resume_state = {"status": PENDING_USER_TURN_RESTORED_STATUS}
        return True

    def _restore_interrupted_turn_if_needed(self) -> dict[str, str]:
        if self._restore_runtime_checkpoint():
            self.session_path = self.session_store.save(self.session)
            return self.resume_state
        if self._restore_pending_user_turn():
            self.session_path = self.session_store.save(self.session)
            return self.resume_state
        self.resume_state = {"status": CHECKPOINT_NONE_STATUS}
        return self.resume_state

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self):
        return self._legacy_prompt_specs_from_registry()

    def build_tool_registry(self):
        return build_standard_tool_registry(self)

    def _legacy_prompt_specs_from_registry(self):
        specs = {}
        for definition in self.tool_registry.get_definitions():
            function = definition.get("function", {})
            name = function.get("name")
            if not name:
                continue
            tool = self.tool_registry.get(name)
            if tool is None:
                continue
            specs[name] = {
                "schema": function.get("parameters", {}),
                "risky": not tool.read_only,
                "description": function.get("description", ""),
            }
        return specs

    def tool_signature(self):
        payload = self.tool_registry.get_definitions()
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        prompt, _ = self.build_model_prompt(user_message)
        return prompt

    def record(self, item):
        sanitized = self._sanitize_message_for_persistence(item) if hasattr(self, "_sanitize_message_for_persistence") else item
        if sanitized is None:
            return
        self.session_manager.append_message(self.session, sanitized)
        self.session_path = self.session_store.save(self.session)

    @staticmethod
    def looks_sensitive_env_name(name):
        upper = str(name).upper()
        return any(upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}") for marker in SENSITIVE_ENV_NAME_MARKERS)

    def is_secret_env_name(self, name):
        upper = str(name).upper()
        return upper in self.secret_env_names or self.looks_sensitive_env_name(upper)

    def configured_secret_env_items(self):
        items = [
            (name, value)
            for name, value in os.environ.items()
            if str(name).upper() in self.secret_env_names and value
        ]
        items.sort(key=lambda item: item[0])
        return items

    def detected_secret_env_items(self):
        items = [
            (name, value)
            for name, value in os.environ.items()
            if self.is_secret_env_name(name) and value
        ]
        items.sort(key=lambda item: item[0])
        return items

    def secret_env_summary(self):
        names = [name for name, _ in self.configured_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def detected_secret_env_summary(self):
        names = [name for name, _ in self.detected_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def redact_text(self, text):
        text = str(text)
        for _, value in sorted(self.detected_secret_env_items(), key=lambda item: len(item[1]), reverse=True):
            text = text.replace(value, REDACTED_VALUE)
        return text

    def redact_artifact(self, value, key=None):
        if key and self.is_secret_env_name(key):
            return REDACTED_VALUE
        if isinstance(value, dict):
            return {
                str(item_key): self.redact_artifact(item_value, key=item_key)
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, tuple):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, str):
            redacted = self.redact_text(value)
            return redacted
        return value

    def shell_env(self):
        env = {
            name: os.environ[name]
            for name in self.shell_env_allowlist
            if name in os.environ
        }
        env["PWD"] = str(self.root)
        if "PATH" not in env and os.environ.get("PATH"):
            env["PATH"] = os.environ["PATH"]
        return env

    def prompt_metadata(self, user_message, prompt):
        _, metadata = self.build_model_prompt(user_message)
        return metadata

    def build_model_prompt(self, user_message):
        invalidated = self.invalidate_stale_memory()
        self._evaluate_legacy_checkpoint()
        prompt, metadata = self.context_manager.build(user_message)
        stable_system = self.context_builder.build_system_prompt(channel="cli")
        system_hash = hashlib.sha256(stable_system.encode("utf-8")).hexdigest()
        metadata.update(
            {
                "workspace_chars": len(self.workspace.text()),
                "memory_chars": len(self.memory.render_memory_text()),
                "request_chars": len(user_message),
                "tool_count": len(self.tools),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "system_hash": system_hash,
                "prompt_cache_key": system_hash,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "tool_signature": self.tool_signature(),
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": len(invalidated),
                "stale_paths": list(self._stale_checkpoint_paths or invalidated),
                "runtime_identity_mismatch_fields": list(self._runtime_identity_mismatch_fields),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        return prompt, metadata

    def build_context_messages_for_turn(self, user_message):
        history = self.session_manager.live_history(self.session, max_messages=0)
        history, reductions = self.context_builder._reduce_history_for_prompt(history)
        self._last_context_budget_reductions = reductions
        return self.context_builder.build_messages(
            history=history,
            current_message=str(user_message or ""),
            channel="cli",
            chat_id=str(self.session.get("id", "")),
            session_summary=self.archived_history_summary(),
        )

    def prompt_metadata_for_messages_for_runner(self, user_message, prompt, messages):
        invalidated = self.invalidate_stale_memory()
        self._evaluate_legacy_checkpoint()
        stable_system = self.context_builder.build_system_prompt(channel="cli")
        system_hash = hashlib.sha256(stable_system.encode("utf-8")).hexdigest()
        metadata = self.context_builder._metadata(
            prompt,
            messages,
            str(user_message or ""),
            relevant_notes=[],
        )
        metadata["budget_reductions"] = list(getattr(self, "_last_context_budget_reductions", []) or [])
        metadata.update(
            {
                "workspace_chars": len(self.workspace.text()),
                "memory_chars": len(self.memory.render_memory_text()),
                "request_chars": len(str(user_message or "")),
                "tool_count": len(self.tools),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "system_hash": system_hash,
                "prompt_cache_key": system_hash,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "tool_signature": self.tool_signature(),
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": len(invalidated),
                "stale_paths": list(self._stale_checkpoint_paths or invalidated),
                "runtime_identity_mismatch_fields": list(self._runtime_identity_mismatch_fields),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        return metadata

    def _build_prompt_and_metadata(self, user_message):
        return self.build_model_prompt(user_message)

    def history_text(self, limit=4000):
        return clip(ContextBuilder.messages_to_prompt(self.session.get("history", [])), limit)

    def _evaluate_legacy_checkpoint(self):
        checkpoint = getattr(self, "_legacy_checkpoint", None)
        if not isinstance(checkpoint, dict):
            checkpoint = checkpointlib.legacy_checkpoint(self.session)
            self._legacy_checkpoint = checkpoint
        if not isinstance(checkpoint, dict):
            return
        runtime_identity = checkpoint.get("runtime_identity")
        if isinstance(runtime_identity, dict):
            expected_fingerprint = str(runtime_identity.get("workspace_fingerprint", "")).strip()
            if expected_fingerprint and expected_fingerprint != self.workspace.fingerprint():
                self.resume_state = {"status": "workspace-mismatch"}
                self._runtime_identity_mismatch_fields = ["workspace_fingerprint"]
                return
        stale_paths = []
        freshness = checkpoint.get("freshness")
        if isinstance(freshness, dict):
            for path, expected in freshness.items():
                current = memorylib.file_freshness(str(path), self.root)
                if expected != current:
                    stale_paths.append(str(path))
        if stale_paths:
            self.resume_state = {"status": "partial-stale"}
            self._stale_checkpoint_paths = stale_paths
        elif self.resume_state.get("status") == CHECKPOINT_NONE_STATUS:
            self.resume_state = {"status": "full-valid"}

    def _emit_legacy_checkpoint_events(self, task_state):
        status = str(self.resume_state.get("status", "")).strip()
        checkpoint = getattr(self, "_legacy_checkpoint", None)
        if not isinstance(checkpoint, dict):
            return
        if status == "partial-stale":
            payload = self.create_checkpoint(
                task_state,
                task_state.user_request,
                trigger="freshness_mismatch",
            )
            self.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": payload["checkpoint_id"],
                    "trigger": "freshness_mismatch",
                    "stale_paths": list(self._stale_checkpoint_paths),
                },
            )
        elif status == "workspace-mismatch":
            self.emit_trace(
                task_state,
                "runtime_identity_mismatch",
                {
                    "fields": list(self._runtime_identity_mismatch_fields),
                    "checkpoint_id": checkpoint.get("checkpoint_id", ""),
                },
            )

    def emit_trace(self, task_state, event, payload=None):
        payload = self.redact_artifact(payload or {})
        payload["event"] = event
        payload["created_at"] = now()
        # trace 是运行中的逐事件时间线，适合回答“这一轮 agent 到底做了什么”。
        self.run_store.append_trace(task_state, payload)
        return payload

    def emit_progress(self, event, payload=None):
        callback = getattr(self, "progress_callback", None)
        if not callable(callback):
            return
        callback(str(event), self.redact_artifact(payload or {}))

    def capture_workspace_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            if not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        changed_paths = []
        summaries = []
        all_paths = sorted(set(before) | set(after))
        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger, *, assistant_message=None, completed_tool_results=None, pending_tool_calls=None):
        checkpoint_id = self._set_runtime_checkpoint(
            {
                "trigger": str(trigger or ""),
                "created_at": now(),
                "user_message": str(user_message or ""),
                "assistant_message": dict(assistant_message or {}) if isinstance(assistant_message, dict) else None,
                "completed_tool_results": list(completed_tool_results or []),
                "pending_tool_calls": list(pending_tool_calls or []),
            }
        )
        task_state.checkpoint_id = checkpoint_id
        return {"checkpoint_id": checkpoint_id, "trigger": trigger}

    def infer_next_step(self, task_state):
        if task_state.status == "completed":
            return "No next step recorded."
        if task_state.stop_reason == "step_limit_reached":
            return "Resume from the latest checkpoint and continue the task."
        if task_state.last_tool:
            return f"Decide the next action after {task_state.last_tool}."
        return "Continue the task from the latest checkpoint."

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。完整结果已经进了
        `history`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整历史，再由这个函数择优沉淀成轻量记忆。
        """
        if not self.feature_enabled("memory"):
            return
        path = args.get("path")
        if not path:
            return

        canonical_path = self.memory.canonical_path(path)
        # 不是所有工具结果都进入工作记忆。
        # 读文件会生成摘要；写文件/patch 会让旧摘要失效，因为它们可能过期了。
        if name in {"read_file", "write_file", "edit_file", "patch_file"}:
            self.memory.remember_file(canonical_path)
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
            self.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
        elif name in {"write_file", "edit_file", "patch_file"}:
            self.memory.invalidate_file_summary(canonical_path)

    def note_tool(self, name, args, result):
        self.update_memory_after_tool(name, args, result)

    def remember_user_message(self, user_message):
        if not self.feature_enabled("memory"):
            return []
        remembered = self.memory.remember_user_facts(user_message)
        if remembered:
            self.session["memory"] = self.memory.to_dict()
            self.session_path = self.session_store.save(self.session)
        return remembered

    def record_durable_contract_from_final(self, final):
        text = str(final or "")
        task_summary = str(self.session.get("memory", {}).get("working", {}).get("task_summary", "")).lower()
        if "durable memory" not in task_summary or "benchmark" not in task_summary:
            return {}
        promotions = []
        rejections = []
        conventions = []
        for raw_line in text.splitlines():
            line = str(raw_line).strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith("project convention:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    promotions.append(f"project-conventions: {value}")
                    conventions.append(value)
            elif lowered.startswith("dependency:") and SECRET_SHAPED_TEXT_PATTERN.search(line):
                rejections.append("dependency-facts:secret_shaped")
            elif lowered.startswith("decision:"):
                value = line.split(":", 1)[1].strip()
                if "current goal" in value.lower() or "debug" in value.lower():
                    rejections.append("key-decisions:transient_task_state")
                elif value:
                    promotions.append(f"key-decisions: {value}")
        if conventions:
            topic_dir = self.root / ".pico" / "memory" / "topics"
            topic_dir.mkdir(parents=True, exist_ok=True)
            topic_path = topic_dir / "project-conventions.md"
            existing = topic_path.read_text(encoding="utf-8") if topic_path.exists() else "# Project Conventions\n"
            lines = [existing.rstrip()]
            for item in conventions:
                bullet = f"- {item}"
                if bullet not in existing:
                    lines.append(bullet)
            topic_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        result = {}
        if promotions:
            result["durable_promotions"] = promotions
        if rejections:
            result["durable_rejections"] = rejections
        return result

    def record_process_note_for_tool(self, name, metadata):
        status = str(metadata.get("tool_status", "")).strip()
        if status not in {"partial_success", "error", "rejected"}:
            return
        affected_paths = [str(path).strip() for path in metadata.get("affected_paths", []) if str(path).strip()]
        path_text = ", ".join(affected_paths) or "workspace"
        if status == "partial_success":
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            text = f"{name} rejected; choose a different action before retry"
        tags = ["process", status, *affected_paths]
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    def ask(self, user_message):
        """执行一次完整的 agent 回合，直到产出最终答案或命中停止条件。

        为什么存在：
        `ask()` 是整个 runtime 的总调度器。它把“用户提一个请求”扩展成一条
        可持续推进的控制循环：记录会话、组 prompt、调用模型、执行工具、
        写 trace/report、更新状态，直到模型给出最终答案或系统主动停下。

        输入 / 输出：
        - 输入：`user_message`，即用户这一次的任务描述
        - 输出：字符串形式的最终回答；如果中途达到步数上限或重试上限，
          返回的是一条停止原因说明

        在 agent 链路里的位置：
        它是 CLI 和底层工具/模型之间的核心桥梁。CLI 收到用户输入后基本只做
        一件事：调用 `agent.ask()`。而 `ask()` 内部再去驱动 `ContextManager`
        组 prompt、`model_client.complete()` 调模型、`run_tool()` 执行动作。
        如果新人想理解 pico 是怎么“从一句话跑成一个 agent 流程”的，
        这里就是最关键的入口。
        """
        self._restore_interrupted_turn_if_needed()
        self.prepare_session_for_turn()
        self.record_user_turn(user_message)
        task_state = self.start_task_run_for_turn(user_message)
        self.run_history_compaction_for_turn(task_state, user_message)
        result = self.runner.run(self.build_run_spec(user_message))
        self.run_history_compaction_for_turn(task_state, user_message)
        return result.final_content or ""

    def build_run_spec(self, user_message) -> PicoRunSpec:
        return PicoRunSpec(
            model_client=self.model_client,
            tool_registry=self.tool_registry,
            tool_executor=self.tool_executor,
            max_iterations=self.max_steps,
            max_new_tokens=self.max_new_tokens,
            initial_messages=self.build_context_messages_for_turn(user_message),
            max_tool_result_chars=getattr(self.runner, "max_tool_result_chars", 16_000),
            user_message=str(user_message or ""),
            concurrent_tools=True,
            fail_on_tool_error=getattr(self.runner, "fail_on_tool_error", False),
            workspace_root=self.root,
            session_id=str(self.session.get("id", "")),
        )

    def _sanitize_persisted_blocks(
        self,
        content,
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ):
        filtered = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue
            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue
            if block.get("type") == "image_url" and str(block.get("image_url", {}).get("url", "")).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": _image_placeholder_text(path)})
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > getattr(self.runner, "max_tool_result_chars", 16_000):
                    text = clip(text, getattr(self.runner, "max_tool_result_chars", 16_000))
                filtered.append({**block, "text": text})
                continue
            filtered.append(block)
        return filtered

    def _sanitize_message_for_persistence(self, message):
        entry = dict(message)
        role, content = entry.get("role"), entry.get("content")
        if role == "assistant" and not content and not entry.get("tool_calls"):
            return None
        if role == "tool":
            if (
                isinstance(content, str)
                and len(content) > getattr(self.runner, "max_tool_result_chars", 16_000)
                and not content.startswith("[Tool result too large;")
            ):
                entry["content"] = clip(content, getattr(self.runner, "max_tool_result_chars", 16_000))
            elif isinstance(content, list):
                filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                if not filtered:
                    return None
                entry["content"] = filtered
        elif role == "user":
            if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                end_marker = ContextBuilder._RUNTIME_CONTEXT_END
                end_pos = content.find(end_marker)
                if end_pos >= 0:
                    after = content[end_pos + len(end_marker):].lstrip("\n")
                    if not after:
                        return None
                    entry["content"] = after
                else:
                    after_tag = content[len(ContextBuilder._RUNTIME_CONTEXT_TAG):].lstrip("\n")
                    if not after_tag.strip():
                        return None
                    entry["content"] = after_tag
            elif isinstance(content, list):
                filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                if not filtered:
                    return None
                entry["content"] = filtered
        return entry

    def _save_turn(self, messages, skip: int = 0):
        for message in list(messages or [])[skip:]:
            entry = self._sanitize_message_for_persistence(message)
            if entry is None:
                continue
            entry.setdefault("created_at", now())
            self.record(entry)

    def run_tool(self, name, args):
        """Execute a tool through the standard Registry -> Executor path."""
        return self.tool_executor.execute(str(name), args or {})

    def repeated_tool_call(self, name, args):
        # agent 很常见的一种坏循环，是在没有新信息的情况下反复发起同一调用。
        # 这里提前挡掉最简单的这种循环。
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if name == "read_file" and self._repeated_read_file_call(args, tool_events):
            return True
        if name in {"write_file", "edit_file", "patch_file", "notebook_edit"}:
            for event in reversed(tool_events[-12:]):
                if event.get("name") != name or event.get("args") != args:
                    continue
                if str(event.get("content", "")).startswith("error:"):
                    continue
                return True
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    def _repeated_read_file_call(self, args, tool_events):
        current = self._read_file_call_signature(args)
        if current is None:
            return False
        current_path, current_start, current_end = current
        for event in reversed(tool_events[-12:]):
            if event.get("name") != "read_file":
                continue
            if str(event.get("content", "")).startswith("error:"):
                continue
            previous = self._read_file_call_signature(event.get("args", {}))
            if previous is None:
                continue
            previous_path, previous_start, previous_end = previous
            overlaps = previous_start <= current_end and current_start <= previous_end
            if previous_path == current_path and overlaps:
                return True
        return False

    def _read_file_call_signature(self, args):
        if not isinstance(args, dict) or "path" not in args:
            return None
        try:
            path = self.path(args["path"])
            start = int(args.get("start", 1))
            end = int(args.get("end", 200))
            if start < 1 or end < start:
                return None
            line_count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        except Exception:
            return None
        effective_end = min(end, max(line_count, start))
        return path.relative_to(self.root).as_posix(), start, effective_end

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        # report 是一次运行的最终摘要；
        # 和 trace 的区别在于，trace 关注过程，report 关注结果与关键指标。
        report = {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "redacted_env": self.detected_secret_env_summary(),
        }
        report.update(self.record_durable_contract_from_final(task_state.final_answer))
        return report

    def tool_example(self, name):
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """Compatibility validator backed by Registry.prepare_call()."""
        _, _, error = self.tool_registry.prepare_call(name, args or {})
        if error:
            if name == "delegate" and "not found" in error and self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")
            raise ValueError(error)

    def tool_list_files(self, args):
        return tool_capabilities.tool_list_files(self, args)

    def tool_read_file(self, args):
        return tool_capabilities.tool_read_file(self, args)

    def tool_search(self, args):
        return tool_capabilities.tool_search(self, args)

    def tool_run_shell(self, args):
        return tool_capabilities.tool_run_shell(self, args)

    def tool_write_file(self, args):
        return tool_capabilities.tool_write_file(self, args)

    def tool_patch_file(self, args):
        return tool_capabilities.tool_patch_file(self, args)

    def tool_delegate(self, args):
        return tool_capabilities.tool_delegate(self, args)

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        “这是工具调用”还是“这是最终答案”。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：模型返回的原始文本 `raw`
        - 输出：`(kind, payload)`，其中 `kind` 可能是 `tool`、`final`、`retry`

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。
        """
        raw = str(raw)
        # 这里支持两种工具格式：
        # 1. <tool>...</tool> 里包 JSON，适合简短调用
        # 2. XML 风格属性/子标签，适合写文件这类多行内容
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = PicoLoop.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", PicoLoop.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", PicoLoop.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", PicoLoop.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", PicoLoop.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = PicoLoop.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", PicoLoop.retry_notice()
        if "<final>" in raw:
            final = PicoLoop.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", PicoLoop.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", PicoLoop.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = PicoLoop.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = PicoLoop.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
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

    @staticmethod
    def extract_raw(text, tag):
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

    def reset(self):
        self.session["history"] = []
        self.session["last_consolidated"] = 0
        self.session["session_metadata"] = {}
        self.session["updated_at"] = now()
        self.session["history_archive"] = memorylib.default_history_archive_state()
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.MemoryManager(self.session["memory"], workspace_root=self.root)
        self._prepared_session_summary = ""
        self.session_store.save(self.session)

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved


__all__ = ["PicoLoop"]
