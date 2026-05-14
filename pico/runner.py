"""Pico runner for model/tool control loops and tool-result governance.

Runner boundary:
- It may request models, parse responses, execute tools through the supplied
  executor, append assistant/tool messages, and govern tool results.
- It must not own product lifecycle state: no TaskState creation, run directory
  creation, report writing, pending-turn management, runtime checkpoint cleanup,
  or product-level deterministic fallback finals.

The current ``ask()`` method is a legacy compatibility entrypoint and still
contains lifecycle calls that are scheduled to move into ``PicoLoop``.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .workspace import clip, now

_BACKFILL_CONTENT = "Error: Task interrupted before this tool finished."
_SNIP_SAFETY_BUFFER = 256
_MICROCOMPACT_KEEP_RECENT = 10
_MICROCOMPACT_MIN_CHARS = 500
_COMPACTABLE_TOOLS = frozenset({
    "read_file", "exec", "grep", "glob",
    "web_search", "web_fetch", "list_dir",
})


@dataclass(slots=True)
class PicoToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class PicoRunSpec:
    """Input contract for one Pico runner turn.

    Product lifecycle objects such as TaskState, run directories, checkpoints,
    reports, and pending user turns belong to PicoLoop and must not be carried
    here.
    """

    model_client: Any
    tool_registry: Any
    tool_executor: Any
    max_iterations: int
    max_new_tokens: int
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tool_result_chars: int = 16_000
    user_message: str = ""
    prompt: str | None = None
    prompt_metadata: dict[str, Any] = field(default_factory=dict)
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    concurrent_tools: bool = True
    fail_on_tool_error: bool = False
    progress_callback: Any | None = None
    trace_callback: Any | None = None
    checkpoint_callback: Any | None = None
    tool_result_limit: int | None = None
    workspace_root: Path | None = None
    session_id: str | None = None
    context_window_tokens: int | None = None
    context_block_limit: int | None = None


@dataclass(slots=True)
class PicoRunResult:
    """Output contract for one Pico runner turn."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PicoModelRequest:
    prompt: str
    prompt_metadata: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None


@dataclass(slots=True)
class PicoParsedResponse:
    kind: str
    payload: Any
    content: str
    tool_calls: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class PicoToolExecution:
    message: dict[str, Any] | None
    final_content: str | None = None


@dataclass(slots=True)
class PicoRawToolExecution:
    tool_call_id: str
    name: str
    args: dict[str, Any]
    result: Any
    metadata: dict[str, Any]
    duration_ms: int
    include_tool_call_id: bool


class PicoRunner:
    """Runner layer for Pico's agent turn loop."""

    def __init__(
        self,
        agent: Any,
        *,
        max_tool_result_chars: int = 16_000,
        fail_on_tool_error: bool = False,
    ) -> None:
        self.agent = agent
        self.max_tool_result_chars = max_tool_result_chars
        self.fail_on_tool_error = fail_on_tool_error
        self._active_spec: PicoRunSpec | None = None

    @staticmethod
    def _messages_to_legacy_prompt(messages: list[dict[str, Any]]) -> str:
        from .context_manager import ContextBuilder

        return ContextBuilder.render_legacy_prompt(messages, include_relevant_memory=False)

    def run(self, spec: PicoRunSpec) -> PicoRunResult:
        """Run one agent turn.

        This compatibility implementation keeps the legacy loop behavior while
        establishing ``PicoRunSpec``/``PicoRunResult`` as the public runner
        contract. Later phase-4 tasks will move the loop internals onto this
        spec directly.
        """

        previous_spec = self._active_spec
        previous_max_tool_result_chars = self.max_tool_result_chars
        self._active_spec = spec
        self.max_tool_result_chars = int(spec.tool_result_limit or spec.max_tool_result_chars)
        try:
            final = self.ask(spec.user_message)
        finally:
            self._active_spec = previous_spec
            self.max_tool_result_chars = previous_max_tool_result_chars
        task_state = getattr(self.agent, "current_task_state", None)
        task_stop_reason = str(getattr(task_state, "stop_reason", "") or "completed")
        stop_reason = self._to_nanobot_stop_reason(task_stop_reason)
        messages = list(getattr(self.agent, "session", {}).get("history", []))
        return PicoRunResult(
            final_content=final,
            messages=messages,
            tools_used=self._tools_used_from_run(task_state, messages),
            stop_reason=stop_reason,
            error=final if stop_reason == "error" else None,
            usage=dict(getattr(self.agent, "last_completion_metadata", {}) or {}),
            metadata={
                "prompt_metadata": dict(getattr(self.agent, "last_prompt_metadata", {}) or {}),
                "completion_metadata": dict(getattr(self.agent, "last_completion_metadata", {}) or {}),
            },
        )

    @staticmethod
    def _tools_used_from_run(task_state: Any, messages: list[dict[str, Any]]) -> list[str]:
        names = [str(name) for name in list(getattr(task_state, "tools_used", []) or []) if str(name).strip()]
        if not names:
            names = [
                str(message.get("name"))
                for message in messages
                if message.get("role") == "tool" and str(message.get("name", "")).strip()
            ]
        seen: set[str] = set()
        ordered: list[str] = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered

    @staticmethod
    def _to_nanobot_stop_reason(stop_reason: str) -> str:
        mapping = {
            "final_answer_returned": "completed",
            "model_error": "error",
            "step_limit_reached": "max_iterations",
            "retry_limit_reached": "max_iterations",
        }
        return mapping.get(str(stop_reason or ""), str(stop_reason or "completed"))

    def _current_spec(self, user_message: str) -> PicoRunSpec:
        if self._active_spec is not None:
            return self._active_spec
        agent = self.agent
        return PicoRunSpec(
            model_client=agent.model_client,
            messages=list(agent.session.get("history", [])),
            tool_registry=agent.tool_registry,
            tool_executor=agent.tool_executor,
            max_iterations=agent.max_steps,
            max_new_tokens=agent.max_new_tokens,
            max_tool_result_chars=self.max_tool_result_chars,
            user_message=user_message,
            fail_on_tool_error=self.fail_on_tool_error,
            workspace_root=getattr(agent, "root", None),
            session_id=str(agent.session.get("id", "")),
            context_window_tokens=getattr(agent, "history_consolidator", None).context_window_tokens
            if getattr(agent, "history_consolidator", None) is not None
            else None,
        )

    def _record_message(self, message: dict[str, Any]) -> None:
        self.agent.record(message)

    def _write_task_state(self, task_state: Any) -> None:
        self.agent.write_task_state_for_runner(task_state)

    def _emit_trace(self, task_state: Any, event: str, payload: dict[str, Any] | None = None) -> Any:
        return self.agent.emit_trace_for_runner(task_state, event, payload)

    def _emit_progress(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.agent.emit_progress_for_runner(event, payload)

    def _create_checkpoint(self, task_state: Any, user_message: str, trigger: str, **kwargs: Any) -> Any:
        return self.agent.create_checkpoint_for_runner(task_state, user_message, trigger=trigger, **kwargs)

    def _clear_pending_user_turn(self) -> None:
        self.agent.clear_pending_user_turn_for_runner()

    def _clear_runtime_checkpoint(self) -> None:
        self.agent.clear_runtime_checkpoint_for_runner()

    def _write_report(self, task_state: Any) -> None:
        self.agent.write_report_for_runner(task_state)

    def _reset_prepared_session_summary(self) -> None:
        self.agent._prepared_session_summary = ""

    def _build_prompt_metadata(
        self,
        user_message: str,
        prompt: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.agent.prompt_metadata_for_messages_for_runner(user_message, prompt, messages)

    def ask(self, user_message: str) -> str:
        agent = self.agent
        spec = self._current_spec(user_message)
        run_started_at = time.monotonic()
        task_state = agent.current_task_state
        if task_state is None:
            task_state = agent.start_task_run_for_turn(user_message)

        tool_steps = 0
        attempts = 0
        max_attempts = max(agent.max_steps * 3, agent.max_steps + 4)

        while tool_steps < agent.max_steps and attempts < max_attempts:
            attempts += 1
            task_state.record_attempt()
            self._write_task_state(task_state)
            initial_messages = list(spec.initial_messages or spec.messages or [])
            governed_messages = self.govern_messages_for_model(initial_messages, spec=spec)
            prompt_started_at = time.monotonic()
            prompt = self._messages_to_legacy_prompt(governed_messages)
            prompt_metadata = self._build_prompt_metadata(user_message, prompt, governed_messages)
            self._emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            if prompt_metadata.get("budget_reductions"):
                checkpoint = self._create_checkpoint(task_state, user_message, trigger="context_reduction")
                self._write_task_state(task_state)
                self._emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            self._emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            self._emit_progress(
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                },
            )

            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(agent.model_client, "supports_prompt_cache", False):
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()

            try:
                model_response = self._complete_model(
                    spec,
                    PicoModelRequest(
                        prompt=prompt,
                        prompt_metadata=prompt_metadata,
                        messages=governed_messages,
                        prompt_cache_key=prompt_cache_key,
                        prompt_cache_retention=prompt_cache_retention,
                    ),
                )
            except Exception as exc:
                final = f"Model request failed: {type(exc).__name__}: {exc}"
                agent.last_completion_metadata = {}
                agent.last_prompt_metadata = prompt_metadata
                task_state.stop_model_error(final)
                self._record_message({"role": "assistant", "content": final, "created_at": now()})
                self._clear_pending_user_turn()
                self._clear_runtime_checkpoint()
                self._write_task_state(task_state)
                self._emit_trace(
                    task_state,
                    "model_error",
                    {
                        "error": final,
                        "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                    },
                )
                self._emit_trace(
                    task_state,
                    "run_finished",
                    {
                        "status": task_state.status,
                        "stop_reason": task_state.stop_reason,
                        "final_answer": final,
                        "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                    },
                )
                self._write_report(task_state)
                self._reset_prepared_session_summary()
                self._emit_progress("model_error", {"error": final})
                return final
            raw = model_response.content
            completion_metadata = dict(getattr(model_response, "metadata", {}) or {})
            if not completion_metadata:
                completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                prompt_metadata.update(completion_metadata)
            if model_response.tool_calls:
                prompt_metadata["tool_call_protocol"] = "native"
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            parsed = self._parse_model_response(model_response, parse_legacy=agent.parse)
            kind, payload = parsed.kind, parsed.payload
            self._emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            if kind == "tool_calls":
                tool_calls = list(payload.get("tool_calls", []))
                assistant_message = {
                    "role": "assistant",
                    "content": raw or "",
                    "tool_calls": [self._to_openai_tool_call(tool_call) for tool_call in tool_calls],
                    "created_at": now(),
                }
                self._record_message(assistant_message)
                pending_tool_calls = [self._to_openai_tool_call(tool_call) for tool_call in tool_calls]
                checkpoint = self._create_checkpoint(
                    task_state,
                    user_message,
                    trigger="tool_requested",
                    assistant_message=assistant_message,
                    pending_tool_calls=pending_tool_calls,
                )
                self._emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_requested",
                        "phase": "awaiting_tools",
                    },
                )
                completed_tool_results = []
                tool_steps += len(tool_calls)
                for execution in self._execute_tool_call_batches_for_loop(
                    task_state,
                    tool_calls,
                    user_message=user_message,
                    run_started_at=run_started_at,
                    concurrent_tools=spec.concurrent_tools,
                ):
                    if execution.final_content is not None:
                        return execution.final_content
                    if execution.message is not None:
                        completed_tool_results.append(execution.message)
                checkpoint = self._create_checkpoint(
                    task_state,
                    user_message,
                    trigger="tool_executed",
                    assistant_message=assistant_message,
                    completed_tool_results=completed_tool_results,
                    pending_tool_calls=[],
                )
                self._write_task_state(task_state)
                self._emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                        "phase": "tools_completed",
                    },
                )
                continue

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                checkpoint = self._create_checkpoint(
                    task_state,
                    user_message,
                    trigger="tool_requested",
                    pending_tool_calls=[
                        {
                            "id": f"tool_{tool_steps}",
                            "function": {"name": name, "arguments": args},
                        }
                    ],
                )
                self._emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_requested",
                    },
                )
                execution = self._execute_tool_call_for_loop(
                    task_state,
                    tool_call_id=f"tool_{tool_steps}",
                    name=name,
                    args=args,
                    user_message=user_message,
                    run_started_at=run_started_at,
                    include_tool_call_id=False,
                )
                if execution.final_content is not None:
                    return execution.final_content
                tool_message = execution.message
                checkpoint = self._create_checkpoint(
                    task_state,
                    user_message,
                    trigger="tool_executed",
                    completed_tool_results=[tool_message] if tool_message is not None else [],
                )
                self._write_task_state(task_state)
                self._emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                continue

            if kind == "retry":
                self._record_message({"role": "assistant", "content": payload, "created_at": now()})
                self._write_task_state(task_state)
                continue

            final = (payload or raw).strip()
            self._record_message({"role": "assistant", "content": final, "created_at": now()})
            self._clear_pending_user_turn()
            self._clear_runtime_checkpoint()
            task_state.finish_success(final)
            self._write_task_state(task_state)
            self._emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            self._write_report(task_state)
            self._reset_prepared_session_summary()
            return final

        if attempts >= max_attempts and tool_steps < agent.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        self._record_message({"role": "assistant", "content": final, "created_at": now()})
        self._clear_pending_user_turn()
        self._clear_runtime_checkpoint()
        self._write_task_state(task_state)
        self._emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        self._write_report(task_state)
        self._reset_prepared_session_summary()
        return final

    def _complete_model(
        self,
        spec: PicoRunSpec,
        request: PicoModelRequest,
    ) -> Any:
        complete_messages_with_tools = getattr(spec.model_client, "complete_messages_with_tools", None)
        if callable(complete_messages_with_tools) and request.messages:
            request.prompt_metadata["tool_call_protocol"] = "native"
            return complete_messages_with_tools(
                request.messages,
                spec.max_new_tokens,
                tools=spec.tool_registry.get_definitions(),
                prompt_cache_key=request.prompt_cache_key,
                prompt_cache_retention=request.prompt_cache_retention,
            )
        complete_with_tools = getattr(spec.model_client, "complete_with_tools", None)
        if callable(complete_with_tools):
            request.prompt_metadata["tool_call_protocol"] = "native"
            return complete_with_tools(
                request.prompt,
                spec.max_new_tokens,
                tools=spec.tool_registry.get_definitions(),
                prompt_cache_key=request.prompt_cache_key,
                prompt_cache_retention=request.prompt_cache_retention,
            )
        request.prompt_metadata["tool_call_protocol"] = "legacy_text"
        from .models import ModelResponse

        return ModelResponse(
            content=spec.model_client.complete(
                request.prompt,
                spec.max_new_tokens,
                prompt_cache_key=request.prompt_cache_key,
                prompt_cache_retention=request.prompt_cache_retention,
            )
        )

    @staticmethod
    def _parse_model_response(model_response: Any, *, parse_legacy: Any) -> PicoParsedResponse:
        content = str(getattr(model_response, "content", "") or "")
        tool_calls = list(getattr(model_response, "tool_calls", []) or [])
        if tool_calls:
            return PicoParsedResponse(
                kind="tool_calls",
                payload={"tool_calls": tool_calls},
                content=content,
                tool_calls=tool_calls,
            )
        kind, payload = parse_legacy(content)
        return PicoParsedResponse(kind=str(kind), payload=payload, content=content)

    def _execute_tool_call_for_loop(
        self,
        task_state: Any,
        *,
        tool_call_id: str,
        name: str,
        args: dict[str, Any],
        user_message: str,
        run_started_at: float,
        include_tool_call_id: bool,
    ) -> PicoToolExecution:
        task_state.record_tool(name)
        raw = self._execute_tool_capability(
            tool_call_id=str(tool_call_id),
            name=name,
            args=args,
            include_tool_call_id=include_tool_call_id,
        )
        return self._commit_tool_execution_for_loop(
            task_state,
            raw,
            user_message=user_message,
            run_started_at=run_started_at,
        )

    def _execute_tool_capability(
        self,
        *,
        tool_call_id: str,
        name: str,
        args: dict[str, Any],
        include_tool_call_id: bool,
    ) -> PicoRawToolExecution:
        tool_started_at = time.monotonic()
        result = self.agent.run_tool(name, args)
        metadata = dict(getattr(self.agent, "_last_tool_result_metadata", {}) or {})
        return PicoRawToolExecution(
            tool_call_id=str(tool_call_id),
            name=name,
            args=dict(args),
            result=result,
            metadata=metadata,
            duration_ms=int((time.monotonic() - tool_started_at) * 1000),
            include_tool_call_id=include_tool_call_id,
        )

    def _commit_tool_execution_for_loop(
        self,
        task_state: Any,
        raw: PicoRawToolExecution,
        *,
        user_message: str,
        run_started_at: float,
    ) -> PicoToolExecution:
        agent = self.agent
        name = raw.name
        args = raw.args
        result = raw.result
        if self.fail_on_tool_error and str(result).startswith("error:"):
            raise RuntimeError(result)
        if agent.is_repeated_tool_result_for_runner(name, result, user_message):
            final = agent.finish_with_repeated_tool_fallback_for_runner(
                task_state,
                run_started_at=run_started_at,
                repeated_result=str(result),
            )
            if final is not None:
                return PicoToolExecution(message=None, final_content=final)
        normalized = self.normalize_tool_result(raw.tool_call_id, str(name), result)
        tool_message = {
            "role": "tool",
            "name": name,
            "args": args,
            "content": normalized,
            "created_at": now(),
        }
        if raw.include_tool_call_id:
            tool_message["tool_call_id"] = raw.tool_call_id
        self._record_message(tool_message)
        self._write_task_state(task_state)
        trace_payload = {
            "name": name,
            "args": args,
            "result": clip(normalized, 500),
            "duration_ms": raw.duration_ms,
            **raw.metadata,
        }
        if raw.include_tool_call_id:
            trace_payload["tool_call_id"] = raw.tool_call_id
        self._emit_trace(task_state, "tool_executed", trace_payload)
        self._emit_progress(
            "tool_executed",
            {
                "name": name,
                "status": raw.metadata.get("tool_status", ""),
            },
        )
        return PicoToolExecution(message=tool_message)

    def _execute_tool_call_batches_for_loop(
        self,
        task_state: Any,
        tool_calls: list[Any],
        *,
        user_message: str,
        run_started_at: float,
        concurrent_tools: bool,
    ) -> list[PicoToolExecution]:
        normalized_calls = [
            PicoToolCall(str(call.id), str(call.name), dict(call.arguments or {}))
            for call in tool_calls
        ]
        executions: list[PicoToolExecution] = []
        for batch in self.partition_tool_batches(normalized_calls, concurrent_tools=concurrent_tools):
            if concurrent_tools and len(batch) > 1:
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    futures = [
                        executor.submit(
                            self._execute_tool_capability,
                            tool_call_id=call.id,
                            name=call.name,
                            args=dict(call.arguments or {}),
                            include_tool_call_id=True,
                        )
                        for call in batch
                    ]
                    raw_results = [future.result() for future in futures]
                for raw in raw_results:
                    executions.append(
                        self._commit_tool_execution_for_loop(
                            task_state,
                            raw,
                            user_message=user_message,
                            run_started_at=run_started_at,
                        )
                    )
                continue
            for call in batch:
                executions.append(
                    self._execute_tool_call_for_loop(
                        task_state,
                        tool_call_id=call.id,
                        name=call.name,
                        args=dict(call.arguments or {}),
                        user_message=user_message,
                        run_started_at=run_started_at,
                        include_tool_call_id=True,
                    )
                )
        return executions

    @staticmethod
    def _to_openai_tool_call(tool_call: Any) -> dict[str, Any]:
        return {
            "id": str(tool_call.id),
            "type": "function",
            "function": {
                "name": str(tool_call.name),
                "arguments": dict(tool_call.arguments or {}),
            },
        }

    def partition_tool_batches(
        self,
        tool_calls: list[PicoToolCall],
        *,
        concurrent_tools: bool = True,
    ) -> list[list[PicoToolCall]]:
        if not concurrent_tools:
            return [[tool_call] for tool_call in tool_calls]

        batches: list[list[PicoToolCall]] = []
        current: list[PicoToolCall] = []
        for tool_call in tool_calls:
            tool = self.agent.tool_registry.get(tool_call.name)
            can_batch = bool(tool and tool.read_only and not tool.exclusive and tool.concurrency_safe)
            if can_batch:
                current.append(tool_call)
                continue
            if current:
                batches.append(current)
                current = []
            batches.append([tool_call])
        if current:
            batches.append(current)
        return batches

    async def execute_tool_calls(
        self,
        tool_calls: list[PicoToolCall],
        *,
        concurrent_tools: bool = True,
    ) -> list[str]:
        results: list[str] = []
        for batch in self.partition_tool_batches(tool_calls, concurrent_tools=concurrent_tools):
            if concurrent_tools and len(batch) > 1:
                results.extend(await asyncio.gather(*(self._execute_one(call) for call in batch)))
            else:
                for call in batch:
                    results.append(await self._execute_one(call))
        return results

    async def _execute_one(self, tool_call: PicoToolCall) -> str:
        result = self.agent.run_tool(tool_call.name, tool_call.arguments)
        if self.fail_on_tool_error and str(result).startswith("error:"):
            raise RuntimeError(result)
        return str(result)

    def normalize_tool_result(self, tool_call_id: str, tool_name: str, result: Any) -> str:
        return self._normalize_tool_result(tool_call_id, tool_name, result)

    def _normalize_tool_result(self, tool_call_id: str, tool_name: str, result: Any) -> str:
        text = self.ensure_nonempty_tool_result(tool_name, result)
        persisted = len(text) > self.max_tool_result_chars
        text = self.maybe_persist_tool_result(tool_call_id, text)
        if persisted:
            return text
        if len(text) > self.max_tool_result_chars:
            return clip(text, self.max_tool_result_chars)
        return text

    @staticmethod
    def ensure_nonempty_tool_result(tool_name: str, result: Any) -> str:
        if result is None:
            return f"[{tool_name} returned no content]"
        text = str(result)
        if not text.strip():
            return f"[{tool_name} returned empty content]"
        return text

    def maybe_persist_tool_result(self, tool_call_id: str, text: str) -> str:
        if len(text) <= self.max_tool_result_chars:
            return text
        root = Path(self.agent.root) / ".pico" / "tool-results" / str(self.agent.session.get("id", "default"))
        root.mkdir(parents=True, exist_ok=True)
        filename = f"{tool_call_id}.txt"
        path = root / filename
        path.write_text(text, encoding="utf-8")
        rel = path.relative_to(self.agent.root).as_posix()
        return f"[Tool result too large; full content saved to {rel}]"

    @staticmethod
    def _drop_orphan_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declared: set[str] = set()
        updated: list[dict[str, Any]] | None = None
        for index, message in enumerate(messages):
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if isinstance(tool_call, dict) and tool_call.get("id"):
                        declared.add(str(tool_call["id"]))
            if message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and str(tool_call_id) not in declared:
                    if updated is None:
                        updated = [dict(item) for item in messages[:index]]
                    continue
            if updated is not None:
                updated.append(dict(message))
        return messages if updated is None else updated

    drop_orphan_tool_results = _drop_orphan_tool_results

    @staticmethod
    def _backfill_missing_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declared: list[tuple[int, str, str]] = []
        fulfilled: set[str] = set()
        for index, message in enumerate(messages):
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict) or not tool_call.get("id"):
                        continue
                    function = tool_call.get("function")
                    name = function.get("name", "") if isinstance(function, dict) else ""
                    declared.append((index, str(tool_call["id"]), str(name)))
            elif message.get("role") == "tool" and message.get("tool_call_id"):
                fulfilled.add(str(message["tool_call_id"]))

        missing = [(idx, call_id, name) for idx, call_id, name in declared if call_id not in fulfilled]
        if not missing:
            return messages

        updated = [dict(message) for message in messages]
        offset = 0
        for assistant_index, call_id, name in missing:
            insert_at = assistant_index + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(
                insert_at,
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": _BACKFILL_CONTENT,
                },
            )
            offset += 1
        return updated

    backfill_missing_tool_results = _backfill_missing_tool_results

    @staticmethod
    def microcompact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compactable = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "tool" and message.get("name") in _COMPACTABLE_TOOLS
        ]
        if len(compactable) <= _MICROCOMPACT_KEEP_RECENT:
            return messages
        stale = compactable[: len(compactable) - _MICROCOMPACT_KEEP_RECENT]
        updated: list[dict[str, Any]] | None = None
        for index in stale:
            content = messages[index].get("content")
            if not isinstance(content, str) or len(content) < _MICROCOMPACT_MIN_CHARS:
                continue
            if updated is None:
                updated = [dict(message) for message in messages]
            name = messages[index].get("name", "tool")
            updated[index]["content"] = f"[{name} result omitted from context]"
        return messages if updated is None else updated

    @staticmethod
    def _estimate_message_tokens(message: dict[str, Any]) -> int:
        return max(1, len(str(message)) // 4)

    def _estimate_prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        tool_defs = []
        try:
            spec = self._active_spec
            if spec is not None:
                tool_defs = spec.tool_registry.get_definitions()
        except Exception:
            tool_defs = []
        return sum(self._estimate_message_tokens(message) for message in messages) + max(0, len(str(tool_defs)) // 4)

    def _snip_history(self, spec: PicoRunSpec, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages or not spec.context_window_tokens:
            return messages
        budget = int(spec.context_block_limit or (spec.context_window_tokens - spec.max_new_tokens - _SNIP_SAFETY_BUFFER))
        if budget <= 0 or self._estimate_prompt_tokens(messages) <= budget:
            return messages
        system_messages = [dict(message) for message in messages if message.get("role") == "system"]
        non_system = [dict(message) for message in messages if message.get("role") != "system"]
        kept: list[dict[str, Any]] = []
        used = sum(self._estimate_message_tokens(message) for message in system_messages)
        for message in reversed(non_system):
            cost = self._estimate_message_tokens(message)
            if kept and used + cost > budget:
                break
            kept.insert(0, message)
            used += cost
        for index, message in enumerate(kept):
            if message.get("role") == "user":
                kept = kept[index:]
                break
        return system_messages + kept

    def govern_messages_for_model(self, messages: list[dict[str, Any]], spec: PicoRunSpec | None = None) -> list[dict[str, Any]]:
        governed = self._drop_orphan_tool_results(messages)
        governed = self._backfill_missing_tool_results(governed)
        governed = self.microcompact(governed)
        governed = self._apply_tool_result_budget(governed)
        if spec is not None:
            governed = self._snip_history(spec, governed)
            governed = self._drop_orphan_tool_results(governed)
            governed = self._backfill_missing_tool_results(governed)
        return governed

    def _apply_tool_result_budget(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated = messages
        for index, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            normalized = self._normalize_tool_result(
                str(message.get("tool_call_id") or f"tool_{index}"),
                str(message.get("name") or "tool"),
                message.get("content"),
            )
            if normalized != message.get("content"):
                if updated is messages:
                    updated = [dict(item) for item in messages]
                updated[index]["content"] = normalized
        return updated
