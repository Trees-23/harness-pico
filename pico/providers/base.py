"""Base provider contracts for Pico."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: Any | None = field(default=None, compare=False)


@dataclass(slots=True)
class ModelResponse:
    content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw: Any | None = field(default=None, compare=False)
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PicoProvider(Protocol):
    supports_prompt_cache: bool
    last_completion_metadata: dict[str, Any]

    def complete(self, prompt: str, max_new_tokens: int, **kwargs: Any) -> str:
        ...

    def complete_messages(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        **kwargs: Any,
    ) -> str:
        ...

    def complete_with_tools(
        self,
        prompt: str,
        max_new_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        ...

    def complete_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        ...
