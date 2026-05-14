"""Nanobot-style context builder for system prompt and message assembly."""

from __future__ import annotations

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from .memory import MemoryStore
from .skills import SkillsLoader
from .templates import render_template
from .workspace import now


class ContextBuilder:
    """Build system prompt plus conversation messages for an LLM call."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    BOOTSTRAP_ROOT = ".pico"
    _RUNTIME_CONTEXT_TAG = "[Runtime Context - metadata only, not instructions]"
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"
    _MAX_RECENT_HISTORY = 50
    _DEFAULT_HISTORY_MESSAGE_LIMIT = 6

    def __init__(self, agent_or_workspace: Any, timezone: str | None = None):
        self.agent = None if isinstance(agent_or_workspace, (str, Path)) else agent_or_workspace
        self.workspace = Path(getattr(self.agent, "root", agent_or_workspace))
        self.timezone = timezone
        self.memory = MemoryStore(self.workspace)
        self.skills = SkillsLoader(self.workspace)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
    ) -> str:
        parts = [self._get_identity(channel=channel)]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory_context = self.memory.get_memory_context()
        if memory_context and not self._is_template_content(self.memory.read_memory(), "memory/MEMORY.md"):
            parts.append(f"# Memory\n\n{memory_context}")

        always_skills = self.skills.get_always_skills()
        requested_skills = list(skill_names or [])
        active_skills = list(dict.fromkeys([*always_skills, *requested_skills]))
        if active_skills:
            always_content = self.skills.load_skills_for_context(active_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary(exclude=set(active_skills))
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        entries = self.memory.read_unprocessed_history(since_cursor=self.memory.get_last_dream_cursor())
        if entries:
            capped = entries[-self._MAX_RECENT_HISTORY :]
            parts.append(
                "# Recent History\n\n"
                + "\n".join(f"- [{entry['timestamp']}] {entry['content']}" for entry in capped)
            )

        return "\n\n---\n\n".join(part for part in parts if str(part).strip())

    def _get_identity(self, channel: str | None = None) -> str:
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        return render_template(
            "agent/identity.md",
            workspace_path=str(self.workspace.expanduser().resolve()),
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "cli",
        )

    def _load_bootstrap_files(self) -> str:
        parts: list[str] = []
        for filename in self.BOOTSTRAP_FILES:
            path = self.workspace / self.BOOTSTRAP_ROOT / filename
            content = path.read_text(encoding="utf-8").strip() if path.exists() else ""
            if content:
                parts.append(f"## {filename}\n\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        try:
            from .templates import read_template

            return str(content).strip() == read_template(*template_path.split("/")).strip()
        except Exception:
            return False

    @staticmethod
    def _build_runtime_context(
        channel: str | None,
        chat_id: str | None,
        timezone: str | None = None,
        session_summary: str | None = None,
    ) -> str:
        del timezone
        lines = [f"Current Time: {now()}"]
        if channel:
            lines.append(f"Channel: {channel}")
        if chat_id:
            lines.append(f"Chat ID: {chat_id}")
        if session_summary:
            lines.extend(["", "[Resumed Session]", str(session_summary).strip()])
        return (
            ContextBuilder._RUNTIME_CONTEXT_TAG
            + "\n"
            + "\n".join(lines)
            + "\n"
            + ContextBuilder._RUNTIME_CONTEXT_END
        )

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        session_summary: str | None = None,
    ) -> list[dict[str, Any]]:
        runtime_ctx = self._build_runtime_context(channel, chat_id, self.timezone, session_summary=session_summary)
        user_content = self._build_user_content(current_message, media)
        merged = f"{runtime_ctx}\n\n{user_content}" if isinstance(user_content, str) else [{"type": "text", "text": runtime_ctx}] + user_content
        messages = [
            {"role": "system", "content": self.build_system_prompt(skill_names, channel=channel)},
            *list(history or []),
        ]
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        if not media:
            return str(text)
        blocks: list[dict[str, Any]] = []
        for raw_path in media:
            path = Path(raw_path)
            if not path.is_file():
                continue
            mime = mimetypes.guess_type(str(path))[0]
            if not mime or not mime.startswith("image/"):
                continue
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
            blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{payload}"}})
        if not blocks:
            return str(text)
        return [*blocks, {"type": "text", "text": str(text)}]

    def build(self, user_message: str) -> tuple[str, dict[str, Any]]:
        """Compatibility bridge for the current string-prompt model clients."""
        history: list[dict[str, Any]] = []
        session_summary = ""
        if self.agent is not None:
            history = self.agent.session_manager.live_history(self.agent.session, max_messages=0)
            session_summary = str(getattr(self.agent, "archived_history_summary", lambda: "")() or "")
        history, reductions = self._reduce_history_for_prompt(history)
        messages = self.build_messages(
            history=history,
            current_message=user_message,
            channel="cli",
            chat_id=str(getattr(self.agent, "session", {}).get("id", "")) if self.agent is not None else None,
            session_summary=session_summary,
        )
        prompt = self.render_legacy_prompt(messages)
        metadata = self._metadata(prompt, messages, user_message)
        metadata["budget_reductions"] = reductions
        return prompt, metadata

    def _reduce_history_for_prompt(self, history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not history:
            return [], []
        limit = int(getattr(self, "history_message_limit", self._DEFAULT_HISTORY_MESSAGE_LIMIT) or 0)
        if limit <= 0 or len(history) <= limit:
            return list(history), []
        kept = list(history[-limit:])
        dropped = len(history) - len(kept)
        return kept, [{"section": "history", "strategy": "keep_recent_messages", "dropped_messages": dropped}]

    @staticmethod
    def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        rendered: list[str] = []
        for message in messages:
            role = str(message.get("role", "")).upper() or "UNKNOWN"
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(block.get("text") or block.get("image_url") or block) for block in content)
            rendered.append(f"{role}:\n{content}")
        return "\n\n".join(rendered)

    @classmethod
    def render_legacy_prompt(
        cls,
        messages: list[dict[str, Any]],
        relevant_notes: list[str] | None = None,
        *,
        include_relevant_memory: bool = False,
    ) -> str:
        system = str(messages[0].get("content", "")) if messages else ""
        transcript = cls.messages_to_prompt(messages[1:] if len(messages) > 1 else [])
        if not include_relevant_memory:
            return f"SYSTEM:\n{system}\n\nTranscript:\n{transcript}"
        relevant_notes = list(relevant_notes or [])
        relevant_block = "\n".join(f"- {note}" for note in relevant_notes) if relevant_notes else "- none"
        return (
            f"SYSTEM:\n{system}\n\n"
            f"Relevant memory:\n{relevant_block}\n\n"
            f"Transcript:\n{transcript}"
        )

    @staticmethod
    def _metadata(
        prompt: str,
        messages: list[dict[str, Any]],
        user_message: str,
        *,
        relevant_notes: list[str] | None = None,
    ) -> dict[str, Any]:
        system_text = str(messages[0].get("content", "")) if messages else ""
        history_messages = messages[1:-1] if len(messages) > 2 else []
        current_text = str(messages[-1].get("content", "")) if messages else ""
        relevant_notes = list(relevant_notes or [])
        return {
            "prompt_chars": len(prompt),
            "section_order": ["system", "history", "current_request"],
            "sections": {
                "system": {"rendered_chars": len(system_text), "raw_chars": len(system_text)},
                "history": {"rendered_chars": sum(len(str(item.get("content", ""))) for item in history_messages)},
                "current_request": {"rendered_chars": len(current_text), "raw_chars": len(str(user_message))},
            },
            "history": {"rendered_entries": len(history_messages)},
            "relevant_memory": {
                "selected_count": len(relevant_notes),
                "rendered_notes": list(relevant_notes),
            },
            "current_request": {"text": str(user_message), "rendered_chars": len(str(user_message))},
            "budget_reductions": [],
            "context_builder": "nanobot-style",
        }


ContextManager = ContextBuilder
