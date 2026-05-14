import asyncio
import os
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pico as mini_pkg
import pico.cli as mini_cli
from pico import checkpoint as checkpointlib
from pico import (
    AnthropicCompatibleModelClient,
    FakeModelClient,
    ModelResponse,
    MiniAgent,
    OllamaModelClient,
    OpenAICompatibleModelClient,
    SessionStore,
    ToolCallRequest,
    WorkspaceContext,
    build_tools_help,
    build_welcome,
)
from pico.memory import MemoryStore


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".pico" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return MiniAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_agent_runs_tool_then_final(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Read the file successfully.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Read the file successfully."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    assert "hello.txt" in agent.session["memory"]["files"]


def test_agent_updates_task_summary_on_each_request(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>First pass.</final>",
            "<final>Second pass.</final>",
        ],
    )

    assert agent.ask("First request") == "First pass."
    assert agent.session["memory"]["working"]["task_summary"] == "First request"

    assert agent.ask("Second request") == "Second pass."
    assert agent.session["memory"]["working"]["task_summary"] == "Second request"


def test_agent_only_stores_reusable_epistemic_notes(tmp_path):
    (tmp_path / "facts.txt").write_text("deploy key is red\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"facts.txt","start":1,"end":1}}</tool>',
            "<final>Done.</final>",
            "<final>It is red.</final>",
        ],
    )

    assert agent.ask("Read the file and remember the fact") == "Done."
    notes = agent.session["memory"]["episodic_notes"]
    assert any("deploy key is red" in note["text"] for note in notes)
    assert not any(note["text"] == "Done." for note in notes)
    assert not any(note["text"] == "Done." for note in notes)

    resumed = MiniAgent.from_session(
        model_client=FakeModelClient(["<final>It is red.</final>"]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("What color is the deploy key?") == "It is red."
    prompt = resumed.model_client.prompts[-1]
    assert "Relevant memory" not in prompt
    assert "deploy key is red" in prompt


def test_file_summary_cache_is_invalidated_on_out_of_band_edit_and_path_spelling(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    agent.memory.set_file_summary("./sample.txt", "sample.txt: alpha")
    agent.memory.remember_file("./sample.txt")
    assert agent.memory.to_dict()["file_summaries"]["sample.txt"]["freshness"]

    assert "sample.txt: alpha" in agent.memory.render_memory_text()
    file_path.write_text("beta\n", encoding="utf-8")

    resumed = MiniAgent.from_session(
        model_client=FakeModelClient([]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert "sample.txt: alpha" not in resumed.memory.render_memory_text()
    resumed.memory.invalidate_file_summary("sample.txt")
    assert "sample.txt" not in resumed.memory.to_dict()["file_summaries"]


def test_agent_retries_after_empty_model_output(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "<final>Recovered after retry.</final>",
        ],
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after retry."
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("empty response" in item for item in notices)


def test_agent_retries_after_malformed_tool_payload(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":"bad"}</tool>',
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>Recovered after malformed tool output.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Recovered after malformed tool output."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("valid <tool> call" in item for item in notices)


def test_agent_accepts_xml_write_file_tool(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="hello.py"><content>print("hi")\n</content></tool>',
            "<final>Done.</final>",
        ],
    )

    answer = agent.ask("Create hello.py")

    assert answer == "Done."
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == 'print("hi")\n'


def test_retries_do_not_consume_the_whole_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "",
            "<final>Recovered after several retries.</final>",
        ],
        max_steps=1,
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after several retries."


def test_agent_saves_and_resumes_session(tmp_path):
    agent = build_agent(tmp_path, ["<final>First pass.</final>"])
    assert agent.ask("Start a session") == "First pass."

    session_dir = tmp_path / ".pico" / "sessions" / agent.session["id"]
    history_path = session_dir / "cli_direct.jsonl"
    assert history_path.exists()
    assert not (session_dir / "session.json").exists()
    assert not (session_dir / "history.jsonl").exists()
    assert "Start a session" in history_path.read_text(encoding="utf-8")

    resumed = MiniAgent.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.session["history"][0]["content"] == "Start a session"
    assert resumed.ask("Continue") == "Resumed."


def test_agent_compacts_old_history_into_archived_session_summary(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "- Current goal: Continue editing runtime.py\n- Key files: runtime.py, memory.py\n- Next step: inspect the recent transcript",
            "<final>Compaction complete.</final>",
        ],
    )

    for index in range(12):
        role = "user" if index % 2 == 0 else "assistant"
        agent.record(
            {
                "role": role,
                "content": f"history-{index} " + ("X" * 420),
                "created_at": f"2026-04-07T09:{index:02d}:00+00:00",
            }
        )

    answer = agent.ask("Continue editing runtime.py")

    assert answer == "Compaction complete."
    assert agent.session["history_archive"]["compaction_count"] == 1
    assert agent.session["history_archive"]["archived_messages"] > 0
    assert "runtime.py" in agent.session["history_archive"]["latest_summary"]
    assert agent.session["last_consolidated"] > 0
    assert len(agent.session["history"]) == 14
    assert "[Resumed Session]" in agent.model_client.prompts[-1]
    assert "runtime.py" in agent.model_client.prompts[-1]
    assert (tmp_path / ".pico" / "memory" / "history.jsonl").exists()


def test_agent_registers_dream_cron_and_cron_processes_pending_archive(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".pico" / "sessions")
    archive_store = MemoryStore(tmp_path)
    archive_store.append_history("- User prefers concise technical responses.")
    agent = MiniAgent(
        model_client=FakeModelClient(
            [
                "<final>Main turn complete.</final>",
                "- Persist the user's stable communication preference into USER.md.",
                '<tool name="patch_file" path=".pico/USER.md"><old_text>- [ ] Technical</old_text><new_text>- [x] Technical</new_text></tool>',
                "<final>done</final>",
            ]
        ),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )

    answer = agent.ask("Continue")

    assert answer == "Main turn complete."
    assert archive_store.get_last_dream_cursor() == 0
    dream_job = next(job for job in agent.cron.list_jobs(include_disabled=True) if job.id == "dream")
    dream_job.state.next_run_at_ms = 1
    asyncio.run(agent.cron._on_timer())

    assert archive_store.get_last_dream_cursor() == 1
    assert "- [x] Technical" in (tmp_path / ".pico" / "USER.md").read_text(encoding="utf-8")


def test_agent_promotes_user_facts_into_history_without_using_assistant_reply(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>真不错！</final>",
            "<final>其实你只有一只猫。</final>",
            "<final>继续。</final>",
        ],
    )

    agent.ask("我有两只猫")
    agent.ask("我有两只狗")
    agent.ask("我有几个宠物")

    prompt = agent.model_client.prompts[-1]

    assert "Relevant memory:" not in prompt
    assert "我有两只猫" in prompt
    assert "我有两只狗" in prompt
    assert "其实你只有一只猫" in prompt


def test_delegate_uses_child_agent(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"inspect README","max_steps":2}}</tool>',
            "<final>Child result.</final>",
            "<final>Parent incorporated the child result.</final>",
        ],
    )

    answer = agent.ask("Use delegation")

    assert answer == "Parent incorporated the child result."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_patch_file_replaces_exact_match(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "patch_file",
        {
            "path": "sample.txt",
            "old_text": "world",
            "new_text": "agent",
        },
    )

    assert result == "patched sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello agent\n"


def test_invalid_risky_tool_does_not_prompt_for_approval(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")

    with patch("builtins.input") as mock_input:
        result = agent.run_tool("write_file", {})

    assert result.startswith("error: invalid arguments for write_file: Invalid parameters")
    assert "missing required path" in result
    assert "missing required content" in result
    assert 'example: <tool name="write_file"' in result
    mock_input.assert_not_called()


def test_run_tool_uses_registry_prepare_call_for_argument_governance(tmp_path):
    agent = build_agent(tmp_path, [])

    with patch("pico.tools.validate_tool", side_effect=AssertionError("legacy validator bypassed")):
        with patch.object(agent.tool_registry, "prepare_call", wraps=agent.tool_registry.prepare_call) as prepare_call:
            result = agent.run_tool("write_file", {})

    assert "missing required path" in result
    assert "missing required content" in result
    prepare_call.assert_called_once()


def test_write_tools_acquire_same_file_lock(tmp_path):
    class SpyLocks:
        def __init__(self):
            self.keys = []

        @contextmanager
        def lock(self, key):
            self.keys.append(key)
            yield

    agent = build_agent(tmp_path, [])
    spy_locks = SpyLocks()
    agent.tool_executor.file_locks = spy_locks

    result = agent.run_tool("write_file", {"path": "locked.txt", "content": "hello\n"})

    assert result == "wrote locked.txt (6 chars)"
    assert spy_locks.keys == [str((tmp_path / "locked.txt").resolve())]


def test_list_files_hides_internal_agent_state(tmp_path):
    agent = build_agent(tmp_path, [])
    (tmp_path / ".pico").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")

    result = agent.run_tool("list_files", {})

    assert ".pico" not in result
    assert ".git" not in result
    assert "[F] hello.txt" in result


def test_repeated_identical_tool_call_is_rejected(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "1"})
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "2"})

    result = agent.run_tool("list_files", {})

    assert result == "error: repeated identical tool call for list_files; choose a different tool or return a final answer"


def test_repeated_read_file_overlapping_range_is_rejected(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "hello.txt", "start": 1, "end": 200},
            "content": "# hello.txt\n   1: alpha\n   2: beta",
            "created_at": "1",
        }
    )

    result = agent.run_tool("read_file", {"path": "hello.txt", "start": 1, "end": 260})

    assert result.startswith("error: repeated read_file call for an overlapping already-read file range")


def test_repeated_write_file_after_success_is_rejected_before_approval(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")
    agent.record(
        {
            "role": "tool",
            "name": "write_file",
            "args": {"path": "hello.txt", "content": "hello"},
            "content": "wrote hello.txt (5 chars)",
            "created_at": "1",
        }
    )

    result = agent.run_tool("write_file", {"path": "hello.txt", "content": "hello"})

    assert result == "error: repeated identical tool call for write_file; choose a different tool or return a final answer"


def test_welcome_screen_keeps_box_shape_for_long_paths(tmp_path):
    deep = tmp_path / "very" / "long" / "path" / "for" / "the" / "mini" / "agent" / "welcome" / "screen"
    deep.mkdir(parents=True)
    agent = build_agent(deep, [])

    welcome = build_welcome(agent, model="qwen3.5:4b", host="http://127.0.0.1:11434")
    lines = welcome.splitlines()

    assert len(lines) >= 5
    assert len({len(line) for line in lines}) == 1
    assert "..." in welcome
    assert "(  o o  )" in welcome
    assert "MINI-CODING-AGENT" not in welcome
    assert "MINI CODING AGENT" not in welcome
    assert "pico" in welcome
    assert "local coding agent" in welcome
    assert "// READY" not in welcome
    assert "SLASH" not in welcome
    assert "READY      " not in welcome
    assert "commands: Commands:" not in welcome


def test_ollama_client_posts_expected_payload():
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OllamaModelClient(
        model="qwen3.5:4b",
        host="http://127.0.0.1:11434",
        temperature=0.2,
        top_p=0.9,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["timeout"] == 30
    assert captured["body"]["model"] == "qwen3.5:4b"
    assert captured["body"]["prompt"] == "hello"
    assert captured["body"]["stream"] is False


def test_openai_compatible_client_posts_expected_responses_payload():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"output_text": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "https://right.codes/v1/responses"
    assert captured["timeout"] == 30
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["User-agent"] == "pico/0.1"
    assert captured["body"] == {
        "model": "right.codes/codex-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "hello",
                    }
                ],
            }
        ],
        "max_output_tokens": 42,
        "stream": False,
        "temperature": 0.2,
    }


def test_openai_compatible_client_sends_prompt_cache_fields_and_records_usage():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "output_text": "<final>ok</final>",
                    "usage": {
                        "input_tokens": 2048,
                        "input_tokens_details": {"cached_tokens": 1536},
                        "output_tokens": 32,
                        "total_tokens": 2080,
                    },
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete(
            "hello",
            42,
            prompt_cache_key="prefix-hash-123",
            prompt_cache_retention="in_memory",
        )

    assert result == "<final>ok</final>"
    assert captured["body"]["prompt_cache_key"] == "prefix-hash-123"
    assert captured["body"]["prompt_cache_retention"] == "in_memory"
    assert client.last_completion_metadata["prompt_cache_supported"] is True
    assert client.last_completion_metadata["cached_tokens"] == 1536
    assert client.last_completion_metadata["cache_hit"] is True
    assert client.last_completion_metadata["input_tokens"] == 2048


def test_openai_compatible_client_complete_with_tools_posts_tool_definitions():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "read_file",
                            "arguments": "{\"path\":\"README.md\",\"start\":\"1\"}",
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]

    with patch("urllib.request.urlopen", fake_urlopen):
        response = client.complete_with_tools("hello", 42, tools=tools)

    assert captured["body"]["tools"] == tools
    assert captured["body"]["tool_choice"] == "auto"
    assert response.tool_calls == [ToolCallRequest("call_1", "read_file", {"path": "README.md", "start": "1"})]


def test_openai_compatible_client_messages_path_uses_responses_not_anthropic_messages():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"output_text": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        response = client.complete_messages_with_tools(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
            42,
            tools=[{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        )

    assert captured["url"] == "https://right.codes/v1/responses"
    assert "x-api-key" not in {key.lower(): value for key, value in captured["headers"].items()}
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["instructions"] == "sys"
    assert response.content == "<final>ok</final>"


def test_openai_compatible_client_chat_completions_mode_for_proxy_stations():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "<final>chat ok</final>"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="gpt-5.5",
        base_url="https://api.s2im7pl7e.xyz/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
        api_style="chat_completions",
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        response = client.complete_messages_with_tools(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
            42,
            tools=[{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        )

    assert captured["url"] == "https://api.s2im7pl7e.xyz/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert captured["body"]["max_tokens"] == 42
    assert captured["body"]["tools"][0]["function"]["name"] == "read_file"
    assert response.content == "<final>chat ok</final>"


def test_openai_compatible_client_allows_user_agent_override():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"output_text": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["headers"] = dict(request.headers)
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
        user_agent="curl/8.5.0",
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        assert client.complete("hello", 42) == "<final>ok</final>"

    assert captured["headers"]["User-agent"] == "curl/8.5.0"


def test_openai_compatible_client_extracts_text_from_event_stream():
    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                'data: {"type":"response.created","response":{"id":"resp_1","output":[]}}\n'
                'data: {"type":"response.completed","response":{"output":[{"content":[{"text":"<final>stream ok</final>"}]}]}}\n'
                "data: [DONE]\n"
            ).encode("utf-8")

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = client.complete("hello", 42)

    assert result == "<final>stream ok</final>"


def test_openai_compatible_client_extracts_text_from_event_stream_deltas():
    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"<final>"}\n'
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"OK"}\n'
                'event: response.output_text.done\n'
                'data: {"type":"response.output_text.done","text":"<final>OK</final>"}\n'
                "data: [DONE]\n"
            ).encode("utf-8")

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = client.complete("hello", 42)

    assert result == "<final>OK</final>"


def test_anthropic_compatible_client_posts_expected_messages_payload():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "<final>ok</final>",
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = AnthropicCompatibleModelClient(
        model="claude-sonnet-4-5-20250929",
        base_url="https://www.right.codes/claude-aws/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "https://www.right.codes/claude-aws/v1/messages"
    assert captured["timeout"] == 30
    assert captured["headers"]["X-api-key"] == "sk-test"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["body"] == {
        "model": "claude-sonnet-4-5-20250929",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                    }
                ],
            }
        ],
        "max_tokens": 42,
        "stream": False,
        "temperature": 0.2,
    }


def test_anthropic_compatible_client_extracts_first_text_block():
    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": "<final>ok</final>"},
                    ]
                }
            ).encode("utf-8")

    client = AnthropicCompatibleModelClient(
        model="claude-sonnet-4-5-20250929",
        base_url="https://www.right.codes/claude-aws/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"


def test_anthropic_compatible_client_complete_with_tools_posts_tool_schema():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read_file",
                            "input": {"path": "README.md", "start": "1"},
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = AnthropicCompatibleModelClient(
        model="claude-sonnet-4-5-20250929",
        base_url="https://www.right.codes/claude-aws/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]

    with patch("urllib.request.urlopen", fake_urlopen):
        response = client.complete_with_tools("hello", 42, tools=tools)

    assert captured["body"]["tools"] == [
        {
            "name": "read_file",
            "description": "Read",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    assert captured["body"]["tool_choice"] == {"type": "auto"}
    assert response.tool_calls == [ToolCallRequest("toolu_1", "read_file", {"path": "README.md", "start": "1"})]


def test_build_agent_uses_openai_provider_and_model_override(tmp_path):
    args = type(
        "Args",
        (),
        {
            "cwd": str(tmp_path),
            "provider": "openai",
            "model": "override-model",
            "base_url": None,
            "host": "http://127.0.0.1:11434",
            "ollama_timeout": 300,
            "temperature": 0.2,
            "top_p": 0.9,
            "resume": None,
            "approval": "ask",
            "secret_env_names": [],
            "max_steps": 6,
            "max_new_tokens": 512,
        },
    )()

    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE": "https://www.right.codes/codex/v1",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "env-model",
        },
        clear=False,
    ):
        with patch(
            "pico.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch("pico.cli.OpenAICompatibleModelClient") as mock_openai:
            fake_client = mock_openai.return_value
            agent = mini_pkg.build_agent(args)

    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["model"] == "override-model"
    assert mock_openai.call_args.kwargs["base_url"] == "https://www.right.codes/codex/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "sk-test"
    assert agent.model_client is fake_client


def test_load_env_file_supports_export_quotes_and_preserves_existing_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '\n'.join(
            [
                'export OPENAI_API_KEY="sk-from-file"',
                "OPENAI_API_BASE='https://env.example/v1'",
                "OPENAI_MODEL=gpt-env",
                "# ignored comment",
                "not an assignment",
            ]
        ),
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"OPENAI_MODEL": "already-set"}, clear=True):
        loaded = mini_cli.load_env_file(env_file)

        assert os.environ["OPENAI_API_KEY"] == "sk-from-file"
        assert os.environ["OPENAI_API_BASE"] == "https://env.example/v1"
        assert os.environ["OPENAI_MODEL"] == "already-set"
        assert loaded == {
            "OPENAI_API_KEY": "sk-from-file",
            "OPENAI_API_BASE": "https://env.example/v1",
        }


def test_build_agent_loads_project_env_before_openai_client(tmp_path):
    args = mini_pkg.build_arg_parser().parse_args(["--cwd", str(tmp_path)])
    env_file = tmp_path / "project.env"
    env_file.write_text(
        '\n'.join(
            [
                'export OPENAI_API_KEY="sk-dotenv"',
                'export OPENAI_API_BASE="https://dotenv.example/v1"',
                'export OPENAI_MODEL="dotenv-model"',
            ]
        ),
        encoding="utf-8",
    )

    with patch.dict(os.environ, {}, clear=True), patch("pico.cli.DEFAULT_ENV_FILE", env_file), patch(
        "pico.cli.OpenAICompatibleModelClient"
    ) as mock_openai:
        agent = mini_pkg.build_agent(args)

    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["model"] == "dotenv-model"
    assert mock_openai.call_args.kwargs["base_url"] == "https://dotenv.example/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "sk-dotenv"
    assert agent.model_client is mock_openai.return_value


def test_project_env_can_force_openai_provider_and_chat_completions_style(tmp_path):
    args = mini_pkg.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--provider", "anthropic"])
    env_file = tmp_path / "project.env"
    env_file.write_text(
        '\n'.join(
            [
                'OPENAI_API_KEY="sk-dotenv"',
                'OPENAI_API_BASE="https://proxy.example/v1"',
                'OPENAI_MODEL="proxy-model"',
                'PICO_PROVIDER="openai"',
                'OPENAI_API_STYLE="chat_completions"',
            ]
        ),
        encoding="utf-8",
    )

    with patch.dict(os.environ, {}, clear=True), patch("pico.cli.DEFAULT_ENV_FILE", env_file), patch(
        "pico.cli.AnthropicCompatibleModelClient",
        side_effect=AssertionError("anthropic client should not be used"),
    ), patch("pico.cli.OpenAICompatibleModelClient") as mock_openai:
        agent = mini_pkg.build_agent(args)

    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["model"] == "proxy-model"
    assert mock_openai.call_args.kwargs["base_url"] == "https://proxy.example/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "sk-dotenv"
    assert mock_openai.call_args.kwargs["api_style"] == "chat_completions"
    assert agent.model_client is mock_openai.return_value


def test_build_arg_parser_defaults_provider_to_openai(tmp_path):
    args = mini_pkg.build_arg_parser().parse_args(["--cwd", str(tmp_path)])

    assert args.provider == "openai"


def test_build_arg_parser_accepts_anthropic_provider(tmp_path):
    args = mini_pkg.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--provider", "anthropic"])

    assert args.provider == "anthropic"


def test_build_agent_uses_anthropic_provider_and_openai_key_fallback(tmp_path):
    args = type(
        "Args",
        (),
        {
            "cwd": str(tmp_path),
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "base_url": None,
            "host": "http://127.0.0.1:11434",
            "ollama_timeout": 300,
            "openai_timeout": 300,
            "temperature": 0.2,
            "top_p": 0.9,
            "resume": None,
            "approval": "ask",
            "secret_env_names": [],
            "max_steps": 6,
            "max_new_tokens": 512,
        },
    )()

    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-openai-fallback",
        },
        clear=True,
    ), patch("pico.cli.DEFAULT_ENV_FILE", tmp_path / "missing.env"):
        with patch(
            "pico.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch(
            "pico.cli.OpenAICompatibleModelClient",
            side_effect=AssertionError("openai client should not be used"),
        ), patch("pico.cli.AnthropicCompatibleModelClient") as mock_anthropic:
            fake_client = mock_anthropic.return_value
            agent = mini_pkg.build_agent(args)

    mock_anthropic.assert_called_once()
    assert mock_anthropic.call_args.kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert mock_anthropic.call_args.kwargs["base_url"] == "https://www.right.codes/claude/v1"
    assert mock_anthropic.call_args.kwargs["api_key"] == "sk-openai-fallback"
    assert agent.model_client is fake_client


def test_build_agent_uses_anthropic_default_model_when_env_is_missing(tmp_path):
    args = mini_pkg.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--provider", "anthropic"])

    with patch.dict(
        os.environ,
        {},
        clear=False,
    ), patch("pico.cli.DEFAULT_ENV_FILE", tmp_path / "missing.env"):
        os.environ.pop("ANTHROPIC_MODEL", None)
        with patch("pico.cli.AnthropicCompatibleModelClient") as mock_anthropic:
            mini_pkg.build_agent(args)

    assert mock_anthropic.call_args.kwargs["model"] == "claude-sonnet-4-6"


def test_build_agent_uses_openai_provider_by_default(tmp_path):
    args = mini_pkg.build_arg_parser().parse_args(["--cwd", str(tmp_path)])

    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE": "https://www.right.codes/codex/v1",
            "OPENAI_API_KEY": "sk-test",
        },
        clear=False,
    ), patch("pico.cli.DEFAULT_ENV_FILE", tmp_path / "missing.env"):
        with patch(
            "pico.cli.OllamaModelClient",
            side_effect=AssertionError("ollama client should not be used"),
        ), patch("pico.cli.OpenAICompatibleModelClient") as mock_openai:
            fake_client = mock_openai.return_value
            agent = mini_pkg.build_agent(args)

    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["model"] == "gpt-5.4"
    assert mock_openai.call_args.kwargs["base_url"] == "https://www.right.codes/codex/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "sk-test"
    assert agent.model_client is fake_client


def test_successful_run_persists_run_artifacts_and_stop_reason(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Finished.</final>",
        ],
    )

    assert agent.ask("Do the thing") == "Finished."

    runs_root = tmp_path / ".pico" / "runs"
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    task_state = json.loads((run_dir / "task_state.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    trace_lines = (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()

    assert task_state["task_id"] != task_state["run_id"]
    assert run_dir.name == task_state["run_id"]
    assert (run_dir / "task_state.json").exists()
    assert (run_dir / "trace.jsonl").exists()
    assert (run_dir / "report.json").exists()
    assert task_state["stop_reason"] == "final_answer_returned"
    assert task_state["final_answer"] == "Finished."
    assert report["stop_reason"] == "final_answer_returned"
    assert report["task_state"]["stop_reason"] == "final_answer_returned"
    assert report["run_id"] == task_state["run_id"]
    trace_events = [json.loads(line)["event"] for line in trace_lines]
    assert trace_events[0] == "run_started"
    assert trace_events[-1] == "run_finished"
    assert trace_events.count("prompt_built") == 2
    assert "tool_executed" in trace_events


def test_trace_and_report_redact_secret_env_values(tmp_path):
    secret = "sk-test-secret-123"
    with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=True):
        agent = build_agent(
            tmp_path,
            [
                '<tool>{"name":"run_shell","args":{"command":"printf \'%s\' \'sk-test-secret-123\'","timeout":20}}</tool>',
                "<final>Masked.</final>",
            ],
        )

        assert agent.ask("Mask the secret") == "Masked."

    runs_root = tmp_path / ".pico" / "runs"
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    report_text = (run_dir / "report.json").read_text(encoding="utf-8")
    trace_events = [json.loads(line) for line in trace_text.splitlines()]

    assert secret not in trace_text
    assert secret not in report_text

    prompt_events = [event for event in trace_events if event["event"] == "prompt_built"]
    assert prompt_events
    assert prompt_events[0]["prompt_metadata"]["secret_env_count"] >= 1
    assert "OPENAI_API_KEY" in prompt_events[0]["prompt_metadata"]["secret_env_names"]

    tool_events = [event for event in trace_events if event["event"] == "tool_executed"]
    assert tool_events
    assert "<redacted>" in tool_events[0]["args"]["command"]
    assert "<redacted>" in tool_events[0]["result"]


def test_prompt_budget_metadata_records_budget_decisions(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    agent.memory.append_note("alpha episodic note " + ("A" * 120), tags=("recall",), created_at="2026-04-07T10:00:00+00:00")
    agent.memory.append_note("beta episodic recall note " + ("B" * 120), created_at="2026-04-07T10:01:00+00:00")
    agent.memory.append_note("gamma episodic note " + ("C" * 120), tags=("recall",), created_at="2026-04-07T10:02:00+00:00")

    for index in range(4):
        agent.record(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"history-{index}-" + ("A" * 240),
                "created_at": f"2026-04-07T10:0{index}:00+00:00",
            }
        )

    agent.context_manager.total_budget = 1000
    agent.context_manager.section_budgets = {
        "prefix": 80,
        "memory": 80,
        "history": 80,
    }

    assert agent.ask("recall") == "Done."

    trace_events = [
        json.loads(line)
        for line in (agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines())
    ]
    prompt_events = [event for event in trace_events if event["event"] == "prompt_built"]
    assert prompt_events
    metadata = prompt_events[0]["prompt_metadata"]

    assert "Relevant memory:" not in agent.model_client.prompts[0]
    assert metadata["relevant_memory"]["selected_count"] == 0
    assert metadata["relevant_memory"]["rendered_notes"] == []
    assert metadata["section_order"] == ["system", "history", "current_request"]
    assert metadata["current_request"]["text"] == "recall"
    assert metadata["current_request"]["rendered_chars"] == len("recall")


def test_prompt_metadata_uses_context_builder_system_hash(tmp_path):
    agent = build_agent(tmp_path, [])

    first = agent.prompt_metadata("first", "")
    second = agent.prompt_metadata("second", "")

    assert first["system_hash"] == second["system_hash"]
    assert second["prompt_cache_key"] == second["system_hash"]

    (tmp_path / "README.md").write_text("demo changed\n", encoding="utf-8")

    third = agent.prompt_metadata("third", "")

    assert third["system_hash"] == second["system_hash"]
    assert third["prompt_cache_key"] == third["system_hash"]


def test_runtime_checkpoint_restores_completed_and_pending_tool_messages(tmp_path):
    agent = build_agent(tmp_path, ["<final>Recovered.</final>"])
    agent.record({"role": "user", "content": "start", "created_at": "2026-04-14T09:00:00+00:00"})
    checkpoint = {
        "checkpoint_id": "ckpt_restore",
        "assistant_message": {"role": "assistant", "content": "Working on it.", "created_at": "2026-04-14T09:00:01+00:00"},
        "completed_tool_results": [
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": "README.md"},
                "content": "demo",
                "created_at": "2026-04-14T09:00:02+00:00",
            }
        ],
        "pending_tool_calls": [
            {"id": "tool_2", "function": {"name": "write_file", "arguments": {"path": "notes.txt"}}}
        ],
    }
    agent.session["session_metadata"]["runtime_checkpoint"] = checkpoint
    agent.session_store.save(agent.session)

    resumed = MiniAgent.from_session(
        model_client=FakeModelClient(["<final>Recovered.</final>"]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue") == "Recovered."
    contents = [item["content"] for item in resumed.session["history"] if item["role"] in {"assistant", "tool"}]
    assert "Working on it." in contents
    assert "demo" in contents
    assert any("Task interrupted before this tool finished." in item for item in contents)
    assert resumed.last_prompt_metadata["resume_status"] == "runtime-checkpoint-restored"
    assert "runtime_checkpoint" not in resumed.session["session_metadata"]


def test_checkpoint_module_materializes_pending_tool_backfill_without_runtime(tmp_path):
    session = {
        "id": "session-1",
        "history": [{"role": "user", "content": "start", "created_at": "1"}],
        "session_metadata": {
            "runtime_checkpoint": {
                "checkpoint_id": "ckpt_restore",
                "assistant_message": {"role": "assistant", "content": "Working"},
                "completed_tool_results": [
                    {"role": "tool", "name": "read_file", "content": "done"}
                ],
                "pending_tool_calls": [
                    {"id": "tool_2", "function": {"name": "write_file", "arguments": {"path": "notes.txt"}}}
                ],
            },
            "pending_user_turn": True,
        },
    }

    state = checkpointlib.restore_interrupted_turn(session)

    assert state["status"] == "runtime-checkpoint-restored"
    assert "runtime_checkpoint" not in session["session_metadata"]
    assert "pending_user_turn" not in session["session_metadata"]
    assert session["history"][-1]["tool_call_id"] == "tool_2"
    assert "Task interrupted before this tool finished." in session["history"][-1]["content"]


def test_pending_user_turn_restores_error_reply_before_new_turn(tmp_path):
    agent = build_agent(tmp_path, ["<final>Recovered.</final>"])
    agent.record({"role": "user", "content": "unfinished request", "created_at": "2026-04-14T09:00:00+00:00"})
    agent.session["session_metadata"]["pending_user_turn"] = True
    agent.session_store.save(agent.session)

    resumed = MiniAgent.from_session(
        model_client=FakeModelClient(["<final>Recovered.</final>"]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue") == "Recovered."
    assistant_messages = [item["content"] for item in resumed.session["history"] if item["role"] == "assistant"]
    assert any("Task interrupted before a response was generated." in item for item in assistant_messages)
    assert resumed.last_prompt_metadata["resume_status"] == "pending-user-turn-restored"
    assert "pending_user_turn" not in resumed.session["session_metadata"]


def test_run_shell_nonzero_with_workspace_change_is_recorded_as_partial_success(tmp_path):
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "run_shell",
        {
            "command": "printf 'changed\\n' > README.md && exit 1",
            "timeout": 20,
        },
    )

    assert "exit_code: 1" in result
    assert agent._last_tool_result_metadata["tool_status"] == "partial_success"
    assert agent._last_tool_result_metadata["affected_paths"] == ["README.md"]
    assert agent._last_tool_result_metadata["workspace_changed"] is True


def test_successful_turn_clears_runtime_checkpoint_and_pending_user_turn(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":1}}</tool>',
            "<final>Done.</final>",
        ],
    )

    assert agent.ask("Inspect README") == "Done."
    metadata = agent.session["session_metadata"]

    assert "runtime_checkpoint" not in metadata
    assert "pending_user_turn" not in metadata


def test_write_file_trace_records_minimum_tool_contract_fields(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"write_file","args":{"path":"notes.txt","content":"hello\\n"}}</tool>',
            "<final>Done.</final>",
        ],
    )

    assert agent.ask("Create notes.txt") == "Done."

    trace_events = [
        json.loads(line)
        for line in agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines()
    ]
    tool_event = [event for event in trace_events if event["event"] == "tool_executed"][-1]

    assert tool_event["name"] == "write_file"
    assert tool_event["risk_level"] == "high"
    assert tool_event["read_only"] is False
    assert tool_event["tool_status"] == "ok"
    assert tool_event["affected_paths"] == ["notes.txt"]
    assert tool_event["workspace_changed"] is True
    assert tool_event["diff_summary"] == ["created:notes.txt"]


def test_partial_success_creates_process_note_for_exploration_history(tmp_path):
    agent = build_agent(tmp_path, [])

    agent.run_tool(
        "run_shell",
        {
            "command": "printf 'changed\\n' > README.md && exit 1",
            "timeout": 20,
        },
    )

    process_notes = [
        note
        for note in agent.memory.to_dict()["episodic_notes"]
        if note.get("kind") == "process"
    ]

    assert process_notes
    assert process_notes[-1]["text"] == "run_shell partial_success on README.md; inspect diff before retry"
    assert "partial_success" in process_notes[-1]["tags"]
    assert "README.md" in process_notes[-1]["tags"]


def test_final_answer_does_not_auto_promote_durable_memory(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>Project convention: Use constrained tools instead of guessing.\n"
            "Decision: Keep durable memory topic-based and lightweight.</final>",
        ],
    )

    answer = agent.ask(
        "Capture the stable facts you already discovered as durable memory. "
        "Respond with exactly the long-term facts."
    )

    memory_text = (tmp_path / ".pico" / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))

    assert "Project convention:" in answer
    assert "Use constrained tools instead of guessing." not in memory_text
    assert "durable_promotions" not in report
    assert "durable_rejections" not in report
    assert "durable_superseded" not in report


def test_final_answer_does_not_auto_promote_chinese_durable_memory(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>项目约定：优先使用受约束工具，不要靠猜。\n"
            "决策：持久记忆保持轻量、按 topic 管理。</final>",
        ],
    )

    answer = agent.ask("请把下面这些稳定事实记住，作为长期记忆保存下来。")

    memory_text = (tmp_path / ".pico" / "memory" / "MEMORY.md").read_text(encoding="utf-8")

    assert "项目约定：" in answer
    assert "优先使用受约束工具，不要靠猜。" not in memory_text


def test_final_answer_does_not_auto_promote_secret_shaped_lines(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>Project convention: Use constrained tools instead of guessing.\n"
            "Dependency: API key is sk-live-secret-abc.\n"
            "Decision: Current goal is fix flaky tests.</final>",
        ],
    )

    agent.ask("Capture these stable facts into durable memory.")

    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    memory_text = (tmp_path / ".pico" / "memory" / "MEMORY.md").read_text(encoding="utf-8")

    assert "durable_promotions" not in report
    assert "durable_rejections" not in report
    assert "Use constrained tools instead of guessing." not in memory_text
    assert "API key is sk-live-secret-abc" not in memory_text


def test_agent_records_model_cache_metadata_in_last_prompt_metadata(tmp_path):
    class CacheAwareFakeModelClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            self.last_completion_metadata = {
                "prompt_cache_supported": True,
                "cached_tokens": 512,
                "cache_hit": True,
                "input_tokens": 1024,
            }
            return super().complete(prompt, max_new_tokens, **kwargs)

    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".pico" / "sessions")
    agent = MiniAgent(
        model_client=CacheAwareFakeModelClient(["<final>Done.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )

    assert agent.ask("Cache aware run") == "Done."

    assert agent.last_prompt_metadata["prompt_cache_supported"] is True
    assert agent.last_prompt_metadata["cached_tokens"] == 512
    assert agent.last_prompt_metadata["cache_hit"] is True
    assert agent.last_prompt_metadata["system_hash"]
    assert agent.last_prompt_metadata["prompt_cache_key"] == agent.last_prompt_metadata["system_hash"]


def test_recent_transcript_entries_stay_richer_than_older_ones(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    old_text = "OLD-" + ("A" * 320)
    recent_text = "RECENT-" + ("B" * 320)

    agent.record({"role": "user", "content": old_text, "created_at": "2026-04-07T09:00:00+00:00"})
    agent.record({"role": "assistant", "content": old_text, "created_at": "2026-04-07T09:01:00+00:00"})
    agent.record({"role": "user", "content": recent_text, "created_at": "2026-04-07T09:02:00+00:00"})
    agent.record({"role": "assistant", "content": recent_text, "created_at": "2026-04-07T09:03:00+00:00"})
    agent.record({"role": "user", "content": recent_text, "created_at": "2026-04-07T09:04:00+00:00"})
    agent.record({"role": "assistant", "content": recent_text, "created_at": "2026-04-07T09:05:00+00:00"})
    agent.record({"role": "user", "content": recent_text, "created_at": "2026-04-07T09:06:00+00:00"})
    agent.record({"role": "assistant", "content": recent_text, "created_at": "2026-04-07T09:07:00+00:00"})

    assert agent.ask("Check the transcript") == "Done."

    prompt = agent.model_client.prompts[-1]

    assert recent_text in prompt
    assert old_text not in prompt


def test_public_api_exports_resolve_through_package_path():
    assert callable(build_welcome)
    assert callable(build_tools_help)
    assert FakeModelClient is not None
    assert MiniAgent is not None
    assert OllamaModelClient is not None
    assert SessionStore is not None
    assert WorkspaceContext is not None
    assert Path(mini_pkg.__file__).as_posix().endswith("/pico/__init__.py")


def test_reviewer_skeleton_docs_exist():
    review_pack = Path("docs/review-pack/README.md")
    architecture = Path("docs/architecture/agent-harness-v1-overview.md")

    assert review_pack.exists()
    assert architecture.exists()

    review_text = review_pack.read_text(encoding="utf-8")
    assert "Project pitch" in review_text
    assert "Architecture map" in review_text
    assert "Benchmark evidence" in review_text
    assert "Sample run artifact list" in review_text

    architecture_text = architecture.read_text(encoding="utf-8")
    assert "Agent Harness v1" in architecture_text
    assert "task state" in architecture_text.lower()


def test_package_import_surface_includes_cli_entrypoints():
    assert callable(mini_pkg.main)
    assert callable(mini_pkg.build_agent)
    assert callable(mini_pkg.build_arg_parser)
    assert callable(mini_pkg.build_tools_help)


def test_build_tools_help_lists_registered_model_callable_tools(tmp_path):
    agent = build_agent(tmp_path, [])

    output = build_tools_help(agent)

    assert "Model-callable tools registered" in output
    assert "not pico> commands" in output
    assert "- list_dir:" in output
    assert "- read_file:" in output
    assert "- exec:" in output
    assert "multi_tool_use.parallel" not in output


def test_module_execution_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "pico", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
