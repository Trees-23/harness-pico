import asyncio
import inspect
import json

from pico import FakeModelClient, MiniAgent, ModelResponse, SessionStore, ToolCallRequest, WorkspaceContext
from pico.models import _anthropic_payload_from_messages, _openai_responses_payload_from_messages
from pico.runner import PicoModelRequest, PicoParsedResponse, PicoRunner, PicoRunResult, PicoRunSpec, PicoToolCall


def build_agent(tmp_path, outputs=None, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    return MiniAgent(
        model_client=FakeModelClient(outputs or []),
        workspace=workspace,
        session_store=SessionStore(tmp_path / ".pico" / "sessions"),
        approval_policy=kwargs.pop("approval_policy", "auto"),
        **kwargs,
    )


def test_runtime_ask_delegates_to_pico_runner(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])

    assert isinstance(agent.runner, PicoRunner)
    assert agent.ask("Finish") == "Done."


def test_loop_ask_runs_compaction_before_building_run_spec(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    events = []

    def wrap(name):
        original = getattr(agent, name)

        def wrapped(*args, **kwargs):
            events.append(name)
            return original(*args, **kwargs)

        return wrapped

    for name in (
        "prepare_session_for_turn",
        "record_user_turn",
        "start_task_run_for_turn",
        "run_history_compaction_for_turn",
        "build_run_spec",
    ):
        monkeypatch.setattr(agent, name, wrap(name))

    assert agent.ask("Finish") == "Done."

    assert events[:5] == [
        "prepare_session_for_turn",
        "record_user_turn",
        "start_task_run_for_turn",
        "run_history_compaction_for_turn",
        "build_run_spec",
    ]
    assert events[-1] == "run_history_compaction_for_turn"
    assert events.count("run_history_compaction_for_turn") == 2


def test_runner_run_returns_contract_result(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])

    result = agent.runner.run(agent.build_run_spec("Finish"))

    assert isinstance(result, PicoRunResult)
    assert result.final_content == "Done."
    assert result.stop_reason == "completed"
    assert result.messages[-1]["role"] == "assistant"


def test_runner_native_tool_calls_use_registry_executor_path(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCallRequest("call_1", "read_file", {"path": "README.md", "start": "1", "end": "1"})
                ],
            ),
            "<final>Done.</final>",
        ],
    )

    assert agent.ask("Inspect README") == "Done."
    assistant_events = [item for item in agent.session["history"] if item["role"] == "assistant"]
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]

    assert assistant_events[0]["tool_calls"][0]["id"] == "call_1"
    assert assistant_events[0]["tool_calls"][0]["function"]["name"] == "read_file"
    assert tool_events[0]["tool_call_id"] == "call_1"
    assert tool_events[0]["name"] == "read_file"
    assert "# README.md" in tool_events[0]["content"]
    assert agent.last_prompt_metadata["tool_call_protocol"] == "native"


def test_runner_legacy_tool_calls_use_registry_executor_path(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"README.md","start":"1","end":"1"}}</tool>',
            "<final>Done.</final>",
        ],
    )

    assert agent.ask("Inspect README") == "Done."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]

    assert tool_events[0]["name"] == "read_file"
    assert tool_events[0]["args"] == {"path": "README.md", "start": "1", "end": "1"}
    assert "# README.md" in tool_events[0]["content"]
    assert agent._last_tool_result_metadata["tool_status"] == "ok"
    assert agent.last_prompt_metadata["tool_call_protocol"] == "native"


def test_runner_native_repeated_read_file_variants_finish_with_previous_result(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCallRequest("call_1", "read_file", {"path": "README.md", "start": "1", "end": "200"})
                ],
            ),
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCallRequest("call_2", "read_file", {"path": "README.md", "start": "1", "end": "260"})
                ],
            ),
        ],
    )

    assert agent.ask("Read README") == "# README.md\n   1: demo"
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assistant_events = [item for item in agent.session["history"] if item["role"] == "assistant"]

    assert tool_events[0]["tool_call_id"] == "call_1"
    assert len(tool_events) == 1
    assert assistant_events[-1]["content"] == "# README.md\n   1: demo"


def test_runner_finishes_after_repeated_write_file_without_second_execution(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCallRequest("call_1", "write_file", {"path": "docs/1.txt", "content": "hello"})
                ],
            ),
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCallRequest("call_2", "write_file", {"path": "docs/1.txt", "content": "hello"})
                ],
            ),
        ],
    )

    assert agent.ask("Write hello") == "Done. Wrote docs/1.txt."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    task_state = json.loads((agent.current_run_dir / "task_state.json").read_text(encoding="utf-8"))

    assert len(tool_events) == 1
    assert tool_events[0]["content"] == "wrote docs/1.txt (5 chars)"
    assert task_state["status"] == "completed"
    assert task_state["stop_reason"] == "final_answer_returned"


def test_runner_model_error_finishes_run_and_clears_pending_state(tmp_path):
    agent = build_agent(tmp_path, [])

    answer = agent.ask("This will exhaust fake outputs")

    assert answer.startswith("Model request failed: RuntimeError: fake model ran out of outputs")
    assert "pending_user_turn" not in agent.session["session_metadata"]
    assert "runtime_checkpoint" not in agent.session["session_metadata"]
    task_state = json.loads((agent.current_run_dir / "task_state.json").read_text(encoding="utf-8"))
    trace_events = [
        json.loads(line)["event"]
        for line in (agent.current_run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert task_state["status"] == "failed"
    assert task_state["stop_reason"] == "model_error"
    assert trace_events[-2:] == ["model_error", "run_finished"]


def test_runner_run_returns_model_error_result(tmp_path):
    agent = build_agent(tmp_path, [])

    result = agent.runner.run(agent.build_run_spec("This will exhaust fake outputs"))

    assert result.stop_reason == "error"
    assert result.error is not None
    assert result.error.startswith("Model request failed: RuntimeError: fake model ran out of outputs")


def test_runner_result_stop_reasons_use_nanobot_terms(tmp_path):
    agent = build_agent(tmp_path, [])
    runner = agent.runner

    assert runner._to_nanobot_stop_reason("final_answer_returned") == "completed"
    assert runner._to_nanobot_stop_reason("model_error") == "error"
    assert runner._to_nanobot_stop_reason("step_limit_reached") == "max_iterations"
    assert runner._to_nanobot_stop_reason("retry_limit_reached") == "max_iterations"


def test_runner_lifecycle_writes_are_behind_adapter_methods():
    source = inspect.getsource(PicoRunner.ask)

    assert "write_report_for_runner" not in source
    assert "write_task_state_for_runner" not in source
    assert "create_checkpoint_for_runner" not in source
    assert "clear_pending_user_turn_for_runner" not in source
    assert "clear_runtime_checkpoint_for_runner" not in source


def test_runner_main_loop_does_not_call_legacy_build_model_prompt():
    source = inspect.getsource(PicoRunner.ask)

    assert "build_model_prompt" not in source


def test_runner_complete_model_uses_spec_contract(tmp_path):
    agent = build_agent(tmp_path, ["agent output"])
    spec_model = FakeModelClient(["spec output"])
    spec = PicoRunSpec(
        model_client=spec_model,
        tool_registry=agent.tool_registry,
        tool_executor=agent.tool_executor,
        max_iterations=1,
        max_new_tokens=7,
        initial_messages=[],
    )
    request = PicoModelRequest(prompt="prompt", prompt_metadata={})

    response = agent.runner._complete_model(spec, request)

    assert response.content == "spec output"
    assert spec_model.prompts == ["prompt"]
    assert agent.model_client.prompts == []
    assert request.prompt_metadata["tool_call_protocol"] == "native"


def test_fake_model_client_records_messages_requests():
    client = FakeModelClient(["ok"])
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]

    response = client.complete_messages_with_tools(messages, 10, tools=[])

    assert response.content == "ok"
    assert client.messages == [messages]
    assert client.prompts == ["SYSTEM:\nsystem\n\nUSER:\nhello"]


def test_openai_messages_payload_preserves_chat_structure():
    payload = _openai_responses_payload_from_messages(
        model="gpt-test",
        messages=[
            {"role": "system", "content": "system policy"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "file body"},
        ],
        max_new_tokens=17,
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        temperature=0,
        prompt_cache_key="cache-key",
        prompt_cache_retention="in_memory",
        supports_prompt_cache=True,
    )

    assert payload["model"] == "gpt-test"
    assert payload["instructions"] == "system policy"
    assert payload["max_output_tokens"] == 17
    assert payload["tools"][0]["function"]["name"] == "read_file"
    assert payload["tool_choice"] == "auto"
    assert payload["prompt_cache_key"] == "cache-key"
    assert payload["prompt_cache_retention"] == "in_memory"
    assert payload["input"][0] == {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    assert payload["input"][1]["role"] == "assistant"
    assert payload["input"][1]["tool_calls"][0]["id"] == "call_1"
    assert payload["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "file body",
    }


def test_anthropic_messages_payload_preserves_chat_structure():
    payload = _anthropic_payload_from_messages(
        model="claude-test",
        messages=[
            {"role": "system", "content": "system policy"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": {"path": "README.md"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "file body"},
        ],
        max_new_tokens=17,
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        temperature=0,
    )

    assert payload["model"] == "claude-test"
    assert payload["system"] == "system policy"
    assert payload["max_tokens"] == 17
    assert payload["tools"][0]["name"] == "read_file"
    assert payload["tool_choice"] == {"type": "auto"}
    assert payload["messages"][0] == {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read_file",
        "input": {"path": "README.md"},
    }
    assert payload["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "file body"}],
    }


def test_runner_uses_model_response_metadata(tmp_path):
    response = ModelResponse(
        "<final>Done.</final>",
        metadata={"input_tokens": 10, "cached_tokens": 3, "cache_hit": True},
        usage={"input_tokens": 10, "cached_tokens": 3},
    )
    agent = build_agent(tmp_path, [response])

    assert agent.ask("Finish") == "Done."

    assert agent.last_completion_metadata["input_tokens"] == 10
    assert agent.last_completion_metadata["cached_tokens"] == 3
    assert agent.last_prompt_metadata["cache_hit"] is True


def test_runner_parse_model_response_unifies_native_and_legacy(tmp_path):
    agent = build_agent(tmp_path)

    native = agent.runner._parse_model_response(
        ModelResponse("", [ToolCallRequest("call_1", "read_file", {"path": "README.md"})]),
        parse_legacy=agent.parse,
    )
    legacy_final = agent.runner._parse_model_response(
        ModelResponse("<final>Done.</final>"),
        parse_legacy=agent.parse,
    )
    malformed = agent.runner._parse_model_response(
        ModelResponse("<tool>{bad json}</tool>"),
        parse_legacy=agent.parse,
    )

    assert isinstance(native, PicoParsedResponse)
    assert native.kind == "tool_calls"
    assert native.tool_calls[0].id == "call_1"
    assert legacy_final.kind == "final"
    assert legacy_final.payload == "Done."
    assert malformed.kind == "retry"


def test_runner_emits_progress_events(tmp_path):
    events = []
    agent = build_agent(tmp_path, ["<final>Done.</final>"], progress_callback=lambda event, payload: events.append((event, payload)))

    assert agent.ask("Finish") == "Done."

    assert events == [("model_requested", {"attempts": 1, "tool_steps": 0})]


def test_runner_uses_governed_history_for_model_prompt_without_rewriting_session(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    agent.record({"role": "user", "content": "start", "created_at": "1"})
    agent.record(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "missing_1", "function": {"name": "read_file", "arguments": {"path": "README.md"}}}],
            "created_at": "2",
        }
    )

    assert agent.ask("Continue") == "Done."

    assert not any(
        item.get("tool_call_id") == "missing_1" and "Task interrupted" in item.get("content", "")
        for item in agent.session["history"]
    )
    assert "Task interrupted before this tool finished" in str(agent.model_client.messages[0])


def test_runner_main_prompt_omits_relevant_memory_section(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])

    assert agent.ask("Finish") == "Done."

    assert "Relevant memory:" not in agent.model_client.prompts[0]
    assert "Relevant memory:" not in str(agent.model_client.messages[0])


def test_runner_partitions_tool_calls_by_tool_attributes(tmp_path):
    runner = build_agent(tmp_path).runner
    calls = [
        PicoToolCall("1", "list_dir", {}),
        PicoToolCall("2", "read_file", {"path": "README.md"}),
        PicoToolCall("3", "write_file", {"path": "a.txt", "content": "a"}),
        PicoToolCall("4", "grep", {"pattern": "demo"}),
        PicoToolCall("5", "exec", {"command": "echo hi", "timeout": 20}),
        PicoToolCall("6", "list_dir", {}),
    ]

    batches = runner.partition_tool_batches(calls)

    assert [[call.name for call in batch] for batch in batches] == [
        ["list_dir", "read_file"],
        ["write_file"],
        ["grep"],
        ["exec"],
        ["list_dir"],
    ]


def test_runner_native_tool_calls_execute_in_attribute_batches(tmp_path, monkeypatch):
    agent = build_agent(
        tmp_path,
        [
            ModelResponse(
                "",
                [
                    ToolCallRequest("call_1", "list_dir", {}),
                    ToolCallRequest("call_2", "read_file", {"path": "README.md"}),
                    ToolCallRequest("call_3", "write_file", {"path": "out.txt", "content": "x"}),
                ],
            ),
            "<final>Done.</final>",
        ],
    )
    seen_batches = []
    original = agent.runner.partition_tool_batches

    def spy_partition(tool_calls, *, concurrent_tools=True):
        batches = original(tool_calls, concurrent_tools=concurrent_tools)
        seen_batches.extend([[call.name for call in batch] for batch in batches])
        return batches

    monkeypatch.setattr(agent.runner, "partition_tool_batches", spy_partition)

    assert agent.ask("Inspect and write") == "Done."

    assert seen_batches == [["list_dir", "read_file"], ["write_file"]]


def test_runner_tool_errors_are_model_consumable_by_default(tmp_path):
    runner = build_agent(tmp_path, approval_policy="never").runner

    result = asyncio.run(
        runner.execute_tool_calls(
            [PicoToolCall("1", "exec", {"command": "echo hi", "timeout": 20})],
        )
    )

    assert result == ["error: approval denied for exec"]


def test_runner_fail_on_tool_error_raises(tmp_path):
    runner = PicoRunner(build_agent(tmp_path, approval_policy="never"), fail_on_tool_error=True)

    try:
        asyncio.run(
            runner.execute_tool_calls(
                [PicoToolCall("1", "exec", {"command": "echo hi", "timeout": 20})],
            )
        )
    except RuntimeError as exc:
        assert "approval denied" in str(exc)
    else:
        raise AssertionError("fail_on_tool_error did not raise")


def test_runner_normalizes_empty_and_persists_large_tool_results(tmp_path):
    runner = PicoRunner(build_agent(tmp_path), max_tool_result_chars=40)

    empty = runner.normalize_tool_result("tool_1", "demo", "")
    large = runner.normalize_tool_result("tool_2", "demo", "x" * 100)

    assert empty == "[demo returned empty content]"
    assert "Tool result too large" in large
    assert ".pico/tool-results" in large
    assert (tmp_path / ".pico" / "tool-results" / runner.agent.session["id"] / "tool_2.txt").read_text(
        encoding="utf-8"
    ) == "x" * 100


def test_runner_run_applies_spec_tool_result_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            ModelResponse(
                "",
                [ToolCallRequest("call_1", "read_file", {"path": "README.md", "start": "1", "end": "1"})],
            ),
            "<final>Done.</final>",
        ],
    )
    spec = agent.build_run_spec("Inspect README")
    spec.max_tool_result_chars = 20

    result = agent.runner.run(spec)
    tool_message = next(item for item in result.messages if item["role"] == "tool")

    assert "Tool result too large" in tool_message["content"]
    assert (tmp_path / ".pico" / "tool-results" / agent.session["id"] / "call_1.txt").exists()


def test_runner_cleans_orphans_backfills_missing_and_microcompacts(tmp_path):
    runner = build_agent(tmp_path).runner
    messages = [
        {"role": "tool", "tool_call_id": "orphan", "name": "read_file", "content": "drop"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "read_file"}},
                {"id": "call_2", "function": {"name": "grep"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "ok"},
    ]
    for index in range(12):
        messages.append(
            {
                "role": "tool",
                "name": "read_file",
                "content": "x" * 600,
            }
        )

    governed = runner.govern_messages_for_model(messages)

    assert all(item.get("tool_call_id") != "orphan" for item in governed)
    assert any(
        item.get("tool_call_id") == "call_2"
        and "Task interrupted before this tool finished" in item.get("content", "")
        for item in governed
    )
    assert any(item.get("content") == "[read_file result omitted from context]" for item in governed)


def test_runner_backfills_missing_tool_results_independently(tmp_path):
    runner = build_agent(tmp_path).runner
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "read_file"}}],
        }
    ]

    updated = runner._backfill_missing_tool_results(messages)

    assert updated[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "read_file",
        "content": "Error: Task interrupted before this tool finished.",
    }


def test_runner_microcompact_uses_nanobot_tool_set(tmp_path):
    runner = build_agent(tmp_path).runner
    messages = [
        {"role": "tool", "name": "grep", "content": "x" * 600}
        for _ in range(11)
    ]

    compacted = runner.microcompact(messages)

    assert compacted[0]["content"] == "[grep result omitted from context]"


def test_runner_snip_history_keeps_system_and_recent_user_boundary(tmp_path):
    agent = build_agent(tmp_path)
    runner = agent.runner
    spec = agent.build_run_spec("now")
    spec.context_window_tokens = 80
    spec.max_new_tokens = 1
    spec.context_block_limit = 60
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old " * 80},
        {"role": "assistant", "content": "old answer " * 80},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]

    snipped = runner._snip_history(spec, messages)

    assert snipped[0] == {"role": "system", "content": "system"}
    assert snipped[1]["role"] == "user"
    assert snipped[1]["content"] == "recent question"
    assert snipped[-1]["content"] == "recent answer"
    assert all("old" not in str(message.get("content")) for message in snipped[1:])
