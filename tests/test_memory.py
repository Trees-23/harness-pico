import asyncio
from pathlib import Path
from unittest.mock import patch

from pico.auto_compact import AutoCompact
from pico.cron import CronJob, CronPayload, CronSchedule, CronService
from pico.dream import Dream
from pico.consolidator import Consolidator, split_history_for_compaction
from pico.context_manager import ContextBuilder
from pico.models import FakeModelClient
from pico.memory import MemoryStore, LayeredMemory, normalize_session_summary
from pico.session_manager import Session, SessionManager


def test_working_memory_tracks_summary_and_recent_files():
    memory = LayeredMemory()

    memory.set_task_summary("Investigate flaky tests")
    memory.remember_file("README.md")
    memory.remember_file("src/app.py")
    memory.remember_file("README.md")

    snapshot = memory.to_dict()

    assert snapshot["working"]["task_summary"] == "Investigate flaky tests"
    assert snapshot["working"]["recent_files"] == ["src/app.py", "README.md"]
    assert snapshot["task"] == "Investigate flaky tests"
    assert snapshot["files"] == ["src/app.py", "README.md"]


def test_episodic_notes_append_and_retrieve_deterministically():
    memory = LayeredMemory()

    memory.append_note("Exact tag note", tags=("recall",), created_at="2026-04-07T10:00:00+00:00")
    memory.append_note("Keyword overlap note about memory", created_at="2026-04-07T10:01:00+00:00")
    memory.append_note("Newest unrelated note", created_at="2026-04-07T10:02:00+00:00")
    memory.append_note("Older unrelated note", created_at="2026-04-07T09:59:00+00:00")

    snapshot = memory.to_dict()
    assert [note["text"] for note in snapshot["episodic_notes"]] == [
        "Exact tag note",
        "Keyword overlap note about memory",
        "Newest unrelated note",
        "Older unrelated note",
    ]
    assert snapshot["notes"] == [
        "Exact tag note",
        "Keyword overlap note about memory",
        "Newest unrelated note",
        "Older unrelated note",
    ]

    lines = [line for line in memory.retrieval_view("recall memory", limit=4).splitlines() if line.startswith("- ")]
    assert lines == [
        "- Exact tag note",
        "- Keyword overlap note about memory",
    ]


def test_file_summaries_use_canonical_paths_and_freshness(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\n", encoding="utf-8")
    memory = LayeredMemory(workspace_root=tmp_path)

    memory.set_file_summary("./sample.txt", "sample.txt: alpha")
    memory.remember_file("./sample.txt")
    snapshot = memory.to_dict()["file_summaries"]["sample.txt"]

    assert snapshot["summary"] == "sample.txt: alpha"
    assert snapshot["freshness"]

    assert "sample.txt: alpha" in memory.render_memory_text()
    file_path.write_text("beta\n", encoding="utf-8")
    assert "sample.txt: alpha" not in memory.render_memory_text()

    memory.invalidate_file_summary("sample.txt")

    assert "sample.txt" not in memory.to_dict()["file_summaries"]


def test_process_notes_keep_kind_and_latest_duplicate_wins():
    memory = LayeredMemory()

    memory.append_note(
        "Shell partial success on README.md; inspect diff before retry",
        tags=("process", "partial_success"),
        created_at="2026-04-07T10:00:00+00:00",
        kind="process",
    )
    memory.append_note(
        "Shell partial success on README.md; inspect diff before retry",
        tags=("process", "partial_success"),
        created_at="2026-04-07T10:01:00+00:00",
        kind="process",
    )

    notes = memory.to_dict()["episodic_notes"]

    assert len(notes) == 1
    assert notes[0]["kind"] == "process"
    assert notes[0]["created_at"] == "2026-04-07T10:01:00+00:00"


def test_memory_state_does_not_expose_durable_topics(tmp_path):
    memory_root = tmp_path / ".pico" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "MEMORY.md").write_text(
        "# Long-term Memory\n\n"
        "## Project Context\n\n"
        "- [project-conventions] Use constrained tools instead of guessing.\n",
        encoding="utf-8",
    )

    memory = LayeredMemory(workspace_root=tmp_path)

    snapshot = memory.to_dict()
    assert "durable_topics" not in snapshot
    assert "Use constrained tools instead of guessing." in memory.store.get_memory_context()


def test_memory_store_appends_jsonl_summary_records(tmp_path):
    store = MemoryStore(tmp_path)

    cursor = store.append_history("- Continue in runtime.py\n- README.md was already checked")

    payload = (tmp_path / ".pico" / "memory" / "history.jsonl").read_text(encoding="utf-8")

    assert cursor == 1
    assert '"cursor": 1' in payload
    assert '"content": "- Continue in runtime.py\\n- README.md was already checked"' in payload
    assert normalize_session_summary("<final>summary line</final>") == "summary line"
    assert hasattr(store.git, "line_ages")


def test_memory_store_migrates_legacy_root_runtime_tree_into_dot_pico(tmp_path):
    legacy_root = tmp_path / "memory"
    legacy_root.mkdir(parents=True)
    (legacy_root / "MEMORY.md").write_text("# Long-term Memory\n\nlegacy\n", encoding="utf-8")
    (legacy_root / "history.jsonl").write_text('{"cursor": 1, "timestamp": "2026-05-06 10:00", "content": "legacy"}\n', encoding="utf-8")
    (legacy_root / ".cursor").write_text("1", encoding="utf-8")
    (legacy_root / ".dream_cursor").write_text("1", encoding="utf-8")

    store = MemoryStore(tmp_path)

    assert store.memory_file == tmp_path / ".pico" / "memory" / "MEMORY.md"
    assert store.history_file == tmp_path / ".pico" / "memory" / "history.jsonl"
    assert store.read_memory() == "# Long-term Memory\n\nlegacy\n"
    assert store.read_unprocessed_history(0)[0]["content"] == "legacy"


def test_memory_store_sync_materializes_runtime_bootstrap_files_without_overwriting(tmp_path):
    store = MemoryStore(tmp_path)

    assert (tmp_path / ".pico" / "AGENTS.md").exists()
    assert (tmp_path / ".pico" / "TOOLS.md").exists()
    assert (tmp_path / ".pico" / "HEARTBEAT.md").exists()
    agents_before = (tmp_path / ".pico" / "AGENTS.md").read_text(encoding="utf-8")
    (tmp_path / ".pico" / "AGENTS.md").write_text("custom agents\n", encoding="utf-8")

    created = store.sync()

    assert (tmp_path / ".pico" / "AGENTS.md").read_text(encoding="utf-8") == "custom agents\n"
    assert all(path.name != "AGENTS.md" for path in created)
    assert agents_before


def test_consolidator_pipeline_is_decoupled_from_memory_state(tmp_path):
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"history-{index} " + ("X" * 420),
        }
        for index in range(13)
    ]
    archive_store = MemoryStore(tmp_path)
    captured = {}

    def summarize(prompt, max_new_tokens):
        captured["prompt"] = prompt
        captured["max_new_tokens"] = max_new_tokens
        return "- Current goal: Continue editing runtime.py\n- Next step: inspect the recent transcript"

    compactor = Consolidator(archive_store, summarize)
    session = {
        "id": "session-1",
        "history": list(history),
        "last_consolidated": 0,
        "history_archive": {
            "latest_summary": "",
            "latest_cursor": 0,
            "compaction_count": 0,
            "archived_messages": 0,
            "last_compacted_at": "",
        },
    }

    result = compactor.maybe_consolidate_by_tokens(
        session,
        session_summary="",
        task_summary="Continue editing runtime.py",
    )

    assert split_history_for_compaction(history)[0]
    assert result is not None
    assert result.archived_messages > 0
    assert result.summary.startswith("- Current goal")
    assert session["last_consolidated"] > 0
    assert captured["max_new_tokens"] == 256
    assert "Existing session summary" not in captured["prompt"]
    assert "Current task:\nContinue editing runtime.py" in captured["prompt"]
    payload = (tmp_path / ".pico" / "memory" / "history.jsonl").read_text(encoding="utf-8")
    assert '"cursor": 1' in payload


def test_consolidator_estimates_tokens_via_context_builder_messages(tmp_path):
    store = MemoryStore(tmp_path)
    (tmp_path / ".pico" / "AGENTS.md").write_text("Follow AGENTS bootstrap.", encoding="utf-8")
    store.write_memory("# Long-term Memory\n\nRemember the architecture decisions.\n")
    builder = ContextBuilder(tmp_path)
    session = {
        "id": "session-1",
        "updated_at": "2026-05-06T10:00:00+00:00",
        "session_metadata": {"channel": "cli"},
        "history": [
            {"role": "user", "content": "请记住我的项目背景。"},
            {"role": "assistant", "content": "我会记住。"},
        ],
        "last_consolidated": 0,
    }
    compactor = Consolidator(
        store,
        lambda prompt, max_new_tokens: "- summary",
        build_messages=builder.build_messages,
        get_tool_definitions=lambda: [{"name": "patch_file", "parameters": {"type": "object"}}],
    )

    estimated = compactor.estimate_session_prompt_tokens(session, session_summary="之前的摘要")

    assert estimated > 0
    assert "session-1" not in compactor._locks


def test_consolidator_caps_boundary_to_user_turn_with_chunk_limit(tmp_path):
    store = MemoryStore(tmp_path)
    history = [{"role": "user", "content": "start"}]
    for index in range(1, 85):
        role = "assistant"
        if index in {30, 61}:
            role = "user"
        history.append({"role": role, "content": f"msg-{index}"})
    compactor = Consolidator(store, lambda prompt, max_new_tokens: "- summary")
    session = {
        "id": "session-1",
        "history": history,
        "last_consolidated": 0,
    }

    capped = compactor._cap_consolidation_boundary(session, 75)

    assert capped == 30


def test_consolidator_reuses_session_lock_and_tracks_last_summary_metadata(tmp_path):
    store = MemoryStore(tmp_path)
    history = [
        {"role": "user", "content": f"user-{index} " + ("X" * 480)}
        if index % 2 == 0
        else {"role": "assistant", "content": f"assistant-{index} " + ("Y" * 480)}
        for index in range(18)
    ]
    summaries = iter(
        [
            "- Round 1 summary about earlier transcript",
            "- Round 2 summary about even older transcript",
        ]
    )
    compactor = Consolidator(
        store,
        lambda prompt, max_new_tokens: next(summaries),
        context_window_tokens=700,
        max_completion_tokens=100,
    )
    session = {
        "id": "session-1",
        "updated_at": "2026-05-06T10:00:00+00:00",
        "history": history,
        "last_consolidated": 0,
        "session_metadata": {},
    }

    first_lock = compactor.get_lock("session-1")
    second_lock = compactor.get_lock("session-1")
    result = compactor.maybe_consolidate_by_tokens(session, session_summary="", task_summary="Continue")

    assert first_lock is second_lock
    assert result is not None
    assert result.archived_messages > 0
    assert result.summary.startswith("- Round")
    assert session["last_consolidated"] > 0
    assert session["session_metadata"]["_last_summary"]["text"] == result.summary
    assert session["session_metadata"]["_last_summary"]["last_active"] == "2026-05-06T10:00:00+00:00"


def test_auto_compact_keeps_recent_suffix_and_injects_summary(tmp_path):
    store = MemoryStore(tmp_path)
    sessions = SessionManager(tmp_path / ".pico" / "sessions")
    compactor = AutoCompact(
        sessions,
        Consolidator(store, lambda prompt, max_new_tokens: "- keep summary"),
        session_ttl_minutes=1,
    )
    session = {
        "id": "session-1",
        "history": [{"role": "user", "content": f"msg-{i}"} for i in range(12)],
        "last_consolidated": 0,
        "updated_at": "2024-04-07T10:00:00+00:00",
        "history_archive": {
            "latest_summary": "",
            "latest_cursor": 0,
            "compaction_count": 0,
            "archived_messages": 0,
            "last_compacted_at": "",
        },
    }

    prepared, summary = compactor.prepare_session(session, "session-1")
    assert len(prepared["history"]) == 8
    assert summary and "Previous conversation summary" in summary


def test_auto_compact_can_schedule_background_archive_for_expired_sessions(tmp_path):
    store = MemoryStore(tmp_path)
    sessions = SessionManager(tmp_path / ".pico" / "sessions")
    compactor = AutoCompact(
        sessions,
        Consolidator(store, lambda prompt, max_new_tokens: "- compacted summary"),
        session_ttl_minutes=1,
    )
    payload = {
        "id": "session-1",
        "created_at": "2026-05-06T10:00:00+00:00",
        "updated_at": "2026-05-06T10:00:00+00:00",
        "history": [{"role": "user", "content": f"msg-{i}"} for i in range(12)],
        "last_consolidated": 0,
        "session_metadata": {},
        "history_archive": {
            "latest_summary": "",
            "latest_cursor": 0,
            "compaction_count": 0,
            "archived_messages": 0,
            "last_compacted_at": "",
        },
    }
    sessions.save(payload)

    scheduled = []

    def schedule_background(coro):
        scheduled.append(coro)

    compactor.check_expired(schedule_background)
    assert len(scheduled) == 1

    asyncio.run(scheduled.pop())
    reloaded = sessions.load("session-1")

    assert len(reloaded["history"]) == 8


def test_session_manager_persists_jsonl_without_session_json(tmp_path):
    sessions = SessionManager(tmp_path / ".pico" / "sessions")
    payload = {
        "id": "session-1",
        "created_at": "2026-05-06T10:00:00+00:00",
        "updated_at": "2026-05-06T10:00:01+00:00",
        "history": [{"role": "user", "content": "hello", "created_at": "2026-05-06T10:00:01+00:00"}],
        "last_consolidated": 0,
        "session_metadata": {"channel": "cli"},
        "memory": {"pico_only": True},
        "history_archive": {"latest_summary": "not persisted in session json"},
    }

    path = sessions.save(payload)
    loaded = sessions.load("session-1")

    assert path == tmp_path / ".pico" / "sessions" / "session-1" / "cli_direct.jsonl"
    assert path.exists()
    assert not (path.parent / "session.json").exists()
    assert loaded["history"] == payload["history"]
    assert loaded["session_metadata"] == {"channel": "cli"}
    assert "memory" not in loaded
    assert "history_archive" not in loaded
    assert sessions.latest() == "session-1"


def test_session_manager_repairs_corrupt_jsonl_and_lists_sessions(tmp_path):
    sessions = SessionManager(tmp_path / ".pico" / "sessions")
    record = Session(key="session-1")
    record.add_message("user", "hello")
    sessions.save_record(record)
    history_path = sessions.history_path("session-1")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write("{bad json\n")
        handle.write('{"role": "assistant", "content": "recovered"}\n')
    sessions.invalidate("session-1")

    repaired = sessions.get_or_create("session-1")
    listed = sessions.list_sessions()
    readonly = sessions.read_session_file("session-1")

    assert [message["content"] for message in repaired.messages] == ["hello", "recovered"]
    assert listed and listed[0]["key"] == "session-1"
    assert readonly is not None
    assert [message["content"] for message in readonly["messages"]] == ["hello", "recovered"]
    assert sessions.delete_session("session-1") is True
    assert sessions.read_session_file("session-1") is None


def test_cron_service_runs_due_job_and_protects_system_jobs(tmp_path):
    events = []

    async def on_job(job):
        events.append(job.name)

    cron = CronService(tmp_path / "cron" / "jobs.json", on_job=on_job)
    job = cron.add_job("regular", CronSchedule(kind="every", every_ms=1), "run")
    system_job = cron.register_system_job(
        CronJob(
            id="dream",
            name="dream",
            schedule=CronSchedule(kind="every", every_ms=1),
            payload=CronPayload(kind="system_event", message="dream"),
        )
    )
    job.state.next_run_at_ms = 1
    system_job.state.next_run_at_ms = 1

    asyncio.run(cron._on_timer())

    assert events == ["regular", "dream"]
    assert cron.remove_job("dream") == "protected"
    assert cron.remove_job(job.id) == "removed"


def test_dream_processes_archived_history_into_user_profile(tmp_path):
    store = MemoryStore(tmp_path)
    store.append_history("- User prefers concise technical answers in Chinese.\n- Main project: pico memory refactor.")

    model = FakeModelClient(
        [
            "- Persist the user's stable communication preferences into USER.md.",
            '<tool name="write_file" path=".pico/USER.md"><content># User Profile\n\nInformation about the user to help personalize interactions.\n\n## Basic Information\n\n- **Name**: (your name)\n- **Timezone**: (your timezone)\n- **Language**: Chinese\n\n## Preferences\n\n### Communication Style\n\n- [ ] Casual\n- [ ] Professional\n- [x] Technical\n\n### Response Length\n\n- [x] Brief and concise\n- [ ] Detailed explanations\n- [ ] Adaptive based on question\n\n### Technical Level\n\n- [ ] Beginner\n- [ ] Intermediate\n- [x] Expert\n\n## Work Context\n\n- **Primary Role**: (your role)\n- **Main Projects**: pico memory refactor\n- **Tools You Use**: (IDEs, languages, frameworks)\n\n## Topics of Interest\n\n- pico memory refactor\n-\n-\n\n## Special Instructions\n\n(Any specific instructions for how the assistant should behave)\n</content></tool>',
            "<final>done</final>",
        ]
    )

    result = Dream(store, model, interval_seconds=0).run()

    assert result is not None
    assert result.completed is True
    assert result.processed_entries == 1
    assert result.new_cursor == 1
    assert store.get_last_dream_cursor() == 1
    user_text = (tmp_path / ".pico" / "USER.md").read_text(encoding="utf-8")
    assert "- **Language**: Chinese" in user_text
    assert "- [x] Brief and concise" in user_text
    assert "- **Main Projects**: pico memory refactor" in user_text


def test_dream_phase1_prompt_can_include_line_ages_and_phase2_lists_skills(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_memory("# Long-term Memory\n\nold fact\n")
    skills_dir = tmp_path / "skills" / "recall_helper"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("name: recall_helper\ndescription: remembers recall workflows\n", encoding="utf-8")
    store.append_history("- Continue refining memory.")

    dream = Dream(store, FakeModelClient(["analysis", "<final>done</final>"]), interval_seconds=0)
    dream.store.git.line_ages = lambda path: [type("Age", (), {"age_days": 0})(), type("Age", (), {"age_days": 0})(), type("Age", (), {"age_days": 30})()]  # type: ignore[method-assign]

    batch = store.read_unprocessed_history(0)
    phase1 = dream._build_phase1_prompt(batch)
    phase2 = dream._build_phase2_user_prompt(batch, "analysis")

    assert "<- 30d" in phase1
    assert "## Existing Skills" in phase2
    assert "recall_helper - remembers recall workflows" in phase2


def test_dream_tools_use_registry_prepare_call(tmp_path):
    store = MemoryStore(tmp_path)
    dream = Dream(store, FakeModelClient([]), interval_seconds=0)

    with patch("pico.tools.validate_tool", side_effect=AssertionError("legacy validator bypassed")):
        result = dream._run_tool("read_file", {"path": ".pico/USER.md", "start": "1", "end": "3"})

    assert "# .pico/USER.md" in result


def test_user_fact_notes_support_chinese_recall_without_assistant_pollution():
    memory = LayeredMemory()

    remembered = memory.remember_user_facts("我有两只猫")
    memory.append_note("你只有一只猫。", tags=("宠物",), source="assistant", kind="episodic")

    assert remembered == ["我有两只猫"]

    lines = [line for line in memory.retrieval_view("我有几个宠物", limit=3).splitlines() if line.startswith("- ")]
    assert "我有两只猫" in lines[0]
    assert all("你只有一只猫" not in line for line in lines)
