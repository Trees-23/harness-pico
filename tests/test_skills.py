from pico.skills import SkillsLoader


def write_skill(root, name, frontmatter, body):
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")


def test_skills_loader_workspace_overrides_builtin_and_summarizes(tmp_path):
    workspace_skills = tmp_path / "skills"
    builtin_skills = tmp_path / "builtin"
    write_skill(
        workspace_skills,
        "demo",
        'description: Workspace demo\nmetadata: {"nanobot": {"always": true}}',
        "Workspace body",
    )
    write_skill(
        builtin_skills,
        "demo",
        "description: Builtin demo",
        "Builtin body",
    )
    write_skill(
        builtin_skills,
        "missing",
        'description: Missing tool\nmetadata: {"nanobot": {"requires": {"bins": ["definitely_missing_pico_bin"]}}}',
        "Missing body",
    )

    loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin_skills)

    assert loader.list_skills(filter_unavailable=False) == [
        {"name": "demo", "path": str(workspace_skills / "demo" / "SKILL.md"), "source": "workspace"},
        {"name": "missing", "path": str(builtin_skills / "missing" / "SKILL.md"), "source": "builtin"},
    ]
    assert loader.load_skill("demo") and "Workspace body" in loader.load_skill("demo")
    assert loader.get_always_skills() == ["demo"]

    loaded = loader.load_skills_for_context(["demo"])
    assert "### Skill: demo" in loaded
    assert "Workspace body" in loaded
    assert "metadata:" not in loaded

    summary = loader.build_skills_summary(exclude={"demo"})
    assert "**missing**" in summary
    assert "unavailable: CLI: definitely_missing_pico_bin" in summary


def test_read_file_can_read_builtin_skill_path(tmp_path, monkeypatch):
    from pico import FakeModelClient, MiniAgent, SessionStore, WorkspaceContext
    import pico.skills as skills_module
    import pico.tooling.capabilities as capabilities
    import pico.tooling.executor as executor

    builtin_skills = tmp_path / "builtin"
    write_skill(builtin_skills, "demo", "description: Demo", "Builtin body")
    monkeypatch.setattr(skills_module, "BUILTIN_SKILLS_DIR", builtin_skills)
    monkeypatch.setattr(capabilities, "BUILTIN_SKILLS_DIR", builtin_skills)
    monkeypatch.setattr(executor, "BUILTIN_SKILLS_DIR", builtin_skills)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=WorkspaceContext.build(workspace),
        session_store=SessionStore(workspace / ".pico" / "sessions"),
        approval_policy="auto",
    )

    result = agent.run_tool("read_file", {"path": str(builtin_skills / "demo" / "SKILL.md"), "start": 1, "end": 20})

    assert "# " in result
    assert "Builtin body" in result
