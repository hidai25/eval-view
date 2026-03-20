"""Tests for init onboarding flow."""

from __future__ import annotations

from click.testing import CliRunner
import yaml

from evalview.core.types import ExpectedBehavior, TestCase, TestInput, Thresholds


def _make_fake_tests(n=2):
    """Create minimal TestCase objects for mocking."""
    tests = []
    for i in range(n):
        tests.append(TestCase(
            name=f"Test {i+1}",
            description="Draft test",
            input=TestInput(query=f"test query {i+1}"),
            expected=ExpectedBehavior(tools=["tool_a"] if i == 0 else []),
            thresholds=Thresholds(min_score=50.0),
            adapter="http",
            endpoint="http://localhost:8000/execute",
            generated=True,
            meta={"behavior_class": "tool_path" if i == 0 else "direct_answer", "prompt_source": "discovery"},
        ))
    return tests


def test_init_generate_path_uses_isolated_onboarding_folder(monkeypatch, tmp_path):
    """`evalview init` should not mix generated onboarding drafts with stale tests."""
    from evalview.commands.init_cmd import init

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.init_cmd._detect_agent_endpoint", lambda: "http://localhost:8000/execute")
    monkeypatch.setattr("evalview.commands.init_cmd._detect_model", lambda: "claude-sonnet-4-6")

    fake_tests = _make_fake_tests(2)

    def _fake_generate(endpoint, out_dir, **kwargs):
        return (
            2,
            {
                "covered": {
                    "tool_paths": 1,
                    "direct_answers": 1,
                    "multi_turn": 0,
                }
            },
            fake_tests,
        )

    monkeypatch.setattr("evalview.commands.init_cmd._generate_init_draft_suite", _fake_generate)
    monkeypatch.setattr("evalview.commands.init_cmd._create_demo_agent", lambda base_path: None)

    # Patch the httpx preflight check so it doesn't hit a real endpoint
    class _FakeResponse:
        status_code = 200
    monkeypatch.setattr("httpx.post", lambda *a, **kw: _FakeResponse())

    runner = CliRunner()
    # input: "2" = choice 2 (generate), "1" = quick budget, "y" = approve tests
    result = runner.invoke(init, ["--dir", str(tmp_path)], input="2\n1\ny\n")

    assert result.exit_code == 0, result.output
    assert "tests/generated-from-init/" in result.output
    assert "evalview snapshot --path tests/generated-from-init" in result.output
    assert "--approve-generated" not in result.output
    assert "evalview check tests/generated-from-init" in result.output
    assert "tests/test-cases/" not in result.output
    assert "Generated Tests" in result.output
    assert "Save these" in result.output
    state = (tmp_path / ".evalview" / "state.json").read_text(encoding="utf-8")
    assert "tests/generated-from-init" in state


def test_init_updates_stale_existing_config(monkeypatch, tmp_path):
    """`evalview init` should refresh stale config when it detects a live agent."""
    from evalview.commands.init_cmd import init

    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".evalview"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        """adapter: http
endpoint: http://localhost:8090/execute
timeout: 30.0
model:
  name: old-model
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("evalview.commands.init_cmd._detect_agent_endpoint", lambda: "http://localhost:8000/execute")
    monkeypatch.setattr("evalview.commands.init_cmd._detect_model", lambda: "claude-sonnet-4-6")
    monkeypatch.setattr(
        "evalview.commands.init_cmd._generate_init_draft_suite",
        lambda endpoint, out_dir, **kw: (1, {"covered": {"tool_paths": 0, "direct_answers": 1, "multi_turn": 0}}, _make_fake_tests(1)),
    )
    monkeypatch.setattr("evalview.commands.init_cmd._create_demo_agent", lambda base_path: None)

    # Patch the httpx preflight check so it doesn't hit a real endpoint
    class _FakeResponse:
        status_code = 200
    monkeypatch.setattr("httpx.post", lambda *a, **kw: _FakeResponse())

    runner = CliRunner()
    result = runner.invoke(init, ["--dir", str(tmp_path)], input="2\n1\ny\n")

    assert result.exit_code == 0, result.output
    assert "Updated .evalview/config.yaml to use http://localhost:8000/execute" in result.output
    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated["endpoint"] == "http://localhost:8000/execute"
    assert updated["adapter"] == "http"
    assert updated["model"]["name"] == "claude-sonnet-4-6"


def test_init_regeneration_replaces_existing_onboarding_drafts(monkeypatch, tmp_path):
    """Rerunning init generation should refresh tests/generated-from-init instead of accumulating drafts."""
    from evalview.commands.init_cmd import init

    monkeypatch.chdir(tmp_path)
    generated_dir = tmp_path / "tests" / "generated-from-init"
    generated_dir.mkdir(parents=True)
    (generated_dir / "stale.yaml").write_text(
        "# Auto-generated by: evalview generate\ngenerated: true\nname: stale\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("evalview.commands.init_cmd._detect_agent_endpoint", lambda: "http://localhost:8000/execute")
    monkeypatch.setattr("evalview.commands.init_cmd._detect_model", lambda: "claude-sonnet-4-6")

    fake_tests = _make_fake_tests(1)

    def _fake_generate(endpoint, out_dir, **kwargs):
        return (1, {"covered": {"tool_paths": 0, "direct_answers": 1, "multi_turn": 0}}, fake_tests)

    monkeypatch.setattr("evalview.commands.init_cmd._generate_init_draft_suite", _fake_generate)
    monkeypatch.setattr("evalview.commands.init_cmd._create_demo_agent", lambda base_path: None)

    # Patch the httpx preflight check
    class _FakeResponse:
        status_code = 200
    monkeypatch.setattr("httpx.post", lambda *a, **kw: _FakeResponse())

    runner = CliRunner()
    result = runner.invoke(init, ["--dir", str(tmp_path)], input="2\n1\ny\n")

    assert result.exit_code == 0, result.output
    # Stale file should be cleaned up by _write_init_suite
    generated_files = list(generated_dir.glob("*.yaml"))
    assert len(generated_files) >= 1
