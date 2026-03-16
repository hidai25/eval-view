"""Tests for init onboarding flow."""

from __future__ import annotations

from click.testing import CliRunner
import yaml


def test_init_generate_path_uses_isolated_onboarding_folder(monkeypatch, tmp_path):
    """`evalview init` should not mix generated onboarding drafts with stale tests."""
    from evalview.commands.init_cmd import init

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.init_cmd._detect_agent_endpoint", lambda: "http://localhost:8000/execute")
    monkeypatch.setattr("evalview.commands.init_cmd._detect_model", lambda: "claude-sonnet-4-6")
    def _fake_generate(endpoint, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preview.yaml").write_text(
            "name: preview\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
            encoding="utf-8",
        )
        return (
            2,
            {
                "covered": {
                    "tool_paths": 1,
                    "direct_answers": 1,
                    "multi_turn": 0,
                }
            },
        )

    monkeypatch.setattr("evalview.commands.init_cmd._generate_init_draft_suite", _fake_generate)
    monkeypatch.setattr("evalview.commands.init_cmd._create_demo_agent", lambda base_path: None)

    runner = CliRunner()
    result = runner.invoke(init, ["--dir", str(tmp_path)], input="2\n")

    assert result.exit_code == 0, result.output
    assert "tests/generated-from-init/" in result.output
    assert "evalview snapshot tests/generated-from-init" in result.output
    assert "--approve-generated" not in result.output
    assert "evalview check tests/generated-from-init" in result.output
    assert "tests/test-cases/" not in result.output
    assert "Only 2 distinct behavior path was discovered" not in result.output
    assert "Generated Test Preview" in result.output
    assert "Behavior:" in result.output
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
        lambda endpoint, out_dir: (1, {"covered": {"tool_paths": 0, "direct_answers": 1, "multi_turn": 0}}),
    )
    monkeypatch.setattr("evalview.commands.init_cmd._create_demo_agent", lambda base_path: None)

    runner = CliRunner()
    result = runner.invoke(init, ["--dir", str(tmp_path)], input="2\n")

    assert result.exit_code == 0, result.output
    assert "Updated .evalview/config.yaml to use http://localhost:8000/execute" in result.output
    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated["endpoint"] == "http://localhost:8000/execute"
    assert updated["adapter"] == "http"
    assert updated["model"]["name"] == "claude-sonnet-4-6"


def test_init_explains_single_draft_as_single_behavior_path(monkeypatch, tmp_path):
    """Init should explain why one generated draft can still be expected."""
    from evalview.commands.init_cmd import init

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.init_cmd._detect_agent_endpoint", lambda: "http://localhost:8000/execute")
    monkeypatch.setattr("evalview.commands.init_cmd._detect_model", lambda: "claude-sonnet-4-6")
    monkeypatch.setattr(
        "evalview.commands.init_cmd._generate_init_draft_suite",
        lambda endpoint, out_dir: (
            1,
            {"covered": {"tool_paths": 0, "direct_answers": 1, "multi_turn": 0, "clarifications": 0, "refusals": 0, "error_paths": 0}},
        ),
    )
    monkeypatch.setattr("evalview.commands.init_cmd._create_demo_agent", lambda base_path: None)

    runner = CliRunner()
    result = runner.invoke(init, ["--dir", str(tmp_path)], input="2\n")

    assert result.exit_code == 0, result.output
    assert "Only 1 distinct behavior path was discovered during the lighter init flow" in result.output
    assert "one representative draft test" in result.output
    assert "clarifications=0" in result.output


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

    def _fake_generate(endpoint, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "fresh.yaml").write_text(
            "name: fresh\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
            encoding="utf-8",
        )
        return (1, {"covered": {"tool_paths": 0, "direct_answers": 1, "multi_turn": 0}})

    monkeypatch.setattr("evalview.commands.init_cmd._generate_init_draft_suite", _fake_generate)
    monkeypatch.setattr("evalview.commands.init_cmd._create_demo_agent", lambda base_path: None)

    runner = CliRunner()
    result = runner.invoke(init, ["--dir", str(tmp_path)], input="2\n")

    assert result.exit_code == 0, result.output
    assert not (generated_dir / "stale.yaml").exists()
    assert (generated_dir / "fresh.yaml").exists()
