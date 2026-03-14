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
    monkeypatch.setattr(
        "evalview.commands.init_cmd._generate_init_draft_suite",
        lambda endpoint, out_dir: (
            2,
            {
                "covered": {
                    "tool_paths": 1,
                    "direct_answers": 1,
                    "multi_turn": 0,
                }
            },
        ),
    )
    monkeypatch.setattr("evalview.commands.init_cmd._create_demo_agent", lambda base_path: None)

    runner = CliRunner()
    result = runner.invoke(init, ["--dir", str(tmp_path)], input="2\n")

    assert result.exit_code == 0, result.output
    assert "tests/generated-from-init/" in result.output
    assert "evalview snapshot tests/generated-from-init" in result.output
    assert "evalview check tests/generated-from-init" in result.output
    assert "tests/test-cases/" not in result.output


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
