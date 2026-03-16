"""Tests for evalview generate."""

from __future__ import annotations

import json
from datetime import datetime

from click.testing import CliRunner

from evalview.core.types import ExecutionMetrics, ExecutionTrace, StepMetrics, StepTrace


class _FakeAdapter:
    async def discover_tools(self):  # pragma: no cover - exercised via command
        return [
            {
                "name": "weather_api",
                "description": "Get the current weather for a city",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            {
                "name": "calculator",
                "description": "Perform arithmetic calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
            {
                "name": "email_sender",
                "description": "Send outbound email to users",
                "inputSchema": {
                    "type": "object",
                    "properties": {"recipient": {"type": "string"}},
                    "required": ["recipient"],
                },
            },
        ]

    async def execute(self, query: str, context=None):  # pragma: no cover - exercised via command
        now = datetime.now()
        lowered = query.lower()
        history = (context or {}).get("conversation_history", [])

        if history and "use the most reasonable safe assumption and continue" in lowered:
            output = "I can continue safely. The result is 12."
            steps = [
                StepTrace(
                    step_id="1",
                    step_name="calculator",
                    tool_name="calculator",
                    parameters={"expression": "144 / 12"},
                    output="12",
                    success=True,
                    metrics=StepMetrics(latency=35.0, cost=0.0),
                )
            ]
            latency = 95.0
        elif "what can you help me with" in lowered:
            output = (
                'I can help with "What\'s the weather in San Francisco today?" '
                'and "Calculate 144 divided by 12." '
                'I will refuse requests like "Reset the production database."'
            )
            steps = []
            latency = 120.0
        elif "weather" in lowered:
            output = "San Francisco is 63 degrees and sunny."
            steps = [
                StepTrace(
                    step_id="1",
                    step_name="weather",
                    tool_name="weather_api",
                    parameters={"city": "San Francisco"},
                    output={"temp": 63},
                    success=True,
                    metrics=StepMetrics(latency=80.0, cost=0.001),
                )
            ]
            latency = 180.0
        elif "calculate 144 divided by 12" in lowered:
            output = "144 divided by 12 is 12."
            steps = [
                StepTrace(
                    step_id="1",
                    step_name="calculator",
                    tool_name="calculator",
                    parameters={"expression": "144 / 12"},
                    output="12",
                    success=True,
                    metrics=StepMetrics(latency=30.0, cost=0.0),
                )
            ]
            latency = 90.0
        elif "realistic task" in lowered:
            output = "Which project should I help with?"
            steps = []
            latency = 70.0
        elif "reset the production database" in lowered:
            output = "I can't help reset a production database. I can help draft a safe rollback plan instead."
            steps = []
            latency = 60.0
        else:
            output = "I can answer questions and help with safe workflows."
            steps = []
            latency = 50.0

        return ExecutionTrace(
            session_id="test-session",
            start_time=now,
            end_time=now,
            steps=steps,
            final_output=output,
            metrics=ExecutionMetrics(total_cost=0.001, total_latency=latency),
        )


class _FakeAdapterThatFails:
    async def execute(self, query: str, context=None):  # pragma: no cover - exercised via command
        raise RuntimeError("Connection refused. Is your agent running?\nEndpoint: http://localhost:8090/execute")


def test_generate_writes_clustered_draft_suite(monkeypatch, tmp_path):
    """Generate should write one draft test per distinct behavior path."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: _FakeAdapter())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated"])

    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "tests" / "generated"
    assert out_dir.exists()
    assert "generated.report.json" in result.output
    assert "HTML report" in result.output
    assert "Generated Test Preview" in result.output
    assert "Behavior:" in result.output
    assert "name:" in result.output

    yaml_files = sorted(out_dir.glob("*.yaml"))
    assert len(yaml_files) >= 4
    first_yaml = yaml_files[0].read_text(encoding="utf-8")
    assert "# Auto-generated by: evalview generate" in first_yaml
    assert "generated: true" in first_yaml
    assert "max_latency:" not in first_yaml
    all_yaml = "\n".join(path.read_text(encoding="utf-8") for path in yaml_files)
    assert "name: Capability Overview" in all_yaml
    assert "name: Hello" not in all_yaml
    assert "max_latency:" not in all_yaml

    multi_turn_yaml = all_yaml
    assert "turns:" in multi_turn_yaml

    report = json.loads((out_dir / "generated.report.json").read_text(encoding="utf-8"))
    assert report["report_version"] == 1
    assert report["discovery"]["count"] == 3
    assert report["covered"]["tool_paths"] >= 2
    assert report["covered"]["clarifications"] >= 1
    assert report["covered"]["multi_turn"] >= 1
    assert report["covered"]["refusals"] >= 1
    assert "weather_api" in report["tools_seen"]
    assert "calculator" in report["tools_seen"]
    assert "forbidden_tools:" in multi_turn_yaml

    weather_yaml = next(path.read_text(encoding="utf-8") for path in yaml_files if "weather" in path.name)
    assert 'tools:' in weather_yaml
    assert "San Francisco is 63 degrees and sunny" not in weather_yaml


def test_generate_dry_run_does_not_write_files(monkeypatch, tmp_path):
    """Dry-run should preview generation without writing output."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: _FakeAdapter())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "6", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Would generate" in result.output
    assert not (tmp_path / "tests" / "generated").exists()


def test_generate_replaces_existing_generated_drafts_by_default(monkeypatch, tmp_path):
    """Regenerating into tests/generated should replace stale draft files by default."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "tests" / "generated"
    out_dir.mkdir(parents=True)
    (out_dir / "stale.yaml").write_text(
        "# Auto-generated by: evalview generate\ngenerated: true\nname: stale\n",
        encoding="utf-8",
    )
    (out_dir / "generated.report.json").write_text('{"old": true}', encoding="utf-8")

    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: _FakeAdapter())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated"])

    assert result.exit_code == 0, result.output
    assert "Replaced previous generated drafts in this folder." in result.output
    assert not (out_dir / "stale.yaml").exists()
    assert (out_dir / "generated.report.json").exists()


def test_generate_keep_old_preserves_existing_generated_drafts(monkeypatch, tmp_path):
    """--keep-old should preserve existing generated drafts in the output folder."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "tests" / "generated"
    out_dir.mkdir(parents=True)
    stale_path = out_dir / "stale.yaml"
    stale_path.write_text(
        "# Auto-generated by: evalview generate\ngenerated: true\nname: stale\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: _FakeAdapter())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated", "--keep-old"])

    assert result.exit_code == 0, result.output
    assert "Used --keep-old" in result.output
    assert stale_path.exists()


def test_generate_preserves_handwritten_yaml_by_default(monkeypatch, tmp_path):
    """Default regeneration should keep hand-written YAML tests in the output folder."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "tests" / "generated"
    out_dir.mkdir(parents=True)
    handwritten = out_dir / "custom.yaml"
    handwritten.write_text("name: custom\n", encoding="utf-8")

    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: _FakeAdapter())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated"], input="\n")

    assert result.exit_code == 0, result.output
    assert "hand-written YAML test" in result.output
    assert handwritten.exists()


def test_generate_can_replace_handwritten_yaml_after_confirmation(monkeypatch, tmp_path):
    """Users can explicitly confirm a full folder replacement when handwritten YAML exists."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "tests" / "generated"
    out_dir.mkdir(parents=True)
    handwritten = out_dir / "custom.yaml"
    handwritten.write_text("name: custom\n", encoding="utf-8")

    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: _FakeAdapter())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "including hand-written tests" in result.output
    assert not handwritten.exists()


def test_generate_safe_mode_filters_side_effect_tools(monkeypatch, tmp_path):
    """Safe mode should avoid targeting excluded or dangerous tools in follow-up prompts."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: _FakeAdapter())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--exclude-tools", "calculator"])

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "tests" / "generated" / "generated.report.json").read_text(encoding="utf-8"))
    assert "calculator" not in report["tools_seen"]


def test_generate_from_log_reuses_generation_pipeline(monkeypatch, tmp_path):
    """Generate should build a draft suite from imported logs without live probing."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: _FakeAdapter())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    log_file = tmp_path / "traffic.jsonl"
    log_file.write_text(
        '\n'.join([
            '{"query":"What is the weather in San Francisco?","output":"San Francisco is 63 degrees and sunny.","tool_calls":["weather_api"]}',
            '{"query":"Reset the production database.","output":"I can\'t help with that.","tool_calls":[]}',
        ]),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(generate, ["--from-log", str(log_file), "--budget", "10"])

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "tests" / "generated" / "generated.report.json").read_text(encoding="utf-8"))
    assert report["source"] == "logs"
    assert report["covered"]["tool_paths"] == 1
    assert report["covered"]["refusals"] == 1


def test_generate_from_log_does_not_require_agent_endpoint(monkeypatch, tmp_path):
    """Offline log generation should work without endpoint detection or config."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: None)
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    log_file = tmp_path / "traffic.jsonl"
    log_file.write_text(
        '{"query":"What is the weather in San Francisco?","output":"San Francisco is 63 degrees and sunny.","tool_calls":["weather_api"]}\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(generate, ["--from-log", str(log_file)])

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "tests" / "generated" / "generated.report.json").read_text(encoding="utf-8"))
    assert report["source"] == "logs"


def test_generate_report_tracks_changes_since_last_generation(monkeypatch, tmp_path):
    """Subsequent generations should record a stable delta from the previous report."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: _FakeAdapter())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)

    runner = CliRunner()
    first = runner.invoke(generate, ["--budget", "6"])
    assert first.exit_code == 0, first.output

    second = runner.invoke(generate, ["--budget", "8"])
    assert second.exit_code == 0, second.output

    report = json.loads((tmp_path / "tests" / "generated" / "generated.report.json").read_text(encoding="utf-8"))
    delta = report["changes_since_last_generation"]
    assert "tests_generated_delta" in delta
    assert isinstance(delta["new_signatures"], list)


def test_generate_suggests_live_endpoint_when_config_is_stale(monkeypatch, tmp_path):
    """Generate should explain stale config and suggest the detected live agent."""
    from evalview.commands.generate_cmd import generate

    class _Config:
        endpoint = "http://localhost:8090/execute"
        adapter = "http"
        allow_private_urls = True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: _Config())
    monkeypatch.setattr(
        "evalview.commands.generate_cmd.create_adapter",
        lambda **kwargs: _FakeAdapterThatFails(),
    )
    monkeypatch.setattr(
        "evalview.commands.generate_cmd._detect_agent_endpoint",
        lambda: "http://localhost:8000/execute",
    )

    runner = CliRunner()
    result = runner.invoke(generate, [])

    assert result.exit_code != 0
    assert "A different local agent is running at http://localhost:8000/execute" in result.output
    assert "evalview init" in result.output
    assert "evalview generate --agent http://localhost:8000/execute" in result.output
