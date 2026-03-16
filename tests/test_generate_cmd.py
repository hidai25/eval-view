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
        elif "example requests" in lowered or "example tasks" in lowered:
            output = (
                'Sure! Try: "What\'s the weather in London in Fahrenheit?" '
                'or "What is 25 times 4?"'
            )
            steps = []
            latency = 80.0
        elif "types of data" in lowered or "information do you work with" in lowered:
            output = "I work with weather data and mathematical expressions."
            steps = []
            latency = 60.0
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


class _NonProgressingFollowUpAdapter(_FakeAdapter):
    async def execute(self, query: str, context=None):  # pragma: no cover - exercised via command
        lowered = query.lower()
        history = (context or {}).get("conversation_history", [])
        if history and "use the most reasonable safe assumption and continue" in lowered:
            now = datetime.now()
            return ExecutionTrace(
                session_id="test-session",
                start_time=now,
                end_time=now,
                steps=[],
                final_output="Please describe the support issue you need help with.",
                metrics=ExecutionMetrics(total_cost=0.0, total_latency=36.0),
            )
        return await super().execute(query, context=context)


class _DomainSeedAdapter:
    async def discover_tools(self):  # pragma: no cover - exercised via command
        return [
            {
                "name": "search_pain_history",
                "description": "Search pain point history for a product or category",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            }
        ]

    async def execute(self, query: str, context=None):  # pragma: no cover - exercised via command
        now = datetime.now()
        lowered = query.lower()

        if "what can you help me with" in lowered:
            output = (
                'You can ask things like "What are the top pain points for Notion this week?" '
                'or "Show me stability issues for Slack."'
            )
            steps = []
            latency = 50.0
        elif "notion" in lowered or "slack" in lowered:
            output = "I found current pain point signals for the requested product."
            steps = [
                StepTrace(
                    step_id="1",
                    step_name="search_pain_history",
                    tool_name="search_pain_history",
                    parameters={"query": query, "days": 7},
                    output={"items": 3},
                    success=True,
                    metrics=StepMetrics(latency=20.0, cost=0.001),
                )
            ]
            latency = 85.0
        elif "coffee shops" in lowered or "eiffel tower" in lowered:
            output = "I focus on product pain points, not local search."
            steps = []
            latency = 35.0
        else:
            output = "I can help analyze product pain points."
            steps = []
            latency = 45.0

        return ExecutionTrace(
            session_id="domain-seed-session",
            start_time=now,
            end_time=now,
            steps=steps,
            final_output=output,
            metrics=ExecutionMetrics(total_cost=0.001, total_latency=latency),
        )


def _patch_generate_for_fake_adapter(monkeypatch, adapter_cls=None):
    """Common monkeypatches for generate tests using fake adapters."""
    monkeypatch.setattr("evalview.commands.generate_cmd.create_adapter", lambda **kwargs: (adapter_cls or _FakeAdapter)())
    monkeypatch.setattr("evalview.commands.generate_cmd._detect_agent_endpoint", lambda: "http://localhost:8000")
    monkeypatch.setattr("evalview.commands.generate_cmd._load_config_if_exists", lambda: None)
    # Prevent real LLM calls during synthesis in tests
    monkeypatch.setattr(
        "evalview.test_generation.AgentTestGenerator._select_synthesis_client",
        staticmethod(lambda model_override=None: None),
    )


def test_generate_writes_clustered_draft_suite(monkeypatch, tmp_path):
    """Generate should write one draft test per distinct behavior path."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    _patch_generate_for_fake_adapter(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated"], input="y\n")

    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "tests" / "generated"
    assert out_dir.exists()
    assert "Full Test YAML" in result.output
    assert "Generated Tests" in result.output
    assert "Prompt sources" in result.output
    assert "Save these" in result.output

    yaml_files = sorted(out_dir.glob("*.yaml"))
    assert len(yaml_files) >= 2
    first_yaml = yaml_files[0].read_text(encoding="utf-8")
    assert "# Auto-generated by: evalview generate" in first_yaml
    assert "generated: true" in first_yaml
    assert "max_latency:" not in first_yaml
    all_yaml = "\n".join(path.read_text(encoding="utf-8") for path in yaml_files)
    # Discovery probes (capability, examples) should NOT become tests
    assert "name: Capability overview" not in all_yaml
    assert "prompt_source:" in all_yaml

    report = json.loads((out_dir / "generated.report.json").read_text(encoding="utf-8"))
    assert report["report_version"] == 1
    assert report["discovery"]["count"] == 3
    assert report["covered"]["tool_paths"] >= 2
    assert report["covered"]["refusals"] >= 1
    assert "weather_api" in report["tools_seen"]
    assert "calculator" in report["tools_seen"]
    assert "prompt_sources" in report

    weather_yaml = next(path.read_text(encoding="utf-8") for path in yaml_files if "weather" in path.name)
    assert 'tools:' in weather_yaml
    assert "San Francisco is 63 degrees and sunny" not in weather_yaml


def test_generate_dry_run_does_not_write_files(monkeypatch, tmp_path):
    """Dry-run should preview generation without writing output."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    _patch_generate_for_fake_adapter(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "6", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Would generate" in result.output
    assert "Prompt sources" in result.output
    assert not (tmp_path / "tests" / "generated").exists()


def test_generate_uses_project_docs_as_cold_start_seed_prompts(monkeypatch, tmp_path):
    """Cold-start generation should prioritize project-domain prompts mined from local docs."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# PainTracker",
                "| Message | What it does |",
                "| --- | --- |",
                "| `What are the top pain points for Notion this week?` | Query current data |",
                "| `Show me stability issues for Slack` | Search a product category |",
            ]
        ),
        encoding="utf-8",
    )
    _patch_generate_for_fake_adapter(monkeypatch, adapter_cls=_DomainSeedAdapter)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "3", "--out", "tests/generated"], input="y\n")

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "tests" / "generated" / "generated.report.json").read_text(encoding="utf-8"))
    drafted_queries = {draft["query"] for draft in report["draft_tests"]}
    assert "What are the top pain points for Notion this week?" in drafted_queries or "Show me stability issues for Slack" in drafted_queries
    assert "Search for coffee shops near the Eiffel Tower." not in drafted_queries
    assert any(source.startswith("project_docs:") for source in report["prompt_sources"])


def test_generate_uses_existing_curated_tests_as_seed_prompts(monkeypatch, tmp_path):
    """Existing non-generated tests should act as high-signal seeds ahead of generic prompts."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    evalview_dir = tmp_path / "tests" / "evalview"
    evalview_dir.mkdir(parents=True)
    (evalview_dir / "notion.yaml").write_text(
        "\n".join(
            [
                'name: "pain-query"',
                "input:",
                '  query: "What are the top pain points for Notion this week?"',
            ]
        ),
        encoding="utf-8",
    )

    _patch_generate_for_fake_adapter(monkeypatch, adapter_cls=_DomainSeedAdapter)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "3", "--out", "tests/generated"], input="y\n")

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "tests" / "generated" / "generated.report.json").read_text(encoding="utf-8"))
    drafted_queries = {draft["query"] for draft in report["draft_tests"]}
    assert "What are the top pain points for Notion this week?" in drafted_queries
    assert report["prompt_sources"].get("existing_tests", 0) >= 1


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

    _patch_generate_for_fake_adapter(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated"], input="y\n")

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

    _patch_generate_for_fake_adapter(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated", "--keep-old"], input="y\n")

    assert result.exit_code == 0, result.output
    assert stale_path.exists()


def test_generate_preserves_handwritten_yaml_by_default(monkeypatch, tmp_path):
    """Default regeneration should keep hand-written YAML tests in the output folder."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "tests" / "generated"
    out_dir.mkdir(parents=True)
    handwritten = out_dir / "custom.yaml"
    handwritten.write_text("name: custom\n", encoding="utf-8")

    _patch_generate_for_fake_adapter(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated"], input="y\n\n")

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

    _patch_generate_for_fake_adapter(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--out", "tests/generated"], input="y\nn\n")

    assert result.exit_code == 0, result.output
    assert "including hand-written tests" in result.output
    assert not handwritten.exists()


def test_generate_safe_mode_filters_side_effect_tools(monkeypatch, tmp_path):
    """Safe mode should avoid targeting excluded or dangerous tools in follow-up prompts."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    _patch_generate_for_fake_adapter(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(generate, ["--budget", "8", "--exclude-tools", "calculator"], input="y\n")

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "tests" / "generated" / "generated.report.json").read_text(encoding="utf-8"))
    assert "calculator" not in report["tools_seen"]


def test_generate_from_log_reuses_generation_pipeline(monkeypatch, tmp_path):
    """Generate should build a draft suite from imported logs without live probing."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    _patch_generate_for_fake_adapter(monkeypatch)

    log_file = tmp_path / "traffic.jsonl"
    log_file.write_text(
        '\n'.join([
            '{"query":"What is the weather in San Francisco?","output":"San Francisco is 63 degrees and sunny.","tool_calls":["weather_api"]}',
            '{"query":"Reset the production database.","output":"I can\'t help with that.","tool_calls":[]}',
        ]),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(generate, ["--from-log", str(log_file), "--budget", "10"], input="y\n")

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
    result = runner.invoke(generate, ["--from-log", str(log_file)], input="y\n")

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "tests" / "generated" / "generated.report.json").read_text(encoding="utf-8"))
    assert report["source"] == "logs"


def test_generate_drops_non_progressing_follow_up_from_multi_turn(monkeypatch, tmp_path):
    """A clarification follow-up should not become a multi-turn draft if it does not advance."""
    import asyncio

    from evalview.test_generation import AgentTestGenerator, ProbeResult

    adapter = _NonProgressingFollowUpAdapter()
    generator = AgentTestGenerator(
        adapter=adapter,
        endpoint="http://localhost:8000",
        adapter_type="http",
    )
    # Prevent real LLM calls — force static fallback for clarification follow-up
    monkeypatch.setattr(
        "evalview.test_generation.AgentTestGenerator._select_synthesis_client",
        staticmethod(lambda model_override=None: None),
    )
    now = datetime.now()
    first_trace = ExecutionTrace(
        session_id="test-session",
        start_time=now,
        end_time=now,
        steps=[],
        final_output="Please describe the support issue you need help with.",
        metrics=ExecutionMetrics(total_cost=0.0, total_latency=40.0),
    )
    probe = ProbeResult(
        query="Hello, what can you help me with?",
        trace=first_trace,
        tools=[],
        signature="clarification",
        behavior_class="clarification",
        rationale="Observed clarification path",
    )

    follow_up = asyncio.run(generator._generate_multi_turn_probe(probe))

    # The non-progressing adapter returns the same output, so the follow-up
    # should be dropped as not meaningful.
    assert follow_up is None


def test_generate_report_tracks_changes_since_last_generation(monkeypatch, tmp_path):
    """Subsequent generations should record a stable delta from the previous report."""
    from evalview.commands.generate_cmd import generate

    monkeypatch.chdir(tmp_path)
    _patch_generate_for_fake_adapter(monkeypatch)

    runner = CliRunner()
    first = runner.invoke(generate, ["--budget", "6"], input="y\n")
    assert first.exit_code == 0, first.output

    second = runner.invoke(generate, ["--budget", "8"], input="y\n")
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
    result = runner.invoke(generate, ["--budget", "4"])

    assert result.exit_code != 0
    assert "A different local agent is running at http://localhost:8000/execute" in result.output
    assert "evalview init" in result.output
    assert "evalview generate --agent http://localhost:8000/execute" in result.output
