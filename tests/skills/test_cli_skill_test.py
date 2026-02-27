import json
from types import SimpleNamespace

import yaml
from click.testing import CliRunner

from evalview.cli import main


def test_skill_test_cli_passes_provider_and_base_url_in_legacy_mode(tmp_path, monkeypatch):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: test-skill\n"
        "description: test skill for cli test\n"
        "---\n\n"
        "# Test Skill\n"
    )

    suite_file = tmp_path / "tests.yaml"
    suite_file.write_text(
        yaml.safe_dump(
            {
                "name": "cli-suite",
                "skill": str(skill_file),
                "tests": [
                    {
                        "name": "t1",
                        "input": "hi",
                        "expected": {"output_contains": ["hello"]},
                    }
                ],
            }
        )
    )

    captured = {}

    fake_result = SimpleNamespace(
        suite_name="cli-suite",
        skill_name="test-skill",
        passed=True,
        total_tests=1,
        passed_tests=1,
        failed_tests=0,
        pass_rate=1.0,
        total_latency_ms=10.0,
        avg_latency_ms=10.0,
        total_tokens=5,
        results=[
            SimpleNamespace(
                test_name="t1",
                passed=True,
                score=100.0,
                input_query="hi",
                output="hello",
                contains_failed=[],
                not_contains_failed=[],
                latency_ms=10.0,
                input_tokens=5,
                output_tokens=5,
                error=None,
            )
        ],
    )

    class FakeSkillRunner:
        def __init__(self, model, provider=None, base_url=None):
            captured["model"] = model
            captured["provider"] = provider
            captured["base_url"] = base_url
            self.provider = provider or "openai"
            self.base_url = base_url
            self.model = model or "gpt-4o-mini"

        def load_test_suite(self, path):
            captured["loaded_test_file"] = path
            return SimpleNamespace(
                name="cli-suite",
                skill=str(skill_file),
                tests=[SimpleNamespace(name="t1")],
                min_pass_rate=1.0,
            )

        def run_suite(self, suite, on_test_complete=None):
            return fake_result

    monkeypatch.setattr("evalview.skills.SkillRunner", FakeSkillRunner)
    monkeypatch.setattr("evalview.cli.print_evalview_banner", lambda *args, **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "skill",
            "test",
            str(suite_file),
            "--provider",
            "openai",
            "--base-url",
            "https://api.deepseek.com/v1",
            "--model",
            "deepseek-chat",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["provider"] == "openai"
    assert captured["base_url"] == "https://api.deepseek.com/v1"
    assert captured["model"] == "deepseek-chat"

    # JSON payload is pretty-printed after the human-readable summary lines.
    json_start = result.output.find("{")
    json_end = result.output.rfind("}")
    assert json_start != -1 and json_end != -1, result.output
    payload = json.loads(result.output[json_start:json_end + 1])
    assert payload["suite_name"] == "cli-suite"
