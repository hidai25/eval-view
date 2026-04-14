"""Tests for the autopr glue: synthesizer + command + monitor incidents feed."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from evalview.commands.autopr_cmd import (
    _commit_message,
    _existing_slugs,
    autopr,
    load_incidents,
    write_regression_test,
)
from evalview.commands.monitor_cmd import (
    DEFAULT_INCIDENTS_PATH,
    _append_incidents,
    _build_incident_record,
)
from evalview.core.diff import DiffStatus
from evalview.core.regression_synth import (
    SynthesisError,
    incident_slug,
    synthesize_regression_test,
    truncate_output,
)
from evalview.core.types import TestCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _incident(**overrides: Any) -> Dict[str, Any]:
    """Minimal-but-realistic incident record for synthesizer tests."""
    base: Dict[str, Any] = {
        "version": 1,
        "timestamp": "2026-04-14T12:34:56Z",
        "test_name": "refund-request",
        "query": "I want a refund for order #123",
        "status": "REGRESSION",
        "score_delta": -30.0,
        "baseline_tools": ["lookup_order", "check_policy", "process_refund"],
        "actual_tools": ["lookup_order", "process_refund", "escalate_to_human"],
        "baseline_output": "After checking our policy, I can confirm a refund.",
        "actual_output": "Sure, I've processed your refund for $999.",
        "model_changed": False,
        "golden_model_id": "claude-opus-4-5-20251101",
        "actual_model_id": "claude-opus-4-5-20251101",
        "source_file": "tests/refund-request.yaml",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Synthesizer — pure function
# ---------------------------------------------------------------------------


class TestSynthesizer:
    def test_produces_valid_test_case(self) -> None:
        test = synthesize_regression_test(_incident())
        # The output must be loadable as a proper TestCase — that's the
        # contract that keeps the synthesized YAML compatible with the rest
        # of evalview.
        tc = TestCase(**test)
        assert tc.input.query == "I want a refund for order #123"
        assert tc.thresholds.min_score == 90.0
        assert tc.suite_type == "regression"
        assert tc.gate == "strict"
        assert "incident" in tc.tags and "autopr" in tc.tags

    def test_forbids_newly_added_tools(self) -> None:
        test = synthesize_regression_test(_incident())
        # `escalate_to_human` was in the actual but not the baseline →
        # the regression test must forbid it.
        assert "escalate_to_human" in test["expected"]["forbidden_tools"]
        # Tools present in both should NOT be in forbidden_tools.
        assert "lookup_order" not in test["expected"].get("forbidden_tools", [])

    def test_preserves_baseline_tools_as_expected(self) -> None:
        test = synthesize_regression_test(_incident())
        assert test["expected"]["tools"] == [
            "lookup_order",
            "check_policy",
            "process_refund",
        ]

    def test_pins_novel_output_phrases(self) -> None:
        test = synthesize_regression_test(_incident())
        not_contains = test["expected"]["output"]["not_contains"]
        # The bad output says "processed your refund for $999" — that phrase
        # should survive into not_contains because it's absent from the
        # baseline.
        assert any("$999" in phrase for phrase in not_contains)

    def test_negative_phrases_prefer_short(self) -> None:
        long_actual = (
            "I've processed your refund. " * 20
            + "Also here is a long novel sentence that should get truncated."
        )
        test = synthesize_regression_test(
            _incident(actual_output=long_actual, baseline_output="")
        )
        not_contains = test["expected"]["output"]["not_contains"]
        assert all(len(p) <= 200 for p in not_contains)
        assert len(not_contains) <= 3

    def test_slug_is_stable_and_unique(self) -> None:
        a = incident_slug(_incident())
        b = incident_slug(_incident())
        assert a == b  # deterministic

        c = incident_slug(_incident(query="something totally different"))
        assert a != c  # query change → different slug

        d = incident_slug(_incident(test_name="billing-dispute"))
        assert a != d  # test name change → different slug

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(SynthesisError):
            synthesize_regression_test({"query": "only a query"})
        with pytest.raises(SynthesisError):
            synthesize_regression_test({"test_name": "only a name"})

    def test_graceful_when_no_baseline_tools(self) -> None:
        test = synthesize_regression_test(
            _incident(baseline_tools=None, actual_tools=None)
        )
        assert "tools" not in test["expected"]
        assert "forbidden_tools" not in test["expected"]

    def test_truncate_output(self) -> None:
        assert truncate_output(None) is None
        assert truncate_output("short") == "short"
        long = "x" * 3000
        trimmed = truncate_output(long, limit=100) or ""
        assert trimmed.endswith("... [truncated]")
        assert len(trimmed) <= 120

    def test_meta_incident_carries_slug(self) -> None:
        test = synthesize_regression_test(_incident())
        assert test["meta"]["incident"]["slug"] == incident_slug(_incident())
        assert test["meta"]["incident"]["added_tools"] == ["escalate_to_human"]


# ---------------------------------------------------------------------------
# autopr command — integration with a fake incidents file
# ---------------------------------------------------------------------------


def _write_incidents(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _init_repo(path: Path) -> None:
    """Initialise a throwaway git repo with a deterministic identity.

    Disables commit signing so the tests don't inherit the outer machine's
    signing configuration. Some sandboxed CI environments set a global
    ``commit.gpgsign=true`` plus a proxy signing helper that fails for
    unauthenticated test runs — keeping the test repo self-contained avoids
    that whole mess.
    """
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    for key, value in {
        "user.email": "ci@example.com",
        "user.name": "CI",
        "commit.gpgsign": "false",
        "tag.gpgsign": "false",
    }.items():
        subprocess.run(
            ["git", "config", "--local", key, value], cwd=path, check=True
        )


class TestAutoprCommand:
    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(incidents, [_incident()])
        tests_dir = tmp_path / "tests" / "regressions"

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                autopr,
                [
                    "--from",
                    str(incidents),
                    "--tests-dir",
                    str(tests_dir),
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0, result.output
        assert not tests_dir.exists()
        assert "Would write" in result.output

    def test_writes_new_test_files(self, tmp_path: Path) -> None:
        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(incidents, [_incident()])
        tests_dir = tmp_path / "tests" / "regressions"

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                autopr,
                [
                    "--from",
                    str(incidents),
                    "--tests-dir",
                    str(tests_dir),
                ],
            )
        assert result.exit_code == 0, result.output
        yaml_files = list(tests_dir.glob("*.yaml"))
        assert len(yaml_files) == 1
        test = yaml.safe_load(yaml_files[0].read_text())
        # Round-trip through TestCase to validate structure.
        TestCase(**test)

    def test_skips_existing_slugs(self, tmp_path: Path) -> None:
        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(incidents, [_incident(), _incident()])
        tests_dir = tmp_path / "tests" / "regressions"

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # First invocation writes one file (both incidents have the
            # same slug because they're identical, so one dedupes).
            runner.invoke(
                autopr,
                ["--from", str(incidents), "--tests-dir", str(tests_dir)],
            )
            # Second invocation sees the existing slug and writes nothing
            # new.
            result = runner.invoke(
                autopr,
                ["--from", str(incidents), "--tests-dir", str(tests_dir)],
            )
        assert result.exit_code == 0
        assert "Skipped" in result.output
        assert len(list(tests_dir.glob("*.yaml"))) == 1

    def test_require_new_fails_cleanly_when_nothing_to_do(self, tmp_path: Path) -> None:
        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(incidents, [])  # empty file
        tests_dir = tmp_path / "tests" / "regressions"

        runner = CliRunner()
        result = runner.invoke(
            autopr,
            [
                "--from",
                str(incidents),
                "--tests-dir",
                str(tests_dir),
                "--require-new",
            ],
        )
        assert result.exit_code == 1

    def test_malformed_incident_is_skipped(self, tmp_path: Path) -> None:
        incidents = tmp_path / "incidents.jsonl"
        incidents.parent.mkdir(parents=True, exist_ok=True)
        with incidents.open("w", encoding="utf-8") as f:
            f.write(json.dumps(_incident()) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps({"test_name": "no-query"}) + "\n")

        records = load_incidents(incidents)
        # The malformed row is dropped; the 'no-query' one survives the
        # JSON parse but will fail synthesis later.
        assert len(records) == 2

        tests_dir = tmp_path / "tests" / "regressions"
        runner = CliRunner()
        result = runner.invoke(
            autopr,
            ["--from", str(incidents), "--tests-dir", str(tests_dir)],
        )
        # Exit 0 because at least one valid test was written.
        assert result.exit_code == 0
        assert len(list(tests_dir.glob("*.yaml"))) == 1

    def test_commit_creates_branch_and_commit(self, tmp_path: Path, monkeypatch) -> None:
        """End-to-end: --commit runs git init + add + commit on a throwaway repo."""
        # Real git, throwaway repo — faster and less mocky than stubbing
        # subprocess.run. Force commit signing off so the test works in
        # sandboxes that enforce signing globally.
        _init_repo(tmp_path)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_path, check=True,
        )

        incidents = tmp_path / ".evalview" / "incidents.jsonl"
        _write_incidents(incidents, [_incident()])

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            autopr,
            [
                "--from",
                str(incidents),
                "--tests-dir",
                "tests/regressions",
                "--commit",
                "--no-push",
            ],
        )
        assert result.exit_code == 0, result.output

        branches = subprocess.run(
            ["git", "branch", "--list"], cwd=tmp_path, capture_output=True, text=True
        ).stdout
        assert "evalview/autopr/" in branches

        log = subprocess.run(
            ["git", "log", "--oneline", "-n", "2"],
            cwd=tmp_path, capture_output=True, text=True,
        ).stdout
        assert "regression tests" in log

    def test_commit_refuses_dirty_tree(self, tmp_path: Path, monkeypatch) -> None:
        _init_repo(tmp_path)
        (tmp_path / "README.md").write_text("hello")
        subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
        # Dirty the tracked file — this should trip the safety guard.
        (tmp_path / "README.md").write_text("dirty")

        incidents = tmp_path / ".evalview" / "incidents.jsonl"
        _write_incidents(incidents, [_incident()])

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            autopr,
            [
                "--from",
                str(incidents),
                "--tests-dir",
                "tests/regressions",
                "--commit",
                "--no-push",
            ],
        )
        assert result.exit_code == 2
        assert "uncommitted changes" in result.output


# ---------------------------------------------------------------------------
# Monitor incident writer
# ---------------------------------------------------------------------------


class _FakeStep:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name


class _FakeTrace:
    def __init__(self, steps: List[str], output: str) -> None:
        self.steps = [_FakeStep(s) for s in steps]
        self.final_output = output


class _FakeDiff:
    def __init__(self) -> None:
        self.overall_severity = DiffStatus.REGRESSION
        self.score_diff = -30.0
        self.model_changed = False
        self.golden_model_id = "claude-opus-4-5"
        self.actual_model_id = "claude-opus-4-5"


class _FakeGolden:
    def __init__(self, steps: List[str], output: str) -> None:
        self.trace = _FakeTrace(steps, output)


class _FakeResult:
    def __init__(self, test_case: str, steps: List[str], output: str) -> None:
        self.test_case = test_case
        self.trace = _FakeTrace(steps, output)


class _FakeTestCase:
    def __init__(self, name: str, query: str, source_file: str) -> None:
        self.name = name
        self.input = MagicMock()
        self.input.query = query
        self.source_file = source_file


class TestMonitorIncidentFeed:
    def test_build_incident_record_captures_trace_state(self) -> None:
        record = _build_incident_record(
            test_name="refund-request",
            diff=_FakeDiff(),
            test_case=_FakeTestCase(
                "refund-request",
                "I want a refund for order #123",
                "tests/refund-request.yaml",
            ),
            result=_FakeResult(
                "refund-request",
                ["lookup_order", "process_refund", "escalate_to_human"],
                "Sure, I processed your refund for $999.",
            ),
            golden=_FakeGolden(
                ["lookup_order", "check_policy", "process_refund"],
                "After checking our policy, I can confirm a refund.",
            ),
            cycle=7,
        )
        assert record["test_name"] == "refund-request"
        assert record["query"] == "I want a refund for order #123"
        assert record["status"] == "regression"
        assert record["score_delta"] == -30.0
        assert record["baseline_tools"] == [
            "lookup_order",
            "check_policy",
            "process_refund",
        ]
        assert "escalate_to_human" in record["actual_tools"]
        assert record["source_file"] == "tests/refund-request.yaml"
        assert record["cycle"] == 7

    def test_append_incidents_round_trips_through_synthesizer(
        self, tmp_path: Path
    ) -> None:
        """The monitor's record must be a valid synthesizer input."""
        path = tmp_path / ".evalview" / "incidents.jsonl"
        tc = _FakeTestCase(
            "refund-request",
            "I want a refund",
            "tests/refund-request.yaml",
        )
        result = _FakeResult("refund-request", ["lookup_order"], "bad")
        golden = _FakeGolden(["lookup_order", "check_policy"], "good")

        n = _append_incidents(
            path,
            [("refund-request", _FakeDiff())],
            {"refund-request": tc},
            {"refund-request": result},
            {"refund-request": golden},
            cycle=1,
        )
        assert n == 1

        records = load_incidents(path)
        assert len(records) == 1
        # End-to-end round trip: monitor → incidents.jsonl → synthesizer
        # → TestCase. This is the whole closed loop.
        test = synthesize_regression_test(records[0])
        TestCase(**test)

    def test_default_incidents_path_points_into_evalview_dir(self) -> None:
        assert str(DEFAULT_INCIDENTS_PATH).endswith("incidents.jsonl")
        assert ".evalview" in str(DEFAULT_INCIDENTS_PATH)
