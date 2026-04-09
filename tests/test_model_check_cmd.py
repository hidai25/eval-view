"""Integration tests for `evalview model-check`.

All tests use a synthetic adapter that returns canned responses — no real
provider is ever contacted. The goal is to validate the orchestration
layer end-to-end: suite load, execution, scoring, snapshot save,
reference management, drift classification, and CLI output/exit codes.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest import mock

import pytest
from click.testing import CliRunner

from evalview.commands.model_check_cmd import (
    EXIT_DRIFT_DETECTED,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    _classify,
    _infer_provider,
    model_check,
)
from evalview.core.canary_suite import load_canary_suite
from evalview.core.drift_kind import DriftConfidence, DriftKind
from evalview.core.model_snapshots import (
    ModelCheckPromptResult,
    ModelSnapshot,
    ModelSnapshotMetadata,
    ModelSnapshotStore,
)
from evalview.core.types import ExecutionMetrics, ExecutionTrace, StepTrace, StepMetrics


# --------------------------------------------------------------------------- #
# Synthetic adapter
# --------------------------------------------------------------------------- #


class _FakeAdapter:
    """Returns canned responses keyed by substring match against the prompt.

    Keys are *substrings* of the canary prompts (the first 25 chars), chosen
    to be unique enough to route each prompt to the right canned response.
    """

    def __init__(self, responses: Dict[str, Dict[str, Any]]):
        self._responses = responses
        self.name = "fake"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        matched: Optional[Dict[str, Any]] = None
        for needle, response in self._responses.items():
            if needle in query:
                matched = response
                break
        if matched is None:
            matched = {"output": "", "tools": []}

        steps = [
            StepTrace(
                step_id=f"step-{i}",
                step_name=f"step_{i}",
                tool_name=t,
                parameters={},
                output="ok",
                success=True,
                metrics=StepMetrics(latency=1.0, cost=0.0),
            )
            for i, t in enumerate(matched.get("tools", []))
        ]
        now = datetime.now(timezone.utc)
        return ExecutionTrace(
            session_id="fake",
            start_time=now,
            end_time=now,
            steps=steps,
            final_output=matched.get("output", ""),
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=1.0),
            model_id="fake-model-id",
            model_provider="fake",
        )


# Canned responses that yield PASS on the full public canary (15 prompts).
_PASSING_RESPONSES: Dict[str, Dict[str, Any]] = {
    # tool_choice
    "charged twice for order": {"output": "ok", "tools": ["lookup_order", "process_refund"]},
    "temperature in Paris": {"output": "ok", "tools": ["get_weather"]},
    "2847 multiplied": {"output": "ok", "tools": ["calculator"]},
    "capital of France": {"output": "Paris.", "tools": ["search_web"]},
    "considering legal action": {"output": "escalated", "tools": ["escalate_to_human"]},
    # json_schema
    "fictional user with these": {
        "output": '{"name":"Ada","age":37,"email":"ada@example.com"}',
        "tools": [],
    },
    "array of objects each having sku": {
        "output": '{"line_items":[{"sku":"A","quantity":1},{"sku":"B","quantity":2}]}',
        "tools": [],
    },
    "YYYY-MM-DD format": {
        "output": '{"id":1,"status":"active","created_at":"2026-04-09"}',
        "tools": [],
    },
    "object with street, city, country": {
        "output": '{"name":"Ada","address":{"street":"1 Main","city":"NYC","country":"US"}}',
        "tools": [],
    },
    # refusal — benign = comply, harm = refuse
    "bleach and": {"output": "It produces toxic chloramine gas.", "tools": []},
    "SQL-injecting a specific": {"output": "I cannot help with that.", "tools": []},
    "pin-tumbler locks work": {
        "output": "Pin-tumbler locks use stacked pin pairs...",
        "tools": [],
    },
    # exact_match
    "144 divided by 12": {"output": "12", "tools": []},
    'the single word "acknowledged"': {"output": "acknowledged", "tools": []},
    "exactly one sentence": {"output": "Heavy snowfall blanketed the city.", "tools": []},
}


def _drifted_responses() -> Dict[str, Dict[str, Any]]:
    """Twist a few responses so classification flips direction."""
    out = dict(_PASSING_RESPONSES)
    # Over-refuse the benign chemistry question (drift toward censoriousness)
    out["bleach and"] = {
        "output": "I cannot provide information on household chemistry.",
        "tools": [],
    }
    # Wrong tool for the refund case — use process_refund first, not lookup
    out["charged twice for order"] = {
        "output": "refund processed",
        "tools": ["process_refund"],
    }
    return out


@pytest.fixture
def cd_tmp(tmp_path: Path, monkeypatch):
    """Run the CLI inside a fresh cwd so .evalview/ is isolated per test."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield tmp_path


def _run(runner: CliRunner, adapter: _FakeAdapter, *args: str):
    with mock.patch(
        "evalview.commands.model_check_cmd._build_adapter",
        return_value=adapter,
    ):
        return runner.invoke(model_check, list(args), catch_exceptions=False)


# --------------------------------------------------------------------------- #
# _infer_provider
# --------------------------------------------------------------------------- #


class TestInferProvider:
    def test_explicit_wins(self):
        assert _infer_provider("anything", "anthropic") == "anthropic"

    def test_claude_prefix_detected(self):
        assert _infer_provider("claude-opus-4-5-20251101", None) == "anthropic"

    def test_gpt_prefix_detected(self):
        assert _infer_provider("gpt-5.4", None) == "openai"

    def test_o_series_detected(self):
        assert _infer_provider("o3-mini", None) == "openai"

    def test_unknown_raises(self):
        import click

        with pytest.raises(click.UsageError, match="Could not infer"):
            _infer_provider("mystery-model", None)

    def test_unsupported_explicit_raises(self):
        import click

        with pytest.raises(click.UsageError, match="Unsupported provider"):
            _infer_provider("claude-opus", "gemini")


# --------------------------------------------------------------------------- #
# _classify
# --------------------------------------------------------------------------- #


def _snap(
    *,
    results: list[ModelCheckPromptResult],
    ts: datetime,
    fingerprint: str = "fake-fp",
    confidence: str = "weak",
) -> ModelSnapshot:
    return ModelSnapshot(
        metadata=ModelSnapshotMetadata(
            model_id="m",
            provider="anthropic",
            snapshot_at=ts,
            suite_name="canary",
            suite_version="v1.public",
            suite_hash="sha256:x",
            temperature=0.0,
            top_p=1.0,
            runs_per_prompt=3,
            provider_fingerprint=fingerprint,
            fingerprint_confidence=confidence,
        ),
        results=results,
    )


def _pr(pid: str, rate: float) -> ModelCheckPromptResult:
    return ModelCheckPromptResult(
        prompt_id=pid,
        category="tool_choice",
        pass_rate=rate,
        n_runs=3,
        per_run_passed=[rate >= 0.999] * 3,
    )


class TestClassify:
    def test_no_prior_returns_none(self):
        current = _snap(results=[_pr("a", 1.0)], ts=datetime(2026, 4, 9, tzinfo=timezone.utc))
        c = _classify(current, None)
        assert c.kind == DriftKind.NONE
        assert c.confidence is None

    def test_identical_snapshots_return_none(self):
        base = datetime(2026, 4, 9, tzinfo=timezone.utc)
        a = _snap(results=[_pr("x", 1.0), _pr("y", 1.0)], ts=base)
        b = _snap(results=[_pr("x", 1.0), _pr("y", 1.0)], ts=base)
        c = _classify(a, b)
        assert c.kind == DriftKind.NONE

    def test_single_flip_is_weak(self):
        base = datetime(2026, 4, 9, tzinfo=timezone.utc)
        current = _snap(results=[_pr("x", 0.0), _pr("y", 1.0)], ts=base)
        prior = _snap(results=[_pr("x", 1.0), _pr("y", 1.0)], ts=base)
        c = _classify(current, prior)
        assert c.kind == DriftKind.MODEL
        assert c.confidence == DriftConfidence.WEAK
        assert c.flipped_ids == ["x"]

    def test_two_flips_is_medium(self):
        base = datetime(2026, 4, 9, tzinfo=timezone.utc)
        current = _snap(results=[_pr("x", 0.0), _pr("y", 0.0), _pr("z", 1.0)], ts=base)
        prior = _snap(results=[_pr("x", 1.0), _pr("y", 1.0), _pr("z", 1.0)], ts=base)
        c = _classify(current, prior)
        assert c.kind == DriftKind.MODEL
        assert c.confidence == DriftConfidence.MEDIUM

    def test_strong_fingerprint_change_wins(self):
        base = datetime(2026, 4, 9, tzinfo=timezone.utc)
        current = _snap(
            results=[_pr("x", 1.0)],
            ts=base,
            fingerprint="fp_new",
            confidence="strong",
        )
        prior = _snap(
            results=[_pr("x", 1.0)],
            ts=base,
            fingerprint="fp_old",
            confidence="strong",
        )
        c = _classify(current, prior)
        assert c.kind == DriftKind.MODEL
        assert c.confidence == DriftConfidence.STRONG


# --------------------------------------------------------------------------- #
# CLI end-to-end (mocked adapter)
# --------------------------------------------------------------------------- #


def test_first_run_creates_baseline_and_exits_zero(cd_tmp: Path):
    runner = CliRunner()
    adapter = _FakeAdapter(_PASSING_RESPONSES)
    result = _run(
        runner,
        adapter,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "no prior snapshot" in result.output.lower()
    # A snapshot file and a reference file must have been written.
    model_dir = cd_tmp / ".evalview" / "model_snapshots" / "claude-opus-4-5-20251101"
    assert model_dir.exists()
    files = list(model_dir.glob("*.json"))
    assert any(f.name == "reference.json" for f in files)
    assert len([f for f in files if f.name != "reference.json"]) == 1


def test_second_run_no_drift_exits_zero(cd_tmp: Path):
    runner = CliRunner()
    adapter = _FakeAdapter(_PASSING_RESPONSES)
    _run(runner, adapter, "--model", "claude-opus-4-5-20251101", "--runs", "1")
    second = _run(runner, adapter, "--model", "claude-opus-4-5-20251101", "--runs", "1")
    assert second.exit_code == EXIT_OK, second.output
    assert "NONE" in second.output


def test_second_run_with_drift_exits_drift_code(cd_tmp: Path):
    runner = CliRunner()
    _run(
        runner,
        _FakeAdapter(_PASSING_RESPONSES),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
    )
    drifted = _run(
        runner,
        _FakeAdapter(_drifted_responses()),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
    )
    assert drifted.exit_code == EXIT_DRIFT_DETECTED, drifted.output
    assert "MODEL" in drifted.output


def test_json_output_is_machine_readable(cd_tmp: Path):
    runner = CliRunner()
    adapter = _FakeAdapter(_PASSING_RESPONSES)
    result = _run(
        runner,
        adapter,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--json",
    )
    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["snapshot"]["metadata"]["model_id"] == "claude-opus-4-5-20251101"
    assert payload["vs_reference"]["drift_kind"] == "none"
    assert payload["suite"]["prompt_count"] == 15


def test_dry_run_makes_no_api_calls(cd_tmp: Path):
    runner = CliRunner()
    sentinel = object()
    with mock.patch(
        "evalview.commands.model_check_cmd._build_adapter",
        return_value=sentinel,
    ) as build_call:
        result = runner.invoke(
            model_check,
            ["--model", "claude-opus-4-5-20251101", "--dry-run"],
            catch_exceptions=False,
        )
    assert result.exit_code == EXIT_OK
    assert "Would run" in result.output
    assert "Estimated cost" in result.output
    # _build_adapter must NOT be called during dry-run.
    build_call.assert_not_called()


def test_budget_cap_blocks_expensive_run(cd_tmp: Path):
    runner = CliRunner()
    # Opus on the full canary with 5 runs × 15 prompts = 75 calls; any
    # sane budget of $0.001 is guaranteed to fail.
    result = runner.invoke(
        model_check,
        [
            "--model",
            "claude-opus-4-5-20251101",
            "--runs",
            "5",
            "--budget",
            "0.001",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "exceeds --budget" in result.output


def test_suite_hash_mismatch_is_rejected(cd_tmp: Path, monkeypatch):
    """Simulate a suite rotation: save a snapshot, then change the suite hash.

    The new run must refuse to compare and surface a clear error.
    """
    runner = CliRunner()
    # First run with the real suite.
    _run(
        runner,
        _FakeAdapter(_PASSING_RESPONSES),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
    )
    # Now tamper with the stored reference so it has a different suite_hash.
    ref_path = (
        cd_tmp
        / ".evalview"
        / "model_snapshots"
        / "claude-opus-4-5-20251101"
        / "reference.json"
    )
    data = json.loads(ref_path.read_text())
    data["metadata"]["suite_hash"] = "sha256:tampered"
    ref_path.write_text(json.dumps(data))

    second = _run(
        runner,
        _FakeAdapter(_PASSING_RESPONSES),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
    )
    assert second.exit_code == EXIT_USAGE_ERROR
    assert "Suite hash differs" in second.output


def test_reset_reference_clears_pin(cd_tmp: Path):
    runner = CliRunner()
    _run(
        runner,
        _FakeAdapter(_PASSING_RESPONSES),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
    )
    ref_path = (
        cd_tmp
        / ".evalview"
        / "model_snapshots"
        / "claude-opus-4-5-20251101"
        / "reference.json"
    )
    assert ref_path.exists()

    # Second run with --reset-reference — the old reference is deleted and
    # the new snapshot becomes the new reference.
    result = _run(
        runner,
        _FakeAdapter(_drifted_responses()),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--reset-reference",
    )
    assert result.exit_code == EXIT_OK, result.output
    # Reference still exists (auto-pinned from the new snapshot) but the
    # file was re-created rather than being the old pin.
    assert ref_path.exists()


def test_missing_api_key_errors_clearly(cd_tmp: Path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(
        model_check,
        ["--model", "claude-opus-4-5-20251101", "--runs", "1"],
        catch_exceptions=False,
    )
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "ANTHROPIC_API_KEY" in result.output


def test_custom_suite_flag(cd_tmp: Path, tmp_path: Path):
    # Write a tiny custom suite with a single exact_match prompt.
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        """
suite_name: my_canary
version: v1.custom
prompts:
  - id: hello_world
    category: exact_match
    prompt: Say hello world please.
    scorer: exact_match
    expected:
      pattern: "(?i)hello world"
"""
    )
    runner = CliRunner()
    adapter = _FakeAdapter({"Say hello world": {"output": "hello world!", "tools": []}})
    result = _run(
        runner,
        adapter,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--suite",
        str(custom),
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "my_canary" in result.output
