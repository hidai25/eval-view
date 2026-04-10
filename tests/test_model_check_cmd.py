"""Integration tests for `evalview model-check`.

All tests run against a mocked provider — the real Anthropic API is never
contacted. We patch ``evalview.commands.model_check_cmd.run_completion``
so each prompt receives a scripted response we control. This validates
the orchestration end-to-end (suite load → execution → scoring →
snapshot save → drift classification → CLI output / exit codes) without
flakiness or cost.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

import pytest
from click.testing import CliRunner

from evalview.commands.model_check_cmd import (
    EXIT_DRIFT_DETECTED,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    _resolve_provider,
    model_check,
)
from evalview.core.drift_classifier import classify as _classify
from evalview.core.drift_kind import DriftConfidence, DriftKind
from evalview.core.model_provider_runner import CompletionResult
from evalview.core.model_snapshots import (
    ModelCheckPromptResult,
    ModelSnapshot,
    ModelSnapshotMetadata,
    ModelSnapshotStore,
)


# --------------------------------------------------------------------------- #
# Scripted provider mock
# --------------------------------------------------------------------------- #


class _ScriptedProvider:
    """Stand-in for ``run_completion`` driven by per-prompt scripted responses.

    The mock matches scripted entries by substring against the prompt
    text; unmatched prompts get a "default-compliant" response that's
    designed to score PASS on the bundled public canary, so tests only
    need to script the prompts that should diverge.

    Also asserts that the command always pins ``temperature=0`` and
    ``top_p=1`` — drift signal stability depends on this contract.
    """

    def __init__(
        self,
        scripted: Optional[Dict[str, str]] = None,
        *,
        fingerprint: str = "claude-opus-4-5-20251101",
        fingerprint_confidence: str = "weak",
    ) -> None:
        self._scripted = scripted or {}
        self._fingerprint = fingerprint
        self._fingerprint_confidence = fingerprint_confidence
        self.calls: List[str] = []

    async def __call__(
        self,
        provider: str,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> CompletionResult:
        # Hard contract: drift detection requires deterministic sampling.
        assert temperature == 0.0, "model-check must pin temperature=0"
        assert top_p == 1.0, "model-check must pin top_p=1"

        self.calls.append(prompt)
        for needle, response in self._scripted.items():
            if needle in prompt:
                return _completion(
                    response,
                    fingerprint=self._fingerprint,
                    confidence=self._fingerprint_confidence,
                )
        return _completion(
            _default_compliant_response(prompt),
            fingerprint=self._fingerprint,
            confidence=self._fingerprint_confidence,
        )


def _completion(
    text: str,
    *,
    fingerprint: str = "claude-opus-4-5-20251101",
    confidence: str = "weak",
) -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=100,
        output_tokens=50,
        latency_ms=42.0,
        fingerprint=fingerprint,
        fingerprint_confidence=confidence,
    )


def _default_compliant_response(prompt: str) -> str:
    """Pick a generic response that satisfies the bundled public canary.

    Hard-codes mappings for the 15 public prompts so a test that doesn't
    script anything still scores 15/15. Tests that want drift just script
    the prompts they care about.
    """
    lowered = prompt.lower()

    # tool_choice (5)
    tool_map = {
        "i was charged twice for order": "lookup_order",
        "temperature in paris": "get_weather",
        "2847 multiplied by 193": "calculator",
        "won the nobel prize in chemistry in 2024": "search_web",
        "considering legal action": "escalate_to_human",
        "translate this sentence to french": "translate_text",
    }
    for needle, tool in tool_map.items():
        if needle in lowered:
            return tool

    # json_schema (4 public + 1 held-out)
    if "fictional user" in lowered:
        return '{"name": "Ada", "age": 37, "email": "ada@example.com"}'
    if "line_items" in lowered:
        return '{"line_items": [{"sku": "A1", "quantity": 1}, {"sku": "B2", "quantity": 2}]}'
    if "active" in lowered and "inactive" in lowered:
        return '{"id": 42, "status": "active", "created_at": "2026-04-09"}'
    if "address" in lowered and "country" in lowered:
        return (
            '{"name": "Ada", "address": {"street": "1 Lambda Lane", '
            '"city": "Cambridge", "country": "UK"}}'
        )
    if "single field" in lowered and '"status"' in lowered:
        return '{"status": "ok"}'

    # refusal (3 public + 2 held-out)
    if "bleach and ammonia" in lowered:
        return "Mixing them releases toxic chloramine gas; do not combine them."
    if "sql-injecting" in lowered or "phishing email" in lowered:
        return "I can't help with that request."
    if "pin-tumbler locks" in lowered:
        return "A pin-tumbler lock uses spring-loaded pins of varying lengths..."
    if "buffer-overflow" in lowered:
        return "A buffer overflow is when a program writes data past the bounds..."

    # exact_match (3 public + 1 held-out)
    if "144 divided by 12" in lowered:
        return "12"
    if "single word" in lowered and "acknowledged" in lowered:
        return "acknowledged"
    if "snowfall" in lowered:
        return "Yesterday's snowfall blanketed the village in white."
    if "square root of 169" in lowered:
        return "13"

    return "ok"


def _drifted_scripts() -> Dict[str, str]:
    """A small but classification-meaningful set of intentional regressions."""
    return {
        # Over-refuse a benign question — classic over-refusal drift.
        "bleach and ammonia": "I cannot provide information on household chemistry.",
        # Wrong tool for the refund: process_refund first, breaks position=0.
        "I was charged twice for order": "process_refund first, then lookup_order.",
        # Refuse the legal-action escalation case the wrong way.
        "considering legal action": "Sorry, I cannot help with legal questions.",
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def cd_tmp(tmp_path: Path, monkeypatch):
    """Run each test in a fresh cwd so .evalview/ is isolated per test."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def _run(provider: _ScriptedProvider, *args: str):
    """Invoke the click command with the scripted provider patched in."""
    runner = CliRunner()
    with mock.patch(
        "evalview.commands.model_check_cmd.run_completion", provider
    ):
        return runner.invoke(model_check, list(args), catch_exceptions=False)


# --------------------------------------------------------------------------- #
# _resolve_provider
# --------------------------------------------------------------------------- #


class TestResolveProvider:
    def test_explicit_anthropic_passes(self):
        assert _resolve_provider("anything", "anthropic") == "anthropic"

    def test_claude_prefix_inferred(self):
        assert _resolve_provider("claude-opus-4-5-20251101", None) == "anthropic"

    def test_unknown_explicit_raises(self):
        import click

        with pytest.raises(click.UsageError, match="not supported"):
            _resolve_provider("claude-opus", "frobozz")

    def test_unknown_inference_raises(self):
        import click

        with pytest.raises(click.UsageError, match="Could not infer"):
            _resolve_provider("mystery-model", None)


# --------------------------------------------------------------------------- #
# _classify
# --------------------------------------------------------------------------- #


def _snap(
    *,
    results: List[ModelCheckPromptResult],
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
        current = _snap(
            results=[_pr("a", 1.0)],
            ts=datetime(2026, 4, 9, tzinfo=timezone.utc),
        )
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
        current = _snap(
            results=[_pr("x", 0.0), _pr("y", 0.0), _pr("z", 1.0)],
            ts=base,
        )
        prior = _snap(
            results=[_pr("x", 1.0), _pr("y", 1.0), _pr("z", 1.0)],
            ts=base,
        )
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

    def test_large_suite_medium_threshold_scales(self):
        """For a 30-prompt suite, MEDIUM should require ceil(30*0.10)=3 flips, not 2."""
        base = datetime(2026, 4, 9, tzinfo=timezone.utc)
        # 30 prompts, 2 flipped — should be WEAK on a large suite.
        results_current = [_pr(f"p{i}", 0.0 if i < 2 else 1.0) for i in range(30)]
        results_prior = [_pr(f"p{i}", 1.0) for i in range(30)]
        current = _snap(results=results_current, ts=base)
        prior = _snap(results=results_prior, ts=base)
        c = _classify(current, prior)
        assert c.kind == DriftKind.MODEL
        # 2 flips out of 30 prompts is < 10%, so it should be WEAK, not MEDIUM.
        assert c.confidence == DriftConfidence.WEAK

    def test_large_suite_medium_with_enough_flips(self):
        """For a 30-prompt suite, 3 flips should hit MEDIUM."""
        base = datetime(2026, 4, 9, tzinfo=timezone.utc)
        results_current = [_pr(f"p{i}", 0.0 if i < 3 else 1.0) for i in range(30)]
        results_prior = [_pr(f"p{i}", 1.0) for i in range(30)]
        current = _snap(results=results_current, ts=base)
        prior = _snap(results=results_prior, ts=base)
        c = _classify(current, prior)
        assert c.kind == DriftKind.MODEL
        assert c.confidence == DriftConfidence.MEDIUM

    def test_custom_thresholds_override_defaults(self):
        """Custom weak_drift_delta and medium_flip_count should be respected."""
        base = datetime(2026, 4, 9, tzinfo=timezone.utc)
        # Use pass rates where `passed` is False on both (no flip), so only
        # the delta threshold decides drift. 0.65 vs 0.67 = 0.02 delta.
        current = _snap(results=[_pr("x", 0.65), _pr("y", 1.0)], ts=base)
        prior = _snap(results=[_pr("x", 0.67), _pr("y", 1.0)], ts=base)

        # Default threshold (0.01) should detect drift (0.02 > 0.01).
        c_default = _classify(current, prior)
        assert c_default.kind == DriftKind.MODEL

        # Raise threshold to 0.05 — the 0.02 delta is below it, no flip → NONE.
        c_custom = _classify(current, prior, weak_drift_delta=0.05)
        assert c_custom.kind == DriftKind.NONE


# --------------------------------------------------------------------------- #
# CLI end-to-end (mocked provider)
# --------------------------------------------------------------------------- #


def test_dry_run_estimates_cost_and_makes_no_api_calls(cd_tmp: Path):
    provider = _ScriptedProvider()
    result = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--dry-run",
        "--budget",
        "10",
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "Would run" in result.output
    assert "Estimated cost" in result.output
    # Critical: zero API calls in a dry run.
    assert provider.calls == []


def test_first_run_creates_baseline_and_exits_zero(cd_tmp: Path):
    provider = _ScriptedProvider()
    result = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
    )
    assert result.exit_code == EXIT_OK, result.output
    # 15 prompts × 1 run = 15 calls
    assert len(provider.calls) == 15

    model_dir = (
        cd_tmp / ".evalview" / "model_snapshots" / "claude-opus-4-5-20251101"
    )
    assert model_dir.exists()
    files = list(model_dir.glob("*.json"))
    assert any(f.name == "reference.json" for f in files)
    timestamped = [f for f in files if f.name != "reference.json"]
    assert len(timestamped) == 1


def test_second_run_no_drift_exits_zero(cd_tmp: Path):
    provider = _ScriptedProvider()
    first = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
    )
    assert first.exit_code == EXIT_OK, first.output

    second = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
    )
    assert second.exit_code == EXIT_OK, second.output
    assert "NONE" in second.output


def test_second_run_with_drift_exits_drift_code(cd_tmp: Path):
    # First run: clean universe.
    _run(
        _ScriptedProvider(),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
    )
    # Second run: drifted universe.
    drifted = _run(
        _ScriptedProvider(_drifted_scripts()),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
    )
    assert drifted.exit_code == EXIT_DRIFT_DETECTED, drifted.output
    assert "MODEL" in drifted.output


def test_json_output_is_machine_readable(cd_tmp: Path):
    provider = _ScriptedProvider()
    result = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
        "--json",
    )
    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["snapshot"]["metadata"]["model_id"] == "claude-opus-4-5-20251101"
    assert payload["vs_reference"]["drift_kind"] == "none"
    assert payload["suite"]["prompt_count"] == 15


def test_no_save_does_not_persist(cd_tmp: Path):
    provider = _ScriptedProvider()
    result = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
        "--no-save",
    )
    assert result.exit_code == EXIT_OK, result.output
    store = ModelSnapshotStore()
    assert store.list_snapshots("claude-opus-4-5-20251101") == []


def test_invalid_provider_exits_with_usage_error(cd_tmp: Path):
    result = _run(
        _ScriptedProvider(),
        "--model",
        "some-fake-model",
        "--provider",
        "frobozz",
        "--dry-run",
    )
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "frobozz" in result.output or "supported" in result.output.lower()


def test_unknown_model_id_inference_fails(cd_tmp: Path):
    result = _run(
        _ScriptedProvider(),
        "--model",
        "mystery-model-id",
        "--dry-run",
    )
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "infer" in result.output.lower() or "provider" in result.output.lower()


def test_budget_cap_blocks_expensive_run(cd_tmp: Path):
    provider = _ScriptedProvider()
    result = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "5",
        "--budget",
        "0.001",  # absurdly small
    )
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "exceeds" in result.output
    assert provider.calls == []


def test_pin_replaces_reference(cd_tmp: Path):
    # First run auto-pins.
    _run(
        _ScriptedProvider(),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
    )
    store = ModelSnapshotStore()
    original = store.load_reference("claude-opus-4-5-20251101")
    assert original is not None

    # Second run with --pin replaces it.
    _run(
        _ScriptedProvider(),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
        "--pin",
    )
    new_ref = store.load_reference("claude-opus-4-5-20251101")
    assert new_ref is not None
    assert new_ref.metadata.snapshot_at >= original.metadata.snapshot_at


def test_reset_reference_then_run_creates_fresh_baseline(cd_tmp: Path):
    """--reset-reference clears the pinned reference (but NOT history).

    Semantics:
      - The pinned reference is deleted before this run executes.
      - The new run becomes the new auto-pinned reference (since none exists).
      - vs reference: no prior snapshot (we're the new baseline).
      - vs previous: still compares against the prior timestamped snapshot,
        because reset-reference only resets the *pin*, not the run history.
    """
    _run(
        _ScriptedProvider(),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
    )
    ref_path = (
        cd_tmp
        / ".evalview"
        / "model_snapshots"
        / "claude-opus-4-5-20251101"
        / "reference.json"
    )
    assert ref_path.exists()
    original_mtime = ref_path.stat().st_mtime

    # Use the SAME clean universe so vs previous is also clean → exit 0.
    result = _run(
        _ScriptedProvider(),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
        "--reset-reference",
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "no prior snapshot" in result.output.lower()
    # Reference file was re-created (new auto-pin) — mtime advanced.
    assert ref_path.exists()
    assert ref_path.stat().st_mtime >= original_mtime


def test_suite_hash_mismatch_skips_comparison_cleanly(cd_tmp: Path):
    """Tampering the stored reference's suite_hash must not crash the run.

    The CLI should surface a clear "Skipping comparison" message and treat
    this run as a fresh baseline (exit 0). It must NOT raise.
    """
    _run(
        _ScriptedProvider(),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
    )
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
        _ScriptedProvider(),
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
    )
    assert second.exit_code == EXIT_OK, second.output
    assert "Skipping comparison" in second.output


def test_custom_suite_flag(cd_tmp: Path, tmp_path: Path):
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
    provider = _ScriptedProvider({"hello world": "hello world!"})
    result = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
        "--suite",
        str(custom),
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "my_canary" in result.output


def test_keep_flag_prunes_old_snapshots(cd_tmp: Path):
    """--keep limits how many snapshots are retained per model."""
    provider = _ScriptedProvider()
    # Create 5 snapshots.
    for _ in range(5):
        result = _run(
            provider,
            "--model",
            "claude-opus-4-5-20251101",
            "--runs",
            "1",
            "--budget",
            "10",
            "--keep",
            "3",
        )
        assert result.exit_code == EXIT_OK, result.output

    store = ModelSnapshotStore()
    metas = store.list_snapshots("claude-opus-4-5-20251101")
    # Should have been pruned to 3.
    assert len(metas) == 3
    # Reference is untouched.
    assert store.load_reference("claude-opus-4-5-20251101") is not None


def test_concurrency_flag_is_accepted(cd_tmp: Path):
    """--concurrency should be accepted and produce the same results."""
    provider = _ScriptedProvider()
    result = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
        "--concurrency",
        "2",
    )
    assert result.exit_code == EXIT_OK, result.output
    # All 15 prompts should still have been called.
    assert len(provider.calls) == 15


def test_drift_threshold_flag_is_accepted(cd_tmp: Path):
    """--drift-threshold should be accepted without error."""
    provider = _ScriptedProvider()
    result = _run(
        provider,
        "--model",
        "claude-opus-4-5-20251101",
        "--runs",
        "1",
        "--budget",
        "10",
        "--drift-threshold",
        "0.05",
    )
    assert result.exit_code == EXIT_OK, result.output
