"""Unit tests for core/drift_classifier.py."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evalview.core.drift_classifier import (
    DEFAULT_MEDIUM_FLIP_COUNT,
    DEFAULT_MEDIUM_FLIP_RATIO,
    DEFAULT_WEAK_DRIFT_DELTA,
    PromptDelta,
    _effective_medium_flip_count,
    classify,
)
from evalview.core.drift_kind import DriftConfidence, DriftKind
from evalview.core.model_snapshots import (
    ModelCheckPromptResult,
    ModelSnapshot,
    ModelSnapshotMetadata,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _snap(*, results, ts, fingerprint="fp", confidence="weak"):
    return ModelSnapshot(
        metadata=ModelSnapshotMetadata(
            model_id="m",
            provider="anthropic",
            snapshot_at=ts,
            suite_name="canary",
            suite_version="v1",
            suite_hash="sha256:x",
            temperature=0.0,
            top_p=1.0,
            runs_per_prompt=3,
            provider_fingerprint=fingerprint,
            fingerprint_confidence=confidence,
        ),
        results=results,
    )


def _pr(pid, rate):
    return ModelCheckPromptResult(
        prompt_id=pid,
        category="tool_choice",
        pass_rate=rate,
        n_runs=3,
        per_run_passed=[rate >= 0.999] * 3,
    )


_BASE = datetime(2026, 4, 9, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# _effective_medium_flip_count
# --------------------------------------------------------------------------- #


class TestEffectiveMediumFlipCount:
    def test_small_suite_uses_fixed_count(self):
        assert _effective_medium_flip_count(15, 2, 0.10) == 2
        assert _effective_medium_flip_count(20, 2, 0.10) == 2

    def test_large_suite_scales_with_ratio(self):
        # 30 * 0.10 = 3.0 → ceil = 3
        assert _effective_medium_flip_count(30, 2, 0.10) == 3
        # 50 * 0.10 = 5.0 → ceil = 5
        assert _effective_medium_flip_count(50, 2, 0.10) == 5

    def test_large_suite_never_below_fixed_minimum(self):
        # Even with a low ratio, should never go below medium_flip_count.
        assert _effective_medium_flip_count(100, 5, 0.01) == 5

    def test_boundary_at_21(self):
        # 21 * 0.10 = 2.1 → ceil = 3
        assert _effective_medium_flip_count(21, 2, 0.10) == 3


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #


class TestClassify:
    def test_none_other_returns_none(self):
        c = classify(_snap(results=[_pr("a", 1.0)], ts=_BASE), None)
        assert c.kind == DriftKind.NONE
        assert c.confidence is None

    def test_identical_returns_none(self):
        a = _snap(results=[_pr("x", 1.0)], ts=_BASE)
        b = _snap(results=[_pr("x", 1.0)], ts=_BASE)
        c = classify(a, b)
        assert c.kind == DriftKind.NONE

    def test_custom_weak_delta_threshold(self):
        # Use pass_rate=0.65 vs 0.67 — both below 0.999 so `passed` is False
        # on both, meaning no flip. Only the delta threshold matters here.
        current = _snap(results=[_pr("x", 0.65)], ts=_BASE)
        prior = _snap(results=[_pr("x", 0.67)], ts=_BASE)

        # Default (0.01): 0.02 delta should detect drift.
        c = classify(current, prior)
        assert c.kind == DriftKind.MODEL

        # Higher threshold (0.05): same 0.02 delta is below it → no drift.
        c = classify(current, prior, weak_drift_delta=0.05)
        assert c.kind == DriftKind.NONE

    def test_custom_medium_flip_count(self):
        # 2 flips — default MEDIUM for small suite.
        current = _snap(results=[_pr("a", 0.0), _pr("b", 0.0), _pr("c", 1.0)], ts=_BASE)
        prior = _snap(results=[_pr("a", 1.0), _pr("b", 1.0), _pr("c", 1.0)], ts=_BASE)

        c = classify(current, prior)
        assert c.confidence == DriftConfidence.MEDIUM

        # Raise the threshold to 3 — now 2 flips is only WEAK.
        c = classify(current, prior, medium_flip_count=3)
        assert c.confidence == DriftConfidence.WEAK

    def test_deltas_populated(self):
        current = _snap(results=[_pr("x", 0.5), _pr("y", 1.0)], ts=_BASE)
        prior = _snap(results=[_pr("x", 1.0), _pr("y", 1.0)], ts=_BASE)
        c = classify(current, prior)
        assert len(c.deltas) == 2
        assert isinstance(c.deltas[0], PromptDelta)
        x_delta = next(d for d in c.deltas if d.prompt_id == "x")
        assert x_delta.delta == pytest.approx(-0.5)


# --------------------------------------------------------------------------- #
# Constants exported
# --------------------------------------------------------------------------- #


def test_default_constants_are_sensible():
    assert DEFAULT_WEAK_DRIFT_DELTA > 0
    assert DEFAULT_MEDIUM_FLIP_COUNT >= 2
    assert 0 < DEFAULT_MEDIUM_FLIP_RATIO < 1
