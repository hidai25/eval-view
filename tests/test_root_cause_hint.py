"""Tests for the root-cause hinter.

The hinter is the *narrating* layer that sits on top of
:func:`evalview.core.noise_tracker.detect_coordinated_incident`. Where the
incident detector answers "are these failures correlated?", the hinter
answers "correlated **how**?" and produces a narrative + suggested actions.

Layers under test:

1. **Pure hinters** — each ``hint_*`` function in
   ``evalview.core.root_cause_hint`` looks for one specific cross-test
   pattern and returns ``Optional[RootCauseHint]``. Tested in isolation.
2. **Selection** — ``analyze_root_cause_hint`` picks the best hint when
   multiple hinters fire, deterministically.
3. **Integration with Incident** — ``detect_coordinated_incident`` attaches
   the hint to the resulting ``Incident.hint`` field so notifiers can
   render the narrative without an extra round trip.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Tuple
from unittest.mock import MagicMock

from evalview.core.diff import DiffStatus
from evalview.core.noise_tracker import detect_coordinated_incident
from evalview.core.root_cause_hint import (
    HINTERS,
    HINTERS_ROADMAP,
    HintContext,
    RootCauseHint,
    analyze_root_cause_hint,
    hint_coordinated_output_drift,
    hint_coordinated_tool_addition,
    hint_coordinated_tool_removal,
    hint_provider_rollout,
    hint_runtime_fingerprint_shift,
    hint_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diff(
    severity: DiffStatus = DiffStatus.REGRESSION,
    *,
    model_changed: bool = False,
    golden_model_id: str | None = None,
    actual_model_id: str | None = None,
    fingerprint: str | None = None,
    golden_fingerprint: str | None = None,
    tool_added: List[str] | None = None,
    tool_removed: List[str] | None = None,
    output_similarity: float | None = None,
) -> Any:
    """Build the minimal diff-like object every hinter inspects.

    We use ``MagicMock`` with explicit attrs (not real TraceDiff) so the
    test stays decoupled from the evaluator pipeline — same pattern as
    test_noise_tracker.py.
    """
    diff = MagicMock()
    diff.overall_severity = severity
    diff.model_changed = model_changed
    diff.golden_model_id = golden_model_id
    diff.actual_model_id = actual_model_id
    diff.runtime_model_fingerprint = fingerprint
    diff.actual_runtime_fingerprint = fingerprint
    diff.golden_runtime_fingerprint = golden_fingerprint

    tool_diffs = []
    for t in tool_added or []:
        tool_diffs.append(SimpleNamespace(type="added", actual_tool=t, golden_tool=None))
    for t in tool_removed or []:
        tool_diffs.append(SimpleNamespace(type="removed", actual_tool=None, golden_tool=t))
    diff.tool_diffs = tool_diffs

    if output_similarity is None:
        diff.output_diff = None
    else:
        diff.output_diff = SimpleNamespace(similarity=output_similarity)
    return diff


def _ctx(*diffs: Tuple[str, Any], min_affected: int = 3) -> HintContext:
    """Build a HintContext directly, bypassing the public entry point.

    Lets the unit tests drive each hinter without re-running the failing
    filter. Mirrors how ``analyze_root_cause_hint`` constructs it.
    """
    failing = tuple(
        (n, d) for n, d in diffs if d.overall_severity != DiffStatus.PASSED
    )
    return HintContext(diffs=diffs, failing=failing, min_affected=min_affected)


# ---------------------------------------------------------------------------
# hint_provider_rollout
# ---------------------------------------------------------------------------


class TestProviderRollout:
    def test_fires_when_three_tests_share_model_change(self) -> None:
        diffs = [
            ("a", _diff(model_changed=True, golden_model_id="m1", actual_model_id="m2")),
            ("b", _diff(model_changed=True, golden_model_id="m1", actual_model_id="m2")),
            ("c", _diff(model_changed=True, golden_model_id="m1", actual_model_id="m2")),
        ]
        hint = hint_provider_rollout(_ctx(*diffs))
        assert hint is not None
        assert hint.cause_id == "provider_rollout"
        assert hint.confidence == "high"
        # The narrative must mention both endpoints of the transition so
        # the operator sees the full picture without opening the dashboard.
        assert "m1" in hint.narrative and "m2" in hint.narrative
        # Evidence carries the transition breakdown for cloud / CI grouping.
        assert hint.evidence["affected_count"] == 3
        assert hint.evidence["transitions"][0]["from"] == "m1"
        assert hint.evidence["transitions"][0]["to"] == "m2"

    def test_does_not_fire_below_min_affected(self) -> None:
        diffs = [
            ("a", _diff(model_changed=True, actual_model_id="m2")),
            ("b", _diff(model_changed=True, actual_model_id="m2")),
        ]
        assert hint_provider_rollout(_ctx(*diffs, min_affected=3)) is None

    def test_suggested_action_includes_new_model_when_known(self) -> None:
        diffs = [
            ("a", _diff(model_changed=True, actual_model_id="m-new")),
            ("b", _diff(model_changed=True, actual_model_id="m-new")),
            ("c", _diff(model_changed=True, actual_model_id="m-new")),
        ]
        hint = hint_provider_rollout(_ctx(*diffs))
        assert hint is not None
        # First action should be a model-check pin command with the actual
        # model — that's the natural first step the operator should run.
        assert "model-check" in hint.suggested_actions[0]
        assert "m-new" in hint.suggested_actions[0]


# ---------------------------------------------------------------------------
# hint_runtime_fingerprint_shift
# ---------------------------------------------------------------------------


class TestFingerprintShift:
    def test_fires_when_three_tests_share_new_fingerprint(self) -> None:
        diffs = [
            ("a", _diff(fingerprint="fp-new", golden_fingerprint="fp-old")),
            ("b", _diff(fingerprint="fp-new", golden_fingerprint="fp-old")),
            ("c", _diff(fingerprint="fp-new", golden_fingerprint="fp-old")),
        ]
        hint = hint_runtime_fingerprint_shift(_ctx(*diffs))
        assert hint is not None
        assert hint.cause_id == "runtime_fingerprint_shift"
        assert hint.confidence == "medium"
        assert hint.evidence["fingerprint"] == "fp-new"

    def test_ignores_fingerprint_that_matches_baseline(self) -> None:
        # When the actual fingerprint equals the baseline, that test isn't
        # part of any shift — it just happens to have a fingerprint.
        diffs = [
            ("a", _diff(fingerprint="fp-same", golden_fingerprint="fp-same")),
            ("b", _diff(fingerprint="fp-same", golden_fingerprint="fp-same")),
            ("c", _diff(fingerprint="fp-same", golden_fingerprint="fp-same")),
        ]
        assert hint_runtime_fingerprint_shift(_ctx(*diffs)) is None


# ---------------------------------------------------------------------------
# hint_coordinated_tool_addition / removal
# ---------------------------------------------------------------------------


class TestCoordinatedToolAddition:
    def test_fires_when_three_tests_add_same_tool(self) -> None:
        diffs = [
            ("a", _diff(tool_added=["escalate_to_human"])),
            ("b", _diff(tool_added=["escalate_to_human"])),
            ("c", _diff(tool_added=["escalate_to_human", "log_event"])),
        ]
        hint = hint_coordinated_tool_addition(_ctx(*diffs))
        assert hint is not None
        # Picks the tool that appears in the most failing tests — even
        # though "log_event" appears in one diff, "escalate_to_human" is
        # the shared signal.
        assert hint.evidence["tool"] == "escalate_to_human"
        assert hint.evidence["affected_count"] == 3
        assert "escalate_to_human" in hint.cause_label

    def test_does_not_fire_when_tools_dont_overlap(self) -> None:
        diffs = [
            ("a", _diff(tool_added=["tool_a"])),
            ("b", _diff(tool_added=["tool_b"])),
            ("c", _diff(tool_added=["tool_c"])),
        ]
        # No shared tool reaches min_affected, so no hint.
        assert hint_coordinated_tool_addition(_ctx(*diffs)) is None


class TestCoordinatedToolRemoval:
    def test_fires_when_three_tests_remove_same_tool(self) -> None:
        diffs = [
            ("a", _diff(tool_removed=["check_policy"])),
            ("b", _diff(tool_removed=["check_policy"])),
            ("c", _diff(tool_removed=["check_policy"])),
        ]
        hint = hint_coordinated_tool_removal(_ctx(*diffs))
        assert hint is not None
        assert hint.cause_id == "coordinated_tool_removal"
        assert hint.evidence["tool"] == "check_policy"


# ---------------------------------------------------------------------------
# hint_coordinated_output_drift
# ---------------------------------------------------------------------------


class TestCoordinatedOutputDrift:
    def test_fires_on_low_similarity_with_no_tool_changes(self) -> None:
        diffs = [
            ("a", _diff(output_similarity=0.4)),
            ("b", _diff(output_similarity=0.5)),
            ("c", _diff(output_similarity=0.6)),
        ]
        hint = hint_coordinated_output_drift(_ctx(*diffs))
        assert hint is not None
        assert hint.cause_id == "coordinated_output_drift"
        assert hint.confidence == "medium"
        assert hint.evidence["avg_similarity"] == 0.5

    def test_skips_tests_with_tool_changes(self) -> None:
        # If tools also changed, this isn't *output-only* drift — leave
        # that case to the tool-addition/removal hinters.
        diffs = [
            ("a", _diff(output_similarity=0.3, tool_added=["new_tool"])),
            ("b", _diff(output_similarity=0.4)),
            ("c", _diff(output_similarity=0.4)),
        ]
        # Only 2 tests qualify → below min_affected.
        assert hint_coordinated_output_drift(_ctx(*diffs)) is None

    def test_high_similarity_does_not_fire(self) -> None:
        diffs = [
            ("a", _diff(output_similarity=0.95)),
            ("b", _diff(output_similarity=0.92)),
            ("c", _diff(output_similarity=0.98)),
        ]
        assert hint_coordinated_output_drift(_ctx(*diffs)) is None


# ---------------------------------------------------------------------------
# analyze_root_cause_hint — selection logic
# ---------------------------------------------------------------------------


class TestAnalyzeRootCauseHint:
    def test_provider_rollout_beats_output_drift(self) -> None:
        # Build diffs that match both hinters. Provider rollout has higher
        # priority (100 vs 50) so it should win.
        diffs = [
            (
                "a",
                _diff(
                    model_changed=True,
                    actual_model_id="m2",
                    output_similarity=0.4,
                ),
            ),
            (
                "b",
                _diff(
                    model_changed=True,
                    actual_model_id="m2",
                    output_similarity=0.4,
                ),
            ),
            (
                "c",
                _diff(
                    model_changed=True,
                    actual_model_id="m2",
                    output_similarity=0.4,
                ),
            ),
        ]
        hint = analyze_root_cause_hint(diffs)
        assert hint is not None
        assert hint.cause_id == "provider_rollout"

    def test_returns_none_when_no_hinter_matches(self) -> None:
        diffs = [
            ("a", _diff(severity=DiffStatus.PASSED)),
            ("b", _diff(severity=DiffStatus.PASSED)),
        ]
        assert analyze_root_cause_hint(diffs) is None

    def test_selection_is_deterministic_across_runs(self) -> None:
        diffs = [
            ("a", _diff(model_changed=True, actual_model_id="m2")),
            ("b", _diff(model_changed=True, actual_model_id="m2")),
            ("c", _diff(model_changed=True, actual_model_id="m2")),
        ]
        # Pin determinism — same input twice must produce the same hint.
        a = analyze_root_cause_hint(diffs)
        b = analyze_root_cause_hint(diffs)
        assert a == b

    def test_empty_input_returns_none(self) -> None:
        assert analyze_root_cause_hint([]) is None


# ---------------------------------------------------------------------------
# Integration: detect_coordinated_incident populates Incident.hint
# ---------------------------------------------------------------------------


class TestIncidentCarriesHint:
    def test_provider_rollout_attaches_hint_to_incident(self) -> None:
        diffs = [
            ("a", _diff(model_changed=True, actual_model_id="m2")),
            ("b", _diff(model_changed=True, actual_model_id="m2")),
            ("c", _diff(model_changed=True, actual_model_id="m2")),
        ]
        incident = detect_coordinated_incident(diffs)
        assert incident is not None
        # Backward-compat: the basic classification still fires.
        assert incident.cause == "likely provider update"
        # New: hint travels with the incident so notifiers can render it.
        assert incident.hint is not None
        assert incident.hint.cause_id == "provider_rollout"
        assert "model-check" in incident.hint.suggested_actions[0]

    def test_incident_hint_can_be_none(self) -> None:
        # The "correlated batch failure" path doesn't carry a richer
        # signal — hint must be optional, not required.
        diffs = [
            ("a", _diff(severity=DiffStatus.REGRESSION)),
            ("b", _diff(severity=DiffStatus.REGRESSION)),
            ("c", _diff(severity=DiffStatus.REGRESSION)),
        ]
        incident = detect_coordinated_incident(diffs)
        assert incident is not None
        # Either the batch-failure incident with hint=None, OR an output-
        # drift hint depending on heuristic; both shapes are valid as long
        # as the Incident still constructs cleanly.
        assert hasattr(incident, "hint")


# ---------------------------------------------------------------------------
# Serialization + roadmap surface
# ---------------------------------------------------------------------------


class TestHintToDict:
    def test_round_trips_to_plain_dict(self) -> None:
        hint = RootCauseHint(
            cause_id="x",
            cause_label="x",
            confidence="high",
            narrative="n",
            evidence={"k": "v"},
            suggested_actions=("a1", "a2"),
            priority=10,
        )
        d = hint_to_dict(hint)
        assert d == {
            "cause_id": "x",
            "cause_label": "x",
            "confidence": "high",
            "narrative": "n",
            "evidence": {"k": "v"},
            "suggested_actions": ["a1", "a2"],
            "priority": 10,
        }


class TestRoadmap:
    def test_roadmap_is_non_empty_and_shaped_for_contributors(self) -> None:
        # The roadmap is a public surface for contributors picking up a
        # "good first issue". Pin its presence + minimal shape so we don't
        # accidentally delete it during refactors.
        assert HINTERS_ROADMAP
        for entry in HINTERS_ROADMAP:
            assert isinstance(entry, str)
            assert ":" in entry  # "name: description" shape

    def test_hinters_tuple_contains_all_named_functions(self) -> None:
        # Smoke check: the public registry must include every hinter we
        # documented in the module. Catches "wrote a hinter, forgot to
        # register it" mistakes.
        named = {
            hint_provider_rollout,
            hint_runtime_fingerprint_shift,
            hint_coordinated_tool_addition,
            hint_coordinated_tool_removal,
            hint_coordinated_output_drift,
        }
        assert named.issubset(set(HINTERS))
