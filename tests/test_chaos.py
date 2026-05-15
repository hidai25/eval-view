"""Tests for `evalview.core.chaos`.

The module is the *plan* layer for chaos injection; integration into
`evalview simulate` is wired separately. These tests pin the plan
shape, the determinism contract (same seed → same scenario), and the
"one disruption per step" rule.
"""
from __future__ import annotations

import json

import pytest

from evalview.core.chaos import (
    CHAOS_MODES_ROADMAP,
    MODE_GOAL_INTERRUPTION,
    MODE_LATENCY_SPIKE,
    MODE_TOOL_FAILURE,
    SHIPPED_MODES,
    build_scenario,
    goal_interruption,
    latency_spike,
    random_scenario,
    tool_failure,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class TestBuilders:
    def test_tool_failure_default_payload(self) -> None:
        d = tool_failure(tool="lookup_order", on_call_index=2)
        assert d.mode == MODE_TOOL_FAILURE
        assert d.step_index == 2
        # Default payload mimics the shape most adapters already pass
        # through, so existing agent error-handling exercises naturally.
        assert d.params["error_payload"] == {"error": "simulated_failure"}

    def test_tool_failure_custom_payload(self) -> None:
        payload = {"code": 503, "message": "service unavailable"}
        d = tool_failure(tool="x", error_payload=payload)
        assert d.params["error_payload"] == payload

    def test_latency_spike(self) -> None:
        d = latency_spike(on_call_index=1, delay_ms=2500)
        assert d.mode == MODE_LATENCY_SPIKE
        assert d.params["delay_ms"] == 2500

    def test_goal_interruption(self) -> None:
        d = goal_interruption(after_step=4, new_message="Wait, change of plan.")
        assert d.mode == MODE_GOAL_INTERRUPTION
        assert d.step_index == 4
        assert d.params["new_message"] == "Wait, change of plan."


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------


class TestBuildScenario:
    def test_sorts_disruptions_by_step(self) -> None:
        scenario = build_scenario(
            seed=42,
            disruptions=[
                latency_spike(on_call_index=5, delay_ms=1000),
                tool_failure(tool="x", on_call_index=2),
            ],
        )
        assert [d.step_index for d in scenario.disruptions] == [2, 5]

    def test_two_disruptions_same_step_rejected(self) -> None:
        # Pin the "one disruption per step" rule — if you need multiple,
        # model them as separate steps in the scenario.
        with pytest.raises(ValueError):
            build_scenario(
                seed=1,
                disruptions=[
                    tool_failure(tool="a", on_call_index=3),
                    latency_spike(on_call_index=3),
                ],
            )

    def test_disruption_at_returns_correct_one(self) -> None:
        scenario = build_scenario(
            seed=1,
            disruptions=[
                tool_failure(tool="a", on_call_index=2),
                latency_spike(on_call_index=5),
            ],
        )
        assert scenario.disruption_at(2) is not None
        assert scenario.disruption_at(2).mode == MODE_TOOL_FAILURE
        assert scenario.disruption_at(99) is None

    def test_to_dict_round_trips_via_json(self) -> None:
        # Scenarios should serialize cleanly so they can be checked in
        # for regression testing or shared across machines.
        scenario = build_scenario(
            seed=7,
            disruptions=[tool_failure(tool="search", on_call_index=1)],
            description="canary",
        )
        payload = json.loads(json.dumps(scenario.to_dict()))
        assert payload["seed"] == 7
        assert payload["disruptions"][0]["mode"] == MODE_TOOL_FAILURE


# ---------------------------------------------------------------------------
# Determinism (the core contract)
# ---------------------------------------------------------------------------


class TestRandomScenarioDeterminism:
    def test_same_seed_same_scenario(self) -> None:
        # The whole point of seeding: a CI run with seed=42 today must
        # produce the same disruptions as one tomorrow.
        a = random_scenario(
            seed=42, available_tools=["a", "b", "c"],
            max_steps=10, n_disruptions=3,
        )
        b = random_scenario(
            seed=42, available_tools=["a", "b", "c"],
            max_steps=10, n_disruptions=3,
        )
        assert a.to_dict() == b.to_dict()

    def test_different_seed_different_scenario(self) -> None:
        # Useful for property-testing-style sweeps: many seeds, many
        # different chaos plans.
        a = random_scenario(
            seed=1, available_tools=["a", "b"], max_steps=10, n_disruptions=2,
        )
        b = random_scenario(
            seed=2, available_tools=["a", "b"], max_steps=10, n_disruptions=2,
        )
        assert a.to_dict() != b.to_dict()

    def test_disruptions_respect_max_steps(self) -> None:
        # Pinning the bound prevents off-by-one regressions when the
        # step-collision loop changes.
        scenario = random_scenario(
            seed=99, available_tools=["a"], max_steps=5, n_disruptions=3,
        )
        for d in scenario.disruptions:
            assert 0 <= d.step_index < 5


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_shipped_modes_include_all_constants(self) -> None:
        # Adding a new MODE_* constant means appending to SHIPPED_MODES;
        # this test keeps the two in lockstep so consumers iterating
        # SHIPPED_MODES never miss a mode.
        assert MODE_TOOL_FAILURE in SHIPPED_MODES
        assert MODE_LATENCY_SPIKE in SHIPPED_MODES
        assert MODE_GOAL_INTERRUPTION in SHIPPED_MODES

    def test_roadmap_is_non_empty_and_shaped_for_contributors(self) -> None:
        # Same contract as the root-cause hinter roadmap: keep the
        # contributor surface visible across refactors.
        assert CHAOS_MODES_ROADMAP
        for entry in CHAOS_MODES_ROADMAP:
            assert isinstance(entry, str)
            assert ":" in entry  # "name: description" shape


# ---------------------------------------------------------------------------
# Type-shape sanity
# ---------------------------------------------------------------------------


class TestChaosDisruptionFrozen:
    def test_disruption_is_immutable(self) -> None:
        # Frozen dataclasses prevent simulator handlers from mutating
        # the plan mid-run. Pin it so a future @dataclass swap doesn't
        # quietly drop the freeze.
        d = tool_failure(tool="x", on_call_index=0)
        with pytest.raises(Exception):
            d.step_index = 999  # type: ignore[misc]
