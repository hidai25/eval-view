"""Tests for record/replay cassettes (`evalview/core/cassette.py`).

Covers the three cases that matter for "is this actually hermetic?":

1. **Recording is transparent** — wrapping a real executor doesn't
   change what the agent sees, but every call lands in the cassette.
2. **Round-trip equivalence** — a recorded cassette, replayed, returns
   the same values without ever invoking the real executor.
3. **Strict semantics** — extra/missing calls behave correctly under
   both lenient and strict modes; per-tool sequential matching is
   robust to inter-tool ordering drift.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml
from click.testing import CliRunner

from evalview.adapters.base import AgentAdapter
from evalview.core.cassette import (
    CASSETTE_FORMAT_VERSION,
    CassetteError,
    CassetteMismatchError,
    RecordingToolExecutor,
    ReplayToolExecutor,
    cassette_path_for,
    load_cassette,
    new_cassette,
    save_cassette,
)
from evalview.core.simulation import Simulator
from evalview.core.types import (
    Cassette,
    ExecutionMetrics,
    ExecutionTrace,
    ExpectedBehavior,
    Interaction,
    MockSpec,
    StepMetrics,
    StepTrace,
    TestCase,
    TestInput,
    Thresholds,
    ToolMock,
)


class _ScriptedAdapter(AgentAdapter):
    """Adapter that walks a fixed plan, calling tool_executor for each step."""

    def __init__(self, plan: List[Dict[str, Any]]) -> None:
        self._plan = plan
        self.tool_executor = None

    @property
    def name(self) -> str:
        return "scripted"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        executor = self.tool_executor
        assert executor is not None
        steps: List[StepTrace] = []
        outputs: List[str] = []
        for i, call in enumerate(self._plan):
            result = executor(call["tool"], call["params"])
            outputs.append(str(result))
            steps.append(StepTrace(
                step_id=f"step-{i}",
                step_name=call["tool"],
                tool_name=call["tool"],
                parameters=call["params"],
                output=result,
                success=True,
                metrics=StepMetrics(latency=1.0, cost=0.0),
            ))
        return ExecutionTrace(
            session_id="t",
            start_time=datetime(2026, 5, 4, 10, 0, 0),
            end_time=datetime(2026, 5, 4, 10, 0, 1),
            steps=steps,
            final_output=" | ".join(outputs),
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=1.0),
        )


def _case(name: str = "t", mocks: Optional[MockSpec] = None) -> TestCase:
    return TestCase(
        name=name,
        input=TestInput(query="q"),
        expected=ExpectedBehavior(),
        thresholds=Thresholds(min_score=0),
        mocks=mocks,
    )


# ============================================================================
# RecordingToolExecutor — transparent capture
# ============================================================================


class TestRecordingToolExecutor:
    def test_records_each_call_with_result(self):
        def real(name, params):
            return {"echo": name, "p": params}

        rec = RecordingToolExecutor(real=real)
        assert rec("search", {"q": "paris"}) == {"echo": "search", "p": {"q": "paris"}}
        assert rec("lookup", {"id": 1}) == {"echo": "lookup", "p": {"id": 1}}

        assert [i.tool for i in rec.interactions] == ["search", "lookup"]
        assert rec.interactions[0].returns == {"echo": "search", "p": {"q": "paris"}}
        assert rec.interactions[0].error is None

    def test_records_errors_then_reraises(self):
        def real(name, params):
            raise ValueError("upstream boom")

        rec = RecordingToolExecutor(real=real)
        with pytest.raises(ValueError, match="upstream boom"):
            rec("flaky", {})
        assert len(rec.interactions) == 1
        assert rec.interactions[0].error == "ValueError: upstream boom"
        assert rec.interactions[0].returns is None

    def test_params_are_copied_not_aliased(self):
        """Mutating the params dict after recording must not alter the cassette."""
        def real(name, params):
            return None

        rec = RecordingToolExecutor(real=real)
        params = {"q": "v1"}
        rec("search", params)
        params["q"] = "v2"
        assert rec.interactions[0].params == {"q": "v1"}


# ============================================================================
# ReplayToolExecutor — hermetic playback
# ============================================================================


class TestReplayToolExecutor:
    def _cassette(self, *interactions: Interaction) -> Cassette:
        return Cassette(
            test_name="t",
            recorded_at="2026-05-04T00:00:00Z",
            interactions=list(interactions),
        )

    def test_serves_calls_in_recorded_order_per_tool(self):
        cas = self._cassette(
            Interaction(tool="search", params={}, returns="r1"),
            Interaction(tool="search", params={}, returns="r2"),
        )
        rep = ReplayToolExecutor(cas)
        assert rep("search", {}) == "r1"
        assert rep("search", {}) == "r2"

    def test_intertool_order_does_not_matter(self):
        """The agent may reorder lookup vs check_policy between runs.

        Per-tool sequential matching keeps replay deterministic anyway:
        each tool name has its own queue, so calls consume independently.
        """
        cas = self._cassette(
            Interaction(tool="lookup", params={}, returns="L1"),
            Interaction(tool="check_policy", params={}, returns="P1"),
            Interaction(tool="lookup", params={}, returns="L2"),
        )
        rep = ReplayToolExecutor(cas)
        assert rep("check_policy", {}) == "P1"
        assert rep("lookup", {}) == "L1"
        assert rep("lookup", {}) == "L2"

    def test_replays_recorded_errors(self):
        cas = self._cassette(
            Interaction(tool="flaky", error="RuntimeError: boom"),
        )
        rep = ReplayToolExecutor(cas)
        with pytest.raises(RuntimeError, match="RuntimeError: boom"):
            rep("flaky", {})

    def test_strict_raises_on_unmatched(self):
        cas = self._cassette(Interaction(tool="search", returns="r"))
        rep = ReplayToolExecutor(cas, strict=True)
        with pytest.raises(CassetteMismatchError):
            rep("unrecorded", {})

    def test_lenient_falls_through_to_real(self):
        cas = self._cassette(Interaction(tool="search", returns="r"))
        live: List[str] = []

        def real(name, params):
            live.append(name)
            return "live"

        rep = ReplayToolExecutor(cas, real=real, strict=False)
        assert rep("unrecorded", {}) == "live"
        assert live == ["unrecorded"]

    def test_lenient_returns_none_with_no_real(self):
        cas = self._cassette(Interaction(tool="search", returns="r"))
        rep = ReplayToolExecutor(cas, real=None, strict=False)
        assert rep("unrecorded", {}) is None

    def test_remaining_reports_unused(self):
        cas = self._cassette(
            Interaction(tool="search", returns="r1"),
            Interaction(tool="search", returns="r2"),
        )
        rep = ReplayToolExecutor(cas)
        rep("search", {})
        assert rep.remaining() == {"search": 1}


# ============================================================================
# On-disk format — versioning + round-trip
# ============================================================================


class TestCassettePersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        cas = new_cassette("my-test", adapter="anthropic")
        cas.interactions.append(Interaction(tool="search", params={"q": "x"}, returns=[1, 2, 3]))
        path = tmp_path / "c.json"
        save_cassette(cas, path)
        loaded = load_cassette(path)
        assert loaded.test_name == "my-test"
        assert loaded.adapter == "anthropic"
        assert loaded.version == CASSETTE_FORMAT_VERSION
        assert loaded.interactions[0].tool == "search"
        assert loaded.interactions[0].returns == [1, 2, 3]

    def test_load_rejects_future_version(self, tmp_path: Path):
        path = tmp_path / "c.json"
        path.write_text('{"version": 99, "test_name": "t", "recorded_at": "x", "interactions": []}')
        with pytest.raises(CassetteError, match="format v99"):
            load_cassette(path)

    def test_cassette_path_replaces_slashes(self):
        p = cassette_path_for("billing/refund")
        assert p.name == "billing__refund.json"


# ============================================================================
# End-to-end: record once, replay deterministically
# ============================================================================


class TestRoundTripWithSimulator:
    @pytest.mark.asyncio
    async def test_record_then_replay_yields_same_outputs(self, tmp_path: Path):
        # The "real" tool is a counter that returns a different value each
        # call. If replay actually serves from the cassette (and not from
        # the live counter), the second pass must produce identical output.
        counter = {"n": 0}

        def real(name, params):
            counter["n"] += 1
            return f"call-{counter['n']}-{name}"

        plan = [
            {"tool": "alpha", "params": {}},
            {"tool": "beta", "params": {}},
            {"tool": "alpha", "params": {}},
        ]

        # ── Record pass ──
        record_adapter = _ScriptedAdapter(plan)
        record_adapter.tool_executor = real
        sim = Simulator(record_adapter, MockSpec())
        rec_trace, rec_result = await sim.run(_case("rt"), record=True)

        assert rec_result.recorded_cassette is not None
        cassette = rec_result.recorded_cassette
        assert len(cassette.interactions) == 3
        assert cassette.adapter == "scripted"

        path = tmp_path / "rt.json"
        save_cassette(cassette, path)

        # ── Replay pass ── against an executor that would explode if used.
        def must_not_call(name, params):
            raise AssertionError(f"replay leaked to live executor: {name}")

        replay_adapter = _ScriptedAdapter(plan)
        replay_adapter.tool_executor = must_not_call
        sim2 = Simulator(replay_adapter, MockSpec(strict=True))
        loaded = load_cassette(path)
        rep_trace, _ = await sim2.run(_case("rt"), replay_cassette=loaded)

        # Replay outputs match what was recorded — not what `real` would
        # return now (counter has advanced).
        assert rep_trace.final_output == rec_trace.final_output

    @pytest.mark.asyncio
    async def test_recording_skipped_for_mocked_calls(self):
        """Synthetic mock results must not pollute the cassette."""
        live_calls: List[str] = []

        def real(name, params):
            live_calls.append(name)
            return f"live-{name}"

        plan = [
            {"tool": "mocked_tool", "params": {}},
            {"tool": "live_tool", "params": {}},
        ]
        adapter = _ScriptedAdapter(plan)
        adapter.tool_executor = real

        spec = MockSpec(tool_mocks=[ToolMock(tool="mocked_tool", returns="from-mock")])
        _, result = await Simulator(adapter, spec).run(_case("mix"), record=True)

        cas = result.recorded_cassette
        assert cas is not None
        # Only the unmocked tool reached the recorder.
        assert [i.tool for i in cas.interactions] == ["live_tool"]
        assert live_calls == ["live_tool"]


# ============================================================================
# CLI integration
# ============================================================================


class TestSimulateCassetteCLI:
    def test_record_and_replay_flags_mutually_exclusive(self, tmp_path: Path):
        from evalview.commands.simulate_cmd import simulate

        (tmp_path / "t.yaml").write_text(yaml.safe_dump({
            "name": "t",
            "input": {"query": "x"},
            "expected": {},
            "thresholds": {"min_score": 0},
        }))
        runner = CliRunner()
        result = runner.invoke(simulate, [str(tmp_path), "--record", "--replay"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_replay_without_cassette_errors(self, tmp_path: Path):
        from evalview.commands.simulate_cmd import simulate

        (tmp_path / "t.yaml").write_text(yaml.safe_dump({
            "name": "no-cassette",
            "adapter": "http",
            "endpoint": "http://127.0.0.1:9999",
            "input": {"query": "x"},
            "expected": {},
            "thresholds": {"min_score": 0},
        }))
        runner = CliRunner()
        result = runner.invoke(
            simulate,
            [str(tmp_path), "--replay", "--cassette-dir", str(tmp_path / "c")],
        )
        assert result.exit_code != 0
        assert "no cassette found" in result.output.lower()
