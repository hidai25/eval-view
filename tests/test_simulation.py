"""Tests for the simulation harness.

Covers:
- MockedToolExecutor matching rules (tool name, param subset, strict)
- Simulator single-run and multi-variant aggregation
- YAML round-trip of the ``mocks:`` section via TestCase
- CLI smoke test via click.testing.CliRunner
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest
import yaml
from click.testing import CliRunner

from evalview.adapters.base import AgentAdapter
from evalview.core.simulation import (
    MockedToolExecutor,
    Simulator,
    UnmatchedMockError,
)
from evalview.core.types import (
    ExecutionMetrics,
    ExecutionTrace,
    ExpectedBehavior,
    HttpMock,
    MockSpec,
    ResponseMock,
    StepMetrics,
    StepTrace,
    TestCase,
    TestInput,
    Thresholds,
    ToolMock,
)


class _FakeAdapter(AgentAdapter):
    """Deterministic adapter for unit tests.

    Calls every ``tool_name`` supplied in ``plan`` via the
    ``context["tool_executor"]`` hook the simulator installs, so the
    test can observe exactly which calls the simulator intercepted.
    """

    def __init__(self, plan: List[Dict[str, Any]]) -> None:
        self._plan = plan
        self.tool_executor = None
        self.last_interceptor: Optional[Simulator] = None

    @property
    def name(self) -> str:
        return "fake"

    def install_mock_interceptor(self, simulator: Simulator) -> None:
        self.last_interceptor = simulator

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        context = context or {}
        # Simulator patches self.tool_executor (Python-adapter convention).
        executor = self.tool_executor
        assert executor is not None, "simulator should install tool_executor"

        start = datetime(2026, 4, 20, 10, 0, 0)
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
        end = datetime(2026, 4, 20, 10, 0, 1)
        return ExecutionTrace(
            session_id="fake",
            start_time=start,
            end_time=end,
            steps=steps,
            final_output=" | ".join(outputs),
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=1.0),
        )


# ============================================================================
# MockedToolExecutor
# ============================================================================


class TestMockedToolExecutor:
    def _executor(self, spec, real=None):
        import random
        from evalview.core.simulation import _MockHitCounter
        counter = _MockHitCounter()
        return MockedToolExecutor(spec, real, counter, random.Random(0)), counter

    def test_matches_by_tool_name(self):
        spec = MockSpec(tool_mocks=[ToolMock(tool="search", returns=[1, 2])])
        ex, counter = self._executor(spec)
        assert ex("search", {}) == [1, 2]
        assert counter.hits[("tool", "search")] == 1

    def test_unmatched_falls_through_to_real(self):
        calls: List[str] = []

        def real(name, params):
            calls.append(name)
            return "real"

        spec = MockSpec(tool_mocks=[ToolMock(tool="search", returns="mock")])
        ex, _ = self._executor(spec, real=real)
        assert ex("edit", {}) == "real"
        assert calls == ["edit"]

    def test_unmatched_returns_none_without_executor(self):
        spec = MockSpec(tool_mocks=[ToolMock(tool="search", returns="x")])
        ex, _ = self._executor(spec, real=None)
        assert ex("unknown", {}) is None

    def test_strict_raises_on_unmatched(self):
        spec = MockSpec(strict=True, tool_mocks=[ToolMock(tool="search", returns="x")])
        ex, _ = self._executor(spec, real=None)
        with pytest.raises(UnmatchedMockError):
            ex("edit", {})

    def test_param_subset_match(self):
        spec = MockSpec(tool_mocks=[ToolMock(
            tool="search",
            match_params={"q": "paris"},
            returns="paris-result",
        )])
        ex, counter = self._executor(spec)
        assert ex("search", {"q": "paris", "lang": "en"}) == "paris-result"
        assert counter.hits[("tool", "search")] == 1

    def test_param_match_misses_on_wrong_value(self):
        real_calls: List[str] = []

        def real(name, params):
            real_calls.append(params.get("q"))
            return "real"

        spec = MockSpec(tool_mocks=[ToolMock(
            tool="search",
            match_params={"q": "paris"},
            returns="mock",
        )])
        ex, _ = self._executor(spec, real=real)
        assert ex("search", {"q": "london"}) == "real"

    def test_error_mock_raises(self):
        spec = MockSpec(tool_mocks=[ToolMock(
            tool="flaky", returns=None, error="upstream boom",
        )])
        ex, counter = self._executor(spec)
        with pytest.raises(RuntimeError, match="upstream boom"):
            ex("flaky", {})
        # Even though it raised, the mock counts as applied.
        assert counter.hits[("tool", "flaky")] == 1


# ============================================================================
# Simulator
# ============================================================================


def _case(mocks: Optional[MockSpec] = None) -> TestCase:
    return TestCase(
        name="sim-test",
        input=TestInput(query="what is the capital of france?"),
        expected=ExpectedBehavior(),
        thresholds=Thresholds(min_score=0),
        mocks=mocks,
    )


class TestSimulatorSingleRun:
    @pytest.mark.asyncio
    async def test_records_applied_mocks_and_path(self):
        adapter = _FakeAdapter(plan=[
            {"tool": "search", "params": {"q": "paris"}},
            {"tool": "summarize", "params": {}},
        ])
        spec = MockSpec(
            seed=42,
            tool_mocks=[
                ToolMock(tool="search", returns=["Paris"]),
                ToolMock(tool="summarize", returns="Paris is the capital."),
            ],
        )
        sim = Simulator(adapter, spec)
        tc = _case(spec)
        trace, result = await sim.run(tc)

        assert result.seed == 42
        assert {m.matcher for m in result.mocks_applied} == {"search", "summarize"}
        assert len(result.branches_explored) == 1
        assert result.branches_explored[0].decision_path == [
            "step-0:search",
            "step-1:summarize",
        ]
        # Adapter saw the installer call.
        assert adapter.last_interceptor is sim
        # Trace output contains the mocked results.
        assert "Paris" in trace.final_output

    @pytest.mark.asyncio
    async def test_seed_override_wins(self):
        adapter = _FakeAdapter(plan=[])
        spec = MockSpec(seed=1)
        _, result = await Simulator(adapter, spec).run(_case(spec), seed_override=99)
        assert result.seed == 99


class TestSimulatorVariants:
    @pytest.mark.asyncio
    async def test_variants_produce_independent_branches(self):
        adapter = _FakeAdapter(plan=[
            {"tool": "search", "params": {"q": "paris"}},
        ])
        spec = MockSpec(
            seed=0,
            tool_mocks=[ToolMock(tool="search", returns=["Paris"])],
        )
        sim = Simulator(adapter, spec)
        traces, result = await sim.run_variants(_case(spec), variants=3)

        assert len(traces) == 3
        assert [b.branch_id for b in result.branches_explored] == ["b0", "b1", "b2"]
        assert [o.variant_index for o in result.variant_outcomes] == [0, 1, 2]
        # search ran once per variant → combined count = 3
        search_hits = [m for m in result.mocks_applied if m.matcher == "search"]
        assert search_hits[0].count == 3

    @pytest.mark.asyncio
    async def test_variants_rejects_zero(self):
        adapter = _FakeAdapter(plan=[])
        sim = Simulator(adapter, MockSpec())
        with pytest.raises(ValueError):
            await sim.run_variants(_case(), variants=0)


# ============================================================================
# Response / HTTP mock matching (public helpers)
# ============================================================================


class TestResponseAndHttpMatch:
    def test_response_mock_substring(self):
        spec = MockSpec(response_mocks=[ResponseMock(match_prompt="summarize", returns="sum")])
        sim = Simulator(_FakeAdapter([]), spec)
        assert sim.response_mock_for("please summarize this").returns == "sum"
        assert sim.response_mock_for("unrelated") is None

    def test_response_mock_regex(self):
        spec = MockSpec(response_mocks=[
            ResponseMock(match_prompt=r"user:\s*\w+", regex=True, returns="hi"),
        ])
        sim = Simulator(_FakeAdapter([]), spec)
        assert sim.response_mock_for("user: alice").returns == "hi"
        assert sim.response_mock_for("userr: nope") is None

    def test_http_mock_method_filter(self):
        spec = MockSpec(http_mocks=[
            HttpMock(url_pattern="api.example.com", method="POST", status=503),
        ])
        sim = Simulator(_FakeAdapter([]), spec)
        assert sim.http_mock_for("https://api.example.com/x", method="POST").status == 503
        assert sim.http_mock_for("https://api.example.com/x", method="GET") is None


# ============================================================================
# YAML round-trip via TestCase loader
# ============================================================================


class TestYAMLIntegration:
    def test_test_case_accepts_mocks_from_yaml(self, tmp_path):
        yaml_doc = {
            "name": "flight-search-sim",
            "input": {"query": "find cheapest flight to Paris"},
            "expected": {"tools": ["search_flights"]},
            "thresholds": {"min_score": 50},
            "adapter": "http",
            "endpoint": "http://127.0.0.1:9999",
            "mocks": {
                "seed": 7,
                "strict": False,
                "tool_mocks": [
                    {
                        "tool": "search_flights",
                        "match_params": {"to": "Paris"},
                        "returns": [{"id": "FL123", "price": 299}],
                        "latency_ms": 10,
                    }
                ],
            },
        }
        f = tmp_path / "t.yaml"
        f.write_text(yaml.safe_dump(yaml_doc))

        from evalview.core.loader import TestCaseLoader
        tc = TestCaseLoader.load_from_file(f)
        assert tc.mocks is not None
        assert tc.mocks.seed == 7
        assert len(tc.mocks.tool_mocks) == 1
        assert tc.mocks.tool_mocks[0].match_params == {"to": "Paris"}
        assert tc.mocks.tool_mocks[0].returns == [{"id": "FL123", "price": 299}]

    def test_test_case_without_mocks_defaults_none(self, tmp_path):
        yaml_doc = {
            "name": "no-mocks",
            "input": {"query": "hi"},
            "expected": {},
            "thresholds": {"min_score": 0},
        }
        f = tmp_path / "nm.yaml"
        f.write_text(yaml.safe_dump(yaml_doc))
        from evalview.core.loader import TestCaseLoader
        tc = TestCaseLoader.load_from_file(f)
        assert tc.mocks is None


# ============================================================================
# CLI
# ============================================================================


class TestSimulateCLI:
    def test_help_renders(self):
        from evalview.commands.simulate_cmd import simulate
        runner = CliRunner()
        result = runner.invoke(simulate, ["--help"])
        assert result.exit_code == 0
        assert "mocks" in result.output.lower()
        assert "--variants" in result.output

    def test_rejects_zero_variants(self, tmp_path):
        from evalview.commands.simulate_cmd import simulate

        # Write a minimal valid test so we get past the loader.
        (tmp_path / "t.yaml").write_text(yaml.safe_dump({
            "name": "t",
            "input": {"query": "x"},
            "expected": {},
            "thresholds": {"min_score": 0},
        }))
        runner = CliRunner()
        result = runner.invoke(simulate, [str(tmp_path), "--variants", "0"])
        assert result.exit_code != 0
        assert "variants" in result.output.lower()
