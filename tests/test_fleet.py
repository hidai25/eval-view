"""Tests for `evalview fleet` — cross-instance monitor history rollup.

Two layers:

1. ``evalview.core.fleet`` — pure rollup math (per-instance summarize,
   anomaly detection, fleet-wide failure detection).
2. ``evalview.commands.fleet_cmd`` — CLI wiring via ``CliRunner``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from click.testing import CliRunner

from evalview.commands.fleet_cmd import fleet_cmd
from evalview.core.fleet import (
    DEFAULT_ANOMALY_SIGMA,
    InstanceSummary,
    build_fleet_report,
    coefficient_of_variation,
    detect_anomalies,
    detect_fleet_wide_failures,
    discover_history_files,
    load_history,
    summarize_instance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_history(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _cycle_record(
    *,
    cycle: int = 1,
    total_tests: int = 10,
    passed: int = 10,
    regressions: int = 0,
    tools_changed: int = 0,
    output_changed: int = 0,
    cost: float = 0.01,
    timestamp: str = "2026-04-14T12:00:00Z",
    failing_tests: List[str] | None = None,
) -> Dict[str, Any]:
    """One cycle-summary record matching the schema monitor_cmd writes."""
    return {
        "timestamp": timestamp,
        "cycle": cycle,
        "total_tests": total_tests,
        "passed": passed,
        "regressions": regressions,
        "tools_changed": tools_changed,
        "output_changed": output_changed,
        "cost": cost,
        "failing_tests": failing_tests or [],
    }


# ---------------------------------------------------------------------------
# load_history & summarize_instance
# ---------------------------------------------------------------------------


class TestLoadHistory:
    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        # Mirrors the autopr / since_cmd contract: nothing to roll up is a
        # normal outcome, not an error.
        assert load_history(tmp_path / "absent.jsonl") == []

    def test_malformed_line_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "history.jsonl"
        path.write_text("{}\nnot json\n{\"cycle\": 1}\n", encoding="utf-8")
        records = load_history(path)
        assert len(records) == 2


class TestSummarizeInstance:
    def test_aggregates_cycles(self) -> None:
        entries = [
            _cycle_record(cycle=1, total_tests=10, passed=10, cost=0.05),
            _cycle_record(cycle=2, total_tests=10, passed=8, regressions=1,
                          tools_changed=1, cost=0.07,
                          failing_tests=["t-a", "t-b"]),
        ]
        s = summarize_instance("pod-a", entries)
        assert s.instance == "pod-a"
        assert s.cycles == 2
        assert s.total_tests_observed == 20
        assert s.passed == 18
        assert s.regressions == 1
        assert abs(s.cost - 0.12) < 1e-9  # float-sum tolerance
        # Failing-test set is the union across cycles.
        assert set(s.failing_tests) == {"t-a", "t-b"}

    def test_ignores_records_without_total_tests(self) -> None:
        # Future-compat: unknown record shapes are skipped silently so a
        # writer adding new event types doesn't crash old rollups.
        entries = [
            _cycle_record(cycle=1),
            {"timestamp": "x", "test_name": "y", "status": "passed"},
        ]
        s = summarize_instance("pod-a", entries)
        assert s.cycles == 1

    def test_empty_input_safe(self) -> None:
        s = summarize_instance("empty", [])
        assert s.cycles == 0
        assert s.pass_rate == 1.0  # nothing failed → vacuously clean


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def _summary(name: str, pass_rate: float, *, failing: List[str] | None = None) -> InstanceSummary:
    """Synthesize an InstanceSummary from a pass-rate target.

    Lets anomaly / fleet-wide tests work directly with rates without
    threading cycle counts through every fixture.
    """
    total = 100
    passed = int(round(total * pass_rate))
    return InstanceSummary(
        instance=name,
        cycles=10,
        total_tests_observed=total,
        passed=passed,
        regressions=total - passed,
        tools_changed=0,
        output_changed=0,
        cost=0.0,
        first_seen=None,
        last_seen=None,
        failing_tests=tuple(failing or []),
    )


class TestDetectAnomalies:
    def test_one_pod_far_below_fleet_mean_is_flagged(self) -> None:
        # Six pods near 100%, one at 30%. With n=7 the bad pod clears
        # the 2σ default by a wide margin without skating the boundary.
        instances = [
            _summary("a", 1.00),
            _summary("b", 0.99),
            _summary("c", 1.00),
            _summary("d", 0.98),
            _summary("f", 1.00),
            _summary("g", 0.99),
            _summary("e", 0.30),
        ]
        anomalies = detect_anomalies(instances, sigma_threshold=2.0)
        names = {a.instance for a in anomalies}
        assert "e" in names
        # Most-anomalous-first ordering is part of the contract.
        assert anomalies[0].instance == "e"
        assert anomalies[0].direction == "below"

    def test_tight_fleet_yields_no_anomalies(self) -> None:
        # All near 100% — std is small but mean is also high; no Z-score
        # crosses 2σ. The use-case here is "fleet is healthy, don't
        # invent reasons to page anyone".
        instances = [_summary(f"pod-{i}", 0.99) for i in range(5)]
        assert detect_anomalies(instances, sigma_threshold=2.0) == []

    def test_below_three_instances_returns_empty(self) -> None:
        # Stats on n<3 isn't meaningful; refuse to call anything an
        # anomaly until we have a real fleet.
        instances = [_summary("a", 1.00), _summary("b", 0.10)]
        assert detect_anomalies(instances) == []


class TestFleetWideFailures:
    def test_test_failing_in_majority_is_surfaced(self) -> None:
        instances = [
            _summary("a", 1.0, failing=["test-x"]),
            _summary("b", 1.0, failing=["test-x"]),
            _summary("c", 1.0, failing=["test-x"]),
            _summary("d", 1.0, failing=[]),
            _summary("e", 1.0, failing=[]),
        ]
        # 3/5 = 60% impact, above the 40% default threshold.
        results = detect_fleet_wide_failures(instances, impact_threshold=0.4)
        assert len(results) == 1
        assert results[0].test_name == "test-x"
        assert results[0].impact_pct == 0.6

    def test_test_failing_in_only_one_pod_is_not_surfaced(self) -> None:
        instances = [
            _summary("a", 1.0, failing=["test-x"]),
            _summary("b", 1.0, failing=[]),
            _summary("c", 1.0, failing=[]),
            _summary("d", 1.0, failing=[]),
            _summary("e", 1.0, failing=[]),
        ]
        # 1/5 = 20%, below threshold.
        assert detect_fleet_wide_failures(instances, impact_threshold=0.4) == []

    def test_sorted_by_impact_desc(self) -> None:
        instances = [
            _summary("a", 1.0, failing=["x", "y"]),
            _summary("b", 1.0, failing=["x"]),
            _summary("c", 1.0, failing=["x"]),
        ]
        # x: 3/3 = 100%, y: 1/3 = 33% → only x surfaces at 0.4 threshold.
        results = detect_fleet_wide_failures(instances, impact_threshold=0.3)
        assert [r.test_name for r in results] == ["x", "y"]


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


class TestBuildFleetReport:
    def test_end_to_end_with_anomaly_and_fleet_wide_failure(self, tmp_path: Path) -> None:
        # Five healthy pods + one bad pod with a fleet-wide test failure.
        # n=7 keeps the Z-score on the bad pod comfortably past 2σ.
        for name in ("monitor-a", "monitor-b", "monitor-c", "monitor-d", "monitor-e"):
            _write_history(
                tmp_path / f"{name}.jsonl",
                [
                    _cycle_record(cycle=i, total_tests=10, passed=10)
                    for i in range(1, 6)
                ],
            )
        # Bad pod: 30% pass rate + a recurring test. Picked low enough
        # that the Z-score against the rest-of-fleet mean is comfortably
        # past 2σ regardless of small variance shifts.
        _write_history(
            tmp_path / "monitor-bad.jsonl",
            [
                _cycle_record(
                    cycle=i,
                    total_tests=10,
                    passed=3,
                    regressions=7,
                    failing_tests=["fleet-wide-test"],
                )
                for i in range(1, 6)
            ],
        )
        # An additional pod that *also* fails the same test (so it's
        # fleet-wide rather than instance-local).
        _write_history(
            tmp_path / "monitor-degraded.jsonl",
            [
                _cycle_record(
                    cycle=i,
                    total_tests=10,
                    passed=9,
                    regressions=1,
                    failing_tests=["fleet-wide-test"],
                )
                for i in range(1, 6)
            ],
        )

        # Lower test_impact_pct so the 2/7 instances carrying the
        # recurring test still count as fleet-wide for this scenario.
        # (Real fleets typically have far more pods sharing a regression.)
        report = build_fleet_report(
            sorted(tmp_path.glob("*.jsonl")),
            test_impact_pct=0.25,
        )
        assert len(report.instances) == 7
        # The bad pod must show up as the worst pass-rate (sorted first).
        assert report.instances[0].instance == "bad"
        # Anomaly detector should flag it.
        assert any(a.instance == "bad" for a in report.anomalies)
        # Fleet-wide failure surfaces the recurring test.
        assert report.fleet_wide_failures
        assert report.fleet_wide_failures[0].test_name == "fleet-wide-test"

    def test_empty_history_files_yield_empty_report(self, tmp_path: Path) -> None:
        # Files exist but contain no parseable records — common at the
        # very start of a monitor session before the first cycle lands.
        for name in ("a", "b"):
            (tmp_path / f"{name}.jsonl").write_text("")
        report = build_fleet_report(list(tmp_path.glob("*.jsonl")))
        assert report.instances == ()
        # Vacuously clean — no instances means no failures.
        assert report.fleet_pass_rate == 1.0


class TestDiscoverHistoryFiles:
    def test_dedupes_overlapping_paths(self, tmp_path: Path) -> None:
        # The same file mentioned twice (once explicit, once via dir)
        # should only contribute one report row.
        f = tmp_path / "monitor-x.jsonl"
        _write_history(f, [_cycle_record()])
        found = discover_history_files([str(f)], [str(tmp_path)])
        # One file, even though it matches both --history and --dir.
        assert len(found) == 1

    def test_directory_scan_picks_up_jsonl(self, tmp_path: Path) -> None:
        for name in ("a", "b", "c"):
            _write_history(tmp_path / f"{name}.jsonl", [_cycle_record()])
        (tmp_path / "not-jsonl.txt").write_text("ignore me")
        found = discover_history_files([], [str(tmp_path)])
        assert len(found) == 3


class TestCoefficientOfVariation:
    def test_zero_for_empty_and_zero_mean(self) -> None:
        # Documented edge cases — no NaNs in our digest output.
        assert coefficient_of_variation([]) == 0.0
        assert coefficient_of_variation([0.0, 0.0, 0.0]) == 0.0

    def test_known_value(self) -> None:
        # σ = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2)/3) = sqrt(2/3) ≈ 0.8165
        # CV = 0.8165 / 2 ≈ 0.4082
        cv = coefficient_of_variation([1.0, 2.0, 3.0])
        assert abs(cv - 0.4082) < 0.001


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestFleetCommand:
    def test_runs_against_directory(self, tmp_path: Path) -> None:
        for name in ("monitor-a", "monitor-b", "monitor-c"):
            _write_history(
                tmp_path / f"{name}.jsonl",
                [_cycle_record(cycle=i, total_tests=10, passed=10) for i in range(1, 4)],
            )
        runner = CliRunner()
        result = runner.invoke(fleet_cmd, ["--dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Instances: 3" in result.output
        assert "Pass rate" in result.output

    def test_json_output_is_parseable(self, tmp_path: Path) -> None:
        _write_history(tmp_path / "monitor-a.jsonl", [_cycle_record()])
        runner = CliRunner()
        result = runner.invoke(fleet_cmd, ["--dir", str(tmp_path), "--json"])
        assert result.exit_code == 0, result.output
        # Skip past any first-run notice that prints to stdout.
        payload = json.loads(result.output[result.output.index("{") :])
        assert payload["instance_count"] == 1
        assert payload["thresholds"]["anomaly_sigma"] == DEFAULT_ANOMALY_SIGMA

    def test_require_clean_exits_nonzero_on_regression(self, tmp_path: Path) -> None:
        _write_history(
            tmp_path / "monitor-a.jsonl",
            [_cycle_record(regressions=1, failing_tests=["bad"])],
        )
        runner = CliRunner()
        result = runner.invoke(
            fleet_cmd, ["--dir", str(tmp_path), "--require-clean"]
        )
        # 1 regression → not clean → exit 1.
        assert result.exit_code == 1, result.output

    def test_require_clean_exits_zero_on_healthy_fleet(self, tmp_path: Path) -> None:
        _write_history(
            tmp_path / "monitor-a.jsonl",
            [_cycle_record(total_tests=5, passed=5)],
        )
        runner = CliRunner()
        result = runner.invoke(
            fleet_cmd, ["--dir", str(tmp_path), "--require-clean"]
        )
        assert result.exit_code == 0, result.output

    def test_no_input_files_does_not_crash(self) -> None:
        # Fresh project, no history yet — should print a friendly hint
        # rather than crashing or scaring the user with stack traces.
        runner = CliRunner()
        result = runner.invoke(fleet_cmd, [])
        assert result.exit_code == 0, result.output
        assert "No history files matched" in result.output

    def test_anomalies_only_skips_per_instance_table(self, tmp_path: Path) -> None:
        # All three pods identical → no anomalies → output stays brief.
        for name in ("a", "b", "c"):
            _write_history(
                tmp_path / f"monitor-{name}.jsonl",
                [_cycle_record(total_tests=10, passed=10)],
            )
        runner = CliRunner()
        result = runner.invoke(
            fleet_cmd, ["--dir", str(tmp_path), "--anomalies-only"]
        )
        assert result.exit_code == 0, result.output
        # No "Per-instance" table when --anomalies-only is set.
        assert "Per-instance" not in result.output
