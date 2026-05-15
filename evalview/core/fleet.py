"""Fleet rollup — aggregate many monitor history files into one view.

`evalview monitor --history monitor.jsonl` writes one record per cycle from
one agent instance. When you scale past one instance — a canary + prod, a
fleet of regional workers, a developer + CI both running monitors — the
single-history commands stop telling you the truth: a passing rollup
hides a regional failure, a failing one hides a single bad pod.

`evalview fleet` is the cross-instance synthesizer. Given N history JSONLs
(each one a monitor session), it produces:

- A fleet-level summary (total sessions, cycles, fleet pass rate, cost).
- A per-instance table sorted by pass rate.
- Anomaly callouts — instances whose pass rate deviates from fleet mean by
  more than `--anomaly-sigma` standard deviations.
- Per-test fleet impact — tests failing in ≥ `--test-impact-pct` of the
  instances are surfaced so you can tell *"this is everywhere"* from
  *"only the eu-west pod is unhappy."*

Pure analytics: no network, no LLM. Reads JSONL only.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ── Defaults ────────────────────────────────────────────────────────────────

# A pod whose pass rate is more than ~2σ off the fleet mean is anomalous.
# 2σ is the standard "stop the line, look at this" threshold — anything
# tighter and a normally-noisy fleet generates false anomalies; anything
# looser and real regional outages slip past.
DEFAULT_ANOMALY_SIGMA = 2.0

# A test that fails in ≥40% of instances is fleet-wide enough that fixing
# one pod won't help. Below that, it's regional or instance-specific.
DEFAULT_TEST_IMPACT_PCT = 0.4


# ── Data shapes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InstanceSummary:
    """Rollup of one history JSONL — one monitor session, one agent."""

    instance: str
    cycles: int
    total_tests_observed: int
    passed: int
    regressions: int
    tools_changed: int
    output_changed: int
    cost: float
    first_seen: Optional[str]
    last_seen: Optional[str]
    failing_tests: Tuple[str, ...]  # union of every failing test across cycles

    @property
    def pass_rate(self) -> float:
        if self.total_tests_observed == 0:
            return 1.0
        return self.passed / self.total_tests_observed


@dataclass(frozen=True)
class FleetReport:
    """Cross-instance synthesis ready for rendering or JSON output."""

    instances: Tuple[InstanceSummary, ...]
    fleet_pass_rate: float
    fleet_cost: float
    fleet_cycles: int
    fleet_regressions: int
    anomalies: Tuple["InstanceAnomaly", ...]
    fleet_wide_failures: Tuple["FleetWideFailure", ...]
    anomaly_sigma: float
    test_impact_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fleet_pass_rate": round(self.fleet_pass_rate, 4),
            "fleet_cost": round(self.fleet_cost, 6),
            "fleet_cycles": self.fleet_cycles,
            "fleet_regressions": self.fleet_regressions,
            "instance_count": len(self.instances),
            "instances": [
                {
                    "instance": s.instance,
                    "cycles": s.cycles,
                    "pass_rate": round(s.pass_rate, 4),
                    "regressions": s.regressions,
                    "tools_changed": s.tools_changed,
                    "output_changed": s.output_changed,
                    "cost": round(s.cost, 6),
                    "first_seen": s.first_seen,
                    "last_seen": s.last_seen,
                    "failing_tests": list(s.failing_tests),
                }
                for s in self.instances
            ],
            "anomalies": [
                {
                    "instance": a.instance,
                    "pass_rate": round(a.pass_rate, 4),
                    "fleet_mean": round(a.fleet_mean, 4),
                    "sigma_distance": round(a.sigma_distance, 2),
                    "direction": a.direction,
                }
                for a in self.anomalies
            ],
            "fleet_wide_failures": [
                {
                    "test_name": f.test_name,
                    "affected_instances": list(f.affected_instances),
                    "impact_pct": round(f.impact_pct, 3),
                }
                for f in self.fleet_wide_failures
            ],
            "thresholds": {
                "anomaly_sigma": self.anomaly_sigma,
                "test_impact_pct": self.test_impact_pct,
            },
        }


@dataclass(frozen=True)
class InstanceAnomaly:
    """An instance whose pass rate is unusually far from fleet mean."""

    instance: str
    pass_rate: float
    fleet_mean: float
    sigma_distance: float
    direction: str  # "below" | "above"


@dataclass(frozen=True)
class FleetWideFailure:
    """A test that's failing across a meaningful fraction of the fleet."""

    test_name: str
    affected_instances: Tuple[str, ...]
    impact_pct: float


# ── History loading ─────────────────────────────────────────────────────────


def load_history(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL history file written by ``evalview monitor --history``.

    Tolerant of missing files and malformed lines — same behavior as
    ``since_cmd._load_history``. We deliberately don't import that
    function to keep the fleet module standalone; the format contract is
    "JSONL with one record per cycle".
    """
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return entries


def _instance_name_from_path(path: Path) -> str:
    """Derive an instance label from the file name.

    ``monitor-eu-west.jsonl`` → ``eu-west``. Common prefixes / suffixes
    stripped so the rendered table doesn't repeat ``monitor-`` for every
    row. Falls back to the stem unchanged.
    """
    stem = path.stem
    for prefix in ("monitor-", "monitor_", "history-", "history_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    for suffix in ("-history", "_history", "-monitor", "_monitor"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or path.name


# ── Per-instance aggregation ────────────────────────────────────────────────


def summarize_instance(name: str, entries: Sequence[Dict[str, Any]]) -> InstanceSummary:
    """Roll up one history file into a single :class:`InstanceSummary`.

    Only consumes cycle-summary records (those carrying ``total_tests``);
    skips any other record shapes a future writer may add. This keeps the
    function forward-compatible with new record types.
    """
    cycles = 0
    total = 0
    passed = 0
    regressions = 0
    tools_changed = 0
    output_changed = 0
    cost = 0.0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    failing: set[str] = set()

    for e in entries:
        if "total_tests" not in e:
            continue
        cycles += 1
        total += int(e.get("total_tests", 0) or 0)
        passed += int(e.get("passed", 0) or 0)
        regressions += int(e.get("regressions", 0) or 0)
        tools_changed += int(e.get("tools_changed", 0) or 0)
        output_changed += int(e.get("output_changed", 0) or 0)
        cost += float(e.get("cost", 0.0) or 0.0)
        ts = e.get("timestamp")
        if ts:
            ts_str = str(ts)
            if first_seen is None or ts_str < first_seen:
                first_seen = ts_str
            if last_seen is None or ts_str > last_seen:
                last_seen = ts_str
        for name_failing in e.get("failing_tests") or []:
            if isinstance(name_failing, str) and name_failing:
                failing.add(name_failing)

    return InstanceSummary(
        instance=name,
        cycles=cycles,
        total_tests_observed=total,
        passed=passed,
        regressions=regressions,
        tools_changed=tools_changed,
        output_changed=output_changed,
        cost=cost,
        first_seen=first_seen,
        last_seen=last_seen,
        failing_tests=tuple(sorted(failing)),
    )


# ── Anomaly detection ───────────────────────────────────────────────────────


def detect_anomalies(
    instances: Sequence[InstanceSummary],
    sigma_threshold: float = DEFAULT_ANOMALY_SIGMA,
) -> List[InstanceAnomaly]:
    """Flag instances whose pass rate is far from the fleet mean.

    Uses a Z-score against fleet pass-rate distribution. We deliberately
    do *not* use the median + MAD because monitor pass rates tend to
    cluster tightly around 1.0; a Z-score on the mean is more responsive
    to the case we actually care about (one pod degrading) without
    drowning in outlier-robust math for a 3-element fleet.

    Returns an empty list when we have fewer than 3 instances — the math
    isn't meaningful below that, and "your one instance is anomalous
    against itself" would just be noise.
    """
    if len(instances) < 3:
        return []
    rates = [s.pass_rate for s in instances]
    mean = statistics.fmean(rates)
    if len(rates) < 2:
        return []
    stdev = statistics.pstdev(rates)
    if stdev == 0:
        return []

    anomalies: List[InstanceAnomaly] = []
    for s in instances:
        z = (s.pass_rate - mean) / stdev
        if abs(z) >= sigma_threshold:
            anomalies.append(
                InstanceAnomaly(
                    instance=s.instance,
                    pass_rate=s.pass_rate,
                    fleet_mean=mean,
                    sigma_distance=z,
                    direction="below" if z < 0 else "above",
                )
            )
    # Most-anomalous first.
    anomalies.sort(key=lambda a: abs(a.sigma_distance), reverse=True)
    return anomalies


def detect_fleet_wide_failures(
    instances: Sequence[InstanceSummary],
    impact_threshold: float = DEFAULT_TEST_IMPACT_PCT,
) -> List[FleetWideFailure]:
    """Find tests failing in ≥ ``impact_threshold`` of instances.

    The threshold is a fraction in ``[0.0, 1.0]``. A test failing on 3 of
    5 instances at threshold 0.4 → impact_pct=0.6, surfaced. A test
    failing on 1 of 5 at the same threshold → impact_pct=0.2, dropped.
    """
    if not instances:
        return []

    test_to_instances: Dict[str, List[str]] = {}
    for inst in instances:
        for test_name in inst.failing_tests:
            test_to_instances.setdefault(test_name, []).append(inst.instance)

    n = len(instances)
    out: List[FleetWideFailure] = []
    for test_name, affected in test_to_instances.items():
        impact_pct = len(affected) / n
        if impact_pct >= impact_threshold:
            out.append(
                FleetWideFailure(
                    test_name=test_name,
                    affected_instances=tuple(sorted(affected)),
                    impact_pct=impact_pct,
                )
            )
    out.sort(key=lambda f: (-f.impact_pct, f.test_name))
    return out


# ── Top-level builder ───────────────────────────────────────────────────────


def build_fleet_report(
    history_files: Sequence[Path],
    *,
    anomaly_sigma: float = DEFAULT_ANOMALY_SIGMA,
    test_impact_pct: float = DEFAULT_TEST_IMPACT_PCT,
) -> FleetReport:
    """Top-level one-shot: paths → fully synthesized :class:`FleetReport`.

    Empty inputs are not an error: a fresh fleet with no history yet
    produces a report with zero instances, pass rate 100% by convention
    (nothing has failed), and no anomalies. Callers render whatever's
    appropriate for "nothing to roll up yet".
    """
    instances: List[InstanceSummary] = []
    for path in history_files:
        entries = load_history(path)
        if not entries:
            continue
        instances.append(
            summarize_instance(_instance_name_from_path(path), entries)
        )
    instances.sort(key=lambda s: (s.pass_rate, s.instance))

    total_observed = sum(s.total_tests_observed for s in instances)
    total_passed = sum(s.passed for s in instances)
    fleet_pass_rate = (
        total_passed / total_observed if total_observed else 1.0
    )
    fleet_cost = sum(s.cost for s in instances)
    fleet_cycles = sum(s.cycles for s in instances)
    fleet_regressions = sum(s.regressions for s in instances)

    anomalies = detect_anomalies(instances, sigma_threshold=anomaly_sigma)
    fleet_wide = detect_fleet_wide_failures(
        instances, impact_threshold=test_impact_pct
    )

    return FleetReport(
        instances=tuple(instances),
        fleet_pass_rate=fleet_pass_rate,
        fleet_cost=fleet_cost,
        fleet_cycles=fleet_cycles,
        fleet_regressions=fleet_regressions,
        anomalies=tuple(anomalies),
        fleet_wide_failures=tuple(fleet_wide),
        anomaly_sigma=anomaly_sigma,
        test_impact_pct=test_impact_pct,
    )


# ── File discovery helpers (used by the CLI) ────────────────────────────────


def discover_history_files(
    paths: Iterable[str],
    directories: Iterable[str] = (),
) -> List[Path]:
    """Expand --history and --dir inputs into a deduplicated, sorted file list.

    Globs in ``paths`` expand via :py:meth:`Path.glob`. ``directories``
    are scanned for ``*.jsonl`` non-recursively (recursive would pick up
    unrelated logs; let users opt in by passing the subdir directly).
    Missing files are dropped silently — fleet is meant to run on noisy
    inputs without crashing.
    """
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if any(ch in raw for ch in "*?["):
            found.extend(sorted(Path().glob(raw)))
        elif p.is_file():
            found.append(p)
    for d in directories:
        dp = Path(d)
        if dp.is_dir():
            found.extend(sorted(dp.glob("*.jsonl")))

    # Dedup while preserving order.
    seen: set[Path] = set()
    out: List[Path] = []
    for p in found:
        resolved = p.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(p)
    return out


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Helper for callers that want a single 'how noisy is this fleet' number.

    σ / μ — invariant to scale, easy to put in a digest line. Returns
    0.0 for empty / zero-mean inputs. Not used by the report itself but
    documented here so notifier authors don't reinvent it.
    """
    if not values:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0 or math.isnan(mean):
        return 0.0
    return statistics.pstdev(values) / mean
