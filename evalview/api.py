"""Public Python API for EvalView.

Use ``gate()`` as a programmatic regression gate — the same checks that
``evalview check`` runs in the terminal, but returning structured data instead
of printing to the console.

Quick start::

    from evalview.api import gate, DiffStatus

    result = gate(test_dir="tests/")

    if not result.passed:
        for d in result.diffs:
            print(f"{d.test_name}: {d.status.value}")

Async variant::

    result = await gate_async(test_dir="tests/")

The gate functions bypass the CLI layer entirely — no Click context, no Rich
console output, no ``sys.exit()``.  They call the same internal execution
pipeline that powers ``evalview check``.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Re-export key types so callers only need ``from evalview.api import ...``
from evalview.core.diff import DiffStatus, TraceDiff  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TestDiff:
    """Per-test diff result — a thin wrapper over TraceDiff with easy access."""

    test_name: str
    status: DiffStatus
    score_delta: float
    output_similarity: Optional[float]
    semantic_similarity: Optional[float]
    tool_changes: int
    model_changed: bool
    raw: "TraceDiff"

    @property
    def passed(self) -> bool:
        return self.status == DiffStatus.PASSED

    def summary(self) -> str:
        return self.raw.summary()


@dataclass
class GateSummary:
    """Aggregate counts across all tests."""

    total: int = 0
    unchanged: int = 0
    regressions: int = 0
    tools_changed: int = 0
    output_changed: int = 0
    execution_failures: int = 0
    requested_tests: int = 0


# Re-export from the shared module so existing callers keep working
from evalview.core.observability import ObservabilitySummary as ObservabilitySignals  # noqa: E402


@dataclass
class GateResult:
    """Structured result from a gate() call.

    Attributes:
        passed: True when all requested tests were compared and no hard
            failure or status in ``fail_on`` was observed.
        exit_code: 0 if passed, 1 if failed.  Matches the semantics of
            ``evalview check --fail-on``.
        status: INCOMPLETE if the suite could not be fully compared, otherwise
            the worst DiffStatus seen across all tests.
        summary: Aggregate pass/fail/change counts.
        diffs: Per-test diff objects with scores, tool diffs, and outputs.
        observability: Aggregate observability signals (anomalies, trust, coherence).
        raw_json: Full result dict for callers that need everything.
    """

    passed: bool
    exit_code: int
    status: DiffStatus
    summary: GateSummary
    diffs: List[TestDiff]
    observability: ObservabilitySignals = field(default_factory=ObservabilitySignals)
    raw_json: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Severity ranking (worst first)
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {
    DiffStatus.REGRESSION: 0,
    DiffStatus.CONTRACT_DRIFT: 1,
    DiffStatus.TOOLS_CHANGED: 2,
    DiffStatus.OUTPUT_CHANGED: 3,
    DiffStatus.PASSED: 4,
}


def _worst_status(diffs: List[Tuple[str, "TraceDiff"]]) -> DiffStatus:
    if not diffs:
        return DiffStatus.PASSED
    return min(
        (d.overall_severity for _, d in diffs),
        key=lambda s: _SEVERITY_RANK.get(s, 99),
    )


# ---------------------------------------------------------------------------
# Core gate implementation
# ---------------------------------------------------------------------------

def _build_gate_result(
    diffs: List[Tuple[str, "TraceDiff"]],
    total_tests: int,
    fail_on: Set[DiffStatus],
    results: Optional[List[Any]] = None,
    error: Optional[str] = None,
) -> GateResult:
    """Convert raw execution output into a GateResult."""
    from evalview.commands.shared import _analyze_check_diffs
    from evalview.core.model_runtime_detector import analyze_model_runtime_change

    analysis = _analyze_check_diffs(diffs)
    model_runtime = analyze_model_runtime_change(diffs)

    # Build per-test diffs
    test_diffs: List[TestDiff] = []
    for name, trace_diff in diffs:
        output_diff = trace_diff.output_diff
        test_diffs.append(TestDiff(
            test_name=name,
            status=trace_diff.overall_severity,
            score_delta=trace_diff.score_diff,
            output_similarity=(
                output_diff.similarity if output_diff else None
            ),
            semantic_similarity=(
                output_diff.semantic_similarity
                if output_diff and hasattr(output_diff, "semantic_similarity")
                else None
            ),
            tool_changes=len(trace_diff.tool_diffs),
            model_changed=getattr(trace_diff, "model_changed", False),
            raw=trace_diff,
        ))

    # Summary counts
    # execution_failures = tests that were submitted but didn't produce a diff
    # (adapter errors, timeouts, missing baselines)
    execution_failures = max(0, total_tests - len(diffs))
    if execution_failures and not error:
        error = f"Only {len(diffs)} of {total_tests} requested tests produced a comparison."
    if total_tests == 0 and not error:
        error = "No matching test cases found."
    if error:
        analysis["all_passed"] = False
    summary = GateSummary(
        total=len(diffs),
        unchanged=sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED),
        regressions=sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION),
        tools_changed=sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED),
        output_changed=sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED),
        execution_failures=execution_failures,
        requested_tests=total_tests,
    )

    worst = DiffStatus.INCOMPLETE if error else _worst_status(diffs)
    has_failure = bool(error) or any(
        d.overall_severity in fail_on or list(d.forbidden_tool_violations) for _, d in diffs
    )

    # Raw JSON for callers that want everything
    raw: Dict[str, Any] = {
        "status": worst.value,
        "passed": not has_failure,
        "summary": {
            "total": summary.total,
            "requested_tests": total_tests,
            "execution_failures": execution_failures,
            "unchanged": summary.unchanged,
            "regressions": summary.regressions,
            "tools_changed": summary.tools_changed,
            "output_changed": summary.output_changed,
        },
        "analysis": analysis,
        "model_runtime": model_runtime.model_dump(),
        "diffs": [
            {
                "test_name": name,
                "status": d.overall_severity.value,
                "forbidden_tool_violations": list(d.forbidden_tool_violations),
                "score_delta": d.score_diff,
                "tool_diffs_count": len(d.tool_diffs),
                "model_changed": getattr(d, "model_changed", False),
                "runtime_fingerprint_changed": getattr(d, "runtime_fingerprint_changed", False),
            }
            for name, d in diffs
        ],
    }
    if error:
        raw["error"] = error

    # Build observability signals from evaluation results
    from evalview.core.observability import extract_observability_summary
    obs = extract_observability_summary(results)

    return GateResult(
        passed=not has_failure,
        exit_code=1 if has_failure else 0,
        status=worst,
        summary=summary,
        diffs=test_diffs,
        observability=obs,
        raw_json=raw,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gate(
    test_dir: str = "tests",
    test_name: Optional[str] = None,
    fail_on: Optional[Set[DiffStatus]] = None,
    judge_model: Optional[str] = None,
    semantic_diff: Optional[bool] = None,
    timeout: float = 30.0,
    quick: bool = False,
) -> GateResult:
    """Run regression checks and return structured results.

    This is the synchronous entry point.  It calls the same execution pipeline
    as ``evalview check`` but returns a :class:`GateResult` instead of printing
    to the console or calling ``sys.exit()``.

    Args:
        test_dir: Path to directory containing YAML test cases.
        test_name: Run only this test (by name).  ``None`` = all tests.
        fail_on: Set of DiffStatus values that count as failure.
            Default: ``{DiffStatus.REGRESSION}``.
        judge_model: Explicit judge model (e.g. ``"gpt-5.4-mini"``).
            ``None`` = use config or env var.
        semantic_diff: Override the project's OpenAI embedding setting. When
            omitted, embeddings require an explicit opt-in in project config.
        timeout: Per-test timeout in seconds.
        quick: If True, skip LLM-as-judge evaluation.  Uses deterministic
            scoring only (tool diff + output comparison).  Free and fast —
            ideal for tight autonomous loops.

    Returns:
        :class:`GateResult` with ``passed``, ``diffs``, ``summary``, etc.

    Example::

        from evalview.api import gate, DiffStatus

        # Full evaluation (default)
        result = gate(test_dir="tests/")

        # Quick mode — no LLM judge, sub-second, $0
        result = gate(test_dir="tests/", quick=True)
    """
    if fail_on is None:
        fail_on = {DiffStatus.REGRESSION}

    return asyncio.run(_gate_impl(
        test_dir=test_dir,
        test_name=test_name,
        fail_on=fail_on,
        judge_model=judge_model,
        semantic_diff=semantic_diff,
        timeout=timeout,
        quick=quick,
    ))


async def gate_async(
    test_dir: str = "tests",
    test_name: Optional[str] = None,
    fail_on: Optional[Set[DiffStatus]] = None,
    judge_model: Optional[str] = None,
    semantic_diff: Optional[bool] = None,
    timeout: float = 30.0,
    quick: bool = False,
) -> GateResult:
    """Async variant of :func:`gate`.

    Use this when you're already inside an async event loop (e.g. inside an
    agent framework or async test runner).

    Args:
        Same as :func:`gate`.

    Returns:
        :class:`GateResult`.
    """
    if fail_on is None:
        fail_on = {DiffStatus.REGRESSION}

    return await _gate_impl(
        test_dir=test_dir,
        test_name=test_name,
        fail_on=fail_on,
        judge_model=judge_model,
        semantic_diff=semantic_diff,
        timeout=timeout,
        quick=quick,
    )


async def _gate_impl(
    test_dir: str,
    test_name: Optional[str],
    fail_on: Set[DiffStatus],
    judge_model: Optional[str],
    semantic_diff: Optional[bool],
    timeout: float,
    quick: bool = False,
) -> GateResult:
    """Shared async implementation for gate() and gate_async()."""
    from evalview.core.loader import TestCaseLoader
    from evalview.commands.shared import (
        _load_config_if_exists,
        apply_judge_model,
    )
    from evalview.core.config import apply_judge_config

    # Load test cases
    test_path = Path(test_dir)
    if not test_path.is_dir():
        return _build_gate_result([], 0, fail_on, error=f"Test directory not found: {test_dir}")

    try:
        test_cases = TestCaseLoader.load_from_directory(test_path)
    except Exception as exc:
        return _build_gate_result([], 0, fail_on, error=f"Failed to load test cases: {exc}")

    if test_name:
        test_cases = [tc for tc in test_cases if tc.name == test_name]

    if not test_cases:
        return _build_gate_result([], 0, fail_on, error="No matching test cases found")

    try:
        config = _load_config_if_exists()
        if not quick:
            if judge_model:
                apply_judge_model(judge_model, interactive=False)
            elif config:
                apply_judge_config(config)
    except Exception as exc:
        return _build_gate_result([], len(test_cases), fail_on, error=f"Invalid gate configuration: {exc}")

    # Execute tests — call the internal pipeline directly.
    # _execute_check_tests uses asyncio.run() internally, but we're already
    # in an async context. Run it in a thread to avoid nested event loop issues.
    import concurrent.futures

    from evalview.commands.shared import _execute_check_tests

    def _run_sync():
        return _execute_check_tests(
            test_cases=test_cases,
            config=config,
            json_output=True,  # suppresses console.print in error paths
            semantic_diff=False if quick else semantic_diff,
            timeout=timeout,
            skip_llm_judge=quick,
        )

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        diffs, results, drift_tracker, golden_traces = await loop.run_in_executor(
            pool, _run_sync
        )

    return _build_gate_result(diffs, len(test_cases), fail_on, results=results)
