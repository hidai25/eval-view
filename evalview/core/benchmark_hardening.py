"""Benchmark hardening and anti-gaming checks for agent evaluations.

Motivated by Berkeley RDI's April 2026 work showing that major agent
benchmarks (SWE-bench, WebArena, OSWorld, GAIA, Terminal-Bench, etc.)
could be exploited to get near-perfect scores without solving the tasks.

This module detects patterns that suggest an agent is gaming a benchmark
rather than genuinely solving it:

1. **Suspiciously fast completion** — task completed far below the
   expected latency floor, suggesting shortcut or pre-computed answer.
2. **Config/test leakage** — agent accessed test configuration files,
   ground truth, or evaluation scripts during execution.
3. **Too-perfect output** — output matches expected answer with 100%
   similarity across multiple tests, which is statistically unlikely
   for genuine multi-step reasoning.
4. **Score without work** — high score despite minimal or no tool calls,
   suggesting the agent produced a memorized answer without execution.
5. **Abnormal tool patterns** — agent accessed files or URLs that
   contain benchmark metadata, answer keys, or evaluation logic.

Usage:
    from evalview.core.benchmark_hardening import check_gaming

    flags = check_gaming(trace, test_case, score)
    for f in flags:
        print(f"[{f.severity}] {f.check}: {f.description}")

All checks are deterministic. Designed to be run as a post-evaluation
pass that annotates results with trust signals.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from evalview.core.types import ExecutionTrace, TestCase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class GamingCheck(str, Enum):
    """Classification of anti-gaming checks."""

    SUSPICIOUSLY_FAST = "suspiciously_fast"
    CONFIG_LEAKAGE = "config_leakage"
    TOO_PERFECT = "too_perfect"
    SCORE_WITHOUT_WORK = "score_without_work"
    ABNORMAL_FILE_ACCESS = "abnormal_file_access"


class FlagSeverity(str, Enum):
    """How suspicious the signal is."""

    INFO = "info"         # Mildly unusual, worth noting
    SUSPICIOUS = "suspicious"  # Likely gaming or a misconfigured test
    CRITICAL = "critical"      # Strong evidence of gaming


@dataclass
class GamingFlag:
    """A single anti-gaming flag raised during evaluation."""

    check: GamingCheck
    severity: FlagSeverity
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check.value,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence,
        }


@dataclass
class HardeningReport:
    """Complete anti-gaming analysis for an evaluation run."""

    flags: List[GamingFlag] = field(default_factory=list)
    trust_score: float = 1.0  # 0.0 = highly suspicious, 1.0 = fully trusted

    @property
    def has_flags(self) -> bool:
        return len(self.flags) > 0

    @property
    def critical_flags(self) -> List[GamingFlag]:
        return [f for f in self.flags if f.severity == FlagSeverity.CRITICAL]

    @property
    def suspicious_flags(self) -> List[GamingFlag]:
        return [f for f in self.flags if f.severity == FlagSeverity.SUSPICIOUS]

    def summary(self) -> str:
        if not self.flags:
            return f"No gaming signals detected (trust: {self.trust_score:.0%})"
        critical = len(self.critical_flags)
        suspicious = len(self.suspicious_flags)
        info = len(self.flags) - critical - suspicious
        parts = []
        if critical:
            parts.append(f"{critical} critical")
        if suspicious:
            parts.append(f"{suspicious} suspicious")
        if info:
            parts.append(f"{info} info")
        return (
            f"Gaming checks: {', '.join(parts)} "
            f"(trust: {self.trust_score:.0%})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flags": [f.to_dict() for f in self.flags],
            "trust_score": round(self.trust_score, 4),
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum expected latency (ms) for a genuine multi-step agent task.
# Tasks completing under this are suspicious. Conservative — even fast
# agents need time for LLM inference + tool execution.
MIN_EXPECTED_LATENCY_MS = 500.0

# Minimum number of tool calls expected for a non-trivial task.
# An agent producing a correct answer with 0 tool calls is suspicious.
#: Scoring ≥80 with fewer than this many tool calls is suspect. 2 catches
#: both "zero tools" (CRITICAL at 95+) and "single tool" shortcut answers
#: while leaving genuinely simple 2-step tasks alone.
MIN_EXPECTED_TOOL_CALLS = 2

# Score threshold above which we check for "too perfect" patterns
PERFECT_SCORE_THRESHOLD = 99.0

# Patterns that suggest config/test leakage — file paths or tool
# parameters that reference evaluation infrastructure
LEAKAGE_PATTERNS: List[str] = [
    r"\.evalview/",
    r"evalview\.ya?ml",
    r"golden/",
    r"baseline/",
    r"ground[_-]?truth",
    r"answer[_-]?key",
    r"expected[_-]?output",
    r"test[_-]?config",
    r"\.eval/",
    r"eval[_-]?results",
    r"scoring[_-]?rubric",
    r"judge[_-]?prompt",
]

# File extensions that should never be accessed during genuine task execution
SUSPICIOUS_EXTENSIONS: Set[str] = {
    ".eval", ".golden", ".baseline", ".answer",
    ".rubric", ".scoring", ".judge",
}

# Compiled patterns for efficiency
_LEAKAGE_RE = [re.compile(p, re.IGNORECASE) for p in LEAKAGE_PATTERNS]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_suspiciously_fast(
    trace: ExecutionTrace,
    min_latency_ms: float = MIN_EXPECTED_LATENCY_MS,
) -> List[GamingFlag]:
    """Flag tasks that complete far below the expected latency floor."""
    flags: List[GamingFlag] = []
    total_latency = trace.metrics.total_latency

    if total_latency <= 0:
        # Latency of 0 means it wasn't measured — skip rather than false-flag
        return flags

    if total_latency < min_latency_ms and len(trace.steps) > 0:
        severity = (
            FlagSeverity.SUSPICIOUS if total_latency < min_latency_ms / 2
            else FlagSeverity.INFO
        )
        flags.append(GamingFlag(
            check=GamingCheck.SUSPICIOUSLY_FAST,
            severity=severity,
            description=(
                f"Task completed in {total_latency:.0f}ms with {len(trace.steps)} "
                f"tool calls. Expected minimum ~{min_latency_ms:.0f}ms for genuine "
                f"multi-step execution."
            ),
            evidence={
                "total_latency_ms": total_latency,
                "step_count": len(trace.steps),
                "threshold_ms": min_latency_ms,
            },
        ))

    return flags


def _check_config_leakage(trace: ExecutionTrace) -> List[GamingFlag]:
    """Flag when the agent accesses test/eval configuration during execution.

    An agent that reads .evalview/, golden baselines, or answer keys is
    not solving the task — it's reading the answer.
    """
    flags: List[GamingFlag] = []
    leaked_paths: List[str] = []

    for step in trace.steps:
        # Check tool parameters for leakage patterns
        params_str = str(step.parameters)
        for pattern in _LEAKAGE_RE:
            match = pattern.search(params_str)
            if match:
                leaked_paths.append(f"step {step.step_id}: {match.group()} in {step.tool_name}")
                break

        # Check output for leakage (agent read eval files and got content back)
        output_str = str(step.output) if step.output else ""
        if len(output_str) > 0:
            for pattern in _LEAKAGE_RE:
                match = pattern.search(output_str)
                if match:
                    leaked_paths.append(
                        f"step {step.step_id}: {match.group()} in output of {step.tool_name}"
                    )
                    break

    if leaked_paths:
        severity = (
            FlagSeverity.CRITICAL if len(leaked_paths) >= 2
            else FlagSeverity.SUSPICIOUS
        )
        flags.append(GamingFlag(
            check=GamingCheck.CONFIG_LEAKAGE,
            severity=severity,
            description=(
                f"Agent accessed {len(leaked_paths)} evaluation/config path(s) "
                f"during execution. This suggests the agent may be reading test "
                f"answers rather than solving the task."
            ),
            evidence={
                "leaked_paths": leaked_paths[:10],
            },
        ))

    return flags


def _check_score_without_work(
    trace: ExecutionTrace,
    score: float,
    min_tools: int = MIN_EXPECTED_TOOL_CALLS,
) -> List[GamingFlag]:
    """Flag high scores achieved with minimal or no tool usage.

    An agent that scores 95+ with zero tool calls probably produced a
    memorized answer rather than executing the task.
    """
    flags: List[GamingFlag] = []

    if score >= 80.0 and len(trace.steps) < min_tools:
        severity = (
            FlagSeverity.CRITICAL if score >= 95.0 and len(trace.steps) == 0
            else FlagSeverity.SUSPICIOUS if score >= 90.0
            else FlagSeverity.INFO
        )
        flags.append(GamingFlag(
            check=GamingCheck.SCORE_WITHOUT_WORK,
            severity=severity,
            description=(
                f"Score of {score:.1f} achieved with only {len(trace.steps)} "
                f"tool call(s). High scores with minimal tool usage suggest a "
                f"memorized or shortcut answer."
            ),
            evidence={
                "score": score,
                "tool_call_count": len(trace.steps),
                "min_expected_tools": min_tools,
            },
        ))

    return flags


def _check_too_perfect(
    trace: ExecutionTrace,
    score: float,
    perfect_threshold: float = PERFECT_SCORE_THRESHOLD,
) -> List[GamingFlag]:
    """Flag suspiciously perfect scores.

    A score of 100.0 on a genuinely difficult multi-step task is
    statistically rare. Combined with other signals (fast completion,
    few tools), it's a strong gaming indicator.
    """
    flags: List[GamingFlag] = []

    if score >= perfect_threshold:
        # Perfect score + fast + few tools = highly suspicious
        # Treat latency <= 0 as unmeasured (don't false-flag)
        latency = trace.metrics.total_latency
        is_fast = latency > 0 and latency < MIN_EXPECTED_LATENCY_MS * 2
        is_light = len(trace.steps) <= 2

        if is_fast and is_light:
            severity = FlagSeverity.CRITICAL
        elif is_fast or is_light:
            severity = FlagSeverity.SUSPICIOUS
        else:
            severity = FlagSeverity.INFO

        flags.append(GamingFlag(
            check=GamingCheck.TOO_PERFECT,
            severity=severity,
            description=(
                f"Near-perfect score ({score:.1f}) achieved"
                f"{' in very low latency' if is_fast else ''}"
                f"{' with minimal tool calls' if is_light else ''}. "
                f"Verify this is genuine task completion."
            ),
            evidence={
                "score": score,
                "latency_ms": trace.metrics.total_latency,
                "tool_count": len(trace.steps),
                "is_fast": is_fast,
                "is_light": is_light,
            },
        ))

    return flags


# Pre-compiled pattern: match file extensions at word boundary or end-of-string.
# Requires the extension to be followed by a non-alphanumeric char or end-of-string
# to avoid false positives like ".evalview" matching ".eval".
_SUSPICIOUS_EXT_RE = re.compile(
    r"(?:"
    + "|".join(re.escape(ext) for ext in SUSPICIOUS_EXTENSIONS)
    + r")(?=[^a-zA-Z0-9]|$)",
    re.IGNORECASE,
)


def _check_abnormal_file_access(trace: ExecutionTrace) -> List[GamingFlag]:
    """Flag access to files with suspicious extensions."""
    flags: List[GamingFlag] = []
    suspicious_accesses: List[str] = []

    for step in trace.steps:
        params_str = str(step.parameters)
        for match in _SUSPICIOUS_EXT_RE.finditer(params_str):
            suspicious_accesses.append(
                f"{step.tool_name} accessed *{match.group()} file"
            )

    if suspicious_accesses:
        flags.append(GamingFlag(
            check=GamingCheck.ABNORMAL_FILE_ACCESS,
            severity=FlagSeverity.SUSPICIOUS,
            description=(
                f"Agent accessed {len(suspicious_accesses)} file(s) with "
                f"evaluation-related extensions during execution."
            ),
            evidence={
                "accesses": suspicious_accesses[:10],
            },
        ))

    return flags


# ---------------------------------------------------------------------------
# Trust computation
# ---------------------------------------------------------------------------

# Trust penalties per severity level
TRUST_PENALTY_CRITICAL = 0.3
TRUST_PENALTY_SUSPICIOUS = 0.15
TRUST_PENALTY_INFO = 0.05


def _compute_trust_score(flags: List[GamingFlag]) -> float:
    """Compute a trust score from gaming flags.

    Starts at 1.0 (fully trusted), reduced by each flag's severity.
    Clamped to [0.0, 1.0].
    """
    trust = 1.0
    for f in flags:
        if f.severity == FlagSeverity.CRITICAL:
            trust -= TRUST_PENALTY_CRITICAL
        elif f.severity == FlagSeverity.SUSPICIOUS:
            trust -= TRUST_PENALTY_SUSPICIOUS
        elif f.severity == FlagSeverity.INFO:
            trust -= TRUST_PENALTY_INFO
    return round(max(0.0, min(1.0, trust)), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_gaming(
    trace: ExecutionTrace,
    score: float,
    min_latency_ms: float = MIN_EXPECTED_LATENCY_MS,
    min_tools: int = MIN_EXPECTED_TOOL_CALLS,
    # Deprecated — accepted for backwards compatibility but unused.
    test_case: Optional[TestCase] = None,
) -> HardeningReport:
    """Run all anti-gaming checks on an evaluation result.

    Args:
        trace: The execution trace to analyze.
        score: The evaluation score (0-100).
        min_latency_ms: Minimum expected latency for genuine execution.
        min_tools: Minimum expected tool calls.

    Returns:
        HardeningReport with all flags and a trust score.
    """
    all_flags: List[GamingFlag] = []

    all_flags.extend(_check_suspiciously_fast(trace, min_latency_ms))
    all_flags.extend(_check_config_leakage(trace))
    all_flags.extend(_check_score_without_work(trace, score, min_tools))
    all_flags.extend(_check_too_perfect(trace, score))
    all_flags.extend(_check_abnormal_file_access(trace))

    return HardeningReport(
        flags=all_flags,
        trust_score=_compute_trust_score(all_flags),
    )


# Coefficient of variation below which latencies are "suspiciously uniform"
BATCH_TIMING_CV_THRESHOLD = 0.05  # 5% — real tasks have natural variance


def check_gaming_batch(
    results: List[Dict[str, Any]],
) -> HardeningReport:
    """Run anti-gaming checks across a batch of results.

    Detects batch-level patterns that individual checks miss:
    - All tests scoring near-perfect (statistically improbable)
    - High rate (80%+) of near-perfect scores
    - All tests completing in near-identical time (suggests automation gaming)

    Args:
        results: List of dicts with ``score`` (float) and optionally
                 ``latency_ms`` (float) for timing-similarity detection.

    Returns:
        HardeningReport with batch-level flags.
    """
    flags: List[GamingFlag] = []

    if not results:
        return HardeningReport(flags=flags)

    scores = [r.get("score", 0) for r in results]
    perfect_count = sum(1 for s in scores if s >= PERFECT_SCORE_THRESHOLD)

    # --- Score distribution checks ---

    # Batch-level: suspiciously high perfect rate
    if len(results) >= 3 and perfect_count == len(results):
        flags.append(GamingFlag(
            check=GamingCheck.TOO_PERFECT,
            severity=FlagSeverity.CRITICAL,
            description=(
                f"All {len(results)} tests scored near-perfect "
                f"({PERFECT_SCORE_THRESHOLD}+). This is statistically "
                f"improbable for genuine multi-step agent tasks and suggests "
                f"benchmark gaming or test contamination."
            ),
            evidence={
                "total_tests": len(results),
                "perfect_count": perfect_count,
                "scores": scores,
            },
        ))
    elif len(results) >= 5 and perfect_count / len(results) >= 0.8:
        flags.append(GamingFlag(
            check=GamingCheck.TOO_PERFECT,
            severity=FlagSeverity.SUSPICIOUS,
            description=(
                f"{perfect_count}/{len(results)} tests scored near-perfect. "
                f"Verify these results reflect genuine task completion."
            ),
            evidence={
                "total_tests": len(results),
                "perfect_count": perfect_count,
                "perfect_rate": perfect_count / len(results),
            },
        ))

    # --- Timing similarity check ---
    # If all tests complete in near-identical time, it suggests pre-computed
    # answers or automation gaming.  Real tasks have natural latency variance.
    latencies = [
        r["latency_ms"] for r in results
        if isinstance(r.get("latency_ms"), (int, float)) and r["latency_ms"] > 0
    ]
    if len(latencies) >= 3:
        mean_lat = sum(latencies) / len(latencies)
        if mean_lat > 0:
            variance = sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)
            std_dev = variance ** 0.5
            cv = std_dev / mean_lat  # coefficient of variation

            if cv < BATCH_TIMING_CV_THRESHOLD:
                flags.append(GamingFlag(
                    check=GamingCheck.SUSPICIOUSLY_FAST,
                    severity=FlagSeverity.SUSPICIOUS,
                    description=(
                        f"All {len(latencies)} tests completed in near-identical time "
                        f"(CV={cv:.1%}, mean={mean_lat:.0f}ms). Real multi-step tasks "
                        f"exhibit natural latency variance."
                    ),
                    evidence={
                        "latency_count": len(latencies),
                        "mean_ms": round(mean_lat, 1),
                        "std_dev_ms": round(std_dev, 1),
                        "cv": round(cv, 4),
                        "threshold_cv": BATCH_TIMING_CV_THRESHOLD,
                    },
                ))

    return HardeningReport(flags=flags, trust_score=_compute_trust_score(flags))
