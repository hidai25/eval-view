"""Behavioral anomaly detection for agent execution traces.

Detects silent failure patterns that pass/fail scoring misses:

1. **Tool loops** — agent calls the same tool with the same parameters
   repeatedly, suggesting it's stuck in a retry loop.
2. **Progress stalls** — agent makes many tool calls but output doesn't
   change or no new tools are introduced, suggesting thrashing.
3. **Brittle recovery** — agent hits an error, then retries the exact same
   action instead of adapting its strategy.
4. **Skipped required steps** — agent reaches a plausible output but skips
   steps that a correct execution path requires.
5. **Excessive retries** — agent retries a failing action more times than
   is reasonable before giving up or succeeding.

These are the "looks-good-but-is-wrong" patterns practitioners complain
about most. Output may seem fine while the decision path is broken.

Usage:
    from evalview.core.behavioral_anomalies import detect_anomalies

    anomalies = detect_anomalies(trace)
    for a in anomalies:
        print(f"[{a.severity}] {a.pattern}: {a.description}")

All detection is deterministic — no LLM calls, no network I/O.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from evalview.core.types import ExecutionTrace, StepTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class AnomalyPattern(str, Enum):
    """Classification of detected behavioral anomalies."""

    TOOL_LOOP = "tool_loop"
    PROGRESS_STALL = "progress_stall"
    BRITTLE_RECOVERY = "brittle_recovery"
    SKIPPED_STEPS = "skipped_steps"
    EXCESSIVE_RETRIES = "excessive_retries"


class AnomalySeverity(str, Enum):
    """How concerning the anomaly is."""

    WARNING = "warning"   # Unusual but possibly intentional
    ERROR = "error"       # Almost certainly a problem


@dataclass
class Anomaly:
    """A single detected behavioral anomaly in an execution trace."""

    pattern: AnomalyPattern
    severity: AnomalySeverity
    description: str
    step_indices: List[int] = field(default_factory=list)
    tool_name: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "pattern": self.pattern.value,
            "severity": self.severity.value,
            "description": self.description,
            "step_indices": self.step_indices,
            "tool_name": self.tool_name,
            "evidence": self.evidence,
        }


@dataclass
class AnomalyReport:
    """Complete anomaly analysis for an execution trace."""

    anomalies: List[Anomaly] = field(default_factory=list)
    total_steps: int = 0
    unique_tools: int = 0
    error_count: int = 0

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0

    @property
    def error_anomalies(self) -> List[Anomaly]:
        return [a for a in self.anomalies if a.severity == AnomalySeverity.ERROR]

    @property
    def warning_anomalies(self) -> List[Anomaly]:
        return [a for a in self.anomalies if a.severity == AnomalySeverity.WARNING]

    def summary(self) -> str:
        """One-line summary for CLI output."""
        if not self.anomalies:
            return "No behavioral anomalies detected"
        errors = len(self.error_anomalies)
        warnings = len(self.warning_anomalies)
        parts = []
        if errors:
            parts.append(f"{errors} error(s)")
        if warnings:
            parts.append(f"{warnings} warning(s)")
        patterns = {a.pattern.value for a in self.anomalies}
        return f"Behavioral anomalies: {', '.join(parts)} — {', '.join(sorted(patterns))}"

    def to_dict(self) -> dict:
        return {
            "anomalies": [a.to_dict() for a in self.anomalies],
            "total_steps": self.total_steps,
            "unique_tools": self.unique_tools,
            "error_count": self.error_count,
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Detection thresholds (conservative defaults)
# ---------------------------------------------------------------------------

# Tool loop: same tool + same params called N+ times in a row
LOOP_MIN_CONSECUTIVE = 3

# Progress stall: N+ consecutive calls with no new tool introduced
STALL_WINDOW = 5

# Brittle recovery: error followed by identical retry
BRITTLE_MAX_IDENTICAL_RETRIES = 2

# Excessive retries: same tool called N+ times total (not necessarily consecutive)
EXCESSIVE_RETRY_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Fingerprinting helpers
# ---------------------------------------------------------------------------


def _step_fingerprint(step: StepTrace) -> str:
    """Create a deterministic fingerprint of a step's tool call.

    Two steps with the same fingerprint called the same tool with the
    same parameters — they are functionally identical calls.
    """
    # Sort dict keys for stability; truncate large values
    def _stable_repr(obj: Any, max_len: int = 200) -> str:
        if isinstance(obj, dict):
            items = sorted(obj.items())
            return "{" + ",".join(f"{k!r}:{_stable_repr(v, max_len)}" for k, v in items) + "}"
        if isinstance(obj, (list, tuple)):
            return "[" + ",".join(_stable_repr(v, max_len) for v in obj) + "]"
        s = repr(obj)
        if len(s) > max_len:
            return s[:max_len] + "..."
        return s

    return f"{step.tool_name}::{_stable_repr(step.parameters)}"


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def _detect_tool_loops(steps: List[StepTrace]) -> List[Anomaly]:
    """Detect consecutive identical tool calls (same tool + same params).

    This catches the classic "agent is stuck" pattern where it calls the
    same tool with the same arguments over and over, expecting a different
    result.
    """
    anomalies: List[Anomaly] = []
    if len(steps) < LOOP_MIN_CONSECUTIVE:
        return anomalies

    run_start = 0
    run_fp = _step_fingerprint(steps[0])

    for i in range(1, len(steps)):
        fp = _step_fingerprint(steps[i])
        if fp == run_fp:
            continue
        # End of a run — check if it was long enough
        run_length = i - run_start
        if run_length >= LOOP_MIN_CONSECUTIVE:
            anomalies.append(Anomaly(
                pattern=AnomalyPattern.TOOL_LOOP,
                severity=AnomalySeverity.ERROR,
                description=(
                    f"Tool '{steps[run_start].tool_name}' called {run_length} times "
                    f"consecutively with identical parameters (steps {run_start + 1}–{i}). "
                    f"Agent appears stuck in a loop."
                ),
                step_indices=list(range(run_start, i)),
                tool_name=steps[run_start].tool_name,
                evidence={
                    "consecutive_count": run_length,
                    "parameters": steps[run_start].parameters,
                },
            ))
        run_start = i
        run_fp = fp

    # Check the final run
    run_length = len(steps) - run_start
    if run_length >= LOOP_MIN_CONSECUTIVE:
        anomalies.append(Anomaly(
            pattern=AnomalyPattern.TOOL_LOOP,
            severity=AnomalySeverity.ERROR,
            description=(
                f"Tool '{steps[run_start].tool_name}' called {run_length} times "
                f"consecutively with identical parameters (steps {run_start + 1}–{len(steps)}). "
                f"Agent appears stuck in a loop."
            ),
            step_indices=list(range(run_start, len(steps))),
            tool_name=steps[run_start].tool_name,
            evidence={
                "consecutive_count": run_length,
                "parameters": steps[run_start].parameters,
            },
        ))

    return anomalies


def _detect_progress_stalls(steps: List[StepTrace]) -> List[Anomaly]:
    """Detect windows where the agent makes calls but introduces no new tools.

    If the agent calls 5+ tools and every single one is a tool it already
    used, it may be thrashing rather than making progress. This is different
    from a loop (same tool + same params) — here the agent is varying its
    calls but not expanding its strategy.
    """
    anomalies: List[Anomaly] = []
    if len(steps) < STALL_WINDOW:
        return anomalies

    seen_tools: Set[str] = set()
    stall_start: Optional[int] = None
    stall_length = 0

    for i, step in enumerate(steps):
        is_new = step.tool_name not in seen_tools
        seen_tools.add(step.tool_name)

        if is_new:
            # Progress! Check if we were in a stall
            if stall_length >= STALL_WINDOW and stall_start is not None:
                anomalies.append(Anomaly(
                    pattern=AnomalyPattern.PROGRESS_STALL,
                    severity=AnomalySeverity.WARNING,
                    description=(
                        f"{stall_length} consecutive tool calls (steps {stall_start + 1}–{i}) "
                        f"used only previously-seen tools. Agent may be thrashing without "
                        f"making progress."
                    ),
                    step_indices=list(range(stall_start, i)),
                    evidence={
                        "stall_length": stall_length,
                        "tools_in_stall": list({steps[j].tool_name for j in range(stall_start, i)}),
                    },
                ))
            stall_start = None
            stall_length = 0
        else:
            if stall_start is None:
                stall_start = i
                stall_length = 1
            else:
                stall_length += 1

    # Check final window
    if stall_length >= STALL_WINDOW and stall_start is not None:
        anomalies.append(Anomaly(
            pattern=AnomalyPattern.PROGRESS_STALL,
            severity=AnomalySeverity.WARNING,
            description=(
                f"{stall_length} consecutive tool calls (steps {stall_start + 1}–{len(steps)}) "
                f"used only previously-seen tools. Agent may be thrashing without "
                f"making progress."
            ),
            step_indices=list(range(stall_start, len(steps))),
            evidence={
                "stall_length": stall_length,
                "tools_in_stall": list({steps[j].tool_name for j in range(stall_start, len(steps))}),
            },
        ))

    return anomalies


def _detect_brittle_recovery(steps: List[StepTrace]) -> List[Anomaly]:
    """Detect when an agent hits an error and retries the exact same call.

    A healthy agent adapts after failure — changes parameters, tries a
    different tool, or asks for clarification. A brittle agent just retries
    the identical call, hoping for a different result.
    """
    anomalies: List[Anomaly] = []
    if len(steps) < 2:
        return anomalies

    i = 0
    while i < len(steps):
        step = steps[i]
        if not step.success and step.error:
            # Found a failed step — look ahead for identical retries
            fp = _step_fingerprint(step)
            retry_count = 0
            j = i + 1
            while j < len(steps):
                if _step_fingerprint(steps[j]) == fp:
                    retry_count += 1
                    j += 1
                else:
                    break

            if retry_count >= BRITTLE_MAX_IDENTICAL_RETRIES:
                anomalies.append(Anomaly(
                    pattern=AnomalyPattern.BRITTLE_RECOVERY,
                    severity=AnomalySeverity.ERROR,
                    description=(
                        f"Tool '{step.tool_name}' failed at step {i + 1} "
                        f"(error: {step.error[:100]}), then was retried {retry_count} "
                        f"time(s) with identical parameters. Agent did not adapt its "
                        f"strategy after failure."
                    ),
                    step_indices=list(range(i, j)),
                    tool_name=step.tool_name,
                    evidence={
                        "error": step.error[:200],
                        "retry_count": retry_count,
                        "parameters": step.parameters,
                    },
                ))
            i = j
        else:
            i += 1

    return anomalies


def _detect_excessive_retries(steps: List[StepTrace]) -> List[Anomaly]:
    """Detect tools called an unusually high number of times.

    Even if not consecutive (which _detect_tool_loops catches), calling
    the same tool 5+ times in a single execution is suspicious and may
    indicate the agent is struggling with a task.
    """
    anomalies: List[Anomaly] = []
    tool_counts: Dict[str, List[int]] = {}

    for i, step in enumerate(steps):
        tool_counts.setdefault(step.tool_name, []).append(i)

    for tool_name, indices in tool_counts.items():
        if len(indices) >= EXCESSIVE_RETRY_THRESHOLD:
            # Check if many of these were failures
            failure_count = sum(
                1 for idx in indices if not steps[idx].success
            )
            severity = (
                AnomalySeverity.ERROR if failure_count > len(indices) // 2
                else AnomalySeverity.WARNING
            )
            anomalies.append(Anomaly(
                pattern=AnomalyPattern.EXCESSIVE_RETRIES,
                severity=severity,
                description=(
                    f"Tool '{tool_name}' called {len(indices)} times "
                    f"({failure_count} failures). This is unusually high and may "
                    f"indicate the agent is struggling with this tool."
                ),
                step_indices=indices,
                tool_name=tool_name,
                evidence={
                    "total_calls": len(indices),
                    "failure_count": failure_count,
                    "success_count": len(indices) - failure_count,
                },
            ))

    return anomalies


def _detect_skipped_steps(
    steps: List[StepTrace],
    required_tools: Optional[List[str]] = None,
) -> List[Anomaly]:
    """Detect when required tools were never called despite task completion.

    This catches the case where an agent produces a plausible output but
    skips critical steps (e.g., a booking agent that responds with a
    confirmation without actually calling the booking API).

    Only fires when `required_tools` is provided — the caller must know
    which tools are required for the task.
    """
    if not required_tools:
        return []

    anomalies: List[Anomaly] = []
    actual_tools = {step.tool_name for step in steps}
    missing = [t for t in required_tools if t not in actual_tools]

    if missing:
        anomalies.append(Anomaly(
            pattern=AnomalyPattern.SKIPPED_STEPS,
            severity=AnomalySeverity.ERROR,
            description=(
                f"Required tool(s) never called: {', '.join(missing)}. "
                f"Agent may have produced a plausible output without actually "
                f"performing the required actions."
            ),
            tool_name=missing[0],
            evidence={
                "required_tools": required_tools,
                "actual_tools": sorted(actual_tools),
                "missing_tools": missing,
            },
        ))

    return anomalies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_anomalies(
    trace: ExecutionTrace,
    required_tools: Optional[List[str]] = None,
) -> AnomalyReport:
    """Run all behavioral anomaly detectors on an execution trace.

    Args:
        trace: The execution trace to analyze.
        required_tools: Optional list of tools that MUST be called for the
                       task to be considered truly complete. When provided,
                       enables skipped-step detection.

    Returns:
        AnomalyReport with all detected anomalies and summary stats.
    """
    steps = trace.steps
    all_anomalies: List[Anomaly] = []

    all_anomalies.extend(_detect_tool_loops(steps))
    all_anomalies.extend(_detect_progress_stalls(steps))
    all_anomalies.extend(_detect_brittle_recovery(steps))
    all_anomalies.extend(_detect_excessive_retries(steps))
    all_anomalies.extend(_detect_skipped_steps(steps, required_tools))

    return AnomalyReport(
        anomalies=all_anomalies,
        total_steps=len(steps),
        unique_tools=len({s.tool_name for s in steps}),
        error_count=sum(1 for s in steps if not s.success),
    )
