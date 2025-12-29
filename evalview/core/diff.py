"""Diff engine for comparing execution traces against golden baselines.

The diff engine provides deterministic comparison that:
1. Compares tool sequences (order matters)
2. Compares outputs (semantic similarity)
3. Highlights specific differences for easy debugging
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
from difflib import SequenceMatcher, unified_diff
import logging

from evalview.core.types import ExecutionTrace, StepTrace
from evalview.core.golden import GoldenTrace

logger = logging.getLogger(__name__)


class DiffStatus(Enum):
    """Status/category of differences found."""

    STABLE = "stable"      # No significant differences - matches baseline
    CHANGED = "changed"    # Minor changes (tools changed but output similar)
    DRIFT = "drift"        # Output changed but score stable (behavioral drift)
    REGRESSION = "regression"  # Score dropped significantly - likely a bug


# Alias for backwards compatibility
DiffSeverity = DiffStatus


@dataclass
class ToolDiff:
    """Difference in tool usage."""

    type: str  # "added", "removed", "changed", "reordered"
    position: int
    golden_tool: Optional[str]
    actual_tool: Optional[str]
    severity: DiffSeverity
    message: str


@dataclass
class OutputDiff:
    """Difference in output."""

    similarity: float  # 0.0 to 1.0
    golden_preview: str
    actual_preview: str
    diff_lines: List[str]  # Unified diff lines
    severity: DiffSeverity


@dataclass
class TraceDiff:
    """Complete diff between golden and actual trace."""

    test_name: str
    has_differences: bool
    tool_diffs: List[ToolDiff]
    output_diff: Optional[OutputDiff]
    score_diff: float  # actual_score - golden_score
    latency_diff: float  # actual_latency - golden_latency (ms)
    overall_severity: DiffSeverity

    def summary(self) -> str:
        """Human-readable summary of differences."""
        if not self.has_differences:
            return "No significant differences"

        parts = []
        if self.tool_diffs:
            parts.append(f"{len(self.tool_diffs)} tool difference(s)")
        if self.output_diff and self.output_diff.similarity < 0.95:
            parts.append(f"output similarity: {self.output_diff.similarity:.0%}")
        if abs(self.score_diff) > 5:
            direction = "improved" if self.score_diff > 0 else "regressed"
            parts.append(f"score {direction} by {abs(self.score_diff):.1f}")

        return ", ".join(parts) if parts else "Minor differences"


class DiffEngine:
    """Engine for comparing traces against golden baselines."""

    def __init__(
        self,
        tool_similarity_threshold: float = 0.8,
        output_similarity_threshold: float = 0.9,
    ):
        """
        Initialize diff engine.

        Args:
            tool_similarity_threshold: Min similarity for tool sequences to be "same"
            output_similarity_threshold: Min similarity for outputs to be "same"
        """
        self.tool_threshold = tool_similarity_threshold
        self.output_threshold = output_similarity_threshold

    def compare(
        self,
        golden: GoldenTrace,
        actual: ExecutionTrace,
        actual_score: float = 0.0,
    ) -> TraceDiff:
        """
        Compare actual trace against golden baseline.

        Args:
            golden: The golden (expected) trace
            actual: The actual trace from test run
            actual_score: Score from the test run

        Returns:
            TraceDiff with all differences
        """
        # Compare tools
        actual_tools = [step.tool_name for step in actual.steps]
        tool_diffs = self._compare_tools(golden.tool_sequence, actual_tools)

        # Compare outputs
        output_diff = self._compare_outputs(
            golden.trace.final_output, actual.final_output
        )

        # Calculate score diff
        score_diff = actual_score - golden.metadata.score

        # Calculate latency diff
        latency_diff = actual.metrics.total_latency - golden.trace.metrics.total_latency

        # Determine overall status using proper terminology:
        # - REGRESSION: score dropped significantly (>5 points)
        # - DRIFT: output changed (>20%) but score is stable
        # - CHANGED: tools changed but output similar
        # - STABLE: matches baseline

        has_tool_changes = bool(tool_diffs)
        has_output_drift = output_diff and output_diff.similarity < 0.80
        has_minor_output_change = output_diff and output_diff.similarity < 0.95
        score_dropped = score_diff < -5

        has_differences = has_tool_changes or has_minor_output_change

        if score_dropped:
            # Score dropped significantly - this is a REGRESSION
            overall_severity = DiffStatus.REGRESSION
        elif has_output_drift:
            # Output changed significantly but score stable - DRIFT
            overall_severity = DiffStatus.DRIFT
        elif has_tool_changes and not has_minor_output_change:
            # Only tools changed, output similar - minor CHANGED
            overall_severity = DiffStatus.CHANGED
        elif has_minor_output_change:
            # Small output change - DRIFT (less severe)
            overall_severity = DiffStatus.DRIFT
        else:
            # No significant differences - STABLE
            overall_severity = DiffStatus.STABLE

        return TraceDiff(
            test_name=golden.metadata.test_name,
            has_differences=has_differences,
            tool_diffs=tool_diffs,
            output_diff=output_diff,
            score_diff=score_diff,
            latency_diff=latency_diff,
            overall_severity=overall_severity,
        )

    def _compare_tools(
        self, golden_tools: List[str], actual_tools: List[str]
    ) -> List[ToolDiff]:
        """Compare tool sequences and return differences."""
        diffs = []

        # Use SequenceMatcher to find the best alignment
        matcher = SequenceMatcher(None, golden_tools, actual_tools)

        for op, g_start, g_end, a_start, a_end in matcher.get_opcodes():
            if op == "equal":
                continue

            elif op == "replace":
                # Tools at same position are different
                for i, (g, a) in enumerate(
                    zip(golden_tools[g_start:g_end], actual_tools[a_start:a_end])
                ):
                    diffs.append(
                        ToolDiff(
                            type="changed",
                            position=g_start + i,
                            golden_tool=g,
                            actual_tool=a,
                            severity=DiffStatus.CHANGED,
                            message=f"Tool changed: '{g}' -> '{a}' at step {g_start + i + 1}",
                        )
                    )

            elif op == "delete":
                # Tools in golden but not in actual
                for i, g in enumerate(golden_tools[g_start:g_end]):
                    diffs.append(
                        ToolDiff(
                            type="removed",
                            position=g_start + i,
                            golden_tool=g,
                            actual_tool=None,
                            severity=DiffStatus.DRIFT,  # Missing tool is drift
                            message=f"Tool removed: '{g}' was at step {g_start + i + 1}",
                        )
                    )

            elif op == "insert":
                # Tools in actual but not in golden
                for i, a in enumerate(actual_tools[a_start:a_end]):
                    diffs.append(
                        ToolDiff(
                            type="added",
                            position=a_start + i,
                            golden_tool=None,
                            actual_tool=a,
                            severity=DiffStatus.STABLE,  # Added tools are often OK
                            message=f"Tool added: '{a}' at step {a_start + i + 1}",
                        )
                    )

        return diffs

    def _compare_outputs(
        self, golden_output: str, actual_output: str
    ) -> OutputDiff:
        """Compare outputs and return diff."""
        # Calculate similarity
        similarity = SequenceMatcher(None, golden_output, actual_output).ratio()

        # Generate unified diff for display
        golden_lines = golden_output.splitlines(keepends=True)
        actual_lines = actual_output.splitlines(keepends=True)
        diff_lines = list(
            unified_diff(
                golden_lines,
                actual_lines,
                fromfile="golden",
                tofile="actual",
                lineterm="",
            )
        )

        # Determine severity (used internally, overall status determined in compare())
        if similarity >= 0.95:
            severity = DiffStatus.STABLE
        elif similarity >= 0.8:
            severity = DiffStatus.DRIFT
        else:
            severity = DiffStatus.REGRESSION

        # Create preview (first 200 chars)
        golden_preview = golden_output[:200] + ("..." if len(golden_output) > 200 else "")
        actual_preview = actual_output[:200] + ("..." if len(actual_output) > 200 else "")

        return OutputDiff(
            similarity=similarity,
            golden_preview=golden_preview,
            actual_preview=actual_preview,
            diff_lines=diff_lines[:50],  # Limit diff output
            severity=severity,
        )


# Convenience function
def compare_to_golden(
    golden: GoldenTrace,
    actual: ExecutionTrace,
    actual_score: float = 0.0,
) -> TraceDiff:
    """Compare an actual trace against a golden baseline."""
    engine = DiffEngine()
    return engine.compare(golden, actual, actual_score)
