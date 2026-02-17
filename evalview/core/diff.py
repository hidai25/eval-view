"""Diff engine for comparing execution traces against golden baselines.

The diff engine provides deterministic comparison that:
1. Compares tool sequences (order matters)
2. Compares outputs (semantic similarity)
3. Highlights specific differences for easy debugging
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from difflib import SequenceMatcher, unified_diff
import logging

from evalview.core.types import ExecutionTrace, StepTrace
from evalview.core.golden import GoldenTrace
from evalview.core.config import DiffConfig

logger = logging.getLogger(__name__)


class DiffStatus(Enum):
    """Result of comparing current run against golden baseline.

    This is a DIFF STATUS (comparison result), not an overall test result.
    A test may have additional pass/fail criteria (cost limits, latency thresholds)
    beyond the diff status.

    Five states with clear developer-friendly terminology:
    - PASSED: Output and tools match within tolerance - safe to ship
    - TOOLS_CHANGED: Tool sequence differs - agent behavior shifted, review before deploy
    - OUTPUT_CHANGED: Same tools but output differs beyond threshold - review before deploy
    - REGRESSION: Score dropped significantly - likely a bug, fix before deploy
    - CONTRACT_DRIFT: External MCP server interface changed - fix integration before deploy
    """

    PASSED = "passed"                # Output and tools match within tolerance
    TOOLS_CHANGED = "tools_changed"  # Tool sequence differs from golden
    OUTPUT_CHANGED = "output_changed"  # Output differs beyond similarity threshold
    REGRESSION = "regression"        # Score dropped >5 points from golden
    CONTRACT_DRIFT = "contract_drift"  # External MCP server interface changed


# Alias for backwards compatibility
DiffSeverity = DiffStatus


@dataclass
class ParameterDiff:
    """Difference in a single tool parameter.

    Tracks changes in tool call arguments at the parameter level,
    enabling precise identification of what changed between runs.
    """

    param_name: str
    golden_value: Any
    actual_value: Any
    diff_type: str  # "value_changed", "type_changed", "missing", "added"
    similarity: Optional[float] = None  # For string values (0.0-1.0)


@dataclass
class ToolDiff:
    """Difference in tool usage."""

    type: str  # "added", "removed", "changed", "reordered"
    position: int
    golden_tool: Optional[str]
    actual_tool: Optional[str]
    severity: DiffSeverity
    message: str
    parameter_diffs: List[ParameterDiff] = field(default_factory=list)  # NEW: Detailed parameter comparison


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
    matched_variant: Optional[str] = None  # Which golden variant was matched (for multi-reference)

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
        config: Optional[DiffConfig] = None,
        tool_similarity_threshold: Optional[float] = None,
        output_similarity_threshold: Optional[float] = None,
    ):
        """
        Initialize diff engine.

        Args:
            config: Optional DiffConfig with threshold settings
            tool_similarity_threshold: Override for tool similarity (deprecated, use config)
            output_similarity_threshold: Override for output similarity (deprecated, use config)
        """
        # Use config if provided, otherwise use individual params or defaults
        if config is None:
            config = DiffConfig()

        # Allow individual params to override config (backward compatibility)
        self.tool_threshold = tool_similarity_threshold if tool_similarity_threshold is not None else config.tool_similarity_threshold
        self.output_threshold = output_similarity_threshold if output_similarity_threshold is not None else config.output_similarity_threshold
        self.score_regression_threshold = config.score_regression_threshold
        self.ignore_whitespace = config.ignore_whitespace
        self.ignore_case_in_output = config.ignore_case_in_output

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
        # Compare tools (pass both tool names and full steps for parameter comparison)
        actual_tools = [step.tool_name for step in actual.steps]
        tool_diffs = self._compare_tools(
            golden.tool_sequence,
            actual_tools,
            golden.trace.steps,
            actual.steps
        )

        # Compare outputs
        output_diff = self._compare_outputs(
            golden.trace.final_output, actual.final_output
        )

        # Calculate score diff
        score_diff = actual_score - golden.metadata.score

        # Calculate latency diff
        latency_diff = actual.metrics.total_latency - golden.trace.metrics.total_latency

        # Determine overall status:
        # - REGRESSION: score dropped significantly - fix before deploy
        # - TOOLS_CHANGED: different tools used - review before deploy
        # - OUTPUT_CHANGED: same tools, different response - review before deploy
        # - PASSED: matches baseline - safe to ship

        has_tool_changes = bool(tool_diffs)
        has_output_change = output_diff.similarity < 0.95
        has_significant_output_change = output_diff.similarity < 0.80
        score_dropped = score_diff < -self.score_regression_threshold

        has_differences = has_tool_changes or has_output_change

        if score_dropped:
            # Score dropped significantly - REGRESSION
            overall_severity = DiffStatus.REGRESSION
        elif has_tool_changes:
            # Tools changed - TOOLS_CHANGED (behavior shifted)
            overall_severity = DiffStatus.TOOLS_CHANGED
        elif has_output_change:
            # Output changed but same tools - OUTPUT_CHANGED
            overall_severity = DiffStatus.OUTPUT_CHANGED
        else:
            # No significant differences - PASSED
            overall_severity = DiffStatus.PASSED

        return TraceDiff(
            test_name=golden.metadata.test_name,
            has_differences=has_differences,
            tool_diffs=tool_diffs,
            output_diff=output_diff,
            score_diff=score_diff,
            latency_diff=latency_diff,
            overall_severity=overall_severity,
        )

    def compare_multi_reference(
        self,
        golden_variants: List[GoldenTrace],
        actual: ExecutionTrace,
        actual_score: float = 0.0,
    ) -> TraceDiff:
        """
        Compare actual trace against multiple golden variants and return best match.

        This enables non-deterministic agent behavior by accepting multiple valid
        execution paths. The actual trace is compared against all variants and
        the closest match is returned.

        Args:
            golden_variants: List of golden trace variants to compare against
            actual: The actual trace from test run
            actual_score: Score from the test run

        Returns:
            TraceDiff with best match, annotated with matched variant

        Raises:
            ValueError: If golden_variants is empty
        """
        if not golden_variants:
            raise ValueError("At least one golden variant required for comparison")

        # Severity ranking (best to worst)
        severity_rank = {
            DiffStatus.PASSED: 0,
            DiffStatus.OUTPUT_CHANGED: 1,
            DiffStatus.TOOLS_CHANGED: 2,
            DiffStatus.REGRESSION: 3,
            DiffStatus.CONTRACT_DRIFT: 4,
        }

        best_diff: Optional[TraceDiff] = None
        best_rank = float('inf')
        best_variant_name = "default"

        # Compare against each variant
        for i, golden in enumerate(golden_variants):
            diff = self.compare(golden, actual, actual_score)

            # Rank this diff
            rank = severity_rank.get(diff.overall_severity, 999)

            # If same severity, prefer lower score diff (closer to golden)
            if rank == best_rank and best_diff:
                if abs(diff.score_diff) < abs(best_diff.score_diff):
                    best_diff = diff
                    best_variant_name = f"variant_{i}" if i > 0 else "default"
            elif rank < best_rank:
                # This variant is a better match
                best_diff = diff
                best_rank = rank
                best_variant_name = f"variant_{i}" if i > 0 else "default"

        # Annotate with matched variant and return
        # best_diff is guaranteed to be set because golden_variants is non-empty
        if best_diff is None:
            # This should never happen due to the empty check above, but satisfy type checker
            raise RuntimeError("Failed to compare variants (this is a bug)")

        best_diff.matched_variant = best_variant_name
        return best_diff

    def _compare_tools(
        self,
        golden_tools: List[str],
        actual_tools: List[str],
        golden_steps: List[StepTrace],
        actual_steps: List[StepTrace]
    ) -> List[ToolDiff]:
        """
        Compare tool sequences and return differences.

        Args:
            golden_tools: List of tool names from golden trace
            actual_tools: List of tool names from actual trace
            golden_steps: Full step traces from golden (for parameter comparison)
            actual_steps: Full step traces from actual (for parameter comparison)

        Returns:
            List of tool differences with parameter-level details
        """
        diffs = []

        # Use SequenceMatcher to find the best alignment
        matcher = SequenceMatcher(None, golden_tools, actual_tools)

        for op, g_start, g_end, a_start, a_end in matcher.get_opcodes():
            if op == "equal":
                # Tools match by name, but check if parameters changed
                for i in range(g_end - g_start):
                    g_idx = g_start + i
                    a_idx = a_start + i
                    if g_idx < len(golden_steps) and a_idx < len(actual_steps):
                        param_diffs = self._compare_tool_parameters(
                            golden_steps[g_idx],
                            actual_steps[a_idx]
                        )
                        # If parameters changed, add a "changed" diff
                        if param_diffs:
                            diffs.append(
                                ToolDiff(
                                    type="changed",
                                    position=g_idx,
                                    golden_tool=golden_tools[g_idx],
                                    actual_tool=actual_tools[a_idx],
                                    severity=DiffStatus.TOOLS_CHANGED,
                                    message=f"Tool '{golden_tools[g_idx]}' parameters changed at step {g_idx + 1}",
                                    parameter_diffs=param_diffs
                                )
                            )

            elif op == "replace":
                # Tools at same position are different
                for i, (g, a) in enumerate(
                    zip(golden_tools[g_start:g_end], actual_tools[a_start:a_end])
                ):
                    g_idx = g_start + i
                    a_idx = a_start + i

                    # Check if tool names are the same (just reordered) vs actually different
                    param_diffs = []
                    if g == a and g_idx < len(golden_steps) and a_idx < len(actual_steps):
                        # Same tool name but might have different parameters
                        param_diffs = self._compare_tool_parameters(
                            golden_steps[g_idx],
                            actual_steps[a_idx]
                        )

                    diffs.append(
                        ToolDiff(
                            type="changed",
                            position=g_idx,
                            golden_tool=g,
                            actual_tool=a,
                            severity=DiffStatus.TOOLS_CHANGED,
                            message=f"Tool changed: '{g}' -> '{a}' at step {g_idx + 1}",
                            parameter_diffs=param_diffs
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
                            severity=DiffStatus.TOOLS_CHANGED,  # Missing tool = behavior shifted
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
                            severity=DiffStatus.TOOLS_CHANGED,  # Added tool = behavior shifted
                            message=f"Tool added: '{a}' at step {a_start + i + 1}",
                        )
                    )

        return diffs

    def _compare_tool_parameters(
        self,
        golden_step: StepTrace,
        actual_step: StepTrace
    ) -> List[ParameterDiff]:
        """
        Compare parameters between golden and actual tool calls.

        Args:
            golden_step: The golden step trace
            actual_step: The actual step trace

        Returns:
            List of parameter differences
        """
        diffs = []

        # Extract parameters from StepTrace.parameters field
        golden_params = golden_step.parameters if golden_step.parameters else {}
        actual_params = actual_step.parameters if actual_step.parameters else {}

        # Check for missing, added, and changed parameters
        all_keys = set(golden_params.keys()) | set(actual_params.keys())

        for key in sorted(all_keys):  # Sort for consistent output
            if key not in actual_params:
                # Parameter was in golden but missing in actual
                diffs.append(ParameterDiff(
                    param_name=key,
                    golden_value=golden_params[key],
                    actual_value=None,
                    diff_type="missing"
                ))
            elif key not in golden_params:
                # Parameter is new in actual (not in golden)
                diffs.append(ParameterDiff(
                    param_name=key,
                    golden_value=None,
                    actual_value=actual_params[key],
                    diff_type="added"
                ))
            else:
                golden_val = golden_params[key]
                actual_val = actual_params[key]

                # Type mismatch
                if type(golden_val) != type(actual_val):
                    diffs.append(ParameterDiff(
                        param_name=key,
                        golden_value=golden_val,
                        actual_value=actual_val,
                        diff_type="type_changed"
                    ))
                # Value changed
                elif golden_val != actual_val:
                    similarity = None
                    # Calculate string similarity if both are strings
                    if isinstance(golden_val, str) and isinstance(actual_val, str):
                        similarity = SequenceMatcher(None, golden_val, actual_val).ratio()

                    diffs.append(ParameterDiff(
                        param_name=key,
                        golden_value=golden_val,
                        actual_value=actual_val,
                        diff_type="value_changed",
                        similarity=similarity
                    ))

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
            severity = DiffStatus.PASSED
        elif similarity >= 0.8:
            severity = DiffStatus.OUTPUT_CHANGED
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
