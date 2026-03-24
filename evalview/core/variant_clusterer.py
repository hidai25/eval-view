"""Cluster agent execution paths and suggest golden variants."""
from __future__ import annotations

from typing import Dict, List

from evalview.core.types import EvaluationResult


class ExecutionCluster:
    """A cluster of similar execution paths."""

    def __init__(self, tool_sequence: List[str], representative: EvaluationResult):
        self.tool_sequence = tool_sequence
        self.representative = representative
        self.members: List[EvaluationResult] = [representative]
        self.frequency: int = 1

    @property
    def sequence_key(self) -> str:
        return " → ".join(self.tool_sequence) if self.tool_sequence else "(no tools)"

    @property
    def avg_score(self) -> float:
        if not self.members:
            return 0.0
        return sum(r.score for r in self.members) / len(self.members)

    @property
    def pass_rate(self) -> float:
        if not self.members:
            return 0.0
        return sum(1 for r in self.members if r.passed) / len(self.members)

    def add(self, result: EvaluationResult) -> None:
        self.members.append(result)
        self.frequency += 1


def _extract_tool_sequence(result: EvaluationResult) -> List[str]:
    """Extract the tool call sequence from a result."""
    tools: List[str] = []
    if result.trace and result.trace.steps:
        for step in result.trace.steps:
            name = getattr(step, "tool_name", None) or getattr(step, "step_name", None)
            if name:
                tools.append(str(name))
    return tools


def cluster_results(results: List[EvaluationResult]) -> List[ExecutionCluster]:
    """Cluster results by their tool call sequences.

    Returns clusters sorted by frequency (most common first).
    """
    clusters: Dict[str, ExecutionCluster] = {}

    for result in results:
        tools = _extract_tool_sequence(result)
        key = ",".join(tools) if tools else "__no_tools__"

        if key in clusters:
            clusters[key].add(result)
        else:
            clusters[key] = ExecutionCluster(tools, result)

    # Sort by frequency descending
    return sorted(clusters.values(), key=lambda c: c.frequency, reverse=True)


def suggest_variants(
    clusters: List[ExecutionCluster],
    min_frequency_pct: float = 0.15,
    max_variants: int = 5,
) -> List[ExecutionCluster]:
    """Select clusters that should become golden variants.

    Args:
        clusters: All execution clusters, sorted by frequency.
        min_frequency_pct: Minimum frequency as fraction of total runs (default 15%).
        max_variants: Maximum variants to suggest (golden limit is 5).

    Returns:
        List of clusters worth saving as variants.
    """
    if not clusters:
        return []

    total = sum(c.frequency for c in clusters)
    min_count = max(1, int(total * min_frequency_pct))

    # Filter: must appear enough times and have reasonable pass rate
    viable = [
        c for c in clusters
        if c.frequency >= min_count and c.pass_rate >= 0.5
    ]

    # Take up to max_variants, prioritizing by frequency
    return viable[:max_variants]


def format_cluster_summary(clusters: List[ExecutionCluster], total_runs: int) -> str:
    """Format cluster analysis for display."""
    lines: List[str] = []

    for i, cluster in enumerate(clusters, 1):
        pct = cluster.frequency / total_runs * 100
        pass_pct = cluster.pass_rate * 100
        lines.append(
            f"  {i}. {cluster.sequence_key}  "
            f"({cluster.frequency}/{total_runs} runs = {pct:.0f}%, "
            f"pass rate: {pass_pct:.0f}%, "
            f"avg score: {cluster.avg_score:.1f})"
        )

    return "\n".join(lines)
