"""Cross-turn coherence analysis for multi-turn agent evaluations.

Detects when an agent loses context, contradicts itself, or forgets
information from earlier turns in a multi-turn conversation.

The core problem: single-turn tests look fine, but agents fall apart
8–10 turns into a real conversation. This module catches the specific
failure modes that turn-level pass/fail scoring misses:

1. **Context amnesia** — agent stops referencing information that was
   established in earlier turns, suggesting it lost the context.
2. **Self-contradiction** — agent's output in a later turn contradicts
   what it said or did in an earlier turn.
3. **Tool regression** — agent used the right tool in turn N but
   switches to a worse tool for the same purpose in turn N+K.
4. **Strategy drift** — the overall approach shifts mid-conversation
   without a clear trigger from the user.

Usage:
    from evalview.core.turn_coherence import analyze_coherence

    report = analyze_coherence(trace)
    for issue in report.issues:
        print(f"Turn {issue.turn_index}: [{issue.category}] {issue.description}")

All checks are deterministic (string matching, tool comparison).
No LLM calls needed for the core analysis.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from evalview.core.types import ExecutionTrace, TurnTrace, StepTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class CoherenceCategory(str, Enum):
    """Classification of cross-turn coherence issues."""

    CONTEXT_AMNESIA = "context_amnesia"
    TOOL_REGRESSION = "tool_regression"
    STRATEGY_DRIFT = "strategy_drift"
    OUTPUT_CONTRADICTION = "output_contradiction"


class CoherenceSeverity(str, Enum):
    """How concerning the coherence issue is."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CoherenceIssue:
    """A single cross-turn coherence issue detected."""

    category: CoherenceCategory
    severity: CoherenceSeverity
    turn_index: int
    description: str
    reference_turn: Optional[int] = None  # Which earlier turn is relevant
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "turn_index": self.turn_index,
            "reference_turn": self.reference_turn,
            "description": self.description,
            "evidence": self.evidence,
        }


@dataclass
class CoherenceReport:
    """Complete cross-turn coherence analysis."""

    issues: List[CoherenceIssue] = field(default_factory=list)
    total_turns: int = 0
    coherence_score: float = 1.0  # 0.0 = totally incoherent, 1.0 = perfect

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def errors(self) -> List[CoherenceIssue]:
        return [i for i in self.issues if i.severity == CoherenceSeverity.ERROR]

    @property
    def warnings(self) -> List[CoherenceIssue]:
        return [i for i in self.issues if i.severity == CoherenceSeverity.WARNING]

    def summary(self) -> str:
        if not self.issues:
            return f"Multi-turn coherence: OK across {self.total_turns} turns"
        errors = len(self.errors)
        warnings = len(self.warnings)
        categories = {i.category.value for i in self.issues}
        parts = []
        if errors:
            parts.append(f"{errors} error(s)")
        if warnings:
            parts.append(f"{warnings} warning(s)")
        return (
            f"Coherence issues across {self.total_turns} turns: "
            f"{', '.join(parts)} — {', '.join(sorted(categories))} "
            f"(score: {self.coherence_score:.0%})"
        )

    def to_dict(self) -> dict:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "total_turns": self.total_turns,
            "coherence_score": round(self.coherence_score, 4),
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def _extract_key_entities(text: str) -> Set[str]:
    """Extract potential key entities from text for reference tracking.

    Simple heuristic: words that look like names, identifiers, or
    specific values (capitalized words, numbers, quoted strings).
    Not a full NER — just enough to detect obvious context loss.
    """
    import re

    entities: Set[str] = set()

    # Capitalized multi-word phrases (likely names, places, products)
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
        entities.add(match.group().lower())

    # Quoted strings (specific values the user mentioned)
    for match in re.finditer(r'["\']([^"\']{2,30})["\']', text):
        entities.add(match.group(1).lower())

    # Numbers with context (prices, IDs, dates)
    for match in re.finditer(r'\b\d{2,}\b', text):
        entities.add(match.group())

    return entities


def _detect_context_amnesia(
    turns: List[TurnTrace],
    steps: List[StepTrace],
) -> List[CoherenceIssue]:
    """Detect when the agent stops referencing earlier context.

    Tracks key entities introduced in early turns and checks whether
    the agent still references them when relevant in later turns.
    """
    issues: List[CoherenceIssue] = []
    if len(turns) < 3:
        return issues

    # Build cumulative entity set from queries (what the user said)
    query_entities_by_turn: Dict[int, Set[str]] = {}
    cumulative_entities: Set[str] = set()

    for turn in turns:
        turn_entities = _extract_key_entities(turn.query)
        query_entities_by_turn[turn.index] = turn_entities
        cumulative_entities |= turn_entities

    if not cumulative_entities:
        return issues

    # Check later turns' outputs for reference to earlier entities
    # Skip the first 2 turns (too early for amnesia)
    for turn in turns[2:]:
        if not turn.output:
            continue

        output_lower = turn.output.lower()

        # Entities from the first half of the conversation
        early_turn_cutoff = max(1, len(turns) // 2)
        early_entities: Set[str] = set()
        for t in turns[:early_turn_cutoff]:
            early_entities |= query_entities_by_turn.get(t.index, set())

        if not early_entities:
            continue

        # Check if any early entities are referenced in this turn's query
        # but NOT in the agent's response
        query_entities = _extract_key_entities(turn.query)
        referenced_early = query_entities & early_entities

        if referenced_early:
            # User is referencing something from earlier — does the agent too?
            missing_in_output = {
                e for e in referenced_early
                if e not in output_lower
            }
            if missing_in_output and len(missing_in_output) >= len(referenced_early):
                issues.append(CoherenceIssue(
                    category=CoherenceCategory.CONTEXT_AMNESIA,
                    severity=CoherenceSeverity.WARNING,
                    turn_index=turn.index,
                    description=(
                        f"Turn {turn.index}: user referenced earlier context "
                        f"({', '.join(list(missing_in_output)[:3])}) but agent's response "
                        f"doesn't acknowledge it. Possible context loss."
                    ),
                    reference_turn=1,
                    evidence={
                        "missing_entities": sorted(missing_in_output),
                        "referenced_entities": sorted(referenced_early),
                    },
                ))

    return issues


def _detect_tool_regression(
    turns: List[TurnTrace],
    steps: List[StepTrace],
) -> List[CoherenceIssue]:
    """Detect when the agent switches to worse tools in later turns.

    If the agent used tool X in turn 2 for purpose P, then uses tool Y
    in turn 5 for the same purpose, and Y is less specific or less
    appropriate, that's a tool regression.
    """
    issues: List[CoherenceIssue] = []
    if len(turns) < 2:
        return issues

    # Group steps by turn
    steps_by_turn: Dict[int, List[str]] = defaultdict(list)
    for step in steps:
        turn_idx = step.turn_index if step.turn_index is not None else 1
        steps_by_turn[turn_idx].append(step.tool_name)

    # Track tool patterns per turn
    turn_tool_sets: List[Tuple[int, Set[str]]] = []
    for turn in turns:
        tools = set(steps_by_turn.get(turn.index, []))
        if tools:
            turn_tool_sets.append((turn.index, tools))

    # Detect when a turn uses strictly fewer tools than an earlier turn
    # with similar query patterns (heuristic for "same purpose")
    for i in range(1, len(turn_tool_sets)):
        curr_idx, curr_tools = turn_tool_sets[i]
        for j in range(i):
            prev_idx, prev_tools = turn_tool_sets[j]

            # If previous turn used more specific tools and current turn
            # dropped some while keeping others, it might be regression
            dropped = prev_tools - curr_tools
            kept = prev_tools & curr_tools
            if dropped and kept and len(dropped) >= len(kept):
                issues.append(CoherenceIssue(
                    category=CoherenceCategory.TOOL_REGRESSION,
                    severity=CoherenceSeverity.WARNING,
                    turn_index=curr_idx,
                    reference_turn=prev_idx,
                    description=(
                        f"Turn {curr_idx} dropped tool(s) {', '.join(sorted(dropped))} "
                        f"that were used in turn {prev_idx}. Agent may be using a "
                        f"less complete strategy."
                    ),
                    evidence={
                        "dropped_tools": sorted(dropped),
                        "kept_tools": sorted(kept),
                        "current_tools": sorted(curr_tools),
                        "previous_tools": sorted(prev_tools),
                    },
                ))

    return issues


def _detect_strategy_drift(
    turns: List[TurnTrace],
    steps: List[StepTrace],
) -> List[CoherenceIssue]:
    """Detect when the agent's tool usage pattern shifts mid-conversation.

    Compares the tool distribution in the first half vs second half of
    the conversation. A significant shift suggests strategy drift.
    """
    issues: List[CoherenceIssue] = []
    if len(turns) < 4:
        return issues

    # Split steps into first half and second half by turn
    mid = len(turns) // 2
    first_half_turns = {t.index for t in turns[:mid]}
    second_half_turns = {t.index for t in turns[mid:]}

    first_tools: List[str] = []
    second_tools: List[str] = []

    for step in steps:
        turn_idx = step.turn_index if step.turn_index is not None else 1
        if turn_idx in first_half_turns:
            first_tools.append(step.tool_name)
        elif turn_idx in second_half_turns:
            second_tools.append(step.tool_name)

    if not first_tools or not second_tools:
        return issues

    # Compare tool distributions
    first_set = set(first_tools)
    second_set = set(second_tools)

    # Tools used in first half but completely absent in second half
    abandoned = first_set - second_set
    # New tools in second half not used before
    new_tools = second_set - first_set

    # Calculate Jaccard similarity between tool sets
    if first_set | second_set:
        jaccard = len(first_set & second_set) / len(first_set | second_set)
    else:
        jaccard = 1.0

    if jaccard < 0.3 and (len(abandoned) >= 2 or len(new_tools) >= 2):
        issues.append(CoherenceIssue(
            category=CoherenceCategory.STRATEGY_DRIFT,
            severity=CoherenceSeverity.WARNING,
            turn_index=turns[mid].index,
            description=(
                f"Agent's tool strategy shifted significantly between first "
                f"and second half of conversation (Jaccard similarity: {jaccard:.0%}). "
                f"Abandoned: {', '.join(sorted(abandoned)) or 'none'}. "
                f"New: {', '.join(sorted(new_tools)) or 'none'}."
            ),
            evidence={
                "jaccard_similarity": round(jaccard, 4),
                "first_half_tools": sorted(first_set),
                "second_half_tools": sorted(second_set),
                "abandoned_tools": sorted(abandoned),
                "new_tools": sorted(new_tools),
            },
        ))

    return issues


def _detect_output_contradiction(
    turns: List[TurnTrace],
) -> List[CoherenceIssue]:
    """Detect when the agent's outputs directly contradict each other.

    Uses simple lexical heuristics to detect:
    - Turn N says "X is Y" and turn M says "X is not Y"
    - Turn N provides value V and turn M provides a different value for
      the same field
    """
    issues: List[CoherenceIssue] = []
    if len(turns) < 2:
        return issues

    # Compare each turn's output against all earlier turns
    outputs = [(t.index, t.output or "") for t in turns if t.output]

    for i in range(1, len(outputs)):
        curr_idx, curr_output = outputs[i]
        curr_lower = curr_output.lower()

        for j in range(i):
            prev_idx, prev_output = outputs[j]
            prev_lower = prev_output.lower()

            if not prev_lower or not curr_lower:
                continue

            # Check for direct negation patterns
            # "X is available" vs "X is not available"
            # "X is correct" vs "X is incorrect"
            # This is a simple heuristic — catches obvious contradictions
            contradiction_found = False

            # Extract short declarative phrases from each output
            import re
            prev_phrases = set(re.findall(r'\b(\w+\s+is\s+\w+)\b', prev_lower))
            curr_phrases = set(re.findall(r'\b(\w+\s+is\s+\w+)\b', curr_lower))

            for pp in prev_phrases:
                # Check if the negation exists in current
                words = pp.split()
                if len(words) >= 3:
                    subject = words[0]
                    predicate = words[2]
                    negated = f"{subject} is not {predicate}"
                    if negated in curr_lower:
                        contradiction_found = True
                        issues.append(CoherenceIssue(
                            category=CoherenceCategory.OUTPUT_CONTRADICTION,
                            severity=CoherenceSeverity.ERROR,
                            turn_index=curr_idx,
                            reference_turn=prev_idx,
                            description=(
                                f"Turn {curr_idx} contradicts turn {prev_idx}: "
                                f"'{pp}' vs '{negated}'"
                            ),
                            evidence={
                                "original_phrase": pp,
                                "contradicting_phrase": negated,
                            },
                        ))
                        break

            if contradiction_found:
                break  # One contradiction per turn pair is enough

    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_coherence(trace: ExecutionTrace) -> CoherenceReport:
    """Run all cross-turn coherence checks on a multi-turn trace.

    Args:
        trace: Execution trace with turn data.

    Returns:
        CoherenceReport with detected issues and coherence score.
    """
    turns = trace.turns or []
    steps = trace.steps

    if len(turns) < 2:
        return CoherenceReport(
            total_turns=len(turns),
            coherence_score=1.0,
        )

    all_issues: List[CoherenceIssue] = []

    all_issues.extend(_detect_context_amnesia(turns, steps))
    all_issues.extend(_detect_tool_regression(turns, steps))
    all_issues.extend(_detect_strategy_drift(turns, steps))
    all_issues.extend(_detect_output_contradiction(turns))

    # Compute coherence score
    score = 1.0
    for issue in all_issues:
        if issue.severity == CoherenceSeverity.ERROR:
            score -= 0.2
        elif issue.severity == CoherenceSeverity.WARNING:
            score -= 0.1
        elif issue.severity == CoherenceSeverity.INFO:
            score -= 0.03
    score = max(0.0, min(1.0, score))

    return CoherenceReport(
        issues=all_issues,
        total_turns=len(turns),
        coherence_score=round(score, 4),
    )
