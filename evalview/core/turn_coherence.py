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
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from evalview.core.types import ExecutionTrace, TurnTrace, StepTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# Minimum turns required for each detector to fire
AMNESIA_MIN_TURNS = 3
REGRESSION_MIN_TURNS = 2
DRIFT_MIN_TURNS = 4
CONTRADICTION_MIN_TURNS = 2

# Jaccard similarity below which tool sets are considered "drifted"
DRIFT_JACCARD_THRESHOLD = 0.3

# Minimum abandoned or new tools to flag strategy drift
DRIFT_MIN_TOOL_CHANGE = 2

# Coherence score penalties per severity level
COHERENCE_PENALTY_ERROR = 0.2
COHERENCE_PENALTY_WARNING = 0.1
COHERENCE_PENALTY_INFO = 0.03


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

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
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
) -> List[CoherenceIssue]:
    """Detect when the agent stops referencing earlier context.

    Tracks key entities introduced in early turns and checks whether
    the agent still references them when relevant in later turns.
    """
    issues: List[CoherenceIssue] = []
    if len(turns) < AMNESIA_MIN_TURNS:
        return issues

    # Build entity set per turn from queries (what the user said)
    query_entities_by_turn: Dict[int, Set[str]] = {}
    cumulative_entities: Set[str] = set()

    for turn in turns:
        turn_entities = _extract_key_entities(turn.query)
        query_entities_by_turn[turn.index] = turn_entities
        cumulative_entities |= turn_entities

    if not cumulative_entities:
        return issues

    # Pre-compute early entities (first half of conversation) — constant
    early_turn_cutoff = max(1, len(turns) // 2)
    early_entities: Set[str] = set()
    # Also track which turn introduced each entity (for reference_turn)
    entity_origin_turn: Dict[str, int] = {}
    for t in turns[:early_turn_cutoff]:
        for entity in query_entities_by_turn.get(t.index, set()):
            early_entities.add(entity)
            if entity not in entity_origin_turn:
                entity_origin_turn[entity] = t.index

    if not early_entities:
        return issues

    # Check later turns' outputs for reference to earlier entities
    # Skip the first 2 turns (too early for amnesia)
    for turn in turns[2:]:
        if not turn.output:
            continue

        output_lower = turn.output.lower()

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
                # Point to the turn that first introduced the missing entity
                first_missing = sorted(missing_in_output)[0]
                origin = entity_origin_turn.get(first_missing, turns[0].index)
                issues.append(CoherenceIssue(
                    category=CoherenceCategory.CONTEXT_AMNESIA,
                    severity=CoherenceSeverity.WARNING,
                    turn_index=turn.index,
                    description=(
                        f"Turn {turn.index}: user referenced earlier context "
                        f"({', '.join(list(missing_in_output)[:3])}) but agent's response "
                        f"doesn't acknowledge it. Possible context loss."
                    ),
                    reference_turn=origin,
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
    """Detect when the agent drops tools it previously used.

    Compares each turn's tool set against earlier turns. If a later turn
    drops at least as many tools as it keeps from an earlier turn, it may
    indicate a less complete strategy.

    Limitation: this is a structural heuristic — it checks tool-set
    differences but does not verify that the turns have the same purpose.
    Different queries naturally use different tools; this detector may
    flag unrelated turns. Best used for conversations where the user's
    intent is consistent across turns.
    """
    issues: List[CoherenceIssue] = []
    if len(turns) < REGRESSION_MIN_TURNS:
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

    # Detect when a turn uses strictly fewer tools than an earlier turn.
    # Emit at most ONE issue per later turn — pick the earlier turn with the
    # largest `dropped` set (most informative reference) to avoid O(n²) near-
    # duplicate warnings on long conversations.
    for i in range(1, len(turn_tool_sets)):
        curr_idx, curr_tools = turn_tool_sets[i]
        best: Optional[Tuple[int, Set[str], Set[str]]] = None  # (prev_idx, dropped, kept)
        for j in range(i):
            prev_idx, prev_tools = turn_tool_sets[j]
            dropped = prev_tools - curr_tools
            kept = prev_tools & curr_tools
            if dropped and kept and len(dropped) >= len(kept):
                if best is None or len(dropped) > len(best[1]):
                    best = (prev_idx, dropped, kept)
        if best is not None:
            prev_idx, dropped, kept = best
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
                    "previous_tools": sorted(dropped | kept),
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
    if len(turns) < DRIFT_MIN_TURNS:
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

    if jaccard < DRIFT_JACCARD_THRESHOLD and (len(abandoned) >= DRIFT_MIN_TOOL_CHANGE or len(new_tools) >= DRIFT_MIN_TOOL_CHANGE):
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


# Pre-compiled patterns for contradiction detection
# "X is Y" — captures subject, predicate for negation check
_IS_PHRASE_RE = re.compile(r'\b(\w+)\s+is\s+(\w+)\b')
# "the X is $50" / "the X is 42" — captures labeled values
_LABELED_VALUE_RE = re.compile(
    r'\bthe\s+(\w+)\s+is\s+'          # "the price is"
    r'(\$?[\d,]+(?:\.\d+)?)\b'        # "$50" / "42" / "1,200.50"
)
# "X has/have Y" vs "X doesn't/don't have Y"
_HAS_PHRASE_RE = re.compile(r'\b(\w+)\s+(has|have)\s+(\w+)\b')


def _detect_output_contradiction(
    turns: List[TurnTrace],
) -> List[CoherenceIssue]:
    """Detect when the agent's outputs directly contradict each other.

    Uses lexical heuristics to detect three types of contradictions:
    1. Negation: "X is Y" vs "X is not Y"
    2. Value change: "the price is $50" vs "the price is $75"
    3. Has/doesn't: "X has Y" vs "X doesn't have Y"

    Limitations: does not catch semantic contradictions that don't match
    these structural patterns (e.g., "available Monday" vs "available Tuesday").
    """
    issues: List[CoherenceIssue] = []
    if len(turns) < CONTRADICTION_MIN_TURNS:
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

            contradiction = _find_contradiction(prev_lower, curr_lower)
            if contradiction:
                original, contradicting, kind = contradiction
                issues.append(CoherenceIssue(
                    category=CoherenceCategory.OUTPUT_CONTRADICTION,
                    severity=CoherenceSeverity.ERROR,
                    turn_index=curr_idx,
                    reference_turn=prev_idx,
                    description=(
                        f"Turn {curr_idx} contradicts turn {prev_idx}: "
                        f"'{original}' vs '{contradicting}'"
                    ),
                    evidence={
                        "original_phrase": original,
                        "contradicting_phrase": contradicting,
                        "contradiction_type": kind,
                    },
                ))
                break  # One contradiction per later turn is enough

    return issues


def _find_contradiction(
    prev: str,
    curr: str,
) -> Optional[Tuple[str, str, str]]:
    """Find a contradiction between two output strings.

    Returns (original_phrase, contradicting_phrase, kind) or None.
    """
    # 1. Negation: "X is Y" vs "X is not Y"
    for match in _IS_PHRASE_RE.finditer(prev):
        subject, predicate = match.group(1), match.group(2)
        negated = f"{subject} is not {predicate}"
        if negated in curr:
            return (match.group(), negated, "negation")

    # Also check reverse: current says "X is Y", prev says "X is not Y"
    for match in _IS_PHRASE_RE.finditer(curr):
        subject, predicate = match.group(1), match.group(2)
        negated = f"{subject} is not {predicate}"
        if negated in prev:
            return (negated, match.group(), "negation")

    # 2. Value change: "the price is $50" vs "the price is $75"
    prev_values: Dict[str, str] = {}
    for match in _LABELED_VALUE_RE.finditer(prev):
        label, value = match.group(1), match.group(2)
        prev_values[label] = value

    for match in _LABELED_VALUE_RE.finditer(curr):
        label, value = match.group(1), match.group(2)
        if label in prev_values and prev_values[label] != value:
            return (
                f"the {label} is {prev_values[label]}",
                f"the {label} is {value}",
                "value_change",
            )

    # 3. Has/doesn't have: "X has Y" vs "X doesn't have Y"
    for match in _HAS_PHRASE_RE.finditer(prev):
        subject, _verb, obj = match.group(1), match.group(2), match.group(3)
        negated_patterns = [
            f"{subject} doesn't have {obj}",
            f"{subject} does not have {obj}",
            f"{subject} don't have {obj}",
            f"{subject} do not have {obj}",
        ]
        for neg in negated_patterns:
            if neg in curr:
                return (match.group(), neg, "has_negation")

    return None


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

    all_issues.extend(_detect_context_amnesia(turns))
    all_issues.extend(_detect_tool_regression(turns, steps))
    all_issues.extend(_detect_strategy_drift(turns, steps))
    all_issues.extend(_detect_output_contradiction(turns))

    # Compute coherence score
    score = 1.0
    for issue in all_issues:
        if issue.severity == CoherenceSeverity.ERROR:
            score -= COHERENCE_PENALTY_ERROR
        elif issue.severity == CoherenceSeverity.WARNING:
            score -= COHERENCE_PENALTY_WARNING
        elif issue.severity == CoherenceSeverity.INFO:
            score -= COHERENCE_PENALTY_INFO
    score = max(0.0, min(1.0, score))

    return CoherenceReport(
        issues=all_issues,
        total_turns=len(turns),
        coherence_score=round(score, 4),
    )
