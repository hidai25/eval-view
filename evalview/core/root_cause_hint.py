"""Root-cause hinter — synthesize coordinated-incident forensics into a narrative.

Where ``detect_coordinated_incident`` in :mod:`evalview.core.noise_tracker`
answers *"are these failures correlated?"*, this module answers the follow-up
question users actually ask in the war room: *"correlated **how**, and what
should I do about it?"*

The hinter is a list of small pure functions that each look for one specific
cross-test pattern and, if it matches, return a :class:`RootCauseHint`
carrying:

- a stable ``cause_id`` (so the cloud / CI can group recurrences),
- a human-readable ``cause_label`` and ``narrative``,
- a structured ``evidence`` dict (raw forensics),
- a tuple of concrete ``suggested_actions`` (CLI commands the operator can
  copy-paste),
- a ``confidence`` rank and a tie-breaker ``priority``.

Design rules:

1. **Pure.** No I/O, no network, no LLM. Each hinter takes a
   :class:`HintContext` and returns ``Optional[RootCauseHint]``. Same input
   → same output, always.
2. **Conservative.** A hinter that's only 70% sure should return ``"low"``
   confidence (or nothing). False-positive root-cause narratives are
   worse than no narrative — they steer humans toward wrong fixes.
3. **Composable.** Hinters live in a single list and are scored uniformly.
   Adding a new heuristic is a one-function patch; see
   ``docs/agent-recipes/add-root-cause-hint.md`` for the contributor recipe.

The selected hint is the one with the highest ``(priority, confidence_rank)``
pair — ties broken by insertion order in :data:`HINTERS` so the result is
deterministic.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Confidence ranking ──────────────────────────────────────────────────────

CONFIDENCE_LEVELS: Tuple[str, ...] = ("low", "medium", "high")
_CONFIDENCE_RANK: Dict[str, int] = {lvl: i for i, lvl in enumerate(CONFIDENCE_LEVELS)}


def _is_failing(diff: Any) -> bool:
    """Best-effort 'this diff represents a failure' predicate.

    Avoids importing :mod:`evalview.core.diff` at module load time —
    keeps the hinter cheap to import in CI hot paths.
    """
    sev = getattr(diff, "overall_severity", None)
    if sev is None:
        return False
    name = getattr(sev, "name", None) or getattr(sev, "value", None) or str(sev)
    return str(name).upper() != "PASSED"


# ── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RootCauseHint:
    """A narrated guess about *why* a batch of tests is failing together."""

    cause_id: str
    cause_label: str
    confidence: str  # "low" | "medium" | "high"
    narrative: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_actions: Tuple[str, ...] = ()
    priority: int = 0

    def confidence_rank(self) -> int:
        return _CONFIDENCE_RANK.get(self.confidence, -1)


@dataclass(frozen=True)
class HintContext:
    """Bundle of pre-computed views over a cycle's diffs.

    Hinters never re-derive what's already in the context — saves work and
    keeps every hinter operating on identical inputs.
    """

    diffs: Tuple[Tuple[str, Any], ...]
    failing: Tuple[Tuple[str, Any], ...]
    min_affected: int

    @property
    def total_tests(self) -> int:
        return len(self.diffs)

    @property
    def failing_count(self) -> int:
        return len(self.failing)


# ── Hinter implementations ──────────────────────────────────────────────────


def hint_provider_rollout(ctx: HintContext) -> Optional[RootCauseHint]:
    """≥N failing tests report ``model_changed=True`` → provider rollout.

    The strongest signal we have: the adapter itself observed a different
    model ID between the golden snapshot and the actual run. This is what
    you see when OpenAI / Anthropic silently rolls a new model under the
    same name, or when an internal deploy switched models without your code
    changing.
    """
    affected: List[str] = []
    transitions: Counter[Tuple[Optional[str], Optional[str]]] = Counter()
    for name, diff in ctx.failing:
        if not getattr(diff, "model_changed", False):
            continue
        affected.append(name)
        transitions[
            (
                getattr(diff, "golden_model_id", None),
                getattr(diff, "actual_model_id", None),
            )
        ] += 1
    if len(affected) < ctx.min_affected:
        return None

    most_common_transition, _ = transitions.most_common(1)[0]
    golden, actual = most_common_transition
    transition_str = (
        f"{golden or '?'} → {actual or '?'}"
        if (golden or actual)
        else "unspecified"
    )

    return RootCauseHint(
        cause_id="provider_rollout",
        cause_label="likely provider rollout",
        confidence="high",
        narrative=(
            f"{len(affected)} failing tests observed a model-ID change "
            f"between snapshot and run ({transition_str}). The agent code "
            f"didn't change — the provider did. Treat as a rebase candidate "
            f"once you've confirmed the new behavior is acceptable."
        ),
        evidence={
            "signal": "model_changed_flag",
            "affected_count": len(affected),
            "transitions": [
                {"from": g, "to": a, "count": c}
                for (g, a), c in transitions.most_common()
            ],
        },
        suggested_actions=(
            f"evalview model-check --model {actual} --pin"
            if actual
            else "evalview model-check --pin",
            "evalview snapshot   # rebase if the new behavior is acceptable",
            "evalview check --statistical 5   # confirm before any rebase",
        ),
        priority=100,
    )


def hint_runtime_fingerprint_shift(ctx: HintContext) -> Optional[RootCauseHint]:
    """≥N failing tests share a new runtime model fingerprint.

    Weaker than ``model_changed`` because some providers don't emit a stable
    ``model_id`` per-response, but they do leak a fingerprint or
    ``system_fingerprint``. Same flavor of cause (runtime change), one
    confidence level lower because the signal is indirect.
    """
    fingerprints: Dict[str, List[str]] = {}
    for name, diff in ctx.failing:
        fp = (
            getattr(diff, "actual_runtime_fingerprint", None)
            or getattr(diff, "runtime_model_fingerprint", None)
            or getattr(diff, "actual_model_id", None)
        )
        if not fp:
            continue
        # Only counts as a shift if the fingerprint differs from baseline.
        baseline_fp = (
            getattr(diff, "golden_runtime_fingerprint", None)
            or getattr(diff, "golden_model_id", None)
        )
        if baseline_fp and baseline_fp == fp:
            continue
        fingerprints.setdefault(str(fp), []).append(name)

    if not fingerprints:
        return None

    fp, tests = max(fingerprints.items(), key=lambda item: (len(item[1]), item[0]))
    if len(tests) < ctx.min_affected:
        return None

    return RootCauseHint(
        cause_id="runtime_fingerprint_shift",
        cause_label="runtime fingerprint shift",
        confidence="medium",
        narrative=(
            f"{len(tests)} failing tests share a runtime fingerprint "
            f"({fp!r}) that doesn't match their golden baselines. The "
            f"provider didn't change the model name, but something behind "
            f"that name moved — a quantization swap, a routing change, or a "
            f"server-side rev. Often invisible in dashboards."
        ),
        evidence={
            "signal": "runtime_fingerprint",
            "fingerprint": fp,
            "affected_count": len(tests),
            "affected": sorted(tests),
        },
        suggested_actions=(
            "evalview model-check --pin   # capture the new fingerprint",
            "evalview check --statistical 5   # confirm the shift isn't a single-cycle blip",
            "evalview snapshot   # rebase if the new behavior is acceptable",
        ),
        priority=80,
    )


def _failing_tool_changes(
    failing: Tuple[Tuple[str, Any], ...],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (added_tool → tests, removed_tool → tests) across failing diffs.

    Helper for the two tool-change hinters below. A tool is "added" when it
    appears in the actual trace but not the golden; "removed" is the inverse.
    """
    added: Dict[str, List[str]] = {}
    removed: Dict[str, List[str]] = {}
    for name, diff in failing:
        for td in getattr(diff, "tool_diffs", None) or []:
            t_type = getattr(td, "type", None)
            if t_type == "added":
                tool_name = getattr(td, "actual_tool", None)
                if tool_name:
                    added.setdefault(tool_name, []).append(name)
            elif t_type == "removed":
                tool_name = getattr(td, "golden_tool", None)
                if tool_name:
                    removed.setdefault(tool_name, []).append(name)
    return added, removed


def hint_coordinated_tool_addition(ctx: HintContext) -> Optional[RootCauseHint]:
    """≥N failing tests all newly call the same tool that wasn't in baseline.

    Almost always a prompt edit or a tool-description tweak that nudged the
    model toward the new tool. Sometimes a new MCP tool that quietly became
    available in the agent's catalog.
    """
    added, _ = _failing_tool_changes(ctx.failing)
    if not added:
        return None
    tool, tests = max(added.items(), key=lambda item: (len(item[1]), item[0]))
    if len(tests) < ctx.min_affected:
        return None

    return RootCauseHint(
        cause_id="coordinated_tool_addition",
        cause_label=f"new tool '{tool}' called across {len(tests)} failing tests",
        confidence="high",
        narrative=(
            f"All {len(tests)} of these failures introduced the same new "
            f"tool call: '{tool}'. The agent didn't randomly drift toward "
            f"it — something steered the model. Usual suspects: a tool "
            f"description change, a prompt edit, or a newly-registered MCP "
            f"tool entering the catalog."
        ),
        evidence={
            "signal": "shared_tool_addition",
            "tool": tool,
            "affected_count": len(tests),
            "affected": sorted(tests),
        },
        suggested_actions=(
            f"git log --since=7.days -- '*{tool}*'   # find the change",
            f"grep -rn '{tool}' src/ tools/   # review tool description",
            "evalview replay <one-test> --trace   # inspect the decision",
        ),
        priority=70,
    )


def hint_coordinated_tool_removal(ctx: HintContext) -> Optional[RootCauseHint]:
    """≥N failing tests all stopped calling the same tool.

    The flip side of addition: a tool that the agent reliably invoked is
    now silently skipped. Often a tool description regression, an MCP
    server going away, or a guardrail mis-filtering the tool list.
    """
    _, removed = _failing_tool_changes(ctx.failing)
    if not removed:
        return None
    tool, tests = max(removed.items(), key=lambda item: (len(item[1]), item[0]))
    if len(tests) < ctx.min_affected:
        return None

    return RootCauseHint(
        cause_id="coordinated_tool_removal",
        cause_label=f"tool '{tool}' no longer called across {len(tests)} failing tests",
        confidence="high",
        narrative=(
            f"All {len(tests)} of these failures stopped invoking '{tool}'. "
            f"Likely causes: the tool was removed from the catalog, its "
            f"description was edited so the model no longer matches it, or "
            f"a guardrail / filter is hiding it from the LLM."
        ),
        evidence={
            "signal": "shared_tool_removal",
            "tool": tool,
            "affected_count": len(tests),
            "affected": sorted(tests),
        },
        suggested_actions=(
            f"evalview adapters --check-tool {tool}   # confirm registration",
            f"git log --since=7.days -- '*{tool}*'   # find the change",
            "evalview replay <one-test> --trace   # inspect tool catalog at runtime",
        ),
        priority=70,
    )


def hint_coordinated_output_drift(ctx: HintContext) -> Optional[RootCauseHint]:
    """≥N failing tests dropped output similarity below 0.7 with no tool change.

    Tools fired the same way, the agent did the right steps — but the
    *wording* of its answer drifted enough to fail evals. The classic
    "model got chattier / terser / refused" pattern. Distinct from the
    fingerprint shift because we don't claim to know the underlying
    runtime moved; just that the prose did.
    """
    affected: List[str] = []
    similarities: List[float] = []
    for name, diff in ctx.failing:
        if getattr(diff, "tool_diffs", None):
            continue  # tool change is a stronger / different signal
        output_diff = getattr(diff, "output_diff", None)
        if output_diff is None:
            continue
        sim = getattr(output_diff, "similarity", None)
        if sim is None or sim >= 0.7:
            continue
        affected.append(name)
        similarities.append(float(sim))

    if len(affected) < ctx.min_affected:
        return None

    avg_sim = sum(similarities) / len(similarities)
    return RootCauseHint(
        cause_id="coordinated_output_drift",
        cause_label="coordinated output-only drift",
        confidence="medium",
        narrative=(
            f"{len(affected)} failing tests kept the same tool sequence but "
            f"their output wording drifted significantly (avg similarity "
            f"{avg_sim:.0%}). The agent is still doing the right *steps* — "
            f"it's the prose that moved. Common after a model or system-"
            f"prompt edit that didn't change tool wiring."
        ),
        evidence={
            "signal": "output_drift_no_tool_change",
            "affected_count": len(affected),
            "avg_similarity": round(avg_sim, 3),
            "affected": sorted(affected),
        },
        suggested_actions=(
            "evalview check --statistical 5   # is the drift stable or flaky?",
            "evalview replay <one-test> --trace   # eyeball the new wording",
            "evalview snapshot   # rebase if the new wording is acceptable",
        ),
        priority=50,
    )


# ── Registry ────────────────────────────────────────────────────────────────


HinterFn = Callable[[HintContext], Optional[RootCauseHint]]


# Order matters only as a deterministic tie-breaker — final selection is
# driven by ``(priority, confidence_rank)``. Append, don't reorder, when
# adding new hinters: it keeps shipped evidence keys stable.
HINTERS: Tuple[HinterFn, ...] = (
    hint_provider_rollout,
    hint_runtime_fingerprint_shift,
    hint_coordinated_tool_addition,
    hint_coordinated_tool_removal,
    hint_coordinated_output_drift,
)


# Roadmap of hinters that would be valuable but aren't shipped yet. Each
# bullet is sized to be a contributor-friendly first PR — implement one
# function, add it to ``HINTERS``, ship a test. See
# ``docs/agent-recipes/add-root-cause-hint.md`` for the recipe.
HINTERS_ROADMAP: Tuple[str, ...] = (
    "coordinated_cost_spike: ≥N failing tests with cost ratio > 2× baseline "
    "→ likely retry storm / expensive model swap.",
    "coordinated_latency_spike: ≥N failing tests with latency > 2× baseline "
    "and no tool change → likely upstream throttling or cold-cache event.",
    "coordinated_refusal: ≥N failing tests where actual output contains "
    "refusal phrases ('I cannot', 'I'm not able to') and baseline didn't "
    "→ likely safety classifier / guardrail change.",
    "coordinated_parameter_drift: ≥N tools called with same parameter value "
    "shift across tests → likely schema / config change.",
    "coordinated_decision_drift: ≥N rationale events show the same "
    "previously-non-preferred alternative being picked → likely "
    "tool-description nudge. (Requires rationale_events on diff.)",
    "coordinated_retrieval_drop: ≥N tests where retrieved chunk overlap "
    "with baseline dropped below 0.5 → likely index / embedding change.",
)


# ── Public entry point ──────────────────────────────────────────────────────


def analyze_root_cause_hint(
    diffs: List[Tuple[str, Any]],
    min_affected: int = 3,
) -> Optional[RootCauseHint]:
    """Return the best-matching root-cause hint, or ``None``.

    Args:
        diffs: The same ``(test_name, diff)`` list that
            :func:`evalview.core.noise_tracker.detect_coordinated_incident`
            already consumes. Mixing passed and failing diffs is fine — the
            hinter filters to failing diffs internally.
        min_affected: Minimum number of correlated failures required for any
            hint to fire. Keep this in lockstep with the noise tracker's
            ``min_affected`` so the two layers agree on what "coordinated"
            means.

    Selection is deterministic: highest ``priority`` wins, ties broken by
    higher confidence, then by ``HINTERS`` registration order. Returns
    ``None`` when no hinter matches — callers should fall back to the
    untriaged incident headline they were already rendering.
    """
    if not diffs:
        return None
    failing = tuple((n, d) for n, d in diffs if _is_failing(d))
    if len(failing) < min_affected:
        return None
    ctx = HintContext(
        diffs=tuple(diffs),
        failing=failing,
        min_affected=min_affected,
    )

    best: Optional[RootCauseHint] = None
    for hinter in HINTERS:
        hint = hinter(ctx)
        if hint is None:
            continue
        if best is None:
            best = hint
            continue
        if (hint.priority, hint.confidence_rank()) > (
            best.priority,
            best.confidence_rank(),
        ):
            best = hint
    return best


def hint_to_dict(hint: RootCauseHint) -> Dict[str, Any]:
    """Render a hint as a plain dict suitable for JSON / Slack payloads.

    Kept as a free function rather than a method so :class:`RootCauseHint`
    stays a pure data carrier (frozen, hashable, dataclass-only).
    """
    return {
        "cause_id": hint.cause_id,
        "cause_label": hint.cause_label,
        "confidence": hint.confidence,
        "narrative": hint.narrative,
        "evidence": dict(hint.evidence),
        "suggested_actions": list(hint.suggested_actions),
        "priority": hint.priority,
    }
