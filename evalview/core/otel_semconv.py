"""OpenTelemetry semantic conventions for AI agents.

`gen_ai.*` exists in upstream OTel for model-call instrumentation, but the
agent layer above it (turns, tool selections, handoffs, memory reads,
retrieval, human interventions) has no shared vocabulary. Every observability
vendor invents their own attribute names, so traces from one stack are
useless to another.

This module defines a portable vocabulary EvalView can both emit and consume
— and that any other tool can adopt without depending on EvalView. The
constants here are the *only* public contract; callers should always import
the named constants rather than hard-coding strings, so a future spec
revision can rename centrally without breaking adapters.

Status: **draft v0**. Field names are stable for any release labeled
``OTEL_SEMCONV_VERSION``; a bump indicates breaking changes. EvalView's
own emitter writes ``otel.semconv.version`` on every root span so consumers
know how to parse it.

Adopting this convention does **not** require pulling in OTel SDKs — the
constants are plain strings. Adapters that already emit OTel spans should
add these attributes alongside ``gen_ai.*``; adapters that don't can store
them as plain dict keys on their existing trace structures.

See ``docs/agent-recipes/add-otel-emission.md`` for the contributor recipe.
"""
from __future__ import annotations

from typing import Final


# ── Version ─────────────────────────────────────────────────────────────────

# Bumped only on breaking changes (renaming an attribute, removing a span
# kind, changing the meaning of an enum value). Additive changes (new
# attributes, new span kinds, new enum values) keep the version stable.
OTEL_SEMCONV_VERSION: Final[str] = "0.1.0"


# ── Span names ──────────────────────────────────────────────────────────────
#
# Span names follow ``agent.<verb>`` so they sort together in any UI that
# alphabetizes spans, and so the first dot reliably separates namespace
# from operation.

SPAN_AGENT_RUN: Final[str] = "agent.run"
"""Root span for one agent invocation. Wraps everything below."""

SPAN_AGENT_TURN: Final[str] = "agent.turn"
"""One conversation turn (one user input → one agent response)."""

SPAN_AGENT_TOOL_CHOICE: Final[str] = "agent.tool_choice"
"""The model's decision about which tool (if any) to call next."""

SPAN_AGENT_TOOL_CALL: Final[str] = "agent.tool_call"
"""Execution of a chosen tool. Wraps the tool's own internal spans."""

SPAN_AGENT_HANDOFF: Final[str] = "agent.handoff"
"""Control transfer to another agent (multi-agent), to a human, or back."""

SPAN_AGENT_MEMORY_READ: Final[str] = "agent.memory.read"
"""Lookup against a memory store (long-term, episodic, working)."""

SPAN_AGENT_MEMORY_WRITE: Final[str] = "agent.memory.write"
"""Persist new content to a memory store."""

SPAN_AGENT_RETRIEVAL: Final[str] = "agent.retrieval"
"""Retrieval-augmented-generation lookup (vector / keyword / hybrid)."""

SPAN_AGENT_INTERVENTION: Final[str] = "agent.intervention"
"""Human-in-the-loop step: review, approval, manual edit, escalation."""

SPAN_AGENT_PLAN: Final[str] = "agent.plan"
"""A planning step that produced one or more candidate trajectories."""


# Convenience tuple for consumers that want to iterate every known span name.
SPAN_NAMES: Final[tuple[str, ...]] = (
    SPAN_AGENT_RUN,
    SPAN_AGENT_TURN,
    SPAN_AGENT_TOOL_CHOICE,
    SPAN_AGENT_TOOL_CALL,
    SPAN_AGENT_HANDOFF,
    SPAN_AGENT_MEMORY_READ,
    SPAN_AGENT_MEMORY_WRITE,
    SPAN_AGENT_RETRIEVAL,
    SPAN_AGENT_INTERVENTION,
    SPAN_AGENT_PLAN,
)


# ── Common attributes ───────────────────────────────────────────────────────
#
# Conventions:
#   - ``agent.*`` for fields specific to the agent abstraction.
#   - ``gen_ai.*`` is upstream OTel; we don't redefine it. Use it directly
#     for model name, prompt tokens, etc.
#   - Booleans use unprefixed names ending in ``_changed`` / ``_used``.
#   - IDs are strings (UUIDs or provider-native).

# Identity / lineage
ATTR_AGENT_ID: Final[str] = "agent.id"
ATTR_AGENT_NAME: Final[str] = "agent.name"
ATTR_AGENT_VERSION: Final[str] = "agent.version"
ATTR_AGENT_FRAMEWORK: Final[str] = "agent.framework"
"""``langgraph``, ``crewai``, ``openai-assistants``, etc."""

ATTR_AGENT_RUN_ID: Final[str] = "agent.run.id"
ATTR_AGENT_TURN_INDEX: Final[str] = "agent.turn.index"
ATTR_AGENT_PARENT_RUN_ID: Final[str] = "agent.parent_run.id"
"""Set when a multi-agent handoff spawned this run."""

# State
ATTR_AGENT_STATE_FINGERPRINT: Final[str] = "agent.state.fingerprint"
"""Hash of the agent's persistent state at span start. Lets consumers
detect that the same agent reached step N from two different histories."""

ATTR_AGENT_STATE_SIZE_BYTES: Final[str] = "agent.state.size_bytes"

# Goal / intent
ATTR_AGENT_GOAL_TEXT: Final[str] = "agent.goal.text"
"""User-stated or system-derived goal at run start."""

ATTR_AGENT_GOAL_FINGERPRINT: Final[str] = "agent.goal.fingerprint"
"""Hash of the goal — useful when text is too long to log per span."""

ATTR_AGENT_GOAL_DRIFT_DELTA: Final[str] = "agent.goal.drift_delta"
"""Float in [0, 1]; 0 = on-goal, 1 = fully drifted from stated goal.
See ``evalview.core.goal_drift`` for the reference computation."""

# Tool choice / call
ATTR_TOOL_NAME: Final[str] = "agent.tool.name"
ATTR_TOOL_VERSION: Final[str] = "agent.tool.version"
ATTR_TOOL_CHOICE_REASON: Final[str] = "agent.tool.choice.reason"
"""Free-text or structured reason from the model. Truncate to a budget
before emitting (see ``RATIONALE_MAX_TEXT_BYTES`` in core.types)."""

ATTR_TOOL_ALTERNATIVES: Final[str] = "agent.tool.alternatives"
"""JSON-encoded list of tools the model considered but didn't pick."""

ATTR_TOOL_PARAMETERS_FINGERPRINT: Final[str] = "agent.tool.parameters.fingerprint"
"""Hash of normalized tool args — lets consumers detect param drift
without storing the raw arguments."""

ATTR_TOOL_RESULT_STATUS: Final[str] = "agent.tool.result.status"
"""``ok`` | ``error`` | ``timeout`` | ``rate_limited``."""

# Retrieval
ATTR_RETRIEVAL_QUERY: Final[str] = "agent.retrieval.query"
ATTR_RETRIEVAL_INDEX: Final[str] = "agent.retrieval.index"
ATTR_RETRIEVAL_TOP_K: Final[str] = "agent.retrieval.top_k"
ATTR_RETRIEVAL_CHUNK_IDS: Final[str] = "agent.retrieval.chunk_ids"
"""JSON-encoded list of returned chunk IDs (in rank order)."""

ATTR_RETRIEVAL_CHUNK_SCORES: Final[str] = "agent.retrieval.chunk_scores"
"""JSON-encoded list of similarity scores parallel to chunk_ids."""

ATTR_RETRIEVAL_INFLUENCE_SCORES: Final[str] = "agent.retrieval.influence_scores"
"""Optional: per-chunk attribution to the final output. See
``evalview.core.retrieval_lineage`` for the deterministic baseline."""

# Memory
ATTR_MEMORY_STORE: Final[str] = "agent.memory.store"
"""``working`` | ``episodic`` | ``semantic`` | ``profile``."""

ATTR_MEMORY_KEY: Final[str] = "agent.memory.key"
ATTR_MEMORY_AGE_SECONDS: Final[str] = "agent.memory.age_seconds"
"""Wall-clock age of the read entry. Helps spot 'stale memory' drift."""

ATTR_MEMORY_HIT: Final[str] = "agent.memory.hit"
"""Boolean: did the read return a value?"""

# Handoff
ATTR_HANDOFF_TO: Final[str] = "agent.handoff.to"
"""Target agent name, ``human``, or ``terminate``."""

ATTR_HANDOFF_REASON: Final[str] = "agent.handoff.reason"

# Intervention
ATTR_INTERVENTION_KIND: Final[str] = "agent.intervention.kind"
"""``approval`` | ``edit`` | ``redact`` | ``escalate``."""

ATTR_INTERVENTION_OUTCOME: Final[str] = "agent.intervention.outcome"
"""``approved`` | ``rejected`` | ``modified`` | ``timeout``."""

# Cost / usage
ATTR_AGENT_COST_USD: Final[str] = "agent.cost.usd"
"""Cumulative cost across this span and its descendants."""

# Verdict (set by EvalView when the run is post-processed)
ATTR_AGENT_VERDICT: Final[str] = "agent.verdict"
"""``safe_to_ship`` | ``ship_with_quarantine`` | ``investigate`` | ``block_release``."""


# Convenience tuple for consumers that want to iterate every known attribute.
ATTRIBUTES: Final[tuple[str, ...]] = (
    ATTR_AGENT_ID,
    ATTR_AGENT_NAME,
    ATTR_AGENT_VERSION,
    ATTR_AGENT_FRAMEWORK,
    ATTR_AGENT_RUN_ID,
    ATTR_AGENT_TURN_INDEX,
    ATTR_AGENT_PARENT_RUN_ID,
    ATTR_AGENT_STATE_FINGERPRINT,
    ATTR_AGENT_STATE_SIZE_BYTES,
    ATTR_AGENT_GOAL_TEXT,
    ATTR_AGENT_GOAL_FINGERPRINT,
    ATTR_AGENT_GOAL_DRIFT_DELTA,
    ATTR_TOOL_NAME,
    ATTR_TOOL_VERSION,
    ATTR_TOOL_CHOICE_REASON,
    ATTR_TOOL_ALTERNATIVES,
    ATTR_TOOL_PARAMETERS_FINGERPRINT,
    ATTR_TOOL_RESULT_STATUS,
    ATTR_RETRIEVAL_QUERY,
    ATTR_RETRIEVAL_INDEX,
    ATTR_RETRIEVAL_TOP_K,
    ATTR_RETRIEVAL_CHUNK_IDS,
    ATTR_RETRIEVAL_CHUNK_SCORES,
    ATTR_RETRIEVAL_INFLUENCE_SCORES,
    ATTR_MEMORY_STORE,
    ATTR_MEMORY_KEY,
    ATTR_MEMORY_AGE_SECONDS,
    ATTR_MEMORY_HIT,
    ATTR_HANDOFF_TO,
    ATTR_HANDOFF_REASON,
    ATTR_INTERVENTION_KIND,
    ATTR_INTERVENTION_OUTCOME,
    ATTR_AGENT_COST_USD,
    ATTR_AGENT_VERDICT,
)


# ── Validation helpers ──────────────────────────────────────────────────────


def is_known_span(name: str) -> bool:
    """True when ``name`` is in the spec.

    Useful for adapter authors who want to assert their emitter never
    invents an off-spec span name. Strict by design — if you need a new
    span, add it to the spec (PR), don't pass an unknown name.
    """
    return name in SPAN_NAMES


def is_known_attribute(key: str) -> bool:
    """True when ``key`` is a documented attribute.

    Same intent as :func:`is_known_span` — pin emitters to the spec at
    test time so drift between the constants and the wire format is
    caught immediately.
    """
    return key in ATTRIBUTES


def attributes_for_span(span: str) -> frozenset[str]:
    """Return the attributes that *should* appear on a given span.

    A reference, not an enforcement: emitters are free to add OTel
    standard attributes (``service.name``, ``error.type``, etc.). The
    intent is to tell adapter authors *"if you emit `agent.tool_call`,
    these are the agent-layer attributes consumers expect."*
    """
    base = frozenset({
        ATTR_AGENT_RUN_ID, ATTR_AGENT_TURN_INDEX,
        ATTR_AGENT_NAME, ATTR_AGENT_FRAMEWORK,
    })
    if span == SPAN_AGENT_RUN:
        return base | {
            ATTR_AGENT_ID, ATTR_AGENT_VERSION, ATTR_AGENT_GOAL_TEXT,
            ATTR_AGENT_STATE_FINGERPRINT, ATTR_AGENT_COST_USD,
            ATTR_AGENT_VERDICT,
        }
    if span == SPAN_AGENT_TURN:
        return base | {ATTR_AGENT_GOAL_DRIFT_DELTA, ATTR_AGENT_STATE_FINGERPRINT}
    if span == SPAN_AGENT_TOOL_CHOICE:
        return base | {
            ATTR_TOOL_NAME, ATTR_TOOL_CHOICE_REASON, ATTR_TOOL_ALTERNATIVES,
        }
    if span == SPAN_AGENT_TOOL_CALL:
        return base | {
            ATTR_TOOL_NAME, ATTR_TOOL_VERSION,
            ATTR_TOOL_PARAMETERS_FINGERPRINT, ATTR_TOOL_RESULT_STATUS,
        }
    if span == SPAN_AGENT_HANDOFF:
        return base | {
            ATTR_HANDOFF_TO, ATTR_HANDOFF_REASON, ATTR_AGENT_PARENT_RUN_ID,
        }
    if span == SPAN_AGENT_MEMORY_READ:
        return base | {
            ATTR_MEMORY_STORE, ATTR_MEMORY_KEY,
            ATTR_MEMORY_AGE_SECONDS, ATTR_MEMORY_HIT,
        }
    if span == SPAN_AGENT_MEMORY_WRITE:
        return base | {ATTR_MEMORY_STORE, ATTR_MEMORY_KEY}
    if span == SPAN_AGENT_RETRIEVAL:
        return base | {
            ATTR_RETRIEVAL_QUERY, ATTR_RETRIEVAL_INDEX,
            ATTR_RETRIEVAL_TOP_K, ATTR_RETRIEVAL_CHUNK_IDS,
            ATTR_RETRIEVAL_CHUNK_SCORES,
        }
    if span == SPAN_AGENT_INTERVENTION:
        return base | {ATTR_INTERVENTION_KIND, ATTR_INTERVENTION_OUTCOME}
    if span == SPAN_AGENT_PLAN:
        return base
    return base
