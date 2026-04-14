"""Deterministic synthesizer: failing production trace -> regression test YAML.

This module is the heart of the ``evalview autopr`` glue. It takes a single
"incident" record (a structured description of a failing test, typically
written by ``evalview monitor`` when the confirmation gate fires) and returns
a dict that is ready to be dumped as a regression test YAML.

The function is **pure**: no I/O, no network, no LLM. That keeps it fast,
testable, and safe to run in CI. The output schema matches
``evalview.core.types.TestCase``.

Schema of an incident record
----------------------------
Every incident is a JSON object with these fields (all optional except
``test_name`` and ``query``)::

    {
        "version": 1,
        "timestamp": "2026-04-14T12:34:56Z",
        "test_name": "refund-request",
        "query": "I want a refund for order #123",
        "status": "REGRESSION",                # DiffStatus value
        "baseline_tools": ["lookup_order", "check_policy", "process_refund"],
        "actual_tools":   ["lookup_order", "process_refund"],
        "baseline_output": "After checking our policy, ...",
        "actual_output":   "Sure, I processed your refund.",
        "score_delta": -30.0,
        "model_changed": false,
        "golden_model_id": "claude-opus-4-5-20251101",
        "actual_model_id": "claude-opus-4-5-20251101",
        "source_file": "tests/refund-request.yaml"
    }

Callers that don't have every field can omit it. The synthesizer degrades
gracefully: missing ``baseline_tools`` means no ``expected.tools`` clause,
missing ``actual_output`` means no ``not_contains`` clause, etc.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Max length for a "must not contain" phrase extracted from the bad output.
# Short phrases are more robust: long ones overfit to exact wording and
# break the moment the model rephrases the same hallucination.
_MAX_PHRASE_LEN = 80

# How many sentences to pull out of the actual (bad) output as negative
# assertions. More = stricter regression test, but also more brittle.
_MAX_NEGATIVE_PHRASES = 3

# Minimum score for generated regression tests. Set aggressively — a
# regression test exists to codify "never again", so it should fail loudly.
_DEFAULT_MIN_SCORE = 90.0


class SynthesisError(ValueError):
    """Raised when an incident record is too incomplete to synthesize a test."""


def incident_slug(incident: Dict[str, Any]) -> str:
    """Return a stable, filesystem-safe slug identifying the incident.

    The slug combines the test name with a short hash of the query so that:

    - the same failing test name always collapses to the same slug (idempotent)
    - different failure modes of the *same* test name get different slugs
      (so you don't lose an older regression test when a new one is written)

    Used by ``evalview autopr`` to skip already-shipped regression tests.
    """
    test_name = incident.get("test_name", "unknown")
    query = incident.get("query", "")
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "-", test_name).strip("-").lower()
    query_hash = hashlib.sha1(query.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"{safe_name}-{query_hash}"


def _extract_negative_phrases(actual_output: str, baseline_output: str) -> List[str]:
    """Pull short, robust phrases out of the bad output for ``not_contains``.

    Strategy: split actual into sentences, keep only the ones that do NOT
    appear verbatim in the baseline (those are the "newly wrong" parts),
    prefer short ones, cap at ``_MAX_NEGATIVE_PHRASES``. If nothing survives
    (e.g. the outputs are entirely different), fall back to the first short
    sentence of the actual output.
    """
    if not actual_output:
        return []

    # Very permissive sentence split — don't pull in a full tokenizer.
    sentences = re.split(r"(?<=[.!?])\s+", actual_output.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    baseline_lower = (baseline_output or "").lower()
    novel = [s for s in sentences if s.lower() not in baseline_lower]
    candidates = novel if novel else sentences

    # Prefer short phrases — shorter = less brittle as a "must not contain".
    candidates = [s for s in candidates if len(s) <= _MAX_PHRASE_LEN]
    if not candidates and sentences:
        # Still nothing short enough? Take the first sentence and truncate at
        # a word boundary so the phrase stays human-readable.
        head = sentences[0][:_MAX_PHRASE_LEN].rsplit(" ", 1)[0]
        if head:
            candidates = [head]

    # Dedupe while preserving order.
    seen: set[str] = set()
    deduped: List[str] = []
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
        if len(deduped) >= _MAX_NEGATIVE_PHRASES:
            break
    return deduped


def _tool_sequence_diff(
    baseline_tools: Optional[List[str]],
    actual_tools: Optional[List[str]],
) -> Dict[str, List[str]]:
    """Return added/removed tools between baseline and actual traces.

    "Added" tools (present in actual but not baseline) are candidates for
    ``forbidden_tools``: if the agent started calling something it wasn't
    supposed to, we never want to see it called for this query again.
    """
    baseline = list(baseline_tools or [])
    actual = list(actual_tools or [])
    baseline_set = set(baseline)
    actual_set = set(actual)
    return {
        "added": [t for t in actual if t in actual_set - baseline_set],
        "removed": [t for t in baseline if t in baseline_set - actual_set],
    }


def synthesize_regression_test(
    incident: Dict[str, Any],
    min_score: float = _DEFAULT_MIN_SCORE,
) -> Dict[str, Any]:
    """Build a regression test dict from an incident record.

    The returned dict is a valid input for ``TestCase(**result)`` and can be
    dumped to YAML directly. The caller is responsible for writing it to the
    right file and avoiding duplicates — use :func:`incident_slug` for that.

    Raises:
        SynthesisError: if the incident is missing both ``test_name`` and
            ``query``, which are the minimum needed to build a test.
    """
    test_name = (incident.get("test_name") or "").strip()
    query = (incident.get("query") or "").strip()
    if not test_name or not query:
        raise SynthesisError(
            "incident must include non-empty 'test_name' and 'query' fields"
        )

    timestamp = incident.get("timestamp") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    # YYYY-MM-DD prefix for human-readable test names.
    date_prefix = timestamp[:10] if len(timestamp) >= 10 else timestamp

    # TestCase.name only allows [A-Za-z0-9 _\-\.], so we can't use colons or
    # parentheses even though they'd read more naturally. Underscores do the
    # job fine and keep the name greppable.
    regression_name = f"regression_{test_name}_{date_prefix}"
    regression_name = re.sub(r"[^a-zA-Z0-9 _\-\.]", "-", regression_name)

    # Build the description — a short incident report that ends up at the
    # top of the YAML so a human reviewer sees the root cause immediately.
    status = incident.get("status") or "REGRESSION"
    score_delta = incident.get("score_delta")
    score_line = (
        f"Score delta: {score_delta:+.1f}" if isinstance(score_delta, (int, float)) else ""
    )
    model_line = ""
    if incident.get("model_changed"):
        model_line = (
            f"Model changed: {incident.get('golden_model_id', '?')} -> "
            f"{incident.get('actual_model_id', '?')}"
        )
    description_parts = [
        f"Auto-generated from production incident ({status}) at {timestamp}.",
        f"Source test: {incident.get('source_file') or test_name}",
    ]
    if score_line:
        description_parts.append(score_line)
    if model_line:
        description_parts.append(model_line)
    description_parts.append(
        "Synthesized by `evalview autopr` — review the assertions before merging."
    )
    description = "\n".join(description_parts)

    tool_diff = _tool_sequence_diff(
        incident.get("baseline_tools"), incident.get("actual_tools")
    )
    expected: Dict[str, Any] = {}
    if incident.get("baseline_tools"):
        expected["tools"] = list(incident["baseline_tools"])
    if tool_diff["added"]:
        # Tools that appeared in the failing run but not the baseline:
        # forbid them so the agent can never call them again for this query.
        expected["forbidden_tools"] = tool_diff["added"]

    negative_phrases = _extract_negative_phrases(
        incident.get("actual_output") or "",
        incident.get("baseline_output") or "",
    )
    if negative_phrases:
        expected.setdefault("output", {})["not_contains"] = negative_phrases

    # Preserve any positive phrases the baseline cared about. If the caller
    # didn't pass them, we still do the right thing — ``contains`` is optional
    # in ``ExpectedOutput``.
    baseline_contains = incident.get("baseline_contains")
    if baseline_contains:
        expected.setdefault("output", {})["contains"] = list(baseline_contains)

    test: Dict[str, Any] = {
        "name": regression_name,
        "description": description,
        "input": {"query": query},
        "expected": expected,
        "thresholds": {"min_score": float(min_score)},
        # Regression tests are a safety net — classify them so reports
        # treat them differently from capability tests.
        "suite_type": "regression",
        # Strict gate: any single-cycle failure fires immediately. Regression
        # tests should never "wait one cycle" to confirm a flake.
        "gate": "strict",
        "tags": ["incident", "autopr"],
        "meta": {
            "incident": {
                "slug": incident_slug(incident),
                "source_test": test_name,
                "timestamp": timestamp,
                "status": status,
                "score_delta": score_delta,
                "model_changed": bool(incident.get("model_changed")),
                "golden_model_id": incident.get("golden_model_id"),
                "actual_model_id": incident.get("actual_model_id"),
                "added_tools": tool_diff["added"],
                "removed_tools": tool_diff["removed"],
            }
        },
    }
    return test


def truncate_output(text: Optional[str], limit: int = 2000) -> Optional[str]:
    """Trim long outputs before writing them to an incident record.

    Incident files are meant to be checked into ``.evalview/`` and browsed
    by humans. Keeping raw outputs unbounded would make that painful for any
    agent that streams long responses.
    """
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"
