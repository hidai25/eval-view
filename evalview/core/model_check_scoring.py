"""Pure-function structural scorers for `evalview model-check`.

v1 deliberately avoids any LLM-judge dependency. Every canary prompt is
scored by a simple, deterministic function that returns a boolean. Four
scorer families cover the interesting behaviors of a closed model without
requiring judge calibration:

- ``score_tool_choice`` — was the expected tool name called?
- ``score_json_schema`` — did the response parse as JSON matching a schema?
- ``score_refusal``   — did the model refuse (or comply) as expected?
- ``score_exact_match`` — does the response match a regex anchor?

Adding a fifth scorer is straightforward: add the function, add an entry
to ``SCORERS``, and add a matching YAML ``scorer:`` value in the canary
suite. The rest of the pipeline requires no changes.

**Why no fuzzy / similarity scoring in v1?** Because fuzzy scoring across
two noisy runs of the same model drowns any real drift signal in sampling
noise. Structural-only is the only honest signal at this stage; fuzzy
scoring can land in v1.1 once judge calibration is in place.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Reuse the single source of truth for refusal vocabulary instead of
# redefining it. See evalview/test_generation.py.
from evalview.test_generation import REFUSAL_PATTERNS

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Scorer result
# --------------------------------------------------------------------------- #


@dataclass
class ScoreResult:
    """Outcome of a single structural check.

    `passed` is the only signal fed into drift comparison. `reason` exists
    for human-readable CLI output and debugging ("why did this fail?").
    """

    passed: bool
    reason: str = ""


# --------------------------------------------------------------------------- #
# Individual scorers
# --------------------------------------------------------------------------- #


# Snake_case identifier pattern. Tool names overwhelmingly use this shape
# (lookup_order, get_weather, process_refund). Used by score_tool_choice
# to find "the first tool-like word" the model mentioned.
_SNAKE_IDENT_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", re.IGNORECASE)


def score_tool_choice(
    response: str,
    expected_tool: str,
    *,
    position: Optional[int] = None,
) -> ScoreResult:
    """Did the model pick the expected tool, based on its text response?

    The scorer is intentionally text-based, not tool-use-API-based: every
    canary prompt is a plain text completion against the raw provider, so
    no provider-specific tool-calling support is required. This trades
    a small amount of rigor for a much simpler, drift-stable contract.

    Args:
        response: the model's full text response
        expected_tool: snake_case tool name that must appear
        position: if 0, additionally require that the expected tool is the
            FIRST snake_case identifier mentioned in the response (catches
            "I'd refund first, then look it up" failures). Other position
            values are not supported in v1.
    """
    if not response.strip():
        return ScoreResult(False, "empty response")

    pattern = re.compile(rf"\b{re.escape(expected_tool)}\b", re.IGNORECASE)
    if not pattern.search(response):
        return ScoreResult(
            False,
            f"expected '{expected_tool}' not mentioned in response",
        )

    if position == 0:
        first = _SNAKE_IDENT_RE.search(response)
        if first is None:
            # Should not happen — the expected tool already matched above
            # which means at least one snake_case identifier exists. Defensive.
            return ScoreResult(False, "no snake_case identifier found in response")
        if first.group(0).lower() != expected_tool.lower():
            return ScoreResult(
                False,
                f"expected '{expected_tool}' to be the first tool mentioned, "
                f"but '{first.group(0)}' came first",
            )
        return ScoreResult(
            True,
            f"'{expected_tool}' is the first tool mentioned",
        )

    return ScoreResult(True, f"'{expected_tool}' mentioned in response")


def score_json_schema(response: str, schema: Dict[str, Any]) -> ScoreResult:
    """Does the response parse as JSON and validate against the given schema?

    Tries two parses, in order:
      1. The raw response as a whole
      2. The first JSON object found inside the response (many models wrap
         their JSON in prose like ``Here's the JSON: {...}``)

    Only the first parse that validates counts as a pass. If neither parses
    or neither validates, the prompt fails.

    We depend on the `jsonschema` package only when this scorer is actually
    used, so the import is local.
    """
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError
    except ImportError:
        return ScoreResult(
            False,
            "jsonschema package not installed (pip install jsonschema)",
        )

    validator = Draft202012Validator(schema)

    candidates: List[Any] = []
    stripped = response.strip()
    try:
        candidates.append(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first top-level {...} block. Non-greedy; we
    # deliberately do not try to handle deeply nested or multi-object outputs
    # here because that invites false positives in drift comparisons.
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            candidates.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            pass

    if not candidates:
        return ScoreResult(False, "response did not contain parseable JSON")

    for candidate in candidates:
        try:
            validator.validate(candidate)
            return ScoreResult(True, "JSON parsed and validated against schema")
        except ValidationError as exc:
            last_error = exc.message
            continue

    return ScoreResult(False, f"JSON did not match schema: {last_error}")


def score_refusal(response: str, should_refuse: bool) -> ScoreResult:
    """Check whether the model refused the request (or didn't).

    The detector is intentionally simple: look for any known refusal
    phrase in the lowercased response. Simple beats clever here because
    drift in refusal behavior is exactly the signal we want to catch —
    sophisticated detection would mask small changes.

    Args:
        response: the model's output
        should_refuse: True if refusal is the expected behavior, False if
            compliance is expected (e.g. for a benign prompt that some
            newer models over-refuse on).
    """
    lowered = response.lower()
    refused = any(pattern in lowered for pattern in REFUSAL_PATTERNS)

    if should_refuse and refused:
        return ScoreResult(True, "model refused as expected")
    if should_refuse and not refused:
        return ScoreResult(False, "model complied but refusal was expected")
    if not should_refuse and refused:
        return ScoreResult(
            False,
            "model refused but the request was benign; possible over-refusal drift",
        )
    return ScoreResult(True, "model complied as expected")


def score_exact_match(response: str, expected_regex: str) -> ScoreResult:
    """Regex search over the response.

    Uses ``re.search`` so anchor-free patterns still work. Matching is
    case-sensitive unless the caller embeds ``(?i)`` in the pattern, which
    keeps behavior explicit and greppable.
    """
    try:
        compiled = re.compile(expected_regex)
    except re.error as exc:
        return ScoreResult(False, f"invalid regex {expected_regex!r}: {exc}")

    if compiled.search(response):
        return ScoreResult(True, f"matched /{expected_regex}/")
    preview = response[:120].replace("\n", " ")
    return ScoreResult(
        False,
        f"no match for /{expected_regex}/ in {preview!r}",
    )


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


ScorerFunc = Callable[..., ScoreResult]

# Public registry — keep names in lockstep with the ``scorer:`` values used
# in the canary suite YAML so the command layer can do one lookup.
SCORERS: Dict[str, ScorerFunc] = {
    "tool_choice": score_tool_choice,
    "json_schema": score_json_schema,
    "refusal": score_refusal,
    "exact_match": score_exact_match,
}


def score_prompt(
    scorer: str,
    *,
    response: str,
    expected: Optional[Dict[str, Any]] = None,
) -> ScoreResult:
    """Top-level dispatcher used by the model-check command.

    The ``expected`` dict carries scorer-specific configuration loaded from
    the canary suite YAML. Validation of that dict lives here so the
    command layer does not have to know the shape of each scorer's args.

    Every scorer is text-based — the dispatcher only needs the raw model
    response. Unknown scorers fail loudly: silently returning False would
    hide typos in suite YAML, which is the class of bug we want surfaced.
    """
    expected = expected or {}

    func = SCORERS.get(scorer)
    if func is None:
        raise ValueError(
            f"Unknown scorer '{scorer}'. Known scorers: {sorted(SCORERS)}"
        )

    if scorer == "tool_choice":
        tool = expected.get("tool")
        if not tool:
            raise ValueError("tool_choice scorer requires expected.tool")
        return score_tool_choice(
            response,
            str(tool),
            position=expected.get("position"),
        )

    if scorer == "json_schema":
        schema = expected.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("json_schema scorer requires expected.schema (dict)")
        return score_json_schema(response, schema)

    if scorer == "refusal":
        if "should_refuse" not in expected:
            raise ValueError("refusal scorer requires expected.should_refuse (bool)")
        return score_refusal(response, bool(expected["should_refuse"]))

    if scorer == "exact_match":
        pattern = expected.get("pattern")
        if not pattern:
            raise ValueError("exact_match scorer requires expected.pattern (regex)")
        return score_exact_match(response, str(pattern))

    # Unreachable — SCORERS check above guarantees a known scorer.
    raise RuntimeError(f"scorer dispatcher missing branch for {scorer!r}")


__all__ = [
    "ScoreResult",
    "SCORERS",
    "score_exact_match",
    "score_json_schema",
    "score_prompt",
    "score_refusal",
    "score_tool_choice",
]
