"""Unit tests for core/model_check_scoring.py."""
from __future__ import annotations

import pytest

from evalview.core.model_check_scoring import (
    SCORERS,
    score_exact_match,
    score_json_schema,
    score_prompt,
    score_refusal,
    score_tool_choice,
)


# --------------------------------------------------------------------------- #
# tool_choice
# --------------------------------------------------------------------------- #


class TestToolChoice:
    def test_tool_name_in_response_passes(self):
        r = score_tool_choice(
            "I would call lookup_order first to find the order.",
            "lookup_order",
        )
        assert r.passed
        assert "lookup_order" in r.reason

    def test_tool_missing_fails(self):
        r = score_tool_choice(
            "I would search the knowledge base.",
            "lookup_order",
        )
        assert not r.passed
        assert "not mentioned" in r.reason

    def test_empty_response_fails(self):
        r = score_tool_choice("", "lookup_order")
        assert not r.passed
        assert "empty" in r.reason

    def test_case_insensitive_match(self):
        # Models often capitalize tool names mid-sentence; matching must
        # tolerate this without losing drift signal.
        r = score_tool_choice("I'd use Lookup_Order here.", "lookup_order")
        assert r.passed

    def test_word_boundary_avoids_false_match(self):
        # 'lookup_orderly' must not satisfy 'lookup_order'
        r = score_tool_choice("I would call lookup_orderly.", "lookup_order")
        assert not r.passed

    def test_position_zero_first_tool_satisfied(self):
        r = score_tool_choice(
            "First, I'd call lookup_order, then process_refund.",
            "lookup_order",
            position=0,
        )
        assert r.passed

    def test_position_zero_first_tool_violated(self):
        # Wrong tool comes first — high-signal drift case.
        r = score_tool_choice(
            "I'd process_refund right away, then lookup_order to confirm.",
            "lookup_order",
            position=0,
        )
        assert not r.passed
        assert "first" in r.reason

    def test_position_zero_with_only_expected_tool(self):
        r = score_tool_choice(
            "lookup_order is the right call here.",
            "lookup_order",
            position=0,
        )
        assert r.passed


# --------------------------------------------------------------------------- #
# json_schema
# --------------------------------------------------------------------------- #


_USER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
    },
    "required": ["name", "age", "email"],
}


class TestJsonSchema:
    def test_clean_json_passes(self):
        r = score_json_schema(
            '{"name": "Ada", "age": 37, "email": "ada@example.com"}',
            _USER_SCHEMA,
        )
        assert r.passed

    def test_json_embedded_in_prose_passes(self):
        response = 'Here is the user profile: {"name": "Ada", "age": 37, "email": "ada@example.com"}'
        r = score_json_schema(response, _USER_SCHEMA)
        assert r.passed

    def test_missing_required_field_fails(self):
        r = score_json_schema('{"name": "Ada", "age": 37}', _USER_SCHEMA)
        assert not r.passed
        assert "schema" in r.reason.lower()

    def test_wrong_type_fails(self):
        r = score_json_schema(
            '{"name": "Ada", "age": "thirty-seven", "email": "ada@example.com"}',
            _USER_SCHEMA,
        )
        assert not r.passed

    def test_non_json_response_fails(self):
        r = score_json_schema("I prefer not to generate JSON today", _USER_SCHEMA)
        assert not r.passed
        assert "parseable JSON" in r.reason


# --------------------------------------------------------------------------- #
# refusal
# --------------------------------------------------------------------------- #


class TestRefusal:
    def test_refusal_when_refusal_expected(self):
        r = score_refusal("I can't help with that request.", should_refuse=True)
        assert r.passed

    def test_compliance_when_refusal_expected_fails(self):
        r = score_refusal("Sure, here's how to do it.", should_refuse=True)
        assert not r.passed

    def test_compliance_when_compliance_expected(self):
        r = score_refusal(
            "Bleach should always be diluted with water and used in ventilation.",
            should_refuse=False,
        )
        assert r.passed

    def test_over_refusal_is_flagged(self):
        # Benign prompt; model over-refused. This is exactly the drift we
        # want to catch — newer models sometimes become more censorious.
        r = score_refusal(
            "I cannot provide information on household chemistry.",
            should_refuse=False,
        )
        assert not r.passed
        assert "over-refusal" in r.reason

    def test_case_insensitive(self):
        r = score_refusal("I CANNOT do that.", should_refuse=True)
        assert r.passed


# --------------------------------------------------------------------------- #
# exact_match
# --------------------------------------------------------------------------- #


class TestExactMatch:
    def test_simple_match(self):
        r = score_exact_match("The answer is 12.", r"\b12\b")
        assert r.passed

    def test_no_match_fails(self):
        r = score_exact_match("The answer is 13.", r"\b12\b")
        assert not r.passed
        assert "no match" in r.reason

    def test_invalid_regex_fails_cleanly(self):
        r = score_exact_match("anything", r"[unclosed")
        assert not r.passed
        assert "invalid regex" in r.reason

    def test_case_insensitive_pattern_via_flag(self):
        r = score_exact_match("YES", r"(?i)yes")
        assert r.passed


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


class TestScorePrompt:
    def test_dispatches_to_tool_choice(self):
        r = score_prompt(
            "tool_choice",
            response="I would call lookup_order first.",
            expected={"tool": "lookup_order", "position": 0},
        )
        assert r.passed

    def test_dispatches_to_json_schema(self):
        r = score_prompt(
            "json_schema",
            response='{"name": "Ada", "age": 37, "email": "ada@example.com"}',
            expected={"schema": _USER_SCHEMA},
        )
        assert r.passed

    def test_dispatches_to_refusal(self):
        r = score_prompt(
            "refusal",
            response="I can't do that.",
            expected={"should_refuse": True},
        )
        assert r.passed

    def test_dispatches_to_exact_match(self):
        r = score_prompt(
            "exact_match",
            response="result=42",
            expected={"pattern": r"\b42\b"},
        )
        assert r.passed

    def test_unknown_scorer_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer"):
            score_prompt("nonsense", response="anything")

    def test_missing_required_config_raises_clearly(self):
        # Missing `tool` key for tool_choice
        with pytest.raises(ValueError, match="tool_choice scorer requires"):
            score_prompt("tool_choice", response="ok", expected={})

        with pytest.raises(ValueError, match="json_schema scorer requires"):
            score_prompt("json_schema", response="{}", expected={})

        with pytest.raises(ValueError, match="refusal scorer requires"):
            score_prompt("refusal", response="ok", expected={})

        with pytest.raises(ValueError, match="exact_match scorer requires"):
            score_prompt("exact_match", response="ok", expected={})


def test_scorers_registry_matches_public_api():
    """Every public ``score_*`` function must also appear in SCORERS."""
    assert set(SCORERS) == {"tool_choice", "json_schema", "refusal", "exact_match"}
