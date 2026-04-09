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
    def test_exact_tool_in_sequence_passes(self):
        r = score_tool_choice(["lookup_order", "check_policy"], "lookup_order")
        assert r.passed
        assert "lookup_order" in r.reason

    def test_tool_missing_fails(self):
        r = score_tool_choice(["search", "summarize"], "lookup_order")
        assert not r.passed
        assert "expected 'lookup_order'" in r.reason

    def test_empty_tool_list_fails(self):
        r = score_tool_choice([], "lookup_order")
        assert not r.passed

    def test_case_sensitive(self):
        # Tool names are identifiers — loose matching would hide drift.
        r = score_tool_choice(["LookupOrder"], "lookup_order")
        assert not r.passed

    def test_position_constraint_satisfied(self):
        r = score_tool_choice(
            ["lookup_order", "process_refund"],
            "lookup_order",
            position=0,
        )
        assert r.passed

    def test_position_constraint_violated(self):
        r = score_tool_choice(
            ["process_refund", "lookup_order"],
            "lookup_order",
            position=0,
        )
        assert not r.passed
        assert "position 0" in r.reason

    def test_position_out_of_range(self):
        r = score_tool_choice(["lookup_order"], "lookup_order", position=3)
        assert not r.passed


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
            response="",
            tool_calls=["lookup_order"],
            expected={"tool": "lookup_order"},
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
            score_prompt("tool_choice", response="", tool_calls=["x"], expected={})

        with pytest.raises(ValueError, match="json_schema scorer requires"):
            score_prompt("json_schema", response="{}", expected={})

        with pytest.raises(ValueError, match="refusal scorer requires"):
            score_prompt("refusal", response="ok", expected={})

        with pytest.raises(ValueError, match="exact_match scorer requires"):
            score_prompt("exact_match", response="ok", expected={})


def test_scorers_registry_matches_public_api():
    """Every public ``score_*`` function must also appear in SCORERS."""
    assert set(SCORERS) == {"tool_choice", "json_schema", "refusal", "exact_match"}
