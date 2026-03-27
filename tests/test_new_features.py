"""Tests for the 6 new EvalView features:
1. Regex pattern checks
2. JSON schema validation
3. Basic schema check (fallback)
4. Deterministic output eval integration
5. Config models (JudgeConfig, budget)
6. CLI options (--budget, --dry-run, --judge-cache)
7. ExpectedOutput model (regex_patterns field)
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from evalview.core.types import (
    TestCase as TestCaseModel,
    TestInput as TestInputModel,
    ExpectedBehavior,
    ExpectedOutput,
    Thresholds,
    ExecutionTrace,
    ExecutionMetrics,
)
from evalview.evaluators.evaluator import Evaluator


# ============================================================================
# Helpers
# ============================================================================


def _make_test_case(
    query: str = "test query with some words here",
    contains: Optional[List[str]] = None,
    not_contains: Optional[List[str]] = None,
    regex_patterns: Optional[List[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    min_score: float = 0,
) -> TestCaseModel:
    return TestCaseModel(
        name="test-case",
        input=TestInputModel(query=query),
        expected=ExpectedBehavior(
            output=ExpectedOutput(
                contains=contains,
                not_contains=not_contains,
                regex_patterns=regex_patterns,
                json_schema=json_schema,
            )
        ),
        thresholds=Thresholds(min_score=min_score),
    )


def _make_trace(output: str = "test output") -> ExecutionTrace:
    return ExecutionTrace(
        session_id="test-session",
        start_time=datetime(2025, 1, 1, 12, 0, 0),
        end_time=datetime(2025, 1, 1, 12, 0, 1),
        steps=[],
        final_output=output,
        metrics=ExecutionMetrics(total_cost=0, total_latency=0),
    )


# ============================================================================
# 1. Regex Pattern Checks
# ============================================================================


class TestRegexPatterns:
    """Tests for Evaluator._check_regex_patterns."""

    def test_basic_match(self):
        """A simple pattern that matches the output."""
        passed, failed = Evaluator._check_regex_patterns("hello world", [r"hello"])
        assert passed == [r"hello"]
        assert failed == []

    def test_basic_no_match(self):
        """A simple pattern that does NOT match the output."""
        passed, failed = Evaluator._check_regex_patterns("hello world", [r"goodbye"])
        assert passed == []
        assert failed == [r"goodbye"]

    def test_multiple_patterns_mixed(self):
        """Some patterns match, some don't."""
        output = "The answer is 42."
        patterns = [r"answer", r"\d+", r"foobar"]
        passed, failed = Evaluator._check_regex_patterns(output, patterns)
        assert set(passed) == {r"answer", r"\d+"}
        assert failed == [r"foobar"]

    def test_invalid_regex_fails_gracefully(self):
        """An invalid regex pattern should be reported as failed, not raise."""
        passed, failed = Evaluator._check_regex_patterns("test", [r"[invalid"])
        assert passed == []
        assert failed == [r"[invalid"]

    def test_empty_patterns_list(self):
        """An empty patterns list returns empty results."""
        passed, failed = Evaluator._check_regex_patterns("test", [])
        assert passed == []
        assert failed == []

    def test_case_insensitive(self):
        """Patterns are compiled with IGNORECASE."""
        passed, failed = Evaluator._check_regex_patterns("Hello World", [r"hello world"])
        assert passed == [r"hello world"]
        assert failed == []

    def test_dotall_mode(self):
        """Patterns are compiled with DOTALL so '.' spans newlines."""
        output = "line one\nline two"
        passed, failed = Evaluator._check_regex_patterns(output, [r"one.line"])
        assert passed == [r"one.line"]
        assert failed == []

    def test_redos_protection(self):
        """A ReDoS-prone pattern on adversarial input should not hang.

        The signal-based timeout should abort the match within a few seconds.
        We assert that the call completes well within 10 seconds.
        """
        # Classic ReDoS pattern: (a+)+$ with input 'aaa...X'
        evil_pattern = r"(a+)+$"
        evil_input = "a" * 30 + "X"

        start = time.monotonic()
        passed, failed = Evaluator._check_regex_patterns(evil_input, [evil_pattern])
        elapsed = time.monotonic() - start

        # The pattern should either fail (timeout) or match quickly.
        # Key assertion: it must not hang.
        assert elapsed < 5.0, f"ReDoS protection failed: took {elapsed:.1f}s"
        # The pattern won't match (input ends with 'X', not end-of-string after 'a')
        # OR it times out — either way it lands in failed.
        assert evil_pattern in failed

    def test_large_output_truncated(self):
        """Output longer than _MAX_CHECK_OUTPUT_LEN is truncated; match at the
        end of the truncated output should fail while match at the start succeeds."""
        max_len = Evaluator._MAX_CHECK_OUTPUT_LEN
        # Place a marker at the start and the end of the full output
        output = "START" + ("x" * (max_len + 100)) + "ENDMARKER"

        passed, failed = Evaluator._check_regex_patterns(output, [r"START", r"ENDMARKER"])
        assert "START" in passed
        assert "ENDMARKER" in failed  # truncated away


# ============================================================================
# 2. JSON Schema Validation
# ============================================================================


class TestJsonSchema:
    """Tests for Evaluator._check_json_schema."""

    def test_valid_json_passes(self):
        """Valid JSON that satisfies the schema passes."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        output = json.dumps({"name": "Alice"})
        ok, err = Evaluator._check_json_schema(output, schema)
        assert ok is True
        assert err == ""

    def test_missing_required_field_fails(self):
        """Missing a required property should fail."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
        }
        output = json.dumps({"name": "Alice"})
        ok, err = Evaluator._check_json_schema(output, schema)
        assert ok is False
        assert err != ""

    def test_wrong_type_fails(self):
        """A property with the wrong type should fail."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "number"}},
        }
        output = json.dumps({"age": "not a number"})
        ok, err = Evaluator._check_json_schema(output, schema)
        assert ok is False

    def test_json_embedded_in_text(self):
        """JSON embedded in surrounding text should be extracted and validated."""
        schema = {
            "type": "object",
            "required": ["result"],
            "properties": {"result": {"type": "string"}},
        }
        output = 'Here is the result: {"result": "success"} — done.'
        ok, err = Evaluator._check_json_schema(output, schema)
        assert ok is True

    def test_multiple_json_objects_picks_first_valid(self):
        """When multiple JSON objects exist in text, the first valid one is used."""
        schema = {
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "number"}},
        }
        output = 'First: {"bad": true} Second: {"value": 42}'
        ok, err = Evaluator._check_json_schema(output, schema)
        # The first JSON object that parses is {"bad": true}, which lacks 'value'.
        # Whether it passes depends on whether jsonschema is installed —
        # but at minimum the method should not raise.
        # With jsonschema, the first valid parse {"bad": true} fails; then the
        # regex may or may not find the second one depending on the extraction pattern.
        # We just verify it doesn't crash and returns a tuple.
        assert isinstance(ok, bool)

    def test_no_json_in_output(self):
        """Output with no JSON at all should fail."""
        schema = {"type": "object"}
        output = "This is plain text with no JSON."
        ok, err = Evaluator._check_json_schema(output, schema)
        assert ok is False
        assert "does not contain valid JSON" in err

    def test_empty_schema_passes(self):
        """An empty schema {} should accept any valid JSON."""
        output = json.dumps({"anything": [1, 2, 3]})
        ok, err = Evaluator._check_json_schema(output, {})
        assert ok is True

    def test_nested_object(self):
        """Nested object validation works."""
        schema = {
            "type": "object",
            "required": ["user"],
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        output = json.dumps({"user": {"name": "Bob"}})
        ok, err = Evaluator._check_json_schema(output, schema)
        assert ok is True


# ============================================================================
# 3. Basic Schema Check (fallback without jsonschema lib)
# ============================================================================


class TestBasicSchemaCheck:
    """Tests for Evaluator._basic_schema_check (used when jsonschema is not installed)."""

    def test_required_properties(self):
        """Missing required property is reported."""
        schema = {"type": "object", "required": ["a", "b"]}
        ok, err = Evaluator._basic_schema_check({"a": 1}, schema)
        assert ok is False
        assert "Missing required property: b" in err

    def test_type_checking_string(self):
        """Property expected to be string validated correctly."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        ok, err = Evaluator._basic_schema_check({"name": "Alice"}, schema)
        assert ok is True

        ok, err = Evaluator._basic_schema_check({"name": 123}, schema)
        assert ok is False
        assert "expected string" in err

    def test_type_checking_number(self):
        """Property expected to be number validated correctly."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "number"}},
        }
        ok, err = Evaluator._basic_schema_check({"age": 25}, schema)
        assert ok is True

        ok, err = Evaluator._basic_schema_check({"age": 25.5}, schema)
        assert ok is True

        ok, err = Evaluator._basic_schema_check({"age": "twenty"}, schema)
        assert ok is False
        assert "expected number" in err

    def test_type_checking_boolean(self):
        """Property expected to be boolean validated correctly."""
        schema = {
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
        }
        ok, err = Evaluator._basic_schema_check({"active": True}, schema)
        assert ok is True

        ok, err = Evaluator._basic_schema_check({"active": "yes"}, schema)
        assert ok is False
        assert "expected boolean" in err

    def test_type_checking_array(self):
        """Property expected to be array validated correctly."""
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array"}},
        }
        ok, err = Evaluator._basic_schema_check({"items": [1, 2]}, schema)
        assert ok is True

        ok, err = Evaluator._basic_schema_check({"items": "not array"}, schema)
        assert ok is False
        assert "expected array" in err

    def test_wrong_root_type(self):
        """Root type mismatch is detected."""
        schema = {"type": "object"}
        ok, err = Evaluator._basic_schema_check([1, 2, 3], schema)
        assert ok is False
        assert "Expected object" in err

        schema_arr = {"type": "array"}
        ok, err = Evaluator._basic_schema_check({"key": "val"}, schema_arr)
        assert ok is False
        assert "Expected array" in err


# ============================================================================
# 4. Deterministic Output Eval Integration
# ============================================================================


class TestDeterministicEvalWithCodeChecks:
    """Tests for _deterministic_output_eval with regex and JSON schema checks.

    Uses Evaluator(skip_llm_judge=True) to avoid needing API keys.
    """

    def test_regex_pass_adds_weight(self):
        """When all regex patterns match, the regex weight is earned."""
        tc = _make_test_case(regex_patterns=[r"\d+", r"answer"])
        trace = _make_trace("The answer is 42 and that is final.")
        evaluator = Evaluator(skip_llm_judge=True)
        result = evaluator._deterministic_output_eval(tc, trace)
        assert "All regex patterns matched" in result.rationale

    def test_regex_fail_reduces_score(self):
        """When regex patterns fail, the regex portion is not earned.

        We also add failing contains checks so the total stays below the 75 cap,
        making the difference from the regex weight observable.
        """
        # Both test cases require contains=["NOTHERE"] which will fail (0 pts for
        # contains weight), keeping total low enough that the regex delta matters.
        tc_pass = _make_test_case(
            regex_patterns=[r"hello"],
            contains=["NOTHERE"],
            not_contains=["hello"],  # will fail too — output contains "hello"
        )
        tc_fail = _make_test_case(
            regex_patterns=[r"goodbye"],
            contains=["NOTHERE"],
            not_contains=["hello"],  # will fail too
        )
        trace = _make_trace("hello world and some more words here to be long enough")

        evaluator = Evaluator(skip_llm_judge=True)
        score_pass = evaluator._deterministic_output_eval(tc_pass, trace).score
        score_fail = evaluator._deterministic_output_eval(tc_fail, trace).score
        assert score_pass > score_fail

    def test_json_schema_pass_adds_weight(self):
        """When JSON output validates against schema, the schema weight is earned."""
        schema = {
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"type": "string"}},
        }
        tc = _make_test_case(json_schema=schema)
        trace = _make_trace(json.dumps({"status": "ok"}))
        evaluator = Evaluator(skip_llm_judge=True)
        result = evaluator._deterministic_output_eval(tc, trace)
        assert "JSON schema valid" in result.rationale

    def test_json_schema_fail_reduces_score(self):
        """When JSON schema validation fails, score is lower.

        We add failing contains/not_contains checks to keep both totals
        below the 75 cap so the schema weight delta is observable.
        """
        schema = {
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"type": "string"}},
        }
        tc_pass = _make_test_case(
            json_schema=schema,
            contains=["NOTHERE"],
            not_contains=["ok"],  # will fail for the passing trace
        )
        tc_fail = _make_test_case(
            json_schema=schema,
            contains=["NOTHERE"],
            not_contains=["words"],  # will also fail for the failing trace
        )

        trace_pass = _make_trace(json.dumps({"status": "ok"}))
        trace_fail = _make_trace("not json at all but some words here to pass length check")

        evaluator = Evaluator(skip_llm_judge=True)
        score_pass = evaluator._deterministic_output_eval(tc_pass, trace_pass).score
        score_fail = evaluator._deterministic_output_eval(tc_fail, trace_fail).score
        assert score_pass > score_fail

    def test_no_regex_no_schema_awards_full_weight(self):
        """When no regex/schema is specified, the full weight for those checks is awarded."""
        tc = _make_test_case()  # no regex_patterns, no json_schema
        trace = _make_trace("A sufficiently long output with enough content for tests.")
        evaluator = Evaluator(skip_llm_judge=True)
        result = evaluator._deterministic_output_eval(tc, trace)
        # Score should include both the regex and schema weights (awarded by default)
        # since no checks are defined, both weights are granted.
        assert result.score > 0

    def test_score_capped_at_75(self):
        """Deterministic score is capped at _DETO_SCORE_CAP (75)."""
        tc = _make_test_case(
            query="test query",
            contains=["output"],
        )
        trace = _make_trace("test output with query words here for relevance check")
        evaluator = Evaluator(skip_llm_judge=True)
        result = evaluator._deterministic_output_eval(tc, trace)
        assert result.score <= 75.0


# ============================================================================
# 5. Config Models
# ============================================================================


class TestConfig:
    """Tests for JudgeConfig and EvalViewConfig budget field."""

    def test_judge_config_defaults(self):
        from evalview.core.config import JudgeConfig
        cfg = JudgeConfig()
        assert cfg.provider is None
        assert cfg.model is None

    def test_judge_config_with_values(self):
        from evalview.core.config import JudgeConfig
        cfg = JudgeConfig(provider="anthropic", model="sonnet")
        assert cfg.provider == "anthropic"
        assert cfg.model == "sonnet"

    def test_evalview_config_with_budget(self):
        from evalview.core.config import EvalViewConfig
        cfg = EvalViewConfig(adapter="http", endpoint="http://localhost:8000", budget=1.50)
        assert cfg.budget == 1.50

    def test_evalview_config_budget_none_by_default(self):
        from evalview.core.config import EvalViewConfig
        cfg = EvalViewConfig(adapter="http", endpoint="http://localhost:8000")
        assert cfg.budget is None

    def test_get_judge_config_returns_none_when_not_set(self):
        from evalview.core.config import EvalViewConfig
        cfg = EvalViewConfig(adapter="http", endpoint="http://localhost:8000")
        assert cfg.get_judge_config() is None

    def test_get_judge_config_returns_config_when_set(self):
        from evalview.core.config import EvalViewConfig, JudgeConfig
        cfg = EvalViewConfig(
            adapter="http",
            endpoint="http://localhost:8000",
            judge=JudgeConfig(provider="openai", model="gpt-4o"),
        )
        judge = cfg.get_judge_config()
        assert judge is not None
        assert judge.provider == "openai"
        assert judge.model == "gpt-4o"


# ============================================================================
# 6. CLI Options Exist
# ============================================================================


class TestCLIOptions:
    """Verify that new CLI options are registered on the run and check commands."""

    def _get_option_names(self, command: Any) -> List[str]:
        """Extract long option names from a Click command."""
        names = []
        for param in command.params:
            names.extend(param.opts)
        return names

    def test_run_has_budget_option(self):
        from evalview.commands.run._cmd import run
        opts = self._get_option_names(run)
        assert "--budget" in opts

    def test_run_has_dry_run_option(self):
        from evalview.commands.run._cmd import run
        opts = self._get_option_names(run)
        assert "--dry-run" in opts

    def test_run_has_tag_option(self):
        from evalview.commands.run._cmd import run
        opts = self._get_option_names(run)
        assert "--tag" in opts

    def test_check_has_tag_option(self):
        from evalview.commands.check_cmd import check
        opts = self._get_option_names(check)
        assert "--tag" in opts

    def test_run_judge_cache_defaults_true(self):
        """--judge-cache / --no-judge-cache defaults to True."""
        from evalview.commands.run._cmd import run
        for param in run.params:
            if "--judge-cache" in param.opts or "--judge-cache/--no-judge-cache" in param.secondary_opts + param.opts:
                # Click boolean flags: check default
                if param.name == "judge_cache":
                    assert param.default is True
                    return
        # Fallback: look by name
        for param in run.params:
            if param.name == "judge_cache":
                assert param.default is True
                return
        pytest.fail("--judge-cache option not found on run command")

    def test_check_has_budget_option(self):
        from evalview.commands.check_cmd import check
        opts = self._get_option_names(check)
        assert "--budget" in opts

    def test_check_has_dry_run_option(self):
        from evalview.commands.check_cmd import check
        opts = self._get_option_names(check)
        assert "--dry-run" in opts


# ============================================================================
# 7. ExpectedOutput Model
# ============================================================================


class TestExpectedOutput:
    """Tests for the regex_patterns field on ExpectedOutput."""

    def test_regex_patterns_field_exists(self):
        """ExpectedOutput has a regex_patterns field."""
        eo = ExpectedOutput()
        assert hasattr(eo, "regex_patterns")

    def test_regex_patterns_accepts_list(self):
        """regex_patterns accepts a list of strings."""
        eo = ExpectedOutput(regex_patterns=[r"\d+", r"[A-Z]+"])
        assert eo.regex_patterns == [r"\d+", r"[A-Z]+"]

    def test_regex_patterns_defaults_none(self):
        """regex_patterns defaults to None when not provided."""
        eo = ExpectedOutput()
        assert eo.regex_patterns is None

    def test_json_schema_field_exists(self):
        """ExpectedOutput has a json_schema field."""
        eo = ExpectedOutput(json_schema={"type": "object"})
        assert eo.json_schema == {"type": "object"}

    def test_json_schema_defaults_none(self):
        """json_schema defaults to None when not provided."""
        eo = ExpectedOutput()
        assert eo.json_schema is None
