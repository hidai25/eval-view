"""Main evaluator orchestrator."""

import json as _json
import logging
import re as _re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Deterministic output scoring weights (must sum to 100)
# ---------------------------------------------------------------------------
# These control the fallback scorer used when no LLM judge is available.
# Adjust here to rebalance; the cap prevents misleadingly high scores
# since deterministic heuristics are inherently approximate.
_DETO_CONTAINS_WEIGHT: float = 30.0      # % score for contains checks
_DETO_NOT_CONTAINS_WEIGHT: float = 15.0  # % score for not_contains checks
_DETO_LENGTH_WEIGHT: float = 15.0        # % score for non-empty output
_DETO_RELEVANCE_WEIGHT: float = 15.0     # % score for query-term overlap
_DETO_REGEX_WEIGHT: float = 15.0         # % score for regex pattern checks
_DETO_JSON_SCHEMA_WEIGHT: float = 10.0   # % score for JSON schema validation
_DETO_SCORE_CAP: float = 75.0           # max score; signals "approximate"
_DETO_MIN_OUTPUT_LENGTH: int = 10        # chars below which output is "short"
_DETO_MIN_QUERY_WORD_LENGTH: int = 3     # minimum chars to treat as a keyword

from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    EvaluationResult,
    Evaluations,
    OutputEvaluation,
    ContainsChecks,
)
from evalview.core.config import ScoringWeights, DEFAULT_WEIGHTS
from evalview.evaluators.tool_call_evaluator import ToolCallEvaluator
from evalview.evaluators.sequence_evaluator import SequenceEvaluator
from evalview.evaluators.output_evaluator import OutputEvaluator
from evalview.evaluators.cost_evaluator import CostEvaluator
from evalview.evaluators.latency_evaluator import LatencyEvaluator
from evalview.evaluators.hallucination_evaluator import HallucinationEvaluator
from evalview.evaluators.safety_evaluator import SafetyEvaluator
from evalview.evaluators.pii_evaluator import PIIEvaluator

if TYPE_CHECKING:
    from evalview.core.judge_cache import JudgeCache


class _RegexTimeoutError(Exception):
    """Raised when a regex pattern match exceeds the time limit (ReDoS protection)."""
    pass

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluator that orchestrates all evaluation components.

    Supports multiple LLM providers for evaluation: OpenAI, Anthropic, Gemini, and Grok.
    Auto-detects available providers based on API keys in environment.
    """

    def __init__(
        self,
        default_weights: Optional[ScoringWeights] = None,
        skip_llm_judge: bool = False,
        judge_cache: Optional["JudgeCache"] = None,
    ):
        """
        Initialize evaluator.

        Args:
            default_weights: Default scoring weights (can be overridden per test case)
            skip_llm_judge: If True, skip LLM-as-judge and use deterministic scoring.
                           Useful when no API key is available.
            judge_cache: Optional JudgeCache instance for caching LLM judge results.
                        Most useful in statistical mode (--runs) to avoid redundant calls.

        Note:
            LLM provider for evaluation is auto-detected from environment variables.
            Set EVAL_PROVIDER to specify a provider, or EVAL_MODEL to specify a model.
        """
        self.tool_evaluator = ToolCallEvaluator()
        self.sequence_evaluator = SequenceEvaluator()
        self.cost_evaluator = CostEvaluator()
        self.latency_evaluator = LatencyEvaluator()
        self.pii_evaluator = PIIEvaluator()
        self.default_weights = default_weights or DEFAULT_WEIGHTS
        self.skip_llm_judge = skip_llm_judge
        self.judge_cache = judge_cache
        self._logged_deterministic_mode = False

        # Only initialize LLM-dependent evaluators when needed.
        # If no API key is configured, fall back to deterministic mode automatically
        # so commands like `evalview check` work without requiring a key.
        if not skip_llm_judge:
            try:
                self.output_evaluator = OutputEvaluator(cache=judge_cache)
                self.hallucination_evaluator = HallucinationEvaluator()
                self.safety_evaluator = SafetyEvaluator()
            except ValueError:
                # No LLM provider API key found — degrade gracefully to deterministic mode.
                logger.debug("No LLM provider API key found; falling back to deterministic scoring.")
                self.skip_llm_judge = True
                self.output_evaluator = None
                self.hallucination_evaluator = None
                self.safety_evaluator = None
        else:
            self.output_evaluator = None
            self.hallucination_evaluator = None
            self.safety_evaluator = None

    async def evaluate(
        self, test_case: TestCase, trace: ExecutionTrace, adapter_name: Optional[str] = None
    ) -> EvaluationResult:
        """
        Run complete evaluation on a test case.

        Args:
            test_case: Test case with expected behavior
            trace: Execution trace from agent
            adapter_name: Name of the adapter used (e.g., "langgraph", "crewai")

        Returns:
            Complete evaluation result
        """
        # Check which evaluations to run based on test case config
        run_hallucination = test_case.checks.hallucination if test_case.checks else True
        run_safety = test_case.checks.safety if test_case.checks else True
        run_pii = test_case.checks.pii if test_case.checks else False

        # Skip LLM evaluations if skip_llm_judge is set
        if self.skip_llm_judge:
            if not self._logged_deterministic_mode:
                logger.info("Running in deterministic mode (no LLM judge) - scores capped at 75")
                self._logged_deterministic_mode = True
            run_hallucination = False
            run_safety = False
            output_quality = self._deterministic_output_eval(test_case, trace)
        else:
            output_quality = await self.output_evaluator.evaluate(test_case, trace)

        # Run all evaluations
        evaluations = Evaluations(
            tool_accuracy=self.tool_evaluator.evaluate(test_case, trace),
            sequence_correctness=self.sequence_evaluator.evaluate(test_case, trace),
            output_quality=output_quality,
            cost=self.cost_evaluator.evaluate(test_case, trace),
            latency=self.latency_evaluator.evaluate(test_case, trace),
            hallucination=await self.hallucination_evaluator.evaluate(test_case, trace) if run_hallucination else None,
            safety=await self.safety_evaluator.evaluate(test_case, trace) if run_safety else None,
            forbidden_tools=self.tool_evaluator.evaluate_forbidden(test_case, trace),
            pii=await self.pii_evaluator.evaluate(test_case, trace) if run_pii else None,
        )

        # Compute overall score
        score = self._compute_overall_score(evaluations, test_case)

        # Determine pass/fail
        passed = self._compute_pass_fail(evaluations, test_case, score)

        return EvaluationResult(
            test_case=test_case.name,
            passed=passed,
            score=score,
            evaluations=evaluations,
            trace=trace,
            timestamp=datetime.now(),
            adapter_name=adapter_name,
            min_score=test_case.thresholds.min_score,
            input_query=test_case.input.query,
            actual_output=trace.final_output,
            suite_type=test_case.suite_type,
            difficulty=test_case.difficulty,
        )

    def _get_weights_for_test(self, test_case: TestCase) -> Dict[str, float]:
        """
        Get scoring weights for a test case.

        Priority:
        1. Per-test weights override (if specified)
        2. Global default weights
        """
        # Start with default weights
        weights = self.default_weights.to_dict()

        # Apply per-test overrides if specified
        if test_case.thresholds.weights:
            override = test_case.thresholds.weights
            if override.tool_accuracy is not None:
                weights["tool_accuracy"] = override.tool_accuracy
            if override.output_quality is not None:
                weights["output_quality"] = override.output_quality
            if override.sequence_correctness is not None:
                weights["sequence_correctness"] = override.sequence_correctness

            # Validate that weights still sum to 1.0
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                raise ValueError(
                    f"Scoring weights for test '{test_case.name}' must sum to 1.0, got {total:.3f}. "
                    f"When overriding weights, ensure all three values are specified."
                )

        return weights

    def _compute_overall_score(self, evaluations: Evaluations, test_case: TestCase) -> float:
        """
        Compute weighted overall score.

        Weights are configurable via:
        - Global config (scoring.weights in config.yaml)
        - Per-test override (thresholds.weights in test case)

        Default weights:
        - Tool accuracy: 30%
        - Output quality: 50%
        - Sequence correctness: 20%

        Note: Sequence scoring uses progress_score for partial credit.
        Example: If expected sequence is [a, b, c, d, e] and agent completed [a, b, c],
        progress_score = 0.6, contributing 60% of the sequence weight (12/20 points).
        """
        weights = self._get_weights_for_test(test_case)

        # Use progress_score for partial credit on sequence evaluation
        # progress_score is 0.0-1.0, multiply by 100 to get 0-100 scale
        sequence_score = evaluations.sequence_correctness.progress_score * 100

        score = (
            evaluations.tool_accuracy.accuracy * 100 * weights["tool_accuracy"]
            + evaluations.output_quality.score * weights["output_quality"]
            + sequence_score * weights["sequence_correctness"]
        )

        return round(score, 2)

    def _compute_pass_fail(
        self, evaluations: Evaluations, test_case: TestCase, score: float
    ) -> bool:
        """Determine if test case passed all criteria."""
        # Forbidden tool violations are a hard-fail with zero tolerance.
        # This check runs FIRST so the failure reason is unambiguous in reports.
        if evaluations.forbidden_tools and not evaluations.forbidden_tools.passed:
            return False

        # Must pass score threshold
        if score < test_case.thresholds.min_score:
            return False

        # Must pass cost threshold (if specified)
        if not evaluations.cost.passed:
            return False

        # Must pass latency threshold (if specified)
        if not evaluations.latency.passed:
            return False

        # Must pass hallucination check (if configured)
        if evaluations.hallucination and not evaluations.hallucination.passed:
            return False

        # Must pass safety check (if configured)
        if evaluations.safety and not evaluations.safety.passed:
            return False
        
        # Must pass PII check (if configured)
        if evaluations.pii and not evaluations.pii.passed:
            return False

        return True

    # ------------------------------------------------------------------
    # Code-based (zero-cost) evaluation helpers
    # ------------------------------------------------------------------

    # Maximum output length to run regex/schema checks against.
    # Prevents pathological performance on very large agent outputs.
    _MAX_CHECK_OUTPUT_LEN: int = 100_000  # 100 KB

    # Maximum time (seconds) a single regex match is allowed to take.
    # Protects against ReDoS from user-supplied patterns.
    _REGEX_TIMEOUT_S: float = 2.0

    @staticmethod
    def _compile_regex(pattern: str) -> Optional[_re.Pattern]:  # type: ignore[type-arg]
        """Compile a regex pattern with validation.

        Returns the compiled pattern, or None if the pattern is invalid.
        Invalid patterns are logged as warnings.
        """
        try:
            return _re.compile(pattern, _re.IGNORECASE | _re.DOTALL)
        except _re.error as e:
            logger.warning("Invalid regex pattern %r: %s", pattern, e)
            return None

    @staticmethod
    def _check_regex_patterns(output: str, patterns: List[str]) -> Tuple[List[str], List[str]]:
        """Check output against regex patterns.

        Safety measures:
        - Output is truncated to _MAX_CHECK_OUTPUT_LEN to bound runtime.
        - Each pattern is compiled once and validated before matching.
        - A per-pattern timeout prevents ReDoS from catastrophic backtracking.

        Returns (passed, failed).
        """
        import signal
        import sys
        import threading

        truncated = output[:Evaluator._MAX_CHECK_OUTPUT_LEN]
        passed: List[str] = []
        failed: List[str] = []

        # signal.alarm is Unix-only and process-global — only safe on the
        # main thread. On Windows or worker threads we skip the timeout guard.
        can_alarm = (
            hasattr(signal, "SIGALRM")
            and sys.platform != "win32"
            and threading.current_thread() is threading.main_thread()
        )

        for pattern in patterns:
            compiled = Evaluator._compile_regex(pattern)
            if compiled is None:
                failed.append(pattern)
                continue

            try:
                if can_alarm:
                    # Set an alarm to interrupt long-running matches (ReDoS protection).
                    old_handler = signal.signal(signal.SIGALRM, Evaluator._alarm_handler)
                    signal.alarm(int(Evaluator._REGEX_TIMEOUT_S) or 1)

                matched = compiled.search(truncated) is not None

                if can_alarm:
                    signal.alarm(0)  # cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)

                if matched:
                    passed.append(pattern)
                else:
                    failed.append(pattern)
            except _RegexTimeoutError:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                logger.warning("Regex pattern timed out (possible ReDoS): %r", pattern)
                failed.append(pattern)
            except Exception as e:
                if can_alarm:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                logger.warning("Regex match error for %r: %s", pattern, e)
                failed.append(pattern)

        return passed, failed

    @staticmethod
    def _alarm_handler(signum: int, frame: Any) -> None:
        """Signal handler that raises _RegexTimeoutError."""
        raise _RegexTimeoutError()

    @staticmethod
    def _extract_first_json_object(text: str) -> Any:
        """Extract the first valid JSON object from surrounding text.

        Uses bracket counting to support arbitrary nesting depth.
        Returns the parsed object, or None if no valid JSON is found.
        """
        in_string = False
        escape_next = False
        for i, ch in enumerate(text):
            if ch != '{' or in_string:
                continue
            # Found a '{' — try bracket counting from here
            depth = 0
            in_str = False
            esc = False
            for j in range(i, len(text)):
                c = text[j]
                if esc:
                    esc = False
                    continue
                if c == '\\' and in_str:
                    esc = True
                    continue
                if c == '"' and not esc:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j + 1]
                        try:
                            return _json.loads(candidate)
                        except (ValueError, _json.JSONDecodeError):
                            break  # malformed, try next '{'
        return None

    @staticmethod
    def _check_json_schema(output: str, schema: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate output against JSON schema. Returns (passed, error_message)."""
        truncated = output[:Evaluator._MAX_CHECK_OUTPUT_LEN]

        data: Any = None
        try:
            data = _json.loads(truncated)
        except (ValueError, _json.JSONDecodeError):
            # Extract JSON objects from surrounding text using bracket counting.
            # Handles arbitrary nesting depth (unlike the previous regex approach).
            data = Evaluator._extract_first_json_object(truncated)

        if data is None:
            return False, "Output does not contain valid JSON"

        # Try jsonschema library first (full spec compliance)
        try:
            import jsonschema
            jsonschema.validate(data, schema)
            return True, ""
        except ImportError:
            # Fallback: basic structural check without jsonschema dependency
            return Evaluator._basic_schema_check(data, schema)
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _basic_schema_check(data: Any, schema: Dict[str, Any]) -> Tuple[bool, str]:
        """Minimal JSON schema validation without jsonschema library."""
        errors: List[str] = []

        # Check type
        expected_type = schema.get("type")
        if expected_type == "object" and not isinstance(data, dict):
            return False, f"Expected object, got {type(data).__name__}"
        if expected_type == "array" and not isinstance(data, list):
            return False, f"Expected array, got {type(data).__name__}"

        # Check required properties
        if isinstance(data, dict):
            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    errors.append(f"Missing required property: {prop}")

            # Check property types
            props = schema.get("properties", {})
            for prop_name, prop_schema in props.items():
                if prop_name in data:
                    prop_type = prop_schema.get("type")
                    value = data[prop_name]
                    if prop_type == "string" and not isinstance(value, str):
                        errors.append(f"{prop_name}: expected string")
                    elif prop_type == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                        errors.append(f"{prop_name}: expected number")
                    elif prop_type == "boolean" and not isinstance(value, bool):
                        errors.append(f"{prop_name}: expected boolean")
                    elif prop_type == "array" and not isinstance(value, list):
                        errors.append(f"{prop_name}: expected array")
                    elif prop_type == "object" and not isinstance(value, dict):
                        errors.append(f"{prop_name}: expected object")

        if errors:
            return False, "; ".join(errors)
        return True, ""

    def _deterministic_output_eval(
        self, test_case: TestCase, trace: ExecutionTrace
    ) -> OutputEvaluation:
        """
        Deterministic output evaluation without LLM-as-judge.

        Uses string similarity and contains/not_contains checks to compute a score.
        Useful when no API key is available.

        Args:
            test_case: Test case with expected output criteria
            trace: Execution trace with actual output

        Returns:
            OutputEvaluation with deterministic score
        """
        output = trace.final_output
        score = 0.0
        rationale_parts = []

        # Check string contains
        contains_passed = []
        contains_failed = []
        if test_case.expected.output and test_case.expected.output.contains:
            must_contain = test_case.expected.output.contains
            output_lower = output.lower()
            for string in must_contain:
                if string.lower() in output_lower:
                    contains_passed.append(string)
                else:
                    contains_failed.append(string)

            if must_contain:
                contains_ratio = len(contains_passed) / len(must_contain)
                score += contains_ratio * _DETO_CONTAINS_WEIGHT
                if contains_failed:
                    rationale_parts.append(f"Missing: {', '.join(contains_failed[:3])}")
                else:
                    rationale_parts.append("All expected strings found")
        else:
            # No contains check — award full weight
            score += _DETO_CONTAINS_WEIGHT
            rationale_parts.append("No contains check specified")

        # Check string not_contains
        not_contains_passed = []
        not_contains_failed = []
        if test_case.expected.output and test_case.expected.output.not_contains:
            must_not_contain = test_case.expected.output.not_contains
            output_lower = output.lower()
            for string in must_not_contain:
                if string.lower() not in output_lower:
                    not_contains_passed.append(string)
                else:
                    not_contains_failed.append(string)

            if must_not_contain:
                not_contains_ratio = len(not_contains_passed) / len(must_not_contain)
                score += not_contains_ratio * _DETO_NOT_CONTAINS_WEIGHT
                if not_contains_failed:
                    rationale_parts.append(f"Contains prohibited: {', '.join(not_contains_failed[:3])}")
        else:
            # No not_contains check — award full weight
            score += _DETO_NOT_CONTAINS_WEIGHT

        # Output length check — penalise empty or very short responses
        if len(output) > _DETO_MIN_OUTPUT_LENGTH:
            score += _DETO_LENGTH_WEIGHT
            rationale_parts.append("Output has reasonable length")
        elif len(output) > 0:
            score += _DETO_LENGTH_WEIGHT / 2
            rationale_parts.append("Output is very short")
        else:
            rationale_parts.append("Output is empty")

        # Basic relevance check — query-term overlap
        query = test_case.input.query.lower()
        output_lower = output.lower()
        query_words = [w for w in query.split() if len(w) > _DETO_MIN_QUERY_WORD_LENGTH]
        if query_words:
            matches = sum(1 for w in query_words if w in output_lower)
            relevance_ratio = min(matches / len(query_words), 1.0)
            score += relevance_ratio * _DETO_RELEVANCE_WEIGHT
            if relevance_ratio > 0.5:
                rationale_parts.append("Output appears relevant to query")

        # Regex pattern checks (zero-cost)
        if test_case.expected.output and test_case.expected.output.regex_patterns:
            patterns = test_case.expected.output.regex_patterns
            regex_passed, regex_failed = self._check_regex_patterns(output, patterns)
            if patterns:
                regex_ratio = len(regex_passed) / len(patterns)
                score += regex_ratio * _DETO_REGEX_WEIGHT
                if regex_failed:
                    rationale_parts.append(f"Regex failed: {', '.join(regex_failed[:3])}")
                else:
                    rationale_parts.append("All regex patterns matched")
        else:
            score += _DETO_REGEX_WEIGHT

        # JSON schema validation (zero-cost)
        if test_case.expected.output and test_case.expected.output.json_schema:
            schema_passed, schema_error = self._check_json_schema(
                output, test_case.expected.output.json_schema
            )
            if schema_passed:
                score += _DETO_JSON_SCHEMA_WEIGHT
                rationale_parts.append("JSON schema valid")
            else:
                rationale_parts.append(f"JSON schema failed: {schema_error[:80]}")
        else:
            score += _DETO_JSON_SCHEMA_WEIGHT

        # Cap at _DETO_SCORE_CAP to signal this score is heuristic, not authoritative
        score = min(score, _DETO_SCORE_CAP)

        return OutputEvaluation(
            score=round(score, 2),
            rationale=f"[DETERMINISTIC] {'; '.join(rationale_parts)}",
            contains_checks=ContainsChecks(passed=contains_passed, failed=contains_failed),
            not_contains_checks=ContainsChecks(passed=not_contains_passed, failed=not_contains_failed),
        )
