"""Main evaluator orchestrator."""

import logging
from datetime import datetime
from typing import Optional, Dict, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Deterministic output scoring weights (must sum to 100)
# ---------------------------------------------------------------------------
# These control the fallback scorer used when no LLM judge is available.
# Adjust here to rebalance; the cap prevents misleadingly high scores
# since deterministic heuristics are inherently approximate.
_DETO_CONTAINS_WEIGHT: float = 40.0      # % score for contains checks
_DETO_NOT_CONTAINS_WEIGHT: float = 20.0  # % score for not_contains checks
_DETO_LENGTH_WEIGHT: float = 20.0        # % score for non-empty output
_DETO_RELEVANCE_WEIGHT: float = 20.0     # % score for query-term overlap
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

if TYPE_CHECKING:
    from evalview.core.judge_cache import JudgeCache

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
        self.default_weights = default_weights or DEFAULT_WEIGHTS
        self.skip_llm_judge = skip_llm_judge
        self.judge_cache = judge_cache
        self._logged_deterministic_mode = False

        # Only initialize LLM-dependent evaluators when needed
        # This avoids requiring API keys for deterministic mode
        if not skip_llm_judge:
            self.output_evaluator = OutputEvaluator(cache=judge_cache)
            self.hallucination_evaluator = HallucinationEvaluator()
            self.safety_evaluator = SafetyEvaluator()
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

        return True

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

        # Cap at _DETO_SCORE_CAP to signal this score is heuristic, not authoritative
        score = min(score, _DETO_SCORE_CAP)

        return OutputEvaluation(
            score=round(score, 2),
            rationale=f"[DETERMINISTIC] {'; '.join(rationale_parts)}",
            contains_checks=ContainsChecks(passed=contains_passed, failed=contains_failed),
            not_contains_checks=ContainsChecks(passed=not_contains_passed, failed=not_contains_failed),
        )
