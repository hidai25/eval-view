"""Orchestrator for two-phase skill test evaluation.

Coordinates Phase 1 (deterministic) and Phase 2 (rubric) evaluation,
handling pass/fail logic and score aggregation.
"""

import logging
from typing import Optional

from evalview.skills.agent_types import (
    DeterministicExpected,
    DeterministicEvaluation,
    RubricConfig,
    RubricEvaluation,
    SkillAgentTrace,
    SkillAgentTest,
    SkillAgentTestResult,
    TestCategory,
)
from evalview.skills.evaluators.deterministic import DeterministicEvaluator
from evalview.skills.evaluators.rubric import RubricEvaluator

logger = logging.getLogger(__name__)


class SkillTestOrchestrator:
    """Orchestrates two-phase skill test evaluation.

    Evaluation flow:
    1. Run Phase 1 deterministic checks (fast, no LLM)
    2. If Phase 1 passes AND rubric is configured, run Phase 2 (LLM judge)
    3. Calculate final score and pass/fail status

    Score calculation:
    - If only Phase 1: use deterministic score
    - If both phases: weighted average (60% Phase 1, 40% Phase 2)
    """

    def __init__(
        self,
        skip_rubric: bool = False,
        rubric_model: Optional[str] = None,
    ):
        """Initialize orchestrator.

        Args:
            skip_rubric: If True, skip Phase 2 evaluation entirely
            rubric_model: Optional model override for rubric evaluation
        """
        self.skip_rubric = skip_rubric
        self.deterministic_evaluator = DeterministicEvaluator()
        self.rubric_evaluator = RubricEvaluator(model=rubric_model)

    async def evaluate(
        self,
        test: SkillAgentTest,
        trace: SkillAgentTrace,
        cwd: Optional[str] = None,
    ) -> SkillAgentTestResult:
        """Evaluate a skill test execution.

        Args:
            test: The test case specification
            trace: Execution trace from agent adapter
            cwd: Working directory for file path resolution

        Returns:
            SkillAgentTestResult with evaluation details
        """
        # Phase 1: Deterministic evaluation
        deterministic_result = self.deterministic_evaluator.evaluate(
            expected=test.expected,
            trace=trace,
            cwd=cwd,
        )

        # Phase 2: Rubric evaluation (if applicable)
        rubric_result: Optional[RubricEvaluation] = None

        if self._should_run_rubric(test, deterministic_result):
            # test.rubric is guaranteed non-None by _should_run_rubric
            assert test.rubric is not None
            rubric_result = await self.rubric_evaluator.evaluate(
                rubric=test.rubric,
                trace=trace,
                skill_name=trace.skill_name,
            )

        # Calculate final score and pass/fail
        final_score, passed = self._calculate_final_result(
            test=test,
            deterministic=deterministic_result,
            rubric=rubric_result,
        )

        return SkillAgentTestResult(
            test_name=test.name,
            category=test.category,
            passed=passed,
            score=final_score,
            input_query=test.input,
            final_output=trace.final_output,
            deterministic=deterministic_result,
            rubric=rubric_result,
            trace=trace if trace else None,
            latency_ms=trace.duration_ms,
            input_tokens=trace.total_input_tokens,
            output_tokens=trace.total_output_tokens,
            error=trace.errors[0] if trace.errors else None,
        )

    def _should_run_rubric(
        self,
        test: SkillAgentTest,
        deterministic_result: DeterministicEvaluation,
    ) -> bool:
        """Determine if Phase 2 rubric evaluation should run.

        Args:
            test: The test case
            deterministic_result: Result of Phase 1

        Returns:
            True if rubric evaluation should run
        """
        # Skip if explicitly disabled
        if self.skip_rubric:
            return False

        # Skip if no rubric configured
        if test.rubric is None:
            return False

        # Skip if Phase 1 failed (unless it's a negative test)
        if not deterministic_result.passed and test.category != TestCategory.NEGATIVE:
            logger.debug(
                f"Skipping rubric for '{test.name}': Phase 1 failed"
            )
            return False

        return True

    def _calculate_final_result(
        self,
        test: SkillAgentTest,
        deterministic: DeterministicEvaluation,
        rubric: Optional[RubricEvaluation],
    ) -> tuple:
        """Calculate final score and pass/fail status.

        Args:
            test: The test case
            deterministic: Phase 1 result
            rubric: Phase 2 result (may be None)

        Returns:
            Tuple of (final_score, passed)
        """
        # Handle negative tests (should NOT trigger)
        if test.category == TestCategory.NEGATIVE:
            return self._evaluate_negative_test(test, deterministic, rubric)

        # Standard test evaluation
        if rubric is not None:
            # Both phases ran - weighted average
            # 60% deterministic, 40% rubric
            final_score = (deterministic.score * 0.6) + (rubric.score * 0.4)
            passed = deterministic.passed and rubric.passed
        else:
            # Only Phase 1 ran
            final_score = deterministic.score
            passed = deterministic.passed

        return final_score, passed

    def _evaluate_negative_test(
        self,
        test: SkillAgentTest,
        deterministic: DeterministicEvaluation,
        rubric: Optional[RubricEvaluation],
    ) -> tuple:
        """Evaluate a negative test (should NOT trigger skill).

        For negative tests with should_trigger=False:
        - The test defines what should NOT happen using 'not_contain' checks
        - If those checks pass, it means the skill correctly did not trigger
        - The deterministic checks directly express the expected behavior

        Args:
            test: The negative test case
            deterministic: Phase 1 result
            rubric: Phase 2 result

        Returns:
            Tuple of (final_score, passed)
        """
        if not test.should_trigger:
            # For negative tests, the expected checks (like tool_calls_not_contain)
            # already express what should NOT happen. If they pass, the test passes.
            return deterministic.score, deterministic.passed
        else:
            # Normal negative test with should_trigger=True
            # (unusual but supported)
            return deterministic.score, deterministic.passed
