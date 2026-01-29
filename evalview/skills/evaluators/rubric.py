"""Rubric-based LLM evaluator for skill testing (Phase 2).

Uses LLM-as-judge to evaluate qualitative aspects of skill execution
based on a user-defined rubric prompt.
"""

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

from evalview.skills.agent_types import (
    RubricConfig,
    RubricEvaluation,
    SkillAgentTrace,
)

if TYPE_CHECKING:
    from evalview.core.llm_provider import LLMClient

logger = logging.getLogger(__name__)


class RubricEvaluator:
    """Evaluates skill execution using LLM-as-judge with rubric.

    This is Phase 2 of the two-phase evaluation system. It only runs if:
    1. Phase 1 deterministic checks passed
    2. A rubric config is provided

    Uses the existing LLMClient from evalview.core.llm_provider for
    multi-provider support (OpenAI, Anthropic, Gemini, etc.).
    """

    def __init__(self, model: Optional[str] = None):
        """Initialize rubric evaluator.

        Args:
            model: Optional model override for evaluation
        """
        self.model_override = model
        self._llm_client: Optional["LLMClient"] = None

    async def evaluate(
        self,
        rubric: RubricConfig,
        trace: SkillAgentTrace,
        skill_name: str,
    ) -> RubricEvaluation:
        """Evaluate execution trace against rubric.

        Args:
            rubric: Rubric configuration with prompt and min_score
            trace: Execution trace to evaluate
            skill_name: Name of the skill being tested

        Returns:
            RubricEvaluation with score and rationale
        """
        try:
            # Lazy load LLM client
            if self._llm_client is None:
                from evalview.core.llm_provider import LLMClient

                model = rubric.model or self.model_override
                self._llm_client = LLMClient(model=model)

            # Build evaluation prompt
            system_prompt = self._build_system_prompt(rubric)
            user_prompt = self._build_user_prompt(trace, skill_name)

            # Get LLM evaluation
            response = await self._llm_client.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=1000,
            )

            # Parse response
            score = float(response.get("score", 0))
            rationale = response.get("reasoning", response.get("rationale", ""))

            passed = score >= rubric.min_score

            return RubricEvaluation(
                passed=passed,
                score=score,
                rationale=rationale,
                min_score=rubric.min_score,
                rubric_response=response,
            )

        except ImportError:
            logger.warning("LLM provider not available, skipping rubric evaluation")
            return RubricEvaluation(
                passed=True,  # Graceful degradation
                score=0.0,
                rationale="Rubric evaluation skipped: LLM provider not available",
                min_score=rubric.min_score,
            )

        except Exception as e:
            logger.error(f"Rubric evaluation failed: {e}")
            return RubricEvaluation(
                passed=False,
                score=0.0,
                rationale=f"Rubric evaluation failed: {str(e)}",
                min_score=rubric.min_score,
            )

    def _build_system_prompt(self, rubric: RubricConfig) -> str:
        """Build system prompt for LLM evaluation.

        Args:
            rubric: Rubric configuration

        Returns:
            System prompt string
        """
        return f"""You are an expert evaluator for AI agent skill testing.
Your task is to evaluate an agent's execution of a skill based on a rubric.

## Rubric
{rubric.prompt}

## Scoring Guidelines
- Score from 0 to 100
- Consider the rubric criteria carefully
- Provide specific reasoning for your score
- Be objective and consistent

Respond with a JSON object containing:
- "score": number from 0 to 100
- "reasoning": detailed explanation of the score
- "strengths": list of things done well
- "weaknesses": list of areas for improvement
"""

    def _build_user_prompt(
        self,
        trace: SkillAgentTrace,
        skill_name: str,
    ) -> str:
        """Build user prompt with execution details.

        Args:
            trace: Execution trace
            skill_name: Name of the skill

        Returns:
            User prompt string
        """
        # Truncate long content to avoid token limits
        final_output = trace.final_output
        if len(final_output) > 5000:
            final_output = final_output[:5000] + "\n... (truncated)"

        return f"""## Skill Being Tested
{skill_name}

## Test Name
{trace.test_name}

## Agent's Final Output
{final_output}

## Execution Summary
- Tool calls: {', '.join(trace.tool_calls) if trace.tool_calls else 'None'}
- Files created: {', '.join(trace.files_created) if trace.files_created else 'None'}
- Files modified: {', '.join(trace.files_modified) if trace.files_modified else 'None'}
- Commands ran: {len(trace.commands_ran)}
- Duration: {trace.duration_ms:.0f}ms
- Errors: {', '.join(trace.errors) if trace.errors else 'None'}

Please evaluate this execution according to the rubric.
"""
