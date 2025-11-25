"""Output quality evaluator using LLM-as-judge."""

import json
import os
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI
from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    OutputEvaluation,
    ContainsChecks,
)


class OutputEvaluator:
    """Evaluates output quality using string checks and LLM-as-judge."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize output evaluator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def evaluate(self, test_case: TestCase, trace: ExecutionTrace) -> OutputEvaluation:
        """
        Evaluate output quality.

        Args:
            test_case: Test case with expected output criteria
            trace: Execution trace with actual output

        Returns:
            OutputEvaluation with quality score and checks
        """
        output = trace.final_output

        # Check string contains/not_contains
        contains_checks = self._check_contains(
            output, test_case.expected.output.contains if test_case.expected.output else []
        )

        not_contains_checks = self._check_not_contains(
            output,
            test_case.expected.output.not_contains if test_case.expected.output else [],
        )

        # LLM-as-judge evaluation
        llm_result = await self._llm_as_judge(test_case, trace)

        return OutputEvaluation(
            score=llm_result["score"],
            rationale=llm_result["rationale"],
            contains_checks=contains_checks,
            not_contains_checks=not_contains_checks,
        )

    def _check_contains(self, output: str, must_contain: Optional[List[str]]) -> ContainsChecks:
        """Check if output contains required strings."""
        if not must_contain:
            return ContainsChecks(passed=[], failed=[])

        passed: List[str] = []
        failed: List[str] = []

        output_lower = output.lower()
        for string in must_contain:
            if string.lower() in output_lower:
                passed.append(string)
            else:
                failed.append(string)

        return ContainsChecks(passed=passed, failed=failed)

    def _check_not_contains(
        self, output: str, must_not_contain: Optional[List[str]]
    ) -> ContainsChecks:
        """Check if output does not contain prohibited strings."""
        if not must_not_contain:
            return ContainsChecks(passed=[], failed=[])

        passed: List[str] = []
        failed: List[str] = []

        output_lower = output.lower()
        for string in must_not_contain:
            if string.lower() not in output_lower:
                passed.append(string)
            else:
                failed.append(string)

        return ContainsChecks(passed=passed, failed=failed)

    async def _llm_as_judge(self, test_case: TestCase, trace: ExecutionTrace) -> Dict[str, Any]:
        """Use LLM to judge output quality."""
        system_prompt = """You are an expert evaluator of AI agent outputs. Rate the quality and correctness of the agent's response on a scale of 0-100.

Consider:
- Accuracy: Is the information correct?
- Completeness: Does it fully answer the query?
- Relevance: Is it on-topic?
- Clarity: Is it well-structured?

Return a JSON object with:
{
  "score": <number 0-100>,
  "rationale": "<brief explanation>"
}"""

        user_content = {
            "query": test_case.input.query,
            "agent_output": trace.final_output,
        }

        if test_case.expected.output and test_case.expected.output.contains:
            user_content["expected_contains"] = test_case.expected.output.contains

        # Use EVAL_MODEL from env (defaults to gpt-4o-mini for backwards compatibility)
        model = os.getenv("EVAL_MODEL", "gpt-4o-mini")

        # GPT-5 models require temperature=1 and max_completion_tokens
        # GPT-4 models use temperature=0.3 and max_tokens
        is_gpt5 = model.startswith("gpt-5")

        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_content, indent=2)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 1 if is_gpt5 else 0.3,
        }

        # GPT-5 uses max_completion_tokens, GPT-4 uses max_tokens
        if is_gpt5:
            params["max_completion_tokens"] = 5000
        else:
            params["max_tokens"] = 1000

        response = await self.client.chat.completions.create(**params)

        result = json.loads(response.choices[0].message.content or "{}")
        return {
            "score": result.get("score", 0),
            "rationale": result.get("rationale", "No rationale provided"),
        }
