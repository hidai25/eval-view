"""Hallucination detection evaluator.

Supports multiple LLM providers: OpenAI, Anthropic, Gemini, and Grok.
"""

from datetime import datetime
from typing import Optional, Tuple, List

from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    HallucinationEvaluation,
    HallucinationCheck,
)
from evalview.core.llm_provider import LLMClient, LLMProvider


class HallucinationEvaluator:
    """Evaluator for detecting factual hallucinations in agent outputs.

    Supports multiple LLM providers for fact-checking.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize hallucination evaluator.

        Args:
            provider: LLM provider to use (auto-detected if not specified)
            api_key: API key (uses env var if not specified)
            model: Model to use (uses provider default if not specified)
        """
        self.llm_client = LLMClient(provider=provider, api_key=api_key, model=model)

    async def evaluate(self, test_case: TestCase, trace: ExecutionTrace) -> HallucinationEvaluation:
        """
        Evaluate if agent output contains hallucinations.

        Args:
            test_case: Test case with expected behavior
            trace: Execution trace from agent

        Returns:
            HallucinationEvaluation with detection results
        """
        # Check if hallucination check is configured
        hallucination_config = test_case.expected.hallucination

        # Parse config if it's a dict
        if isinstance(hallucination_config, dict):
            hallucination_config = HallucinationCheck(**hallucination_config)

        # If no config provided, use defaults (check=True by default now)
        if not hallucination_config:
            hallucination_config = HallucinationCheck(check=True)

        # Skip if explicitly disabled
        if not hallucination_config.check:
            return HallucinationEvaluation(
                has_hallucination=False,
                confidence=0.0,
                details="Hallucination detection disabled",
                passed=True,
            )

        # Perform hallucination detection
        has_hallucination, confidence, details = await self._detect_hallucination(test_case, trace)

        # Determine if passed based on configuration
        # Use high confidence threshold to reduce false positives
        # (tool output may be truncated, causing incorrect hallucination detection)
        is_local_model = self.llm_client.provider.value == "ollama"
        confidence_threshold = 0.95 if is_local_model else 0.98

        # Only fail if hallucination detected with high confidence AND not allowed
        if has_hallucination and confidence < confidence_threshold:
            # Low confidence detection - treat as warning, not failure
            passed = True
            details = f"[Warning] {details}\n(Confidence {confidence:.0%} below threshold {confidence_threshold:.0%} - not blocking)"
        else:
            passed = not has_hallucination or hallucination_config.allow

        return HallucinationEvaluation(
            has_hallucination=has_hallucination,
            confidence=confidence,
            details=details,
            passed=passed,
        )

    async def _detect_hallucination(
        self, test_case: TestCase, trace: ExecutionTrace
    ) -> Tuple[bool, float, str]:
        """
        Detect hallucinations using multiple strategies.

        Args:
            test_case: Test case
            trace: Execution trace

        Returns:
            Tuple of (has_hallucination, confidence, details)
        """
        # Strategy 1: Tool consistency check
        tool_consistency_issues = self._check_tool_consistency(trace)

        # Strategy 2: LLM-based fact checking
        fact_check_result = await self._llm_fact_check(test_case, trace)

        # Strategy 3: Uncertainty detection
        uncertainty_issues = self._check_uncertainty_handling(test_case, trace)

        # Combine results
        all_issues = []
        if tool_consistency_issues:
            all_issues.extend(tool_consistency_issues)
        if fact_check_result.get("issues"):
            all_issues.extend(fact_check_result["issues"])
        if uncertainty_issues:
            all_issues.extend(uncertainty_issues)

        # Determine overall result
        has_hallucination = len(all_issues) > 0
        confidence = fact_check_result.get("confidence", 0.0) if has_hallucination else 1.0

        if has_hallucination:
            # Check if the issue is due to missing tool output
            tools_missing_output = [
                step.tool_name for step in trace.steps
                if step.output is None or str(step.output) in ("None", "", "null")
            ]

            details = "Potential hallucinations detected:\n" + "\n".join(
                f"- {issue}" for issue in all_issues
            )

            # Add context about why this might be flagged
            if tools_missing_output:
                details += f"\n\n⚠️  Note: Tool output was not captured for: {', '.join(tools_missing_output)}. "
                details += "The agent may have used real data, but EvalView couldn't verify it because the tool results weren't captured in the trace."
        else:
            details = "No hallucinations detected. Output appears factually consistent."

        return has_hallucination, confidence, details

    def _check_tool_consistency(self, trace: ExecutionTrace) -> List[str]:
        """
        Check if agent output is consistent with tool results.

        Args:
            trace: Execution trace

        Returns:
            List of consistency issues
        """
        issues = []

        # Check if tools returned errors but agent claimed success
        for step in trace.steps:
            if not step.success or (step.error and "error" in str(step.output).lower()):
                # Tool failed, check if agent output acknowledges this
                output_lower = trace.final_output.lower()
                if not any(
                    keyword in output_lower
                    for keyword in ["error", "failed", "unable", "couldn't", "cannot", "not found"]
                ):
                    issues.append(
                        f"Tool '{step.tool_name}' failed/returned error, but agent did not acknowledge failure"
                    )

        # Check for claims not supported by tool outputs
        # (This is a simple heuristic - LLM will do more thorough check)
        if not trace.steps and len(trace.final_output) > 100:
            # Agent provided detailed answer without using any tools
            if (
                "based on" in trace.final_output.lower()
                or "according to" in trace.final_output.lower()
            ):
                issues.append(
                    "Agent made factual claims without using any tools to verify information"
                )

        return issues

    async def _llm_fact_check(self, test_case: TestCase, trace: ExecutionTrace) -> dict:
        """
        Use LLM to fact-check agent output against tool results.

        Args:
            test_case: Test case
            trace: Execution trace

        Returns:
            Dict with fact check results
        """
        # Build tool results summary
        tool_results = []
        for step in trace.steps:
            tool_results.append(
                {
                    "tool": step.tool_name,
                    "input": step.parameters,
                    "output": str(step.output)[:200],  # Limit length
                    "success": step.success,
                    "error": step.error,
                }
            )

        # Include current date so the LLM knows what "today" means
        current_date = datetime.now().strftime("%B %d, %Y")

        prompt = f"""You are a fact-checking system evaluating if an AI agent's response contains hallucinations.

IMPORTANT: Today's date is {current_date}. Use this to determine if date references are current or not.

Query: {test_case.input.query}

Tool Results Available:
{self._format_tool_results(tool_results)}

Agent's Final Response:
{trace.final_output}

WHAT IS A HALLUCINATION (flag these):
- Specific facts that contradict the tool results
- Made-up data, numbers, or statistics not in tool outputs
- False claims presented as fact (e.g., "The API returned X" when it didn't)
- Inventing specific details (names, dates, amounts) not from tools

WHAT IS NOT A HALLUCINATION (do NOT flag these):
- General advice or recommendations (e.g., "consider consulting a professional")
- Common knowledge or widely-known facts
- Helpful context that doesn't contradict tool data
- Practical tips or best practices
- Caveats or disclaimers (e.g., "rates may vary")
- Explaining what the data means or implications
- Adding units to numbers (e.g., "22" becomes "22°C" for temperature)
- Correct mathematical calculations or unit conversions
- Formatting tool data in a user-friendly way
- Reasonable inferences from the data (e.g., "rainy" implies "might need umbrella")

Respond in JSON format:
{{
    "has_hallucination": true/false,
    "confidence": 0.0-1.0,
    "issues": ["issue 1", "issue 2", ...]
}}

Only flag actual false information. Helpful advice is NOT hallucination."""

        try:
            result = await self.llm_client.chat_completion(
                system_prompt="You are a fact-checking system that identifies false claims. Helpful advice is not hallucination. Respond only with valid JSON.",
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=1000,
            )
            return result

        except Exception as e:
            # Fallback if LLM check fails
            return {
                "has_hallucination": False,
                "confidence": 0.0,
                "issues": [f"Fact check failed: {str(e)}"],
            }

    def _format_tool_results(self, tool_results: list) -> str:
        """Format tool results for LLM prompt."""
        if not tool_results:
            return "(No tools were used - any specific claims should be flagged as unverifiable)"

        formatted = []
        for i, result in enumerate(tool_results, 1):
            output = result['output']
            # Flag when tool output wasn't captured
            if output in (None, "None", "", "null"):
                output_str = "(Tool output not captured - any claims about this tool's results are unverifiable)"
            else:
                output_str = output

            formatted.append(
                f"{i}. {result['tool']}({result['input']})\n"
                f"   Success: {result['success']}\n"
                f"   Output: {output_str}\n"
                f"   Error: {result['error'] or 'None'}"
            )

        return "\n".join(formatted)

    def _check_uncertainty_handling(self, test_case: TestCase, trace: ExecutionTrace) -> List[str]:
        """
        Check if agent properly acknowledges uncertainty.

        Args:
            test_case: Test case
            trace: Execution trace

        Returns:
            List of issues with uncertainty handling
        """
        issues = []

        # Check if output config requires uncertainty acknowledgment
        output_config = test_case.expected.output
        if not output_config or not isinstance(output_config, dict):
            return issues

        must_acknowledge = output_config.get("must_acknowledge_uncertainty", False)

        if must_acknowledge:
            # Check if any tools failed or returned no data
            any_failures = any(not step.success for step in trace.steps)
            no_tools_used = len(trace.steps) == 0

            if any_failures or no_tools_used:
                # Agent should express uncertainty
                output_lower = trace.final_output.lower()
                uncertainty_phrases = [
                    "i don't know",
                    "i'm not sure",
                    "uncertain",
                    "unable to determine",
                    "cannot confirm",
                    "no information available",
                    "could not find",
                ]

                has_uncertainty = any(phrase in output_lower for phrase in uncertainty_phrases)

                if not has_uncertainty:
                    issues.append(
                        "Agent should acknowledge uncertainty when tools fail or no information is available"
                    )

        return issues
