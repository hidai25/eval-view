"""Hallucination detection evaluator.

Uses the Ragas-style faithfulness pattern:
1. Extract factual claims from agent response
2. Verify each claim against tool outputs
3. Score = supported claims / total claims

This is the industry-standard approach used by Ragas, Patronus AI, and others.
It dramatically reduces false positives compared to single-prompt fact-checking
because the judge evaluates one claim at a time with full context.
"""

import json
import logging
from typing import Optional, Tuple, List, Dict, Any

from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    HallucinationEvaluation,
    HallucinationCheck,
)
from evalview.core.llm_provider import LLMClient, LLMProvider

logger = logging.getLogger(__name__)


class HallucinationEvaluator:
    """Evaluator for detecting factual hallucinations in agent outputs.

    Architecture (Ragas faithfulness pattern):
      Step 1: Extract discrete factual claims from agent response
      Step 2: For each claim, check if ANY tool output supports it
      Step 3: Faithfulness = supported_claims / total_claims
      Step 4: Flag unsupported claims as potential hallucinations
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.llm_client = LLMClient(provider=provider, api_key=api_key, model=model)

    async def evaluate(self, test_case: TestCase, trace: ExecutionTrace) -> HallucinationEvaluation:
        """Evaluate if agent output contains hallucinations."""
        hallucination_config = test_case.expected.hallucination

        if isinstance(hallucination_config, dict):
            hallucination_config = HallucinationCheck(**hallucination_config)

        if not hallucination_config:
            hallucination_config = HallucinationCheck(check=True)

        if not hallucination_config.check:
            return HallucinationEvaluation(
                has_hallucination=False,
                confidence=0.0,
                details="Hallucination detection disabled",
                passed=True,
            )

        has_hallucination, confidence, details = await self._detect_hallucination(test_case, trace)

        # Determine pass/fail
        is_local_model = self.llm_client.provider.value == "ollama"
        confidence_threshold = 0.95 if is_local_model else 0.98

        if has_hallucination and confidence < confidence_threshold:
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
        """Detect hallucinations using claim-level verification.

        Returns:
            Tuple of (has_hallucination, confidence, details)
        """
        # Deterministic checks
        tool_issues = self._check_tool_consistency(trace)
        uncertainty_issues = self._check_uncertainty_handling(test_case, trace)

        # Skip LLM-based checks if no output to verify
        if not trace.final_output or not trace.final_output.strip():
            all_deterministic = tool_issues + uncertainty_issues
            if all_deterministic:
                return True, 0.8, "\n".join(f"- {i}" for i in all_deterministic)
            return False, 1.0, "No output to verify."

        # Build full tool context (no aggressive truncation)
        tool_context = self._build_tool_context(trace)

        # Step 1: Extract claims from agent response
        claims, extraction_failed = await self._extract_claims(trace.final_output, test_case.input.query)

        if extraction_failed:
            # LLM call failed — report it clearly, don't mask as "no claims"
            all_deterministic = tool_issues + uncertainty_issues
            if all_deterministic:
                details = "Claim extraction unavailable (LLM error). Deterministic issues found:\n"
                details += "\n".join(f"- {i}" for i in all_deterministic)
                return True, 0.7, details
            return False, 0.0, "Claim extraction unavailable (LLM error). No deterministic issues found."

        if not claims:
            # Genuinely no verifiable claims — check deterministic issues
            all_deterministic = tool_issues + uncertainty_issues
            if all_deterministic:
                return True, 0.7, "\n".join(f"- {i}" for i in all_deterministic)
            return False, 1.0, "No verifiable factual claims found in output."

        # Step 2: Verify each claim against tool outputs
        verdicts = await self._verify_claims(claims, tool_context)

        # Step 3: Calculate faithfulness score
        supported = sum(1 for v in verdicts if v["supported"])
        total = len(verdicts)
        faithfulness = supported / total if total > 0 else 1.0

        # Collect unsupported claims
        unsupported = [v for v in verdicts if not v["supported"]]

        # Combine all issues: deterministic + LLM-verified
        all_issues: List[str] = list(tool_issues) + list(uncertainty_issues)
        for v in unsupported:
            all_issues.append(f"{v['claim']} — {v['reason']}")

        # Deterministic issues AND unsupported claims both drive the verdict
        has_hallucination = len(all_issues) > 0
        if has_hallucination:
            unsupported_count = len(unsupported) + len(tool_issues) + len(uncertainty_issues)
            total_checks = total + len(tool_issues) + len(uncertainty_issues)
            confidence = min(0.5 + (unsupported_count / max(total_checks, 1)) * 0.5, 0.99)
        else:
            confidence = faithfulness

        if has_hallucination:
            details = (
                f"Faithfulness: {faithfulness:.0%} ({supported}/{total} claims supported)\n"
                f"Issues:\n"
                + "\n".join(f"- {issue}" for issue in all_issues)
            )
        else:
            details = f"Faithfulness: {faithfulness:.0%} ({supported}/{total} claims verified against tool outputs)."

        return has_hallucination, confidence, details

    async def _extract_claims(self, response: str, query: str) -> Tuple[List[str], bool]:
        """Step 1: Extract discrete factual claims from agent response.

        Returns:
            Tuple of (claims_list, extraction_failed).
            extraction_failed is True when the LLM call itself errored,
            distinguishing "no claims found" from "couldn't check."
        """
        prompt = f"""Extract all specific factual claims from this AI agent response.

Rules:
- Each claim should be a single, simple factual statement
- Replace pronouns with the actual entities they refer to
- Skip opinions, advice, recommendations, and meta-commentary
- Skip hedged/uncertain statements ("might", "could", "possibly")
- Only include claims that reference specific data, numbers, names, or events
- If the response is mostly general advice with no specific factual claims, return an empty list

Query: {query}

Response:
{response[:3000]}

Return a JSON array of claim strings. Example:
["User X burned through $1000 in costs", "The error rate was 40%", "LangChain agents ignored 515 stop commands"]

If no specific factual claims exist, return: []"""

        try:
            result = await self.llm_client.chat_completion(
                system_prompt="You extract factual claims from text. Return only a JSON array of strings. No explanations.",
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=1500,
            )
            # Result should be a list of strings
            if isinstance(result, list):
                return [str(c) for c in result if c], False
            if isinstance(result, dict) and "claims" in result:
                return [str(c) for c in result["claims"] if c], False
            # Try to parse as JSON array from string
            if isinstance(result, str):
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    return [str(c) for c in parsed if c], False
            return [], False  # LLM succeeded but found no claims
        except Exception as e:
            logger.debug("Claim extraction failed: %s", e)
            return [], True  # LLM call failed — flag as extraction_failed

    async def _verify_claims(
        self, claims: List[str], tool_context: str
    ) -> List[Dict[str, Any]]:
        """Step 2: Verify each claim against the full tool output context.

        For each claim, determines if it is supported by the tool outputs.
        Uses a single batched LLM call for efficiency.
        """
        claims_numbered = "\n".join(f"{i+1}. {claim}" for i, claim in enumerate(claims))

        prompt = f"""You are verifying whether each factual claim is supported by the tool outputs below.

Tool Outputs (this is what the agent had access to):
{tool_context}

Claims to verify:
{claims_numbered}

For each claim, determine:
- "supported": true if the tool outputs contain evidence for this claim (even paraphrased, reorganized, or summarized)
- "supported": false ONLY if the claim clearly contradicts the tool outputs OR has absolutely no basis in them

Be generous — if data could plausibly support the claim, mark it as supported.
Agents commonly summarize, round numbers, group data, or rephrase — these are NOT hallucinations.

Return a JSON array with one object per claim:
[
  {{"claim": "...", "supported": true, "reason": "Found in tool output N"}},
  {{"claim": "...", "supported": false, "reason": "No evidence in any tool output"}}
]"""

        try:
            result = await self.llm_client.chat_completion(
                system_prompt="You verify factual claims against evidence. Return only a JSON array. Be generous — paraphrasing is not hallucination.",
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=2000,
            )

            verdicts = self._parse_verdicts(result, claims)
            return verdicts

        except Exception as e:
            logger.debug("Claim verification failed: %s", e)
            # On failure, assume all claims are supported (fail open, not closed)
            return [{"claim": c, "supported": True, "reason": "Verification unavailable"} for c in claims]

    def _parse_verdicts(self, result: Any, claims: List[str]) -> List[Dict[str, Any]]:
        """Parse LLM verification result into structured verdicts."""
        verdicts = []

        if isinstance(result, list):
            verdicts = result
        elif isinstance(result, dict) and "verdicts" in result:
            verdicts = result["verdicts"]
        elif isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    verdicts = parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Ensure we have a verdict for each claim
        if len(verdicts) < len(claims):
            for i in range(len(verdicts), len(claims)):
                verdicts.append({
                    "claim": claims[i],
                    "supported": True,
                    "reason": "No verdict returned — assuming supported",
                })

        # Normalize verdict format
        normalized = []
        for i, v in enumerate(verdicts):
            if isinstance(v, dict):
                normalized.append({
                    "claim": v.get("claim", claims[i] if i < len(claims) else "unknown"),
                    "supported": bool(v.get("supported", True)),
                    "reason": str(v.get("reason", "")),
                })
            else:
                normalized.append({
                    "claim": claims[i] if i < len(claims) else "unknown",
                    "supported": True,
                    "reason": "Unparseable verdict — assuming supported",
                })

        return normalized

    def _build_tool_context(self, trace: ExecutionTrace) -> str:
        """Build full tool context string for verification.

        Includes generous output limits so the judge can verify
        claims the agent made from deeper in the results.
        """
        if not trace.steps:
            return "(No tools were used — any specific factual claims are unverifiable)"

        parts = []
        for i, step in enumerate(trace.steps, 1):
            output_str = str(step.output) if step.output is not None else "(no output captured)"
            # Allow up to 3000 chars per tool — enough for most API responses
            if len(output_str) > 3000:
                output_str = output_str[:3000] + f"\n... (truncated from {len(output_str)} chars)"

            params_str = str(step.parameters)[:500] if step.parameters else ""
            parts.append(
                f"[Tool {i}: {step.tool_name}]\n"
                f"Input: {params_str}\n"
                f"Output: {output_str}"
            )

        return "\n\n".join(parts)

    def _check_tool_consistency(self, trace: ExecutionTrace) -> List[str]:
        """Deterministic check: did tools fail but agent didn't acknowledge it?"""
        issues = []

        for step in trace.steps:
            if not step.success or (step.error and "error" in str(step.output).lower()):
                output_lower = (trace.final_output or "").lower()
                if not any(
                    keyword in output_lower
                    for keyword in ["error", "failed", "unable", "couldn't", "cannot", "not found", "no results", "no data"]
                ):
                    issues.append(
                        f"Tool '{step.tool_name}' failed/returned error, but agent did not acknowledge the failure"
                    )

        if not trace.steps and len(trace.final_output or "") > 100:
            lower = (trace.final_output or "").lower()
            if "based on" in lower or "according to" in lower:
                issues.append(
                    "Agent made factual claims without using any tools to verify"
                )

        return issues

    def _check_uncertainty_handling(self, test_case: TestCase, trace: ExecutionTrace) -> List[str]:
        """Check if agent properly acknowledges uncertainty when required."""
        issues = []

        output_config = test_case.expected.output
        if not output_config:
            return issues

        if isinstance(output_config, dict):
            must_acknowledge = output_config.get("must_acknowledge_uncertainty", False)
        else:
            must_acknowledge = output_config.must_acknowledge_uncertainty or False

        if must_acknowledge:
            any_failures = any(not step.success for step in trace.steps)
            no_tools_used = len(trace.steps) == 0

            if any_failures or no_tools_used:
                output_lower = (trace.final_output or "").lower()
                uncertainty_phrases = [
                    "i don't know", "i'm not sure", "uncertain",
                    "unable to determine", "cannot confirm",
                    "no information available", "could not find",
                ]
                if not any(phrase in output_lower for phrase in uncertainty_phrases):
                    issues.append(
                        "Agent should acknowledge uncertainty when tools fail or no information is available"
                    )

        return issues
