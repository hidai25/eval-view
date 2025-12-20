"""Skill test runner - executes skills against Claude."""

import time
from pathlib import Path
from typing import Any, Optional
import yaml  # type: ignore[import-untyped]

from evalview.skills.types import (
    Skill,
    SkillTestSuite,
    SkillTest,
    SkillTestResult,
    SkillTestSuiteResult,
    SkillExpectedBehavior,
)
from evalview.skills.parser import SkillParser


class SkillRunner:
    """Runs skill tests against Claude API.

    Loads a skill, sends test queries, and evaluates responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the skill runner.

        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
            model: Model to use for testing
        """
        import os

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )
        self.model = model
        self._client: Optional[Any] = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client

    def load_test_suite(self, yaml_path: str) -> SkillTestSuite:
        """Load a test suite from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Test suite not found: {yaml_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Resolve skill path relative to the YAML file's directory
        if "skill" in data and not Path(data["skill"]).is_absolute():
            yaml_dir = path.parent
            data["skill"] = str((yaml_dir / data["skill"]).resolve())

        return SkillTestSuite(**data)

    def run_suite(self, suite: SkillTestSuite) -> SkillTestSuiteResult:
        """
        Run all tests in a test suite.

        Args:
            suite: The test suite to run

        Returns:
            SkillTestSuiteResult with all results
        """
        # Load the skill
        skill = SkillParser.parse_file(suite.skill)

        # Run each test
        results = []
        for test in suite.tests:
            result = self.run_test(skill, test, model=suite.model)
            results.append(result)

        # Calculate stats
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = len(results) - passed_tests
        pass_rate = passed_tests / len(results) if results else 0.0

        total_latency = sum(r.latency_ms for r in results)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in results)

        return SkillTestSuiteResult(
            suite_name=suite.name,
            skill_name=skill.metadata.name,
            passed=pass_rate >= suite.min_pass_rate,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            pass_rate=pass_rate,
            results=results,
            total_latency_ms=total_latency,
            avg_latency_ms=total_latency / len(results) if results else 0.0,
            total_tokens=total_tokens,
        )

    def run_test(
        self,
        skill: Skill,
        test: SkillTest,
        model: Optional[str] = None,
    ) -> SkillTestResult:
        """
        Run a single test against a skill.

        Args:
            skill: The loaded skill
            test: The test to run
            model: Model override

        Returns:
            SkillTestResult
        """
        model = model or self.model

        # Build system prompt with skill instructions
        system_prompt = self._build_system_prompt(skill)

        # Call Claude
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": test.input}],
            )
            latency_ms = (time.time() - start_time) * 1000

            output = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            error = None

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            output = ""
            input_tokens = 0
            output_tokens = 0
            error = str(e)

        # Evaluate the response
        evaluation = self._evaluate_response(output, test.expected)

        return SkillTestResult(
            test_name=test.name,
            passed=evaluation["passed"] and error is None,
            score=evaluation["score"],
            input_query=test.input,
            output=output,
            contains_passed=evaluation["contains_passed"],
            contains_failed=evaluation["contains_failed"],
            not_contains_passed=evaluation["not_contains_passed"],
            not_contains_failed=evaluation["not_contains_failed"],
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )

    def _build_system_prompt(self, skill: Skill) -> str:
        """Build system prompt with skill loaded."""
        return f"""You are Claude, an AI assistant with the following skill loaded:

# Skill: {skill.metadata.name}

{skill.metadata.description}

## Instructions

{skill.instructions}

---

Follow the skill instructions above when responding to user queries.
"""

    def _evaluate_response(
        self,
        output: str,
        expected: SkillExpectedBehavior,
    ) -> dict:
        """
        Evaluate a response against expected behavior.

        Returns dict with: passed, score, contains_passed/failed, not_contains_passed/failed
        """
        output_lower = output.lower()
        total_checks = 0
        passed_checks = 0

        contains_passed = []
        contains_failed = []
        not_contains_passed = []
        not_contains_failed = []

        # Check output_contains
        if expected.output_contains:
            for phrase in expected.output_contains:
                total_checks += 1
                if phrase.lower() in output_lower:
                    passed_checks += 1
                    contains_passed.append(phrase)
                else:
                    contains_failed.append(phrase)

        # Check output_not_contains
        if expected.output_not_contains:
            for phrase in expected.output_not_contains:
                total_checks += 1
                if phrase.lower() not in output_lower:
                    passed_checks += 1
                    not_contains_passed.append(phrase)
                else:
                    not_contains_failed.append(phrase)

        # Check max_length
        if expected.max_length:
            total_checks += 1
            if len(output) <= expected.max_length:
                passed_checks += 1

        # Calculate score
        if total_checks == 0:
            # No checks defined, pass by default
            score = 100.0
            passed = True
        else:
            score = (passed_checks / total_checks) * 100
            passed = len(contains_failed) == 0 and len(not_contains_failed) == 0

        return {
            "passed": passed,
            "score": score,
            "contains_passed": contains_passed,
            "contains_failed": contains_failed,
            "not_contains_passed": not_contains_passed,
            "not_contains_failed": not_contains_failed,
        }
