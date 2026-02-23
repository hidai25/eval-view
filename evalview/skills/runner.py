"""Skill test runner - executes skills against Anthropic or OpenAI-compatible APIs."""

import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple
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
    """Runs skill tests against Anthropic or OpenAI-compatible APIs.

    Loads a skill, sends test queries, and evaluates responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the skill runner.

        Args:
            api_key: Provider API key (or uses env vars)
            model: Model to use for testing
            provider: Provider name — "anthropic" or "openai" (covers all OpenAI-compatible APIs)
            base_url: Optional base URL for OpenAI-compatible providers
        """
        self.model = model
        self.provider, self.api_key, self.base_url = self._resolve_provider_config(
            api_key=api_key,
            provider=provider,
            base_url=base_url,
        )
        self._client: Optional[Any] = None

    @property
    def client(self):
        """Lazy-load provider client."""
        if self._client is None:
            if self.provider == "anthropic":
                try:
                    import anthropic
                except ImportError:
                    raise ImportError("anthropic package required. Install with: pip install anthropic")
                self._client = anthropic.Anthropic(api_key=self.api_key)
            else:
                try:
                    from openai import OpenAI
                except ImportError:
                    raise ImportError("openai package required. Install with: pip install openai")
                if self.base_url:
                    self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                else:
                    self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _resolve_provider_config(
        self,
        api_key: Optional[str],
        provider: Optional[str],
        base_url: Optional[str],
    ) -> Tuple[str, str, Optional[str]]:
        """Resolve provider configuration from args/env.

        Env vars supported:
        - `SKILL_TEST_PROVIDER` (`anthropic`, `openai`, `openai-compatible`)
        - `SKILL_TEST_API_KEY`
        - `SKILL_TEST_BASE_URL`
        - Anthropic: `ANTHROPIC_API_KEY`
        - OpenAI-compatible: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `KIMI_API_KEY`, `MOONSHOT_API_KEY`
        - Base URL aliases: `OPENAI_BASE_URL`, `DEEPSEEK_BASE_URL`, `KIMI_BASE_URL`, `MOONSHOT_BASE_URL`
        """
        provider = (provider or os.environ.get("SKILL_TEST_PROVIDER") or "").strip().lower() or None
        explicit_api_key = api_key or os.environ.get("SKILL_TEST_API_KEY")
        explicit_base_url = base_url or os.environ.get("SKILL_TEST_BASE_URL")

        if provider in {"openai-compatible", "openai_compatible"}:
            provider = "openai"

        if provider == "anthropic":
            key = explicit_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "API key required for Anthropic. Set ANTHROPIC_API_KEY or SKILL_TEST_API_KEY."
                )
            return "anthropic", key, None

        if provider == "openai":
            openai_key_source = None
            if explicit_api_key:
                key = explicit_api_key
            else:
                openai_key_source, key = self._first_env_item(
                    "OPENAI_API_KEY",
                    "DEEPSEEK_API_KEY",
                    "KIMI_API_KEY",
                    "MOONSHOT_API_KEY",
                )
            if not key:
                raise ValueError(
                    "API key required for OpenAI-compatible provider. "
                    "Set one of OPENAI_API_KEY / DEEPSEEK_API_KEY / KIMI_API_KEY / MOONSHOT_API_KEY "
                    "or SKILL_TEST_API_KEY."
                )
            resolved_base_url = self._resolve_openai_compatible_base_url(
                key_source=openai_key_source,
                explicit_base_url=explicit_base_url,
            )
            self._validate_openai_compatible_base_url(
                key_source=openai_key_source,
                resolved_base_url=resolved_base_url,
            )
            return "openai", key, resolved_base_url

        # Auto-detect provider preference:
        # 1) If an Anthropic key exists and no OpenAI-compatible base URL/key is set, prefer Anthropic.
        # 2) Otherwise use OpenAI-compatible if any relevant key is present.
        anthropic_key = explicit_api_key or os.environ.get("ANTHROPIC_API_KEY")
        openai_key_source = None
        openai_key: Optional[str]
        if explicit_api_key:
            openai_key = explicit_api_key
        else:
            openai_key_source, openai_key = self._first_env_item(
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "KIMI_API_KEY",
                "MOONSHOT_API_KEY",
            )
        openai_base_url = self._resolve_openai_compatible_base_url(
            key_source=openai_key_source,
            explicit_base_url=explicit_base_url,
        )

        if openai_key and (openai_base_url or not anthropic_key):
            self._validate_openai_compatible_base_url(
                key_source=openai_key_source,
                resolved_base_url=openai_base_url,
            )
            return "openai", openai_key, openai_base_url
        if anthropic_key:
            return "anthropic", anthropic_key, None
        if openai_key:
            self._validate_openai_compatible_base_url(
                key_source=openai_key_source,
                resolved_base_url=openai_base_url,
            )
            return "openai", openai_key, openai_base_url

        raise ValueError(
            "No supported API key found. Set ANTHROPIC_API_KEY for Anthropic, or "
            "an OpenAI-compatible key (OPENAI_API_KEY / DEEPSEEK_API_KEY / KIMI_API_KEY / MOONSHOT_API_KEY). "
            "Optional overrides: SKILL_TEST_PROVIDER, SKILL_TEST_API_KEY, SKILL_TEST_BASE_URL."
        )

    @staticmethod
    def _first_env(*names: str) -> Optional[str]:
        for name in names:
            value = os.environ.get(name)
            if value:
                return value
        return None

    @staticmethod
    def _first_env_item(*names: str) -> Tuple[Optional[str], Optional[str]]:
        for name in names:
            value = os.environ.get(name)
            if value:
                return name, value
        return None, None

    @staticmethod
    def _validate_openai_compatible_base_url(
        key_source: Optional[str],
        resolved_base_url: Optional[str],
    ) -> None:
        """Prevent vendor-specific keys from defaulting to OpenAI's endpoint."""
        if key_source in {"DEEPSEEK_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY"} and not resolved_base_url:
            base_var = key_source.replace("_API_KEY", "_BASE_URL")
            raise ValueError(
                f"{key_source} detected but no base URL configured. "
                f"Set {base_var} or SKILL_TEST_BASE_URL."
            )

    @staticmethod
    def _resolve_openai_compatible_base_url(
        key_source: Optional[str],
        explicit_base_url: Optional[str],
    ) -> Optional[str]:
        """Resolve base URL matched to the selected OpenAI-compatible key source.

        Order:
        1) Explicit override (`SKILL_TEST_BASE_URL` / passed arg)
        2) Matching alias-specific base URL for the selected key env var
        3) None
        """
        if explicit_base_url:
            return explicit_base_url

        key_to_base_var = {
            "OPENAI_API_KEY": "OPENAI_BASE_URL",
            "DEEPSEEK_API_KEY": "DEEPSEEK_BASE_URL",
            "KIMI_API_KEY": "KIMI_BASE_URL",
            "MOONSHOT_API_KEY": "MOONSHOT_BASE_URL",
        }
        if not key_source:
            return None
        base_var = key_to_base_var.get(key_source)
        if not base_var:
            return None
        return os.environ.get(base_var)

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

        # Call provider
        start_time = time.time()
        try:
            output, input_tokens, output_tokens = self._invoke_model(
                model=model,
                system_prompt=system_prompt,
                user_input=test.input,
            )
            latency_ms = (time.time() - start_time) * 1000
            error = None

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            output = ""
            input_tokens = 0
            output_tokens = 0
            error = self._categorize_model_error(e)

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

    def _invoke_model(self, model: str, system_prompt: str, user_input: str) -> Tuple[str, int, int]:
        """Invoke configured provider and normalize response."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
            )
            output = ""
            if getattr(response, "content", None):
                chunks = []
                for block in response.content:
                    text = getattr(block, "text", None)
                    if text:
                        chunks.append(text)
                output = "\n".join(chunks).strip()
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            return output, input_tokens, output_tokens

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=4096,
        )
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "")
        if isinstance(content, list):
            # Some OpenAI-compatible providers may return structured content blocks.
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(str(text))
            output = "\n".join(p for p in parts if p).strip()
        else:
            output = str(content or "")
        usage = getattr(response, "usage", None)
        input_tokens = int(
            getattr(usage, "prompt_tokens", 0)
            or getattr(usage, "input_tokens", 0)
            or 0
        )
        output_tokens = int(
            getattr(usage, "completion_tokens", 0)
            or getattr(usage, "output_tokens", 0)
            or 0
        )
        return output, input_tokens, output_tokens

    @staticmethod
    def _categorize_model_error(exc: Exception) -> str:
        """Return a user-facing error label with light categorization."""
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()

        if any(token in lowered for token in ("authentication", "unauthorized", "invalid api key", "401", "forbidden", "403")):
            return f"Authentication error: {message}"
        if any(token in lowered for token in ("api key", "base url", "base_url", "provider")):
            return f"Provider configuration error: {message}"
        if "timeout" in lowered:
            return f"Timeout error: {message}"
        if any(token in lowered for token in ("connection", "dns", "name resolution", "network", "nodename nor servname")):
            return f"Connection error: {message}"
        return message

    def _build_system_prompt(self, skill: Skill) -> str:
        """Build system prompt with skill loaded."""
        return f"""You are an AI assistant with the following skill loaded:

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
