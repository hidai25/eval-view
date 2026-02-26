"""Test generator for Claude Code skills.

Auto-generates comprehensive test suites from SKILL.md files using LLM-powered few-shot learning.
"""

import uuid
import json
import logging
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, NamedTuple, Union, cast
from datetime import datetime

import yaml  # type: ignore[import-untyped]

from evalview.core.llm_provider import (
    LLMClient,
    LLMProvider,
    detect_available_providers,
    PROVIDER_CONFIGS,
)
from evalview.skills.parser import SkillParser
from evalview.skills.agent_types import (
    SkillAgentTestSuite,
    SkillAgentTest,
    TestCategory,
    DeterministicExpected,
    RubricConfig,
    AgentConfig,
    AgentType,
)
from evalview.skills.types import Skill

logger = logging.getLogger(__name__)


class ModelOption(NamedTuple):
    """Available model option with cost info."""

    provider: LLMProvider
    model: str
    display_name: str
    cost_per_gen: float  # Estimated cost per generation
    description: str


class SkillTestGenerator:
    """Auto-generates test suites for Claude Code skills using LLM.

    Uses few-shot learning from golden examples to generate high-quality
    test cases across all categories: explicit, implicit, contextual, negative.
    """

    # Available models by provider (name, cost_per_gen, description)
    AVAILABLE_MODELS = {
        "anthropic": [
            ("claude-haiku-4-5-20251001", 0.004, "Fast & cheap (recommended)"),
            ("claude-sonnet-4-5-20250929", 0.020, "Higher quality, slower"),
            ("claude-opus-4-6", 0.050, "Best quality, expensive"),
        ],
        "openai": [
            ("gpt-4o-mini", 0.008, "Fast & affordable (recommended)"),
            ("gpt-4o", 0.030, "Higher quality"),
            ("gpt-5", 0.040, "Latest model (beta)"),
        ],
        "gemini": [
            ("gemini-2.0-flash", 0.001, "Free tier (recommended)"),
            ("gemini-1.5-pro", 0.015, "Higher quality"),
        ],
        "deepseek": [
            ("deepseek-chat", 0.0007, "Fast & ultra-cheap (recommended)"),
            ("deepseek-reasoner", 0.0055, "Best quality, slower"),
        ],
    }

    # Cost-optimized defaults (first model in each provider list)
    DEFAULT_MODELS = {
        "anthropic": "claude-haiku-4-5-20251001",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "deepseek": "deepseek-chat",
    }

    def __init__(self, model: Optional[str] = None):
        """Initialize generator with LLM client.

        Args:
            model: Optional model override. If None, auto-selects cheapest available.
        """
        # Auto-select provider and model if not specified
        if model:
            # User specified a model, let LLMClient handle provider selection
            self.client = LLMClient(model=model)
        else:
            # Auto-select cheapest provider, ignoring EVAL_PROVIDER
            provider_info = self._auto_select_provider()
            auto_model = self.DEFAULT_MODELS.get(provider_info.provider.value, "claude-haiku-4-5-20251001")
            self.client = LLMClient(
                provider=provider_info.provider,
                api_key=provider_info.api_key,
                model=auto_model
            )
            logger.debug(f"Auto-selected: {provider_info.provider.value} / {auto_model}")

        self.generation_id = str(uuid.uuid4())
        self.generation_cost = 0.0

    def _auto_select_provider(self):
        """Select cheapest available provider based on API keys (ignoring EVAL_PROVIDER)."""
        from evalview.core.llm_provider import AvailableProvider

        providers = detect_available_providers()
        if not providers:
            raise ValueError(
                "No LLM provider available. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or DEEPSEEK_API_KEY"
            )

        # Return first available (detect_available_providers already sorts by priority)
        return providers[0]

    @classmethod
    def get_available_models(cls) -> List[ModelOption]:
        """Get all available models across all providers with API keys.

        Returns:
            List of ModelOption sorted by cost (cheapest first)
        """
        providers = detect_available_providers()
        options = []

        for provider_info in providers:
            provider_name = provider_info.provider.value
            if provider_name not in cls.AVAILABLE_MODELS:
                continue

            provider_config = PROVIDER_CONFIGS[provider_info.provider]
            for model, cost, desc in cls.AVAILABLE_MODELS[provider_name]:
                options.append(
                    ModelOption(
                        provider=provider_info.provider,
                        model=model,
                        display_name=f"{provider_config.display_name} - {model}",
                        cost_per_gen=cost,
                        description=desc,
                    )
                )

        # Sort by cost (cheapest first)
        options.sort(key=lambda x: x.cost_per_gen)
        return options

    @classmethod
    def select_model_interactive(cls, console) -> Tuple[LLMProvider, str, str]:
        """Interactive model selection with cost display.

        Args:
            console: Rich Console instance

        Returns:
            Tuple of (provider, api_key, model)
        """
        from rich.table import Table
        import click

        options = cls.get_available_models()

        if not options:
            raise ValueError(
                "No LLM provider available. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or DEEPSEEK_API_KEY"
            )

        # If only one option, use it
        if len(options) == 1:
            option = options[0]
            providers = detect_available_providers()
            api_key = next(p.api_key for p in providers if p.provider == option.provider)
            return option.provider, api_key, option.model

        # Show table
        console.print()
        console.print("[bold]Available Models for Test Generation[/bold]")
        console.print()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Model", style="cyan")
        table.add_column("Cost/Generation", justify="right", style="green")
        table.add_column("Description", style="dim")

        for i, option in enumerate(options, 1):
            cost_str = f"${option.cost_per_gen:.4f}" if option.cost_per_gen > 0 else "Free"
            table.add_row(
                str(i),
                option.display_name,
                cost_str,
                option.description,
            )

        console.print(table)
        console.print()

        # Prompt for selection
        console.print(
            f"[dim]Cheapest option: {options[0].display_name} (${options[0].cost_per_gen:.4f}/gen)[/dim]"
        )
        console.print()

        choice = click.prompt(
            "Select model",
            type=click.IntRange(1, len(options)),
            default=1,
            show_default=True,
        )

        selected = options[choice - 1]
        console.print()
        console.print(
            f"[green]✓[/green] Selected: [bold]{selected.display_name}[/bold] "
            f"(~${selected.cost_per_gen:.4f}/generation estimate)"
        )

        # Get API key for this provider
        providers = detect_available_providers()
        api_key = next(p.api_key for p in providers if p.provider == selected.provider)

        return selected.provider, api_key, selected.model

    async def generate_test_suite(
        self,
        skill: Skill,
        count: int = 10,
        categories: Optional[List[TestCategory]] = None,
    ) -> SkillAgentTestSuite:
        """Generate complete test suite for a skill.

        Args:
            skill: Parsed skill object
            count: Number of tests to generate (default: 10, max: 50)
            categories: Test categories to include (default: all 4)

        Returns:
            Complete SkillAgentTestSuite ready to save

        Raises:
            ValueError: If generation fails after retries or invalid parameters
        """
        # Input validation
        if not isinstance(count, int):
            raise ValueError(f"count must be an integer, got {type(count).__name__}")

        if not 1 <= count <= 50:
            raise ValueError(
                f"count must be between 1 and 50 to prevent excessive costs. Got: {count}\n"
                f"For large test suites, generate multiple times or increase limit in code."
            )

        if categories is None:
            categories = [
                TestCategory.EXPLICIT,
                TestCategory.IMPLICIT,
                TestCategory.CONTEXTUAL,
                TestCategory.NEGATIVE,
            ]

        # Load golden example for few-shot learning
        golden_examples = self._load_golden_example()

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(skill, count, categories, golden_examples)

        # Generate with retry logic and exponential backoff
        last_error = None
        for attempt in range(3):
            try:
                # Call LLM
                logger.debug(f"Generation attempt {attempt + 1}/3")
                response = await self.client.chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    max_tokens=4096,
                )

                # Parse response
                tests_data = self._parse_llm_response(response)

                if not tests_data:
                    raise ValueError("LLM returned empty test list")

                # Convert to SkillAgentTest objects
                tests = [self._dict_to_test(t) for t in tests_data]

                # Build suite
                suite = SkillAgentTestSuite(
                    name=f"test-{skill.metadata.name}",
                    description=f"Auto-generated tests for {skill.metadata.name}",
                    skill=skill.file_path or "./SKILL.md",
                    agent=AgentConfig(type=AgentType.CLAUDE_CODE),
                    tests=tests,
                    min_pass_rate=0.8,
                )

                # Estimate cost
                self.generation_cost = self._estimate_cost(user_prompt, str(response))

                logger.info(f"Generated {len(tests)} tests (cost: ~${self.generation_cost:.4f})")
                return suite

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt + 1}/3 failed: {e}")

                if attempt < 2:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff_seconds = 2 ** attempt
                    logger.debug(f"Retrying in {backoff_seconds}s...")
                    await asyncio.sleep(backoff_seconds)
                    continue
                else:
                    # Final attempt failed
                    error_type = type(last_error).__name__
                    raise ValueError(
                        f"Generation failed after 3 attempts with exponential backoff.\n"
                        f"Last error ({error_type}): {str(last_error)[:200]}\n\n"
                        f"Possible causes:\n"
                        f"  - LLM returned invalid JSON (try different model with --model)\n"
                        f"  - Rate limit exceeded (wait a minute and retry)\n"
                        f"  - Skill instructions too complex (simplify SKILL.md)\n"
                    )

        # Unreachable: loop always returns or raises
        raise RuntimeError("Unreachable code: all retry attempts exhausted")

    def _load_golden_example(self) -> List[Dict[str, Any]]:
        """Load code-reviewer tests.yaml as few-shot example.

        Returns up to 3 example tests from the golden suite.
        """
        # Use bundled package location only (not hard-coded user paths)
        golden_path = (
            Path(__file__).parent.parent / "examples" / "skills" / "code-reviewer" / "tests.yaml"
        )

        if golden_path.exists():
            try:
                with open(golden_path) as f:
                    suite = yaml.safe_load(f)
                    tests = suite.get("tests", [])[:3]  # First 3 examples
                    logger.debug(f"Loaded {len(tests)} golden examples from {golden_path}")
                    return tests
            except Exception as e:
                logger.warning(f"Failed to load golden example: {e}")
                return []

        logger.warning(
            f"Golden example not found at {golden_path}. "
            "Proceeding without few-shot examples (quality may be reduced)."
        )
        return []

    def _build_system_prompt(self) -> str:
        """Build system prompt with category definitions and rules."""
        return """You are an expert test engineer for Claude Code skills.

Generate comprehensive test cases across 4 categories per OpenAI eval guidelines:

1. EXPLICIT - Direct skill invocation
   Example: "Use the code-reviewer skill to check this code"

2. IMPLICIT - Natural language implying skill use
   Example: "Can you review this code for bugs?"

3. CONTEXTUAL - Realistic, noisy prompts with multiple concerns
   Example: "I'm reviewing PR #123. Make sure it's secure. Also tests are failing."

4. NEGATIVE - Should NOT trigger skill
   Example: "How do I install Python?" (for code-reviewer skill)

Output JSON with this exact structure:
{
  "tests": [
    {
      "name": "test-name-kebab-case",
      "category": "explicit",
      "description": "What this tests",
      "input": "User query",
      "should_trigger": true,
      "expected": {
        "tool_calls_contain": ["Read", "Write"],
        "files_created": ["code-review.md"],
        "output_contains": ["security", "vulnerability"]
      },
      "rubric": {
        "prompt": "Evaluate if the agent:\\n1. Found the issue\\n2. Explained it well",
        "min_score": 70
      }
    }
  ]
}

Assertion inference rules:
- If skill uses Write tool → files_created: ["*.md"] or specific filename
- If skill uses Bash tool → commands_ran: ["command"]
- If skill mentions security → output_contains: ["security", "vulnerability"]
- If skill mentions code review → output_contains: ["review"]
- Negative tests MUST have should_trigger: false and use output_not_contains

Important:
- Test names must be kebab-case
- Rubric prompts should be specific and measurable
- Expected assertions should be realistic (don't over-specify)
- Include both deterministic checks (expected) and quality checks (rubric)
"""

    def _build_user_prompt(
        self,
        skill: Skill,
        count: int,
        categories: List[TestCategory],
        golden_examples: List[Dict[str, Any]],
    ) -> str:
        """Build user prompt with skill details and few-shot examples."""
        # Calculate distribution (3-3-2-2 for 10 tests)
        distribution = self._calculate_distribution(count, categories)

        # Truncate instructions to avoid token overflow
        original_length = len(skill.instructions)
        instructions = skill.instructions[:2000]

        if original_length > 2000:
            instructions += "\n\n[... truncated for brevity ...]"
            logger.warning(
                f"Skill instructions truncated from {original_length} to 2000 chars. "
                f"Consider simplifying SKILL.md for better test generation quality."
            )

        prompt = f"""Generate {count} tests for this skill:

## Skill Metadata
- Name: {skill.metadata.name}
- Description: {skill.metadata.description}
- Tools: {skill.metadata.tools or 'Not specified'}
- Triggers: {skill.metadata.triggers or 'Not specified'}

## Skill Instructions
{instructions}

## Test Distribution
{self._format_distribution(distribution)}

"""

        # Add few-shot examples
        if golden_examples:
            prompt += "\n## Example Tests (for reference)\n\n"
            for i, ex in enumerate(golden_examples[:2], 1):
                prompt += f"### Example {i}\n```yaml\n{yaml.dump(ex, default_flow_style=False)}```\n\n"

        prompt += "\nGenerate similar high-quality tests for the new skill above. Return valid JSON only."
        return prompt

    def _calculate_distribution(
        self, count: int, categories: List[TestCategory]
    ) -> Dict[TestCategory, int]:
        """Calculate equal distribution of tests across categories."""
        if len(categories) == 4:
            # Equal split with remainder distributed: 3+3+2+2 for 10 tests
            base = count // 4
            remainder = count % 4

            distribution = {
                TestCategory.EXPLICIT: base + (1 if remainder > 0 else 0),
                TestCategory.IMPLICIT: base + (1 if remainder > 1 else 0),
                TestCategory.CONTEXTUAL: base + (1 if remainder > 2 else 0),
                TestCategory.NEGATIVE: base,
            }
        else:
            # Distribute evenly across provided categories
            per_category = count // len(categories)
            remainder = count % len(categories)
            distribution = {}
            for i, cat in enumerate(categories):
                distribution[cat] = per_category + (1 if i < remainder else 0)

        return distribution

    def _format_distribution(self, distribution: Dict[TestCategory, int]) -> str:
        """Format distribution for prompt."""
        lines = []
        for cat, cnt in distribution.items():
            lines.append(f"- {cat.value}: {cnt} tests")
        return "\n".join(lines)

    def _parse_llm_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse LLM response to extract tests array.

        Handles both direct JSON and markdown-wrapped JSON.
        """
        # If response is already parsed
        if isinstance(response, dict):
            if "tests" in response:
                return response["tests"]
            # Some models return the tests directly
            if isinstance(response.get("message"), dict):
                content = response["message"].get("content", "")
            else:
                content = str(response)
        else:
            content = str(response)

        # Try to extract JSON from content
        if isinstance(content, str):
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            content = content.strip()
            parsed = json.loads(content)

            if isinstance(parsed, dict):
                return parsed.get("tests", [])
            elif isinstance(parsed, list):
                return parsed

        raise ValueError("Could not parse tests from LLM response")

    def _dict_to_test(self, data: Dict[str, Any]) -> SkillAgentTest:
        """Convert LLM JSON to SkillAgentTest object."""
        # Build expected assertions
        expected = None
        if data.get("expected"):
            expected = DeterministicExpected(**data["expected"])

        # Build rubric
        rubric = None
        if data.get("rubric"):
            rubric = RubricConfig(**data["rubric"])

        return SkillAgentTest(
            name=data["name"],
            description=data.get("description"),
            input=data["input"],
            category=TestCategory(data["category"]),
            should_trigger=data.get("should_trigger", True),
            expected=expected,
            rubric=rubric,
        )

    def validate_test_suite(self, suite: SkillAgentTestSuite) -> List[str]:
        """Pre-save validation.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check unique names
        names = [t.name for t in suite.tests]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            errors.append(f"Duplicate test names: {', '.join(set(duplicates))}")

        # Check negative tests
        for test in suite.tests:
            if test.category == TestCategory.NEGATIVE and test.should_trigger:
                errors.append(
                    f"Test '{test.name}' is NEGATIVE category but should_trigger=True"
                )

        # Check non-empty assertions for positive tests
        for test in suite.tests:
            if test.should_trigger and not test.expected and not test.rubric:
                errors.append(
                    f"Test '{test.name}' has no assertions (expected or rubric)"
                )

        return errors

    def save_as_yaml(self, suite: SkillAgentTestSuite, path: Path):
        """Save test suite as YAML with metadata header using atomic write.

        Uses temp file + rename pattern to ensure file is never left in corrupted state.

        Args:
            suite: Test suite to save
            path: Output path
        """
        output = {
            "name": suite.name,
            "description": suite.description,
            "skill": suite.skill,
            "agent": {
                "type": suite.agent.type.value,
                "max_turns": suite.agent.max_turns,
                "timeout": suite.agent.timeout,
            },
            "min_pass_rate": suite.min_pass_rate,
            "tests": [self._serialize_test(t) for t in suite.tests],
        }

        # Ensure parent directory exists
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first (atomic operation)
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".yaml",
            dir=path.parent,  # Same directory for atomic rename
        ) as tmp:
            # Write metadata header
            tmp.write(f"# Auto-generated by: evalview skill generate-tests\n")
            tmp.write(f"# Generation ID: {self.generation_id}\n")
            tmp.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            tmp.write(f"# Model: {self.client.model}\n")
            tmp.write(f"# Cost: ~${self.generation_cost:.4f} (estimate)\n\n")

            # Write YAML content
            yaml.dump(output, tmp, default_flow_style=False, sort_keys=False)
            tmp_path = tmp.name

        # Atomic rename (OS-level atomic operation on Unix/Linux/macOS)
        # On Windows, this may not be fully atomic, but still safer than direct write
        try:
            shutil.move(tmp_path, path)
            logger.info(f"Saved test suite to {path}")
        except Exception as e:
            # Cleanup temp file if rename fails
            try:
                Path(tmp_path).unlink()
            except:
                pass
            raise RuntimeError(f"Failed to save test suite to {path}: {e}")

    def _serialize_test(self, test: SkillAgentTest) -> Dict[str, Any]:
        """Serialize test to dict for YAML output."""
        result: Dict[str, Any] = {
            "name": test.name,
            "category": test.category.value,
            "input": test.input,
            "should_trigger": test.should_trigger,
        }

        if test.description:
            result["description"] = test.description

        if test.expected:
            result["expected"] = self._serialize_expected(test.expected)

        if test.rubric:
            result["rubric"] = {
                "prompt": test.rubric.prompt,
                "min_score": test.rubric.min_score,
            }

        return result

    def _serialize_expected(self, expected: DeterministicExpected) -> Dict[str, Any]:
        """Serialize expected assertions to dict (only non-None fields)."""
        result: Dict[str, Any] = {}

        if expected.tool_calls_contain:
            result["tool_calls_contain"] = expected.tool_calls_contain
        if expected.tool_calls_not_contain:
            result["tool_calls_not_contain"] = expected.tool_calls_not_contain
        if expected.tool_sequence:
            result["tool_sequence"] = expected.tool_sequence

        if expected.files_created:
            result["files_created"] = expected.files_created
        if expected.files_modified:
            result["files_modified"] = expected.files_modified
        if expected.files_not_modified:
            result["files_not_modified"] = expected.files_not_modified

        if expected.commands_ran:
            result["commands_ran"] = expected.commands_ran
        if expected.commands_not_ran:
            result["commands_not_ran"] = expected.commands_not_ran

        if expected.output_contains:
            result["output_contains"] = expected.output_contains
        if expected.output_not_contains:
            result["output_not_contains"] = expected.output_not_contains

        if expected.max_output_tokens:
            result["max_output_tokens"] = expected.max_output_tokens
        if expected.max_input_tokens:
            result["max_input_tokens"] = expected.max_input_tokens
        if expected.max_total_tokens:
            result["max_total_tokens"] = expected.max_total_tokens

        return result

    def get_generation_cost(self) -> float:
        """Return estimated generation cost in USD."""
        return self.generation_cost

    def get_category_distribution(self, suite: SkillAgentTestSuite) -> Dict[str, int]:
        """Get test count per category for telemetry.

        Returns:
            Dict mapping category name to count
        """
        distribution: Dict[str, int] = {}
        for test in suite.tests:
            cat = test.category.value
            distribution[cat] = distribution.get(cat, 0) + 1
        return distribution

    def _estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate generation cost based on tokens.

        Uses rough 4 chars/token heuristic and model pricing.
        Note: This is an approximation and may vary ±40% from actual cost.
        For accurate tracking, check your provider's billing dashboard.
        """
        # Rough estimate: 4 chars per token
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4

        # Model pricing (per 1M tokens) - input, output
        pricing = {
            "claude-haiku-4-5-20251001": (0.25, 1.25),
            "gpt-4o-mini": (0.15, 0.60),
            "gemini-2.0-flash": (0.0, 0.0),  # Free tier
            "gpt-4o": (2.50, 10.00),
            "claude-sonnet-4-5-20250929": (3.00, 15.00),
            "deepseek-chat": (0.14, 0.28),
            "deepseek-reasoner": (0.55, 2.19),
        }

        model = self.client.model
        input_price, output_price = pricing.get(model, (0.15, 0.60))  # Default to gpt-4o-mini

        cost = (input_tokens / 1_000_000 * input_price) + (
            output_tokens / 1_000_000 * output_price
        )
        return cost
