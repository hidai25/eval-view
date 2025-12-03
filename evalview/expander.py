"""Test case expander - LLM-assisted test variation generation."""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import yaml  # type: ignore[import-untyped]

from evalview.core.types import TestCase, TestInput, ExpectedBehavior, Thresholds


class TestExpander:
    """Expands test cases into variations using LLM."""

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize expander with auto-detected or specified provider.

        Args:
            provider: Force a specific provider ('openai' or 'anthropic').
                     If None, auto-detects based on available API keys.
        """
        self.provider, self.message = self._detect_provider(provider)
        self.client: Any = None
        self._init_client()

    def _detect_provider(self, forced_provider: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Detect which provider to use based on available API keys."""
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if forced_provider:
            return forced_provider, None

        if openai_key and anthropic_key:
            return "openai", "Both OpenAI and Anthropic keys found, using OpenAI for expansion"
        elif openai_key:
            return "openai", None
        elif anthropic_key:
            return "anthropic", None
        else:
            raise ValueError(
                "No API key found. Set either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
            )

    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def expand(
        self,
        base_test: TestCase,
        count: int = 10,
        include_edge_cases: bool = True,
        variation_focus: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate test variations from a base test.

        Args:
            base_test: Base test case to expand
            count: Number of variations to generate
            include_edge_cases: Include edge cases (empty input, invalid data, etc.)
            variation_focus: Optional focus for variations (e.g., "different tickers")

        Returns:
            List of generated test case dictionaries
        """
        # Build the prompt
        prompt = self._build_expansion_prompt(
            base_test, count, include_edge_cases, variation_focus
        )

        system_prompt = """You are a test case generator for AI agents. Given a base test case, generate variations that test the same functionality with different inputs.

Rules:
1. Vary the inputs meaningfully (different entities, values, edge cases)
2. For edge cases: test empty inputs, invalid data, boundary conditions
3. Output valid JSON array of test cases
4. Each test MUST specify expected_tools - only the tools actually needed for that specific query
5. Keep expected_contains reasonable - only include things that MUST be in the output
6. Use flexible patterns in expected_contains (e.g., use "weather" not "weather in London")

IMPORTANT for expected_tools:
- Analyze each query to determine which tools it actually needs
- If a query only asks about weather, only include weather tools
- If a query only asks about conversion, only include conversion tools
- Only include multiple tools if the query explicitly requires multiple operations

Return JSON in this format:
{
  "tests": [
    {
      "name": "Test Name",
      "description": "What this tests",
      "query": "The actual query to send",
      "expected_tools": ["tool1"],
      "expected_contains": ["word1", "word2"],
      "is_edge_case": false
    }
  ]
}"""

        # Call appropriate provider
        if self.provider == "openai":
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
        else:  # anthropic
            response = await self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract JSON from response
            content = response.content[0].text
            # Try to parse JSON, handling potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())

        variations = result.get("tests", result.get("variations", []))

        return variations

    def _build_expansion_prompt(
        self,
        base_test: TestCase,
        count: int,
        include_edge_cases: bool,
        variation_focus: Optional[str],
    ) -> str:
        """Build the LLM prompt for test expansion."""

        # Extract base test info
        base_query = base_test.input.query
        base_name = base_test.name
        base_description = base_test.description or ""

        # Expected output patterns
        expected_contains = []
        if base_test.expected.output and isinstance(base_test.expected.output, dict):
            expected_contains = base_test.expected.output.get("contains", [])

        # Get available tools from base test
        available_tools = []
        if base_test.tools:
            for tool in base_test.tools:
                tool_name = tool.get("name", "") if isinstance(tool, dict) else str(tool)
                tool_desc = tool.get("description", "") if isinstance(tool, dict) else ""
                available_tools.append({"name": tool_name, "description": tool_desc})

        # Get expected tools from base test
        base_expected_tools = base_test.expected.tools or []

        prompt = f"""Base test case:
- Name: {base_name}
- Description: {base_description}
- Query: "{base_query}"
- Expected tools for base query: {base_expected_tools}
- Expected output contains: {expected_contains}

Available tools that the agent can use:
{json.dumps(available_tools, indent=2)}

Generate {count} test variations.

IMPORTANT: For each variation, analyze what the query actually needs and set expected_tools accordingly.
- If a query only asks about weather for cities, expected_tools should only include "get_weather"
- If a query only asks about temperature conversion, expected_tools should only include "convert_temperature"
- Only include both tools if the query explicitly asks for both weather AND conversion
"""

        if include_edge_cases:
            edge_count = min(3, count // 3)
            prompt += f"""
Include {edge_count} edge cases such as:
- Empty or minimal input
- Invalid/malformed input
- Boundary conditions
- Unexpected but valid input
"""

        if variation_focus:
            prompt += f"""
Focus variations on: {variation_focus}
"""

        prompt += """
Return JSON in this format:
{
  "tests": [
    {
      "name": "Test Name",
      "description": "What this tests",
      "query": "The actual query to send",
      "expected_tools": ["tool1"],
      "expected_contains": ["word1", "word2"],
      "is_edge_case": false
    }
  ]
}
"""
        return prompt

    def convert_to_test_case(
        self,
        variation: Dict[str, Any],
        base_test: TestCase,
        index: int,
    ) -> TestCase:
        """
        Convert a variation dict to a TestCase object.

        Args:
            variation: Generated variation dictionary
            base_test: Original base test (for inheriting thresholds, etc.)
            index: Variation index for naming

        Returns:
            TestCase object
        """
        # Build expected behavior
        expected_contains = variation.get("expected_contains", [])

        # Use variation-specific expected_tools if provided, otherwise fall back to base
        expected_tools = variation.get("expected_tools", base_test.expected.tools)

        expected = ExpectedBehavior(
            tools=expected_tools,
            output={"contains": expected_contains} if expected_contains else None,
        )

        # Inherit thresholds from base test, with buffer for edge cases
        if base_test.thresholds:
            # Edge cases might fail or be slower, so relax thresholds
            is_edge = variation.get("is_edge_case", False)
            thresholds = Thresholds(
                min_score=base_test.thresholds.min_score * (0.7 if is_edge else 1.0),
                max_cost=base_test.thresholds.max_cost * (1.5 if is_edge else 1.2) if base_test.thresholds.max_cost else None,
                max_latency=base_test.thresholds.max_latency * (1.5 if is_edge else 1.2) if base_test.thresholds.max_latency else None,
            )
        else:
            thresholds = Thresholds(min_score=0.0)

        # Inherit input context (e.g., system_prompt) from base test
        input_context = base_test.input.context if base_test.input.context else None

        return TestCase(
            name=variation.get("name", f"{base_test.name} - Variation {index}"),
            description=variation.get("description", f"Auto-generated variation of {base_test.name}"),
            input=TestInput(query=variation.get("query", ""), context=input_context),
            expected=expected,
            thresholds=thresholds,
            adapter=base_test.adapter,
            endpoint=base_test.endpoint,
            adapter_config=base_test.adapter_config,
            tools=base_test.tools,  # Inherit tool definitions from base test
            model=base_test.model,  # Inherit model from base test
        )

    def save_variations(
        self,
        variations: List[TestCase],
        output_dir: Path,
        prefix: str = "expanded",
    ) -> List[Path]:
        """
        Save generated variations to YAML files.

        Args:
            variations: List of TestCase objects
            output_dir: Directory to save files
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for i, test_case in enumerate(variations, 1):
            # Generate filename
            filename = f"{prefix}-{i:03d}.yaml"
            filepath = output_dir / filename

            # Convert to dict
            input_dict: Dict[str, Any] = {"query": test_case.input.query}
            if test_case.input.context:
                input_dict["context"] = test_case.input.context

            test_dict: Dict[str, Any] = {
                "name": test_case.name,
                "description": test_case.description,
                "input": input_dict,
                "expected": {},
            }

            # Add expected fields
            if test_case.expected.tools:
                test_dict["expected"]["tools"] = test_case.expected.tools
            if test_case.expected.output:
                # Convert Pydantic model to dict if needed, filtering out None values
                if hasattr(test_case.expected.output, "model_dump"):
                    output_dict = test_case.expected.output.model_dump(exclude_none=True)
                elif hasattr(test_case.expected.output, "dict"):
                    output_dict = {k: v for k, v in test_case.expected.output.dict().items() if v is not None}
                elif isinstance(test_case.expected.output, dict):
                    output_dict = {k: v for k, v in test_case.expected.output.items() if v is not None}
                else:
                    output_dict = test_case.expected.output
                test_dict["expected"]["output"] = output_dict

            # Add thresholds
            if test_case.thresholds:
                test_dict["thresholds"] = {
                    "min_score": test_case.thresholds.min_score,
                }
                if test_case.thresholds.max_cost:
                    test_dict["thresholds"]["max_cost"] = test_case.thresholds.max_cost
                if test_case.thresholds.max_latency:
                    test_dict["thresholds"]["max_latency"] = test_case.thresholds.max_latency

            # Add adapter config if present
            if test_case.adapter:
                test_dict["adapter"] = test_case.adapter
            if test_case.model:
                test_dict["model"] = test_case.model
            if test_case.endpoint:
                test_dict["endpoint"] = test_case.endpoint

            # Copy tool definitions from base test (important for adapters like Anthropic)
            if test_case.tools:
                test_dict["tools"] = test_case.tools

            # Write file
            with open(filepath, "w") as f:
                f.write("# Auto-generated by: evalview expand\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                yaml.dump(test_dict, f, default_flow_style=False, sort_keys=False)

            saved_paths.append(filepath)

        return saved_paths
