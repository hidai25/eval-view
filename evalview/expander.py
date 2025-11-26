"""Test case expander - LLM-assisted test variation generation."""

import os
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import yaml
from openai import AsyncOpenAI

from evalview.core.types import TestCase, TestInput, ExpectedBehavior, Thresholds


class TestExpander:
    """Expands test cases into variations using LLM."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize expander.

        Args:
            openai_api_key: OpenAI API key (uses env var if not provided)
        """
        self.client = AsyncOpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

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

        # Call LLM
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a test case generator for AI agents. Given a base test case, generate variations that test the same functionality with different inputs.

Rules:
1. Keep the same structure and expected behavior pattern
2. Vary the inputs meaningfully (different entities, values, edge cases)
3. For edge cases: test empty inputs, invalid data, boundary conditions
4. Output valid JSON array of test cases
5. Each test should have: name, description, query, expected_contains (list of strings that should appear in output)
6. Keep expected_contains reasonable - only include things that MUST be in the output"""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        # Parse response
        result = json.loads(response.choices[0].message.content)
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

        prompt = f"""Base test case:
- Name: {base_name}
- Description: {base_description}
- Query: "{base_query}"
- Expected output contains: {expected_contains}

Generate {count} test variations.
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

        expected = ExpectedBehavior(
            tools=base_test.expected.tools,  # Inherit tools expectation
            output={"contains": expected_contains} if expected_contains else None,
        )

        # Inherit thresholds from base test, with buffer for edge cases
        thresholds = None
        if base_test.thresholds:
            # Edge cases might fail or be slower, so relax thresholds
            is_edge = variation.get("is_edge_case", False)
            thresholds = Thresholds(
                min_score=base_test.thresholds.min_score * (0.7 if is_edge else 1.0),
                max_cost=base_test.thresholds.max_cost * (1.5 if is_edge else 1.2) if base_test.thresholds.max_cost else None,
                max_latency=base_test.thresholds.max_latency * (1.5 if is_edge else 1.2) if base_test.thresholds.max_latency else None,
            )

        return TestCase(
            name=variation.get("name", f"{base_test.name} - Variation {index}"),
            description=variation.get("description", f"Auto-generated variation of {base_test.name}"),
            input=TestInput(query=variation.get("query", "")),
            expected=expected,
            thresholds=thresholds,
            adapter=base_test.adapter,
            endpoint=base_test.endpoint,
            adapter_config=base_test.adapter_config,
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
            test_dict = {
                "name": test_case.name,
                "description": test_case.description,
                "input": {"query": test_case.input.query},
                "expected": {},
            }

            # Add expected fields
            if test_case.expected.tools:
                test_dict["expected"]["tools"] = test_case.expected.tools
            if test_case.expected.output:
                test_dict["expected"]["output"] = test_case.expected.output

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
            if test_case.endpoint:
                test_dict["endpoint"] = test_case.endpoint

            # Write file
            with open(filepath, "w") as f:
                f.write(f"# Auto-generated by: evalview expand\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                yaml.dump(test_dict, f, default_flow_style=False, sort_keys=False)

            saved_paths.append(filepath)

        return saved_paths
