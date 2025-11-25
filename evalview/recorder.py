"""Test case recorder for generating test cases from agent interactions."""

import re
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from evalview.core.types import TestCase, ExecutionTrace, TestInput, ExpectedBehavior, Thresholds
from evalview.adapters.base import AgentAdapter


class RecordedInteraction:
    """Represents a recorded agent interaction."""

    def __init__(
        self,
        query: str,
        trace: ExecutionTrace,
        timestamp: datetime,
    ):
        self.query = query
        self.trace = trace
        self.timestamp = timestamp


class TestCaseRecorder:
    """Records agent interactions and generates test cases."""

    def __init__(self, adapter: AgentAdapter):
        """
        Initialize recorder.

        Args:
            adapter: Agent adapter to use for recording
        """
        self.adapter = adapter

    async def record_interaction(self, query: str) -> RecordedInteraction:
        """
        Record a single agent interaction.

        Args:
            query: User query to send to agent

        Returns:
            RecordedInteraction with trace data
        """
        # Execute query against agent
        trace = await self.adapter.execute(query)

        return RecordedInteraction(
            query=query,
            trace=trace,
            timestamp=datetime.now(),
        )

    def generate_test_case(
        self,
        interaction: RecordedInteraction,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> TestCase:
        """
        Generate a test case from recorded interaction.

        Args:
            interaction: Recorded interaction data
            name: Optional custom name for test case
            description: Optional custom description

        Returns:
            Generated TestCase
        """
        # Extract tool names from trace
        tools = [tc.name for tc in interaction.trace.tool_calls]

        # Extract key phrases from output
        key_phrases = self._extract_key_phrases(interaction.trace.final_output)

        # Generate thresholds with 20% buffer
        thresholds = self._generate_thresholds(interaction.trace)

        # Generate default name if not provided
        if not name:
            name = self._generate_test_name(interaction.query, tools)

        # Generate default description
        if not description:
            description = f"Auto-recorded test case from interaction at {interaction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

        return TestCase(
            name=name,
            description=description,
            input=TestInput(query=interaction.query),
            expected=ExpectedBehavior(
                tools=tools if tools else None,
                sequence=tools if len(tools) > 1 else None,
                output={"contains": key_phrases} if key_phrases else None,
            ),
            thresholds=thresholds,
        )

    def _extract_key_phrases(self, output: str, max_phrases: int = 5) -> List[str]:
        """
        Extract key phrases from agent output.

        Args:
            output: Agent output text
            max_phrases: Maximum number of phrases to extract

        Returns:
            List of key phrases
        """
        if not output:
            return []

        phrases = []

        # Extract numbers (likely important data)
        numbers = re.findall(r"\b\d+\.?\d*\b", output)
        phrases.extend(numbers[:2])  # Take first 2 numbers

        # Extract capitalized words (likely entities/proper nouns)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", output)
        phrases.extend(capitalized[:2])  # Take first 2 capitalized words

        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', output)
        phrases.extend(quoted[:1])

        # Extract words after common indicators
        indicators = [
            r"is\s+(\w+)",
            r"are\s+(\w+)",
            r"shows\s+(\w+)",
            r"indicates\s+(\w+)",
        ]
        for pattern in indicators:
            matches = re.findall(pattern, output.lower())
            phrases.extend(matches[:1])

        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in phrases:
            if phrase.lower() not in seen and len(phrase) > 1:
                seen.add(phrase.lower())
                unique_phrases.append(phrase)

        return unique_phrases[:max_phrases]

    def _generate_thresholds(self, trace: ExecutionTrace) -> Thresholds:
        """
        Generate thresholds with 20% buffer from actual execution.

        Args:
            trace: Execution trace

        Returns:
            Generated thresholds
        """
        # Default min score
        min_score = 75.0

        # Cost threshold with 20% buffer (if available)
        max_cost = None
        if trace.cost is not None and trace.cost > 0:
            max_cost = round(trace.cost * 1.2, 4)

        # Latency threshold with 20% buffer (if available)
        max_latency = None
        if trace.latency is not None and trace.latency > 0:
            max_latency = int(trace.latency * 1.2)

        return Thresholds(
            min_score=min_score,
            max_cost=max_cost,
            max_latency=max_latency,
        )

    def _generate_test_name(self, query: str, tools: List[str]) -> str:
        """
        Generate a descriptive test name.

        Args:
            query: User query
            tools: List of tools used

        Returns:
            Generated test name
        """
        # Extract key words from query
        words = re.findall(r"\b\w+\b", query)
        key_words = [
            w
            for w in words
            if len(w) > 3
            and w.lower()
            not in {
                "what",
                "when",
                "where",
                "which",
                "how",
                "can",
                "could",
                "would",
                "should",
                "the",
                "this",
                "that",
                "these",
                "those",
                "with",
                "from",
                "about",
            }
        ]

        # Create name from key words
        if key_words:
            name_base = " ".join(key_words[:3]).title()
        else:
            name_base = "Agent Query"

        # Add tool hint if single tool
        if len(tools) == 1:
            name_base += f" - {tools[0]}"
        elif len(tools) > 1:
            name_base += " - Multi-Tool"

        return name_base

    def save_to_yaml(
        self,
        test_case: TestCase,
        output_path: Path,
        append_metadata: bool = True,
    ) -> None:
        """
        Save test case to YAML file.

        Args:
            test_case: Test case to save
            output_path: Path to save YAML file
            append_metadata: Whether to append recording metadata
        """
        import yaml

        # Convert test case to dict
        test_dict = {
            "name": test_case.name,
            "description": test_case.description,
            "input": {"query": test_case.input.query},
            "expected": {},
            "thresholds": {},
        }

        # Add expected fields
        if test_case.expected.tools:
            test_dict["expected"]["tools"] = test_case.expected.tools

        if test_case.expected.sequence:
            test_dict["expected"]["sequence"] = test_case.expected.sequence

        if test_case.expected.output:
            test_dict["expected"]["output"] = test_case.expected.output

        # Add thresholds
        if test_case.thresholds:
            test_dict["thresholds"]["min_score"] = test_case.thresholds.min_score
            if test_case.thresholds.max_cost:
                test_dict["thresholds"]["max_cost"] = test_case.thresholds.max_cost
            if test_case.thresholds.max_latency:
                test_dict["thresholds"]["max_latency"] = test_case.thresholds.max_latency

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(output_path, "w") as f:
            if append_metadata:
                f.write("# Auto-generated by: evalview record\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            yaml.dump(test_dict, f, default_flow_style=False, sort_keys=False)

    def generate_filename(self, base_dir: Path, prefix: str = "recorded") -> Path:
        """
        Generate unique filename for recorded test case.

        Args:
            base_dir: Base directory for test cases
            prefix: Filename prefix

        Returns:
            Unique file path
        """
        # Find next available number
        existing = list(base_dir.glob(f"{prefix}-*.yaml"))
        numbers = []
        for path in existing:
            match = re.search(rf"{prefix}-(\d+)\.yaml", path.name)
            if match:
                numbers.append(int(match.group(1)))

        next_num = max(numbers, default=0) + 1

        return base_dir / f"{prefix}-{next_num:03d}.yaml"
