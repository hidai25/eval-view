"""Agent-based skill test runner.

Orchestrates agent-based skill testing:
1. Loads test suites from YAML
2. Resolves appropriate adapter for agent type
3. Executes tests and captures traces
4. Saves JSONL traces for debugging
5. Runs two-phase evaluation
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from evalview.skills.agent_types import (
    AgentConfig,
    AgentType,
    SkillAgentTest,
    SkillAgentTestSuite,
    SkillAgentTestResult,
    SkillAgentTestSuiteResult,
    TestCategory,
)
from evalview.skills.adapters import SkillAdapterRegistry
from evalview.skills.adapters.base import SkillAgentAdapterError
from evalview.skills.evaluators import SkillTestOrchestrator
from evalview.skills.parser import SkillParser

logger = logging.getLogger(__name__)


class SkillAgentRunner:
    """Runs agent-based skill tests.

    Loads test suites, executes tests through real agents, captures
    traces, and runs two-phase evaluation.

    Usage:
        runner = SkillAgentRunner()
        suite = runner.load_test_suite("tests.yaml")
        result = await runner.run_suite(suite)
    """

    def __init__(
        self,
        verbose: bool = False,
        skip_rubric: bool = False,
        trace_dir: Optional[str] = None,
        rubric_model: Optional[str] = None,
    ):
        """Initialize skill agent runner.

        Args:
            verbose: Enable verbose logging
            skip_rubric: Skip Phase 2 rubric evaluation
            trace_dir: Directory to save JSONL traces
            rubric_model: Model override for rubric evaluation
        """
        self.verbose = verbose
        self.skip_rubric = skip_rubric
        self.trace_dir = trace_dir
        self.rubric_model = rubric_model

        self.orchestrator = SkillTestOrchestrator(
            skip_rubric=skip_rubric,
            rubric_model=rubric_model,
        )

        if verbose:
            logging.getLogger("evalview.skills").setLevel(logging.DEBUG)

    def load_test_suite(
        self,
        yaml_path: str,
        agent_type_override: Optional[AgentType] = None,
        cwd_override: Optional[str] = None,
        max_turns_override: Optional[int] = None,
    ) -> SkillAgentTestSuite:
        """Load a test suite from YAML file.

        Args:
            yaml_path: Path to YAML test file
            agent_type_override: Override agent type from CLI
            cwd_override: Override working directory from CLI
            max_turns_override: Override max turns from CLI

        Returns:
            Loaded and validated test suite
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Test suite not found: {yaml_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Resolve skill path relative to YAML file
        if "skill" in data and not Path(data["skill"]).is_absolute():
            yaml_dir = path.parent
            data["skill"] = str((yaml_dir / data["skill"]).resolve())

        # Apply CLI overrides
        if agent_type_override:
            if "agent" not in data:
                data["agent"] = {}
            data["agent"]["type"] = agent_type_override.value

        if cwd_override:
            if "agent" not in data:
                data["agent"] = {}
            data["agent"]["cwd"] = cwd_override

        if max_turns_override:
            if "agent" not in data:
                data["agent"] = {}
            data["agent"]["max_turns"] = max_turns_override

        return SkillAgentTestSuite(**data)

    async def run_suite(
        self,
        suite: SkillAgentTestSuite,
    ) -> SkillAgentTestSuiteResult:
        """Run all tests in a test suite.

        Args:
            suite: The test suite to run

        Returns:
            SkillAgentTestSuiteResult with all results
        """
        # Load the skill
        skill = SkillParser.parse_file(suite.skill)

        # Create adapter
        adapter = SkillAdapterRegistry.create(suite.agent)

        # Check adapter health
        if not await adapter.health_check():
            logger.warning(f"Adapter health check failed for {adapter.name}")

        # Setup trace directory
        trace_dir = self._setup_trace_dir(suite.name)

        # Run each test
        results: List[SkillAgentTestResult] = []
        for test in suite.tests:
            if self.verbose:
                logger.info(f"Running test: {test.name}")

            result = await self._run_test(
                adapter=adapter,
                skill=skill,
                test=test,
                config=suite.agent,
                trace_dir=trace_dir,
            )
            results.append(result)

        # Calculate stats
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = len(results) - passed_tests
        pass_rate = passed_tests / len(results) if results else 0.0

        total_latency = sum(r.latency_ms for r in results)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in results)

        # Stats by category
        by_category = self._calculate_category_stats(results)

        return SkillAgentTestSuiteResult(
            suite_name=suite.name,
            skill_name=skill.metadata.name,
            agent_type=suite.agent.type,
            passed=pass_rate >= suite.min_pass_rate,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            pass_rate=pass_rate,
            by_category=by_category,
            results=results,
            total_latency_ms=total_latency,
            avg_latency_ms=total_latency / len(results) if results else 0.0,
            total_tokens=total_tokens,
        )

    async def _run_test(
        self,
        adapter,
        skill,
        test: SkillAgentTest,
        config: AgentConfig,
        trace_dir: Optional[str],
    ) -> SkillAgentTestResult:
        """Run a single test case.

        Args:
            adapter: The skill adapter to use
            skill: The loaded skill
            test: The test case
            config: Agent configuration
            trace_dir: Directory to save traces

        Returns:
            SkillAgentTestResult
        """
        try:
            # Execute through adapter
            context = {
                "test_name": test.name,
                "cwd": config.cwd,
            }

            trace = await adapter.execute(
                skill=skill,
                query=test.input,
                context=context,
            )

            # Save trace if configured
            trace_path = None
            if trace_dir and config.capture_trace:
                trace_path = self._save_trace(trace, trace_dir)

            # Run evaluation
            result = await self.orchestrator.evaluate(
                test=test,
                trace=trace,
                cwd=config.cwd,
            )

            # Add trace path
            if trace_path:
                result.trace_path = trace_path

            return result

        except SkillAgentAdapterError as e:
            logger.error(f"Adapter error for test '{test.name}': {e}")
            return SkillAgentTestResult(
                test_name=test.name,
                category=test.category,
                passed=False,
                score=0.0,
                input_query=test.input,
                final_output="",
                error=str(e),
            )

        except Exception as e:
            logger.error(f"Error running test '{test.name}': {e}")
            return SkillAgentTestResult(
                test_name=test.name,
                category=test.category,
                passed=False,
                score=0.0,
                input_query=test.input,
                final_output="",
                error=str(e),
            )

    def _setup_trace_dir(self, suite_name: str) -> Optional[str]:
        """Setup trace directory for saving JSONL traces.

        Args:
            suite_name: Name of the test suite

        Returns:
            Path to trace directory, or None if not configured
        """
        if not self.trace_dir:
            return None

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_path = Path(self.trace_dir) / f"{suite_name}_{timestamp}"
        trace_path.mkdir(parents=True, exist_ok=True)

        return str(trace_path)

    def _save_trace(self, trace, trace_dir: str) -> str:
        """Save trace to JSONL file.

        Args:
            trace: The execution trace
            trace_dir: Directory to save to

        Returns:
            Path to saved trace file
        """
        filename = f"{trace.test_name}_{trace.session_id}.jsonl"
        filepath = Path(trace_dir) / filename

        with open(filepath, "w") as f:
            # Write metadata line
            metadata = {
                "session_id": trace.session_id,
                "skill_name": trace.skill_name,
                "test_name": trace.test_name,
                "start_time": trace.start_time.isoformat(),
                "end_time": trace.end_time.isoformat(),
                "duration_ms": trace.duration_ms,
            }
            f.write(json.dumps(metadata) + "\n")

            # Write each event
            for event in trace.events:
                event_dict = event.model_dump(mode="json", exclude_none=True)
                f.write(json.dumps(event_dict) + "\n")

            # Write summary
            summary = {
                "type": "summary",
                "tool_calls": trace.tool_calls,
                "files_created": trace.files_created,
                "files_modified": trace.files_modified,
                "commands_ran": trace.commands_ran,
                "total_input_tokens": trace.total_input_tokens,
                "total_output_tokens": trace.total_output_tokens,
                "errors": trace.errors,
            }
            f.write(json.dumps(summary) + "\n")

        return str(filepath)

    def _calculate_category_stats(
        self,
        results: List[SkillAgentTestResult],
    ) -> Dict[TestCategory, Dict[str, int]]:
        """Calculate pass/fail stats by test category.

        Args:
            results: List of test results

        Returns:
            Dict mapping category to stats dict
        """
        stats: Dict[TestCategory, Dict[str, int]] = {}

        for category in TestCategory:
            category_results = [r for r in results if r.category == category]
            if category_results:
                stats[category] = {
                    "total": len(category_results),
                    "passed": sum(1 for r in category_results if r.passed),
                    "failed": sum(1 for r in category_results if not r.passed),
                }

        return stats


async def run_agent_tests(
    test_file: str,
    agent_type: Optional[str] = None,
    trace_dir: Optional[str] = None,
    skip_rubric: bool = False,
    cwd: Optional[str] = None,
    max_turns: Optional[int] = None,
    verbose: bool = False,
    rubric_model: Optional[str] = None,
) -> SkillAgentTestSuiteResult:
    """Convenience function to run agent-based skill tests.

    Args:
        test_file: Path to YAML test file
        agent_type: Agent type string (e.g., "claude-code")
        trace_dir: Directory to save traces
        skip_rubric: Skip Phase 2 rubric evaluation
        cwd: Working directory override
        max_turns: Max turns override
        verbose: Enable verbose logging
        rubric_model: Model override for rubric evaluation

    Returns:
        SkillAgentTestSuiteResult
    """
    # Convert agent type string to enum
    agent_type_enum = None
    if agent_type:
        try:
            agent_type_enum = AgentType(agent_type)
        except ValueError:
            raise ValueError(f"Unknown agent type: {agent_type}")

    runner = SkillAgentRunner(
        verbose=verbose,
        skip_rubric=skip_rubric,
        trace_dir=trace_dir,
        rubric_model=rubric_model,
    )

    suite = runner.load_test_suite(
        yaml_path=test_file,
        agent_type_override=agent_type_enum,
        cwd_override=cwd,
        max_turns_override=max_turns,
    )

    return await runner.run_suite(suite)
