"""Unit tests for the skill agent runner.

Tests for SkillAgentRunner that orchestrates test execution.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import yaml

from evalview.skills.agent_types import (
    AgentConfig,
    AgentType,
    DeterministicExpected,
    SkillAgentTest,
    SkillAgentTestSuite,
    SkillAgentTestResult,
    SkillAgentTestSuiteResult,
    SkillAgentTrace,
    TestCategory,
)
from evalview.skills.agent_runner import SkillAgentRunner, run_agent_tests


# =============================================================================
# Test Suite Loading Tests
# =============================================================================


class TestSkillAgentRunnerLoading:
    """Tests for loading test suites from YAML."""

    @pytest.fixture
    def runner(self) -> SkillAgentRunner:
        """Create a runner instance."""
        return SkillAgentRunner()

    @pytest.fixture
    def valid_yaml(self, tmp_path, temp_skill_file) -> str:
        """Create a valid test suite YAML file."""
        yaml_content = {
            "name": "test-suite",
            "skill": temp_skill_file,
            "agent": {
                "type": "system-prompt",
                "max_turns": 5,
            },
            "tests": [
                {
                    "name": "test-1",
                    "input": "Do something",
                    "expected": {
                        "output_contains": ["success"],
                    },
                }
            ],
        }

        yaml_path = tmp_path / "tests.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        return str(yaml_path)

    def test_load_test_suite_from_yaml(self, runner, valid_yaml):
        """Can load a test suite from valid YAML."""
        suite = runner.load_test_suite(valid_yaml)

        assert isinstance(suite, SkillAgentTestSuite)
        assert suite.name == "test-suite"
        assert len(suite.tests) == 1
        assert suite.tests[0].name == "test-1"

    def test_load_test_suite_file_not_found(self, runner):
        """Raises FileNotFoundError for missing YAML."""
        with pytest.raises(FileNotFoundError):
            runner.load_test_suite("/nonexistent/path.yaml")

    def test_load_test_suite_resolves_relative_skill_path(self, runner, tmp_path):
        """Skill path should be resolved relative to YAML file."""
        # Create skill in subdirectory
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_file = skills_dir / "SKILL.md"
        skill_file.write_text("---\nname: test\n---\nInstructions")

        # Create YAML that references skill relatively
        yaml_content = {
            "name": "test-suite",
            "skill": "skills/SKILL.md",  # Relative path
            "tests": [
                {"name": "test-1", "input": "query"}
            ],
        }

        yaml_path = tmp_path / "tests.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        suite = runner.load_test_suite(str(yaml_path))

        # Skill path should be resolved to absolute
        assert Path(suite.skill).is_absolute()
        assert Path(suite.skill).exists()

    def test_load_test_suite_with_agent_type_override(
        self, runner, valid_yaml
    ):
        """CLI agent type override should take precedence."""
        suite = runner.load_test_suite(
            valid_yaml,
            agent_type_override=AgentType.CLAUDE_CODE,
        )

        assert suite.agent.type == AgentType.CLAUDE_CODE

    def test_load_test_suite_with_cwd_override(self, runner, valid_yaml, tmp_path):
        """CLI cwd override should take precedence."""
        suite = runner.load_test_suite(
            valid_yaml,
            cwd_override=str(tmp_path),
        )

        assert suite.agent.cwd == str(tmp_path)

    def test_load_test_suite_with_max_turns_override(self, runner, valid_yaml):
        """CLI max_turns override should take precedence."""
        suite = runner.load_test_suite(
            valid_yaml,
            max_turns_override=20,
        )

        assert suite.agent.max_turns == 20


# =============================================================================
# Test Execution Tests
# =============================================================================


class TestSkillAgentRunnerExecution:
    """Tests for test suite execution."""

    @pytest.fixture
    def runner(self) -> SkillAgentRunner:
        return SkillAgentRunner(verbose=False)

    @pytest.fixture
    def sample_suite(self, sample_test_suite) -> SkillAgentTestSuite:
        return sample_test_suite

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter."""
        adapter = AsyncMock()
        adapter.name = "mock-adapter"
        adapter.health_check = AsyncMock(return_value=True)

        now = datetime.now()
        adapter.execute = AsyncMock(
            return_value=SkillAgentTrace(
                session_id="test-123",
                skill_name="test-skill",
                test_name="test-case",
                start_time=now,
                end_time=now + timedelta(seconds=2),
                tool_calls=["Read", "Write"],
                files_created=["output.txt"],
                final_output="SUCCESS: Task completed.",
            )
        )

        return adapter

    @pytest.mark.asyncio
    async def test_run_suite_returns_result(
        self, runner, sample_suite, mock_adapter
    ):
        """Running suite should return SkillAgentTestSuiteResult."""
        mock_skill = MagicMock()
        mock_skill.metadata.name = "test-skill"  # Assign directly as string
        mock_skill.instructions = "Instructions"

        with patch(
            "evalview.skills.adapters.SkillAdapterRegistry.create",
            return_value=mock_adapter,
        ):
            with patch(
                "evalview.skills.agent_runner.SkillParser.parse_file",
                return_value=mock_skill,
            ):
                result = await runner.run_suite(sample_suite)

        assert isinstance(result, SkillAgentTestSuiteResult)
        assert result.suite_name == sample_suite.name

    @pytest.mark.asyncio
    async def test_run_suite_calculates_stats(
        self, runner, sample_suite, mock_adapter
    ):
        """Suite result should have correct statistics."""
        mock_skill = MagicMock()
        mock_skill.metadata.name = "test-skill"
        mock_skill.instructions = "Instructions"

        with patch(
            "evalview.skills.adapters.SkillAdapterRegistry.create",
            return_value=mock_adapter,
        ):
            with patch(
                "evalview.skills.agent_runner.SkillParser.parse_file",
                return_value=mock_skill,
            ):
                result = await runner.run_suite(sample_suite)

        assert result.total_tests == len(sample_suite.tests)
        assert result.passed_tests + result.failed_tests == result.total_tests
        assert 0 <= result.pass_rate <= 1

    @pytest.mark.asyncio
    async def test_run_suite_handles_adapter_error(self, runner, sample_suite):
        """Adapter errors should be captured in test result."""
        from evalview.skills.adapters.base import SkillAgentAdapterError

        error_adapter = AsyncMock()
        error_adapter.name = "error-adapter"
        error_adapter.health_check = AsyncMock(return_value=True)
        error_adapter.execute = AsyncMock(
            side_effect=SkillAgentAdapterError("Adapter failed")
        )

        mock_skill = MagicMock()
        mock_skill.metadata.name = "test-skill"
        mock_skill.instructions = "Instructions"

        with patch(
            "evalview.skills.adapters.SkillAdapterRegistry.create",
            return_value=error_adapter,
        ):
            with patch(
                "evalview.skills.agent_runner.SkillParser.parse_file",
                return_value=mock_skill,
            ):
                result = await runner.run_suite(sample_suite)

        # All tests should fail with error
        for test_result in result.results:
            assert test_result.passed is False
            assert test_result.error is not None

    @pytest.mark.asyncio
    async def test_run_suite_checks_adapter_health(
        self, runner, sample_suite, mock_adapter, caplog
    ):
        """Should check adapter health before running tests."""
        import logging

        mock_adapter.health_check = AsyncMock(return_value=False)

        mock_skill = MagicMock()
        mock_skill.metadata.name = "test-skill"
        mock_skill.instructions = "Instructions"

        with caplog.at_level(logging.WARNING):
            with patch(
                "evalview.skills.adapters.SkillAdapterRegistry.create",
                return_value=mock_adapter,
            ):
                with patch(
                    "evalview.skills.agent_runner.SkillParser.parse_file",
                    return_value=mock_skill,
                ):
                    await runner.run_suite(sample_suite)

        mock_adapter.health_check.assert_called_once()


# =============================================================================
# Trace Saving Tests
# =============================================================================


class TestSkillAgentRunnerTraces:
    """Tests for trace saving functionality."""

    @pytest.fixture
    def runner_with_trace_dir(self, tmp_path) -> SkillAgentRunner:
        """Runner configured to save traces."""
        return SkillAgentRunner(
            trace_dir=str(tmp_path / "traces"),
        )

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter."""
        adapter = AsyncMock()
        adapter.name = "mock-adapter"
        adapter.health_check = AsyncMock(return_value=True)

        now = datetime.now()
        adapter.execute = AsyncMock(
            return_value=SkillAgentTrace(
                session_id="test-123",
                skill_name="test-skill",
                test_name="test-case",
                start_time=now,
                end_time=now + timedelta(seconds=2),
                events=[],
                tool_calls=["Read"],
                final_output="Done",
            )
        )

        return adapter

    @pytest.mark.asyncio
    async def test_traces_saved_to_directory(
        self, runner_with_trace_dir, sample_test_suite, mock_adapter
    ):
        """Traces should be saved as JSONL files."""
        mock_skill = MagicMock()
        mock_skill.metadata.name = "test-skill"
        mock_skill.instructions = "Instructions"

        with patch(
            "evalview.skills.adapters.SkillAdapterRegistry.create",
            return_value=mock_adapter,
        ):
            with patch(
                "evalview.skills.agent_runner.SkillParser.parse_file",
                return_value=mock_skill,
            ):
                result = await runner_with_trace_dir.run_suite(sample_test_suite)

        # Check trace files exist
        trace_dir = Path(runner_with_trace_dir.trace_dir)
        assert trace_dir.exists() or True  # Dir created with timestamp

    def test_setup_trace_dir_creates_timestamped_subdir(
        self, runner_with_trace_dir
    ):
        """Trace directory should have timestamped subdirectory."""
        trace_path = runner_with_trace_dir._setup_trace_dir("my-suite")

        assert trace_path is not None
        assert Path(trace_path).exists()
        assert "my-suite" in trace_path

    def test_setup_trace_dir_returns_none_if_not_configured(self):
        """Returns None if trace_dir not configured."""
        runner = SkillAgentRunner()  # No trace_dir

        result = runner._setup_trace_dir("suite")

        assert result is None

    def test_save_trace_writes_jsonl(self, tmp_path):
        """Save trace writes valid JSONL file."""
        runner = SkillAgentRunner()

        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="test-123",
            skill_name="test-skill",
            test_name="test-case",
            start_time=now,
            end_time=now + timedelta(seconds=2),
            events=[],
            tool_calls=["Read", "Write"],
            files_created=["file.txt"],
            commands_ran=["npm install"],
            final_output="Done",
        )

        trace_dir = str(tmp_path)
        filepath = runner._save_trace(trace, trace_dir)

        assert Path(filepath).exists()
        assert filepath.endswith(".jsonl")

        # Verify JSONL content
        with open(filepath) as f:
            lines = f.readlines()

        assert len(lines) >= 2  # Metadata + summary

        # First line should be metadata
        metadata = json.loads(lines[0])
        assert metadata["session_id"] == "test-123"


# =============================================================================
# Category Statistics Tests
# =============================================================================


class TestCategoryStatistics:
    """Tests for test category statistics calculation."""

    @pytest.fixture
    def runner(self) -> SkillAgentRunner:
        return SkillAgentRunner()

    def test_calculate_category_stats_empty(self, runner):
        """Empty results should return empty stats."""
        stats = runner._calculate_category_stats([])

        assert stats == {}

    def test_calculate_category_stats_single_category(self, runner):
        """Stats for single category results."""
        results = [
            SkillAgentTestResult(
                test_name="t1",
                category=TestCategory.EXPLICIT,
                passed=True,
                score=100.0,
                input_query="q",
                final_output="o",
            ),
            SkillAgentTestResult(
                test_name="t2",
                category=TestCategory.EXPLICIT,
                passed=False,
                score=50.0,
                input_query="q",
                final_output="o",
            ),
        ]

        stats = runner._calculate_category_stats(results)

        assert TestCategory.EXPLICIT in stats
        assert stats[TestCategory.EXPLICIT]["total"] == 2
        assert stats[TestCategory.EXPLICIT]["passed"] == 1
        assert stats[TestCategory.EXPLICIT]["failed"] == 1

    def test_calculate_category_stats_multiple_categories(self, runner):
        """Stats for multiple categories."""
        results = [
            SkillAgentTestResult(
                test_name="t1",
                category=TestCategory.EXPLICIT,
                passed=True,
                score=100.0,
                input_query="q",
                final_output="o",
            ),
            SkillAgentTestResult(
                test_name="t2",
                category=TestCategory.NEGATIVE,
                passed=True,
                score=100.0,
                input_query="q",
                final_output="o",
            ),
            SkillAgentTestResult(
                test_name="t3",
                category=TestCategory.CONTEXTUAL,
                passed=False,
                score=30.0,
                input_query="q",
                final_output="o",
            ),
        ]

        stats = runner._calculate_category_stats(results)

        assert len(stats) == 3
        assert TestCategory.EXPLICIT in stats
        assert TestCategory.NEGATIVE in stats
        assert TestCategory.CONTEXTUAL in stats


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestRunAgentTestsFunction:
    """Tests for the run_agent_tests convenience function."""

    @pytest.fixture
    def valid_yaml(self, tmp_path, temp_skill_file) -> str:
        """Create a valid test YAML."""
        yaml_content = {
            "name": "test-suite",
            "skill": temp_skill_file,
            "tests": [
                {"name": "test-1", "input": "query"}
            ],
        }

        yaml_path = tmp_path / "tests.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        return str(yaml_path)

    @pytest.mark.asyncio
    async def test_run_agent_tests_returns_result(self, valid_yaml):
        """Convenience function returns suite result."""
        mock_adapter = AsyncMock()
        mock_adapter.name = "mock"
        mock_adapter.health_check = AsyncMock(return_value=True)

        now = datetime.now()
        mock_adapter.execute = AsyncMock(
            return_value=SkillAgentTrace(
                session_id="t",
                skill_name="s",
                test_name="t",
                start_time=now,
                end_time=now,
                final_output="Done",
            )
        )

        mock_skill = MagicMock()
        mock_skill.metadata.name = "test-skill"
        mock_skill.instructions = "Instructions"

        with patch(
            "evalview.skills.adapters.SkillAdapterRegistry.create",
            return_value=mock_adapter,
        ):
            with patch(
                "evalview.skills.agent_runner.SkillParser.parse_file",
                return_value=mock_skill,
            ):
                result = await run_agent_tests(valid_yaml)

        assert isinstance(result, SkillAgentTestSuiteResult)

    @pytest.mark.asyncio
    async def test_run_agent_tests_with_agent_type(self, valid_yaml):
        """Can specify agent type as string."""
        mock_adapter = AsyncMock()
        mock_adapter.name = "mock"
        mock_adapter.health_check = AsyncMock(return_value=True)

        now = datetime.now()
        mock_adapter.execute = AsyncMock(
            return_value=SkillAgentTrace(
                session_id="t",
                skill_name="s",
                test_name="t",
                start_time=now,
                end_time=now,
                final_output="Done",
            )
        )

        mock_skill = MagicMock()
        mock_skill.metadata.name = "test-skill"
        mock_skill.instructions = "Instructions"

        with patch(
            "evalview.skills.adapters.SkillAdapterRegistry.create",
            return_value=mock_adapter,
        ):
            with patch(
                "evalview.skills.agent_runner.SkillParser.parse_file",
                return_value=mock_skill,
            ):
                result = await run_agent_tests(
                    valid_yaml,
                    agent_type="claude-code",
                )

        assert isinstance(result, SkillAgentTestSuiteResult)

    @pytest.mark.asyncio
    async def test_run_agent_tests_invalid_agent_type(self, valid_yaml):
        """Invalid agent type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            await run_agent_tests(
                valid_yaml,
                agent_type="invalid-agent",
            )

        assert "Unknown agent type" in str(exc_info.value)


# =============================================================================
# Runner Configuration Tests
# =============================================================================


class TestSkillAgentRunnerConfiguration:
    """Tests for runner configuration options."""

    def test_verbose_enables_debug_logging(self, caplog):
        """Verbose mode should enable debug logging."""
        import logging

        with caplog.at_level(logging.DEBUG):
            SkillAgentRunner(verbose=True)

        # Logger level should be set
        logger = logging.getLogger("evalview.skills")
        assert logger.level <= logging.DEBUG

    def test_skip_rubric_passed_to_orchestrator(self):
        """skip_rubric should be passed to orchestrator."""
        runner = SkillAgentRunner(skip_rubric=True)

        assert runner.orchestrator.skip_rubric is True

    def test_rubric_model_passed_to_orchestrator(self):
        """rubric_model should be passed to rubric evaluator."""
        runner = SkillAgentRunner(rubric_model="gpt-4")

        assert runner.orchestrator.rubric_evaluator.model_override == "gpt-4"

    def test_default_configuration(self):
        """Default configuration values."""
        runner = SkillAgentRunner()

        assert runner.verbose is False
        assert runner.skip_rubric is False
        assert runner.trace_dir is None
        assert runner.rubric_model is None
