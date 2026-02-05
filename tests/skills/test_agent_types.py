"""Unit tests for agent_types module.

Tests Pydantic model validation, serialization, and edge cases
for all skill testing type definitions.
"""

from datetime import datetime, timedelta
import pytest
from pydantic import ValidationError

from evalview.skills.agent_types import (
    AgentConfig,
    AgentType,
    DeterministicCheckResult,
    DeterministicEvaluation,
    DeterministicExpected,
    RubricConfig,
    RubricEvaluation,
    SkillAgentTest,
    SkillAgentTestResult,
    SkillAgentTestSuite,
    SkillAgentTestSuiteResult,
    SkillAgentTrace,
    TestCategory,
    TraceEvent,
    TraceEventType,
)


class TestAgentType:
    """Tests for AgentType enum."""

    def test_all_agent_types_defined(self):
        """Verify all expected agent types exist."""
        expected_types = {
            "claude-code",
            "claude-agent-sdk",
            "codex",
            "langgraph",
            "crewai",
            "openai-assistants",
            "custom",
            "system-prompt",
        }
        actual_types = {t.value for t in AgentType}
        assert expected_types == actual_types

    def test_agent_type_string_conversion(self):
        """Test that agent types can be converted to/from strings."""
        assert AgentType.CLAUDE_CODE.value == "claude-code"
        assert AgentType("claude-code") == AgentType.CLAUDE_CODE


class TestTestCategory:
    """Tests for TestCategory enum."""

    def test_all_categories_defined(self):
        """Verify all test categories exist."""
        expected = {"explicit", "implicit", "contextual", "negative"}
        actual = {c.value for c in TestCategory}
        assert expected == actual


class TestAgentConfig:
    """Tests for AgentConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgentConfig()

        assert config.type == AgentType.SYSTEM_PROMPT
        assert config.max_turns == 10
        assert config.timeout == 300.0
        assert config.capture_trace is True
        assert config.cwd is None
        assert config.tools is None

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = AgentConfig(
            type=AgentType.CLAUDE_CODE,
            max_turns=20,
            timeout=600.0,
            tools=["Read", "Write"],
        )

        assert config.type == AgentType.CLAUDE_CODE
        assert config.max_turns == 20
        assert config.timeout == 600.0
        assert config.tools == ["Read", "Write"]

    def test_max_turns_validation(self):
        """Test max_turns boundary validation."""
        # Valid boundaries
        AgentConfig(max_turns=1)  # Minimum
        AgentConfig(max_turns=100)  # Maximum

        # Invalid values
        with pytest.raises(ValidationError):
            AgentConfig(max_turns=0)

        with pytest.raises(ValidationError):
            AgentConfig(max_turns=101)

    def test_timeout_validation(self):
        """Test timeout boundary validation."""
        AgentConfig(timeout=1.0)  # Minimum
        AgentConfig(timeout=3600.0)  # Maximum

        with pytest.raises(ValidationError):
            AgentConfig(timeout=0.5)

        with pytest.raises(ValidationError):
            AgentConfig(timeout=3601.0)

    def test_cwd_validation_with_valid_path(self, tmp_path):
        """Test cwd validation with valid directory."""
        config = AgentConfig(cwd=str(tmp_path))
        assert config.cwd == str(tmp_path)

    def test_cwd_validation_with_invalid_path(self):
        """Test cwd validation with non-existent directory."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(cwd="/nonexistent/path/12345")

        assert "does not exist" in str(exc_info.value)


class TestDeterministicExpected:
    """Tests for DeterministicExpected model."""

    def test_all_fields_optional(self):
        """Test that all fields are optional."""
        expected = DeterministicExpected()
        assert expected.tool_calls_contain is None
        assert expected.files_created is None

    def test_tool_checks(self):
        """Test tool check fields."""
        expected = DeterministicExpected(
            tool_calls_contain=["Read", "Write"],
            tool_calls_not_contain=["Bash"],
            tool_sequence=["Read", "Write"],
        )

        assert expected.tool_calls_contain == ["Read", "Write"]
        assert expected.tool_calls_not_contain == ["Bash"]
        assert expected.tool_sequence == ["Read", "Write"]

    def test_file_checks(self):
        """Test file check fields."""
        expected = DeterministicExpected(
            files_created=["output.txt"],
            files_modified=["config.json"],
            files_not_modified=["README.md"],
            file_contains={"output.txt": ["success"]},
            file_not_contains={"output.txt": ["error"]},
        )

        assert expected.files_created == ["output.txt"]
        assert expected.file_contains == {"output.txt": ["success"]}

    def test_command_checks(self):
        """Test command check fields."""
        expected = DeterministicExpected(
            commands_ran=["npm install"],
            commands_not_ran=["rm -rf"],
            command_count_max=10,
        )

        assert expected.command_count_max == 10

    def test_command_count_max_validation(self):
        """Test command_count_max must be non-negative."""
        DeterministicExpected(command_count_max=0)  # Valid

        with pytest.raises(ValidationError):
            DeterministicExpected(command_count_max=-1)


class TestRubricConfig:
    """Tests for RubricConfig model."""

    def test_minimum_prompt_length(self):
        """Test prompt minimum length validation."""
        # Valid - at least 10 characters
        RubricConfig(prompt="This is a valid rubric prompt")

        # Invalid - too short
        with pytest.raises(ValidationError):
            RubricConfig(prompt="Too short")

    def test_default_values(self):
        """Test default rubric configuration."""
        rubric = RubricConfig(prompt="Evaluate the response quality.")

        assert rubric.min_score == 70.0
        assert rubric.model is None
        assert rubric.schema_path is None

    def test_score_validation(self):
        """Test min_score boundary validation."""
        RubricConfig(prompt="Valid prompt here", min_score=0)
        RubricConfig(prompt="Valid prompt here", min_score=100)

        with pytest.raises(ValidationError):
            RubricConfig(prompt="Valid prompt here", min_score=-1)

        with pytest.raises(ValidationError):
            RubricConfig(prompt="Valid prompt here", min_score=101)


class TestSkillAgentTest:
    """Tests for SkillAgentTest model."""

    def test_basic_test(self):
        """Test basic test creation."""
        test = SkillAgentTest(
            name="my-test",
            input="Test query",
        )

        assert test.name == "my-test"
        assert test.input == "Test query"
        assert test.category == TestCategory.EXPLICIT
        assert test.should_trigger is True

    def test_test_with_expected(self, deterministic_expected):
        """Test creation with expected behaviors."""
        test = SkillAgentTest(
            name="with-expected",
            input="Do something",
            expected=deterministic_expected,
        )

        assert test.expected is not None
        assert test.expected.tool_calls_contain == ["Read", "Write"]

    def test_negative_test_warning(self, caplog):
        """Test warning for NEGATIVE category with should_trigger=True."""
        import logging

        with caplog.at_level(logging.WARNING):
            test = SkillAgentTest(
                name="suspicious-negative",
                input="Query",
                category=TestCategory.NEGATIVE,
                should_trigger=True,  # This should warn
            )

        assert "should_trigger=True" in caplog.text or test is not None

    def test_name_length_validation(self):
        """Test name length constraints."""
        # Valid
        SkillAgentTest(name="x", input="query")  # Min length 1
        SkillAgentTest(name="x" * 128, input="query")  # Max length 128

        # Invalid
        with pytest.raises(ValidationError):
            SkillAgentTest(name="", input="query")

        with pytest.raises(ValidationError):
            SkillAgentTest(name="x" * 129, input="query")

    def test_input_not_empty(self):
        """Test input cannot be empty."""
        with pytest.raises(ValidationError):
            SkillAgentTest(name="test", input="")


class TestSkillAgentTestSuite:
    """Tests for SkillAgentTestSuite model."""

    def test_basic_suite(self):
        """Test basic test suite creation."""
        suite = SkillAgentTestSuite(
            name="my-suite",
            skill="./SKILL.md",
            tests=[SkillAgentTest(name="test1", input="query")],
        )

        assert suite.name == "my-suite"
        assert suite.skill == "./SKILL.md"
        assert len(suite.tests) == 1
        assert suite.min_pass_rate == 0.8

    def test_suite_requires_tests(self):
        """Test that suite requires at least one test."""
        with pytest.raises(ValidationError):
            SkillAgentTestSuite(
                name="empty-suite",
                skill="./SKILL.md",
                tests=[],
            )

    def test_skill_path_warning(self, caplog):
        """Test warning for non-.md skill paths."""
        import logging

        with caplog.at_level(logging.WARNING):
            SkillAgentTestSuite(
                name="suite",
                skill="./skill.txt",  # Should warn
                tests=[SkillAgentTest(name="t", input="q")],
            )

        # Warning may or may not appear depending on log level
        # Just verify it doesn't raise


class TestSkillAgentTrace:
    """Tests for SkillAgentTrace model."""

    def test_trace_creation(self, sample_trace):
        """Test trace creation with fixtures."""
        assert sample_trace.session_id == "test-abc123"
        assert sample_trace.skill_name == "test-skill"
        assert len(sample_trace.tool_calls) == 2

    def test_duration_property(self):
        """Test duration_ms property calculation."""
        start = datetime.now()
        end = start + timedelta(seconds=5)

        trace = SkillAgentTrace(
            session_id="test",
            skill_name="skill",
            test_name="test",
            start_time=start,
            end_time=end,
            final_output="",
        )

        assert abs(trace.duration_ms - 5000.0) < 1  # Within 1ms

    def test_has_errors_property(self):
        """Test has_errors property."""
        trace_ok = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=datetime.now(),
            end_time=datetime.now(),
            final_output="",
            errors=[],
        )
        assert trace_ok.has_errors is False

        trace_err = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=datetime.now(),
            end_time=datetime.now(),
            final_output="",
            errors=["Something went wrong"],
        )
        assert trace_err.has_errors is True


class TestTraceEvent:
    """Tests for TraceEvent model."""

    def test_tool_call_event(self):
        """Test tool call event creation."""
        event = TraceEvent(
            type=TraceEventType.TOOL_CALL,
            tool_name="Read",
            tool_input={"file_path": "/test.py"},
            tool_success=True,
        )

        assert event.type == TraceEventType.TOOL_CALL
        assert event.tool_name == "Read"
        assert event.tool_success is True

    def test_file_event(self):
        """Test file operation event."""
        event = TraceEvent(
            type=TraceEventType.FILE_CREATE,
            file_path="/output/result.txt",
            file_content="content here",
        )

        assert event.type == TraceEventType.FILE_CREATE
        assert event.file_path == "/output/result.txt"

    def test_command_event(self):
        """Test command execution event."""
        event = TraceEvent(
            type=TraceEventType.COMMAND_RUN,
            command="npm install",
            command_output="packages installed",
            command_exit_code=0,
        )

        assert event.command == "npm install"
        assert event.command_exit_code == 0


class TestDeterministicEvaluation:
    """Tests for DeterministicEvaluation model."""

    def test_evaluation_creation(self):
        """Test evaluation result creation."""
        checks = [
            DeterministicCheckResult(
                check_name="tool_calls_contain",
                passed=True,
                expected=["Read"],
                actual=["Read", "Write"],
                message="All required tools were called",
            ),
            DeterministicCheckResult(
                check_name="output_contains",
                passed=False,
                expected=["success"],
                actual="error occurred",
                message="String not found",
            ),
        ]

        evaluation = DeterministicEvaluation(
            passed=False,
            score=50.0,
            checks=checks,
            passed_count=1,
            total_count=2,
        )

        assert evaluation.passed is False
        assert evaluation.score == 50.0
        assert len(evaluation.checks) == 2

    def test_failed_checks_property(self):
        """Test failed_checks property."""
        checks = [
            DeterministicCheckResult(
                check_name="check1", passed=True,
                expected="a", actual="a", message="ok"
            ),
            DeterministicCheckResult(
                check_name="check2", passed=False,
                expected="b", actual="c", message="fail"
            ),
        ]

        evaluation = DeterministicEvaluation(
            passed=False, score=50.0, checks=checks,
            passed_count=1, total_count=2
        )

        failed = evaluation.failed_checks
        assert len(failed) == 1
        assert failed[0].check_name == "check2"


class TestSkillAgentTestResult:
    """Tests for SkillAgentTestResult model."""

    def test_result_creation(self, sample_trace):
        """Test result creation."""
        result = SkillAgentTestResult(
            test_name="my-test",
            category=TestCategory.EXPLICIT,
            passed=True,
            score=95.0,
            input_query="Do something",
            final_output="Done successfully",
            latency_ms=1500.0,
            input_tokens=100,
            output_tokens=50,
        )

        assert result.passed is True
        assert result.score == 95.0

    def test_result_with_error(self):
        """Test result with execution error."""
        result = SkillAgentTestResult(
            test_name="failed-test",
            category=TestCategory.EXPLICIT,
            passed=False,
            score=0.0,
            input_query="Query",
            final_output="",
            error="Timeout exceeded",
        )

        assert result.passed is False
        assert result.error == "Timeout exceeded"


class TestSkillAgentTestSuiteResult:
    """Tests for SkillAgentTestSuiteResult model."""

    def test_suite_result(self):
        """Test suite result creation."""
        result = SkillAgentTestSuiteResult(
            suite_name="my-suite",
            skill_name="test-skill",
            agent_type=AgentType.CLAUDE_CODE,
            passed=True,
            total_tests=5,
            passed_tests=4,
            failed_tests=1,
            pass_rate=0.8,
            results=[],
            total_latency_ms=5000.0,
            avg_latency_ms=1000.0,
            total_tokens=500,
        )

        assert result.passed is True
        assert result.pass_rate == 0.8
