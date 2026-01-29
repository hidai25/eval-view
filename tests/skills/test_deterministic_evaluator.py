"""Unit tests for deterministic evaluator.

Comprehensive tests for Phase 1 deterministic checks including:
- Tool call verification
- File system assertions
- Command execution checks
- Output string matching
"""

from datetime import datetime, timedelta
from pathlib import Path
import pytest

from evalview.skills.agent_types import (
    DeterministicExpected,
    SkillAgentTrace,
)
from evalview.skills.evaluators.deterministic import DeterministicEvaluator


class TestDeterministicEvaluatorBasic:
    """Basic evaluation scenarios."""

    @pytest.fixture
    def evaluator(self) -> DeterministicEvaluator:
        """Create a fresh evaluator instance."""
        return DeterministicEvaluator()

    @pytest.fixture
    def minimal_trace(self) -> SkillAgentTrace:
        """Create a minimal valid trace."""
        now = datetime.now()
        return SkillAgentTrace(
            session_id="test-123",
            skill_name="test-skill",
            test_name="test-case",
            start_time=now,
            end_time=now + timedelta(seconds=1),
            final_output="Test output",
        )

    def test_none_expected_returns_passed(self, evaluator, minimal_trace):
        """When expected is None, evaluation should pass with 100%."""
        result = evaluator.evaluate(None, minimal_trace)

        assert result.passed is True
        assert result.score == 100.0
        assert len(result.checks) == 0

    def test_empty_expected_returns_passed(self, evaluator, minimal_trace):
        """When expected has no fields set, evaluation should pass."""
        expected = DeterministicExpected()
        result = evaluator.evaluate(expected, minimal_trace)

        assert result.passed is True
        assert result.score == 100.0
        assert len(result.checks) == 0


class TestToolCallChecks:
    """Tests for tool call verification."""

    @pytest.fixture
    def evaluator(self) -> DeterministicEvaluator:
        return DeterministicEvaluator()

    @pytest.fixture
    def trace_with_tools(self) -> SkillAgentTrace:
        """Trace with Read, Write, Bash tool calls."""
        now = datetime.now()
        return SkillAgentTrace(
            session_id="test-123",
            skill_name="test-skill",
            test_name="test-case",
            start_time=now,
            end_time=now + timedelta(seconds=2),
            tool_calls=["Read", "Write", "Bash", "Read"],
            final_output="Success",
        )

    def test_tool_calls_contain_all_present(self, evaluator, trace_with_tools):
        """Pass when all required tools were called."""
        expected = DeterministicExpected(
            tool_calls_contain=["Read", "Write"]
        )
        result = evaluator.evaluate(expected, trace_with_tools)

        assert result.passed is True
        assert result.score == 100.0
        assert len(result.checks) == 1
        assert result.checks[0].check_name == "tool_calls_contain"

    def test_tool_calls_contain_missing_tool(self, evaluator, trace_with_tools):
        """Fail when a required tool was not called."""
        expected = DeterministicExpected(
            tool_calls_contain=["Read", "Grep"]  # Grep not in trace
        )
        result = evaluator.evaluate(expected, trace_with_tools)

        assert result.passed is False
        assert result.checks[0].passed is False
        assert "Grep" in result.checks[0].message

    def test_tool_calls_not_contain_success(self, evaluator, trace_with_tools):
        """Pass when no forbidden tools were called."""
        expected = DeterministicExpected(
            tool_calls_not_contain=["Delete", "Exec"]
        )
        result = evaluator.evaluate(expected, trace_with_tools)

        assert result.passed is True
        assert result.checks[0].check_name == "tool_calls_not_contain"

    def test_tool_calls_not_contain_violation(self, evaluator, trace_with_tools):
        """Fail when a forbidden tool was called."""
        expected = DeterministicExpected(
            tool_calls_not_contain=["Bash"]  # Bash is in trace
        )
        result = evaluator.evaluate(expected, trace_with_tools)

        assert result.passed is False
        assert "Bash" in result.checks[0].message

    def test_tool_sequence_found(self, evaluator, trace_with_tools):
        """Pass when tool sequence appears in order."""
        expected = DeterministicExpected(
            tool_sequence=["Read", "Write", "Bash"]
        )
        result = evaluator.evaluate(expected, trace_with_tools)

        assert result.passed is True
        assert result.checks[0].check_name == "tool_sequence"
        assert "matched" in result.checks[0].message.lower()

    def test_tool_sequence_not_found(self, evaluator, trace_with_tools):
        """Fail when tool sequence doesn't appear in order."""
        expected = DeterministicExpected(
            tool_sequence=["Write", "Read", "Bash"]  # Wrong order
        )
        result = evaluator.evaluate(expected, trace_with_tools)

        assert result.passed is False

    def test_tool_sequence_partial_match(self, evaluator):
        """Report partial match in message."""
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            tool_calls=["Read", "Bash"],  # Write missing after Read
            final_output="",
        )
        expected = DeterministicExpected(
            tool_sequence=["Read", "Write", "Bash"]
        )

        result = evaluator.evaluate(expected, trace)

        assert result.passed is False
        # Should mention what it found up to
        assert "up to" in result.checks[0].message.lower()

    def test_tool_sequence_subsequence_matching(self, evaluator):
        """Tool sequence should match as subsequence, not contiguous."""
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            tool_calls=["Read", "Grep", "Write", "Bash"],  # Grep in middle
            final_output="",
        )
        expected = DeterministicExpected(
            tool_sequence=["Read", "Write"]  # Not contiguous
        )

        result = evaluator.evaluate(expected, trace)

        assert result.passed is True


class TestFileChecks:
    """Tests for file system assertions."""

    @pytest.fixture
    def evaluator(self) -> DeterministicEvaluator:
        return DeterministicEvaluator()

    @pytest.fixture
    def trace_with_files(self) -> SkillAgentTrace:
        """Trace with file operations."""
        now = datetime.now()
        return SkillAgentTrace(
            session_id="test-123",
            skill_name="test-skill",
            test_name="test-case",
            start_time=now,
            end_time=now,
            files_created=["output.txt", "data.json"],
            files_modified=["config.yaml", "README.md"],
            final_output="Done",
        )

    def test_files_created_all_present(self, evaluator, trace_with_files):
        """Pass when all expected files were created."""
        expected = DeterministicExpected(
            files_created=["output.txt"]
        )
        result = evaluator.evaluate(expected, trace_with_files)

        assert result.passed is True
        assert result.checks[0].check_name == "files_created"

    def test_files_created_missing(self, evaluator, trace_with_files):
        """Fail when expected file was not created."""
        expected = DeterministicExpected(
            files_created=["missing.txt"]
        )
        result = evaluator.evaluate(expected, trace_with_files)

        assert result.passed is False
        assert "missing.txt" in str(result.checks[0].message)

    def test_files_modified_all_present(self, evaluator, trace_with_files):
        """Pass when all expected files were modified."""
        expected = DeterministicExpected(
            files_modified=["config.yaml"]
        )
        result = evaluator.evaluate(expected, trace_with_files)

        assert result.passed is True

    def test_files_not_modified_success(self, evaluator, trace_with_files):
        """Pass when forbidden files were not modified."""
        expected = DeterministicExpected(
            files_not_modified=["secret.key", "credentials.json"]
        )
        result = evaluator.evaluate(expected, trace_with_files)

        assert result.passed is True

    def test_files_not_modified_violation(self, evaluator, trace_with_files):
        """Fail when a forbidden file was modified."""
        expected = DeterministicExpected(
            files_not_modified=["README.md"]  # README.md is in modified list
        )
        result = evaluator.evaluate(expected, trace_with_files)

        assert result.passed is False

    def test_file_contains_strings_present(self, evaluator, tmp_path):
        """Pass when file contains expected strings."""
        # Create test file
        test_file = tmp_path / "output.txt"
        test_file.write_text("Operation completed successfully!\nAll tests passed.")

        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="",
        )

        expected = DeterministicExpected(
            file_contains={str(test_file): ["completed", "passed"]}
        )

        result = evaluator.evaluate(expected, trace, cwd=str(tmp_path))

        assert result.passed is True

    def test_file_contains_case_insensitive(self, evaluator, tmp_path):
        """File contains check should be case-insensitive."""
        test_file = tmp_path / "output.txt"
        test_file.write_text("SUCCESS")

        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="",
        )

        expected = DeterministicExpected(
            file_contains={str(test_file): ["success"]}  # lowercase
        )

        result = evaluator.evaluate(expected, trace)

        assert result.passed is True

    def test_file_contains_missing_file(self, evaluator, tmp_path):
        """Fail when file to check doesn't exist."""
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="",
        )

        expected = DeterministicExpected(
            file_contains={"/nonexistent/file.txt": ["something"]}
        )

        result = evaluator.evaluate(expected, trace)

        assert result.passed is False
        assert "not found" in result.checks[0].message.lower()

    def test_file_not_contains_success(self, evaluator, tmp_path):
        """Pass when file doesn't contain forbidden strings."""
        test_file = tmp_path / "output.txt"
        test_file.write_text("Clean output with no errors")

        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="",
        )

        expected = DeterministicExpected(
            file_not_contains={str(test_file): ["FATAL", "EXCEPTION"]}
        )

        result = evaluator.evaluate(expected, trace)

        assert result.passed is True

    def test_file_not_contains_missing_file_is_ok(self, evaluator):
        """Missing file is acceptable for file_not_contains check."""
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="",
        )

        expected = DeterministicExpected(
            file_not_contains={"/nonexistent/file.txt": ["error"]}
        )

        result = evaluator.evaluate(expected, trace)

        assert result.passed is True
        assert "acceptable" in result.checks[0].message.lower()


class TestCommandChecks:
    """Tests for command execution verification."""

    @pytest.fixture
    def evaluator(self) -> DeterministicEvaluator:
        return DeterministicEvaluator()

    @pytest.fixture
    def trace_with_commands(self) -> SkillAgentTrace:
        """Trace with command executions."""
        now = datetime.now()
        return SkillAgentTrace(
            session_id="test-123",
            skill_name="test-skill",
            test_name="test-case",
            start_time=now,
            end_time=now,
            commands_ran=[
                "npm install",
                "npm run build",
                "npm test -- --coverage",
            ],
            final_output="Build successful",
        )

    def test_commands_ran_substring_match(self, evaluator, trace_with_commands):
        """Commands use substring matching."""
        expected = DeterministicExpected(
            commands_ran=["npm install", "npm test"]
        )
        result = evaluator.evaluate(expected, trace_with_commands)

        assert result.passed is True

    def test_commands_ran_missing(self, evaluator, trace_with_commands):
        """Fail when expected command was not run."""
        expected = DeterministicExpected(
            commands_ran=["yarn install"]  # Used npm, not yarn
        )
        result = evaluator.evaluate(expected, trace_with_commands)

        assert result.passed is False

    def test_commands_ran_case_insensitive(self, evaluator, trace_with_commands):
        """Command matching should be case-insensitive."""
        expected = DeterministicExpected(
            commands_ran=["NPM INSTALL"]
        )
        result = evaluator.evaluate(expected, trace_with_commands)

        assert result.passed is True

    def test_commands_not_ran_success(self, evaluator, trace_with_commands):
        """Pass when forbidden commands were not run."""
        expected = DeterministicExpected(
            commands_not_ran=["rm -rf", "sudo"]
        )
        result = evaluator.evaluate(expected, trace_with_commands)

        assert result.passed is True

    def test_commands_not_ran_violation(self, evaluator, trace_with_commands):
        """Fail when forbidden command was run."""
        expected = DeterministicExpected(
            commands_not_ran=["npm run build"]
        )
        result = evaluator.evaluate(expected, trace_with_commands)

        assert result.passed is False

    def test_command_count_max_within_limit(self, evaluator, trace_with_commands):
        """Pass when command count is within limit."""
        expected = DeterministicExpected(
            command_count_max=5
        )
        result = evaluator.evaluate(expected, trace_with_commands)

        assert result.passed is True
        assert "3 <= 5" in result.checks[0].message

    def test_command_count_max_exceeded(self, evaluator, trace_with_commands):
        """Fail when command count exceeds limit."""
        expected = DeterministicExpected(
            command_count_max=2
        )
        result = evaluator.evaluate(expected, trace_with_commands)

        assert result.passed is False
        assert "3 > 2" in result.checks[0].message

    def test_command_count_max_zero(self, evaluator):
        """Handle zero command limit."""
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            commands_ran=[],
            final_output="",
        )

        expected = DeterministicExpected(command_count_max=0)
        result = evaluator.evaluate(expected, trace)

        assert result.passed is True


class TestOutputChecks:
    """Tests for output string matching."""

    @pytest.fixture
    def evaluator(self) -> DeterministicEvaluator:
        return DeterministicEvaluator()

    @pytest.fixture
    def trace_with_output(self) -> SkillAgentTrace:
        """Trace with final output."""
        now = datetime.now()
        return SkillAgentTrace(
            session_id="test-123",
            skill_name="test-skill",
            test_name="test-case",
            start_time=now,
            end_time=now,
            final_output="Operation completed successfully. Created 3 files.",
        )

    def test_output_contains_all_strings(self, evaluator, trace_with_output):
        """Pass when output contains all expected strings."""
        expected = DeterministicExpected(
            output_contains=["completed", "successfully"]
        )
        result = evaluator.evaluate(expected, trace_with_output)

        assert result.passed is True

    def test_output_contains_case_insensitive(self, evaluator, trace_with_output):
        """Output matching should be case-insensitive."""
        expected = DeterministicExpected(
            output_contains=["COMPLETED", "SUCCESSFULLY"]
        )
        result = evaluator.evaluate(expected, trace_with_output)

        assert result.passed is True

    def test_output_contains_missing_string(self, evaluator, trace_with_output):
        """Fail when expected string not in output."""
        expected = DeterministicExpected(
            output_contains=["completed", "ERROR"]
        )
        result = evaluator.evaluate(expected, trace_with_output)

        assert result.passed is False
        assert "ERROR" in str(result.checks[0].message)

    def test_output_not_contains_success(self, evaluator, trace_with_output):
        """Pass when output doesn't contain forbidden strings."""
        expected = DeterministicExpected(
            output_not_contains=["ERROR", "FAILED", "exception"]
        )
        result = evaluator.evaluate(expected, trace_with_output)

        assert result.passed is True

    def test_output_not_contains_violation(self, evaluator, trace_with_output):
        """Fail when output contains forbidden string."""
        expected = DeterministicExpected(
            output_not_contains=["Created"]
        )
        result = evaluator.evaluate(expected, trace_with_output)

        assert result.passed is False

    def test_output_truncated_in_actual(self, evaluator):
        """Long output should be truncated in check results."""
        now = datetime.now()
        long_output = "x" * 1000
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output=long_output,
        )

        expected = DeterministicExpected(output_contains=["x"])
        result = evaluator.evaluate(expected, trace)

        assert result.passed is True
        # Actual should be truncated
        assert "..." in str(result.checks[0].actual)


class TestMultipleChecks:
    """Tests for combining multiple check types."""

    @pytest.fixture
    def evaluator(self) -> DeterministicEvaluator:
        return DeterministicEvaluator()

    @pytest.fixture
    def comprehensive_trace(self, tmp_path) -> SkillAgentTrace:
        """Trace with all types of data."""
        # Create a test file
        test_file = tmp_path / "output.txt"
        test_file.write_text("Build completed successfully")

        now = datetime.now()
        return SkillAgentTrace(
            session_id="test-123",
            skill_name="test-skill",
            test_name="test-case",
            start_time=now,
            end_time=now,
            tool_calls=["Read", "Write", "Bash"],
            files_created=["output.txt"],
            files_modified=[],
            commands_ran=["npm install", "npm build"],
            final_output="SUCCESS: All operations completed",
        )

    def test_all_checks_pass(self, evaluator, comprehensive_trace, tmp_path):
        """All checks pass returns 100% score."""
        expected = DeterministicExpected(
            tool_calls_contain=["Read", "Write"],
            tool_calls_not_contain=["Delete"],
            files_created=["output.txt"],
            commands_ran=["npm install"],
            command_count_max=5,
            output_contains=["SUCCESS"],
            output_not_contains=["ERROR"],
        )

        result = evaluator.evaluate(expected, comprehensive_trace, cwd=str(tmp_path))

        assert result.passed is True
        assert result.score == 100.0
        assert result.passed_count == result.total_count
        assert len(result.failed_checks) == 0

    def test_partial_checks_pass(self, evaluator, comprehensive_trace, tmp_path):
        """Calculate correct score for partial pass."""
        expected = DeterministicExpected(
            tool_calls_contain=["Read"],  # Pass
            tool_calls_not_contain=["Bash"],  # Fail - Bash is in trace
            output_contains=["SUCCESS"],  # Pass
        )

        result = evaluator.evaluate(expected, comprehensive_trace, cwd=str(tmp_path))

        assert result.passed is False
        assert result.passed_count == 2
        assert result.total_count == 3
        # Score should be approximately 66.67%
        assert 66 < result.score < 67

    def test_failed_checks_property(self, evaluator, comprehensive_trace, tmp_path):
        """failed_checks property returns only failures."""
        expected = DeterministicExpected(
            tool_calls_contain=["Read"],  # Pass
            tool_calls_not_contain=["Bash"],  # Fail
            files_created=["missing.txt"],  # Fail
            output_contains=["SUCCESS"],  # Pass
        )

        result = evaluator.evaluate(expected, comprehensive_trace, cwd=str(tmp_path))

        assert len(result.failed_checks) == 2
        failed_names = [c.check_name for c in result.failed_checks]
        assert "tool_calls_not_contain" in failed_names
        assert "files_created" in failed_names


class TestPathNormalization:
    """Tests for file path handling."""

    @pytest.fixture
    def evaluator(self) -> DeterministicEvaluator:
        return DeterministicEvaluator()

    def test_filename_only_matching(self, evaluator):
        """File matching uses basename for cross-platform support."""
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            files_created=["/some/path/output.txt"],
            final_output="",
        )

        expected = DeterministicExpected(
            files_created=["output.txt"]  # Just filename
        )

        result = evaluator.evaluate(expected, trace)

        assert result.passed is True

    def test_relative_path_resolution(self, evaluator, tmp_path):
        """Relative paths should be resolved against cwd."""
        # Create file
        subdir = tmp_path / "sub"
        subdir.mkdir()
        test_file = subdir / "output.txt"
        test_file.write_text("content")

        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="",
        )

        expected = DeterministicExpected(
            file_contains={"sub/output.txt": ["content"]}
        )

        result = evaluator.evaluate(expected, trace, cwd=str(tmp_path))

        assert result.passed is True
