"""Deterministic evaluators for skill testing (Phase 1).

Fast, debuggable checks that don't require LLM calls:
- Tool call verification
- File system assertions
- Command execution checks
- Output string matching

Each check returns a DeterministicCheckResult with:
- passed: bool
- expected: what was expected
- actual: what was found
- message: human-readable explanation
"""

import os
import logging
from typing import List, Optional, Set

from evalview.skills.agent_types import (
    DeterministicExpected,
    DeterministicCheckResult,
    DeterministicEvaluation,
    SkillAgentTrace,
)

logger = logging.getLogger(__name__)


class DeterministicEvaluator:
    """Evaluates deterministic checks against execution trace.

    This is Phase 1 of the two-phase evaluation system. It runs fast,
    deterministic checks that don't require LLM calls:

    - Tool call verification (contain, not contain, sequence)
    - File system assertions (created, modified, not modified, contains)
    - Command execution checks (ran, not ran, count)
    - Output string matching (contains, not contains)
    """

    def evaluate(
        self,
        expected: Optional[DeterministicExpected],
        trace: SkillAgentTrace,
        cwd: Optional[str] = None,
    ) -> DeterministicEvaluation:
        """Run all deterministic checks.

        Args:
            expected: Expected behaviors (None means no checks)
            trace: Execution trace to evaluate
            cwd: Working directory for file path resolution

        Returns:
            DeterministicEvaluation with all check results
        """
        if expected is None:
            return DeterministicEvaluation(passed=True, score=100.0)

        checks: List[DeterministicCheckResult] = []

        # Tool call checks
        if expected.tool_calls_contain:
            checks.append(
                self._check_tools_contain(
                    expected.tool_calls_contain, trace.tool_calls
                )
            )

        if expected.tool_calls_not_contain:
            checks.append(
                self._check_tools_not_contain(
                    expected.tool_calls_not_contain, trace.tool_calls
                )
            )

        if expected.tool_sequence:
            checks.append(
                self._check_tool_sequence(expected.tool_sequence, trace.tool_calls)
            )

        # File checks
        if expected.files_created:
            checks.append(
                self._check_files_created(
                    expected.files_created, trace.files_created, cwd
                )
            )

        if expected.files_modified:
            checks.append(
                self._check_files_modified(
                    expected.files_modified, trace.files_modified, cwd
                )
            )

        if expected.files_not_modified:
            checks.append(
                self._check_files_not_modified(
                    expected.files_not_modified, trace.files_modified, cwd
                )
            )

        if expected.file_contains:
            for file_path, strings in expected.file_contains.items():
                checks.append(self._check_file_contains(file_path, strings, cwd))

        if expected.file_not_contains:
            for file_path, strings in expected.file_not_contains.items():
                checks.append(self._check_file_not_contains(file_path, strings, cwd))

        # Command checks
        if expected.commands_ran:
            checks.append(
                self._check_commands_ran(expected.commands_ran, trace.commands_ran)
            )

        if expected.commands_not_ran:
            checks.append(
                self._check_commands_not_ran(
                    expected.commands_not_ran, trace.commands_ran
                )
            )

        if expected.command_count_max is not None:
            checks.append(
                self._check_command_count_max(
                    expected.command_count_max, len(trace.commands_ran)
                )
            )

        # Output checks
        if expected.output_contains:
            checks.append(
                self._check_output_contains(
                    expected.output_contains, trace.final_output
                )
            )

        if expected.output_not_contains:
            checks.append(
                self._check_output_not_contains(
                    expected.output_not_contains, trace.final_output
                )
            )

        # Calculate overall result
        passed_count = sum(1 for c in checks if c.passed)
        total_count = len(checks)
        score = (passed_count / total_count * 100) if total_count > 0 else 100.0

        return DeterministicEvaluation(
            passed=(passed_count == total_count),
            score=score,
            checks=checks,
            passed_count=passed_count,
            total_count=total_count,
        )

    def _check_tools_contain(
        self, expected_tools: List[str], actual_tools: List[str]
    ) -> DeterministicCheckResult:
        """Check that all expected tools were called."""
        actual_set = set(actual_tools)
        missing = [t for t in expected_tools if t not in actual_set]

        if missing:
            return DeterministicCheckResult(
                check_name="tool_calls_contain",
                passed=False,
                expected=expected_tools,
                actual=actual_tools,
                message=f"Missing tool calls: {missing}",
            )

        return DeterministicCheckResult(
            check_name="tool_calls_contain",
            passed=True,
            expected=expected_tools,
            actual=actual_tools,
            message=f"All required tools were called",
        )

    def _check_tools_not_contain(
        self, forbidden_tools: List[str], actual_tools: List[str]
    ) -> DeterministicCheckResult:
        """Check that no forbidden tools were called."""
        actual_set = set(actual_tools)
        forbidden_found = [t for t in forbidden_tools if t in actual_set]

        if forbidden_found:
            return DeterministicCheckResult(
                check_name="tool_calls_not_contain",
                passed=False,
                expected=f"NOT {forbidden_tools}",
                actual=actual_tools,
                message=f"Forbidden tools were called: {forbidden_found}",
            )

        return DeterministicCheckResult(
            check_name="tool_calls_not_contain",
            passed=True,
            expected=f"NOT {forbidden_tools}",
            actual=actual_tools,
            message="No forbidden tools were called",
        )

    def _check_tool_sequence(
        self, expected_sequence: List[str], actual_tools: List[str]
    ) -> DeterministicCheckResult:
        """Check that tools appear in order (subsequence match)."""
        # Check if expected_sequence is a subsequence of actual_tools
        seq_idx = 0
        for tool in actual_tools:
            if seq_idx < len(expected_sequence) and tool == expected_sequence[seq_idx]:
                seq_idx += 1

        passed = seq_idx == len(expected_sequence)

        if not passed:
            found_up_to = expected_sequence[:seq_idx] if seq_idx > 0 else []
            return DeterministicCheckResult(
                check_name="tool_sequence",
                passed=False,
                expected=expected_sequence,
                actual=actual_tools,
                message=f"Tool sequence not found. Got up to: {found_up_to}",
            )

        return DeterministicCheckResult(
            check_name="tool_sequence",
            passed=True,
            expected=expected_sequence,
            actual=actual_tools,
            message="Tool sequence matched",
        )

    def _check_files_created(
        self, expected_files: List[str], created_files: List[str], cwd: Optional[str]
    ) -> DeterministicCheckResult:
        """Check that expected files were created."""
        # Normalize paths for comparison
        created_set = self._normalize_paths(created_files, cwd)
        expected_set = self._normalize_paths(expected_files, cwd)

        missing = [f for f in expected_set if f not in created_set]

        if missing:
            return DeterministicCheckResult(
                check_name="files_created",
                passed=False,
                expected=expected_files,
                actual=list(created_files),
                message=f"Files not created: {missing}",
            )

        return DeterministicCheckResult(
            check_name="files_created",
            passed=True,
            expected=expected_files,
            actual=list(created_files),
            message="All expected files were created",
        )

    def _check_files_modified(
        self, expected_files: List[str], modified_files: List[str], cwd: Optional[str]
    ) -> DeterministicCheckResult:
        """Check that expected files were modified."""
        modified_set = self._normalize_paths(modified_files, cwd)
        expected_set = self._normalize_paths(expected_files, cwd)

        missing = [f for f in expected_set if f not in modified_set]

        if missing:
            return DeterministicCheckResult(
                check_name="files_modified",
                passed=False,
                expected=expected_files,
                actual=list(modified_files),
                message=f"Files not modified: {missing}",
            )

        return DeterministicCheckResult(
            check_name="files_modified",
            passed=True,
            expected=expected_files,
            actual=list(modified_files),
            message="All expected files were modified",
        )

    def _check_files_not_modified(
        self, forbidden_files: List[str], modified_files: List[str], cwd: Optional[str]
    ) -> DeterministicCheckResult:
        """Check that forbidden files were NOT modified."""
        modified_set = self._normalize_paths(modified_files, cwd)
        forbidden_set = self._normalize_paths(forbidden_files, cwd)

        modified_forbidden = [f for f in forbidden_set if f in modified_set]

        if modified_forbidden:
            return DeterministicCheckResult(
                check_name="files_not_modified",
                passed=False,
                expected=f"NOT {forbidden_files}",
                actual=list(modified_files),
                message=f"Forbidden files were modified: {modified_forbidden}",
            )

        return DeterministicCheckResult(
            check_name="files_not_modified",
            passed=True,
            expected=f"NOT {forbidden_files}",
            actual=list(modified_files),
            message="No forbidden files were modified",
        )

    def _check_file_contains(
        self, file_path: str, expected_strings: List[str], cwd: Optional[str]
    ) -> DeterministicCheckResult:
        """Check that a file contains expected strings."""
        full_path = self._resolve_path(file_path, cwd)

        if not os.path.exists(full_path):
            return DeterministicCheckResult(
                check_name=f"file_contains[{file_path}]",
                passed=False,
                expected=expected_strings,
                actual=None,
                message=f"File not found: {file_path}",
            )

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return DeterministicCheckResult(
                check_name=f"file_contains[{file_path}]",
                passed=False,
                expected=expected_strings,
                actual=None,
                message=f"Could not read file: {e}",
            )

        content_lower = content.lower()
        missing = [s for s in expected_strings if s.lower() not in content_lower]

        if missing:
            return DeterministicCheckResult(
                check_name=f"file_contains[{file_path}]",
                passed=False,
                expected=expected_strings,
                actual=f"<file content, {len(content)} chars>",
                message=f"Strings not found in file: {missing}",
            )

        return DeterministicCheckResult(
            check_name=f"file_contains[{file_path}]",
            passed=True,
            expected=expected_strings,
            actual=f"<file content, {len(content)} chars>",
            message="All expected strings found in file",
        )

    def _check_file_not_contains(
        self, file_path: str, forbidden_strings: List[str], cwd: Optional[str]
    ) -> DeterministicCheckResult:
        """Check that a file does NOT contain forbidden strings."""
        full_path = self._resolve_path(file_path, cwd)

        if not os.path.exists(full_path):
            # File not existing is OK for "not contains" check
            return DeterministicCheckResult(
                check_name=f"file_not_contains[{file_path}]",
                passed=True,
                expected=f"NOT {forbidden_strings}",
                actual=None,
                message=f"File not found (acceptable): {file_path}",
            )

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return DeterministicCheckResult(
                check_name=f"file_not_contains[{file_path}]",
                passed=False,
                expected=f"NOT {forbidden_strings}",
                actual=None,
                message=f"Could not read file: {e}",
            )

        content_lower = content.lower()
        found = [s for s in forbidden_strings if s.lower() in content_lower]

        if found:
            return DeterministicCheckResult(
                check_name=f"file_not_contains[{file_path}]",
                passed=False,
                expected=f"NOT {forbidden_strings}",
                actual=f"<file content, {len(content)} chars>",
                message=f"Forbidden strings found in file: {found}",
            )

        return DeterministicCheckResult(
            check_name=f"file_not_contains[{file_path}]",
            passed=True,
            expected=f"NOT {forbidden_strings}",
            actual=f"<file content, {len(content)} chars>",
            message="No forbidden strings found in file",
        )

    def _check_commands_ran(
        self, expected_commands: List[str], actual_commands: List[str]
    ) -> DeterministicCheckResult:
        """Check that expected commands were run (substring match)."""
        missing = []
        for expected in expected_commands:
            found = any(expected.lower() in cmd.lower() for cmd in actual_commands)
            if not found:
                missing.append(expected)

        if missing:
            return DeterministicCheckResult(
                check_name="commands_ran",
                passed=False,
                expected=expected_commands,
                actual=actual_commands,
                message=f"Commands not found: {missing}",
            )

        return DeterministicCheckResult(
            check_name="commands_ran",
            passed=True,
            expected=expected_commands,
            actual=actual_commands,
            message="All expected commands were run",
        )

    def _check_commands_not_ran(
        self, forbidden_commands: List[str], actual_commands: List[str]
    ) -> DeterministicCheckResult:
        """Check that forbidden commands were NOT run."""
        found = []
        for forbidden in forbidden_commands:
            if any(forbidden.lower() in cmd.lower() for cmd in actual_commands):
                found.append(forbidden)

        if found:
            return DeterministicCheckResult(
                check_name="commands_not_ran",
                passed=False,
                expected=f"NOT {forbidden_commands}",
                actual=actual_commands,
                message=f"Forbidden commands were run: {found}",
            )

        return DeterministicCheckResult(
            check_name="commands_not_ran",
            passed=True,
            expected=f"NOT {forbidden_commands}",
            actual=actual_commands,
            message="No forbidden commands were run",
        )

    def _check_command_count_max(
        self, max_count: int, actual_count: int
    ) -> DeterministicCheckResult:
        """Check that command count doesn't exceed maximum."""
        passed = actual_count <= max_count

        if not passed:
            return DeterministicCheckResult(
                check_name="command_count_max",
                passed=False,
                expected=f"<= {max_count}",
                actual=actual_count,
                message=f"Too many commands: {actual_count} > {max_count}",
            )

        return DeterministicCheckResult(
            check_name="command_count_max",
            passed=True,
            expected=f"<= {max_count}",
            actual=actual_count,
            message=f"Command count within limit: {actual_count} <= {max_count}",
        )

    def _check_output_contains(
        self, expected_strings: List[str], output: str
    ) -> DeterministicCheckResult:
        """Check that output contains expected strings."""
        output_lower = output.lower()
        missing = [s for s in expected_strings if s.lower() not in output_lower]

        if missing:
            return DeterministicCheckResult(
                check_name="output_contains",
                passed=False,
                expected=expected_strings,
                actual=output[:500] + "..." if len(output) > 500 else output,
                message=f"Strings not found in output: {missing}",
            )

        return DeterministicCheckResult(
            check_name="output_contains",
            passed=True,
            expected=expected_strings,
            actual=output[:500] + "..." if len(output) > 500 else output,
            message="All expected strings found in output",
        )

    def _check_output_not_contains(
        self, forbidden_strings: List[str], output: str
    ) -> DeterministicCheckResult:
        """Check that output does NOT contain forbidden strings."""
        output_lower = output.lower()
        found = [s for s in forbidden_strings if s.lower() in output_lower]

        if found:
            return DeterministicCheckResult(
                check_name="output_not_contains",
                passed=False,
                expected=f"NOT {forbidden_strings}",
                actual=output[:500] + "..." if len(output) > 500 else output,
                message=f"Forbidden strings found in output: {found}",
            )

        return DeterministicCheckResult(
            check_name="output_not_contains",
            passed=True,
            expected=f"NOT {forbidden_strings}",
            actual=output[:500] + "..." if len(output) > 500 else output,
            message="No forbidden strings found in output",
        )

    def _normalize_paths(
        self, paths: List[str], cwd: Optional[str]
    ) -> Set[str]:
        """Normalize a list of file paths for comparison."""
        normalized = set()
        for path in paths:
            normalized.add(self._normalize_path(path, cwd))
        return normalized

    def _normalize_path(self, path: str, cwd: Optional[str]) -> str:
        """Normalize a single file path."""
        # Get just the filename for comparison (handles different cwd)
        return os.path.basename(path)

    def _resolve_path(self, path: str, cwd: Optional[str]) -> str:
        """Resolve a path relative to cwd."""
        if os.path.isabs(path):
            return path
        if cwd:
            return os.path.join(cwd, path)
        return os.path.abspath(path)
