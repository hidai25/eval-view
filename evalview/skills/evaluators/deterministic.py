"""Deterministic evaluators for skill testing (Phase 1).

Fast, debuggable checks that don't require LLM calls:
- Tool call verification
- File system assertions
- Command execution checks
- Output string matching
- Token budget enforcement
- Build verification
- Runtime smoke tests
- Repository cleanliness
- Permission/security checks

Each check returns a DeterministicCheckResult with:
- passed: bool
- expected: what was expected
- actual: what was found
- message: human-readable explanation
"""

import os
import re
import signal
import subprocess
import logging
from typing import Any, List, Optional, Set, Tuple

from evalview.skills.agent_types import (
    DeterministicExpected,
    DeterministicCheckResult,
    DeterministicEvaluation,
    SkillAgentTrace,
    SmokeTest,
)

logger = logging.getLogger(__name__)

# Patterns considered dangerous for security checks
SUDO_PATTERNS = [
    r'\bsudo\b',
    r'\bsu\s+-',
    r'\bdoas\b',
]

EXTERNAL_NETWORK_PATTERNS = [
    r'\bcurl\s+https?://(?!localhost|127\.0\.0\.1)',
    r'\bwget\s+https?://(?!localhost|127\.0\.0\.1)',
    r'\bfetch\s+https?://(?!localhost|127\.0\.0\.1)',
]


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

        # Token budget checks
        if expected.max_input_tokens is not None:
            checks.append(
                self._check_max_input_tokens(
                    expected.max_input_tokens, trace.total_input_tokens
                )
            )

        if expected.max_output_tokens is not None:
            checks.append(
                self._check_max_output_tokens(
                    expected.max_output_tokens, trace.total_output_tokens
                )
            )

        if expected.max_total_tokens is not None:
            total_tokens = trace.total_input_tokens + trace.total_output_tokens
            checks.append(
                self._check_max_total_tokens(expected.max_total_tokens, total_tokens)
            )

        # Build verification
        if expected.build_must_pass:
            for build_cmd in expected.build_must_pass:
                checks.append(self._check_build_command(build_cmd, cwd))

        # Runtime smoke tests
        if expected.smoke_tests:
            for smoke_test in expected.smoke_tests:
                checks.append(self._check_smoke_test(smoke_test, cwd))

        # Repository cleanliness
        if expected.git_clean is True:
            checks.append(self._check_git_clean(cwd))

        # Permission/security checks
        if expected.forbidden_patterns:
            checks.append(
                self._check_forbidden_patterns(
                    expected.forbidden_patterns, trace.commands_ran
                )
            )

        if expected.no_sudo is True:
            checks.append(self._check_no_sudo(trace.commands_ran))

        if expected.no_network_external is True:
            checks.append(self._check_no_external_network(trace.commands_ran))

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

    # =========================================================================
    # Token Budget Checks
    # =========================================================================

    def _check_max_input_tokens(
        self, max_tokens: int, actual_tokens: int
    ) -> DeterministicCheckResult:
        """Check that input tokens don't exceed budget."""
        passed = actual_tokens <= max_tokens

        if not passed:
            return DeterministicCheckResult(
                check_name="max_input_tokens",
                passed=False,
                expected=f"<= {max_tokens}",
                actual=actual_tokens,
                message=f"Input token budget exceeded: {actual_tokens} > {max_tokens}",
            )

        return DeterministicCheckResult(
            check_name="max_input_tokens",
            passed=True,
            expected=f"<= {max_tokens}",
            actual=actual_tokens,
            message=f"Input tokens within budget: {actual_tokens} <= {max_tokens}",
        )

    def _check_max_output_tokens(
        self, max_tokens: int, actual_tokens: int
    ) -> DeterministicCheckResult:
        """Check that output tokens don't exceed budget."""
        passed = actual_tokens <= max_tokens

        if not passed:
            return DeterministicCheckResult(
                check_name="max_output_tokens",
                passed=False,
                expected=f"<= {max_tokens}",
                actual=actual_tokens,
                message=f"Output token budget exceeded: {actual_tokens} > {max_tokens}",
            )

        return DeterministicCheckResult(
            check_name="max_output_tokens",
            passed=True,
            expected=f"<= {max_tokens}",
            actual=actual_tokens,
            message=f"Output tokens within budget: {actual_tokens} <= {max_tokens}",
        )

    def _check_max_total_tokens(
        self, max_tokens: int, actual_tokens: int
    ) -> DeterministicCheckResult:
        """Check that total tokens don't exceed budget."""
        passed = actual_tokens <= max_tokens

        if not passed:
            return DeterministicCheckResult(
                check_name="max_total_tokens",
                passed=False,
                expected=f"<= {max_tokens}",
                actual=actual_tokens,
                message=f"Total token budget exceeded: {actual_tokens} > {max_tokens}",
            )

        return DeterministicCheckResult(
            check_name="max_total_tokens",
            passed=True,
            expected=f"<= {max_tokens}",
            actual=actual_tokens,
            message=f"Total tokens within budget: {actual_tokens} <= {max_tokens}",
        )

    # =========================================================================
    # Build Verification
    # =========================================================================

    def _check_build_command(
        self, command: str, cwd: Optional[str]
    ) -> DeterministicCheckResult:
        """Run a build command and verify it succeeds (exit code 0).

        Args:
            command: Build command to run (e.g., "npm run build")
            cwd: Working directory

        Returns:
            DeterministicCheckResult with pass/fail status
        """
        check_name = f"build_must_pass[{command[:30]}...]" if len(command) > 30 else f"build_must_pass[{command}]"

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for builds
            )

            if result.returncode != 0:
                error_output = result.stderr[:500] if result.stderr else result.stdout[:500]
                return DeterministicCheckResult(
                    check_name=check_name,
                    passed=False,
                    expected="exit code 0",
                    actual=f"exit code {result.returncode}",
                    message=f"Build failed: {error_output}",
                )

            return DeterministicCheckResult(
                check_name=check_name,
                passed=True,
                expected="exit code 0",
                actual="exit code 0",
                message=f"Build succeeded: {command}",
            )

        except subprocess.TimeoutExpired:
            return DeterministicCheckResult(
                check_name=check_name,
                passed=False,
                expected="exit code 0",
                actual="timeout",
                message=f"Build timed out after 300s: {command}",
            )
        except Exception as e:
            return DeterministicCheckResult(
                check_name=check_name,
                passed=False,
                expected="exit code 0",
                actual=str(e),
                message=f"Build command failed to execute: {e}",
            )

    # =========================================================================
    # Runtime Smoke Tests
    # =========================================================================

    def _check_smoke_test(
        self, smoke_test: SmokeTest, cwd: Optional[str]
    ) -> DeterministicCheckResult:
        """Run a smoke test to verify runtime behavior.

        Supports:
        - Simple command execution (exit code check)
        - Background processes with wait_for string
        - HTTP health checks

        Args:
            smoke_test: SmokeTest configuration
            cwd: Working directory

        Returns:
            DeterministicCheckResult with pass/fail status
        """
        check_name = f"smoke_test[{smoke_test.command[:25]}...]" if len(smoke_test.command) > 25 else f"smoke_test[{smoke_test.command}]"
        process: Optional[subprocess.Popen[str]] = None

        try:
            if smoke_test.background:
                # Run in background and wait for specific output or health check
                return self._run_background_smoke_test(smoke_test, cwd, check_name)
            else:
                # Simple foreground command
                result = subprocess.run(
                    smoke_test.command,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=smoke_test.timeout,
                )

                if result.returncode != 0:
                    return DeterministicCheckResult(
                        check_name=check_name,
                        passed=False,
                        expected="exit code 0",
                        actual=f"exit code {result.returncode}",
                        message=f"Smoke test failed: {result.stderr[:200] if result.stderr else 'unknown error'}",
                    )

                return DeterministicCheckResult(
                    check_name=check_name,
                    passed=True,
                    expected="exit code 0",
                    actual="exit code 0",
                    message="Smoke test passed",
                )

        except subprocess.TimeoutExpired:
            return DeterministicCheckResult(
                check_name=check_name,
                passed=False,
                expected="completion",
                actual="timeout",
                message=f"Smoke test timed out after {smoke_test.timeout}s",
            )
        except Exception as e:
            return DeterministicCheckResult(
                check_name=check_name,
                passed=False,
                expected="success",
                actual=str(e),
                message=f"Smoke test error: {e}",
            )

    def _run_background_smoke_test(
        self, smoke_test: SmokeTest, cwd: Optional[str], check_name: str
    ) -> DeterministicCheckResult:
        """Run a background smoke test (e.g., dev server).

        Starts the process, waits for ready signal or health check,
        then cleans up.
        """
        import time

        process: Optional[subprocess.Popen[str]] = None

        try:
            # Start background process
            process = subprocess.Popen(
                smoke_test.command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid if os.name != 'nt' else None,
            )

            start_time = time.time()
            output_lines: List[str] = []

            # Wait for ready signal or timeout
            while time.time() - start_time < smoke_test.timeout:
                if process.stdout:
                    # Non-blocking read
                    import select
                    if os.name != 'nt' and select.select([process.stdout], [], [], 0.5)[0]:
                        line = process.stdout.readline()
                        if line:
                            output_lines.append(line)
                            # Check for wait_for string
                            if smoke_test.wait_for and smoke_test.wait_for.lower() in line.lower():
                                break

                # Check health endpoint if specified
                if smoke_test.health_check:
                    try:
                        import urllib.request
                        response = urllib.request.urlopen(
                            smoke_test.health_check, timeout=2
                        )
                        if response.status == smoke_test.expected_status:
                            return DeterministicCheckResult(
                                check_name=check_name,
                                passed=True,
                                expected=f"HTTP {smoke_test.expected_status}",
                                actual=f"HTTP {response.status}",
                                message=f"Health check passed: {smoke_test.health_check}",
                            )
                    except Exception:
                        pass  # Keep waiting

                time.sleep(0.5)

            # Check if we got the wait_for string
            if smoke_test.wait_for:
                full_output = ''.join(output_lines)
                if smoke_test.wait_for.lower() in full_output.lower():
                    return DeterministicCheckResult(
                        check_name=check_name,
                        passed=True,
                        expected=f"output contains '{smoke_test.wait_for}'",
                        actual="found",
                        message="Background process ready",
                    )
                else:
                    return DeterministicCheckResult(
                        check_name=check_name,
                        passed=False,
                        expected=f"output contains '{smoke_test.wait_for}'",
                        actual=full_output[:200] + "..." if len(full_output) > 200 else full_output,
                        message="Ready signal not found in output",
                    )

            # If health check was specified but never succeeded
            if smoke_test.health_check:
                return DeterministicCheckResult(
                    check_name=check_name,
                    passed=False,
                    expected=f"HTTP {smoke_test.expected_status} at {smoke_test.health_check}",
                    actual="health check failed",
                    message="Health check endpoint not responding",
                )

            return DeterministicCheckResult(
                check_name=check_name,
                passed=False,
                expected="ready signal",
                actual="timeout",
                message=f"Background process did not become ready within {smoke_test.timeout}s",
            )

        finally:
            # Cleanup: kill the background process
            if process:
                try:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    else:
                        process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    if process:
                        process.kill()

            # Run cleanup command if specified
            if smoke_test.cleanup:
                try:
                    subprocess.run(
                        smoke_test.cleanup,
                        shell=True,
                        cwd=cwd,
                        timeout=10,
                        capture_output=True,
                    )
                except Exception as e:
                    logger.warning(f"Cleanup command failed: {e}")

    # =========================================================================
    # Repository Cleanliness
    # =========================================================================

    def _check_git_clean(self, cwd: Optional[str]) -> DeterministicCheckResult:
        """Check that git working directory is clean (no uncommitted changes).

        Args:
            cwd: Working directory (should be a git repository)

        Returns:
            DeterministicCheckResult with pass/fail status
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return DeterministicCheckResult(
                    check_name="git_clean",
                    passed=False,
                    expected="clean working directory",
                    actual="git command failed",
                    message=f"git status failed: {result.stderr}",
                )

            status_output = result.stdout.strip()

            if status_output:
                # There are uncommitted changes
                lines = status_output.split('\n')
                num_changes = len(lines)
                preview = '\n'.join(lines[:5])
                if num_changes > 5:
                    preview += f"\n... and {num_changes - 5} more"

                return DeterministicCheckResult(
                    check_name="git_clean",
                    passed=False,
                    expected="clean working directory",
                    actual=f"{num_changes} uncommitted changes",
                    message=f"Working directory not clean:\n{preview}",
                )

            return DeterministicCheckResult(
                check_name="git_clean",
                passed=True,
                expected="clean working directory",
                actual="clean",
                message="Git working directory is clean",
            )

        except subprocess.TimeoutExpired:
            return DeterministicCheckResult(
                check_name="git_clean",
                passed=False,
                expected="clean working directory",
                actual="timeout",
                message="git status timed out",
            )
        except FileNotFoundError:
            return DeterministicCheckResult(
                check_name="git_clean",
                passed=False,
                expected="clean working directory",
                actual="git not found",
                message="git command not found",
            )
        except Exception as e:
            return DeterministicCheckResult(
                check_name="git_clean",
                passed=False,
                expected="clean working directory",
                actual=str(e),
                message=f"Error checking git status: {e}",
            )

    # =========================================================================
    # Permission/Security Checks
    # =========================================================================

    def _check_forbidden_patterns(
        self, patterns: List[str], commands: List[str]
    ) -> DeterministicCheckResult:
        """Check that no commands match forbidden patterns.

        Args:
            patterns: Regex patterns that are forbidden
            commands: List of commands that were executed

        Returns:
            DeterministicCheckResult with pass/fail status
        """
        violations: List[Tuple[str, str]] = []

        for pattern in patterns:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for cmd in commands:
                    if regex.search(cmd):
                        violations.append((pattern, cmd))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        if violations:
            violation_msgs = [f"'{cmd}' matches '{pat}'" for pat, cmd in violations[:3]]
            return DeterministicCheckResult(
                check_name="forbidden_patterns",
                passed=False,
                expected=f"no commands matching {patterns}",
                actual=f"{len(violations)} violations",
                message=f"Forbidden patterns found: {'; '.join(violation_msgs)}",
            )

        return DeterministicCheckResult(
            check_name="forbidden_patterns",
            passed=True,
            expected=f"no commands matching {patterns}",
            actual="no violations",
            message="No forbidden patterns found in commands",
        )

    def _check_no_sudo(self, commands: List[str]) -> DeterministicCheckResult:
        """Check that no sudo/privilege escalation commands were used.

        Args:
            commands: List of commands that were executed

        Returns:
            DeterministicCheckResult with pass/fail status
        """
        sudo_commands: List[str] = []

        for cmd in commands:
            for pattern in SUDO_PATTERNS:
                if re.search(pattern, cmd, re.IGNORECASE):
                    sudo_commands.append(cmd)
                    break

        if sudo_commands:
            return DeterministicCheckResult(
                check_name="no_sudo",
                passed=False,
                expected="no privilege escalation",
                actual=f"{len(sudo_commands)} sudo commands",
                message=f"Privilege escalation detected: {sudo_commands[0][:50]}...",
            )

        return DeterministicCheckResult(
            check_name="no_sudo",
            passed=True,
            expected="no privilege escalation",
            actual="none found",
            message="No sudo/privilege escalation commands used",
        )

    def _check_no_external_network(
        self, commands: List[str]
    ) -> DeterministicCheckResult:
        """Check that no external network calls were made.

        Allows localhost/127.0.0.1 but blocks external URLs.

        Args:
            commands: List of commands that were executed

        Returns:
            DeterministicCheckResult with pass/fail status
        """
        external_calls: List[str] = []

        for cmd in commands:
            for pattern in EXTERNAL_NETWORK_PATTERNS:
                if re.search(pattern, cmd, re.IGNORECASE):
                    external_calls.append(cmd)
                    break

        if external_calls:
            return DeterministicCheckResult(
                check_name="no_network_external",
                passed=False,
                expected="no external network calls",
                actual=f"{len(external_calls)} external calls",
                message=f"External network call detected: {external_calls[0][:50]}...",
            )

        return DeterministicCheckResult(
            check_name="no_network_external",
            passed=True,
            expected="no external network calls",
            actual="none found",
            message="No external network calls detected",
        )
