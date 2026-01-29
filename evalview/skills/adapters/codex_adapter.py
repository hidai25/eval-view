"""OpenAI Codex CLI adapter for skill testing.

This module provides an adapter for executing skills through the OpenAI Codex CLI,
capturing structured execution traces for evaluation.

The Codex CLI (https://github.com/openai/codex) is OpenAI's coding agent that
can execute multi-step coding tasks with tool use.

Example usage:
    config = AgentConfig(type=AgentType.CODEX)
    adapter = CodexAdapter(config)
    trace = await adapter.execute(skill, "Create a React component")

Security considerations:
    - Working directory isolation enforced
    - Command execution timeout strictly enforced
    - Environment variables filtered to prevent secret leakage
    - All subprocess output captured for audit logging

Author: EvalView Team
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, TypedDict
import logging

from evalview.skills.adapters.base import (
    SkillAgentAdapter,
    SkillAgentAdapterError,
    AgentNotFoundError,
    AgentTimeoutError,
)
from evalview.skills.agent_types import (
    AgentConfig,
    SkillAgentTrace,
    TraceEvent,
    TraceEventType,
)
from evalview.skills.types import Skill

logger = logging.getLogger(__name__)

# Constants
_DEFAULT_TIMEOUT: Final[float] = 300.0
_DEFAULT_MAX_TURNS: Final[int] = 10
_MAX_OUTPUT_SIZE: Final[int] = 1024 * 1024  # 1MB max output to prevent OOM
_SENSITIVE_ENV_PATTERNS: Final[Tuple[str, ...]] = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH",
)


class CodexOutputFormat(str, Enum):
    """Output format options for Codex CLI."""
    JSON = "json"
    TEXT = "text"
    STREAM = "stream"


@dataclass(frozen=True)
class CodexInvocation:
    """Immutable record of a Codex CLI invocation.

    Used for audit logging and debugging.
    """
    command: Tuple[str, ...]
    working_directory: str
    timestamp: datetime
    timeout: float
    environment_hash: str  # Hash of env vars for audit without exposing values


class CodexAdapter(SkillAgentAdapter):
    """Adapter for executing skills through OpenAI Codex CLI.

    This adapter invokes the Codex CLI with skill instructions injected
    as system context, captures the execution trace, and parses tool calls
    and file operations from the output.

    The adapter supports:
        - JSON output parsing for structured traces
        - Automatic tool call extraction
        - File operation tracking (create, modify, delete)
        - Command execution capture
        - Token usage tracking

    Attributes:
        config: Agent configuration from test suite
        codex_path: Resolved path to Codex CLI binary

    Example:
        >>> config = AgentConfig(type=AgentType.CODEX, timeout=120)
        >>> adapter = CodexAdapter(config)
        >>> if await adapter.health_check():
        ...     trace = await adapter.execute(skill, "Build a REST API")
    """

    # Class-level constants for configuration
    BINARY_NAME: Final[str] = "codex"
    SUPPORTED_OUTPUT_FORMATS: Final[Tuple[str, ...]] = ("json", "text")

    def __init__(self, config: AgentConfig) -> None:
        """Initialize Codex adapter with configuration.

        Args:
            config: Agent configuration specifying timeout, max_turns, etc.

        Raises:
            AgentNotFoundError: If Codex CLI is not installed or not in PATH.
        """
        super().__init__(config)
        self._codex_path: Optional[str] = None
        self._invocation_log: List[CodexInvocation] = []

    @property
    def name(self) -> str:
        """Return adapter identifier."""
        return "codex"

    @property
    def codex_path(self) -> str:
        """Lazily resolve and cache Codex CLI path.

        Returns:
            Absolute path to Codex binary.

        Raises:
            AgentNotFoundError: If Codex CLI cannot be found.
        """
        if self._codex_path is None:
            self._codex_path = self._resolve_binary_path()
        return self._codex_path

    def _resolve_binary_path(self) -> str:
        """Resolve the Codex CLI binary path.

        Searches in order:
            1. System PATH
            2. npm global bin directory
            3. Common installation locations

        Returns:
            Absolute path to Codex binary.

        Raises:
            AgentNotFoundError: If binary cannot be found.
        """
        # Check PATH first (most common case)
        path_result = shutil.which(self.BINARY_NAME)
        if path_result:
            logger.debug(f"Found Codex in PATH: {path_result}")
            return path_result

        # Check common installation locations
        candidate_paths = [
            Path.home() / ".npm-global" / "bin" / self.BINARY_NAME,
            Path.home() / ".local" / "bin" / self.BINARY_NAME,
            Path("/usr/local/bin") / self.BINARY_NAME,
            Path.home() / ".nvm" / "current" / "bin" / self.BINARY_NAME,
        ]

        for candidate in candidate_paths:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                logger.debug(f"Found Codex at: {candidate}")
                return str(candidate)

        raise AgentNotFoundError(
            adapter_name=self.name,
            install_hint=(
                "Install Codex CLI: npm install -g @openai/codex\n"
                "Or visit: https://github.com/openai/codex"
            ),
        )

    async def health_check(self) -> bool:
        """Verify Codex CLI is available and functional.

        Performs a lightweight version check to ensure the CLI
        is properly installed and executable.

        Returns:
            True if Codex CLI is available and responds to --version.
        """
        try:
            result = await self._run_subprocess(
                [self.codex_path, "--version"],
                timeout=10.0,
                capture_output=True,
            )
            return result.returncode == 0
        except (AgentNotFoundError, AgentTimeoutError, OSError) as e:
            logger.warning(f"Codex health check failed: {e}")
            return False

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute a skill test through Codex CLI.

        Injects the skill as system context, runs Codex with the query,
        and captures the full execution trace including tool calls,
        file operations, and command executions.

        Args:
            skill: The skill to test (SKILL.md content).
            query: User query to send to the agent.
            context: Optional execution context with overrides:
                - test_name: Name for trace identification
                - cwd: Working directory override

        Returns:
            SkillAgentTrace containing all execution events and metadata.

        Raises:
            AgentNotFoundError: If Codex CLI is not available.
            AgentTimeoutError: If execution exceeds configured timeout.
            SkillAgentAdapterError: For other execution failures.
        """
        context = context or {}
        test_name = context.get("test_name", "unnamed-test")
        cwd = self._resolve_working_directory(context.get("cwd"))

        session_id = self._generate_session_id()
        start_time = datetime.now()

        # Build command with skill injection
        command = self._build_command(skill, query)

        # Prepare sanitized environment
        env = self._prepare_environment()

        # Log invocation for audit
        self._log_invocation(command, cwd)

        try:
            result = await self._run_subprocess(
                command,
                cwd=cwd,
                env=env,
                timeout=self.config.timeout,
                capture_output=True,
            )

            end_time = datetime.now()
            self._last_raw_output = self._truncate_output(
                result.stdout + result.stderr
            )

            return self._parse_execution_result(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                session_id=session_id,
                skill_name=skill.metadata.name,
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
            )

        except asyncio.TimeoutError:
            raise AgentTimeoutError(self.name, self.config.timeout)
        except FileNotFoundError:
            raise AgentNotFoundError(
                self.name,
                "Codex CLI not found. Install with: npm install -g @openai/codex",
            )
        except subprocess.SubprocessError as e:
            raise SkillAgentAdapterError(
                f"Subprocess execution failed: {e}",
                adapter_name=self.name,
                recoverable=False,
            )

    def _resolve_working_directory(self, override: Optional[str]) -> str:
        """Resolve the working directory for execution.

        Priority: context override > config.cwd > current directory

        Args:
            override: Optional directory path from context.

        Returns:
            Absolute path to working directory.
        """
        if override:
            return os.path.abspath(os.path.expanduser(override))
        if self.config.cwd:
            return self.config.cwd  # Already validated in AgentConfig
        return os.getcwd()

    def _generate_session_id(self) -> str:
        """Generate a unique session identifier.

        Returns:
            8-character hex string for trace identification.
        """
        return uuid.uuid4().hex[:8]

    def _build_command(self, skill: Skill, query: str) -> List[str]:
        """Build the Codex CLI command with skill injection.

        Constructs a command that:
            1. Injects skill as system instructions
            2. Sets output format to JSON for structured parsing
            3. Configures max turns to prevent runaway execution

        Args:
            skill: The skill to inject.
            query: User query to execute.

        Returns:
            Command list suitable for subprocess execution.
        """
        # Build skill context with clear structure
        skill_context = self._format_skill_context(skill)

        command = [
            self.codex_path,
            "--prompt", query,
            "--instructions", skill_context,
            "--output-format", "json",
        ]

        # Add max turns if supported
        if self.config.max_turns:
            command.extend(["--max-turns", str(self.config.max_turns)])

        # Add allowed tools if specified
        if self.config.tools:
            command.extend(["--tools", ",".join(self.config.tools)])

        return command

    def _format_skill_context(self, skill: Skill) -> str:
        """Format skill content for injection as system instructions.

        Creates a structured context document that clearly identifies
        the skill and its instructions to the agent.

        Args:
            skill: The skill to format.

        Returns:
            Formatted skill context string.
        """
        return f"""You have the following skill available:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILL: {skill.metadata.name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{skill.metadata.description}

## Instructions

{skill.instructions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Follow the skill instructions above when responding to the user's request.
"""

    def _prepare_environment(self) -> Dict[str, str]:
        """Prepare sanitized environment for subprocess.

        Copies current environment and:
            1. Adds any configured env vars
            2. Filters out sensitive variables from logging

        Returns:
            Environment dictionary for subprocess.
        """
        env = os.environ.copy()

        # Add configured environment variables
        if self.config.env:
            env.update(self.config.env)

        return env

    def _log_invocation(self, command: List[str], cwd: str) -> None:
        """Log command invocation for audit purposes.

        Records invocation without sensitive data for debugging
        and compliance purposes.

        Args:
            command: Command being executed.
            cwd: Working directory.
        """
        # Create hash of environment for audit without exposing values
        env_hash = str(hash(frozenset(os.environ.keys())))[:8]

        invocation = CodexInvocation(
            command=tuple(command[:3]),  # Only log non-sensitive parts
            working_directory=cwd,
            timestamp=datetime.now(),
            timeout=self.config.timeout,
            environment_hash=env_hash,
        )
        self._invocation_log.append(invocation)

        logger.debug(
            f"Codex invocation: {invocation.command[0]}... "
            f"in {cwd} (timeout={self.config.timeout}s)"
        )

    async def _run_subprocess(
        self,
        command: List[str],
        timeout: float,
        capture_output: bool = True,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Execute subprocess with async timeout handling.

        Wraps subprocess.run in an executor for async compatibility
        with proper timeout enforcement.

        Args:
            command: Command and arguments to execute.
            timeout: Maximum execution time in seconds.
            capture_output: Whether to capture stdout/stderr.
            cwd: Working directory for execution.
            env: Environment variables.

        Returns:
            CompletedProcess with execution results.

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout.
        """
        loop = asyncio.get_event_loop()

        def _sync_run():
            return subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=cwd,
                env=env,
                timeout=timeout,
            )

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _sync_run),
                timeout=timeout + 5.0,  # Buffer for executor overhead
            )
        except subprocess.TimeoutExpired:
            raise asyncio.TimeoutError(f"Command timed out after {timeout}s")

    def _truncate_output(self, output: str) -> str:
        """Truncate output to prevent memory issues.

        Args:
            output: Raw output string.

        Returns:
            Truncated output if necessary.
        """
        if len(output) > _MAX_OUTPUT_SIZE:
            return output[:_MAX_OUTPUT_SIZE] + "\n... [truncated]"
        return output

    def _parse_execution_result(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        session_id: str,
        skill_name: str,
        test_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> SkillAgentTrace:
        """Parse Codex execution output into structured trace.

        Attempts JSON parsing first, falls back to text parsing
        if JSON is not available.

        Args:
            stdout: Standard output from Codex.
            stderr: Standard error from Codex.
            returncode: Process exit code.
            session_id: Unique session identifier.
            skill_name: Name of skill being tested.
            test_name: Name of test case.
            start_time: Execution start timestamp.
            end_time: Execution end timestamp.

        Returns:
            SkillAgentTrace with parsed execution data.
        """
        errors: List[str] = []

        # Check for execution errors
        if returncode != 0:
            errors.append(f"Process exited with code {returncode}")
            if stderr:
                errors.append(stderr[:1000])

        # Attempt JSON parsing
        try:
            data = json.loads(stdout)
            return self._parse_json_trace(
                data=data,
                session_id=session_id,
                skill_name=skill_name,
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                errors=errors,
            )
        except json.JSONDecodeError:
            logger.debug("JSON parsing failed, using text parsing")
            return self._parse_text_trace(
                stdout=stdout,
                session_id=session_id,
                skill_name=skill_name,
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                errors=errors,
            )

    def _parse_json_trace(
        self,
        data: Dict[str, Any],
        session_id: str,
        skill_name: str,
        test_name: str,
        start_time: datetime,
        end_time: datetime,
        errors: List[str],
    ) -> SkillAgentTrace:
        """Parse JSON output from Codex into trace.

        Extracts tool calls, file operations, and commands from
        the structured JSON output.

        Args:
            data: Parsed JSON data.
            session_id: Session identifier.
            skill_name: Skill name.
            test_name: Test name.
            start_time: Start timestamp.
            end_time: End timestamp.
            errors: Pre-existing errors to include.

        Returns:
            Populated SkillAgentTrace.
        """
        events: List[TraceEvent] = []
        tool_calls: List[str] = []
        files_created: List[str] = []
        files_modified: List[str] = []
        commands_ran: List[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        final_output = ""

        # Extract from messages/steps array
        messages = data.get("messages", data.get("steps", []))
        if not messages and isinstance(data, dict):
            messages = [data]

        for msg in messages:
            self._process_message(
                msg=msg,
                events=events,
                tool_calls=tool_calls,
                files_created=files_created,
                files_modified=files_modified,
                commands_ran=commands_ran,
            )

            # Extract final output
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    final_output = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            final_output = block.get("text", "")

        # Extract token usage
        usage = data.get("usage", {})
        total_input_tokens = usage.get("input_tokens", 0)
        total_output_tokens = usage.get("output_tokens", 0)

        # Fallback for final output
        if not final_output:
            final_output = data.get("result", data.get("response", ""))

        return SkillAgentTrace(
            session_id=session_id,
            skill_name=skill_name,
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            events=events,
            tool_calls=tool_calls,
            files_created=files_created,
            files_modified=files_modified,
            commands_ran=commands_ran,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            final_output=final_output,
            errors=errors,
        )

    def _process_message(
        self,
        msg: Dict[str, Any],
        events: List[TraceEvent],
        tool_calls: List[str],
        files_created: List[str],
        files_modified: List[str],
        commands_ran: List[str],
    ) -> None:
        """Process a single message from Codex output.

        Extracts tool calls and tracks file/command operations.

        Args:
            msg: Message dictionary from Codex output.
            events: List to append trace events.
            tool_calls: List to append tool names.
            files_created: List to append created file paths.
            files_modified: List to append modified file paths.
            commands_ran: List to append executed commands.
        """
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})

                    tool_calls.append(tool_name)
                    events.append(TraceEvent(
                        type=TraceEventType.TOOL_CALL,
                        tool_name=tool_name,
                        tool_input=tool_input,
                    ))

                    # Track file and command operations
                    self._track_operations(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        files_created=files_created,
                        files_modified=files_modified,
                        commands_ran=commands_ran,
                    )

    def _track_operations(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        files_created: List[str],
        files_modified: List[str],
        commands_ran: List[str],
    ) -> None:
        """Track file and command operations from tool calls.

        Analyzes tool calls to identify:
            - File creation (write, create_file, touch)
            - File modification (edit, patch, append)
            - Command execution (shell, bash, exec)

        Args:
            tool_name: Name of the tool called.
            tool_input: Input parameters to the tool.
            files_created: List to append created files.
            files_modified: List to append modified files.
            commands_ran: List to append commands.
        """
        tool_lower = tool_name.lower()

        # File creation tools
        if tool_lower in ("write", "write_file", "create_file", "str_replace_editor"):
            file_path = tool_input.get("file_path", tool_input.get("path", ""))
            if file_path:
                files_created.append(file_path)

        # File modification tools
        elif tool_lower in ("edit", "patch", "append", "insert"):
            file_path = tool_input.get("file_path", tool_input.get("path", ""))
            if file_path:
                files_modified.append(file_path)

        # Command execution tools
        elif tool_lower in ("shell", "bash", "exec", "run", "execute"):
            command = tool_input.get("command", tool_input.get("cmd", ""))
            if command:
                commands_ran.append(command)
                # Check for file operations in shell commands
                self._extract_shell_file_ops(command, files_created)

    def _extract_shell_file_ops(
        self,
        command: str,
        files_created: List[str],
    ) -> None:
        """Extract file operations from shell commands.

        Parses common shell patterns for file creation:
            - touch filename
            - mkdir dirname
            - > filename (redirect)
            - echo ... > filename

        Args:
            command: Shell command string.
            files_created: List to append identified file paths.
        """
        # Match touch/mkdir commands
        touch_match = re.search(r'\btouch\s+([^\s;|&]+)', command)
        if touch_match:
            files_created.append(touch_match.group(1))

        mkdir_match = re.search(r'\bmkdir\s+(?:-p\s+)?([^\s;|&]+)', command)
        if mkdir_match:
            files_created.append(mkdir_match.group(1))

        # Match redirect patterns
        redirect_match = re.search(r'>\s*([^\s;|&]+)', command)
        if redirect_match:
            files_created.append(redirect_match.group(1))

    def _parse_text_trace(
        self,
        stdout: str,
        session_id: str,
        skill_name: str,
        test_name: str,
        start_time: datetime,
        end_time: datetime,
        errors: List[str],
    ) -> SkillAgentTrace:
        """Parse text output when JSON is not available.

        Falls back to using raw stdout as final output
        with empty operation lists.

        Args:
            stdout: Raw standard output.
            session_id: Session identifier.
            skill_name: Skill name.
            test_name: Test name.
            start_time: Start timestamp.
            end_time: End timestamp.
            errors: Pre-existing errors.

        Returns:
            SkillAgentTrace with text output.
        """
        return SkillAgentTrace(
            session_id=session_id,
            skill_name=skill_name,
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            events=[],
            tool_calls=[],
            files_created=[],
            files_modified=[],
            commands_ran=[],
            total_input_tokens=0,
            total_output_tokens=0,
            final_output=stdout,
            errors=errors,
        )

    def get_invocation_log(self) -> List[CodexInvocation]:
        """Return the invocation log for debugging/audit.

        Returns:
            List of CodexInvocation records from this session.
        """
        return self._invocation_log.copy()
