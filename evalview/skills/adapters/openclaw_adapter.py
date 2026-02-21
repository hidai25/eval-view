"""OpenClaw adapter for skill testing.

This module provides an adapter for executing skills through the OpenClaw CLI,
capturing structured execution traces for evaluation.

OpenClaw (https://github.com/openclaw/openclaw) is an open-source autonomous
AI agent that runs locally and can execute tasks via large language models.
It uses AgentSkills (SKILL.md files) to extend its capabilities.

Example usage:
    config = AgentConfig(type=AgentType.OPENCLAW)
    adapter = OpenClawAdapter(config)
    trace = await adapter.execute(skill, "Create a React component")

Security considerations:
    - Working directory isolation enforced
    - Command execution timeout strictly enforced
    - Environment variables filtered to prevent secret leakage
    - All subprocess output captured for audit logging
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple
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
_MAX_OUTPUT_SIZE: Final[int] = 1024 * 1024  # 1MB max output to prevent OOM
_SENSITIVE_ENV_PATTERNS: Final[Tuple[str, ...]] = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH",
)
_SESSION_ID_LENGTH: Final[int] = 8


@dataclass(frozen=True)
class OpenClawInvocation:
    """Immutable record of an OpenClaw CLI invocation.

    Used for audit logging and debugging.
    """
    command: Tuple[str, ...]
    working_directory: str
    timestamp: datetime
    timeout: float
    environment_hash: str


class OpenClawAdapter(SkillAgentAdapter):
    """Adapter for executing skills through OpenClaw CLI.

    This adapter invokes the OpenClaw CLI with skill instructions injected
    as context, captures the execution trace, and parses tool calls
    and file operations from the output.

    OpenClaw uses AgentSkills (SKILL.md files with YAML frontmatter) to
    extend its capabilities. The adapter injects skills via the
    --skill-path flag or --instructions flag depending on availability.

    The adapter supports:
        - JSON output parsing for structured traces
        - Automatic tool call extraction
        - File operation tracking (create, modify, delete)
        - Command execution capture
        - Token usage tracking

    Attributes:
        config: Agent configuration from test suite
        openclaw_path: Resolved path to OpenClaw CLI binary

    Example:
        >>> config = AgentConfig(type=AgentType.OPENCLAW, timeout=120)
        >>> adapter = OpenClawAdapter(config)
        >>> if await adapter.health_check():
        ...     trace = await adapter.execute(skill, "Build a REST API")
    """

    BINARY_NAME: Final[str] = "openclaw"

    def __init__(self, config: AgentConfig) -> None:
        """Initialize OpenClaw adapter with configuration.

        Args:
            config: Agent configuration specifying timeout, max_turns, etc.

        Raises:
            AgentNotFoundError: If OpenClaw CLI is not installed or not in PATH.
        """
        super().__init__(config)
        self._openclaw_path: Optional[str] = None
        self._invocation_log: List[OpenClawInvocation] = []

    @property
    def name(self) -> str:
        """Return adapter identifier."""
        return "openclaw"

    @property
    def openclaw_path(self) -> str:
        """Lazily resolve and cache OpenClaw CLI path.

        Returns:
            Absolute path to OpenClaw binary.

        Raises:
            AgentNotFoundError: If OpenClaw CLI cannot be found.
        """
        if self._openclaw_path is None:
            self._openclaw_path = self._resolve_binary_path()
        return self._openclaw_path

    def _resolve_binary_path(self) -> str:
        """Resolve the OpenClaw CLI binary path.

        Searches in order:
            1. System PATH
            2. pip/pipx installation paths
            3. Common installation locations
            4. Homebrew paths

        Returns:
            Absolute path to OpenClaw binary.

        Raises:
            AgentNotFoundError: If binary cannot be found.
        """
        # Check PATH first (most common case)
        path_result = shutil.which(self.BINARY_NAME)
        if path_result:
            logger.debug(f"Found OpenClaw in PATH: {path_result}")
            return path_result

        # Check common installation locations
        candidate_paths = [
            # pip/pipx installations
            Path.home() / ".local" / "bin" / self.BINARY_NAME,
            # npm global (OpenClaw also distributes via npm)
            Path.home() / ".npm-global" / "bin" / self.BINARY_NAME,
            # Homebrew
            Path("/usr/local/bin") / self.BINARY_NAME,
            Path("/opt/homebrew/bin") / self.BINARY_NAME,
            # nvm
            Path.home() / ".nvm" / "current" / "bin" / self.BINARY_NAME,
            # OpenClaw workspace default
            Path.home() / ".openclaw" / "bin" / self.BINARY_NAME,
        ]

        for candidate in candidate_paths:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                logger.debug(f"Found OpenClaw at: {candidate}")
                return str(candidate)

        raise AgentNotFoundError(
            adapter_name=self.name,
            install_hint=(
                "Install OpenClaw: pip install openclaw\n"
                "Or visit: https://github.com/openclaw/openclaw"
            ),
        )

    async def health_check(self) -> bool:
        """Verify OpenClaw CLI is available and functional.

        Performs a lightweight version check to ensure the CLI
        is properly installed and executable.

        Returns:
            True if OpenClaw CLI is available and responds to --version.
        """
        try:
            result = await self._run_subprocess(
                [self.openclaw_path, "--version"],
                timeout=10.0,
                capture_output=True,
            )
            return result.returncode == 0
        except (AgentNotFoundError, AgentTimeoutError, OSError) as e:
            logger.warning(f"OpenClaw health check failed: {e}")
            return False

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute a skill test through OpenClaw CLI.

        Injects the skill as context, runs OpenClaw with the query,
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
            AgentNotFoundError: If OpenClaw CLI is not available.
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
                "OpenClaw CLI not found. Install with: pip install openclaw",
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
        import uuid
        return uuid.uuid4().hex[:_SESSION_ID_LENGTH]

    def _build_command(self, skill: Skill, query: str) -> List[str]:
        """Build the OpenClaw CLI command with skill injection.

        Constructs a command that:
            1. Injects skill as instructions context
            2. Sets output format to JSON for structured parsing
            3. Configures max turns to prevent runaway execution
            4. Runs in non-interactive (headless) mode

        OpenClaw skills use SKILL.md files with YAML frontmatter. The adapter
        injects the skill content via --instructions when a skill path isn't
        available, or via --skill-path if the skill has a file path.

        Args:
            skill: The skill to inject.
            query: User query to execute.

        Returns:
            Command list suitable for subprocess execution.
        """
        skill_context = self._format_skill_context(skill)

        command = [
            self.openclaw_path,
            "run",
            "--prompt", query,
            "--instructions", skill_context,
            "--output-format", "json",
            "--headless",  # Non-interactive mode for testing
        ]

        # Add max turns if configured
        if self.config.max_turns:
            command.extend(["--max-turns", str(self.config.max_turns)])

        # Add allowed tools if specified
        if self.config.tools:
            command.extend(["--tools", ",".join(self.config.tools)])

        # Add skill path if available (OpenClaw can load skills from path)
        if skill.file_path and os.path.isfile(skill.file_path):
            command.extend(["--skill-path", skill.file_path])

        return command

    def _format_skill_context(self, skill: Skill) -> str:
        """Format skill content for injection as instructions.

        Creates a structured context document that clearly identifies
        the skill and its instructions to the agent. Follows the
        OpenClaw AgentSkills format.

        Args:
            skill: The skill to format.

        Returns:
            Formatted skill context string.
        """
        return f"""You have the following AgentSkill available:

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
            2. Filters out sensitive variables to prevent secret leakage

        Returns:
            Environment dictionary for subprocess.
        """
        env = os.environ.copy()

        # Filter out sensitive environment variables
        filtered_keys = [
            key for key in env
            if any(pattern in key.upper() for pattern in _SENSITIVE_ENV_PATTERNS)
        ]
        for key in filtered_keys:
            logger.debug(f"Filtered sensitive env var from OpenClaw execution: {key}")
            del env[key]

        # Add configured environment variables (these are intentional, so not filtered)
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
        env_hash = str(hash(frozenset(os.environ.keys())))[:8]

        invocation = OpenClawInvocation(
            command=tuple(command[:3]),  # Only log non-sensitive parts
            working_directory=cwd,
            timestamp=datetime.now(),
            timeout=self.config.timeout,
            environment_hash=env_hash,
        )
        self._invocation_log.append(invocation)

        logger.debug(
            f"OpenClaw invocation: {invocation.command[0]}... "
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
        loop = asyncio.get_running_loop()

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
        """Parse OpenClaw execution output into structured trace.

        Attempts JSON parsing first, then JSONL (stream format),
        falls back to text parsing if neither is available.

        Args:
            stdout: Standard output from OpenClaw.
            stderr: Standard error from OpenClaw.
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

        # Attempt JSON parsing (single JSON object)
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
            pass

        # Attempt JSONL parsing (one JSON per line, like stream format)
        lines = [l for l in stdout.strip().split("\n") if l.strip()]
        json_lines = []
        for line in lines:
            try:
                json_lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if json_lines:
            return self._parse_jsonl_trace(
                events_data=json_lines,
                session_id=session_id,
                skill_name=skill_name,
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                errors=errors,
            )

        # Fallback to text parsing
        logger.debug("JSON/JSONL parsing failed, using text parsing")
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
        """Parse JSON output from OpenClaw into trace.

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

        # Extract from messages/steps/actions array
        messages = data.get("messages", data.get("steps", data.get("actions", [])))
        if not messages and isinstance(data, dict):
            messages = [data]

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            self._process_message(
                msg=msg,
                events=events,
                tool_calls=tool_calls,
                files_created=files_created,
                files_modified=files_modified,
                commands_ran=commands_ran,
            )

            # Extract final output from assistant messages
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
        total_input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        total_output_tokens = usage.get(
            "output_tokens", usage.get("completion_tokens", 0)
        )

        # Fallback for final output
        if not final_output:
            final_output = (
                data.get("result", "")
                or data.get("response", "")
                or data.get("output", "")
            )

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

    def _parse_jsonl_trace(
        self,
        events_data: List[Dict[str, Any]],
        session_id: str,
        skill_name: str,
        test_name: str,
        start_time: datetime,
        end_time: datetime,
        errors: List[str],
    ) -> SkillAgentTrace:
        """Parse JSONL (stream) output from OpenClaw into trace.

        OpenClaw can output one JSON event per line in stream mode.

        Args:
            events_data: List of parsed JSON objects (one per line).
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

        for data in events_data:
            msg_type = data.get("type", "")

            # Extract final result
            if msg_type == "result":
                final_output = data.get("result", data.get("output", ""))
                usage = data.get("usage", {})
                total_input_tokens = usage.get(
                    "input_tokens", usage.get("prompt_tokens", 0)
                )
                total_output_tokens = usage.get(
                    "output_tokens", usage.get("completion_tokens", 0)
                )

            # Extract tool calls from assistant messages
            elif msg_type == "assistant":
                message = data.get("message", data)
                content = message.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "tool_use":
                            tool_name = item.get("name", "")
                            tool_input = item.get("input", {})
                            if tool_name:
                                tool_calls.append(tool_name)
                                self._track_operations(
                                    tool_name, tool_input,
                                    files_created, files_modified, commands_ran,
                                )
                                events.append(TraceEvent(
                                    type=TraceEventType.TOOL_CALL,
                                    tool_name=tool_name,
                                    tool_input=tool_input,
                                ))

            # Extract tool call events
            elif msg_type == "tool_call":
                tool_name = data.get("name", data.get("tool", ""))
                tool_input = data.get("input", data.get("arguments", {}))
                if tool_name:
                    tool_calls.append(tool_name)
                    self._track_operations(
                        tool_name, tool_input,
                        files_created, files_modified, commands_ran,
                    )
                    events.append(TraceEvent(
                        type=TraceEventType.TOOL_CALL,
                        tool_name=tool_name,
                        tool_input=tool_input,
                    ))

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
        """Process a single message from OpenClaw output.

        Extracts tool calls and tracks file/command operations.

        Args:
            msg: Message dictionary from OpenClaw output.
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
            - Command execution (shell, bash, exec, run_command)

        OpenClaw uses a mix of standard tool names and its own conventions.

        Args:
            tool_name: Name of the tool called.
            tool_input: Input parameters to the tool.
            files_created: List to append created files.
            files_modified: List to append modified files.
            commands_ran: List to append commands.
        """
        tool_lower = tool_name.lower()

        # File creation tools
        if tool_lower in (
            "write", "write_file", "create_file", "str_replace_editor",
            "file_write", "save_file",
        ):
            file_path = tool_input.get("file_path", tool_input.get("path", ""))
            if file_path:
                files_created.append(file_path)

        # File modification tools
        elif tool_lower in ("edit", "patch", "append", "insert", "file_edit"):
            file_path = tool_input.get("file_path", tool_input.get("path", ""))
            if file_path:
                files_modified.append(file_path)

        # Command execution tools (OpenClaw uses run_command and shell)
        elif tool_lower in (
            "shell", "bash", "exec", "run", "execute",
            "run_command", "terminal", "cmd",
        ):
            command = tool_input.get("command", tool_input.get("cmd", ""))
            if command:
                commands_ran.append(command)
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
        touch_match = re.search(r'\btouch\s+([^\s;|&]+)', command)
        if touch_match:
            files_created.append(touch_match.group(1))

        mkdir_match = re.search(r'\bmkdir\s+(?:-p\s+)?([^\s;|&]+)', command)
        if mkdir_match:
            files_created.append(mkdir_match.group(1))

        # Match stdout redirects (> file or >> file) but not fd redirects (2>)
        redirect_match = re.search(r'(?:^|[^0-9])>{1,2}\s*([^\s;|&]+)', command)
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

    def get_invocation_log(self) -> List[OpenClawInvocation]:
        """Return the invocation log for debugging/audit.

        Returns:
            List of OpenClawInvocation records from this session.
        """
        return self._invocation_log.copy()
