"""Base adapters for skill agent testing.

This module defines:
    - SkillAgentAdapter: Abstract interface that all skill adapters must implement.
    - CLIAgentAdapter: Shared base for CLI-based adapters (Codex, OpenClaw, etc.)
      that handles subprocess execution, output parsing, trace building, and
      environment sanitization.

Unlike the main AgentAdapter (which tests agents via REST APIs), skill adapters:
1. Inject skills into agents as context
2. Capture detailed execution traces
3. Monitor file system and command execution
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple
import logging

from evalview.skills.agent_types import (
    AgentConfig,
    SkillAgentTrace,
    TraceEvent,
    TraceEventType,
)
from evalview.skills.types import Skill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SkillAgentAdapterError(Exception):
    """Base exception for skill adapter errors.

    Attributes:
        message: Human-readable error description
        adapter_name: Which adapter raised the error
        recoverable: Whether retry might succeed
    """

    def __init__(
        self,
        message: str,
        adapter_name: str = "unknown",
        recoverable: bool = False,
    ):
        self.message = message
        self.adapter_name = adapter_name
        self.recoverable = recoverable
        super().__init__(f"[{adapter_name}] {message}")


class AgentNotFoundError(SkillAgentAdapterError):
    """Raised when agent CLI/binary is not found."""

    def __init__(self, adapter_name: str, install_hint: str):
        super().__init__(
            f"Agent not found. {install_hint}",
            adapter_name=adapter_name,
            recoverable=False,
        )
        self.install_hint = install_hint


class AgentTimeoutError(SkillAgentAdapterError):
    """Raised when agent execution times out."""

    def __init__(self, adapter_name: str, timeout: float):
        super().__init__(
            f"Execution timed out after {timeout}s",
            adapter_name=adapter_name,
            recoverable=True,
        )
        self.timeout = timeout


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------


class SkillAgentAdapter(ABC):
    """Abstract adapter for executing skills through AI agents.

    Each adapter implementation handles:
    1. Building the command/request to invoke the agent
    2. Injecting the skill into agent context
    3. Executing the agent with timeout
    4. Parsing output to extract trace events
    5. Building SkillAgentTrace from execution

    Security considerations:
    - Working directories should be isolated
    - Commands should be logged for audit
    - Timeouts must be enforced
    - Environment variables should not leak secrets
    """

    def __init__(self, config: AgentConfig):
        """Initialize with agent configuration.

        Args:
            config: Agent configuration from test suite
        """
        self.config = config
        self._last_raw_output: Optional[str] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter identifier (e.g., 'claude-code', 'codex')."""
        pass

    @abstractmethod
    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute a skill test and capture trace.

        Args:
            skill: The loaded skill to test
            query: User query to send to the agent
            context: Optional execution context (test_name, cwd override)

        Returns:
            SkillAgentTrace with all execution events

        Raises:
            AgentNotFoundError: If agent binary not found
            AgentTimeoutError: If execution exceeds timeout
            SkillAgentAdapterError: For other execution errors
        """
        pass

    async def health_check(self) -> bool:
        """Check if the agent is available and working.

        Returns:
            True if agent can be executed, False otherwise
        """
        return True

    def get_last_raw_output(self) -> Optional[str]:
        """Get raw output from last execution for debugging.

        Returns:
            Raw stdout/stderr from last execution, or None
        """
        return self._last_raw_output


# ---------------------------------------------------------------------------
# Constants for CLI adapters
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT: Final[float] = 300.0
_MAX_OUTPUT_SIZE: Final[int] = 1024 * 1024  # 1 MB max output to prevent OOM
_SESSION_ID_LENGTH: Final[int] = 8
_SENSITIVE_ENV_PATTERNS: Final[Tuple[str, ...]] = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH",
)


@dataclass(frozen=True)
class CLIInvocation:
    """Immutable record of a CLI invocation for audit logging."""

    adapter_name: str
    command: Tuple[str, ...]
    working_directory: str
    timestamp: datetime
    timeout: float
    environment_hash: str  # hash of env keys — never actual values


# ---------------------------------------------------------------------------
# CLI adapter base — shared subprocess / parsing logic for Codex, OpenClaw
# ---------------------------------------------------------------------------


class CLIAgentAdapter(SkillAgentAdapter):
    """Shared base for CLI-based agent adapters.

    Handles:
        - Binary resolution (PATH + common install locations)
        - Subprocess execution with async timeout
        - JSON / JSONL / text output parsing
        - Tool-call extraction and file/command operation tracking
        - Environment sanitization (filters secrets)
        - Audit-grade invocation logging

    Subclasses must implement four hook methods:
        ``name``             — adapter identifier string
        ``binary_name``      — CLI binary name to search for (e.g. "codex")
        ``_candidate_paths`` — extra filesystem locations to probe
        ``_build_command``   — construct the full CLI command list
        ``_install_hint``    — human-readable install instructions

    Optionally override:
        ``_format_skill_context`` — customise skill injection prompt
        ``_file_creation_tools``  — extend recognised tool names
        ``_file_modification_tools``
        ``_command_execution_tools``
    """

    # Subclasses set these -----------------------------------------------

    @property
    @abstractmethod
    def binary_name(self) -> str:
        """CLI binary name to search for (e.g. ``codex``, ``openclaw``)."""

    @abstractmethod
    def _candidate_paths(self) -> List[Path]:
        """Return extra filesystem paths to search for the binary."""

    @abstractmethod
    def _build_command(self, skill: Skill, query: str) -> List[str]:
        """Build the full CLI command list for execution."""

    @abstractmethod
    def _install_hint(self) -> str:
        """Human-readable install instructions for error messages."""

    # Tool-name sets — subclasses can override to add agent-specific names
    def _file_creation_tools(self) -> Tuple[str, ...]:
        return ("write", "write_file", "create_file", "str_replace_editor")

    def _file_modification_tools(self) -> Tuple[str, ...]:
        return ("edit", "patch", "append", "insert")

    def _command_execution_tools(self) -> Tuple[str, ...]:
        return ("shell", "bash", "exec", "run", "execute")

    # Initialisation -----------------------------------------------------

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._binary_path: Optional[str] = None
        self._invocation_log: List[CLIInvocation] = []

    # Binary resolution --------------------------------------------------

    @property
    def cli_path(self) -> str:
        """Lazily resolve and cache CLI binary path.

        Raises:
            AgentNotFoundError: If the binary cannot be found.
        """
        if self._binary_path is None:
            self._binary_path = self._resolve_binary_path()
        return self._binary_path

    def _resolve_binary_path(self) -> str:
        """Search PATH then candidate locations for the binary.

        Returns:
            Absolute path to the binary.

        Raises:
            AgentNotFoundError: If the binary cannot be found.
        """
        import shutil

        path_result = shutil.which(self.binary_name)
        if path_result:
            logger.debug(f"Found {self.binary_name} in PATH: {path_result}")
            return path_result

        for candidate in self._candidate_paths():
            if candidate.is_file() and os.access(candidate, os.X_OK):
                logger.debug(f"Found {self.binary_name} at: {candidate}")
                return str(candidate)

        raise AgentNotFoundError(
            adapter_name=self.name,
            install_hint=self._install_hint(),
        )

    # Health check -------------------------------------------------------

    async def health_check(self) -> bool:
        """Verify the CLI binary responds to ``--version``."""
        try:
            result = await self._run_subprocess(
                [self.cli_path, "--version"],
                timeout=10.0,
                capture_output=True,
            )
            return result.returncode == 0
        except (AgentNotFoundError, AgentTimeoutError, OSError) as exc:
            logger.warning(f"{self.name} health check failed: {exc}")
            return False

    # Execution ----------------------------------------------------------

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        context = context or {}
        test_name = context.get("test_name", "unnamed-test")
        cwd = self._resolve_working_directory(context.get("cwd"))

        session_id = self._generate_session_id()
        start_time = datetime.now()

        command = self._build_command(skill, query)
        env = self._prepare_environment()
        self._log_invocation(command, cwd)

        try:
            result = await self._run_subprocess(
                command, cwd=cwd, env=env,
                timeout=self.config.timeout, capture_output=True,
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
            raise AgentNotFoundError(self.name, self._install_hint())
        except subprocess.SubprocessError as exc:
            raise SkillAgentAdapterError(
                f"Subprocess execution failed: {exc}",
                adapter_name=self.name,
                recoverable=False,
            )

    # Subprocess ---------------------------------------------------------

    async def _run_subprocess(
        self,
        command: List[str],
        timeout: float,
        capture_output: bool = True,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Execute subprocess with async timeout."""
        loop = asyncio.get_running_loop()

        def _sync_run() -> subprocess.CompletedProcess:
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
                timeout=timeout + 5.0,  # buffer for executor overhead
            )
        except subprocess.TimeoutExpired:
            raise asyncio.TimeoutError(f"Command timed out after {timeout}s")

    # Environment --------------------------------------------------------

    def _prepare_environment(self) -> Dict[str, str]:
        """Build a sanitised environment dict for subprocess execution.

        1. Copies the current environment.
        2. Removes variables whose names contain sensitive patterns.
        3. Applies configured ``env`` overrides (intentional, so kept).
        """
        env = os.environ.copy()

        filtered_keys = [
            key for key in env
            if any(p in key.upper() for p in _SENSITIVE_ENV_PATTERNS)
        ]
        for key in filtered_keys:
            logger.debug(f"Filtered sensitive env var from {self.name} execution: {key}")
            del env[key]

        if self.config.env:
            env.update(self.config.env)

        return env

    # Working directory --------------------------------------------------

    def _resolve_working_directory(self, override: Optional[str]) -> str:
        """Priority: context override > config.cwd > cwd."""
        if override:
            return os.path.abspath(os.path.expanduser(override))
        if self.config.cwd:
            return self.config.cwd
        return os.getcwd()

    # Session id ---------------------------------------------------------

    @staticmethod
    def _generate_session_id() -> str:
        """8-char hex string for trace identification."""
        return uuid.uuid4().hex[:_SESSION_ID_LENGTH]

    # Output truncation --------------------------------------------------

    @staticmethod
    def _truncate_output(output: str) -> str:
        if len(output) > _MAX_OUTPUT_SIZE:
            return output[:_MAX_OUTPUT_SIZE] + "\n... [truncated]"
        return output

    # Audit logging ------------------------------------------------------

    def _log_invocation(self, command: List[str], cwd: str) -> None:
        env_hash = str(hash(frozenset(os.environ.keys())))[:8]
        invocation = CLIInvocation(
            adapter_name=self.name,
            command=tuple(command[:3]),
            working_directory=cwd,
            timestamp=datetime.now(),
            timeout=self.config.timeout,
            environment_hash=env_hash,
        )
        self._invocation_log.append(invocation)
        logger.debug(
            f"{self.name} invocation: {invocation.command[0]}... "
            f"in {cwd} (timeout={self.config.timeout}s)"
        )

    def get_invocation_log(self) -> List[CLIInvocation]:
        """Return invocation log for debugging / audit."""
        return self._invocation_log.copy()

    # Skill formatting ---------------------------------------------------

    def _format_skill_context(self, skill: Skill) -> str:
        """Format skill as system-prompt text for injection.

        Subclasses may override for agent-specific phrasing.
        """
        return (
            f"You have the following skill available:\n\n"
            f"{'━' * 80}\n"
            f"SKILL: {skill.metadata.name}\n"
            f"{'━' * 80}\n\n"
            f"{skill.metadata.description}\n\n"
            f"## Instructions\n\n"
            f"{skill.instructions}\n\n"
            f"{'━' * 80}\n\n"
            f"Follow the skill instructions above when responding "
            f"to the user's request.\n"
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

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
        """Parse execution output into a structured trace.

        Tries JSON first, then JSONL (one JSON per line), then plain text.
        """
        errors: List[str] = []
        if returncode != 0:
            errors.append(f"Process exited with code {returncode}")
            if stderr:
                errors.append(stderr[:1000])

        # 1. Try single-JSON parse
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

        # 2. Try JSONL parse
        json_lines: List[Dict[str, Any]] = []
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
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

        # 3. Plain text fallback
        logger.debug("JSON/JSONL parsing failed, using text fallback")
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
        """Parse a single JSON object into a trace."""
        events: List[TraceEvent] = []
        tool_calls: List[str] = []
        files_created: List[str] = []
        files_modified: List[str] = []
        commands_ran: List[str] = []
        final_output = ""

        messages = data.get("messages", data.get("steps", data.get("actions", [])))
        if not messages and isinstance(data, dict):
            messages = [data]

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            self._process_message(
                msg, events, tool_calls,
                files_created, files_modified, commands_ran,
            )
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    final_output = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            final_output = block.get("text", "")

        usage = data.get("usage", {})
        total_input = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        total_output = usage.get("output_tokens", usage.get("completion_tokens", 0))

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
            total_input_tokens=total_input,
            total_output_tokens=total_output,
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
        """Parse JSONL (one JSON per line) into a trace."""
        events: List[TraceEvent] = []
        tool_calls: List[str] = []
        files_created: List[str] = []
        files_modified: List[str] = []
        commands_ran: List[str] = []
        total_input = 0
        total_output = 0
        final_output = ""

        for data in events_data:
            msg_type = data.get("type", "")

            if msg_type == "result":
                final_output = data.get("result", data.get("output", ""))
                usage = data.get("usage", {})
                total_input = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                total_output = usage.get("output_tokens", usage.get("completion_tokens", 0))

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
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            final_output=final_output,
            errors=errors,
        )

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
        """Plain-text fallback: treat stdout as final output."""
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

    # ------------------------------------------------------------------
    # Message / tool-call processing
    # ------------------------------------------------------------------

    def _process_message(
        self,
        msg: Dict[str, Any],
        events: List[TraceEvent],
        tool_calls: List[str],
        files_created: List[str],
        files_modified: List[str],
        commands_ran: List[str],
    ) -> None:
        """Extract tool calls from a single message dict."""
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
                        tool_name, tool_input,
                        files_created, files_modified, commands_ran,
                    )

    def _track_operations(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        files_created: List[str],
        files_modified: List[str],
        commands_ran: List[str],
    ) -> None:
        """Classify a tool call as file-create, file-modify, or command."""
        tool_lower = tool_name.lower()

        if tool_lower in self._file_creation_tools():
            path = tool_input.get("file_path", tool_input.get("path", ""))
            if path:
                files_created.append(path)

        elif tool_lower in self._file_modification_tools():
            path = tool_input.get("file_path", tool_input.get("path", ""))
            if path:
                files_modified.append(path)

        elif tool_lower in self._command_execution_tools():
            cmd = tool_input.get("command", tool_input.get("cmd", ""))
            if cmd:
                commands_ran.append(cmd)
                self._extract_shell_file_ops(cmd, files_created)

    @staticmethod
    def _extract_shell_file_ops(command: str, files_created: List[str]) -> None:
        """Infer file creation from common shell patterns."""
        match = re.search(r'\btouch\s+([^\s;|&]+)', command)
        if match:
            files_created.append(match.group(1))

        match = re.search(r'\bmkdir\s+(?:-p\s+)?([^\s;|&]+)', command)
        if match:
            files_created.append(match.group(1))

        # stdout redirect (> or >>) but not fd redirect (2>)
        match = re.search(r'(?:^|[^0-9])>{1,2}\s*([^\s;|&]+)', command)
        if match:
            files_created.append(match.group(1))
