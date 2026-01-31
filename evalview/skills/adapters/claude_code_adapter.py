"""Claude Code adapter for skill testing.

Executes skills through the Claude Code CLI and captures structured traces.
Uses `claude --print -p <query>` for non-interactive execution.
"""

import asyncio
import json
import os
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import shutil

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


class ClaudeCodeAdapter(SkillAgentAdapter):
    """Adapter for executing skills through Claude Code CLI.

    Uses the claude CLI with --print flag for non-interactive execution.
    Parses JSON output (--output-format=json) for structured traces.

    Tool calls are extracted from the JSON output. File operations and
    commands are inferred from specific tool calls (Write, Edit, Bash).
    """

    def __init__(self, config: AgentConfig):
        """Initialize Claude Code adapter.

        Args:
            config: Agent configuration
        """
        super().__init__(config)
        self.claude_path = self._find_claude_binary()

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "claude-code"

    def _find_claude_binary(self) -> str:
        """Find the claude CLI binary path.

        Returns:
            Path to claude binary

        Raises:
            AgentNotFoundError: If claude not found
        """
        # Check if claude is in PATH
        claude_path = shutil.which("claude")
        if claude_path:
            return claude_path

        # Check common installation locations
        common_paths = [
            os.path.expanduser("~/.npm-global/bin/claude"),
            os.path.expanduser("~/.local/bin/claude"),
            "/usr/local/bin/claude",
        ]

        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        raise AgentNotFoundError(
            adapter_name=self.name,
            install_hint="Install Claude Code: npm install -g @anthropic-ai/claude-code",
        )

    async def health_check(self) -> bool:
        """Check if Claude Code CLI is available.

        Returns:
            True if claude CLI is accessible
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.claude_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                await asyncio.wait_for(process.communicate(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False
            return process.returncode == 0
        except Exception as e:
            logger.warning(f"Claude Code health check failed: {e}")
            return False

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute skill test with Claude Code CLI.

        Args:
            skill: The loaded skill to test
            query: User query to send to the agent
            context: Optional execution context

        Returns:
            SkillAgentTrace with execution details

        Raises:
            AgentNotFoundError: If claude CLI not found
            AgentTimeoutError: If execution times out
            SkillAgentAdapterError: For other errors
        """
        context = context or {}
        test_name = context.get("test_name", "unnamed-test")
        cwd = context.get("cwd") or self.config.cwd or os.getcwd()

        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        # Build command
        cmd = self._build_command(skill, query)
        logger.debug(f"Executing: {' '.join(cmd[:5])}...")

        # Prepare environment
        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)

        try:
            # Run claude CLI using asyncio subprocess (more reliable in threads)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise AgentTimeoutError(self.name, self.config.timeout)

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            returncode = process.returncode

            end_time = datetime.now()
            self._last_raw_output = stdout + stderr

            # Parse output
            trace = self._parse_output(
                stdout,
                stderr,
                returncode,
                session_id=session_id,
                skill_name=skill.metadata.name,
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
            )

            return trace

        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError: {e}")
            raise AgentNotFoundError(
                self.name,
                "Install Claude Code: npm install -g @anthropic-ai/claude-code",
            )

        except AgentTimeoutError:
            raise

        except Exception as e:
            logger.error(f"Exception type: {type(e).__name__}, message: {e}")
            raise SkillAgentAdapterError(
                f"Execution failed: {type(e).__name__}: {e}",
                adapter_name=self.name,
                recoverable=False,
            )

    def _build_command(self, skill: Skill, query: str) -> List[str]:
        """Build claude CLI command with skill injection.

        Args:
            skill: Skill to inject as system prompt
            query: User query

        Returns:
            Command list for subprocess
        """
        # Build skill injection prompt
        skill_prompt = f"""You have the following skill loaded:

# Skill: {skill.metadata.name}

{skill.metadata.description}

## Instructions

{skill.instructions}

---

Follow the skill instructions above when responding to user queries.
"""

        cmd = [
            self.claude_path,
            "--print",
            "-p",
            query,
            "--append-system-prompt",
            skill_prompt,
            "--output-format",
            "json",
        ]

        # Add allowed tools if specified
        if self.config.tools:
            cmd.extend(["--allowedTools", ",".join(self.config.tools)])

        return cmd

    def _parse_output(
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
        """Parse claude CLI output into structured trace.

        Handles both JSON and text output formats.

        Args:
            stdout: Standard output from CLI
            stderr: Standard error from CLI
            returncode: Process return code
            session_id: Unique session identifier
            skill_name: Name of skill being tested
            test_name: Name of test being run
            start_time: Execution start time
            end_time: Execution end time

        Returns:
            SkillAgentTrace with parsed data
        """
        events: List[TraceEvent] = []
        tool_calls: List[str] = []
        files_created: List[str] = []
        files_modified: List[str] = []
        commands_ran: List[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        final_output = ""
        errors: List[str] = []

        # Check for errors
        if returncode != 0:
            errors.append(f"Process exited with code {returncode}")
            if stderr:
                errors.append(stderr[:1000])

        # Try to parse JSON output
        try:
            data = json.loads(stdout)
            final_output, events, tool_calls, files_created, files_modified, commands_ran, total_input_tokens, total_output_tokens = self._parse_json_output(data)
        except json.JSONDecodeError:
            # Fall back to text output parsing
            final_output = stdout
            logger.debug("Could not parse JSON output, using raw text")

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

    def _parse_json_output(
        self, data: Dict[str, Any]
    ) -> tuple:
        """Parse JSON output from claude CLI.

        Claude Code outputs a structured JSON with messages and tool calls.

        Args:
            data: Parsed JSON data

        Returns:
            Tuple of (final_output, events, tool_calls, files_created,
                      files_modified, commands_ran, input_tokens, output_tokens)
        """
        events: List[TraceEvent] = []
        tool_calls: List[str] = []
        files_created: List[str] = []
        files_modified: List[str] = []
        commands_ran: List[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        final_output = ""

        # Extract from messages array if present
        messages = data.get("messages", [])
        if not messages and isinstance(data, dict):
            # Maybe data itself is the message
            messages = [data]

        for msg in messages:
            # Extract text content
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, str):
                    final_output = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                final_output = block.get("text", "")
                            elif block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
                                tool_calls.append(tool_name)
                                tool_input = block.get("input", {})

                                # Create trace event
                                event = TraceEvent(
                                    type=TraceEventType.TOOL_CALL,
                                    tool_name=tool_name,
                                    tool_input=tool_input,
                                )
                                events.append(event)

                                # Track file operations
                                self._track_file_operations(
                                    tool_name,
                                    tool_input,
                                    files_created,
                                    files_modified,
                                    commands_ran,
                                )

            # Extract usage info
            if "usage" in msg:
                usage = msg["usage"]
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)

        # Also check top-level usage
        if "usage" in data:
            usage = data["usage"]
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)

        # Check for result/response field
        if not final_output:
            final_output = data.get("result", "") or data.get("response", "")

        return (
            final_output,
            events,
            tool_calls,
            files_created,
            files_modified,
            commands_ran,
            total_input_tokens,
            total_output_tokens,
        )

    def _track_file_operations(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        files_created: List[str],
        files_modified: List[str],
        commands_ran: List[str],
    ) -> None:
        """Track file operations from tool calls.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            files_created: List to append created files
            files_modified: List to append modified files
            commands_ran: List to append ran commands
        """
        tool_lower = tool_name.lower()

        # Write tool creates files
        if tool_lower == "write":
            file_path = tool_input.get("file_path") or tool_input.get("path", "")
            if file_path:
                files_created.append(file_path)

        # Edit tool modifies files
        elif tool_lower == "edit":
            file_path = tool_input.get("file_path") or tool_input.get("path", "")
            if file_path:
                files_modified.append(file_path)

        # Bash tool runs commands
        elif tool_lower == "bash":
            command = tool_input.get("command", "")
            if command:
                commands_ran.append(command)

                # Check for file operations in commands
                if "touch " in command or "mkdir " in command:
                    # Extract paths (simple heuristic)
                    parts = command.split()
                    for i, part in enumerate(parts):
                        if part in ("touch", "mkdir", ">") and i + 1 < len(parts):
                            files_created.append(parts[i + 1])

        # Read tool (just tracking, no modification)
        elif tool_lower == "read":
            pass  # Just reading, no modification

        # Glob/Grep tools (search operations)
        elif tool_lower in ("glob", "grep"):
            pass  # Search operations, no modification
