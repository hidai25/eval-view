"""Claude Code Agent Teams adapter for skill testing.

Tests multi-agent team workflows that run inside Claude Code. Agent Teams
is a Claude Code feature where agents coordinate via TeammateTool and
SendMessage — this adapter runs Claude Code with team-aware configuration
and captures inter-agent delegation in the trace.

This extends the same CLI approach as ClaudeCodeAdapter (claude --print)
but adds trace analysis for team coordination patterns: which agents were
delegated to, how many handoffs occurred, and whether the final output
synthesized specialist responses correctly.
"""

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
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

# Tool names that indicate inter-agent delegation in Claude Code Agent Teams
TEAM_DELEGATION_TOOLS = frozenset({
    "SendMessage",
    "TeammateTool",
})


class ClaudeAgentTeamsAdapter(SkillAgentAdapter):
    """Adapter for testing Claude Code Agent Teams workflows.

    Agent Teams is a Claude Code feature (not a standalone SDK). Teams are
    configured within Claude Code and agents coordinate via TeammateTool /
    SendMessage. This adapter:

    1. Invokes Claude Code CLI with --print and stream-json output
    2. Parses the structured trace for tool calls including team delegations
    3. Tracks which specialist agents were invoked via SendMessage/TeammateTool
    4. Reports delegation patterns alongside standard tool/file/command traces

    Usage in test YAML:
        agent:
          type: claude-agent-teams
          timeout: 120
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._claude_path = self._find_claude_binary()

    @property
    def name(self) -> str:
        return "claude-agent-teams"

    def _find_claude_binary(self) -> str:
        """Find the claude CLI binary.

        Returns:
            Path to the claude binary

        Raises:
            AgentNotFoundError: If claude is not installed
        """
        claude_path = shutil.which("claude")
        if claude_path:
            return claude_path

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
            install_hint=(
                "Install Claude Code: npm install -g @anthropic-ai/claude-code\n"
                "Agent Teams requires Claude Code with team configuration."
            ),
        )

    async def health_check(self) -> bool:
        """Check if Claude Code CLI is available."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._claude_path,
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
        """Execute a skill test through Claude Code with Agent Teams.

        Runs Claude Code CLI and parses the stream-json output for both
        standard tool calls and team delegation events (SendMessage,
        TeammateTool).

        Args:
            skill: The loaded skill to test
            query: User query to send to the agent team
            context: Optional execution context (test_name, cwd override)

        Returns:
            SkillAgentTrace with tool calls, delegations, and outputs

        Raises:
            AgentNotFoundError: If claude CLI is not found
            AgentTimeoutError: If execution exceeds timeout
            SkillAgentAdapterError: For other execution errors
        """
        context = context or {}
        test_name = context.get("test_name", "unnamed-test")
        cwd = context.get("cwd") or self.config.cwd or os.getcwd()

        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        cmd = self._build_command(skill, query)
        logger.debug(f"Executing: {' '.join(cmd[:5])}...")

        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)

        try:
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
            returncode = process.returncode or 0
            end_time = datetime.now()
            self._last_raw_output = stdout + stderr

            trace = self._parse_output(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                session_id=session_id,
                skill_name=skill.metadata.name,
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
            )
            return trace

        except FileNotFoundError:
            raise AgentNotFoundError(
                self.name,
                "Install Claude Code: npm install -g @anthropic-ai/claude-code",
            )
        except (AgentTimeoutError, AgentNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Claude Agent Teams execution failed: {type(e).__name__}: {e}")
            raise SkillAgentAdapterError(
                f"Execution failed: {type(e).__name__}: {e}",
                adapter_name=self.name,
                recoverable=False,
            )

    def _build_command(self, skill: Skill, query: str) -> List[str]:
        """Build claude CLI command with skill and team context.

        Args:
            skill: Skill to inject as system prompt
            query: User query

        Returns:
            Command list for subprocess
        """
        skill_prompt = f"""You have the following skill loaded:

# Skill: {skill.metadata.name}

{skill.metadata.description}

## Instructions

{skill.instructions}

---

Follow the skill instructions above when responding to user queries.
Use your available team agents (via SendMessage/TeammateTool) to delegate
tasks to specialists when appropriate.
"""

        cmd = [
            self._claude_path,
            "--print",
            "-p",
            query,
            "--append-system-prompt",
            skill_prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]

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
        """Parse claude CLI stream-json output into a structured trace.

        Extracts tool calls (including team delegations), file operations,
        commands, and token usage from the JSON stream.

        Args:
            stdout: Standard output from CLI (stream-json lines)
            stderr: Standard error from CLI
            returncode: Process exit code
            session_id: Unique session identifier
            skill_name: Name of the skill under test
            test_name: Name of the test case
            start_time: Execution start time
            end_time: Execution end time

        Returns:
            SkillAgentTrace with parsed events and metadata
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

        if returncode != 0:
            errors.append(f"Process exited with code {returncode}")
            if stderr.strip():
                errors.append(stderr[:1000])

        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                msg_type = data.get("type", "")

                if msg_type == "result":
                    final_output = data.get("result", "")
                    usage = data.get("usage", {})
                    total_input_tokens = usage.get("input_tokens", 0)
                    total_output_tokens = usage.get("output_tokens", 0)

                elif msg_type == "assistant":
                    message = data.get("message", {})
                    content = message.get("content", [])
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "tool_use":
                            tool_name = item.get("name", "")
                            tool_input = item.get("input", {})
                            if tool_name:
                                tool_calls.append(tool_name)
                                self._track_file_operations(
                                    tool_name, tool_input,
                                    files_created, files_modified, commands_ran
                                )
                                events.append(TraceEvent(
                                    type=TraceEventType.TOOL_CALL,
                                    tool_name=tool_name,
                                    tool_input=tool_input,
                                ))

                elif msg_type == "user":
                    tool_result = data.get("tool_use_result", {})
                    if tool_result and isinstance(tool_result, dict):
                        result_type = tool_result.get("type", "")
                        file_path = tool_result.get("filePath", "")
                        if result_type == "create" and file_path:
                            if file_path not in files_created:
                                files_created.append(file_path)
                        elif result_type in ("edit", "modify") and file_path:
                            if file_path not in files_modified:
                                files_modified.append(file_path)

            except json.JSONDecodeError:
                logger.debug(f"Could not parse JSON line: {line[:100]}")

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

    def _track_file_operations(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        files_created: List[str],
        files_modified: List[str],
        commands_ran: List[str],
    ) -> None:
        """Track file and command side effects from tool calls.

        Also detects team delegation tools (SendMessage, TeammateTool)
        and logs the delegation target for trace visibility.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            files_created: Accumulator for created file paths
            files_modified: Accumulator for modified file paths
            commands_ran: Accumulator for executed commands
        """
        tool_lower = tool_name.lower()

        if tool_lower == "write":
            file_path = tool_input.get("file_path") or tool_input.get("path", "")
            if file_path and file_path not in files_created:
                files_created.append(file_path)

        elif tool_lower == "edit":
            file_path = tool_input.get("file_path") or tool_input.get("path", "")
            if file_path and file_path not in files_modified:
                files_modified.append(file_path)

        elif tool_lower == "bash":
            command = tool_input.get("command", "")
            if command:
                commands_ran.append(command)

        elif tool_name in TEAM_DELEGATION_TOOLS:
            target = tool_input.get("agent") or tool_input.get("to", "unknown")
            logger.debug(f"Team delegation: {tool_name} -> {target}")
