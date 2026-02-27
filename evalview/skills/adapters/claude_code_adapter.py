"""Claude Code adapter for skill testing.

Executes skills through the Claude Code CLI and captures structured traces.
Uses `claude --print -p <query>` for non-interactive execution.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

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
        self._last_raw_output: str = ""

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
        # Remove Claude Code session markers + any inherited auth token that may
        # be a short-lived session value (not the user's real API key).
        # The inner claude will fall back to ~/.claude.json credentials instead.
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        env.pop("ANTHROPIC_API_KEY", None)
        if self.config.env:
            env.update(self.config.env)

        try:
            # Use temp files + run_in_executor (thread pool) to avoid two issues:
            # 1. claude --print hangs when stdout is a pipe (asyncio creates pipes)
            # 2. asyncio subprocess behaves differently from shell subprocess for claude
            # Running in a thread via subprocess.run matches what works in a shell.
            #
            # mkstemp() is used instead of mktemp() to avoid the TOCTOU race:
            # the file descriptor is opened atomically and then closed so the
            # subprocess can write to it by path.
            stdout_fd, stdout_path = tempfile.mkstemp(suffix=".stdout")
            stderr_fd, stderr_path = tempfile.mkstemp(suffix=".stderr")
            os.close(stdout_fd)
            os.close(stderr_fd)

            timeout = self.config.timeout

            def _run_blocking() -> int:
                with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.DEVNULL,
                        stdout=out_f,
                        stderr=err_f,
                        cwd=cwd,
                        env=env,
                        # Detach from Claude Code's process group — otherwise
                        # claude detects it's a child of Claude Code and fails.
                        start_new_session=True,
                    )
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        raise
                return proc.returncode if proc.returncode is not None else 0

            try:
                loop = asyncio.get_running_loop()
                try:
                    returncode = await asyncio.wait_for(
                        loop.run_in_executor(None, _run_blocking),
                        timeout=timeout + 5,
                    )
                except asyncio.TimeoutError:
                    raise AgentTimeoutError(self.name, timeout)
                except subprocess.TimeoutExpired:
                    raise AgentTimeoutError(self.name, timeout)

                with open(stdout_path, "r", errors="replace") as f:
                    stdout = f.read()
                with open(stderr_path, "r", errors="replace") as f:
                    stderr = f.read()
            finally:
                for p in (stdout_path, stderr_path):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

            end_time = datetime.now()
            self._last_raw_output = stdout + stderr

            # Detect missing tool errors and surface them clearly
            self._check_for_missing_tools(stdout, stderr, skill)

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
        # Build skill injection prompt.
        # IMPORTANT: Do NOT say "you have the skill loaded" — claude will try to
        # invoke it via the built-in Skill tool, which fails if the skill isn't
        # installed, producing a misleading "Invalid API key" error.
        # Instead, inject the skill content directly as behavioural instructions.
        skill_prompt = f"""--- SKILL: {skill.metadata.name} ---

{skill.metadata.description}

{skill.instructions}

--- END SKILL ---

You MUST follow the instructions above when responding. Do NOT use the Skill tool \
to invoke this skill — the instructions are already loaded here as your guidelines."""

        # Default to haiku for speed in skill tests; override via agent.model in YAML
        model = self.config.model or "claude-haiku-4-5-20251001"

        cmd = [
            self.claude_path,
            "--print",
            "-p",
            query,
            "--model",
            model,
            "--append-system-prompt",
            skill_prompt,
            "--output-format",
            "stream-json",  # Stream format includes tool calls
            "--verbose",  # Required for stream-json with --print
            "--dangerously-skip-permissions",  # Required for non-interactive testing
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

        # Parse stream-json format (one JSON per line)
        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                msg_type = data.get("type", "")

                # Extract final result
                if msg_type == "result":
                    final_output = data.get("result", "")
                    usage = data.get("usage", {})
                    total_input_tokens = usage.get("input_tokens", 0)
                    total_output_tokens = usage.get("output_tokens", 0)

                # Extract tool calls from assistant messages
                elif msg_type == "assistant":
                    message = data.get("message", {})
                    content = message.get("content", [])
                    for item in content:
                        # Skip string items (plain text responses)
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

                # Extract file operations from tool results
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
                # Skip malformed lines
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

    def _check_for_missing_tools(
        self, stdout: str, stderr: str, skill: Skill
    ) -> None:
        """Log clear warnings when skill requires tools that aren't installed.

        Detects common patterns like mcporter not found, openclaw missing, etc.
        Logs actionable hints rather than raising — the test will still run and
        fail with a meaningful message, but the user also sees a clear hint.
        """
        combined = (stdout + stderr).lower()

        hints = []

        if "mcporter" in combined and (
            "command not found" in combined
            or "no such file" in combined
            or "not found" in combined
        ):
            hints.append(
                "mcporter is not installed. This is an OpenClaw skill.\n"
                "  Install OpenClaw: pip install openclaw\n"
                "  Then configure Exa MCP: mcporter config add exa https://mcp.exa.ai/mcp"
            )

        if "openclaw" in combined and "command not found" in combined:
            hints.append(
                "openclaw CLI is not installed.\n"
                "  Install: pip install openclaw"
            )

        if "invalid api key" in combined or "invalid_api_key" in combined:
            # Could be an MCP server API key issue
            skill_name = skill.metadata.name if skill and skill.metadata else "this skill"
            hints.append(
                f"An external API key required by '{skill_name}' is missing or invalid.\n"
                "  Check the skill's SKILL.md for required API keys and MCP server setup."
            )

        for hint in hints:
            logger.warning(f"\n⚠️  Skill testing hint:\n  {hint}")

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
