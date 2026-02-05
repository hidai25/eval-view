"""Claude Agent SDK adapter for skill testing.

Executes skills through multi-agent teams built with the Claude Agent SDK.
Captures structured traces of inter-agent coordination, tool calls, and outputs.

Supports two execution modes:
    1. Script mode: Run a user-provided Python script that uses the Agent SDK
    2. Default mode: Generate a single-agent runner with the skill as system prompt

The adapter parses JSONL trace lines emitted by the Agent SDK for structured
observability into tool calls, file operations, and agent-to-agent messages.
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

# Trace event types emitted by the Agent SDK that map to inter-agent messaging
_AGENT_MESSAGE_TOOLS = frozenset({"SendMessage", "TeammateTool", "delegate", "handoff"})


class ClaudeAgentSDKAdapter(SkillAgentAdapter):
    """Adapter for executing skills through Claude Agent SDK (Agent Teams).

    Tests multi-agent workflows by running a Python script that uses the
    Claude Agent SDK. The script is expected to print structured JSONL to
    stdout for trace capture.

    Each JSONL line should be a JSON object with at minimum a "type" field.
    Recognized types: tool_call, llm_call, file_create, file_modify,
    command_run, error.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._python_path = self._find_python()

    @property
    def name(self) -> str:
        return "claude-agent-sdk"

    def _find_python(self) -> str:
        """Locate python3 binary.

        Returns:
            Path to python3

        Raises:
            AgentNotFoundError: If python3 is not available
        """
        python_path = shutil.which("python3") or shutil.which("python")
        if python_path:
            return python_path

        raise AgentNotFoundError(
            adapter_name=self.name,
            install_hint="python3 is required. Install Python 3.9+ from https://python.org",
        )

    async def health_check(self) -> bool:
        """Check if the Agent SDK is importable."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._python_path,
                "-c",
                "import claude_agent_sdk; print(claude_agent_sdk.__version__)",
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
            logger.warning(f"Claude Agent SDK health check failed: {e}")
            return False

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute a skill test through a Claude Agent SDK team.

        Args:
            skill: The loaded skill to test
            query: User query to send to the agent team
            context: Optional execution context (test_name, cwd override)

        Returns:
            SkillAgentTrace with execution events and inter-agent messages

        Raises:
            AgentNotFoundError: If python3 is not found
            AgentTimeoutError: If execution exceeds timeout
            SkillAgentAdapterError: For other execution errors
        """
        context = context or {}
        test_name = context.get("test_name", "unnamed-test")
        cwd = context.get("cwd") or self.config.cwd or os.getcwd()

        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        script_path = self.config.script_path
        generated_script = False
        if not script_path:
            script_path = self._write_default_runner(cwd, skill, query)
            generated_script = True

        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)
        env["EVALVIEW_QUERY"] = query
        env["EVALVIEW_MODEL"] = env.get("EVALVIEW_MODEL", "claude-opus-4-6")
        env["EVALVIEW_SKILL_NAME"] = skill.metadata.name
        env["EVALVIEW_SKILL_INSTRUCTIONS"] = skill.instructions or ""

        try:
            process = await asyncio.create_subprocess_exec(
                self._python_path,
                script_path,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
                "python3 is required. Install Python 3.9+ from https://python.org",
            )
        except (AgentTimeoutError, AgentNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Claude Agent SDK execution failed: {type(e).__name__}: {e}")
            raise SkillAgentAdapterError(
                f"Execution failed: {type(e).__name__}: {e}",
                adapter_name=self.name,
                recoverable=False,
            )
        finally:
            if generated_script and os.path.exists(script_path):
                try:
                    os.unlink(script_path)
                except OSError:
                    logger.debug(f"Could not clean up generated script: {script_path}")

    def _write_default_runner(self, cwd: str, skill: Skill, query: str) -> str:
        """Generate a default runner script for agent teams without a custom script.

        Creates a minimal script that instantiates a single agent with the
        skill as system prompt and runs the query. Users should provide their
        own script_path for real multi-agent team testing.

        The generated script reads its model and skill from environment
        variables so the adapter controls configuration centrally.

        Args:
            cwd: Working directory to write the script
            skill: The skill to inject
            query: The user query

        Returns:
            Path to the generated script
        """
        import tempfile

        fd, script_path = tempfile.mkstemp(
            prefix="evalview_agent_sdk_", suffix=".py", dir=cwd
        )
        os.close(fd)

        runner_code = '''\
"""Auto-generated EvalView runner for Claude Agent SDK.

Override this by setting script_path in your test YAML agent config.
"""
import json
import os
import sys

query = os.environ.get("EVALVIEW_QUERY", "")
model = os.environ.get("EVALVIEW_MODEL", "claude-opus-4-6")
skill_name = os.environ.get("EVALVIEW_SKILL_NAME", "")
skill_instructions = os.environ.get("EVALVIEW_SKILL_INSTRUCTIONS", "")

system_prompt = f"Skill: {skill_name}" + chr(10) + chr(10) + skill_instructions

try:
    from claude_agent_sdk import Agent

    agent = Agent(model=model, system_prompt=system_prompt)
    result = agent.run(query)
    print(result)

except ImportError:
    print(
        json.dumps({"type": "error", "message": "claude-agent-sdk not installed. "
                    "Install with: pip install claude-agent-sdk"}),
        file=sys.stderr,
    )
    sys.exit(1)
'''
        with open(script_path, "w") as f:
            f.write(runner_code)
        return script_path

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
        """Parse Agent SDK output into a structured trace.

        Handles JSONL trace lines and plain text output. JSONL lines starting
        with '{"type":' are parsed as structured events; remaining text is
        collected as final_output.

        Args:
            stdout: Standard output from the script
            stderr: Standard error from the script
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
        output_lines: List[str] = []
        errors: List[str] = []

        if returncode != 0:
            errors.append(f"Process exited with code {returncode}")
            if stderr.strip():
                errors.append(stderr[:1000])

        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Try to parse structured JSONL trace events
            if stripped.startswith('{"type":'):
                try:
                    evt = json.loads(stripped)
                    event = self._parse_trace_event(evt)
                    if event:
                        events.append(event)
                        if event.tool_name:
                            tool_calls.append(event.tool_name)
                            self._track_side_effects(
                                event, files_created, files_modified, commands_ran
                            )
                        if event.input_tokens:
                            total_input_tokens += event.input_tokens
                        if event.output_tokens:
                            total_output_tokens += event.output_tokens
                    continue
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Could not parse JSONL trace line: {stripped[:100]}: {e}")

            # Non-JSONL lines are part of the final output
            output_lines.append(line)

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
            final_output="\n".join(output_lines),
            errors=errors,
        )

    def _parse_trace_event(self, evt: Dict[str, Any]) -> Optional[TraceEvent]:
        """Parse a single JSONL event dict into a TraceEvent.

        Args:
            evt: Parsed JSON object with at least a "type" field

        Returns:
            TraceEvent if parseable, None otherwise
        """
        raw_type = evt.get("type", "")
        try:
            event_type = TraceEventType(raw_type)
        except ValueError:
            logger.debug(f"Unknown trace event type: {raw_type}")
            return None

        return TraceEvent(
            type=event_type,
            tool_name=evt.get("tool_name"),
            tool_input=evt.get("tool_input"),
            tool_output=evt.get("tool_output"),
            tool_success=evt.get("tool_success"),
            tool_error=evt.get("tool_error"),
            file_path=evt.get("file_path"),
            file_content=evt.get("file_content"),
            command=evt.get("command"),
            command_output=evt.get("command_output"),
            command_exit_code=evt.get("exit_code"),
            model=evt.get("model"),
            input_tokens=evt.get("input_tokens"),
            output_tokens=evt.get("output_tokens"),
        )

    def _track_side_effects(
        self,
        event: TraceEvent,
        files_created: List[str],
        files_modified: List[str],
        commands_ran: List[str],
    ) -> None:
        """Track file and command side effects from a trace event.

        Args:
            event: The trace event to inspect
            files_created: Accumulator for created file paths
            files_modified: Accumulator for modified file paths
            commands_ran: Accumulator for executed commands
        """
        if event.type == TraceEventType.FILE_CREATE and event.file_path:
            if event.file_path not in files_created:
                files_created.append(event.file_path)
        elif event.type == TraceEventType.FILE_MODIFY and event.file_path:
            if event.file_path not in files_modified:
                files_modified.append(event.file_path)
        elif event.type == TraceEventType.COMMAND_RUN and event.command:
            commands_ran.append(event.command)
        elif event.type == TraceEventType.TOOL_CALL:
            tool_input = event.tool_input or {}
            tool_name = (event.tool_name or "").lower()

            if tool_name == "write":
                path = tool_input.get("file_path") or tool_input.get("path", "")
                if path and path not in files_created:
                    files_created.append(path)
            elif tool_name == "edit":
                path = tool_input.get("file_path") or tool_input.get("path", "")
                if path and path not in files_modified:
                    files_modified.append(path)
            elif tool_name == "bash":
                cmd = tool_input.get("command", "")
                if cmd:
                    commands_ran.append(cmd)
