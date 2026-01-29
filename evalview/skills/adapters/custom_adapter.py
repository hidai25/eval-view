"""Custom script adapter for skill testing.

Executes skills through a user-provided script, enabling integration
with any agent that can be invoked via command line.
"""

import asyncio
import json
import os
import subprocess
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


class CustomAdapter(SkillAgentAdapter):
    """Adapter for executing skills through custom scripts.

    The custom script receives:
    - SKILL_PATH: Path to the skill file
    - SKILL_NAME: Name of the skill
    - QUERY: The user query to process
    - CWD: Working directory

    The script should output JSON to stdout with:
    - output: Final text output
    - tool_calls: List of tool names called
    - files_created: List of created file paths
    - files_modified: List of modified file paths
    - commands_ran: List of commands executed
    - input_tokens: Optional token count
    - output_tokens: Optional token count
    """

    def __init__(self, config: AgentConfig):
        """Initialize custom adapter.

        Args:
            config: Agent configuration with script_path

        Raises:
            SkillAgentAdapterError: If script_path not configured
        """
        super().__init__(config)

        if not config.script_path:
            raise SkillAgentAdapterError(
                "Custom adapter requires script_path in agent config",
                adapter_name="custom",
            )

        self.script_path = os.path.abspath(os.path.expanduser(config.script_path))

        if not os.path.isfile(self.script_path):
            raise AgentNotFoundError(
                adapter_name="custom",
                install_hint=f"Script not found: {self.script_path}",
            )

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "custom"

    async def health_check(self) -> bool:
        """Check if the custom script is available.

        Returns:
            True if script exists and is executable
        """
        return os.path.isfile(self.script_path) and os.access(
            self.script_path, os.X_OK
        )

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute skill test with custom script.

        Args:
            skill: The loaded skill to test
            query: User query to send to the agent
            context: Optional execution context

        Returns:
            SkillAgentTrace with execution details

        Raises:
            AgentTimeoutError: If execution times out
            SkillAgentAdapterError: For other errors
        """
        context = context or {}
        test_name = context.get("test_name", "unnamed-test")
        cwd = context.get("cwd") or self.config.cwd or os.getcwd()

        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        # Prepare environment
        env = os.environ.copy()
        env["SKILL_PATH"] = skill.file_path or ""
        env["SKILL_NAME"] = skill.metadata.name
        env["QUERY"] = query
        env["CWD"] = cwd
        env["SKILL_INSTRUCTIONS"] = skill.instructions

        if self.config.env:
            env.update(self.config.env)

        # Build command
        cmd = [self.script_path, query]

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    env=env,
                    timeout=self.config.timeout,
                ),
            )

            end_time = datetime.now()
            self._last_raw_output = result.stdout + result.stderr

            # Parse output
            return self._parse_output(
                result.stdout,
                result.stderr,
                result.returncode,
                session_id=session_id,
                skill_name=skill.metadata.name,
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
            )

        except subprocess.TimeoutExpired:
            raise AgentTimeoutError(self.name, self.config.timeout)

        except Exception as e:
            raise SkillAgentAdapterError(
                f"Script execution failed: {e}",
                adapter_name=self.name,
                recoverable=False,
            )

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
        """Parse script output into structured trace.

        Args:
            stdout: Standard output from script
            stderr: Standard error from script
            returncode: Process return code
            session_id: Unique session identifier
            skill_name: Name of skill being tested
            test_name: Name of test being run
            start_time: Execution start time
            end_time: Execution end time

        Returns:
            SkillAgentTrace with parsed data
        """
        errors: List[str] = []

        if returncode != 0:
            errors.append(f"Script exited with code {returncode}")
            if stderr:
                errors.append(stderr[:1000])

        # Try to parse JSON output
        try:
            data = json.loads(stdout)

            return SkillAgentTrace(
                session_id=session_id,
                skill_name=skill_name,
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                events=[],  # Custom scripts don't provide detailed events
                tool_calls=data.get("tool_calls", []),
                files_created=data.get("files_created", []),
                files_modified=data.get("files_modified", []),
                commands_ran=data.get("commands_ran", []),
                total_input_tokens=data.get("input_tokens", 0),
                total_output_tokens=data.get("output_tokens", 0),
                final_output=data.get("output", ""),
                errors=errors,
            )

        except json.JSONDecodeError:
            # Use raw output if not JSON
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
