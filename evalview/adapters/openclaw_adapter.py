"""OpenClaw CLI adapter for agent testing.

OpenClaw (https://github.com/openclaw/openclaw) is an open-source autonomous
AI agent that runs locally. This adapter runs OpenClaw via CLI and captures
its tool calls and outputs for evaluation.

Requirements:
    - OpenClaw CLI installed: pip install openclaw
    - OpenClaw configured

Usage in YAML test cases:
    adapter: openclaw

    input:
      query: "Create a React component for a login form"
      context:
        cwd: "./my-project"           # Working directory
        max_turns: 10                 # Max conversation turns
        tools: ["read", "write"]      # Specific tools to enable
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionMetrics,
    ExecutionTrace,
    StepMetrics,
    StepTrace,
    TokenUsage,
    SpanKind,
)
from evalview.core.tracing import Tracer

logger = logging.getLogger(__name__)


class OpenClawAdapter(AgentAdapter):
    """Adapter for the OpenClaw AI agent.

    OpenClaw is a local-first AI agent that uses AgentSkills (SKILL.md)
    to extend its capabilities. This adapter executes OpenClaw via its
    CLI and parses the output for evaluation.

    The adapter uses ``openclaw run`` with JSON output format to capture
    structured data about tool calls, outputs, and metrics.
    """

    def __init__(
        self,
        endpoint: str = "",  # Not used for CLI, kept for registry compatibility
        timeout: float = 300.0,
        cwd: Optional[str] = None,
        tools: Optional[List[str]] = None,
        max_turns: Optional[int] = None,
        skill_path: Optional[str] = None,
        openclaw_path: str = "openclaw",
        **kwargs: Any,
    ):
        """Initialize OpenClaw adapter.

        Args:
            endpoint: Not used (OpenClaw runs locally), kept for registry compatibility
            timeout: Maximum execution time in seconds (default: 300)
            cwd: Working directory for OpenClaw commands
            tools: Specific tools to enable
            max_turns: Maximum conversation turns
            skill_path: Path to a SKILL.md file to load
            openclaw_path: Path to openclaw binary (default: "openclaw")
        """
        self.timeout = timeout
        self.cwd = cwd
        self.tools = tools
        self.max_turns = max_turns
        self.skill_path = skill_path
        self.openclaw_path = openclaw_path
        self._last_raw_output: Optional[str] = None

    @property
    def name(self) -> str:
        return "openclaw"

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Execute a task with OpenClaw and capture the execution trace.

        Args:
            query: The instruction/task for OpenClaw to execute
            context: Optional context with:
                - cwd: Working directory (overrides init setting)
                - tools: Tools to enable
                - max_turns: Maximum conversation turns
                - skill_path: Path to SKILL.md

        Returns:
            ExecutionTrace with tool calls, output, and metrics
        """
        context = context or {}

        cmd = self._build_command(query, context)

        cwd = context.get("cwd", self.cwd)
        if cwd:
            cwd = os.path.abspath(os.path.expanduser(cwd))

        cmd_str = " ".join(cmd)
        logger.info(f"Executing OpenClaw: {cmd_str}")
        if cwd:
            logger.info(f"Working directory: {cwd}")

        start_time = datetime.now()

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=self.timeout,
                    env=self._build_env(context),
                ),
            )

            end_time = datetime.now()
            self._last_raw_output = result.stdout

            if result.stderr:
                logger.debug(f"OpenClaw stderr: {result.stderr}")

            return self._parse_output(
                result.stdout, result.stderr, result.returncode, start_time, end_time
            )

        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            logger.error(f"OpenClaw timed out after {self.timeout}s")
            return self._create_error_trace(
                f"OpenClaw timed out after {self.timeout} seconds",
                start_time,
                end_time,
            )
        except FileNotFoundError:
            end_time = datetime.now()
            logger.error("OpenClaw CLI not found. Is it installed?")
            return self._create_error_trace(
                "OpenClaw CLI not found. Install with: pip install openclaw\n"
                "Or visit: https://github.com/openclaw/openclaw",
                start_time,
                end_time,
            )
        except Exception as e:
            end_time = datetime.now()
            logger.error(f"Error executing OpenClaw: {e}")
            return self._create_error_trace(str(e), start_time, end_time)

    def _build_command(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Build the openclaw CLI command."""
        cmd = [
            self.openclaw_path,
            "run",
            "--prompt", query,
            "--output-format", "json",
            "--headless",
        ]

        # Tools
        tools = context.get("tools", self.tools)
        if tools:
            cmd.extend(["--tools", ",".join(tools)])

        # Max turns
        max_turns = context.get("max_turns", self.max_turns)
        if max_turns:
            cmd.extend(["--max-turns", str(max_turns)])

        # Skill path
        skill_path = context.get("skill_path", self.skill_path)
        if skill_path and os.path.isfile(skill_path):
            cmd.extend(["--skill-path", skill_path])

        return cmd

    def _build_env(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Build environment variables for the subprocess."""
        env = os.environ.copy()
        if "env" in context:
            env.update(context["env"])
        return env

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        start_time: datetime,
        end_time: datetime,
    ) -> ExecutionTrace:
        """Parse OpenClaw output into ExecutionTrace."""
        session_id = f"openclaw-{uuid.uuid4().hex[:8]}"
        total_latency = (end_time - start_time).total_seconds() * 1000

        # Try JSON first (from --output-format json)
        try:
            data = json.loads(stdout)
            return self._parse_json_output(data, session_id, start_time, end_time)
        except json.JSONDecodeError:
            pass

        # Fall back to text parsing
        steps = self._extract_tool_calls_from_text(stdout)
        final_output = self._extract_final_output(stdout)

        if returncode != 0:
            error_msg = stderr or f"OpenClaw exited with code {returncode}"
            if not final_output:
                final_output = error_msg

        tracer = Tracer()
        with tracer.start_span("OpenClaw Execution", SpanKind.AGENT):
            for step in steps:
                tracer.record_tool_call(
                    tool_name=step.tool_name,
                    parameters=step.parameters,
                    result=step.output,
                    error=step.error,
                    duration_ms=step.metrics.latency if step.metrics else 0.0,
                )

        return ExecutionTrace(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=ExecutionMetrics(
                total_cost=self._estimate_cost(steps, stdout),
                total_latency=total_latency,
                total_tokens=self._extract_tokens(stdout),
            ),
            trace_context=tracer.build_trace_context(),
        )

    def _parse_json_output(
        self,
        data: Dict[str, Any],
        session_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> ExecutionTrace:
        """Parse JSON output format from OpenClaw."""
        steps = []

        tool_calls = data.get("tool_calls", data.get("steps", []))
        for i, call in enumerate(tool_calls):
            tool_name = call.get("tool", call.get("name", call.get("tool_name", "unknown")))
            steps.append(
                StepTrace(
                    step_id=f"step-{i}",
                    step_name=call.get("step_name", f"Step {i + 1}"),
                    tool_name=tool_name,
                    parameters=call.get("parameters", call.get("arguments", call.get("input", {}))),
                    output=call.get("output", call.get("result", "")),
                    success=call.get("success", True),
                    error=call.get("error"),
                    metrics=StepMetrics(
                        latency=call.get("latency", 0.0),
                        cost=call.get("cost", 0.0),
                    ),
                )
            )

        final_output = data.get("response", data.get("output", data.get("final_response", "")))
        total_latency = (end_time - start_time).total_seconds() * 1000

        tokens_data = data.get("tokens", data.get("usage", {}))
        total_tokens = None
        if tokens_data:
            if isinstance(tokens_data, dict):
                total_tokens = TokenUsage(
                    input_tokens=int(
                        tokens_data.get("input", tokens_data.get("input_tokens", 0)) or 0
                    ),
                    output_tokens=int(
                        tokens_data.get("output", tokens_data.get("output_tokens", 0)) or 0
                    ),
                    cached_tokens=int(tokens_data.get("cached", 0) or 0),
                )
            elif isinstance(tokens_data, int):
                total_tokens = TokenUsage(output_tokens=tokens_data)

        return ExecutionTrace(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=ExecutionMetrics(
                total_cost=data.get("cost", data.get("total_cost", 0.0)),
                total_latency=total_latency,
                total_tokens=total_tokens,
            ),
        )

    def _extract_tool_calls_from_text(self, output: str) -> List[StepTrace]:
        """Extract tool calls from OpenClaw's text output."""
        steps = []
        step_count = 0

        # OpenClaw tool patterns
        known_tools = {
            "bash", "shell", "read", "write", "edit", "search", "grep", "find",
            "write_file", "create_file", "str_replace_editor", "file_write",
            "file_edit", "patch", "append", "insert", "run", "execute",
            "run_command", "terminal",
        }

        # Look for tool call patterns in output
        tool_pattern = r"(?:Tool|Action|Step)\s*(?:call)?:?\s*(\w+)"
        for match in re.finditer(tool_pattern, output, re.IGNORECASE):
            tool_name = match.group(1).lower()
            if tool_name in known_tools or len(tool_name) > 2:
                step_count += 1

                # Try to extract parameters after the match
                remaining = output[match.end():match.end() + 500]
                params = {}
                param_match = re.search(r'\{.*?\}', remaining, re.DOTALL)
                if param_match:
                    try:
                        params = json.loads(param_match.group())
                    except json.JSONDecodeError:
                        pass

                steps.append(
                    StepTrace(
                        step_id=f"step-{step_count}",
                        step_name=f"Step {step_count}",
                        tool_name=tool_name,
                        parameters=params,
                        output="",
                        success=True,
                        error=None,
                        metrics=StepMetrics(),
                    )
                )

        return steps

    def _extract_final_output(self, output: str) -> str:
        """Extract the final response from OpenClaw output."""
        # Remove ANSI escape codes
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean_output = ansi_escape.sub("", output)

        # Take the last substantive paragraphs as the response
        paragraphs = clean_output.strip().split("\n\n")
        response_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para or para.startswith("$") or para.startswith(">>>"):
                continue
            response_paragraphs.append(para)

        if response_paragraphs:
            return "\n\n".join(response_paragraphs[-3:])

        return clean_output.strip()

    def _estimate_cost(self, steps: List[StepTrace], output: str) -> float:
        """Rough cost estimate based on output length and tool calls."""
        total_chars = len(output)
        estimated_tokens = total_chars / 4
        estimated_cost = (estimated_tokens / 1000) * 0.003
        estimated_cost += len(steps) * 0.001
        return round(estimated_cost, 6)

    def _extract_tokens(self, output: str) -> Optional[TokenUsage]:
        """Try to extract token usage from output."""
        patterns = [
            r"tokens?:\s*(\d+)",
            r"(\d+)\s*tokens?",
            r"input:\s*(\d+).*output:\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    return TokenUsage(
                        input_tokens=int(match.group(1)),
                        output_tokens=int(match.group(2)),
                    )
                else:
                    return TokenUsage(output_tokens=int(match.group(1)))
        return None

    def _create_error_trace(
        self, error_msg: str, start_time: datetime, end_time: datetime
    ) -> ExecutionTrace:
        """Create an ExecutionTrace for error cases."""
        return ExecutionTrace(
            session_id=f"openclaw-error-{uuid.uuid4().hex[:8]}",
            start_time=start_time,
            end_time=end_time,
            steps=[
                StepTrace(
                    step_id="error",
                    step_name="Error",
                    tool_name="error",
                    parameters={},
                    output=error_msg,
                    success=False,
                    error=error_msg,
                    metrics=StepMetrics(),
                )
            ],
            final_output=error_msg,
            metrics=ExecutionMetrics(
                total_cost=0.0,
                total_latency=(end_time - start_time).total_seconds() * 1000,
                total_tokens=None,
            ),
        )

    async def health_check(self) -> bool:
        """Check if OpenClaw CLI is available."""
        try:
            result = subprocess.run(
                [self.openclaw_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info(f"OpenClaw version: {result.stdout.strip()}")
                return True
            return False
        except Exception as e:
            logger.warning(f"OpenClaw health check failed: {e}")
            return False
