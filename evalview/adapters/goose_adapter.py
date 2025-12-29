"""Goose CLI adapter for Block's open-source AI agent.

Goose (https://github.com/block/goose) is a local, extensible AI agent
that automates engineering tasks. This adapter runs Goose via CLI and
captures its tool calls and outputs for evaluation.

Requirements:
    - Goose CLI installed: curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
    - Goose configured: goose configure

Usage:
    adapter: goose

    input:
      query: "Run the tests in this project"
      context:
        cwd: "./my-project"           # Working directory for Goose
        extensions: ["developer"]      # Builtin extensions to enable
        provider: "anthropic"          # Optional: override LLM provider
        model: "claude-sonnet-4-20250514"  # Optional: override model
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
)

logger = logging.getLogger(__name__)


class GooseAdapter(AgentAdapter):
    """Adapter for Block's Goose AI agent.

    Goose is a local-first AI agent that runs commands, edits files,
    and automates engineering tasks. This adapter executes Goose via
    its CLI and parses the output for evaluation.

    The adapter uses `goose run` with JSON output format to capture
    structured data about tool calls, outputs, and metrics.
    """

    def __init__(
        self,
        endpoint: str = "",  # Not used for CLI, but required by registry
        timeout: float = 300.0,  # 5 minutes default for complex tasks
        cwd: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        goose_path: str = "goose",  # Path to goose binary
        **kwargs: Any,
    ):
        """Initialize Goose adapter.

        Args:
            endpoint: Not used (Goose runs locally), kept for registry compatibility
            timeout: Maximum execution time in seconds (default: 300)
            cwd: Working directory for Goose commands
            extensions: Builtin extensions to enable (e.g., ["developer", "computercontroller"])
            provider: LLM provider override (e.g., "anthropic", "openai")
            model: Model override (e.g., "claude-sonnet-4-20250514")
            goose_path: Path to goose binary (default: "goose")
        """
        self.timeout = timeout
        self.cwd = cwd
        self.extensions = extensions or []
        self.provider = provider
        self.model = model
        self.goose_path = goose_path
        self._last_raw_output: Optional[str] = None

    @property
    def name(self) -> str:
        return "goose"

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Execute a task with Goose and capture the execution trace.

        Args:
            query: The instruction/task for Goose to execute
            context: Optional context with:
                - cwd: Working directory (overrides init setting)
                - extensions: Builtin extensions to add
                - provider: LLM provider override
                - model: Model override
                - max_turns: Maximum conversation turns

        Returns:
            ExecutionTrace with tool calls, output, and metrics
        """
        context = context or {}

        # Build the goose command
        cmd = self._build_command(query, context)

        # Determine working directory
        cwd = context.get("cwd", self.cwd)
        if cwd:
            cwd = os.path.abspath(os.path.expanduser(cwd))

        # Log the full command for debugging
        cmd_str = ' '.join(cmd)
        logger.info(f"Executing Goose: {cmd_str}")
        print(f"[DEBUG] Goose command: {cmd_str}")  # Visible in console
        if cwd:
            logger.info(f"Working directory: {cwd}")

        start_time = datetime.now()

        try:
            # Run goose as subprocess
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

            # Store raw output for debugging
            self._last_raw_output = result.stdout

            # Log any stderr (warnings, debug info)
            if result.stderr:
                logger.debug(f"Goose stderr: {result.stderr}")

            # Parse the output
            return self._parse_output(
                result.stdout, result.stderr, result.returncode, start_time, end_time
            )

        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            logger.error(f"Goose timed out after {self.timeout}s")
            return self._create_error_trace(
                f"Goose timed out after {self.timeout} seconds",
                start_time,
                end_time,
            )
        except FileNotFoundError:
            end_time = datetime.now()
            logger.error("Goose CLI not found. Is it installed?")
            return self._create_error_trace(
                "Goose CLI not found. Install with: curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash",
                start_time,
                end_time,
            )
        except Exception as e:
            end_time = datetime.now()
            logger.error(f"Error executing Goose: {e}")
            return self._create_error_trace(str(e), start_time, end_time)

    def _build_command(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Build the goose CLI command."""
        cmd = [
            self.goose_path,
            "run",
            "-t",
            query,
            "--no-session",  # Clean runs for testing
        ]

        # Add extensions (deduplicate)
        extensions = list(set(context.get("extensions", []) + self.extensions))
        if extensions:
            cmd.extend(["--with-builtin", ",".join(extensions)])

        # Add provider/model overrides
        provider = context.get("provider", self.provider)
        model = context.get("model", self.model)

        if provider:
            cmd.extend(["--provider", provider])
        if model:
            cmd.extend(["--model", model])

        # Add max turns if specified
        max_turns = context.get("max_turns")
        if max_turns:
            cmd.extend(["--max-turns", str(max_turns)])

        return cmd

    def _build_env(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Build environment variables for the subprocess."""
        env = os.environ.copy()

        # Allow context to pass additional env vars
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
        """Parse Goose output into ExecutionTrace.

        Goose outputs a mix of:
        - Conversational text
        - Tool calls (shown with specific formatting)
        - Final response

        We parse this to extract tool calls and the final output.
        """
        session_id = f"goose-{uuid.uuid4().hex[:8]}"
        total_latency = (end_time - start_time).total_seconds() * 1000

        # Try to parse as JSON first (if --output-format json was used)
        try:
            data = json.loads(stdout)
            return self._parse_json_output(data, session_id, start_time, end_time)
        except json.JSONDecodeError:
            pass

        # Parse text output (default Goose format)
        steps = self._extract_tool_calls_from_text(stdout)
        final_output = self._extract_final_output(stdout)

        # Check for errors
        if returncode != 0:
            error_msg = stderr or f"Goose exited with code {returncode}"
            if not final_output:
                final_output = error_msg

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
        )

    def _parse_json_output(
        self,
        data: Dict[str, Any],
        session_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> ExecutionTrace:
        """Parse JSON output format from Goose."""
        steps = []

        # Extract tool calls from JSON
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

        # Extract tokens if available
        tokens_data = data.get("tokens", data.get("usage", {}))
        total_tokens = None
        if tokens_data:
            if isinstance(tokens_data, dict):
                total_tokens = TokenUsage(
                    input_tokens=tokens_data.get("input", tokens_data.get("input_tokens", 0)),
                    output_tokens=tokens_data.get("output", tokens_data.get("output_tokens", 0)),
                    cached_tokens=tokens_data.get("cached", 0),
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
        """Extract tool calls from Goose's text output.

        Goose shows tool usage in format:
        ─── shell | developer ──────────────────────────
        command: ls -la
        <command output>
        <assistant response>
        """
        steps = []
        step_count = 0

        # Tool header pattern: "─── shell | developer ───" or "─── text_editor | developer ───"
        tool_header_pattern = r"[─━]+\s*(\w+)\s*\|\s*\w+\s*[─━]+"

        # Common tool names in Goose
        known_tools = {
            "bash", "shell", "read", "write", "edit", "search", "grep", "find",
            "git", "npm", "pip", "python", "node", "cargo", "make",
            "developer", "computercontroller", "memory", "text_editor",
            "ripgrep", "list_directory",
        }

        # Split output into sections by tool headers
        lines = output.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for tool header
            match = re.search(tool_header_pattern, line, re.IGNORECASE)
            if match:
                tool_name = match.group(1).lower()
                # Normalize tool names
                if tool_name == "shell":
                    tool_name = "bash"

                if tool_name in known_tools or len(tool_name) > 2:
                    step_count += 1

                    # Collect parameters and output until next section
                    i += 1
                    params = {}
                    tool_output_lines = []

                    # First line after header is usually the command/parameter
                    if i < len(lines):
                        param_line = lines[i].strip()
                        if param_line.startswith("command:"):
                            params["command"] = param_line[8:].strip()
                            i += 1
                        elif param_line.startswith("path:"):
                            params["path"] = param_line[5:].strip()
                            i += 1

                    # Collect output until we hit another tool header or end
                    while i < len(lines):
                        next_line = lines[i]
                        # Check if this is another tool header
                        if re.search(tool_header_pattern, next_line, re.IGNORECASE):
                            break
                        # Check if this looks like assistant text (starts with capital, no indentation)
                        if next_line and not next_line.startswith(" ") and next_line[0:1].isupper():
                            # Could be assistant response, check if it's not output
                            if not any(c in next_line for c in ["│", "├", "└", "-rw", "drw"]):
                                break
                        tool_output_lines.append(next_line)
                        i += 1

                    # Trim empty lines from output
                    tool_output = "\n".join(tool_output_lines).strip()
                    # Limit output size for storage
                    if len(tool_output) > 2000:
                        tool_output = tool_output[:2000] + "..."

                    steps.append(
                        StepTrace(
                            step_id=f"step-{step_count}",
                            step_name=f"Step {step_count}",
                            tool_name=tool_name,
                            parameters=params,
                            output=tool_output,
                            success=True,
                            error=None,
                            metrics=StepMetrics(),
                        )
                    )
                    continue
            i += 1

        # Deduplicate consecutive same-tool calls (but keep different outputs)
        if steps:
            deduped = [steps[0]]
            for step in steps[1:]:
                # Only dedupe if same tool AND no output on current
                if step.tool_name != deduped[-1].tool_name or step.output:
                    deduped.append(step)
            steps = deduped

        return steps

    def _extract_final_output(self, output: str) -> str:
        """Extract the final response from Goose output.

        The final output is everything after the last tool execution block.
        """
        # Remove ANSI escape codes
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean_output = ansi_escape.sub("", output)

        # Find the last tool header and take everything after it
        tool_header_pattern = r"[─━]+\s*\w+\s*\|\s*\w+\s*[─━]+"

        lines = clean_output.split("\n")
        last_tool_end = -1
        in_tool_block = False

        for i, line in enumerate(lines):
            # Check if this is a tool header
            if re.search(tool_header_pattern, line):
                in_tool_block = True
                continue

            # If we're in a tool block, look for where output ends
            if in_tool_block:
                # Tool output ends when we see a line that looks like assistant text
                # (starts with capital letter, not indented, not part of command output)
                if line and not line.startswith(" ") and not line.startswith("\t"):
                    # Check if it's the start of a new sentence/response
                    if line[0:1].isupper() and not any(c in line for c in ["-rw", "drw", "total ", "│", "├", "└"]):
                        last_tool_end = i
                        in_tool_block = False

        # If we found a response after tool output, return everything from there
        if last_tool_end > 0:
            response_lines = lines[last_tool_end:]
            # Filter out session metadata lines
            response_lines = [l for l in response_lines if not l.startswith("starting session")]
            return "\n".join(response_lines).strip()

        # Fallback: return the last substantive paragraphs
        paragraphs = clean_output.strip().split("\n\n")
        response_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if para.startswith("$") or para.startswith(">>>"):
                continue
            if re.match(r"^[─━═]+", para):
                continue
            if para.startswith("starting session"):
                continue
            response_paragraphs.append(para)

        if response_paragraphs:
            # Return all response paragraphs after the last tool-related one
            return "\n\n".join(response_paragraphs[-3:])  # Last 3 paragraphs

        return clean_output.strip()

    def _estimate_cost(self, steps: List[StepTrace], output: str) -> float:
        """Estimate cost based on output length and tool calls.

        This is a rough estimate since Goose doesn't always report cost.
        For accurate cost tracking, check the LLM provider's dashboard.
        """
        # Rough estimate: ~$0.003 per 1K tokens for Claude Sonnet
        # Assume ~4 chars per token
        total_chars = len(output)
        estimated_tokens = total_chars / 4
        estimated_cost = (estimated_tokens / 1000) * 0.003

        # Add cost per step (tool calls use tokens)
        estimated_cost += len(steps) * 0.001

        return round(estimated_cost, 6)

    def _extract_tokens(self, output: str) -> Optional[TokenUsage]:
        """Try to extract token usage from output.

        Returns None if token info is not available.
        """
        # Look for token counts in output
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
            session_id=f"goose-error-{uuid.uuid4().hex[:8]}",
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
        """Check if Goose CLI is available."""
        try:
            result = subprocess.run(
                [self.goose_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info(f"Goose version: {result.stdout.strip()}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Goose health check failed: {e}")
            return False
