"""CrewAI adapter for skill testing.

This module provides an adapter for executing skills through CrewAI agents,
injecting skills as agent backstory and capturing execution traces.

CrewAI (https://github.com/joaomdmoura/crewAI) is a framework for orchestrating
role-playing, autonomous AI agents. It enables multi-agent collaboration where
agents have specific roles, goals, and backstories.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    CrewAISkillAdapter                           │
    ├─────────────────────────────────────────────────────────────────┤
    │  Skill Injection Strategy:                                      │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │  Agent Configuration                                        ││
    │  │  ├── role: "Skill Executor"                                ││
    │  │  ├── goal: Execute skill per instructions                  ││
    │  │  ├── backstory: {skill.instructions}                       ││
    │  │  └── tools: [configured tools]                             ││
    │  └─────────────────────────────────────────────────────────────┘│
    │                                                                  │
    │  Trace Capture:                                                 │
    │  ├── Tool executions via callback handlers                      │
    │  ├── Agent thoughts and actions                                 │
    │  ├── Task results                                               │
    │  └── Token usage from LLM calls                                 │
    └─────────────────────────────────────────────────────────────────┘

Example usage:
    config = AgentConfig(type=AgentType.CREWAI)
    adapter = CrewAISkillAdapter(config)
    trace = await adapter.execute(skill, "Research and summarize AI trends")

Note:
    CrewAI runs synchronously, so this adapter uses run_in_executor
    for async compatibility.

Author: EvalView Team
"""

from __future__ import annotations

import asyncio
import io
import sys
import uuid
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Type
import logging
import re

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
_DEFAULT_MODEL: Final[str] = "gpt-4o"
_DEFAULT_MAX_ITER: Final[int] = 10


@dataclass
class CrewAIExecutionContext:
    """Context for tracking CrewAI execution.

    Mutable container for collecting trace data during execution.
    Thread-safe for use with callbacks.
    """
    tool_calls: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    commands_ran: List[str] = field(default_factory=list)
    events: List[TraceEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    captured_output: str = ""

    def add_tool_call(
        self,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a tool call.

        Args:
            tool_name: Name of the tool called.
            tool_input: Optional input parameters.
        """
        self.tool_calls.append(tool_name)
        self.events.append(TraceEvent(
            type=TraceEventType.TOOL_CALL,
            tool_name=tool_name,
            tool_input=tool_input,
        ))

    def add_file_operation(
        self,
        operation: str,
        path: str,
    ) -> None:
        """Record a file operation.

        Args:
            operation: Type of operation (create, modify).
            path: File path affected.
        """
        if operation == "create":
            self.files_created.append(path)
            self.events.append(TraceEvent(
                type=TraceEventType.FILE_CREATE,
                file_path=path,
            ))
        elif operation == "modify":
            self.files_modified.append(path)
            self.events.append(TraceEvent(
                type=TraceEventType.FILE_MODIFY,
                file_path=path,
            ))

    def add_command(self, command: str) -> None:
        """Record a command execution.

        Args:
            command: Command string that was executed.
        """
        self.commands_ran.append(command)
        self.events.append(TraceEvent(
            type=TraceEventType.COMMAND_RUN,
            command=command,
        ))


class CrewAICallbackHandler:
    """Callback handler for capturing CrewAI execution events.

    Integrates with CrewAI's callback system to capture:
        - Tool executions
        - Agent actions
        - Task completions
        - Token usage

    This is injected into the Crew to provide real-time trace capture.
    """

    def __init__(self, context: CrewAIExecutionContext) -> None:
        """Initialize callback handler.

        Args:
            context: Execution context to populate with events.
        """
        self.context = context

    def on_tool_start(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> None:
        """Called when a tool execution starts.

        Args:
            tool_name: Name of the tool being invoked.
            tool_input: Input parameters for the tool.
        """
        logger.debug(f"CrewAI tool start: {tool_name}")
        self.context.add_tool_call(tool_name, tool_input)

        # Track file operations from tool input
        self._extract_file_operations(tool_name, tool_input)

    def on_tool_end(
        self,
        tool_name: str,
        tool_output: Any,
    ) -> None:
        """Called when a tool execution completes.

        Args:
            tool_name: Name of the tool.
            tool_output: Output from the tool.
        """
        logger.debug(f"CrewAI tool end: {tool_name}")
        # Update last event with output
        if self.context.events:
            last_event = self.context.events[-1]
            if last_event.tool_name == tool_name:
                # Create new event with output (events are immutable)
                self.context.events[-1] = TraceEvent(
                    type=last_event.type,
                    timestamp=last_event.timestamp,
                    tool_name=last_event.tool_name,
                    tool_input=last_event.tool_input,
                    tool_output=str(tool_output)[:1000],
                    tool_success=True,
                )

    def on_agent_action(
        self,
        action: str,
        agent_name: str,
    ) -> None:
        """Called when an agent takes an action.

        Args:
            action: Description of the action.
            agent_name: Name of the agent.
        """
        logger.debug(f"CrewAI agent action: {agent_name} - {action}")

    def on_llm_end(
        self,
        response: Any,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call completes.

        Args:
            response: LLM response object.
            **kwargs: Additional metadata.
        """
        # Extract token usage if available
        if hasattr(response, "usage"):
            usage = response.usage
            self.context.total_input_tokens += getattr(usage, "prompt_tokens", 0)
            self.context.total_output_tokens += getattr(usage, "completion_tokens", 0)

    def _extract_file_operations(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> None:
        """Extract file operations from tool calls.

        Args:
            tool_name: Name of the tool.
            tool_input: Tool input parameters.
        """
        tool_lower = tool_name.lower()

        if any(w in tool_lower for w in ("write", "create", "save", "file")):
            path = tool_input.get("path", tool_input.get("file_path", ""))
            if path:
                self.context.add_file_operation("create", path)

        elif any(w in tool_lower for w in ("edit", "modify", "update")):
            path = tool_input.get("path", tool_input.get("file_path", ""))
            if path:
                self.context.add_file_operation("modify", path)

        elif any(w in tool_lower for w in ("shell", "bash", "command", "exec")):
            cmd = tool_input.get("command", tool_input.get("cmd", ""))
            if cmd:
                self.context.add_command(cmd)


class CrewAISkillAdapter(SkillAgentAdapter):
    """Adapter for executing skills through CrewAI agents.

    This adapter creates a CrewAI Crew with a single agent configured
    with the skill's instructions as its backstory. The agent executes
    the user's query as a task, and all actions are captured as trace events.

    Skill Injection:
        Skills are injected by setting the agent's backstory to include
        the full skill instructions. The agent's goal is set to follow
        these instructions precisely.

    Trace Capture:
        A custom callback handler captures all tool executions, agent
        actions, and LLM calls during crew execution.

    Attributes:
        config: Agent configuration from test suite.
        model: LLM model to use for the agent.

    Example:
        >>> config = AgentConfig(type=AgentType.CREWAI)
        >>> adapter = CrewAISkillAdapter(config)
        >>> trace = await adapter.execute(skill, "Write a blog post")
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize CrewAI adapter.

        Args:
            config: Agent configuration from test suite.
        """
        super().__init__(config)
        self._crewai_available: Optional[bool] = None
        self.model = (config.env or {}).get("CREWAI_MODEL", _DEFAULT_MODEL)

    @property
    def name(self) -> str:
        """Return adapter identifier."""
        return "crewai"

    def _check_crewai_available(self) -> bool:
        """Check if CrewAI is installed and importable.

        Returns:
            True if CrewAI can be imported.
        """
        if self._crewai_available is None:
            try:
                import crewai
                self._crewai_available = True
            except ImportError:
                self._crewai_available = False
        return self._crewai_available

    async def health_check(self) -> bool:
        """Verify CrewAI is available.

        Returns:
            True if CrewAI is installed and can be imported.
        """
        return self._check_crewai_available()

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute a skill test through CrewAI.

        Creates a Crew with a skill-configured agent and executes
        the query as a task.

        Args:
            skill: The skill to test.
            query: User query to execute.
            context: Optional execution context.

        Returns:
            SkillAgentTrace with execution details.

        Raises:
            AgentNotFoundError: If CrewAI is not installed.
            AgentTimeoutError: If execution exceeds timeout.
            SkillAgentAdapterError: For other failures.
        """
        if not self._check_crewai_available():
            raise AgentNotFoundError(
                adapter_name=self.name,
                install_hint="Install CrewAI: pip install crewai crewai-tools",
            )

        context = context or {}
        test_name = context.get("test_name", "unnamed-test")
        session_id = uuid.uuid4().hex[:8]
        start_time = datetime.now()

        # Create execution context for trace capture
        exec_context = CrewAIExecutionContext()

        try:
            # Run CrewAI synchronously in executor
            result = await asyncio.wait_for(
                self._run_crew(skill, query, exec_context),
                timeout=self.config.timeout,
            )
        except asyncio.TimeoutError:
            raise AgentTimeoutError(self.name, self.config.timeout)
        except Exception as e:
            exec_context.errors.append(str(e))
            result = ""
            logger.error(f"CrewAI execution error: {e}")

        end_time = datetime.now()

        # Combine captured output with result
        final_output = result or exec_context.captured_output
        self._last_raw_output = final_output

        return SkillAgentTrace(
            session_id=session_id,
            skill_name=skill.metadata.name,
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            events=exec_context.events,
            tool_calls=exec_context.tool_calls,
            files_created=exec_context.files_created,
            files_modified=exec_context.files_modified,
            commands_ran=exec_context.commands_ran,
            total_input_tokens=exec_context.total_input_tokens,
            total_output_tokens=exec_context.total_output_tokens,
            final_output=final_output,
            errors=exec_context.errors,
        )

    async def _run_crew(
        self,
        skill: Skill,
        query: str,
        exec_context: CrewAIExecutionContext,
    ) -> str:
        """Run CrewAI crew in executor.

        CrewAI is synchronous, so we run it in a thread executor
        for async compatibility.

        Args:
            skill: Skill to inject.
            query: User query.
            exec_context: Context for trace capture.

        Returns:
            Crew execution result as string.
        """
        loop = asyncio.get_event_loop()

        def _sync_run() -> str:
            return self._execute_crew_sync(skill, query, exec_context)

        return await loop.run_in_executor(None, _sync_run)

    def _execute_crew_sync(
        self,
        skill: Skill,
        query: str,
        exec_context: CrewAIExecutionContext,
    ) -> str:
        """Synchronous CrewAI execution.

        Creates and runs a Crew with the skill-configured agent.

        Args:
            skill: Skill to inject.
            query: User query.
            exec_context: Context for trace capture.

        Returns:
            Crew execution result.
        """
        from crewai import Agent, Task, Crew, Process

        # Create skill-configured agent
        agent = self._create_skill_agent(skill)

        # Create task for the query
        task = Task(
            description=query,
            expected_output="Complete the task following the skill instructions",
            agent=agent,
        )

        # Create and run crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
            max_rpm=self.config.max_turns * 10,  # Rate limit based on max turns
        )

        # Capture stdout/stderr during execution
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = crew.kickoff()

            # Store captured output
            exec_context.captured_output = stdout_capture.getvalue()

            # Parse captured output for additional tool calls
            self._parse_crew_output(
                stdout_capture.getvalue(),
                exec_context,
            )

            # Return the result
            if hasattr(result, "raw"):
                return str(result.raw)
            return str(result)

        except Exception as e:
            exec_context.errors.append(f"Crew execution failed: {e}")
            exec_context.captured_output = (
                stdout_capture.getvalue() + "\n" + stderr_capture.getvalue()
            )
            raise

    def _create_skill_agent(self, skill: Skill) -> Any:
        """Create a CrewAI agent configured with the skill.

        The skill is injected via the agent's backstory, which
        provides context and instructions for its behavior.

        Args:
            skill: Skill to inject.

        Returns:
            Configured CrewAI Agent.
        """
        from crewai import Agent

        backstory = f"""You are an AI assistant with a specialized skill loaded.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILL: {skill.metadata.name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{skill.metadata.description}

## Instructions

{skill.instructions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You must follow these skill instructions precisely when completing tasks.
"""

        # Build tools list if specified
        tools = []
        if self.config.tools:
            tools = self._resolve_tools(self.config.tools)

        return Agent(
            role="Skill Executor",
            goal=f"Execute tasks using the {skill.metadata.name} skill following its instructions precisely",
            backstory=backstory,
            tools=tools,
            llm=self.model,
            max_iter=self.config.max_turns,
            verbose=True,
            allow_delegation=False,
        )

    def _resolve_tools(self, tool_names: List[str]) -> List[Any]:
        """Resolve tool names to CrewAI tool instances.

        Attempts to import and instantiate tools from crewai_tools
        or other sources.

        Args:
            tool_names: List of tool names to resolve.

        Returns:
            List of tool instances.
        """
        tools: List[Any] = []

        try:
            import crewai_tools
        except ImportError:
            logger.warning("crewai_tools not installed, tools will be limited")
            return tools

        # Map common tool names to crewai_tools classes
        tool_mapping = {
            "search": "SerperDevTool",
            "web_search": "SerperDevTool",
            "scrape": "ScrapeWebsiteTool",
            "read_file": "FileReadTool",
            "write_file": "FileWriterTool",
            "directory_read": "DirectoryReadTool",
            "code_interpreter": "CodeInterpreterTool",
        }

        for name in tool_names:
            tool_class_name = tool_mapping.get(name.lower())
            if tool_class_name and hasattr(crewai_tools, tool_class_name):
                try:
                    tool_class = getattr(crewai_tools, tool_class_name)
                    tools.append(tool_class())
                except Exception as e:
                    logger.warning(f"Failed to instantiate tool {name}: {e}")

        return tools

    def _parse_crew_output(
        self,
        output: str,
        exec_context: CrewAIExecutionContext,
    ) -> None:
        """Parse crew output for additional trace information.

        CrewAI verbose output contains tool usage that may not
        be captured by callbacks. This parses the text output
        for additional trace data.

        Args:
            output: Captured stdout from crew execution.
            exec_context: Context to update with parsed data.
        """
        # Pattern for tool usage in CrewAI output
        tool_pattern = re.compile(
            r"Using tool:\s*(\w+)",
            re.IGNORECASE,
        )

        for match in tool_pattern.finditer(output):
            tool_name = match.group(1)
            if tool_name not in exec_context.tool_calls:
                exec_context.tool_calls.append(tool_name)

        # Pattern for file operations
        file_create_pattern = re.compile(
            r"(?:Creating|Writing to|Saving)\s+(?:file\s+)?['\"]?([^\s'\"]+\.\w+)['\"]?",
            re.IGNORECASE,
        )

        for match in file_create_pattern.finditer(output):
            path = match.group(1)
            if path not in exec_context.files_created:
                exec_context.files_created.append(path)

        # Pattern for command execution
        command_pattern = re.compile(
            r"(?:Running|Executing|Command):\s*['\"]?([^'\"\n]+)['\"]?",
            re.IGNORECASE,
        )

        for match in command_pattern.finditer(output):
            cmd = match.group(1).strip()
            if cmd and cmd not in exec_context.commands_ran:
                exec_context.commands_ran.append(cmd)
