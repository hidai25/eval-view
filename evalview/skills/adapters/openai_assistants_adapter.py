"""OpenAI Assistants API adapter for skill testing.

This module provides an adapter for executing skills through OpenAI's
Assistants API, capturing detailed execution traces via the Run Steps API.

The OpenAI Assistants API (https://platform.openai.com/docs/assistants)
provides a stateful, multi-turn conversation interface with built-in
tool use, code interpreter, and file handling capabilities.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                OpenAIAssistantsSkillAdapter                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  Skill Injection:                                               │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │  Assistant Configuration                                    ││
    │  │  ├── instructions: {skill.instructions}                    ││
    │  │  ├── tools: [code_interpreter, file_search, functions]     ││
    │  │  └── model: gpt-4o                                         ││
    │  └─────────────────────────────────────────────────────────────┘│
    │                                                                  │
    │  Execution Flow:                                                │
    │  1. Create/reuse Assistant with skill instructions              │
    │  2. Create Thread for conversation                              │
    │  3. Add user message with query                                 │
    │  4. Create Run and poll for completion                          │
    │  5. Retrieve Run Steps for trace capture                        │
    │  6. Parse tool calls, code execution, outputs                   │
    │                                                                  │
    │  Trace Capture via Run Steps:                                   │
    │  ├── message_creation: Final assistant response                 │
    │  ├── tool_calls: Function calls, code interpreter              │
    │  └── Token usage from Run metadata                              │
    └─────────────────────────────────────────────────────────────────┘

Example usage:
    config = AgentConfig(
        type=AgentType.OPENAI_ASSISTANTS,
        env={"OPENAI_API_KEY": "sk-..."}
    )
    adapter = OpenAIAssistantsSkillAdapter(config)
    trace = await adapter.execute(skill, "Analyze this dataset")

Security considerations:
    - API key validated but never logged
    - Assistants are cleaned up after test (optional)
    - File uploads sanitized and size-limited
    - Request timeouts enforced

Author: EvalView Team
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Final, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from evalview.skills.adapters.base import (
    SkillAgentAdapter,
    SkillAgentAdapterError,
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
_DEFAULT_POLL_INTERVAL: Final[float] = 1.0
_MAX_POLL_ATTEMPTS: Final[int] = 300  # 5 minutes with 1s interval


class RunStatus(str, Enum):
    """OpenAI Run status values."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    REQUIRES_ACTION = "requires_action"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class AssistantConfig:
    """Configuration for OpenAI Assistant creation.

    Attributes:
        api_key: OpenAI API key.
        model: Model to use (e.g., gpt-4o).
        assistant_id: Optional existing assistant ID to reuse.
        cleanup_assistant: Whether to delete assistant after test.
    """
    api_key: str
    model: str = _DEFAULT_MODEL
    assistant_id: Optional[str] = None
    cleanup_assistant: bool = True

    @classmethod
    def from_agent_config(cls, config: AgentConfig) -> "AssistantConfig":
        """Create AssistantConfig from AgentConfig.

        Args:
            config: Agent configuration from test suite.

        Returns:
            Configured AssistantConfig.

        Raises:
            SkillAgentAdapterError: If API key not found.
        """
        env = config.env or {}

        api_key = env.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not api_key:
            raise SkillAgentAdapterError(
                "OPENAI_API_KEY required for OpenAI Assistants adapter",
                adapter_name="openai-assistants",
            )

        return cls(
            api_key=api_key,
            model=env.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)),
            assistant_id=env.get("OPENAI_ASSISTANT_ID"),
            cleanup_assistant=env.get("CLEANUP_ASSISTANT", "true").lower() == "true",
        )


class OpenAIAssistantsSkillAdapter(SkillAgentAdapter):
    """Adapter for executing skills through OpenAI Assistants API.

    This adapter creates an OpenAI Assistant configured with the skill's
    instructions, executes queries via the Threads API, and captures
    detailed traces from Run Steps.

    Features:
        - Automatic Assistant creation with skill injection
        - Thread-based conversation management
        - Detailed trace capture via Run Steps API
        - Support for Code Interpreter and File Search tools
        - Token usage tracking from Run metadata

    Attributes:
        config: Agent configuration from test suite.
        assistant_config: OpenAI-specific configuration.
        _client: Cached OpenAI client instance.

    Example:
        >>> config = AgentConfig(type=AgentType.OPENAI_ASSISTANTS)
        >>> adapter = OpenAIAssistantsSkillAdapter(config)
        >>> trace = await adapter.execute(skill, "Write Python code")
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize OpenAI Assistants adapter.

        Args:
            config: Agent configuration from test suite.
        """
        super().__init__(config)
        self.assistant_config = AssistantConfig.from_agent_config(config)
        self._client: Optional["AsyncOpenAI"] = None
        self._created_assistants: List[str] = []

    @property
    def name(self) -> str:
        """Return adapter identifier."""
        return "openai-assistants"

    @property
    def client(self):
        """Lazily initialize and return OpenAI client.

        Returns:
            OpenAI client instance.

        Raises:
            SkillAgentAdapterError: If openai package not installed.
        """
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.assistant_config.api_key)
            except ImportError:
                raise SkillAgentAdapterError(
                    "openai package required. Install with: pip install openai",
                    adapter_name=self.name,
                )
        return self._client

    async def health_check(self) -> bool:
        """Verify OpenAI API is accessible.

        Attempts to list models to verify API connectivity.

        Returns:
            True if API is accessible.
        """
        try:
            # Simple API call to verify connectivity
            await asyncio.wait_for(
                self.client.models.list(),
                timeout=10.0,
            )
            return True
        except Exception as e:
            logger.debug(f"OpenAI Assistants health check failed: {e}")
            return False

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute a skill test through OpenAI Assistants API.

        Creates an Assistant with skill instructions, runs a conversation,
        and captures the full execution trace.

        Args:
            skill: The skill to test.
            query: User query to execute.
            context: Optional execution context.

        Returns:
            SkillAgentTrace with execution details.

        Raises:
            AgentTimeoutError: If execution exceeds timeout.
            SkillAgentAdapterError: For other failures.
        """
        context = context or {}
        test_name = context.get("test_name", "unnamed-test")
        session_id = uuid.uuid4().hex[:8]
        start_time = datetime.now()

        events: List[TraceEvent] = []
        tool_calls: List[str] = []
        files_created: List[str] = []
        files_modified: List[str] = []
        commands_ran: List[str] = []
        errors: List[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        final_output = ""

        assistant_id = None
        thread_id = None

        try:
            # Create or get assistant
            assistant_id = await self._get_or_create_assistant(skill)

            # Create thread
            thread = await self.client.beta.threads.create()
            thread_id = thread.id

            # Add user message
            await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=query,
            )

            # Create and run
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                max_completion_tokens=4096,
            )

            # Poll for completion
            run = await self._poll_run(
                thread_id=thread_id,
                run_id=run.id,
            )

            # Extract token usage
            if run.usage:
                total_input_tokens = run.usage.prompt_tokens
                total_output_tokens = run.usage.completion_tokens

            # Get run steps for trace
            steps = await self.client.beta.threads.runs.steps.list(
                thread_id=thread_id,
                run_id=run.id,
            )

            # Process steps for trace
            for step in steps.data:
                self._process_run_step(
                    step=step,
                    events=events,
                    tool_calls=tool_calls,
                    files_created=files_created,
                    commands_ran=commands_ran,
                )

            # Get final output from messages
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_id,
                order="desc",
                limit=1,
            )

            if messages.data:
                last_message = messages.data[0]
                if last_message.role == "assistant":
                    for content_block in last_message.content:
                        if content_block.type == "text":
                            final_output = content_block.text.value

            # Check for run failure
            if run.status == RunStatus.FAILED:
                error_msg = "Run failed"
                if run.last_error:
                    error_msg = f"{run.last_error.code}: {run.last_error.message}"
                errors.append(error_msg)

        except asyncio.TimeoutError:
            raise AgentTimeoutError(self.name, self.config.timeout)
        except SkillAgentAdapterError:
            raise
        except Exception as e:
            errors.append(str(e))
            logger.error(f"OpenAI Assistants execution error: {e}")

        finally:
            # Cleanup assistant if configured
            if (
                self.assistant_config.cleanup_assistant
                and assistant_id
                and assistant_id in self._created_assistants
            ):
                try:
                    await self.client.beta.assistants.delete(assistant_id)
                    self._created_assistants.remove(assistant_id)
                except Exception as e:
                    logger.warning(f"Failed to cleanup assistant: {e}")

        end_time = datetime.now()
        self._last_raw_output = final_output

        return SkillAgentTrace(
            session_id=session_id,
            skill_name=skill.metadata.name,
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

    async def _get_or_create_assistant(self, skill: Skill) -> str:
        """Get existing or create new assistant with skill.

        Args:
            skill: Skill to inject into assistant.

        Returns:
            Assistant ID.
        """
        # Use existing assistant if specified
        if self.assistant_config.assistant_id:
            return self.assistant_config.assistant_id

        # Build assistant instructions with skill
        instructions = self._build_skill_instructions(skill)

        # Determine tools to enable
        tools = self._get_tools_config()

        # Create new assistant
        assistant = await self.client.beta.assistants.create(
            name=f"EvalView Skill Test: {skill.metadata.name}",
            instructions=instructions,
            model=self.assistant_config.model,
            tools=tools,
        )

        self._created_assistants.append(assistant.id)
        return assistant.id

    def _build_skill_instructions(self, skill: Skill) -> str:
        """Build assistant instructions with skill injection.

        Args:
            skill: Skill to inject.

        Returns:
            Complete instructions string.
        """
        return f"""You are an AI assistant with a specialized skill loaded.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILL: {skill.metadata.name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{skill.metadata.description}

## Instructions

{skill.instructions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Follow the skill instructions above precisely when responding to user queries.
Use the available tools (code interpreter, file search) as needed to complete tasks.
"""

    def _get_tools_config(self) -> List[Dict[str, Any]]:
        """Get tools configuration for assistant.

        Returns:
            List of tool configurations.
        """
        tools = []

        # Check configured tools
        if self.config.tools:
            tool_set = set(t.lower() for t in self.config.tools)

            if "code_interpreter" in tool_set or "code" in tool_set:
                tools.append({"type": "code_interpreter"})

            if "file_search" in tool_set or "retrieval" in tool_set:
                tools.append({"type": "file_search"})

        else:
            # Default: enable code interpreter
            tools.append({"type": "code_interpreter"})

        return tools

    async def _poll_run(
        self,
        thread_id: str,
        run_id: str,
    ) -> Any:
        """Poll run until completion or timeout.

        Args:
            thread_id: Thread ID.
            run_id: Run ID.

        Returns:
            Completed Run object.

        Raises:
            asyncio.TimeoutError: If polling exceeds timeout.
        """
        deadline = asyncio.get_event_loop().time() + self.config.timeout

        terminal_statuses = {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.EXPIRED,
        }

        while asyncio.get_event_loop().time() < deadline:
            run = await self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id,
            )

            if run.status in terminal_statuses:
                return run

            if run.status == RunStatus.REQUIRES_ACTION:
                # Handle required actions (tool outputs)
                # For skill testing, we typically don't have custom functions
                # that require user-provided outputs
                logger.warning(
                    f"Run requires action but no handler configured: {run.required_action}"
                )
                # Cancel the run to avoid hanging
                await self.client.beta.threads.runs.cancel(
                    thread_id=thread_id,
                    run_id=run_id,
                )
                raise SkillAgentAdapterError(
                    "Run requires action (custom function outputs) which is not supported",
                    adapter_name=self.name,
                )

            await asyncio.sleep(_DEFAULT_POLL_INTERVAL)

        raise asyncio.TimeoutError("Run polling timeout exceeded")

    def _process_run_step(
        self,
        step: Any,
        events: List[TraceEvent],
        tool_calls: List[str],
        files_created: List[str],
        commands_ran: List[str],
    ) -> None:
        """Process a run step for trace capture.

        Extracts tool calls and outputs from the step details.

        Args:
            step: Run step object.
            events: List to append trace events.
            tool_calls: List to append tool names.
            files_created: List to append created files.
            commands_ran: List to append commands.
        """
        if step.type == "tool_calls":
            for tool_call in step.step_details.tool_calls:
                tool_type = tool_call.type

                if tool_type == "code_interpreter":
                    tool_calls.append("code_interpreter")

                    # Extract code input
                    code_input = tool_call.code_interpreter.input
                    events.append(TraceEvent(
                        type=TraceEventType.TOOL_CALL,
                        tool_name="code_interpreter",
                        tool_input={"code": code_input},
                    ))

                    # Track as command execution
                    if code_input:
                        commands_ran.append(f"[python] {code_input[:100]}...")

                    # Check for file outputs
                    for output in tool_call.code_interpreter.outputs:
                        if output.type == "image":
                            files_created.append(f"[generated_image:{output.image.file_id}]")

                elif tool_type == "file_search":
                    tool_calls.append("file_search")
                    events.append(TraceEvent(
                        type=TraceEventType.TOOL_CALL,
                        tool_name="file_search",
                    ))

                elif tool_type == "function":
                    func_name = tool_call.function.name
                    tool_calls.append(func_name)

                    try:
                        import json
                        func_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        func_args = {"raw": tool_call.function.arguments}

                    events.append(TraceEvent(
                        type=TraceEventType.TOOL_CALL,
                        tool_name=func_name,
                        tool_input=func_args,
                    ))

                    # Track file operations from function args
                    self._track_function_operations(
                        func_name=func_name,
                        func_args=func_args,
                        files_created=files_created,
                        commands_ran=commands_ran,
                    )

        elif step.type == "message_creation":
            events.append(TraceEvent(
                type=TraceEventType.LLM_CALL,
                tool_name="message_creation",
            ))

    def _track_function_operations(
        self,
        func_name: str,
        func_args: Dict[str, Any],
        files_created: List[str],
        commands_ran: List[str],
    ) -> None:
        """Track file and command operations from function calls.

        Args:
            func_name: Function name.
            func_args: Function arguments.
            files_created: List to append created files.
            commands_ran: List to append commands.
        """
        func_lower = func_name.lower()

        # File operations
        if any(w in func_lower for w in ("write", "create", "save")):
            path = func_args.get("path", func_args.get("file_path", ""))
            if path:
                files_created.append(path)

        # Command operations
        elif any(w in func_lower for w in ("run", "exec", "shell", "command")):
            cmd = func_args.get("command", func_args.get("cmd", ""))
            if cmd:
                commands_ran.append(cmd)

    async def cleanup(self) -> None:
        """Clean up all created assistants.

        Call this to delete any assistants created during testing
        that weren't automatically cleaned up.
        """
        for assistant_id in self._created_assistants[:]:
            try:
                await self.client.beta.assistants.delete(assistant_id)
                self._created_assistants.remove(assistant_id)
                logger.debug(f"Cleaned up assistant: {assistant_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup assistant {assistant_id}: {e}")
