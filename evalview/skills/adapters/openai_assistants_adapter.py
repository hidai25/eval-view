"""OpenAI Responses API adapter for skill testing.

This module provides an adapter for executing skills through OpenAI's
Responses API, capturing detailed execution traces from the response's
typed output items.

Formerly built on the Assistants API (assistants/threads/runs), which
OpenAI removed on August 26, 2026. The Responses API
(https://platform.openai.com/docs/api-reference/responses) is its
replacement: a single synchronous call that runs the model with
instructions and built-in tools and returns tool activity inline, with
the Conversations API available for multi-turn state.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                OpenAIAssistantsSkillAdapter                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  Skill Injection:                                               │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │  Response Request                                           ││
    │  │  ├── instructions: {skill.instructions}                    ││
    │  │  ├── tools: [code_interpreter, file_search, functions]     ││
    │  │  └── model: gpt-4o                                         ││
    │  └─────────────────────────────────────────────────────────────┘│
    │                                                                  │
    │  Execution Flow:                                                │
    │  1. Build instructions from skill                               │
    │  2. Create Response with query, instructions, and tools         │
    │  3. Await completion (no polling — the call is synchronous)     │
    │  4. Walk typed output items for trace capture                   │
    │  5. Parse tool calls, code execution, outputs                   │
    │                                                                  │
    │  Trace Capture via Output Items:                                │
    │  ├── message: Final assistant response                          │
    │  ├── *_call items: Function calls, code interpreter             │
    │  └── Token usage from response.usage                            │
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
    - No server-side objects created (nothing to clean up)
    - Request timeouts enforced

Author: EvalView Team
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
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
_MAX_OUTPUT_TOKENS: Final[int] = 4096


@dataclass
class ResponsesConfig:
    """Configuration for OpenAI Responses API execution.

    Attributes:
        api_key: OpenAI API key.
        model: Model to use (e.g., gpt-4o).
        prompt_id: Optional dashboard-managed Prompt object id (pmpt_...)
            to run instead of inline instructions.
    """
    api_key: str
    model: str = _DEFAULT_MODEL
    prompt_id: Optional[str] = None

    @classmethod
    def from_agent_config(cls, config: AgentConfig) -> "ResponsesConfig":
        """Create ResponsesConfig from AgentConfig.

        Args:
            config: Agent configuration from test suite.

        Returns:
            Configured ResponsesConfig.

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

        if env.get("OPENAI_ASSISTANT_ID") or os.getenv("OPENAI_ASSISTANT_ID"):
            logger.warning(
                "OPENAI_ASSISTANT_ID is ignored: OpenAI removed the Assistants "
                "API on 2026-08-26. Skill instructions are now passed inline "
                "via the Responses API (or set OPENAI_PROMPT_ID)."
            )

        return cls(
            api_key=api_key,
            model=env.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)),
            prompt_id=env.get("OPENAI_PROMPT_ID", os.getenv("OPENAI_PROMPT_ID")),
        )


# Backwards-compatible alias for the pre-migration config name.
AssistantConfig = ResponsesConfig


class OpenAIAssistantsSkillAdapter(SkillAgentAdapter):
    """Adapter for executing skills through the OpenAI Responses API.

    This adapter injects the skill's instructions into a Responses API
    request, executes the query, and captures detailed traces from the
    response's typed output items.

    Features:
        - Skill injection via per-request instructions
        - Detailed trace capture from response output items
        - Support for Code Interpreter and File Search tools
        - Token usage tracking from response.usage

    Attributes:
        config: Agent configuration from test suite.
        responses_config: OpenAI-specific configuration.
        _client: Cached OpenAI client instance.

    Example:
        >>> config = AgentConfig(type=AgentType.OPENAI_ASSISTANTS)
        >>> adapter = OpenAIAssistantsSkillAdapter(config)
        >>> trace = await adapter.execute(skill, "Write Python code")
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize OpenAI Responses adapter.

        Args:
            config: Agent configuration from test suite.
        """
        super().__init__(config)
        self.responses_config = ResponsesConfig.from_agent_config(config)
        self._client: Optional["AsyncOpenAI"] = None

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
                self._client = AsyncOpenAI(api_key=self.responses_config.api_key)
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
            logger.debug(f"OpenAI Responses health check failed: {e}")
            return False

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute a skill test through the OpenAI Responses API.

        Builds a request with the skill's instructions, runs it, and
        captures the full execution trace from the response.

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

        try:
            request: Dict[str, Any] = {
                "model": self.responses_config.model,
                "input": query,
                "instructions": self._build_skill_instructions(skill),
                "max_output_tokens": _MAX_OUTPUT_TOKENS,
            }
            if self.responses_config.prompt_id:
                request["prompt"] = {"id": self.responses_config.prompt_id}
            tools = self._get_tools_config()
            if tools:
                request["tools"] = tools

            # The Responses API is synchronous — no run creation/polling.
            response = await asyncio.wait_for(
                self.client.responses.create(**request),
                timeout=self.config.timeout,
            )

            # Extract token usage
            if response.usage:
                total_input_tokens = response.usage.input_tokens
                total_output_tokens = response.usage.output_tokens

            # Process typed output items for trace
            for item in response.output or []:
                self._process_output_item(
                    item=item,
                    events=events,
                    tool_calls=tool_calls,
                    files_created=files_created,
                    commands_ran=commands_ran,
                )

            # Final output: aggregated text across message items
            final_output = getattr(response, "output_text", "") or ""

            # Check for response failure
            if response.status == "failed":
                error_msg = "Response failed"
                if response.error:
                    error_msg = f"{response.error.code}: {response.error.message}"
                errors.append(error_msg)
            elif response.status == "incomplete":
                reason = ""
                details = getattr(response, "incomplete_details", None)
                if details:
                    reason = f": {getattr(details, 'reason', '')}"
                errors.append(f"Response incomplete{reason}")

        except asyncio.TimeoutError:
            raise AgentTimeoutError(self.name, self.config.timeout)
        except SkillAgentAdapterError:
            raise
        except Exception as e:
            errors.append(str(e))
            logger.error(f"OpenAI Responses execution error: {e}")

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

    def _build_skill_instructions(self, skill: Skill) -> str:
        """Build system instructions with skill injection.

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
        """Get tools configuration for the Responses API request.

        Returns:
            List of tool configurations.
        """
        tools: List[Dict[str, Any]] = []

        # Default to code_interpreter — unless a dashboard Prompt object is
        # in use, whose own tool configuration would be silently overridden
        # by request-level tools.
        enable_code_interpreter = not self.responses_config.prompt_id
        enable_file_search = False

        # Check configured tools
        if self.config.tools:
            tool_set = set(t.lower() for t in self.config.tools)
            enable_code_interpreter = "code_interpreter" in tool_set or "code" in tool_set
            enable_file_search = "file_search" in tool_set or "retrieval" in tool_set

        if enable_code_interpreter:
            tools.append({"type": "code_interpreter", "container": {"type": "auto"}})

        if enable_file_search:
            # Unlike Assistants, the Responses API requires explicit
            # vector store ids for file_search.
            env = self.config.env or {}
            raw_ids = env.get(
                "OPENAI_VECTOR_STORE_IDS", os.getenv("OPENAI_VECTOR_STORE_IDS", "")
            )
            vector_store_ids = [vs.strip() for vs in raw_ids.split(",") if vs.strip()]
            if vector_store_ids:
                tools.append({"type": "file_search", "vector_store_ids": vector_store_ids})
            else:
                logger.warning(
                    "file_search requires vector store ids in the Responses API; "
                    "set OPENAI_VECTOR_STORE_IDS to enable it. Skipping."
                )

        return tools

    def _process_output_item(
        self,
        item: Any,
        events: List[TraceEvent],
        tool_calls: List[str],
        files_created: List[str],
        commands_ran: List[str],
    ) -> None:
        """Process a response output item for trace capture.

        Extracts tool calls and outputs from the typed output items that
        replaced the Assistants Run Steps API.

        Args:
            item: Response output item.
            events: List to append trace events.
            tool_calls: List to append tool names.
            files_created: List to append created files.
            commands_ran: List to append commands.
        """
        item_type = getattr(item, "type", "")

        if item_type == "code_interpreter_call":
            tool_calls.append("code_interpreter")

            # Extract code input
            code_input = item.code
            events.append(TraceEvent(
                type=TraceEventType.TOOL_CALL,
                tool_name="code_interpreter",
                tool_input={"code": code_input},
            ))

            # Track as command execution
            if code_input:
                commands_ran.append(f"[python] {code_input[:100]}...")

            # Check for file outputs
            for output in item.outputs or []:
                if getattr(output, "type", "") == "image":
                    files_created.append(f"[generated_image:{output.url}]")

        elif item_type == "file_search_call":
            tool_calls.append("file_search")
            events.append(TraceEvent(
                type=TraceEventType.TOOL_CALL,
                tool_name="file_search",
            ))

        elif item_type == "function_call":
            func_name = item.name
            tool_calls.append(func_name)

            try:
                import json
                func_args = json.loads(item.arguments)
            except Exception:
                func_args = {"raw": item.arguments}

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

        elif item_type == "message":
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
        """Clean up adapter resources.

        Retained for interface compatibility. The Responses API creates
        no server-side Assistant objects, so there is nothing to delete.
        """
        return None
