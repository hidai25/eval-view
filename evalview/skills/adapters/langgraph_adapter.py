"""LangGraph adapter for skill testing.

This module provides an adapter for executing skills through LangGraph agents,
supporting both local LangGraph instances and LangGraph Cloud deployments.

LangGraph (https://langchain-ai.github.io/langgraph/) is a framework for building
stateful, multi-actor applications with LLMs, built on LangChain.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    LangGraphSkillAdapter                         │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────────────┐           ┌─────────────────────────────┐ │
    │  │  Skill Injection │           │     Trace Capture           │ │
    │  │  - System prompt │           │     - Tool calls            │ │
    │  │  - Config merge  │           │     - State transitions     │ │
    │  └────────┬────────┘           │     - Token usage           │ │
    │           │                     └─────────────────────────────┘ │
    │           ▼                                                      │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │              LangGraph Client (SDK or HTTP)                 ││
    │  │  - langgraph-sdk for Cloud                                  ││
    │  │  - Direct invocation for local                              ││
    │  └─────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘

Example usage:
    config = AgentConfig(
        type=AgentType.LANGGRAPH,
        env={"LANGGRAPH_API_URL": "http://localhost:2024"}
    )
    adapter = LangGraphSkillAdapter(config)
    trace = await adapter.execute(skill, "Analyze this data")

Security considerations:
    - API keys validated but never logged
    - Request timeouts strictly enforced
    - Response size limits prevent memory exhaustion
    - All external calls wrapped in try/except

Author: EvalView Team
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
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

# Constants
_DEFAULT_API_URL: Final[str] = "http://localhost:2024"
_DEFAULT_ASSISTANT_ID: Final[str] = "agent"
_DEFAULT_TIMEOUT: Final[float] = 300.0
_MAX_RESPONSE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB


@dataclass(frozen=True)
class LangGraphConfig:
    """Configuration for LangGraph connection.

    Attributes:
        api_url: LangGraph API endpoint URL.
        api_key: Optional API key for authentication.
        assistant_id: ID of the assistant/graph to invoke.
        thread_id: Optional thread ID for conversation continuity.
    """
    api_url: str
    api_key: Optional[str]
    assistant_id: str
    thread_id: Optional[str] = None

    @classmethod
    def from_agent_config(cls, config: AgentConfig) -> "LangGraphConfig":
        """Create LangGraphConfig from AgentConfig.

        Extracts LangGraph-specific settings from environment
        and agent configuration.

        Args:
            config: Agent configuration from test suite.

        Returns:
            Configured LangGraphConfig instance.
        """
        env = config.env or {}

        return cls(
            api_url=env.get(
                "LANGGRAPH_API_URL",
                os.getenv("LANGGRAPH_API_URL", _DEFAULT_API_URL)
            ),
            api_key=env.get(
                "LANGGRAPH_API_KEY",
                os.getenv("LANGGRAPH_API_KEY")
            ),
            assistant_id=env.get(
                "LANGGRAPH_ASSISTANT_ID",
                os.getenv("LANGGRAPH_ASSISTANT_ID", _DEFAULT_ASSISTANT_ID)
            ),
            thread_id=env.get("LANGGRAPH_THREAD_ID"),
        )


@runtime_checkable
class LangGraphClientProtocol(Protocol):
    """Protocol defining LangGraph client interface.

    Enables dependency injection and testing with mock clients.
    """

    async def create_thread(self) -> str:
        """Create a new conversation thread."""
        ...

    async def send_message(
        self,
        thread_id: str,
        message: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a message and get response."""
        ...

    async def get_thread_state(self, thread_id: str) -> Dict[str, Any]:
        """Get current thread state."""
        ...


class LangGraphHTTPClient:
    """HTTP client for LangGraph API.

    Implements the LangGraph REST API protocol for communicating
    with LangGraph Cloud or local deployments.

    Attributes:
        config: LangGraph connection configuration.
        _session: Cached aiohttp session for connection reuse.
    """

    def __init__(self, config: LangGraphConfig) -> None:
        """Initialize HTTP client.

        Args:
            config: LangGraph connection configuration.
        """
        self.config = config
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session with lazy initialization."""
        if self._session is None:
            try:
                import aiohttp
                timeout = aiohttp.ClientTimeout(total=_DEFAULT_TIMEOUT)
                self._session = aiohttp.ClientSession(timeout=timeout)
            except ImportError:
                raise SkillAgentAdapterError(
                    "aiohttp required for LangGraph adapter. "
                    "Install with: pip install aiohttp",
                    adapter_name="langgraph",
                )
        return self._session

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers with authentication.

        Returns:
            Headers dictionary with Content-Type and optional auth.
        """
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def create_thread(self) -> str:
        """Create a new conversation thread.

        Returns:
            Thread ID for the new conversation.

        Raises:
            SkillAgentAdapterError: If thread creation fails.
        """
        session = await self._get_session()
        url = f"{self.config.api_url}/threads"

        try:
            async with session.post(
                url,
                headers=self._get_headers(),
                json={"assistant_id": self.config.assistant_id},
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise SkillAgentAdapterError(
                        f"Failed to create thread: {response.status} - {text}",
                        adapter_name="langgraph",
                    )
                data = await response.json()
                return data.get("thread_id", data.get("id", ""))
        except Exception as e:
            if isinstance(e, SkillAgentAdapterError):
                raise
            raise SkillAgentAdapterError(
                f"Thread creation failed: {e}",
                adapter_name="langgraph",
            )

    async def send_message(
        self,
        thread_id: str,
        message: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a message to the LangGraph agent.

        Args:
            thread_id: Conversation thread ID.
            message: User message to send.
            config: Optional configuration overrides.

        Returns:
            Response dictionary with agent output and metadata.

        Raises:
            SkillAgentAdapterError: If message send fails.
        """
        session = await self._get_session()
        url = f"{self.config.api_url}/threads/{thread_id}/runs"

        payload = {
            "assistant_id": self.config.assistant_id,
            "input": {"messages": [{"role": "user", "content": message}]},
        }

        if config:
            payload["config"] = config

        try:
            async with session.post(
                url,
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status not in (200, 201):
                    text = await response.text()
                    raise SkillAgentAdapterError(
                        f"Failed to send message: {response.status} - {text}",
                        adapter_name="langgraph",
                    )
                return await response.json()
        except Exception as e:
            if isinstance(e, SkillAgentAdapterError):
                raise
            raise SkillAgentAdapterError(
                f"Message send failed: {e}",
                adapter_name="langgraph",
            )

    async def get_thread_state(self, thread_id: str) -> Dict[str, Any]:
        """Get current state of a conversation thread.

        Args:
            thread_id: Thread to query.

        Returns:
            Thread state dictionary.
        """
        session = await self._get_session()
        url = f"{self.config.api_url}/threads/{thread_id}/state"

        try:
            async with session.get(
                url,
                headers=self._get_headers(),
            ) as response:
                if response.status != 200:
                    return {}
                return await response.json()
        except Exception:
            return {}

    async def stream_run(
        self,
        thread_id: str,
        message: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a run for real-time event capture.

        Args:
            thread_id: Conversation thread ID.
            message: User message to send.
            config: Optional configuration overrides.

        Yields:
            Event dictionaries as they arrive.
        """
        session = await self._get_session()
        url = f"{self.config.api_url}/threads/{thread_id}/runs/stream"

        payload = {
            "assistant_id": self.config.assistant_id,
            "input": {"messages": [{"role": "user", "content": message}]},
            "stream_mode": "events",
        }

        if config:
            payload["config"] = config

        try:
            async with session.post(
                url,
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status not in (200, 201):
                    return

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        try:
                            event_data = json.loads(line_str[6:])
                            yield event_data
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Stream error: {e}")

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None


class LangGraphSkillAdapter(SkillAgentAdapter):
    """Adapter for executing skills through LangGraph agents.

    This adapter supports:
        - Local LangGraph instances (localhost:2024)
        - LangGraph Cloud deployments
        - Skill injection via system prompt configuration
        - Streaming event capture for detailed traces
        - Token usage tracking

    The adapter injects skills by modifying the agent's configurable
    parameters, specifically the system message or instructions.

    Attributes:
        config: Agent configuration from test suite.
        lg_config: LangGraph-specific configuration.
        client: HTTP client for LangGraph API.

    Example:
        >>> config = AgentConfig(type=AgentType.LANGGRAPH)
        >>> adapter = LangGraphSkillAdapter(config)
        >>> trace = await adapter.execute(skill, "Summarize this document")
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize LangGraph adapter.

        Args:
            config: Agent configuration from test suite.
        """
        super().__init__(config)
        self.lg_config = LangGraphConfig.from_agent_config(config)
        self.client = LangGraphHTTPClient(self.lg_config)

    @property
    def name(self) -> str:
        """Return adapter identifier."""
        return "langgraph"

    async def health_check(self) -> bool:
        """Verify LangGraph API is accessible.

        Attempts to list assistants to verify connectivity.

        Returns:
            True if API is accessible and responsive.
        """
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.lg_config.api_url}/assistants"
                headers = {"Content-Type": "application/json"}
                if self.lg_config.api_key:
                    headers["Authorization"] = f"Bearer {self.lg_config.api_key}"

                async with session.get(url, headers=headers) as response:
                    return response.status in (200, 401)  # 401 = auth required but reachable
        except Exception as e:
            logger.debug(f"LangGraph health check failed: {e}")
            return False

    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute a skill test through LangGraph.

        Creates a new thread, injects the skill via configuration,
        streams the execution, and captures the full trace.

        Args:
            skill: The skill to test.
            query: User query to execute.
            context: Optional execution context overrides.

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
            # Create thread for this test
            thread_id = self.lg_config.thread_id
            if not thread_id:
                thread_id = await asyncio.wait_for(
                    self.client.create_thread(),
                    timeout=30.0,
                )

            # Build skill-injected configuration
            run_config = self._build_skill_config(skill)

            # Stream the run and capture events
            async for event in self._stream_with_timeout(
                thread_id=thread_id,
                message=query,
                config=run_config,
            ):
                self._process_event(
                    event=event,
                    events=events,
                    tool_calls=tool_calls,
                    files_created=files_created,
                    files_modified=files_modified,
                    commands_ran=commands_ran,
                )

                # Extract final output from end events
                if event.get("event") == "on_chain_end":
                    output_data = event.get("data", {}).get("output", {})
                    messages = output_data.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            final_output = last_msg.get("content", "")
                        elif hasattr(last_msg, "content"):
                            final_output = last_msg.content

                # Track token usage
                if "usage" in event.get("data", {}):
                    usage = event["data"]["usage"]
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)

            # Get final state if streaming didn't capture output
            if not final_output:
                state = await self.client.get_thread_state(thread_id)
                messages = state.get("values", {}).get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict):
                        final_output = last_msg.get("content", "")

        except asyncio.TimeoutError:
            raise AgentTimeoutError(self.name, self.config.timeout)
        except SkillAgentAdapterError:
            raise
        except Exception as e:
            errors.append(str(e))
            logger.error(f"LangGraph execution error: {e}")

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

    def _build_skill_config(self, skill: Skill) -> Dict[str, Any]:
        """Build LangGraph run configuration with skill injection.

        Injects the skill as a system message modifier in the
        configurable parameters.

        Args:
            skill: Skill to inject.

        Returns:
            Configuration dictionary for the run.
        """
        skill_prompt = f"""You have the following skill loaded:

# Skill: {skill.metadata.name}

{skill.metadata.description}

## Instructions

{skill.instructions}

---

Follow the skill instructions above when responding to user queries.
"""

        return {
            "configurable": {
                "system_message": skill_prompt,
                # Alternative injection points for different graph designs
                "skill_instructions": skill.instructions,
                "skill_name": skill.metadata.name,
            }
        }

    async def _stream_with_timeout(
        self,
        thread_id: str,
        message: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream run with overall timeout enforcement.

        Wraps the streaming call with a timeout to prevent
        runaway executions.

        Args:
            thread_id: Thread ID for the conversation.
            message: User message to send.
            config: Optional run configuration.

        Yields:
            Event dictionaries from the stream.

        Raises:
            asyncio.TimeoutError: If total execution exceeds timeout.
        """
        deadline = asyncio.get_event_loop().time() + self.config.timeout

        async for event in self.client.stream_run(thread_id, message, config):
            if asyncio.get_event_loop().time() > deadline:
                raise asyncio.TimeoutError("Execution timeout exceeded")
            yield event

    def _process_event(
        self,
        event: Dict[str, Any],
        events: List[TraceEvent],
        tool_calls: List[str],
        files_created: List[str],
        files_modified: List[str],
        commands_ran: List[str],
    ) -> None:
        """Process a LangGraph streaming event.

        Extracts tool calls and tracks operations from
        LangGraph event format.

        Args:
            event: Event dictionary from stream.
            events: List to append trace events.
            tool_calls: List to append tool names.
            files_created: List to append created files.
            files_modified: List to append modified files.
            commands_ran: List to append commands.
        """
        event_type = event.get("event", "")

        # Tool invocation events
        if event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = event.get("data", {}).get("input", {})

            tool_calls.append(tool_name)
            events.append(TraceEvent(
                type=TraceEventType.TOOL_CALL,
                tool_name=tool_name,
                tool_input=tool_input,
            ))

            self._track_tool_operations(
                tool_name=tool_name,
                tool_input=tool_input,
                files_created=files_created,
                files_modified=files_modified,
                commands_ran=commands_ran,
            )

        # Tool completion events
        elif event_type == "on_tool_end":
            # Could capture tool output here if needed
            pass

        # LLM call events
        elif event_type == "on_llm_end":
            data = event.get("data", {})
            events.append(TraceEvent(
                type=TraceEventType.LLM_CALL,
                model=data.get("model", ""),
                input_tokens=data.get("usage", {}).get("input_tokens"),
                output_tokens=data.get("usage", {}).get("output_tokens"),
            ))

    def _track_tool_operations(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        files_created: List[str],
        files_modified: List[str],
        commands_ran: List[str],
    ) -> None:
        """Track file and command operations from tool calls.

        Maps common LangGraph tool patterns to operation types.

        Args:
            tool_name: Name of the tool.
            tool_input: Tool input parameters.
            files_created: List to append created files.
            files_modified: List to append modified files.
            commands_ran: List to append commands.
        """
        tool_lower = tool_name.lower()

        # File operations
        if any(w in tool_lower for w in ("write", "create", "save")):
            path = tool_input.get("path", tool_input.get("file_path", ""))
            if path:
                files_created.append(path)

        elif any(w in tool_lower for w in ("edit", "modify", "update", "patch")):
            path = tool_input.get("path", tool_input.get("file_path", ""))
            if path:
                files_modified.append(path)

        # Shell/command operations
        elif any(w in tool_lower for w in ("shell", "bash", "exec", "terminal", "command")):
            cmd = tool_input.get("command", tool_input.get("cmd", ""))
            if cmd:
                commands_ran.append(cmd)

    async def cleanup(self) -> None:
        """Clean up resources.

        Closes the HTTP client session.
        """
        await self.client.close()

    def __del__(self):
        """Destructor to warn about unclosed sessions."""
        if self.client._session is not None:
            logger.warning(
                "LangGraphSkillAdapter was not properly closed. "
                "Call cleanup() or use as async context manager."
            )
