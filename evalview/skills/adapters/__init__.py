"""Skill agent adapters for executing skills through various AI agents.

This module provides adapters for testing skills through real AI agents
rather than simple system-prompt-based testing.

Available Adapters:
    - ClaudeCodeAdapter: Claude Code CLI
    - CodexAdapter: OpenAI Codex CLI
    - LangGraphSkillAdapter: LangGraph SDK/Cloud
    - CrewAISkillAdapter: CrewAI multi-agent framework
    - OpenAIAssistantsSkillAdapter: OpenAI Assistants API
    - CustomAdapter: User-provided scripts

Usage:
    from evalview.skills.adapters import SkillAdapterRegistry
    from evalview.skills.agent_types import AgentConfig, AgentType

    config = AgentConfig(type=AgentType.CLAUDE_CODE)
    adapter = SkillAdapterRegistry.create(config)
    trace = await adapter.execute(skill, "Your query")
"""

from evalview.skills.adapters.base import (
    SkillAgentAdapter,
    SkillAgentAdapterError,
    AgentNotFoundError,
    AgentTimeoutError,
)
from evalview.skills.adapters.registry import SkillAdapterRegistry, get_skill_adapter

# Import concrete adapters with graceful fallback for optional dependencies
try:
    from evalview.skills.adapters.claude_code_adapter import ClaudeCodeAdapter
except ImportError:
    ClaudeCodeAdapter = None  # type: ignore

try:
    from evalview.skills.adapters.codex_adapter import CodexAdapter
except ImportError:
    CodexAdapter = None  # type: ignore

try:
    from evalview.skills.adapters.langgraph_adapter import LangGraphSkillAdapter
except ImportError:
    LangGraphSkillAdapter = None  # type: ignore

try:
    from evalview.skills.adapters.crewai_adapter import CrewAISkillAdapter
except ImportError:
    CrewAISkillAdapter = None  # type: ignore

try:
    from evalview.skills.adapters.openai_assistants_adapter import (
        OpenAIAssistantsSkillAdapter,
    )
except ImportError:
    OpenAIAssistantsSkillAdapter = None  # type: ignore

try:
    from evalview.skills.adapters.custom_adapter import CustomAdapter
except ImportError:
    CustomAdapter = None  # type: ignore

__all__ = [
    # Base classes and exceptions
    "SkillAgentAdapter",
    "SkillAgentAdapterError",
    "AgentNotFoundError",
    "AgentTimeoutError",
    # Registry
    "SkillAdapterRegistry",
    "get_skill_adapter",
    # Concrete adapters (may be None if dependencies missing)
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "LangGraphSkillAdapter",
    "CrewAISkillAdapter",
    "OpenAIAssistantsSkillAdapter",
    "CustomAdapter",
]
