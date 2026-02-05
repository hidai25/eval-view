"""Agent-based skill testing type definitions.

This module defines the data models for testing skills through real AI agents
rather than simple system-prompt-based testing. It supports:
- Multiple agent types (Claude Code, Codex, LangGraph, etc.)
- Two-phase evaluation (deterministic + rubric-based)
- Structured trace capture for debugging

All models follow Pydantic patterns from evalview/core/types.py.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Supported agent types for skill testing.

    Each agent type has a corresponding adapter that knows how to:
    1. Inject the skill into the agent's context
    2. Execute the agent with a user query
    3. Capture structured traces of tool calls, file changes, commands
    """

    CLAUDE_CODE = "claude-code"
    CLAUDE_AGENT_TEAMS = "claude-agent-teams"
    CODEX = "codex"
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    OPENAI_ASSISTANTS = "openai-assistants"
    CUSTOM = "custom"
    SYSTEM_PROMPT = "system-prompt"  # Legacy mode


class TestCategory(str, Enum):
    """Categories for skill tests per OpenAI eval guidelines.

    Testing across categories ensures skills work in realistic scenarios:
    - EXPLICIT: Direct skill invocation (baseline)
    - IMPLICIT: Natural language that implies skill use
    - CONTEXTUAL: Real-world noisy prompts
    - NEGATIVE: Prompts that should NOT trigger the skill
    """

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CONTEXTUAL = "contextual"
    NEGATIVE = "negative"


class AgentConfig(BaseModel):
    """Configuration for the agent used in skill testing.

    Security considerations:
    - cwd is validated to exist and be accessible
    - timeout prevents runaway agent execution
    - max_turns limits conversation depth to prevent loops

    Args:
        type: Agent type (determines which adapter to use)
        config_path: Path to agent-specific configuration file
        tools: Tools to enable (adapter-specific)
        max_turns: Maximum conversation turns (default: 10)
        capture_trace: Whether to save JSONL trace (default: True)
        timeout: Execution timeout in seconds (default: 300)
        cwd: Working directory for agent execution
        env: Additional environment variables
        script_path: For CUSTOM type, path to runner script
    """

    type: AgentType = Field(default=AgentType.SYSTEM_PROMPT)
    config_path: Optional[str] = Field(default=None)
    tools: Optional[List[str]] = Field(default=None)
    max_turns: int = Field(default=10, ge=1, le=100)
    capture_trace: bool = Field(default=True)
    timeout: float = Field(default=300.0, ge=1.0, le=3600.0)
    cwd: Optional[str] = Field(default=None)
    env: Optional[Dict[str, str]] = Field(default=None)
    script_path: Optional[str] = Field(default=None)

    @field_validator("cwd", mode="before")
    @classmethod
    def validate_cwd(cls, v: Optional[str]) -> Optional[str]:
        """Validate working directory exists."""
        if v is not None:
            import os

            expanded = os.path.abspath(os.path.expanduser(v))
            if not os.path.isdir(expanded):
                raise ValueError(f"Working directory does not exist: {v}")
            return expanded
        return v


class SmokeTest(BaseModel):
    """Configuration for a runtime smoke test.

    Smoke tests verify that the generated application actually works
    by running commands and checking their behavior.

    Attributes:
        command: Command to run (e.g., "npm run dev")
        background: Run in background (for servers)
        wait_for: String to wait for in output (for background processes)
        timeout: Timeout in seconds
        health_check: URL to curl for health verification
        expected_status: Expected HTTP status code (default: 200)
        cleanup: Command to run after test (e.g., kill server)
    """

    command: str = Field(description="Command to execute")
    background: bool = Field(default=False, description="Run in background")
    wait_for: Optional[str] = Field(
        default=None, description="Wait for this string in output"
    )
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    health_check: Optional[str] = Field(
        default=None, description="URL to check (e.g., http://localhost:3000)"
    )
    expected_status: int = Field(default=200, ge=100, le=599)
    cleanup: Optional[str] = Field(
        default=None, description="Cleanup command after test"
    )


class DeterministicExpected(BaseModel):
    """Expected behaviors for Phase 1 deterministic checks.

    All checks are optional. Only specified checks are evaluated.
    Checks use substring/subsequence matching for flexibility.

    Tool checks:
        tool_calls_contain: Tools that MUST be called
        tool_calls_not_contain: Tools that MUST NOT be called
        tool_sequence: Tools that must appear in order (subsequence)

    File checks:
        files_created: Files that must be created
        files_modified: Files that must be modified
        files_not_modified: Files that must NOT be modified
        file_contains: {path: [strings]} - strings that must appear in file
        file_not_contains: {path: [strings]} - strings that must NOT appear

    Command checks:
        commands_ran: Commands that must be executed (substring match)
        commands_not_ran: Commands that must NOT be executed
        command_count_max: Maximum shell commands allowed (catch loops)

    Output checks:
        output_contains: Strings in agent's final output
        output_not_contains: Strings NOT in agent's final output

    Token budget checks:
        max_input_tokens: Maximum input tokens allowed
        max_output_tokens: Maximum output tokens allowed
        max_total_tokens: Maximum total tokens (input + output)

    Build verification:
        build_must_pass: List of build commands that must exit with code 0

    Runtime smoke tests:
        smoke_tests: List of smoke test configurations

    Repository cleanliness:
        git_clean: If True, working directory must be clean (no uncommitted changes)

    Permission/security checks:
        forbidden_patterns: Command patterns that must NOT appear (e.g., "sudo", "rm -rf /")
        no_sudo: If True, no sudo commands allowed
        no_network_external: If True, no external network calls allowed
    """

    # Tool checks
    tool_calls_contain: Optional[List[str]] = Field(default=None)
    tool_calls_not_contain: Optional[List[str]] = Field(default=None)
    tool_sequence: Optional[List[str]] = Field(default=None)

    # File checks
    files_created: Optional[List[str]] = Field(default=None)
    files_modified: Optional[List[str]] = Field(default=None)
    files_not_modified: Optional[List[str]] = Field(default=None)
    file_contains: Optional[Dict[str, List[str]]] = Field(default=None)
    file_not_contains: Optional[Dict[str, List[str]]] = Field(default=None)

    # Command checks
    commands_ran: Optional[List[str]] = Field(default=None)
    commands_not_ran: Optional[List[str]] = Field(default=None)
    command_count_max: Optional[int] = Field(default=None, ge=0)

    # Output checks (compatible with legacy SkillExpectedBehavior)
    output_contains: Optional[List[str]] = Field(default=None)
    output_not_contains: Optional[List[str]] = Field(default=None)

    # Token budget checks
    max_input_tokens: Optional[int] = Field(default=None, ge=0)
    max_output_tokens: Optional[int] = Field(default=None, ge=0)
    max_total_tokens: Optional[int] = Field(default=None, ge=0)

    # Build verification
    build_must_pass: Optional[List[str]] = Field(
        default=None,
        description="Build commands that must succeed (exit code 0)",
    )

    # Runtime smoke tests
    smoke_tests: Optional[List[SmokeTest]] = Field(
        default=None,
        description="Runtime smoke tests to verify application works",
    )

    # Repository cleanliness
    git_clean: Optional[bool] = Field(
        default=None,
        description="If True, git working directory must be clean",
    )

    # Permission/security checks
    forbidden_patterns: Optional[List[str]] = Field(
        default=None,
        description="Command patterns that are forbidden (security)",
    )
    no_sudo: Optional[bool] = Field(
        default=None,
        description="If True, no sudo commands allowed",
    )
    no_network_external: Optional[bool] = Field(
        default=None,
        description="If True, block external network calls",
    )


class RubricConfig(BaseModel):
    """Configuration for Phase 2 rubric-based LLM grading.

    Rubric evaluation only runs if:
    1. Phase 1 deterministic checks passed
    2. This config is provided

    Args:
        schema_path: Optional JSON schema for structured scoring
        prompt: Rubric prompt describing evaluation criteria
        min_score: Minimum score to pass (0-100)
        model: Optional model override for evaluation
    """

    schema_path: Optional[str] = Field(default=None)
    prompt: str = Field(min_length=10)
    min_score: float = Field(default=70.0, ge=0, le=100)
    model: Optional[str] = Field(default=None)


class SkillAgentTest(BaseModel):
    """A single test case for agent-based skill testing.

    Each test specifies:
    - An input query to send to the agent
    - Expected behaviors (deterministic checks)
    - Optional rubric for qualitative grading
    - Category for reporting and analysis
    """

    __test__ = False  # pytest compatibility

    name: str = Field(min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=1024)
    input: str = Field(min_length=1)
    category: TestCategory = Field(default=TestCategory.EXPLICIT)
    should_trigger: bool = Field(default=True)
    expected: Optional[DeterministicExpected] = Field(default=None)
    rubric: Optional[RubricConfig] = Field(default=None)

    @model_validator(mode="after")
    def validate_negative_test(self) -> "SkillAgentTest":
        """Negative tests should have should_trigger=False."""
        if self.category == TestCategory.NEGATIVE and self.should_trigger:
            logger.warning(
                f"Test '{self.name}' is NEGATIVE category but should_trigger=True. "
                "Consider setting should_trigger=False."
            )
        return self


class SkillAgentTestSuite(BaseModel):
    """Complete test suite for agent-based skill testing.

    Loaded from YAML, this defines:
    - Which skill to test (path to SKILL.md)
    - Agent configuration
    - All test cases
    - Pass rate threshold
    """

    __test__ = False

    name: str = Field(min_length=1, max_length=128)
    description: Optional[str] = Field(default=None)
    skill: str = Field(description="Path to SKILL.md file")
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tests: List[SkillAgentTest] = Field(min_length=1)
    min_pass_rate: float = Field(default=0.8, ge=0, le=1)

    @field_validator("skill", mode="before")
    @classmethod
    def validate_skill_path(cls, v: str) -> str:
        """Warn if skill path doesn't look valid."""
        if v and not v.endswith(".md"):
            logger.warning(f"Skill path '{v}' doesn't end with .md")
        return v


# ============================================================================
# Trace Types
# ============================================================================


class TraceEventType(str, Enum):
    """Types of events captured in execution trace."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    FILE_CREATE = "file_create"
    FILE_MODIFY = "file_modify"
    FILE_DELETE = "file_delete"
    COMMAND_RUN = "command_run"
    ERROR = "error"


class TraceEvent(BaseModel):
    """A single event in the execution trace.

    Events are polymorphic - different fields populated based on type.
    This enables JSONL serialization where each line is one event.
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    type: TraceEventType

    # Tool call fields
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    tool_success: Optional[bool] = None
    tool_error: Optional[str] = None

    # File operation fields
    file_path: Optional[str] = None
    file_content: Optional[str] = None

    # Command fields
    command: Optional[str] = None
    command_output: Optional[str] = None
    command_exit_code: Optional[int] = None

    # LLM call fields
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class SkillAgentTrace(BaseModel):
    """Complete trace from agent skill execution.

    Contains both raw events and aggregated data for quick access.
    The trace can be serialized to JSONL for debugging.
    """

    session_id: str
    skill_name: str
    test_name: str
    start_time: datetime
    end_time: datetime

    # Raw events
    events: List[TraceEvent] = Field(default_factory=list)

    # Aggregated for quick access (populated from events)
    tool_calls: List[str] = Field(default_factory=list)
    files_created: List[str] = Field(default_factory=list)
    files_modified: List[str] = Field(default_factory=list)
    commands_ran: List[str] = Field(default_factory=list)

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Final output
    final_output: str = ""

    # Errors encountered
    errors: List[str] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Execution duration in milliseconds."""
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0


# ============================================================================
# Evaluation Result Types
# ============================================================================


class DeterministicCheckResult(BaseModel):
    """Result of a single deterministic check."""

    check_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str


class DeterministicEvaluation(BaseModel):
    """Results of all Phase 1 deterministic checks."""

    passed: bool
    score: float = Field(ge=0, le=100)
    checks: List[DeterministicCheckResult] = Field(default_factory=list)
    passed_count: int = 0
    total_count: int = 0

    @property
    def failed_checks(self) -> List[DeterministicCheckResult]:
        """Get only failed checks for debugging."""
        return [c for c in self.checks if not c.passed]


class RubricEvaluation(BaseModel):
    """Results of Phase 2 rubric-based grading."""

    passed: bool
    score: float = Field(ge=0, le=100)
    rationale: str
    min_score: float
    rubric_response: Optional[Dict[str, Any]] = None


class SkillAgentTestResult(BaseModel):
    """Result of running a single agent-based skill test."""

    __test__ = False

    test_name: str
    category: TestCategory
    passed: bool
    score: float = Field(ge=0, le=100)

    # Input/output
    input_query: str
    final_output: str

    # Evaluation results
    deterministic: Optional[DeterministicEvaluation] = None
    rubric: Optional[RubricEvaluation] = None

    # Trace reference (full trace or path to JSONL)
    trace: Optional[SkillAgentTrace] = None
    trace_path: Optional[str] = None

    # Metrics
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    # Error if execution failed
    error: Optional[str] = None

    timestamp: datetime = Field(default_factory=datetime.now)


class SkillAgentTestSuiteResult(BaseModel):
    """Result of running a complete agent-based skill test suite."""

    __test__ = False

    suite_name: str
    skill_name: str
    agent_type: AgentType
    passed: bool

    # Overall stats
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float

    # Stats by category
    by_category: Dict[TestCategory, Dict[str, int]] = Field(default_factory=dict)

    # Individual results
    results: List[SkillAgentTestResult] = Field(default_factory=list)

    # Aggregate metrics
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0

    timestamp: datetime = Field(default_factory=datetime.now)
