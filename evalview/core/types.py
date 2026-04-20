"""Core type definitions for EvalView."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
import re

from evalview.core.observability import (
    AnomalyReportDict,
    CoherenceReportDict,
    TrustReportDict,
)


# --- Enums and Literal Types ---

Difficulty = Literal["trivial", "easy", "medium", "hard", "expert"]

logger = logging.getLogger(__name__)

# Wire-format schema version for cloud payloads. Bumped to 2 when the
# simulation and rationale_events fields were introduced. Sent as
# X-EvalView-Schema header; older clouds that only understand v1 ignore
# the new fields but still accept the core payload.
SCHEMA_VERSION = 2

# Caps for decision-rationale capture. Enforced by the collector in
# evalview/core/rationale.py (Phase 2). Kept here so cloud ingest can
# mirror the same limits in its Zod validator.
RATIONALE_MAX_EVENTS_PER_RUN = 500
RATIONALE_MAX_TEXT_BYTES = 4096


# --- Test Case Types ---


class TestInput(BaseModel):
    """Input for a test case."""

    __test__ = False  # Tell pytest this is not a test class

    query: str
    context: Optional[Dict[str, Any]] = None


class ExpectedOutput(BaseModel):
    """Expected output criteria."""

    contains: Optional[List[str]] = None
    not_contains: Optional[List[str]] = None
    json_schema: Optional[Dict[str, Any]] = None
    regex_patterns: Optional[List[str]] = None
    must_acknowledge_uncertainty: Optional[bool] = None
    no_pii: Optional[bool] = None


class HallucinationCheck(BaseModel):
    """Configuration for hallucination detection."""

    check: bool = False
    allow: bool = False
    confidence_threshold: float = Field(default=0.8, ge=0, le=1)


class SafetyCheck(BaseModel):
    """Configuration for safety evaluation."""

    check: bool = False
    allow_harmful: bool = False
    categories: Optional[List[str]] = None  # violence, hate_speech, etc.
    severity_threshold: str = "medium"  # "low", "medium", "high"


class MetricThreshold(BaseModel):
    """Threshold for a specific metric."""

    value: float
    tolerance: float


class ExpectedBehavior(BaseModel):
    """Expected behavior of the agent."""

    tools: Optional[List[str]] = None
    tool_categories: Optional[List[str]] = None  # Flexible matching by category
    tool_sequence: Optional[List[str]] = None
    sequence: Optional[List[str]] = None  # Alias for tool_sequence
    output: Optional[Union[ExpectedOutput, Dict[str, Any]]] = None
    metrics: Optional[Dict[str, MetricThreshold]] = None
    hallucination: Optional[Union[HallucinationCheck, Dict[str, Any]]] = None
    safety: Optional[Union[SafetyCheck, Dict[str, Any]]] = None

    # Safety contract: tools that must NEVER be called.
    # Any violation is an immediate hard-fail regardless of score.
    # Example: forbidden_tools: [edit_file, bash] for a read-only research agent.
    forbidden_tools: Optional[List[str]] = Field(
        default=None,
        description=(
            "Tools that must never be invoked. "
            "If any forbidden tool appears in the trace the test fails immediately "
            "with score=0, regardless of output quality."
        ),
    )


class ScoringWeightsOverride(BaseModel):
    """Optional per-test scoring weight overrides."""

    tool_accuracy: Optional[float] = Field(default=None, ge=0, le=1)
    output_quality: Optional[float] = Field(default=None, ge=0, le=1)
    sequence_correctness: Optional[float] = Field(default=None, ge=0, le=1)


class VarianceConfig(BaseModel):
    """Configuration for statistical/variance testing mode.

    When enabled, the test runs multiple times and pass/fail is determined
    by statistical thresholds rather than a single run.
    """

    runs: int = Field(default=10, ge=2, le=100, description="Number of times to run the test")
    pass_rate: float = Field(default=0.8, ge=0, le=1, description="Required pass rate (0.0-1.0)")
    min_mean_score: Optional[float] = Field(default=None, ge=0, le=100, description="Minimum mean score across runs")
    max_std_dev: Optional[float] = Field(default=None, ge=0, description="Maximum allowed standard deviation")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for intervals")


class Thresholds(BaseModel):
    """Performance thresholds for the test."""

    min_score: float = Field(ge=0, le=100)
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None

    # Optional: Override global scoring weights for this test
    weights: Optional[ScoringWeightsOverride] = None

    # Optional: Statistical mode configuration
    variance: Optional[VarianceConfig] = None


class ChecksConfig(BaseModel):
    """Enable/disable specific evaluation checks per test."""

    hallucination: bool = True  # Check for hallucinations
    safety: bool = True  # Check for safety issues
    pii: bool = False  # Check for PII leaks (opt-in)


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation test.

    Used inside the ``turns`` list of a multi-turn ``TestCase``.

    Example YAML::

        turns:
          - query: "I want to book a flight to Paris"
            expected:
              tools: [search_flights]
          - query: "Book the cheapest one"
            expected:
              tools: [book_flight, confirm_booking]
              output:
                contains: ["confirmation", "Paris"]
    """

    __test__ = False

    query: str
    expected: Optional[ExpectedBehavior] = None
    context: Optional[Dict[str, Any]] = None


class TestCase(BaseModel):
    """Test case definition (loaded from YAML)."""

    __test__ = False  # Tell pytest this is not a test class

    name: str
    description: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    # Single-turn: provide ``input``.  Multi-turn: provide ``turns`` — ``input``
    # is auto-populated from the first turn before field validation so that all
    # downstream code can access ``test_case.input.query`` unconditionally.
    input: TestInput
    turns: Optional[List["ConversationTurn"]] = None
    expected: ExpectedBehavior
    thresholds: Thresholds

    # Optional: Enable/disable specific checks for this test
    checks: Optional[ChecksConfig] = None

    # Optional: Override global adapter/endpoint for this test
    adapter: Optional[str] = None  # e.g., "langgraph", "tapescope", "http"
    endpoint: Optional[str] = None  # e.g., "http://127.0.0.1:2024"
    adapter_config: Optional[Dict[str, Any]] = None  # Additional adapter settings

    # Optional: Tool definitions for adapters that support them (e.g., Anthropic, OpenAI)
    # Each tool should have: name, description, input_schema
    tools: Optional[List[Dict[str, Any]]] = None

    # Optional: Model override for this specific test
    model: Optional[str] = None  # e.g., "claude-sonnet-4-5-20250929", "gpt-4o"

    # Optional: Suite type for categorization (capability vs regression)
    # - "capability": Tests that measure what the agent CAN do (expect lower pass rates, hill-climbing)
    # - "regression": Tests that verify the agent STILL works (expect ~100% pass rate, safety net)
    # This affects reporting thresholds and how failures are interpreted.
    suite_type: Optional[str] = Field(
        default=None,
        description="Test suite type: 'capability' (hill-climbing) or 'regression' (safety net)"
    )

    # Optional: Difficulty level for benchmarking and filtering
    # - "trivial": Sanity checks, basic functionality
    # - "easy": Simple tasks most agents should handle
    # - "medium": Standard complexity, requires proper tool use
    # - "hard": Complex multi-step reasoning or edge cases
    # - "expert": Near human-expert level tasks
    difficulty: Optional[Difficulty] = Field(
        default=None,
        description="Task difficulty: 'trivial', 'easy', 'medium', 'hard', or 'expert'"
    )

    # Optional: behavior tags used for focused runs and grouped reporting.
    # Examples: tool_use, retrieval, clarification, multi_step
    tags: List[str] = Field(
        default_factory=list,
        description="Behavior tags for filtering and grouped reporting."
    )

    # Optional: confirmation-gate mode. Default is "relaxed", meaning a
    # failure must persist into a second monitor cycle before it fires an
    # alert — this suppresses single-cycle flakes. Set to "strict" for
    # safety-critical tests (auth, payments, PII, refund flows, etc.)
    # where even a one-cycle blip is worth investigating immediately.
    # Strict tests bypass the ConfirmationGate and alert on n=1.
    gate: Optional[str] = Field(
        default=None,
        description=(
            "Confirmation-gate mode: 'strict' (alert on n=1, bypass the "
            "gate — use for safety-critical tests) or omit/None for "
            "relaxed (wait one cycle to confirm before alerting)."
        ),
    )

    # Set to True by evalview init auto-generation. Never set by users.
    # Enables strict quality gating — generated tests with bad queries are
    # skipped before running so they don't pollute agent scores.
    generated: Optional[bool] = Field(
        default=None,
        description="True when this test was auto-generated by evalview init. User-written tests omit this field."
    )

    # Populated by the loader so commands can update the underlying YAML file.
    source_file: Optional[str] = Field(default=None, exclude=True)

    # Simulation mocks. When present, ``evalview simulate`` runs the
    # test hermetically by intercepting matching tool calls and serving
    # the mock response instead of hitting the real tool_executor.
    # Ignored by ``evalview check`` and ``evalview snapshot``.
    mocks: Optional["MockSpec"] = None

    # ----- Computed properties -----
    @property
    def is_multi_turn(self) -> bool:
        """True when this test has multiple conversation turns."""
        return bool(self.turns)

    # ----- Validators -----
    @model_validator(mode="before")
    @classmethod
    def _populate_input_from_first_turn(cls, data: Any) -> Any:
        """Pre-populate ``input`` from ``turns[0]`` before field validation.

        This runs before Pydantic validates individual fields, so ``input``
        arrives as a real value and its type stays ``TestInput`` (not Optional).
        Downstream code can therefore access ``test_case.input.query`` safely
        without None guards.
        """
        if not isinstance(data, dict):
            return data
        turns = data.get("turns")
        if turns and "input" not in data:
            first = turns[0]
            if isinstance(first, dict):
                data = {**data, "input": {"query": first["query"], "context": first.get("context")}}
            elif hasattr(first, "query"):
                data = {**data, "input": {"query": first.query, "context": getattr(first, "context", None)}}
        return data

    @model_validator(mode="after")
    def _validate_turns(self) -> "TestCase":
        """Enforce multi-turn constraints after all fields are set."""
        if self.turns is not None and len(self.turns) < 2:
            raise ValueError("Multi-turn tests require at least 2 turns")
        return self

    @field_validator("name", mode="before")
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        if not re.match(r"^[a-zA-Z0-9 _\-\.]+$", v):
            raise ValueError(
                "name must contain only alphanumeric characters, spaces, hyphens, underscores, and dots"
            )
        return v

    @field_validator("suite_type", mode="before")
    def validate_suite_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"capability", "regression"}
        if v not in allowed:
            raise ValueError(
                f"suite_type must be either None or one of: {', '.join(allowed)}"
            )
        return v

    @field_validator("adapter", mode="before")
    def validate_adapter(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {
            "http", "langgraph", "anthropic", "openai", "ollama", "crewai",
            "tapescope", "openai-assistants", "streaming", "huggingface", "goose", "mcp", "cohere", "mistral", "opencode", "aider",
        }
        if v not in allowed:
            raise ValueError(
                f"adapter must be either None or one of: {', '.join(allowed)}"
            )
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            values = [v]
        elif isinstance(v, list):
            values = v
        else:
            raise ValueError("tags must be a list of strings")

        normalized: List[str] = []
        for item in values:
            if not isinstance(item, str):
                raise ValueError("tags must contain only strings")
            tag = item.strip().lower()
            if not tag:
                continue
            normalized.append(tag)

        return sorted(set(normalized))

# --- Rationale Capture Types ---
# Structured decision-rationale logging. Optional adapter hooks populate
# these; local HTML replay renders them inline; cloud persists them for
# cross-run analytics. See docs/rationale.md (Phase 2) for the capture
# contract. Caps are advisory here and enforced at collection time.


DecisionType = Literal["tool_choice", "branch", "refusal", "retry"]


class RationaleEvent(BaseModel):
    """A single decision the agent made during execution.

    Captured by adapter hooks (Anthropic ``thinking`` blocks, OpenAI
    reasoning summaries, LangGraph node transitions, etc.). Grouped
    across runs by ``input_hash`` so cloud analytics can answer
    "every time the agent saw this state, which branch did it take?".
    """

    step_id: str = Field(description="Stable id shared with the corresponding StepTrace/Span.")
    turn: Optional[int] = Field(
        default=None,
        description="Turn index for multi-turn tests; None for single-turn.",
    )
    decision_type: DecisionType = Field(
        description="What kind of decision this was: tool_choice, branch, refusal, retry."
    )
    chosen: str = Field(description="The option the agent picked (tool name, branch label, etc.)")
    alternatives: List[str] = Field(
        default_factory=list,
        description="Other options the agent could have taken. Empty when unknown.",
    )
    rationale_text: Optional[str] = Field(
        default=None,
        description=(
            f"Free-form reasoning, typically from CoT/thinking/reasoning fields. "
            f"Truncated to {RATIONALE_MAX_TEXT_BYTES} bytes by the collector."
        ),
    )
    input_hash: str = Field(
        description=(
            "sha256 of normalized (prompt + tool-state). Used for cross-run grouping. "
            "Identical hashes across runs identify 'same situation, different decisions'."
        ),
    )
    model_reported_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence 0.0-1.0 when the model supplies it; None otherwise.",
    )
    truncated: bool = Field(
        default=False,
        description="True when the collector truncated rationale_text to fit the cap.",
    )


# --- Simulation Types ---
# Pre-deployment simulation harness. OSS owns the engine entirely. The
# simulator wraps an adapter and serves mocks for tool calls, LLM
# responses, and outbound HTTP so tests can run hermetically and
# what-if scenarios can be explored before shipping.


class ToolMock(BaseModel):
    """Intercepts a named tool invocation.

    Matching is exact on ``tool`` plus optional ``match_params`` (subset-
    match on parameters). Unmatched tool calls fall through to the real
    adapter unless ``strict`` is set on the parent MockSpec.
    """

    tool: str = Field(description="Tool name to intercept (exact match).")
    match_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional parameter subset to match. Omit to match any call to the tool.",
    )
    returns: Any = Field(description="Value returned to the agent in place of the real tool result.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Simulated latency for realism.")
    error: Optional[str] = Field(
        default=None,
        description="If set, the mock raises this error instead of returning a value.",
    )


class ResponseMock(BaseModel):
    """Intercepts an LLM completion.

    Matched on ``match_prompt`` (substring or regex when ``regex=True``).
    ``returns`` replaces the full completion string.
    """

    match_prompt: str = Field(description="Substring or regex to match against the prompt.")
    regex: bool = Field(default=False, description="Treat match_prompt as a regex when true.")
    returns: str = Field(description="Completion text to return to the agent.")
    finish_reason: str = Field(default="stop", description="Finish reason surfaced to the caller.")


class HttpMock(BaseModel):
    """Intercepts outbound HTTP calls from tools or the agent runtime."""

    url_pattern: str = Field(description="URL substring or regex to match.")
    regex: bool = Field(default=False, description="Treat url_pattern as a regex when true.")
    method: Optional[str] = Field(default=None, description="HTTP method filter (GET/POST/...). None matches any.")
    status: int = Field(default=200, ge=100, le=599)
    body: Any = Field(default=None, description="Response body (string or JSON-serializable).")
    headers: Dict[str, str] = Field(default_factory=dict)


class MockSpec(BaseModel):
    """Full mock configuration for a simulated test run.

    Loaded from the ``mocks:`` section of a test YAML:

        mocks:
          seed: 42
          strict: false
          tool_mocks:
            - tool: search_flights
              returns: [{"id": "FL123", "price": 299}]
          response_mocks:
            - match_prompt: "summarize"
              returns: "Summary: ..."
          http_mocks:
            - url_pattern: api.example.com
              status: 503
    """

    seed: int = Field(default=0, description="Deterministic RNG seed for reproducibility.")
    strict: bool = Field(
        default=False,
        description=(
            "When true, any tool/LLM/HTTP call that doesn't match a mock raises. "
            "When false (default), unmatched calls fall through to the real adapter."
        ),
    )
    tool_mocks: List[ToolMock] = Field(default_factory=list)
    response_mocks: List[ResponseMock] = Field(default_factory=list)
    http_mocks: List[HttpMock] = Field(default_factory=list)


class AppliedMock(BaseModel):
    """Record of a mock that was actually triggered during simulation."""

    kind: Literal["tool", "response", "http"]
    matcher: str = Field(description="The tool name, prompt pattern, or URL pattern that matched.")
    count: int = Field(default=1, ge=1, description="How many times this mock fired in the run.")


class BranchExploration(BaseModel):
    """A single branch the simulator explored.

    For what-if runs with ``--variants N``, the simulator can fan out at
    each decision point and record every path taken. Each branch carries
    the chain of chosen tools and the final output.
    """

    branch_id: str
    parent_branch_id: Optional[str] = None
    decision_path: List[str] = Field(
        default_factory=list,
        description="Ordered list of (step_id:chosen) tokens describing the path.",
    )
    final_output: Optional[str] = None
    passed: Optional[bool] = None


class VariantOutcome(BaseModel):
    """Aggregate outcome for one variant of a simulated test."""

    variant_index: int = Field(ge=0)
    branch_id: str
    passed: bool
    score: Optional[float] = Field(default=None, ge=0, le=100)
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    notes: Optional[str] = None


class SimulationResult(BaseModel):
    """Payload attached to an EvaluationResult when run under ``evalview simulate``.

    Cloud receives this under the ``simulation`` key alongside the
    existing GateResult fields. Cloud stores ``mocks_applied`` and
    ``branches_explored`` in the ``simulations`` table and renders them
    on the simulation tab; it never runs simulations itself.
    """

    seed: int = 0
    mocks_applied: List[AppliedMock] = Field(default_factory=list)
    branches_explored: List[BranchExploration] = Field(default_factory=list)
    variant_outcomes: List[VariantOutcome] = Field(default_factory=list)


# --- Execution Trace Types ---


class TokenUsage(BaseModel):
    """Token usage breakdown."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens + self.cached_tokens


class StepMetrics(BaseModel):
    """Metrics for a single step."""

    latency: float = 0.0  # in milliseconds (default to 0.0 for flexibility)
    cost: float = 0.0  # in dollars (default to 0.0 for flexibility)
    tokens: Optional[TokenUsage] = None

    @field_validator("latency", "cost", mode="before")
    @classmethod
    def coerce_to_float(cls, v, info: ValidationInfo):
        """Convert None or invalid values to 0.0 with DEBUG logging."""
        if v is None:
            logger.debug(f"Coerced {info.field_name} from None to 0.0")
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(
                f"Expected numeric value for {info.field_name}, got {type(v).__name__}: {v}. "
                f"Ensure your adapter returns numeric values for metrics."
            )

    @field_validator("tokens", mode="before")
    @classmethod
    def coerce_tokens(cls, v):
        """Convert int/dict to TokenUsage with DEBUG logging."""
        if v is None:
            return None
        if isinstance(v, int):
            logger.debug(f"Coerced tokens from int ({v}) to TokenUsage(output_tokens={v})")
            return TokenUsage(output_tokens=v)
        if isinstance(v, dict):
            logger.debug("Coerced tokens from dict to TokenUsage")
            return TokenUsage(**v)
        if isinstance(v, TokenUsage):
            return v
        raise ValueError(
            f"tokens must be TokenUsage, dict, or int, got {type(v).__name__}. "
            f"Example: {{'input_tokens': 100, 'output_tokens': 200}}"
        )


class StepTrace(BaseModel):
    """Trace of a single agent step."""

    step_id: str
    step_name: str
    tool_name: str
    parameters: Dict[str, Any]
    output: Any
    success: bool
    error: Optional[str] = None
    metrics: StepMetrics
    turn_index: Optional[int] = None
    turn_query: Optional[str] = None


class ExecutionMetrics(BaseModel):
    """Overall execution metrics."""

    total_cost: float
    total_latency: float
    total_tokens: Optional[TokenUsage] = None

    @field_validator("total_tokens", mode="before")
    @classmethod
    def coerce_total_tokens(cls, v):
        """Convert int/dict to TokenUsage with DEBUG logging."""
        if v is None:
            return None
        if isinstance(v, int):
            logger.debug(
                f"Coerced total_tokens from int ({v}) to TokenUsage(output_tokens={v})"
            )
            return TokenUsage(output_tokens=v)
        if isinstance(v, dict):
            logger.debug("Coerced total_tokens from dict to TokenUsage")
            return TokenUsage(**v)
        if isinstance(v, TokenUsage):
            return v
        raise ValueError(
            f"total_tokens must be TokenUsage, dict, or int, got {type(v).__name__}. "
                f"Check your adapter's _calculate_metrics() method."
            )


class TurnTrace(BaseModel):
    """Summary of a single conversation turn in a multi-turn execution."""

    index: int
    query: str
    output: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    cost: float = 0.0
    evaluation: Optional["TurnEvaluation"] = None


class TurnEvaluation(BaseModel):
    """Per-turn evaluation results (diagnostic — does not affect overall score)."""

    turn_index: int
    passed: bool
    tool_accuracy: Optional[float] = None          # 0.0–1.0
    forbidden_violations: List[str] = Field(default_factory=list)
    contains_passed: List[str] = Field(default_factory=list)
    contains_failed: List[str] = Field(default_factory=list)
    not_contains_passed: List[str] = Field(default_factory=list)
    not_contains_failed: List[str] = Field(default_factory=list)
    details: str = ""


# --- Tracing Types (OpenTelemetry-aligned) ---


class SpanKind(str, Enum):
    """Type of span in the execution trace.

    Aligned with OpenTelemetry GenAI semantic conventions.
    """

    AGENT = "agent"  # Top-level agent execution
    LLM = "llm"  # LLM inference call
    TOOL = "tool"  # Tool/function execution


class LLMCallInfo(BaseModel):
    """Information about an LLM inference call."""

    model: str
    provider: str
    prompt: Optional[str] = None
    completion: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None


class ToolCallInfo(BaseModel):
    """Information about a tool/function execution."""

    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None


class Span(BaseModel):
    """A single span in the execution trace.

    Follows OpenTelemetry GenAI semantic conventions for future
    compatibility with OTEL export.
    """

    span_id: str
    parent_span_id: Optional[str] = None
    trace_id: str
    kind: SpanKind
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: Literal["ok", "error", "unset"] = "unset"
    error_message: Optional[str] = None
    llm: Optional[LLMCallInfo] = None  # Populated for SpanKind.LLM
    tool: Optional[ToolCallInfo] = None  # Populated for SpanKind.TOOL
    cost: float = 0.0

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def coerce_span_datetime(cls, v, info: ValidationInfo):
        """Convert ISO string to datetime."""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(
                    f"Invalid datetime format for {info.field_name}: {v}. "
                    f"Use ISO format (YYYY-MM-DDTHH:MM:SS) or datetime object."
                )
        return v


class TraceContext(BaseModel):
    """Complete trace context for an agent execution.

    Contains all spans (LLM calls, tool executions) organized
    in a hierarchical structure for debugging and visualization.
    """

    trace_id: str
    root_span_id: str
    spans: List[Span] = Field(default_factory=list)
    start_time: datetime
    end_time: Optional[datetime] = None
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_cost: float = 0.0

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def coerce_trace_datetime(cls, v, info: ValidationInfo):
        """Convert ISO string to datetime."""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(
                    f"Invalid datetime format for {info.field_name}: {v}. "
                    f"Use ISO format (YYYY-MM-DDTHH:MM:SS) or datetime object."
                )
        return v


class ExecutionTrace(BaseModel):
    """Execution trace captured from agent run."""

    session_id: str
    start_time: datetime
    end_time: datetime
    steps: List[StepTrace]
    final_output: str
    metrics: ExecutionMetrics

    # Optional detailed trace context with LLM call spans
    # Defaults to None for backward compatibility with existing adapters
    trace_context: Optional[TraceContext] = None

    # Model fingerprint — captured from the API response by the adapter.
    # Optional so adapters that don't capture this field still work.
    model_id: Optional[str] = None        # e.g. "claude-3-5-sonnet-20241022"
    model_provider: Optional[str] = None  # e.g. "anthropic"
    turns: Optional[List[TurnTrace]] = None

    # Structured decision rationales captured by adapter hooks. Empty by
    # default; populated when the adapter supports it (Anthropic
    # thinking, OpenAI reasoning summary, LangGraph node transitions).
    # Caps enforced by evalview/core/rationale.py at collection time.
    rationale_events: List[RationaleEvent] = Field(default_factory=list)

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def coerce_datetime(cls, v, info: ValidationInfo):
        """Convert ISO string to datetime with DEBUG logging."""
        if isinstance(v, str):
            try:
                # Handle ISO format with optional timezone
                result = datetime.fromisoformat(v.replace("Z", "+00:00"))
                logger.debug(f"Coerced {info.field_name} from string to datetime")
                return result
            except ValueError:
                raise ValueError(
                    f"Invalid datetime format for {info.field_name}: {v}. "
                    f"Use ISO format (YYYY-MM-DDTHH:MM:SS) or datetime object."
                )
        return v


# --- Evaluation Result Types ---


class CategoryResult(BaseModel):
    """Result for a single tool category check."""

    category: str
    satisfied: bool
    matched_tools: List[str] = Field(default_factory=list)


class ReasonCode(BaseModel):
    """Structured reason code for evaluation failures.

    Provides actionable, machine-readable error codes with helpful guidance.
    """

    code: str = Field(description="Error code (e.g., 'TOOL_MISSING', 'PARAM_TYPE_MISMATCH')")
    severity: Literal["error", "warning", "info"] = Field(description="Severity level")
    message: str = Field(description="Human-readable description of the issue")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    remediation: Optional[str] = Field(default=None, description="Suggested fix or next steps")


class ToolEvaluation(BaseModel):
    """Tool call accuracy evaluation."""

    accuracy: float = Field(ge=0, le=1)
    missing: List[str] = Field(default_factory=list)
    unexpected: List[str] = Field(default_factory=list)
    correct: List[str] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list, description="Helpful hints for fixing mismatches")
    # Category-based evaluation results
    category_results: List[CategoryResult] = Field(default_factory=list)
    categories_satisfied: int = 0
    categories_total: int = 0
    # Structured reason codes (new)
    reason_codes: List[ReasonCode] = Field(default_factory=list, description="Structured error reasons")


class SequenceEvaluation(BaseModel):
    """Tool sequence correctness evaluation."""

    correct: bool
    expected_sequence: List[str]
    actual_sequence: List[str]
    violations: List[str] = Field(default_factory=list)
    # Progress score for partial credit (0.0 to 1.0)
    # - 1.0 = perfect sequence match
    # - 0.0 = no expected tools found in order
    # - 0.6 = found 3 of 5 expected tools in order
    progress_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Partial credit score: proportion of expected sequence completed"
    )
    # Structured reason codes (new)
    reason_codes: List[ReasonCode] = Field(default_factory=list, description="Structured error reasons")


class ContainsChecks(BaseModel):
    """Results of contains/not_contains checks."""

    passed: List[str] = Field(default_factory=list)
    failed: List[str] = Field(default_factory=list)


class OutputEvaluation(BaseModel):
    """Output quality evaluation."""

    score: float = Field(ge=0, le=100)
    rationale: str
    contains_checks: ContainsChecks
    not_contains_checks: ContainsChecks


class CostBreakdown(BaseModel):
    """Cost breakdown by step."""

    step_id: str
    cost: float


class CostEvaluation(BaseModel):
    """Cost threshold evaluation."""

    total_cost: float
    threshold: float
    passed: bool
    breakdown: List[CostBreakdown] = Field(default_factory=list)


class LatencyBreakdown(BaseModel):
    """Latency breakdown by step."""

    step_id: str
    latency: float


class LatencyEvaluation(BaseModel):
    """Latency threshold evaluation."""

    total_latency: float
    threshold: float
    passed: bool
    breakdown: List[LatencyBreakdown] = Field(default_factory=list)


class HallucinationEvaluation(BaseModel):
    """Hallucination detection evaluation."""

    has_hallucination: bool
    confidence: float = Field(ge=0, le=1)
    details: str
    passed: bool  # True if no hallucination or allowed


class SafetyEvaluation(BaseModel):
    """Safety evaluation."""

    is_safe: bool
    categories_flagged: List[str] = Field(default_factory=list)
    severity: str  # "safe", "low", "medium", "high"
    details: str
    passed: bool  # True if safe or harmful content is allowed

class PIIEvaluation(BaseModel):
    """PII detection evaluation."""

    has_pii: bool
    types_detected: List[str] = Field(default_factory=list)
    details: str
    passed: bool  # True if no PII or PII is allowed

class ForbiddenToolEvaluation(BaseModel):
    """Evaluation of the forbidden_tools safety contract.

    A forbidden tool violation is a hard-fail condition: the test receives
    score=0 and passed=False regardless of output quality or other metrics.
    This is intentionally strict — forbidden tools represent security or
    contract boundaries that must never be crossed.
    """

    violations: List[str] = Field(
        default_factory=list,
        description="Forbidden tool names that were actually invoked.",
    )
    passed: bool = Field(
        description="True only when zero forbidden tools were called.",
    )


class Evaluations(BaseModel):
    """All evaluation results."""

    tool_accuracy: ToolEvaluation
    sequence_correctness: SequenceEvaluation
    output_quality: OutputEvaluation
    cost: CostEvaluation
    latency: LatencyEvaluation
    hallucination: Optional[HallucinationEvaluation] = None
    safety: Optional[SafetyEvaluation] = None
    # Present only when the test case declares forbidden_tools.
    forbidden_tools: Optional[ForbiddenToolEvaluation] = None
    pii: Optional[PIIEvaluation] = None


class EvaluationResult(BaseModel):
    """Complete evaluation result for a test case."""

    test_case: str
    passed: bool
    score: float = Field(ge=0, le=100)
    evaluations: Evaluations
    trace: ExecutionTrace
    timestamp: datetime

    # Adapter info for dynamic display
    adapter_name: Optional[str] = None  # e.g., "langgraph", "crewai", "tapescope"

    # Threshold info for failure reporting
    min_score: Optional[float] = None  # The minimum score threshold from test case

    # User-facing fields for reports
    input_query: Optional[str] = None
    actual_output: Optional[str] = None

    # Suite type for categorization (capability vs regression)
    suite_type: Optional[str] = None  # "capability" or "regression"

    # Difficulty level for benchmarking
    difficulty: Optional[Difficulty] = None

    # Per-turn evaluation results for multi-turn tests
    turn_evaluations: Optional[List[TurnEvaluation]] = None

    # Behavioral anomaly detection results (tool loops, stalls, brittle recovery).
    anomaly_report: Optional[AnomalyReportDict] = None

    # Benchmark trust / anti-gaming check results.
    trust_report: Optional[TrustReportDict] = None

    # Cross-turn coherence analysis (context amnesia, contradictions).
    coherence_report: Optional[CoherenceReportDict] = None

    # Simulation payload. Populated only when the test ran under
    # ``evalview simulate``; None for normal check runs.
    simulation: Optional[SimulationResult] = None


# --- Statistical/Variance Evaluation Types ---


class StatisticalMetrics(BaseModel):
    """Statistical metrics computed across multiple test runs."""

    mean: float = Field(description="Mean value")
    std_dev: float = Field(description="Standard deviation")
    variance: float = Field(description="Variance (std_dev squared)")
    min_value: float = Field(description="Minimum value")
    max_value: float = Field(description="Maximum value")
    median: float = Field(description="Median (50th percentile)")
    percentile_25: float = Field(description="25th percentile")
    percentile_75: float = Field(description="75th percentile")
    percentile_95: float = Field(description="95th percentile")
    confidence_interval_lower: float = Field(description="Lower bound of confidence interval")
    confidence_interval_upper: float = Field(description="Upper bound of confidence interval")
    confidence_level: float = Field(default=0.95, description="Confidence level used")


class FlakinessScore(BaseModel):
    """Flakiness assessment for a test based on variance analysis."""

    score: float = Field(ge=0, le=1, description="Flakiness score (0=stable, 1=highly flaky)")
    category: str = Field(description="stable, low_variance, moderate_variance, high_variance, flaky")
    pass_rate: float = Field(ge=0, le=1, description="Proportion of runs that passed")
    score_coefficient_of_variation: float = Field(description="CV of scores (std_dev/mean)")
    output_consistency: Optional[float] = Field(default=None, description="How consistent outputs are (0-1)")
    contributing_factors: List[str] = Field(default_factory=list, description="Factors contributing to flakiness")


class StatisticalEvaluationResult(BaseModel):
    """Complete statistical evaluation result for a test case run multiple times."""

    test_case: str
    passed: bool = Field(description="Whether the test passed statistical thresholds")
    total_runs: int = Field(description="Number of test executions")
    successful_runs: int = Field(description="Number of runs that passed individually")
    failed_runs: int = Field(description="Number of runs that failed individually")

    # Statistical metrics for key measures
    score_stats: StatisticalMetrics = Field(description="Statistics for overall scores")
    cost_stats: Optional[StatisticalMetrics] = Field(default=None, description="Statistics for cost")
    latency_stats: Optional[StatisticalMetrics] = Field(default=None, description="Statistics for latency")

    # Flakiness assessment
    flakiness: FlakinessScore = Field(description="Flakiness assessment")

    # Pass/fail reasoning
    pass_rate: float = Field(ge=0, le=1, description="Proportion of individual runs that passed")
    required_pass_rate: float = Field(ge=0, le=1, description="Required pass rate threshold")
    failure_reasons: List[str] = Field(default_factory=list, description="Reasons for statistical failure")

    # Industry-standard reliability metrics
    # pass@k: P(at least 1 success in k trials) = 1 - (1 - pass_rate)^k
    # Answers: "Will it work if I give it a few tries?"
    pass_at_k: float = Field(
        ge=0, le=1,
        description="Probability of at least one success in k trials. High pass@k means 'it usually finds a solution eventually'"
    )
    # pass^k: P(all k trials succeed) = pass_rate^k
    # Answers: "Will it work reliably every time?"
    pass_power_k: float = Field(
        ge=0, le=1,
        description="Probability of all k trials succeeding. High pass^k means 'it works consistently'"
    )

    # Individual run results (for detailed analysis)
    individual_results: List[EvaluationResult] = Field(default_factory=list, description="Results from each run")

    # Metadata
    timestamp: datetime = Field(description="When the statistical evaluation completed")
    variance_config: VarianceConfig = Field(description="Configuration used for this evaluation")
