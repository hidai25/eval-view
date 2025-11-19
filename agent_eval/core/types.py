"""Core type definitions for AgentEval."""

from datetime import datetime
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field


# ============================================================================
# Test Case Types
# ============================================================================


class TestInput(BaseModel):
    """Input for a test case."""

    query: str
    context: Optional[Dict[str, Any]] = None


class ExpectedOutput(BaseModel):
    """Expected output criteria."""

    contains: Optional[List[str]] = None
    not_contains: Optional[List[str]] = None
    json_schema: Optional[Dict[str, Any]] = None


class MetricThreshold(BaseModel):
    """Threshold for a specific metric."""

    value: float
    tolerance: float


class ExpectedBehavior(BaseModel):
    """Expected behavior of the agent."""

    tools: Optional[List[str]] = None
    tool_sequence: Optional[List[str]] = None
    output: Optional[ExpectedOutput] = None
    metrics: Optional[Dict[str, MetricThreshold]] = None


class Thresholds(BaseModel):
    """Performance thresholds for the test."""

    min_score: float = Field(ge=0, le=100)
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None


class TestCase(BaseModel):
    """Test case definition (loaded from YAML)."""

    name: str
    description: Optional[str] = None
    input: TestInput
    expected: ExpectedBehavior
    thresholds: Thresholds


# ============================================================================
# Execution Trace Types
# ============================================================================


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

    latency: float  # in milliseconds
    cost: float  # in dollars
    tokens: Optional[TokenUsage] = None


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


class ExecutionMetrics(BaseModel):
    """Overall execution metrics."""

    total_cost: float
    total_latency: float
    total_tokens: Optional[TokenUsage] = None


class ExecutionTrace(BaseModel):
    """Execution trace captured from agent run."""

    session_id: str
    start_time: datetime
    end_time: datetime
    steps: List[StepTrace]
    final_output: str
    metrics: ExecutionMetrics


# ============================================================================
# Evaluation Result Types
# ============================================================================


class ToolEvaluation(BaseModel):
    """Tool call accuracy evaluation."""

    accuracy: float = Field(ge=0, le=1)
    missing: List[str] = Field(default_factory=list)
    unexpected: List[str] = Field(default_factory=list)
    correct: List[str] = Field(default_factory=list)


class SequenceEvaluation(BaseModel):
    """Tool sequence correctness evaluation."""

    correct: bool
    expected_sequence: List[str]
    actual_sequence: List[str]
    violations: List[str] = Field(default_factory=list)


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


class Evaluations(BaseModel):
    """All evaluation results."""

    tool_accuracy: ToolEvaluation
    sequence_correctness: SequenceEvaluation
    output_quality: OutputEvaluation
    cost: CostEvaluation
    latency: LatencyEvaluation


class EvaluationResult(BaseModel):
    """Complete evaluation result for a test case."""

    test_case: str
    passed: bool
    score: float = Field(ge=0, le=100)
    evaluations: Evaluations
    trace: ExecutionTrace
    timestamp: datetime
