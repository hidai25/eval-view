"""Core type definitions for EvalView."""

import logging
from datetime import datetime
from typing import Any, Optional, List, Dict, Union
from pydantic import BaseModel, Field, field_validator, ValidationInfo

logger = logging.getLogger(__name__)


# ============================================================================
# Test Case Types
# ============================================================================


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


class TestCase(BaseModel):
    """Test case definition (loaded from YAML)."""

    __test__ = False  # Tell pytest this is not a test class

    name: str
    description: Optional[str] = None
    input: TestInput
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


class ExecutionTrace(BaseModel):
    """Execution trace captured from agent run."""

    session_id: str
    start_time: datetime
    end_time: datetime
    steps: List[StepTrace]
    final_output: str
    metrics: ExecutionMetrics

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


# ============================================================================
# Evaluation Result Types
# ============================================================================


class CategoryResult(BaseModel):
    """Result for a single tool category check."""

    category: str
    satisfied: bool
    matched_tools: List[str] = Field(default_factory=list)


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


class Evaluations(BaseModel):
    """All evaluation results."""

    tool_accuracy: ToolEvaluation
    sequence_correctness: SequenceEvaluation
    output_quality: OutputEvaluation
    cost: CostEvaluation
    latency: LatencyEvaluation
    hallucination: Optional[HallucinationEvaluation] = None
    safety: Optional[SafetyEvaluation] = None


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


# ============================================================================
# Statistical/Variance Evaluation Types
# ============================================================================


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

    # Individual run results (for detailed analysis)
    individual_results: List[EvaluationResult] = Field(default_factory=list, description="Results from each run")

    # Metadata
    timestamp: datetime = Field(description="When the statistical evaluation completed")
    variance_config: VarianceConfig = Field(description="Configuration used for this evaluation")
