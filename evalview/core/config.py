"""Configuration models for EvalView."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class ScoringWeights(BaseModel):
    """Scoring weights for evaluation components."""

    tool_accuracy: float = Field(default=0.3, ge=0, le=1)
    output_quality: float = Field(default=0.5, ge=0, le=1)
    sequence_correctness: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def validate_sum(self):
        """Ensure weights sum to 1.0."""
        total = self.tool_accuracy + self.output_quality + self.sequence_correctness
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.3f}. "
                f"Current: tool_accuracy={self.tool_accuracy}, "
                f"output_quality={self.output_quality}, "
                f"sequence_correctness={self.sequence_correctness}"
            )
        return self

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for backward compatibility."""
        return {
            "tool_accuracy": self.tool_accuracy,
            "output_quality": self.output_quality,
            "sequence_correctness": self.sequence_correctness,
        }


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(default=0, ge=0, le=10)
    base_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay: float = Field(default=30.0, ge=1.0, le=300.0)
    exponential: bool = True
    jitter: bool = True


class ScoringConfig(BaseModel):
    """Scoring configuration section."""

    weights: ScoringWeights = Field(default_factory=ScoringWeights)


class CIConfig(BaseModel):
    """CI/CD configuration for exit codes and failure handling.

    Example in config.yaml:
        ci:
          fail_on: [REGRESSION, TOOLS_CHANGED]
          warn_on: [OUTPUT_CHANGED]
    """

    fail_on: list = Field(
        default=["REGRESSION"],
        description="Diff statuses that cause exit code 1"
    )
    warn_on: list = Field(
        default=["TOOLS_CHANGED", "OUTPUT_CHANGED"],
        description="Diff statuses that print warning but exit 0"
    )


class DiffConfig(BaseModel):
    """Configuration for diff comparison thresholds.

    Example in config.yaml:
        diff:
          tool_similarity_threshold: 0.85
          output_similarity_threshold: 0.92
          score_regression_threshold: 3.0
          ignore_whitespace: true
          ignore_case_in_output: false
    """

    tool_similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for tool sequences to be considered 'same'"
    )
    output_similarity_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for outputs to be considered 'same'"
    )
    score_regression_threshold: float = Field(
        default=5.0,
        ge=0.0,
        description="Score drop threshold to trigger REGRESSION status"
    )
    ignore_whitespace: bool = Field(
        default=True,
        description="Ignore whitespace differences in output comparison"
    )
    ignore_case_in_output: bool = Field(
        default=False,
        description="Ignore case differences in output comparison"
    )

    # Semantic similarity via embeddings (opt-in — requires OPENAI_API_KEY)
    # When enabled, blends cosine similarity of embeddings with lexical similarity
    # to catch semantic drift that SequenceMatcher misses (e.g. same meaning,
    # different wording after a model update).
    semantic_diff_enabled: bool = Field(
        default=False,
        description=(
            "Enable embedding-based semantic similarity comparison (opt-in). "
            "Requires OPENAI_API_KEY. Adds ~$0.00004 per check."
        ),
    )
    semantic_similarity_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of semantic (embedding) similarity vs lexical similarity. "
            "0.0 = lexical only, 1.0 = semantic only. Default 0.7 means "
            "70% semantic + 30% lexical."
        ),
    )


class EvalViewConfig(BaseModel):
    """Complete EvalView configuration (loaded from config.yaml)."""

    adapter: str
    endpoint: str
    timeout: float = 30.0
    headers: Dict[str, str] = Field(default_factory=dict)
    allow_private_urls: bool = True
    model: Optional[Dict[str, Any]] = None

    # New configuration sections
    scoring: Optional[ScoringConfig] = None
    retry: Optional[RetryConfig] = None
    ci: Optional[CIConfig] = None
    diff: Optional[DiffConfig] = None

    def get_scoring_weights(self) -> ScoringWeights:
        """Get scoring weights with defaults."""
        if self.scoring:
            return self.scoring.weights
        return ScoringWeights()

    def get_retry_config(self) -> RetryConfig:
        """Get retry config with defaults."""
        if self.retry:
            return self.retry
        return RetryConfig()

    def get_ci_config(self) -> CIConfig:
        """Get CI config with defaults."""
        if self.ci:
            return self.ci
        return CIConfig()

    def get_diff_config(self) -> DiffConfig:
        """Get diff config with defaults."""
        if self.diff:
            return self.diff
        return DiffConfig()


# Default weights for backward compatibility
DEFAULT_WEIGHTS = ScoringWeights()
