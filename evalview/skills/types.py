"""Type definitions for skills testing."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field, field_validator


class SkillSeverity(str, Enum):
    """Severity level for validation errors."""

    ERROR = "error"  # Must fix - skill won't work
    WARNING = "warning"  # Should fix - may cause issues
    INFO = "info"  # Suggestion for improvement


class SkillValidationError(BaseModel):
    """A single validation error or warning."""

    code: str = Field(description="Error code (e.g., 'MISSING_NAME', 'INVALID_TRIGGER')")
    message: str = Field(description="Human-readable error message")
    severity: SkillSeverity = Field(default=SkillSeverity.ERROR)
    line: Optional[int] = Field(default=None, description="Line number if applicable")
    suggestion: Optional[str] = Field(default=None, description="How to fix this issue")


class SkillMetadata(BaseModel):
    """Metadata from SKILL.md frontmatter."""

    # Required fields
    name: str = Field(description="Unique identifier for the skill (lowercase, hyphens)")
    description: str = Field(description="What the skill does and when to use it")

    # Optional fields (based on extended spec)
    version: Optional[str] = Field(default=None, description="Skill version")
    author: Optional[str] = Field(default=None, description="Skill author")
    triggers: Optional[List[str]] = Field(
        default=None, description="Keywords that activate the skill"
    )
    tools: Optional[List[str]] = Field(default=None, description="Tools this skill uses")
    model_requirements: Optional[List[str]] = Field(
        default=None, description="Required model capabilities"
    )

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v):
        """Validate skill name format."""
        if not v:
            raise ValueError("Skill name is required")
        if not isinstance(v, str):
            raise ValueError("Skill name must be a string")
        # Name should be lowercase with hyphens
        v = str(v).strip()
        if not v:
            raise ValueError("Skill name cannot be empty")
        return v

    @field_validator("description", mode="before")
    @classmethod
    def validate_description(cls, v):
        """Validate skill description."""
        if not v:
            raise ValueError("Skill description is required")
        if not isinstance(v, str):
            raise ValueError("Skill description must be a string")
        v = str(v).strip()
        if not v:
            raise ValueError("Skill description cannot be empty")
        if len(v) < 10:
            raise ValueError("Skill description should be at least 10 characters")
        return v


class Skill(BaseModel):
    """A parsed Claude Code skill."""

    # Parsed content
    metadata: SkillMetadata = Field(description="Frontmatter metadata")
    instructions: str = Field(description="Markdown instructions body")
    raw_content: str = Field(description="Original file content")

    # File info
    file_path: Optional[str] = Field(default=None, description="Path to SKILL.md file")

    # Computed properties
    @property
    def token_estimate(self) -> int:
        """Rough estimate of tokens when skill is loaded."""
        # ~4 chars per token is a rough estimate
        return len(self.instructions) // 4

    @property
    def is_lightweight(self) -> bool:
        """Check if skill is under the ~5k token recommendation."""
        return self.token_estimate < 5000


class SkillValidationResult(BaseModel):
    """Result of validating a skill."""

    valid: bool = Field(description="True if no errors (warnings allowed)")
    skill: Optional[Skill] = Field(default=None, description="Parsed skill if valid")
    errors: List[SkillValidationError] = Field(default_factory=list)
    warnings: List[SkillValidationError] = Field(default_factory=list)
    info: List[SkillValidationError] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def total_issues(self) -> int:
        """Total number of issues found."""
        return len(self.errors) + len(self.warnings) + len(self.info)


# ============================================================================
# Skill Test Types
# ============================================================================


class SkillTestInput(BaseModel):
    """Input for a skill test case."""

    query: str = Field(description="User query that should trigger the skill")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class SkillExpectedBehavior(BaseModel):
    """Expected behavior when skill is invoked."""

    should_activate: bool = Field(default=True, description="Should this query activate the skill?")
    output_contains: Optional[List[str]] = Field(
        default=None, description="Strings that should appear in output"
    )
    output_not_contains: Optional[List[str]] = Field(
        default=None, description="Strings that should NOT appear"
    )
    tone: Optional[str] = Field(
        default=None, description="Expected tone: professional, casual, technical, etc."
    )
    max_length: Optional[int] = Field(
        default=None, description="Maximum response length in characters"
    )


class SkillTest(BaseModel):
    """A single test within a skill test suite."""

    __test__ = False  # Tell pytest this is not a test class

    name: str = Field(description="Test name")
    description: Optional[str] = Field(default=None)
    input: str = Field(description="User query to test")
    expected: SkillExpectedBehavior


class SkillTestSuite(BaseModel):
    """A complete test suite for a skill (loaded from YAML)."""

    __test__ = False  # Tell pytest this is not a test class

    name: str = Field(description="Test suite name")
    description: Optional[str] = Field(default=None)
    skill: str = Field(description="Path to SKILL.md file")

    # Optional config
    model: str = Field(default="claude-sonnet-4-20250514", description="Model to use")

    # Test cases
    tests: List[SkillTest] = Field(description="List of test cases")

    # Thresholds
    min_pass_rate: float = Field(default=0.8, ge=0, le=1, description="Minimum pass rate (0-1)")


class SkillTestResult(BaseModel):
    """Result of running a single skill test."""

    __test__ = False  # Tell pytest this is not a test class

    test_name: str
    passed: bool
    score: float = Field(ge=0, le=100)

    # What happened
    input_query: str
    output: str

    # Evaluation breakdown
    contains_passed: List[str] = Field(default_factory=list)
    contains_failed: List[str] = Field(default_factory=list)
    not_contains_passed: List[str] = Field(default_factory=list)
    not_contains_failed: List[str] = Field(default_factory=list)

    # Metrics
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    # Error if any
    error: Optional[str] = None

    timestamp: datetime = Field(default_factory=datetime.now)


class SkillTestSuiteResult(BaseModel):
    """Result of running a complete skill test suite."""

    __test__ = False  # Tell pytest this is not a test class

    suite_name: str
    skill_name: str
    passed: bool

    # Stats
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float

    # Individual results
    results: List[SkillTestResult] = Field(default_factory=list)

    # Aggregate metrics
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0

    timestamp: datetime = Field(default_factory=datetime.now)
