"""Skills testing module for EvalView.

This module provides tools for parsing, validating, and testing
Claude Code skills and MCP servers.

Includes both legacy (system-prompt) testing and new agent-based testing
that executes skills through real AI agents like Claude Code.
"""

from evalview.skills.types import (
    Skill,
    SkillMetadata,
    SkillValidationResult,
    SkillValidationError,
    SkillTestSuite,
    SkillTest,
    SkillTestResult,
    SkillTestSuiteResult,
)
from evalview.skills.parser import SkillParser
from evalview.skills.validator import SkillValidator
from evalview.skills.runner import SkillRunner

# Agent-based testing types
from evalview.skills.agent_types import (
    AgentType,
    TestCategory,
    AgentConfig,
    DeterministicExpected,
    RubricConfig,
    SmokeTest,
    SkillAgentTest,
    SkillAgentTestSuite,
    TraceEventType,
    TraceEvent,
    SkillAgentTrace,
    DeterministicCheckResult,
    DeterministicEvaluation,
    RubricEvaluation,
    SkillAgentTestResult,
    SkillAgentTestSuiteResult,
)

# Agent-based runner
from evalview.skills.agent_runner import SkillAgentRunner, run_agent_tests

# Test generator
from evalview.skills.test_generator import SkillTestGenerator

__all__ = [
    # Legacy Types
    "Skill",
    "SkillMetadata",
    "SkillValidationResult",
    "SkillValidationError",
    "SkillTestSuite",
    "SkillTest",
    "SkillTestResult",
    "SkillTestSuiteResult",
    # Parser
    "SkillParser",
    # Validator
    "SkillValidator",
    # Legacy Runner
    "SkillRunner",
    # Agent Types
    "AgentType",
    "TestCategory",
    "AgentConfig",
    "DeterministicExpected",
    "RubricConfig",
    "SmokeTest",
    "SkillAgentTest",
    "SkillAgentTestSuite",
    "TraceEventType",
    "TraceEvent",
    "SkillAgentTrace",
    "DeterministicCheckResult",
    "DeterministicEvaluation",
    "RubricEvaluation",
    "SkillAgentTestResult",
    "SkillAgentTestSuiteResult",
    # Agent Runner
    "SkillAgentRunner",
    "run_agent_tests",
    # Test Generator
    "SkillTestGenerator",
]
