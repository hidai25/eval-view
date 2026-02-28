"""Pytest fixtures for skill testing unit tests.

This module provides reusable fixtures for testing the skill testing
infrastructure, including mock skills, traces, and adapter configurations.

Fixtures are organized by category:
    - Skills: Mock skill objects for testing
    - Traces: Sample execution traces
    - Configs: Agent configurations
    - Mocks: Subprocess and HTTP mocks

Usage:
    def test_evaluator(sample_skill, sample_trace):
        result = evaluator.evaluate(sample_skill, sample_trace)
        assert result.passed
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from evalview.skills.types import Skill, SkillMetadata
from evalview.skills.agent_types import (
    AgentConfig,
    AgentType,
    DeterministicExpected,
    RubricConfig,
    SkillAgentTest,
    SkillAgentTestSuite,
    SkillAgentTrace,
    TestCategory,
    TraceEvent,
    TraceEventType,
)


# =============================================================================
# Skill Fixtures
# =============================================================================


@pytest.fixture
def sample_skill_metadata() -> SkillMetadata:
    """Create a sample skill metadata object."""
    return SkillMetadata(
        name="test-skill",
        description="A test skill for unit testing purposes.",
        version="1.0.0",
        author="EvalView Test Suite",
        triggers=["test", "verify"],
        tools=["Read", "Write", "Bash"],
    )


@pytest.fixture
def sample_skill(sample_skill_metadata: SkillMetadata) -> Skill:
    """Create a sample skill for testing."""
    return Skill(
        metadata=sample_skill_metadata,
        instructions="""## Test Skill Instructions

When activated, this skill should:
1. Read relevant files
2. Process the input
3. Write output to a file

Always respond with "SUCCESS" when complete.
""",
        raw_content="---\nname: test-skill\n---\n# Instructions...",
        file_path="/fake/path/SKILL.md",
    )


@pytest.fixture
def code_review_skill() -> Skill:
    """Create a code review skill for realistic testing."""
    return Skill(
        metadata=SkillMetadata(
            name="code-review",
            description="Reviews code for bugs, security issues, and style violations.",
            tools=["Read", "Grep"],
        ),
        instructions="""## Code Review Instructions

1. Read the target file(s)
2. Analyze for:
   - Security vulnerabilities (SQL injection, XSS, etc.)
   - Logic errors
   - Style violations
3. Provide specific, actionable feedback

Always mention specific line numbers when possible.
""",
        raw_content="",
        file_path="/skills/code-review/SKILL.md",
    )


# =============================================================================
# Trace Fixtures
# =============================================================================


@pytest.fixture
def sample_trace() -> SkillAgentTrace:
    """Create a sample execution trace."""
    start = datetime.now()
    end = start + timedelta(seconds=5)

    return SkillAgentTrace(
        session_id="test-abc123",
        skill_name="test-skill",
        test_name="basic-test",
        start_time=start,
        end_time=end,
        events=[
            TraceEvent(
                type=TraceEventType.TOOL_CALL,
                tool_name="Read",
                tool_input={"file_path": "/src/main.py"},
                tool_success=True,
            ),
            TraceEvent(
                type=TraceEventType.TOOL_CALL,
                tool_name="Write",
                tool_input={"file_path": "output.txt", "content": "Result"},
                tool_success=True,
            ),
        ],
        tool_calls=["Read", "Write"],
        files_created=["output.txt"],
        files_modified=[],
        commands_ran=[],
        total_input_tokens=1500,
        total_output_tokens=500,
        final_output="SUCCESS: Operation completed.",
        errors=[],
    )


@pytest.fixture
def trace_with_errors() -> SkillAgentTrace:
    """Create a trace with errors for failure testing."""
    start = datetime.now()
    end = start + timedelta(seconds=2)

    return SkillAgentTrace(
        session_id="test-error-456",
        skill_name="test-skill",
        test_name="error-test",
        start_time=start,
        end_time=end,
        events=[
            TraceEvent(
                type=TraceEventType.TOOL_CALL,
                tool_name="Bash",
                tool_input={"command": "rm -rf /"},
                tool_success=False,
                tool_error="Permission denied",
            ),
            TraceEvent(
                type=TraceEventType.ERROR,
                tool_error="Command execution failed",
            ),
        ],
        tool_calls=["Bash"],
        files_created=[],
        files_modified=[],
        commands_ran=["rm -rf /"],
        total_input_tokens=100,
        total_output_tokens=50,
        final_output="ERROR: Command failed.",
        errors=["Permission denied", "Command execution failed"],
    )


@pytest.fixture
def trace_with_commands() -> SkillAgentTrace:
    """Create a trace with command executions."""
    start = datetime.now()
    end = start + timedelta(seconds=10)

    return SkillAgentTrace(
        session_id="test-cmd-789",
        skill_name="test-skill",
        test_name="command-test",
        start_time=start,
        end_time=end,
        events=[],
        tool_calls=["Bash", "Bash", "Read"],
        files_created=["package.json", "src/index.ts"],
        files_modified=["README.md"],
        commands_ran=[
            "npm init -y",
            "npm install typescript",
            "tsc --init",
        ],
        total_input_tokens=2000,
        total_output_tokens=1000,
        final_output="Project initialized successfully.",
        errors=[],
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def agent_config_claude() -> AgentConfig:
    """Create Claude Code agent configuration."""
    return AgentConfig(
        type=AgentType.CLAUDE_CODE,
        max_turns=10,
        timeout=120.0,
        capture_trace=True,
    )


@pytest.fixture
def agent_config_custom() -> AgentConfig:
    """Create custom script agent configuration."""
    return AgentConfig(
        type=AgentType.CUSTOM,
        script_path="/path/to/runner.sh",
        max_turns=5,
        timeout=60.0,
    )


@pytest.fixture
def agent_config_langgraph() -> AgentConfig:
    """Create LangGraph agent configuration."""
    return AgentConfig(
        type=AgentType.LANGGRAPH,
        env={
            "LANGGRAPH_API_URL": "http://localhost:2024",
            "LANGGRAPH_API_KEY": "test-key",
        },
        max_turns=15,
        timeout=180.0,
    )


# =============================================================================
# Test Suite Fixtures
# =============================================================================


@pytest.fixture
def deterministic_expected() -> DeterministicExpected:
    """Create sample expected behaviors for deterministic checks."""
    return DeterministicExpected(
        tool_calls_contain=["Read", "Write"],
        tool_calls_not_contain=["Bash"],
        files_created=["output.txt"],
        output_contains=["SUCCESS"],
        output_not_contains=["ERROR", "FAILED"],
    )


@pytest.fixture
def rubric_config() -> RubricConfig:
    """Create sample rubric configuration."""
    return RubricConfig(
        prompt="""Evaluate the output quality based on:
1. Completeness - Did it address all requirements?
2. Accuracy - Is the information correct?
3. Clarity - Is the response clear and well-structured?

Score from 0-100.""",
        min_score=70.0,
        model="gpt-4o-mini",
    )


@pytest.fixture
def sample_test(deterministic_expected: DeterministicExpected) -> SkillAgentTest:
    """Create a sample skill test."""
    return SkillAgentTest(
        name="basic-functionality",
        description="Tests basic skill functionality",
        input="Process the input and write to output.txt",
        category=TestCategory.EXPLICIT,
        should_trigger=True,
        expected=deterministic_expected,
    )


@pytest.fixture
def negative_test() -> SkillAgentTest:
    """Create a negative control test."""
    return SkillAgentTest(
        name="should-not-trigger",
        description="Tests that skill doesn't trigger on irrelevant input",
        input="What's the weather like today?",
        category=TestCategory.NEGATIVE,
        should_trigger=False,
        expected=DeterministicExpected(
            tool_calls_not_contain=["Write"],
            output_not_contains=["SUCCESS"],
        ),
    )


@pytest.fixture
def sample_test_suite(
    sample_skill: Skill,
    sample_test: SkillAgentTest,
    negative_test: SkillAgentTest,
    agent_config_claude: AgentConfig,
) -> SkillAgentTestSuite:
    """Create a complete test suite."""
    return SkillAgentTestSuite(
        name="test-skill-suite",
        description="Comprehensive test suite for test-skill",
        skill=sample_skill.file_path or "/fake/path/SKILL.md",
        agent=agent_config_claude,
        tests=[sample_test, negative_test],
        min_pass_rate=0.8,
    )


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for adapter tests."""
    with patch("subprocess.run") as mock:
        # Default successful response
        mock.return_value = MagicMock(
            stdout='{"result": "SUCCESS", "tool_calls": ["Read", "Write"]}',
            stderr="",
            returncode=0,
        )
        yield mock


@pytest.fixture
def mock_subprocess_timeout():
    """Mock subprocess.run to raise timeout."""
    import subprocess

    with patch("subprocess.run") as mock:
        mock.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=30)
        yield mock


@pytest.fixture
def mock_subprocess_not_found():
    """Mock subprocess.run to raise file not found."""
    with patch("subprocess.run") as mock:
        mock.side_effect = FileNotFoundError("claude not found")
        yield mock


@pytest.fixture
def mock_async_subprocess():
    """Mock asyncio.create_subprocess_exec for async adapter tests.

    Returns stream-json format (JSONL) that matches Claude Code CLI output.
    """
    import asyncio

    async def create_mock_process(*args, **kwargs):
        mock_process = AsyncMock()
        mock_process.returncode = 0
        # Stream-json format: one JSON per line
        stream_json_output = b'{"type": "assistant", "message": {"content": [{"type": "text", "text": "Task completed."}]}}\n{"type": "result", "result": "Task completed.", "usage": {"input_tokens": 100, "output_tokens": 50}}'
        mock_process.communicate = AsyncMock(
            return_value=(stream_json_output, b"")
        )
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()
        return mock_process

    with patch("asyncio.create_subprocess_exec", side_effect=create_mock_process) as mock:
        yield mock


@pytest.fixture
def mock_async_subprocess_timeout():
    """Mock asyncio.create_subprocess_exec to simulate timeout."""
    import asyncio

    async def create_mock_process(*args, **kwargs):
        mock_process = AsyncMock()
        mock_process.returncode = None

        async def mock_communicate():
            raise asyncio.TimeoutError()

        mock_process.communicate = mock_communicate
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()
        return mock_process

    with patch("asyncio.create_subprocess_exec", side_effect=create_mock_process) as mock:
        yield mock


@pytest.fixture
def mock_async_subprocess_not_found():
    """Mock asyncio.create_subprocess_exec to raise file not found."""
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("claude not found")) as mock:
        yield mock


@pytest.fixture
def mock_claude_popen():
    """Mock subprocess.Popen for ClaudeCodeAdapter tests.

    ClaudeCodeAdapter uses subprocess.Popen (not asyncio) — it writes
    to temp files passed as stdout/stderr handles, then reads them back.
    This fixture mimics that by writing JSONL output to the stdout handle.
    """
    import subprocess as _subprocess

    jsonl_output = (
        b'{"type": "assistant", "message": {"content": [{"type": "text", "text": "Task completed."}]}}\n'
        b'{"type": "result", "result": "Task completed.", "usage": {"input_tokens": 100, "output_tokens": 50}}'
    )

    def _fake_popen(cmd, stdin=None, stdout=None, stderr=None, **kwargs):
        if stdout is not None:
            stdout.write(jsonl_output)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None
        return mock_proc

    with patch("subprocess.Popen", side_effect=_fake_popen) as mock:
        yield mock


@pytest.fixture
def mock_claude_popen_timeout():
    """Mock subprocess.Popen for ClaudeCodeAdapter — simulate process timeout."""
    import subprocess as _subprocess

    def _fake_popen(cmd, stdin=None, stdout=None, stderr=None, **kwargs):
        mock_proc = MagicMock()
        mock_proc.returncode = None
        # First wait() call (with timeout=) raises; second (after kill) returns.
        mock_proc.wait = MagicMock(
            side_effect=[_subprocess.TimeoutExpired(cmd, timeout=300), None]
        )
        mock_proc.kill = MagicMock()
        return mock_proc

    with patch("subprocess.Popen", side_effect=_fake_popen) as mock:
        yield mock


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp.ClientSession for HTTP adapter tests."""
    with patch("aiohttp.ClientSession") as mock_class:
        mock_session = AsyncMock()

        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "thread_id": "thread-123",
            "messages": [
                {"role": "assistant", "content": "Task completed successfully."}
            ],
        })
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_class.return_value.__aexit__ = AsyncMock(return_value=None)

        yield mock_session


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for Assistants adapter tests."""
    with patch("openai.AsyncOpenAI") as mock_class:
        mock_client = AsyncMock()

        # Mock models.list for health check
        mock_client.models.list = AsyncMock(return_value=MagicMock(data=[]))

        # Mock beta.assistants.create
        mock_client.beta.assistants.create = AsyncMock(
            return_value=MagicMock(id="asst_123")
        )
        mock_client.beta.assistants.delete = AsyncMock()

        # Mock beta.threads.create
        mock_client.beta.threads.create = AsyncMock(
            return_value=MagicMock(id="thread_123")
        )

        # Mock beta.threads.messages
        mock_client.beta.threads.messages.create = AsyncMock()
        mock_client.beta.threads.messages.list = AsyncMock(
            return_value=MagicMock(
                data=[
                    MagicMock(
                        role="assistant",
                        content=[
                            MagicMock(type="text", text=MagicMock(value="Success!"))
                        ],
                    )
                ]
            )
        )

        # Mock beta.threads.runs
        mock_run = MagicMock(
            id="run_123",
            status="completed",
            usage=MagicMock(prompt_tokens=100, completion_tokens=50),
            last_error=None,
        )
        mock_client.beta.threads.runs.create = AsyncMock(return_value=mock_run)
        mock_client.beta.threads.runs.retrieve = AsyncMock(return_value=mock_run)

        # Mock beta.threads.runs.steps
        mock_client.beta.threads.runs.steps.list = AsyncMock(
            return_value=MagicMock(data=[])
        )

        mock_class.return_value = mock_client
        yield mock_client


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def temp_skill_file(tmp_path):
    """Create a temporary SKILL.md file."""
    skill_content = """---
name: temp-test-skill
description: Temporary skill for testing
---

# Temporary Test Skill

This is a temporary skill for testing file loading.

## Instructions

1. Do something
2. Return success
"""
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(skill_content)
    return str(skill_path)


@pytest.fixture
def temp_test_yaml(tmp_path, temp_skill_file):
    """Create a temporary test YAML file."""
    yaml_content = f"""name: temp-test-suite
skill: {temp_skill_file}
agent:
  type: system-prompt
tests:
  - name: basic-test
    input: "Test query"
    expected:
      output_contains: ["success"]
"""
    yaml_path = tmp_path / "tests.yaml"
    yaml_path.write_text(yaml_content)
    return str(yaml_path)
