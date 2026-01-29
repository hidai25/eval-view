"""Unit tests for rubric-based LLM evaluator.

Tests Phase 2 rubric evaluation with mocked LLM responses.
"""

from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from evalview.skills.agent_types import (
    RubricConfig,
    SkillAgentTrace,
)
from evalview.skills.evaluators.rubric import RubricEvaluator


class TestRubricEvaluatorBasic:
    """Basic rubric evaluator tests."""

    @pytest.fixture
    def evaluator(self) -> RubricEvaluator:
        """Create a fresh evaluator instance."""
        return RubricEvaluator()

    @pytest.fixture
    def rubric(self) -> RubricConfig:
        """Create a sample rubric configuration."""
        return RubricConfig(
            prompt="Evaluate the response for accuracy and completeness.",
            min_score=70.0,
            model="test-model",
        )

    @pytest.fixture
    def trace(self) -> SkillAgentTrace:
        """Create a sample trace."""
        now = datetime.now()
        return SkillAgentTrace(
            session_id="test-123",
            skill_name="test-skill",
            test_name="test-case",
            start_time=now,
            end_time=now + timedelta(seconds=5),
            tool_calls=["Read", "Write"],
            files_created=["output.txt"],
            final_output="Task completed successfully with all requirements met.",
        )


class TestRubricEvaluatorSuccess:
    """Tests for successful rubric evaluation."""

    @pytest.fixture
    def evaluator(self) -> RubricEvaluator:
        return RubricEvaluator()

    @pytest.fixture
    def rubric(self) -> RubricConfig:
        return RubricConfig(
            prompt="Evaluate for accuracy.",
            min_score=70.0,
        )

    @pytest.fixture
    def trace(self) -> SkillAgentTrace:
        now = datetime.now()
        return SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="Perfect output",
        )

    @pytest.mark.asyncio
    async def test_high_score_passes(self, evaluator, rubric, trace):
        """High LLM score should result in passed evaluation."""
        mock_response = {
            "score": 95,
            "reasoning": "Excellent work. All criteria met.",
            "strengths": ["Complete", "Accurate"],
            "weaknesses": [],
        }

        with patch.object(evaluator, "_llm_client") as mock_client:
            mock_client.chat_completion = AsyncMock(return_value=mock_response)
            evaluator._llm_client = mock_client

            result = await evaluator.evaluate(rubric, trace, "test-skill")

            assert result.passed is True
            assert result.score == 95
            assert "Excellent" in result.rationale
            assert result.min_score == 70.0

    @pytest.mark.asyncio
    async def test_score_at_threshold_passes(self, evaluator, rubric, trace):
        """Score exactly at min_score should pass."""
        mock_response = {
            "score": 70,
            "reasoning": "Meets minimum requirements.",
        }

        with patch.object(evaluator, "_llm_client") as mock_client:
            mock_client.chat_completion = AsyncMock(return_value=mock_response)
            evaluator._llm_client = mock_client

            result = await evaluator.evaluate(rubric, trace, "test-skill")

            assert result.passed is True
            assert result.score == 70

    @pytest.mark.asyncio
    async def test_low_score_fails(self, evaluator, rubric, trace):
        """Low LLM score should result in failed evaluation."""
        mock_response = {
            "score": 50,
            "reasoning": "Does not meet minimum requirements.",
        }

        with patch.object(evaluator, "_llm_client") as mock_client:
            mock_client.chat_completion = AsyncMock(return_value=mock_response)
            evaluator._llm_client = mock_client

            result = await evaluator.evaluate(rubric, trace, "test-skill")

            assert result.passed is False
            assert result.score == 50

    @pytest.mark.asyncio
    async def test_rubric_response_preserved(self, evaluator, rubric, trace):
        """Full LLM response should be preserved in result."""
        mock_response = {
            "score": 85,
            "reasoning": "Good work.",
            "custom_field": "extra data",
        }

        with patch.object(evaluator, "_llm_client") as mock_client:
            mock_client.chat_completion = AsyncMock(return_value=mock_response)
            evaluator._llm_client = mock_client

            result = await evaluator.evaluate(rubric, trace, "test-skill")

            assert result.rubric_response == mock_response
            assert result.rubric_response.get("custom_field") == "extra data"


class TestRubricEvaluatorPromptBuilding:
    """Tests for prompt construction."""

    @pytest.fixture
    def evaluator(self) -> RubricEvaluator:
        return RubricEvaluator()

    def test_system_prompt_contains_rubric(self, evaluator):
        """System prompt should contain the rubric text."""
        rubric = RubricConfig(
            prompt="Check for code quality and documentation.",
            min_score=80.0,
        )

        system_prompt = evaluator._build_system_prompt(rubric)

        assert "code quality" in system_prompt
        assert "documentation" in system_prompt
        assert "0 to 100" in system_prompt

    def test_user_prompt_contains_trace_details(self, evaluator):
        """User prompt should contain trace execution details."""
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="test-123",
            skill_name="my-skill",
            test_name="my-test",
            start_time=now,
            end_time=now + timedelta(seconds=3),
            tool_calls=["Read", "Write"],
            files_created=["output.txt"],
            files_modified=["config.json"],
            commands_ran=["npm test"],
            final_output="All tests passed.",
            errors=[],
        )

        user_prompt = evaluator._build_user_prompt(trace, "my-skill")

        assert "my-skill" in user_prompt
        assert "my-test" in user_prompt
        assert "All tests passed" in user_prompt
        assert "Read, Write" in user_prompt
        assert "output.txt" in user_prompt
        assert "config.json" in user_prompt

    def test_user_prompt_truncates_long_output(self, evaluator):
        """Very long output should be truncated."""
        now = datetime.now()
        long_output = "x" * 10000  # 10k chars, exceeds 5k limit
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output=long_output,
        )

        user_prompt = evaluator._build_user_prompt(trace, "skill")

        assert "truncated" in user_prompt.lower()
        assert len(user_prompt) < len(long_output)

    def test_user_prompt_handles_empty_lists(self, evaluator):
        """User prompt handles empty tool/file lists."""
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            tool_calls=[],
            files_created=[],
            files_modified=[],
            final_output="Output",
        )

        user_prompt = evaluator._build_user_prompt(trace, "skill")

        assert "None" in user_prompt

    def test_user_prompt_shows_errors(self, evaluator):
        """User prompt should include errors from trace."""
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="Output",
            errors=["Connection timeout", "Retry failed"],
        )

        user_prompt = evaluator._build_user_prompt(trace, "skill")

        assert "Connection timeout" in user_prompt
        assert "Retry failed" in user_prompt


class TestRubricEvaluatorErrors:
    """Tests for error handling."""

    @pytest.fixture
    def evaluator(self) -> RubricEvaluator:
        return RubricEvaluator()

    @pytest.fixture
    def rubric(self) -> RubricConfig:
        return RubricConfig(
            prompt="Evaluate something.",
            min_score=70.0,
        )

    @pytest.fixture
    def trace(self) -> SkillAgentTrace:
        now = datetime.now()
        return SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="Output",
        )

    @pytest.mark.asyncio
    async def test_general_exception_returns_failure(self, evaluator, rubric, trace):
        """General exceptions should be handled gracefully."""
        # Set up an LLM client that raises an exception
        mock_client = AsyncMock()
        mock_client.chat_completion = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )
        evaluator._llm_client = mock_client

        result = await evaluator.evaluate(rubric, trace, "skill")

        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_llm_exception_returns_failure(self, evaluator, rubric, trace):
        """General LLM exception should return failure."""
        with patch.object(evaluator, "_llm_client") as mock_client:
            mock_client.chat_completion = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )
            evaluator._llm_client = mock_client

            result = await evaluator.evaluate(rubric, trace, "skill")

            assert result.passed is False
            assert result.score == 0.0
            assert "rate limit" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_missing_score_in_response(self, evaluator, rubric, trace):
        """Handle LLM response without score field."""
        mock_response = {
            "reasoning": "Good work but no score provided.",
        }

        with patch.object(evaluator, "_llm_client") as mock_client:
            mock_client.chat_completion = AsyncMock(return_value=mock_response)
            evaluator._llm_client = mock_client

            result = await evaluator.evaluate(rubric, trace, "skill")

            # Score defaults to 0 when not provided
            assert result.score == 0.0
            assert result.passed is False


class TestRubricEvaluatorConfiguration:
    """Tests for evaluator configuration."""

    def test_model_override_at_init(self):
        """Model can be overridden at init time."""
        evaluator = RubricEvaluator(model="custom-model")

        assert evaluator.model_override == "custom-model"

    @pytest.mark.asyncio
    async def test_rubric_model_takes_precedence(self):
        """Model in rubric config takes precedence over init."""
        evaluator = RubricEvaluator(model="init-model")
        rubric = RubricConfig(
            prompt="Evaluate something in detail.",
            min_score=70.0,
            model="rubric-model",  # This should take precedence
        )
        now = datetime.now()
        trace = SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="Output",
        )

        # Create a mock LLM client
        mock_client = AsyncMock()
        mock_client.chat_completion = AsyncMock(
            return_value={"score": 80, "reasoning": "Good"}
        )
        evaluator._llm_client = mock_client

        result = await evaluator.evaluate(rubric, trace, "skill")

        assert result.score == 80


class TestRubricEvaluatorResponseParsing:
    """Tests for parsing LLM responses."""

    @pytest.fixture
    def evaluator(self) -> RubricEvaluator:
        return RubricEvaluator()

    @pytest.fixture
    def rubric(self) -> RubricConfig:
        return RubricConfig(
            prompt="Evaluate the output for accuracy and completeness.",
            min_score=70.0,
        )

    @pytest.fixture
    def trace(self) -> SkillAgentTrace:
        now = datetime.now()
        return SkillAgentTrace(
            session_id="t",
            skill_name="s",
            test_name="t",
            start_time=now,
            end_time=now,
            final_output="Output",
        )

    @pytest.mark.asyncio
    async def test_parses_reasoning_field(self, evaluator, rubric, trace):
        """Parses 'reasoning' field from response."""
        mock_response = {
            "score": 85,
            "reasoning": "Well done.",
        }

        with patch.object(evaluator, "_llm_client") as mock_client:
            mock_client.chat_completion = AsyncMock(return_value=mock_response)
            evaluator._llm_client = mock_client

            result = await evaluator.evaluate(rubric, trace, "skill")

            assert result.rationale == "Well done."

    @pytest.mark.asyncio
    async def test_parses_rationale_field_fallback(self, evaluator, rubric, trace):
        """Falls back to 'rationale' if 'reasoning' not present."""
        mock_response = {
            "score": 85,
            "rationale": "Alternative field name.",
        }

        with patch.object(evaluator, "_llm_client") as mock_client:
            mock_client.chat_completion = AsyncMock(return_value=mock_response)
            evaluator._llm_client = mock_client

            result = await evaluator.evaluate(rubric, trace, "skill")

            assert result.rationale == "Alternative field name."

    @pytest.mark.asyncio
    async def test_converts_string_score_to_float(self, evaluator, rubric, trace):
        """Score can be string and gets converted to float."""
        mock_response = {
            "score": "75.5",
            "reasoning": "Good.",
        }

        with patch.object(evaluator, "_llm_client") as mock_client:
            mock_client.chat_completion = AsyncMock(return_value=mock_response)
            evaluator._llm_client = mock_client

            result = await evaluator.evaluate(rubric, trace, "skill")

            assert result.score == 75.5
            assert result.passed is True  # 75.5 >= 70
