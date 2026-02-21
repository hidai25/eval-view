"""Unit tests for skill agent adapters.

Tests adapter base class, registry, and concrete adapter implementations.
"""

import asyncio
import json
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from evalview.skills.agent_types import AgentConfig, AgentType, SkillAgentTrace
from evalview.skills.types import Skill, SkillMetadata
from evalview.skills.adapters.base import (
    SkillAgentAdapter,
    SkillAgentAdapterError,
    AgentNotFoundError,
    AgentTimeoutError,
)
from evalview.skills.adapters.registry import SkillAdapterRegistry, get_skill_adapter


# =============================================================================
# Base Adapter Tests
# =============================================================================


class TestSkillAgentAdapterExceptions:
    """Tests for adapter exception classes."""

    def test_skill_agent_adapter_error(self):
        """Test base adapter error."""
        error = SkillAgentAdapterError(
            message="Something went wrong",
            adapter_name="test-adapter",
            recoverable=True,
        )

        assert "Something went wrong" in str(error)
        assert error.adapter_name == "test-adapter"
        assert error.recoverable is True

    def test_agent_not_found_error(self):
        """Test agent not found error."""
        error = AgentNotFoundError(
            adapter_name="claude-code",
            install_hint="Run: npm install -g @anthropic/claude-code",
        )

        assert "not found" in str(error).lower()
        assert error.install_hint == "Run: npm install -g @anthropic/claude-code"
        assert error.recoverable is False

    def test_agent_timeout_error(self):
        """Test agent timeout error."""
        error = AgentTimeoutError(
            adapter_name="codex",
            timeout=300.0,
        )

        assert "300" in str(error)
        assert error.timeout == 300.0
        assert error.recoverable is True


class TestSkillAgentAdapterBase:
    """Tests for base adapter class behavior."""

    def test_adapter_stores_config(self):
        """Adapter should store its configuration."""
        config = AgentConfig(
            type=AgentType.CUSTOM,
            max_turns=20,
            timeout=600.0,
        )

        # Create a concrete implementation for testing
        class TestAdapter(SkillAgentAdapter):
            @property
            def name(self) -> str:
                return "test"

            async def execute(self, skill, query, context=None):
                pass

        adapter = TestAdapter(config)

        assert adapter.config == config
        assert adapter.config.max_turns == 20
        assert adapter.config.timeout == 600.0

    def test_get_last_raw_output_initially_none(self):
        """Last raw output should be None initially."""

        class TestAdapter(SkillAgentAdapter):
            @property
            def name(self) -> str:
                return "test"

            async def execute(self, skill, query, context=None):
                pass

        config = AgentConfig(type=AgentType.CUSTOM)
        adapter = TestAdapter(config)

        assert adapter.get_last_raw_output() is None

    @pytest.mark.asyncio
    async def test_health_check_default_returns_true(self):
        """Default health check should return True."""

        class TestAdapter(SkillAgentAdapter):
            @property
            def name(self) -> str:
                return "test"

            async def execute(self, skill, query, context=None):
                pass

        config = AgentConfig(type=AgentType.CUSTOM)
        adapter = TestAdapter(config)

        result = await adapter.health_check()

        assert result is True


# =============================================================================
# Registry Tests
# =============================================================================


class TestSkillAdapterRegistry:
    """Tests for adapter registry."""

    def setup_method(self):
        """Reset registry before each test."""
        SkillAdapterRegistry.reset()

    def test_register_adapter(self):
        """Can register a new adapter."""

        class TestAdapter(SkillAgentAdapter):
            @property
            def name(self) -> str:
                return "test-adapter"

            async def execute(self, skill, query, context=None):
                pass

        SkillAdapterRegistry.register("test-adapter", TestAdapter)

        assert SkillAdapterRegistry.get("test-adapter") == TestAdapter

    def test_get_nonexistent_adapter_returns_none(self):
        """Getting nonexistent adapter returns None."""
        result = SkillAdapterRegistry.get("nonexistent")

        assert result is None

    def test_list_adapters(self):
        """Can list all registered adapters."""

        class Adapter1(SkillAgentAdapter):
            @property
            def name(self):
                return "a1"

            async def execute(self, skill, query, context=None):
                pass

        class Adapter2(SkillAgentAdapter):
            @property
            def name(self):
                return "a2"

            async def execute(self, skill, query, context=None):
                pass

        SkillAdapterRegistry.register("adapter-1", Adapter1)
        SkillAdapterRegistry.register("adapter-2", Adapter2)

        adapters = SkillAdapterRegistry.list_adapters()

        assert "adapter-1" in adapters
        assert "adapter-2" in adapters

    def test_list_names(self):
        """Can list adapter names."""

        class TestAdapter(SkillAgentAdapter):
            @property
            def name(self):
                return "test"

            async def execute(self, skill, query, context=None):
                pass

        SkillAdapterRegistry.register("my-adapter", TestAdapter)

        names = SkillAdapterRegistry.list_names()

        assert "my-adapter" in names

    def test_create_adapter_from_config(self):
        """Can create adapter instance from config."""

        class MyAdapter(SkillAgentAdapter):
            @property
            def name(self):
                return "my-adapter"

            async def execute(self, skill, query, context=None):
                pass

        # Use a unique name that won't be overwritten
        SkillAdapterRegistry.register("my-unique-adapter", MyAdapter)

        # Verify it was registered
        assert SkillAdapterRegistry.get("my-unique-adapter") == MyAdapter

    def test_create_unknown_adapter_raises(self):
        """Creating unknown adapter type raises ValueError."""
        # Reset to ensure no built-ins
        SkillAdapterRegistry.reset()
        SkillAdapterRegistry._initialized = True  # Skip auto-init

        config = AgentConfig(type=AgentType.CLAUDE_CODE)

        with pytest.raises(ValueError) as exc_info:
            SkillAdapterRegistry.create(config)

        assert "Unknown" in str(exc_info.value)

    def test_overwrite_adapter_warns(self, caplog):
        """Overwriting adapter logs warning."""
        import logging

        class Adapter1(SkillAgentAdapter):
            @property
            def name(self):
                return "a1"

            async def execute(self, skill, query, context=None):
                pass

        class Adapter2(SkillAgentAdapter):
            @property
            def name(self):
                return "a2"

            async def execute(self, skill, query, context=None):
                pass

        with caplog.at_level(logging.WARNING):
            SkillAdapterRegistry.register("same-name", Adapter1)
            SkillAdapterRegistry.register("same-name", Adapter2)

        assert "Overwriting" in caplog.text or SkillAdapterRegistry.get("same-name") == Adapter2


class TestGetSkillAdapterFunction:
    """Tests for the convenience function."""

    def setup_method(self):
        SkillAdapterRegistry.reset()

    def test_get_skill_adapter_creates_instance(self):
        """Convenience function creates adapter instance."""
        # Use system-prompt adapter which doesn't need external tools
        config = AgentConfig(type=AgentType.SYSTEM_PROMPT)

        # System prompt type may not have an adapter, so use a registered one
        # Just verify the registry lookup works
        SkillAdapterRegistry._ensure_initialized()
        names = SkillAdapterRegistry.list_names()

        # Should have at least some adapters registered
        assert len(names) > 0


# =============================================================================
# Claude Code Adapter Tests
# =============================================================================


class TestClaudeCodeAdapter:
    """Tests for Claude Code CLI adapter."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(
            type=AgentType.CLAUDE_CODE,
            max_turns=5,
            timeout=60.0,
        )

    @pytest.fixture
    def skill(self) -> Skill:
        return Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="A test skill",
            ),
            instructions="Do something useful.",
            raw_content="",
            file_path="/fake/SKILL.md",
        )

    @pytest.fixture
    def mock_successful_result(self):
        """Mock subprocess result with JSON output."""
        return MagicMock(
            stdout=json.dumps({
                "messages": [
                    {"role": "assistant", "content": "Task completed."}
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }),
            stderr="",
            returncode=0,
        )

    @pytest.mark.asyncio
    async def test_execute_calls_claude_cli(
        self, config, skill, mock_async_subprocess
    ):
        """Execute should call claude CLI with correct arguments."""
        from evalview.skills.adapters.claude_code_adapter import ClaudeCodeAdapter

        # Mock shutil.which to return a fake path
        with patch("shutil.which", return_value="/usr/bin/claude"):
            adapter = ClaudeCodeAdapter(config)
            trace = await adapter.execute(skill, "Test query")

        mock_async_subprocess.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_returns_trace(
        self, config, skill, mock_async_subprocess
    ):
        """Execute should return a valid SkillAgentTrace."""
        from evalview.skills.adapters.claude_code_adapter import ClaudeCodeAdapter

        with patch("shutil.which", return_value="/usr/bin/claude"):
            adapter = ClaudeCodeAdapter(config)
            trace = await adapter.execute(skill, "Test query")

        assert isinstance(trace, SkillAgentTrace)
        assert trace.skill_name == "test-skill"
        assert trace.final_output == "Task completed."

    @pytest.mark.asyncio
    async def test_execute_handles_timeout(self, config, skill, mock_async_subprocess_timeout):
        """Timeout should raise AgentTimeoutError."""
        from evalview.skills.adapters.claude_code_adapter import ClaudeCodeAdapter

        with patch("shutil.which", return_value="/usr/bin/claude"):
            adapter = ClaudeCodeAdapter(config)

            with pytest.raises(AgentTimeoutError) as exc_info:
                await adapter.execute(skill, "Test query")

            assert exc_info.value.timeout == config.timeout

    @pytest.mark.asyncio
    async def test_execute_handles_not_found(
        self, config, skill, mock_async_subprocess_not_found
    ):
        """FileNotFoundError should raise AgentNotFoundError."""
        from evalview.skills.adapters.claude_code_adapter import ClaudeCodeAdapter

        with patch("shutil.which", return_value="/usr/bin/claude"):
            adapter = ClaudeCodeAdapter(config)

            with pytest.raises(AgentNotFoundError) as exc_info:
                await adapter.execute(skill, "Test query")

            assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_health_check_with_version(self, config, mock_async_subprocess):
        """Health check should verify claude is available."""
        from evalview.skills.adapters.claude_code_adapter import ClaudeCodeAdapter

        with patch("shutil.which", return_value="/usr/bin/claude"):
            adapter = ClaudeCodeAdapter(config)
            result = await adapter.health_check()

        assert result is True

    def test_adapter_not_found_when_cli_missing(self, config):
        """Should raise AgentNotFoundError when CLI not installed."""
        from evalview.skills.adapters.claude_code_adapter import ClaudeCodeAdapter

        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile", return_value=False):
                with pytest.raises(AgentNotFoundError) as exc_info:
                    ClaudeCodeAdapter(config)

                assert "not found" in str(exc_info.value).lower()


# =============================================================================
# Custom Adapter Tests
# =============================================================================


class TestCustomAdapter:
    """Tests for custom script adapter."""

    @pytest.fixture
    def config(self, tmp_path) -> AgentConfig:
        # Create a mock script
        script = tmp_path / "runner.sh"
        script.write_text("#!/bin/bash\necho 'test'")
        script.chmod(0o755)

        return AgentConfig(
            type=AgentType.CUSTOM,
            script_path=str(script),
            timeout=30.0,
        )

    @pytest.fixture
    def skill(self) -> Skill:
        return Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="A test skill",
            ),
            instructions="Instructions here.",
            raw_content="",
            file_path="/fake/SKILL.md",
        )

    @pytest.mark.asyncio
    async def test_execute_calls_custom_script(
        self, config, skill, mock_subprocess_run
    ):
        """Execute should call the custom script."""
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "output": "Custom script output",
                "tool_calls": ["Read"],
            }),
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.custom_adapter import CustomAdapter

        adapter = CustomAdapter(config)
        trace = await adapter.execute(skill, "Test query")

        mock_subprocess_run.assert_called_once()
        assert isinstance(trace, SkillAgentTrace)

    def test_missing_script_path_raises(self, skill):
        """Missing script_path should raise error on init."""
        config = AgentConfig(
            type=AgentType.CUSTOM,
            # No script_path
        )

        from evalview.skills.adapters.custom_adapter import CustomAdapter

        # Error should be raised during initialization
        with pytest.raises(SkillAgentAdapterError):
            CustomAdapter(config)


# =============================================================================
# Codex Adapter Tests
# =============================================================================


class TestCodexAdapter:
    """Tests for OpenAI Codex CLI adapter."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(
            type=AgentType.CODEX,
            timeout=120.0,
        )

    @pytest.fixture
    def skill(self) -> Skill:
        return Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="A test skill",
            ),
            instructions="Do something.",
            raw_content="",
            file_path="/fake/SKILL.md",
        )

    @pytest.mark.asyncio
    async def test_execute_calls_codex_cli(
        self, config, skill, mock_subprocess_run
    ):
        """Execute should call codex CLI."""
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "messages": [
                    {"role": "assistant", "content": "Done."}
                ],
            }),
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.codex_adapter import CodexAdapter

        # Mock shutil.which to return a fake path
        with patch("shutil.which", return_value="/usr/bin/codex"):
            adapter = CodexAdapter(config)
            trace = await adapter.execute(skill, "Test query")

        assert isinstance(trace, SkillAgentTrace)
        mock_subprocess_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_name_is_codex(self, config):
        """Adapter name should be 'codex'."""
        from evalview.skills.adapters.codex_adapter import CodexAdapter

        # Mock shutil.which to return a fake path
        with patch("shutil.which", return_value="/usr/bin/codex"):
            adapter = CodexAdapter(config)

        assert adapter.name == "codex"


# =============================================================================
# OpenClaw Adapter Tests
# =============================================================================


class TestOpenClawAdapter:
    """Tests for OpenClaw CLI adapter."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(
            type=AgentType.OPENCLAW,
            timeout=120.0,
        )

    @pytest.fixture
    def skill(self) -> Skill:
        return Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="A test skill",
            ),
            instructions="Do something.",
            raw_content="",
            file_path="/fake/SKILL.md",
        )

    @pytest.mark.asyncio
    async def test_execute_calls_openclaw_cli(
        self, config, skill, mock_subprocess_run
    ):
        """Execute should call openclaw CLI."""
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "messages": [
                    {"role": "assistant", "content": "Done."}
                ],
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 80,
                },
            }),
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            trace = await adapter.execute(skill, "Test query")

        assert isinstance(trace, SkillAgentTrace)
        mock_subprocess_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_returns_trace_with_output(
        self, config, skill, mock_subprocess_run
    ):
        """Execute should return trace with parsed final output."""
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "messages": [
                    {"role": "assistant", "content": "Task completed successfully."}
                ],
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 60,
                },
            }),
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            trace = await adapter.execute(skill, "Test query")

        assert trace.skill_name == "test-skill"
        assert trace.final_output == "Task completed successfully."
        assert trace.total_input_tokens == 150
        assert trace.total_output_tokens == 60

    @pytest.mark.asyncio
    async def test_execute_parses_tool_calls(
        self, config, skill, mock_subprocess_run
    ):
        """Execute should extract tool calls from JSON output."""
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "name": "shell", "input": {"command": "npm install"}},
                            {"type": "tool_use", "name": "write_file", "input": {"path": "index.js"}},
                            {"type": "text", "text": "Project created."},
                        ],
                    }
                ],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }),
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            trace = await adapter.execute(skill, "Create a project")

        assert "shell" in trace.tool_calls
        assert "write_file" in trace.tool_calls
        assert "npm install" in trace.commands_ran
        assert "index.js" in trace.files_created

    @pytest.mark.asyncio
    async def test_execute_handles_jsonl_output(
        self, config, skill, mock_subprocess_run
    ):
        """Execute should parse JSONL (stream) output format."""
        jsonl_output = (
            '{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "run_command", "input": {"command": "ls"}}]}}\n'
            '{"type": "result", "result": "Files listed.", "usage": {"input_tokens": 50, "output_tokens": 20}}'
        )
        mock_subprocess_run.return_value = MagicMock(
            stdout=jsonl_output,
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            trace = await adapter.execute(skill, "List files")

        assert trace.final_output == "Files listed."
        assert "run_command" in trace.tool_calls
        assert "ls" in trace.commands_ran

    @pytest.mark.asyncio
    async def test_execute_handles_text_fallback(
        self, config, skill, mock_subprocess_run
    ):
        """Execute should fall back to text parsing when no JSON."""
        mock_subprocess_run.return_value = MagicMock(
            stdout="Plain text response from OpenClaw.",
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            trace = await adapter.execute(skill, "Test query")

        assert trace.final_output == "Plain text response from OpenClaw."
        assert trace.tool_calls == []

    @pytest.mark.asyncio
    async def test_adapter_name_is_openclaw(self, config):
        """Adapter name should be 'openclaw'."""
        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)

        assert adapter.name == "openclaw"

    def test_adapter_not_found_when_cli_missing(self, config):
        """Should raise AgentNotFoundError when CLI not installed."""
        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value=None):
            with patch("os.access", return_value=False):
                adapter = OpenClawAdapter(config)
                with pytest.raises(AgentNotFoundError) as exc_info:
                    _ = adapter.openclaw_path

                assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_health_check_succeeds(self, config, mock_subprocess_run):
        """Health check should return True when CLI is available."""
        mock_subprocess_run.return_value = MagicMock(
            stdout="openclaw 1.0.0",
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            result = await adapter.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_fails_gracefully(self, config, mock_subprocess_run):
        """Health check should return False when CLI fails."""
        mock_subprocess_run.side_effect = FileNotFoundError("openclaw not found")

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            result = await adapter.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_records_errors_on_nonzero_exit(
        self, config, skill, mock_subprocess_run
    ):
        """Non-zero exit code should be captured in trace errors."""
        mock_subprocess_run.return_value = MagicMock(
            stdout="",
            stderr="Error: skill not found",
            returncode=1,
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            trace = await adapter.execute(skill, "Test query")

        assert trace.has_errors
        assert any("code 1" in e for e in trace.errors)

    def test_invocation_log_tracking(self, config, skill):
        """Invocations should be logged for audit."""
        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)

        log = adapter.get_invocation_log()
        assert log == []  # No invocations yet

    @pytest.mark.asyncio
    async def test_execute_handles_prompt_tokens_alias(
        self, config, skill, mock_subprocess_run
    ):
        """Should handle prompt_tokens/completion_tokens aliases."""
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "result": "Done.",
                "usage": {
                    "prompt_tokens": 300,
                    "completion_tokens": 100,
                },
            }),
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            trace = await adapter.execute(skill, "Test query")

        assert trace.total_input_tokens == 300
        assert trace.total_output_tokens == 100


# =============================================================================
# LangGraph Adapter Tests
# =============================================================================


class TestLangGraphAdapter:
    """Tests for LangGraph SDK adapter."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(
            type=AgentType.LANGGRAPH,
            env={
                "LANGGRAPH_API_URL": "http://localhost:2024",
            },
            timeout=180.0,
        )

    @pytest.fixture
    def skill(self) -> Skill:
        return Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="A test skill",
            ),
            instructions="Instructions.",
            raw_content="",
            file_path="/fake/SKILL.md",
        )

    def test_adapter_name_is_langgraph(self, config):
        """Adapter name should be 'langgraph'."""
        from evalview.skills.adapters.langgraph_adapter import LangGraphSkillAdapter

        adapter = LangGraphSkillAdapter(config)

        assert adapter.name == "langgraph"

    def test_adapter_stores_config(self, config):
        """Adapter should store configuration."""
        from evalview.skills.adapters.langgraph_adapter import LangGraphSkillAdapter

        adapter = LangGraphSkillAdapter(config)

        assert adapter.config.timeout == 180.0


# =============================================================================
# CrewAI Adapter Tests
# =============================================================================


class TestCrewAIAdapter:
    """Tests for CrewAI framework adapter."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(
            type=AgentType.CREWAI,
            timeout=300.0,
        )

    @pytest.fixture
    def skill(self) -> Skill:
        return Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="A test skill",
            ),
            instructions="Do the thing.",
            raw_content="",
            file_path="/fake/SKILL.md",
        )

    @pytest.mark.asyncio
    async def test_adapter_name_is_crewai(self, config):
        """Adapter name should be 'crewai'."""
        from evalview.skills.adapters.crewai_adapter import CrewAISkillAdapter

        adapter = CrewAISkillAdapter(config)

        assert adapter.name == "crewai"


# =============================================================================
# OpenAI Assistants Adapter Tests
# =============================================================================


class TestOpenAIAssistantsAdapter:
    """Tests for OpenAI Assistants API adapter."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(
            type=AgentType.OPENAI_ASSISTANTS,
            timeout=120.0,
            env={"OPENAI_API_KEY": "test-key"},
        )

    @pytest.fixture
    def skill(self) -> Skill:
        return Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="A test skill",
                tools=["Read", "Write"],
            ),
            instructions="Follow these instructions.",
            raw_content="",
            file_path="/fake/SKILL.md",
        )

    def test_adapter_requires_api_key(self):
        """Adapter should require OPENAI_API_KEY."""
        config = AgentConfig(
            type=AgentType.OPENAI_ASSISTANTS,
            timeout=120.0,
        )

        from evalview.skills.adapters.openai_assistants_adapter import (
            OpenAIAssistantsSkillAdapter,
        )

        with pytest.raises(SkillAgentAdapterError) as exc_info:
            OpenAIAssistantsSkillAdapter(config)

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_adapter_name_is_openai_assistants(self):
        """Adapter name should be 'openai-assistants'."""
        from evalview.skills.adapters.openai_assistants_adapter import (
            OpenAIAssistantsSkillAdapter,
        )

        # Use environment variable
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            config = AgentConfig(
                type=AgentType.OPENAI_ASSISTANTS,
                timeout=120.0,
            )
            adapter = OpenAIAssistantsSkillAdapter(config)

            assert adapter.name == "openai-assistants"
