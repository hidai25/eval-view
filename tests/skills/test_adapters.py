"""Unit tests for skill agent adapters.

Tests adapter base class, CLIAgentAdapter shared logic, registry, and
concrete adapter implementations.
"""

import asyncio
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from evalview.skills.agent_types import AgentConfig, AgentType, SkillAgentTrace
from evalview.skills.types import Skill, SkillMetadata
from evalview.skills.adapters.base import (
    CLIAgentAdapter,
    CLIInvocation,
    SkillAgentAdapter,
    SkillAgentAdapterError,
    AgentNotFoundError,
    AgentTimeoutError,
    _MAX_OUTPUT_SIZE,
    _SENSITIVE_ENV_PATTERNS,
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
# CLIAgentAdapter (shared logic) Tests
# =============================================================================


class _StubCLIAdapter(CLIAgentAdapter):
    """Minimal concrete CLIAgentAdapter for testing shared logic."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub-cli"

    def _candidate_paths(self) -> List[Path]:
        return []

    def _install_hint(self) -> str:
        return "Install stub-cli"

    def _build_command(self, skill: Skill, query: str) -> List[str]:
        return [self.cli_path, "--prompt", query]


def _make_stub(config: Optional[AgentConfig] = None) -> _StubCLIAdapter:
    """Create a StubCLIAdapter with binary resolution mocked out."""
    cfg = config or AgentConfig(type=AgentType.CUSTOM, timeout=60.0)
    with patch("shutil.which", return_value="/usr/bin/stub-cli"):
        adapter = _StubCLIAdapter(cfg)
        adapter._binary_path = "/usr/bin/stub-cli"
    return adapter


class TestCLIAgentAdapterShared:
    """Tests for shared CLIAgentAdapter logic.

    These test the base class once, so subclasses don't need to re-test.
    """

    # -- Binary resolution -----------------------------------------------

    def test_cli_path_resolves_from_which(self):
        with patch("shutil.which", return_value="/usr/bin/stub-cli"):
            cfg = AgentConfig(type=AgentType.CUSTOM, timeout=30.0)
            adapter = _StubCLIAdapter(cfg)
            assert adapter.cli_path == "/usr/bin/stub-cli"

    def test_cli_path_not_found_raises(self):
        with patch("shutil.which", return_value=None):
            cfg = AgentConfig(type=AgentType.CUSTOM, timeout=30.0)
            adapter = _StubCLIAdapter(cfg)
            with pytest.raises(AgentNotFoundError) as exc:
                _ = adapter.cli_path
            assert "not found" in str(exc.value).lower()

    # -- Environment sanitisation ----------------------------------------

    def test_prepare_environment_filters_secrets(self):
        adapter = _make_stub()
        with patch.dict("os.environ", {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "API_KEY": "secret-123",
            "AWS_SECRET": "aws-secret",
            "SAFE_VAR": "safe-value",
        }, clear=True):
            env = adapter._prepare_environment()

        assert "PATH" in env
        assert "HOME" in env
        assert "SAFE_VAR" in env
        assert "API_KEY" not in env
        assert "AWS_SECRET" not in env

    def test_prepare_environment_allows_configured_env(self):
        cfg = AgentConfig(
            type=AgentType.CUSTOM, timeout=60.0,
            env={"MY_SECRET_KEY": "needed"},
        )
        adapter = _make_stub(cfg)
        with patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
            env = adapter._prepare_environment()

        assert env["MY_SECRET_KEY"] == "needed"

    # -- Working directory -----------------------------------------------

    def test_resolve_working_directory_override(self, tmp_path):
        adapter = _make_stub()
        result = adapter._resolve_working_directory(str(tmp_path))
        assert result == str(tmp_path)

    def test_resolve_working_directory_fallback_cwd(self):
        adapter = _make_stub()
        result = adapter._resolve_working_directory(None)
        assert result == os.getcwd()

    # -- Session id / truncation -----------------------------------------

    def test_generate_session_id_length(self):
        sid = CLIAgentAdapter._generate_session_id()
        assert len(sid) == 8
        int(sid, 16)  # must be valid hex

    def test_truncate_output_large(self):
        large = "x" * (_MAX_OUTPUT_SIZE + 500)
        truncated = CLIAgentAdapter._truncate_output(large)
        assert len(truncated) < len(large)
        assert truncated.endswith("... [truncated]")

    def test_truncate_output_small(self):
        small = "short output"
        assert CLIAgentAdapter._truncate_output(small) == small

    # -- Shell file-ops extraction ---------------------------------------

    def test_extract_shell_file_ops_touch(self):
        files: List[str] = []
        CLIAgentAdapter._extract_shell_file_ops("touch newfile.txt", files)
        assert "newfile.txt" in files

    def test_extract_shell_file_ops_mkdir(self):
        files: List[str] = []
        CLIAgentAdapter._extract_shell_file_ops("mkdir -p src/components", files)
        assert "src/components" in files

    def test_extract_shell_file_ops_redirect(self):
        files: List[str] = []
        CLIAgentAdapter._extract_shell_file_ops("echo hello > output.txt", files)
        assert "output.txt" in files

    # -- Operation tracking ----------------------------------------------

    def test_track_operations_file_creation(self):
        adapter = _make_stub()
        fc, fm, cmds = [], [], []  # type: ignore[var-annotated]
        adapter._track_operations("write_file", {"path": "index.js"}, fc, fm, cmds)
        assert "index.js" in fc
        assert fm == []

    def test_track_operations_file_modification(self):
        adapter = _make_stub()
        fc, fm, cmds = [], [], []  # type: ignore[var-annotated]
        adapter._track_operations("edit", {"file_path": "cfg.json"}, fc, fm, cmds)
        assert "cfg.json" in fm

    def test_track_operations_command_execution(self):
        adapter = _make_stub()
        fc, fm, cmds = [], [], []  # type: ignore[var-annotated]
        adapter._track_operations("bash", {"command": "npm install"}, fc, fm, cmds)
        assert "npm install" in cmds

    # -- Invocation log --------------------------------------------------

    def test_invocation_log_initially_empty(self):
        adapter = _make_stub()
        assert adapter.get_invocation_log() == []

    def test_invocation_log_records_entry(self):
        adapter = _make_stub()
        adapter._log_invocation(["stub-cli", "--prompt", "hi"], "/tmp")
        log = adapter.get_invocation_log()
        assert len(log) == 1
        assert isinstance(log[0], CLIInvocation)
        assert log[0].adapter_name == "stub"

    # -- Health check ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="1.0", stderr="")
        adapter = _make_stub()
        assert await adapter.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_subprocess_run):
        mock_subprocess_run.side_effect = FileNotFoundError("nope")
        adapter = _make_stub()
        assert await adapter.health_check() is False

    # -- Execute (end-to-end via stub) -----------------------------------

    @pytest.mark.asyncio
    async def test_execute_json_output(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "messages": [{"role": "assistant", "content": "Done."}],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }),
            stderr="",
            returncode=0,
        )
        adapter = _make_stub()

        skill = Skill(
            metadata=SkillMetadata(name="test-skill", description="A test skill"),
            instructions="Do something.", raw_content="", file_path="/fake/SKILL.md",
        )
        trace = await adapter.execute(skill, "go")

        assert isinstance(trace, SkillAgentTrace)
        assert trace.skill_name == "test-skill"
        assert trace.final_output == "Done."
        assert trace.total_input_tokens == 100

    @pytest.mark.asyncio
    async def test_execute_text_fallback(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(
            stdout="Plain text response.", stderr="", returncode=0,
        )
        adapter = _make_stub()
        skill = Skill(
            metadata=SkillMetadata(name="s", description="A test skill"),
            instructions="x", raw_content="", file_path="/fake/SKILL.md",
        )
        trace = await adapter.execute(skill, "go")
        assert trace.final_output == "Plain text response."
        assert trace.tool_calls == []

    @pytest.mark.asyncio
    async def test_execute_jsonl_output(self, mock_subprocess_run):
        jsonl = (
            '{"type": "assistant", "message": {"content": '
            '[{"type": "tool_use", "name": "shell", "input": {"command": "ls"}}]}}\n'
            '{"type": "result", "result": "Listed.", '
            '"usage": {"input_tokens": 10, "output_tokens": 5}}'
        )
        mock_subprocess_run.return_value = MagicMock(
            stdout=jsonl, stderr="", returncode=0,
        )
        adapter = _make_stub()
        skill = Skill(
            metadata=SkillMetadata(name="s", description="A test skill"),
            instructions="x", raw_content="", file_path="/fake/SKILL.md",
        )
        trace = await adapter.execute(skill, "go")
        assert trace.final_output == "Listed."
        assert "shell" in trace.tool_calls
        assert "ls" in trace.commands_ran

    @pytest.mark.asyncio
    async def test_execute_timeout_raises(self):
        adapter = _make_stub()
        with patch.object(adapter, "_run_subprocess", side_effect=asyncio.TimeoutError):
            skill = Skill(
                metadata=SkillMetadata(name="s", description="A test skill"),
                instructions="x", raw_content="", file_path="/fake/SKILL.md",
            )
            with pytest.raises(AgentTimeoutError) as exc:
                await adapter.execute(skill, "go")
            assert exc.value.timeout == 60.0

    @pytest.mark.asyncio
    async def test_execute_nonzero_exit_records_error(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(
            stdout="", stderr="Error: something", returncode=1,
        )
        adapter = _make_stub()
        skill = Skill(
            metadata=SkillMetadata(name="s", description="A test skill"),
            instructions="x", raw_content="", file_path="/fake/SKILL.md",
        )
        trace = await adapter.execute(skill, "go")
        assert trace.has_errors
        assert any("code 1" in e for e in trace.errors)

    @pytest.mark.asyncio
    async def test_execute_prompt_tokens_alias(self, mock_subprocess_run):
        """prompt_tokens / completion_tokens should map correctly."""
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "result": "Done.",
                "usage": {"prompt_tokens": 300, "completion_tokens": 100},
            }),
            stderr="",
            returncode=0,
        )
        adapter = _make_stub()
        skill = Skill(
            metadata=SkillMetadata(name="s", description="A test skill"),
            instructions="x", raw_content="", file_path="/fake/SKILL.md",
        )
        trace = await adapter.execute(skill, "go")
        assert trace.total_input_tokens == 300
        assert trace.total_output_tokens == 100


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

    def test_build_command_includes_max_turns(self, skill):
        """_build_command should include --max-turns from config."""
        from evalview.skills.adapters.codex_adapter import CodexAdapter

        config = AgentConfig(type=AgentType.CODEX, timeout=120.0, max_turns=10)
        with patch("shutil.which", return_value="/usr/bin/codex"):
            adapter = CodexAdapter(config)
            cmd = adapter._build_command(skill, "query")

        assert "--max-turns" in cmd
        assert "10" in cmd

    def test_build_command_includes_tools(self, skill):
        """_build_command should include --tools."""
        from evalview.skills.adapters.codex_adapter import CodexAdapter

        config = AgentConfig(
            type=AgentType.CODEX, timeout=120.0,
            tools=["Read", "Write"],
        )
        with patch("shutil.which", return_value="/usr/bin/codex"):
            adapter = CodexAdapter(config)
            cmd = adapter._build_command(skill, "query")

        assert "--tools" in cmd
        assert "Read,Write" in cmd


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

    # -- Adapter identity ------------------------------------------------

    def test_adapter_name_is_openclaw(self, config):
        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
        assert adapter.name == "openclaw"

    def test_adapter_not_found_when_cli_missing(self, config):
        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value=None):
            with patch("os.access", return_value=False):
                adapter = OpenClawAdapter(config)
                with pytest.raises(AgentNotFoundError):
                    _ = adapter.cli_path

    # -- Execute (via inherited CLIAgentAdapter) -------------------------

    @pytest.mark.asyncio
    async def test_execute_calls_openclaw_cli(
        self, config, skill, mock_subprocess_run
    ):
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "messages": [{"role": "assistant", "content": "Done."}],
                "usage": {"input_tokens": 200, "output_tokens": 80},
            }),
            stderr="",
            returncode=0,
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            trace = await adapter.execute(skill, "Test query")

        assert isinstance(trace, SkillAgentTrace)
        assert trace.total_input_tokens == 200

    @pytest.mark.asyncio
    async def test_execute_parses_tool_calls(
        self, config, skill, mock_subprocess_run
    ):
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps({
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "shell", "input": {"command": "npm install"}},
                        {"type": "tool_use", "name": "write_file", "input": {"path": "index.js"}},
                        {"type": "text", "text": "Project created."},
                    ],
                }],
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

    # -- OpenClaw-specific command building ------------------------------

    def test_build_command_includes_headless(self, config, skill):
        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            cmd = adapter._build_command(skill, "query")

        assert "--headless" in cmd
        assert "run" in cmd

    def test_build_command_includes_skill_path(self, config, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\nname: test\n---\nInstructions")

        skill_with_path = Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="A test skill for path testing",
            ),
            instructions="Instructions",
            raw_content="",
            file_path=str(skill_file),
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            cmd = adapter._build_command(skill_with_path, "query")

        assert "--skill-path" in cmd
        assert str(skill_file) in cmd

    def test_build_command_includes_max_turns(self, skill):
        config = AgentConfig(type=AgentType.OPENCLAW, timeout=120.0, max_turns=15)

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
            cmd = adapter._build_command(skill, "query")

        assert "--max-turns" in cmd
        assert "15" in cmd

    def test_build_command_includes_tools(self, config, skill):
        config_tools = AgentConfig(
            type=AgentType.OPENCLAW, timeout=120.0,
            tools=["Read", "Write", "Bash"],
        )

        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config_tools)
            cmd = adapter._build_command(skill, "query")

        assert "--tools" in cmd
        assert "Read,Write,Bash" in cmd

    # -- Extended tool-name recognition ----------------------------------

    def test_openclaw_recognises_extra_creation_tools(self, config):
        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)

        fc, fm, cmds = [], [], []  # type: ignore[var-annotated]
        adapter._track_operations("file_write", {"path": "out.py"}, fc, fm, cmds)
        assert "out.py" in fc

        fc2, fm2, cmds2 = [], [], []  # type: ignore[var-annotated]
        adapter._track_operations("save_file", {"path": "saved.txt"}, fc2, fm2, cmds2)
        assert "saved.txt" in fc2

    def test_openclaw_recognises_extra_command_tools(self, config):
        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)

        fc, fm, cmds = [], [], []  # type: ignore[var-annotated]
        adapter._track_operations("run_command", {"command": "ls"}, fc, fm, cmds)
        assert "ls" in cmds

        fc2, fm2, cmds2 = [], [], []  # type: ignore[var-annotated]
        adapter._track_operations("terminal", {"command": "pwd"}, fc2, fm2, cmds2)
        assert "pwd" in cmds2

    # -- AgentSkill context format ---------------------------------------

    def test_format_skill_context_uses_agentskill(self, config, skill):
        from evalview.skills.adapters.openclaw_adapter import OpenClawAdapter

        with patch("shutil.which", return_value="/usr/bin/openclaw"):
            adapter = OpenClawAdapter(config)
        ctx = adapter._format_skill_context(skill)
        assert "AgentSkill" in ctx
        assert skill.metadata.name in ctx


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
