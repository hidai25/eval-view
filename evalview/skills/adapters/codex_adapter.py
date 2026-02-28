"""OpenAI Codex CLI adapter for skill testing.

Executes skills through the Codex CLI and captures structured traces.
Delegates subprocess management, output parsing, and environment
sanitisation to :class:`CLIAgentAdapter`.

Example::

    config = AgentConfig(type=AgentType.CODEX)
    adapter = CodexAdapter(config)
    trace = await adapter.execute(skill, "Create a React component")
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from evalview.skills.adapters.base import CLIAgentAdapter
from evalview.skills.types import Skill


class CodexAdapter(CLIAgentAdapter):
    """Adapter for executing skills through OpenAI Codex CLI.

    Thin subclass of :class:`CLIAgentAdapter` that provides Codex-specific
    binary resolution and command construction.  All execution, parsing,
    and trace-building logic lives in the shared base class.
    """

    # -- Required hooks --------------------------------------------------

    @property
    def name(self) -> str:
        return "codex"

    @property
    def binary_name(self) -> str:
        return "codex"

    def _install_hint(self) -> str:
        return (
            "Install Codex CLI: npm install -g @openai/codex\n"
            "Or visit: https://github.com/openai/codex"
        )

    def _candidate_paths(self) -> List[Path]:
        return [
            Path.home() / ".npm-global" / "bin" / "codex",
            Path.home() / ".local" / "bin" / "codex",
            Path("/usr/local/bin") / "codex",
            Path.home() / ".nvm" / "current" / "bin" / "codex",
        ]

    def _build_command(self, skill: Skill, query: str) -> List[str]:
        skill_context = self._format_skill_context(skill)

        command = [
            self.cli_path,
            "--prompt", query,
            "--instructions", skill_context,
            "--output-format", "json",
        ]

        if self.config.max_turns:
            command.extend(["--max-turns", str(self.config.max_turns)])

        if self.config.tools:
            command.extend(["--tools", ",".join(self.config.tools)])

        return command
