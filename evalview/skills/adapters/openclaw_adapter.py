"""OpenClaw adapter for skill testing.

Executes skills through the OpenClaw CLI and captures structured traces.
Delegates subprocess management, output parsing, and environment
sanitisation to :class:`CLIAgentAdapter`.

OpenClaw (https://github.com/openclaw/openclaw) is an open-source autonomous
AI agent that runs locally.  It uses AgentSkills (SKILL.md files) to extend
its capabilities.

Example::

    config = AgentConfig(type=AgentType.OPENCLAW)
    adapter = OpenClawAdapter(config)
    trace = await adapter.execute(skill, "Create a React component")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from evalview.skills.adapters.base import CLIAgentAdapter
from evalview.skills.types import Skill


class OpenClawAdapter(CLIAgentAdapter):
    """Adapter for executing skills through OpenClaw CLI.

    Thin subclass of :class:`CLIAgentAdapter` that provides OpenClaw-specific
    binary resolution, command construction, and extended tool-name recognition.
    All execution, parsing, and trace-building logic lives in the shared base.
    """

    # -- Required hooks --------------------------------------------------

    @property
    def name(self) -> str:
        return "openclaw"

    @property
    def binary_name(self) -> str:
        return "openclaw"

    def _install_hint(self) -> str:
        return (
            "Install OpenClaw: pip install openclaw\n"
            "Or visit: https://github.com/openclaw/openclaw"
        )

    def _candidate_paths(self) -> List[Path]:
        return [
            Path.home() / ".local" / "bin" / "openclaw",
            Path.home() / ".npm-global" / "bin" / "openclaw",
            Path("/usr/local/bin") / "openclaw",
            Path("/opt/homebrew/bin") / "openclaw",
            Path.home() / ".nvm" / "current" / "bin" / "openclaw",
            Path.home() / ".openclaw" / "bin" / "openclaw",
        ]

    def _build_command(self, skill: Skill, query: str) -> List[str]:
        skill_context = self._format_skill_context(skill)

        command = [
            self.cli_path,
            "run",
            "--prompt", query,
            "--instructions", skill_context,
            "--output-format", "json",
            "--headless",
        ]

        if self.config.max_turns:
            command.extend(["--max-turns", str(self.config.max_turns)])

        if self.config.tools:
            command.extend(["--tools", ",".join(self.config.tools)])

        if skill.file_path and os.path.isfile(skill.file_path):
            command.extend(["--skill-path", skill.file_path])

        return command

    # -- Extended tool-name sets for OpenClaw ----------------------------

    def _format_skill_context(self, skill: Skill) -> str:
        """Format using OpenClaw's AgentSkill terminology."""
        return (
            f"You have the following AgentSkill available:\n\n"
            f"{'━' * 80}\n"
            f"SKILL: {skill.metadata.name}\n"
            f"{'━' * 80}\n\n"
            f"{skill.metadata.description}\n\n"
            f"## Instructions\n\n"
            f"{skill.instructions}\n\n"
            f"{'━' * 80}\n\n"
            f"Follow the skill instructions above when responding "
            f"to the user's request.\n"
        )

    def _file_creation_tools(self) -> Tuple[str, ...]:
        return (
            "write", "write_file", "create_file", "str_replace_editor",
            "file_write", "save_file",
        )

    def _file_modification_tools(self) -> Tuple[str, ...]:
        return ("edit", "patch", "append", "insert", "file_edit")

    def _command_execution_tools(self) -> Tuple[str, ...]:
        return (
            "shell", "bash", "exec", "run", "execute",
            "run_command", "terminal", "cmd",
        )
