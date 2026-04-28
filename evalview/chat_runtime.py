"""Shared runtime helpers for the interactive chat interface."""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Tuple

from rich.console import Console

from evalview.core.llm_provider import (
    LLMProvider,
    detect_available_providers,
)

# Commands that are safe to auto-run without confirmation (read-only)
SAFE_COMMANDS = {"demo", "list", "adapters", "help", "--help", "--version"}

# Small models that may hallucinate - show warning
SMALL_OLLAMA_MODELS = {
    "llama3.2", "llama3.2:1b", "llama3.2:3b",
    "phi3", "phi3:mini", "gemma:2b", "gemma2:2b",
    "qwen2:0.5b", "qwen2:1.5b", "tinyllama"
}

# Recommended larger models for better results
RECOMMENDED_MODELS = ["llama3:70b", "mixtral", "qwen2:72b", "llama3.1:70b"]


def get_installed_ollama_models() -> set[str]:
    """Get list of installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return set()

        models: set[str] = set()
        for line in result.stdout.strip().split("\n")[1:]:
            if not line.strip():
                continue
            model_name = line.split()[0]
            models.add(model_name)
            if ":" in model_name:
                models.add(model_name.split(":")[0])
        return models
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()


def get_project_context() -> str:
    """Gather context about the current project for the LLM."""
    context_parts = []

    test_dirs = ["tests/test-cases", "tests", "test-cases", "."]
    test_count = 0
    test_locations = []

    for test_dir in test_dirs:
        if not os.path.isdir(test_dir):
            continue
        yaml_files = glob.glob(f"{test_dir}/**/*.yaml", recursive=True)
        yaml_files += glob.glob(f"{test_dir}/**/*.yml", recursive=True)
        yaml_files = [f for f in yaml_files if "config" not in f.lower()]
        if not yaml_files:
            continue
        test_count += len(yaml_files)
        test_locations.append(f"{test_dir}/ ({len(yaml_files)} files)")

    if test_count > 0:
        context_parts.append(f"- Found {test_count} test case(s) in: {', '.join(test_locations)}")
    else:
        context_parts.append("- No test cases found yet (use 'evalview init' or 'evalview demo')")

    evalview_dir = Path(".evalview")
    if evalview_dir.exists():
        results_dir = evalview_dir / "results"
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json"))
            if result_files:
                latest = max(result_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        passed = data.get("passed", 0)
                        failed = data.get("failed", 0)
                        total = data.get("total", passed + failed)
                        context_parts.append(
                            f"- Last run: {passed}/{total} passed, {failed} failed ({latest.name})"
                        )
                except (json.JSONDecodeError, KeyError):
                    context_parts.append(f"- Last run: {latest.name}")

        golden_dir = evalview_dir / "golden"
        if golden_dir.exists() and list(golden_dir.glob("*.json")):
            context_parts.append("- Golden baseline exists (can use --diff for regression detection)")
        else:
            context_parts.append("- No golden baseline yet (save one with 'evalview golden save')")

        if (evalview_dir / "config.yaml").exists():
            context_parts.append("- Config file: .evalview/config.yaml")
    else:
        context_parts.append("- EvalView not initialized (run 'evalview init' or 'evalview demo')")

    if os.path.isdir("examples"):
        example_dirs = [d for d in os.listdir("examples") if os.path.isdir(f"examples/{d}")]
        if example_dirs:
            context_parts.append(f"- Example tests available: {', '.join(example_dirs[:5])}")

    return "\n".join(context_parts) if context_parts else "No project context available."


def get_command_key(cmd: str) -> str:
    """Get a key for command permission tracking."""
    parts = cmd.split()
    if len(parts) < 2:
        return cmd
    return parts[1]


class CommandPermissions:
    """Track which commands the user has allowed to auto-run."""

    def __init__(self):
        self.always_allow: set[str] = set(SAFE_COMMANDS)

    def is_allowed(self, cmd: str) -> bool:
        return get_command_key(cmd) in self.always_allow

    def allow_always(self, cmd: str) -> None:
        self.always_allow.add(get_command_key(cmd))

    def get_allowed_list(self) -> list[str]:
        return sorted(self.always_allow)


def derive_chat_allowlists() -> Tuple[set[str], dict[str, set[str]]]:
    """Build the chat-validator allowlist directly from the Click registry.

    Returns ``(commands, flags_by_command)`` where ``commands`` is the set of
    valid top-level command names, and ``flags_by_command`` maps each command
    name to its set of valid flag spellings (e.g. ``"--verbose"``, ``"-v"``).

    Sourcing these from ``evalview.cli.main`` instead of hand-maintained sets
    means a new command or flag added to the Click registry is picked up
    automatically — the chat validator can never silently drift out of sync
    with the real CLI.
    """
    import click

    from evalview.cli import main

    commands: set[str] = set()
    flags_by_command: dict[str, set[str]] = {}

    for name, cmd in main.commands.items():
        if cmd.hidden:
            continue
        commands.add(name)
        flags: set[str] = {"--help"}
        for param in cmd.params:
            if isinstance(param, click.Option):
                flags.update(param.opts)
                flags.update(param.secondary_opts)
        flags_by_command[name] = flags

    return commands, flags_by_command


def validate_command(
    cmd: str,
    valid_commands: set[str],
    flags_by_command: dict[str, set[str]],
) -> Tuple[bool, str]:
    """Validate that a command is a valid evalview command."""
    if not cmd.startswith("evalview"):
        return False, "Not an evalview command"

    parts = cmd.split()
    if len(parts) < 2:
        return True, ""

    subcommand = parts[1]
    if subcommand.startswith("-"):
        return True, ""

    if subcommand not in valid_commands:
        return False, f"Unknown command: {subcommand}. Valid: {', '.join(sorted(valid_commands))}"

    valid_flags = flags_by_command.get(subcommand)
    if valid_flags:
        for part in parts[2:]:
            if not part.startswith("-"):
                continue
            flag = part.split("=")[0]
            if flag not in valid_flags:
                return False, f"Unknown flag '{flag}' for '{subcommand}'. Use: evalview {subcommand} --help"

    return True, ""


def extract_commands(response: str) -> list[str]:
    """Extract executable commands from response."""
    commands = []
    matches = re.findall(r"```command\s*\n(.*?)\n```", response, re.DOTALL)
    for match in matches:
        cmd = match.strip()
        if cmd.startswith("evalview"):
            commands.append(cmd)
    return commands


def extract_slash_commands(response: str) -> list[str]:
    """Extract slash commands from LLM response."""
    slash_commands = []
    patterns = [
        r"`(/(?:test|run|adapters|trace-script|trace|compare)\s*[^`]*)`",
        r"^(/(?:test|run|adapters|trace-script|trace|compare)\s*.*)$",
        r"\s(/(?:test|run|adapters|trace-script|trace|compare)\s+\S.*)(?:\s|$)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.MULTILINE)
        for match in matches:
            cmd = match.strip().rstrip("`.,;:")
            if cmd and cmd not in slash_commands:
                slash_commands.append(cmd)

    return slash_commands


def select_provider(console: Console) -> Tuple[LLMProvider, str]:
    """Select which LLM provider to use for chat."""
    available = detect_available_providers()

    for provider, key in available:
        if provider == LLMProvider.OLLAMA:
            return provider, key

    if available:
        return available[0]

    console.print("[red]No LLM provider available.[/red]")
    console.print("\nTo use chat mode, either:")
    console.print("  1. Start Ollama: [cyan]ollama serve[/cyan] (free)")
    console.print("  2. Set an API key: [cyan]export OPENAI_API_KEY=...[/cyan]")
    raise SystemExit(1)


def print_banner(console: Console, provider_info: str = "") -> None:
    """Print the EvalView chat banner."""
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗██╗   ██╗ █████╗ ██╗    ██╗   ██╗██╗███████╗██╗    ██╗[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔════╝██║   ██║██╔══██╗██║    ██║   ██║██║██╔════╝██║    ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]█████╗  ██║   ██║███████║██║    ██║   ██║██║█████╗  ██║ █╗ ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔══╝  ╚██╗ ██╔╝██╔══██║██║    ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗ ╚████╔╝ ██║  ██║███████╗╚████╔╝ ██║███████╗╚███╔███╔╝[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ [/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]              [bold yellow]Interactive Chat Mode[/bold yellow]                            [bold cyan]║[/bold cyan]")
    if provider_info:
        padded = f"  {provider_info}".ljust(66)
        console.print(f"[bold cyan]║[/bold cyan][dim]{padded}[/dim][bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [dim]Type 'exit' to leave • Type 'help' for tips[/dim]                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print()


def format_stats(elapsed_seconds: float, total_tokens: int) -> str:
    """Format the stats string."""
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    elapsed_str = f"{minutes}:{seconds:02d}"
    tokens_str = f"{total_tokens:,}"
    return f"  Elapsed: {elapsed_str}  │  Tokens: {tokens_str}"


def print_separator(console: Console) -> None:
    """Print a horizontal separator line."""
    console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
