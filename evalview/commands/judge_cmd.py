"""Judge command — configure LLM-as-judge provider and model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import yaml

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("judge", hidden=True)
@click.argument("provider", required=False, type=click.Choice(["openai", "anthropic", "gemini", "grok", "ollama"]))
@click.argument("model", required=False)
@track_command("judge", lambda **kw: {"provider": kw.get("provider")})
def judge(provider: Optional[str], model: Optional[str]):
    """Set the LLM-as-judge provider and model.

    Examples:
        evalview judge                     # Show current judge config
        evalview judge openai              # Switch to OpenAI (default model)
        evalview judge openai gpt-4o       # Switch to OpenAI with specific model
        evalview judge anthropic           # Switch to Anthropic
        evalview judge ollama llama3.2     # Use local Ollama
    """
    config_path = Path(".evalview/config.yaml")

    # Load existing config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # If no provider specified, show current config
    if not provider:
        current = config.get("judge", {})
        if current:
            console.print("\n[bold]Current LLM-as-judge:[/bold]")
            console.print(f"  Provider: [cyan]{current.get('provider', 'not set')}[/cyan]")
            console.print(f"  Model: [cyan]{current.get('model', 'default')}[/cyan]\n")
        else:
            console.print("\n[dim]No judge configured. Using interactive selection.[/dim]")
            console.print("\n[bold]Set a judge:[/bold]")
            console.print("  evalview judge openai gpt-4o")
            console.print("  evalview judge anthropic claude-sonnet-4-5-20250929")
            console.print("  evalview judge ollama llama3.2\n")
        return

    # Default models per provider — from central config
    from evalview.core.llm_configs import DEFAULT_MODELS

    # Set the judge config
    config["judge"] = {
        "provider": provider,
        "model": model or DEFAULT_MODELS.get(provider, "default"),
    }

    # Ensure directory exists
    config_path.parent.mkdir(exist_ok=True)

    # Write config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]✓[/green] Judge set to [bold]{provider}[/bold] / [cyan]{config['judge']['model']}[/cyan]")
    console.print(f"[dim]  Saved to {config_path}[/dim]\n")
