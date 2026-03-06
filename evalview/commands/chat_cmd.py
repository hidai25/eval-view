"""Chat command — interactive AI assistant for eval guidance."""
from __future__ import annotations

import asyncio
import sys
from typing import Optional

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("chat")
@click.option(
    "--provider",
    type=click.Choice(["ollama", "openai", "anthropic"]),
    default=None,
    help="LLM provider to use (default: auto-detect, prefers Ollama)",
)
@click.option(
    "--model",
    default=None,
    help="Model to use (default: provider's default)",
)
@click.option(
    "--judge-model",
    type=str,
    help="Model for LLM-as-judge (e.g., gpt-5, sonnet, llama-70b, gpt-4o). Aliases auto-resolve to full names.",
)
@click.option(
    "--judge-provider",
    type=click.Choice(["openai", "anthropic", "huggingface", "gemini", "grok", "ollama"]),
    help="Provider for LLM-as-judge evaluation (ollama = free local)",
)
@click.option("--demo_1", is_flag=True, help="Run '3am panic' demo")
@click.option("--demo_2", is_flag=True, help="Run 'instant action' demo")
@click.option("--demo_3", is_flag=True, help="Run 'cost explosion' demo")
@click.option("--demo_chat", is_flag=True, help="Run 'interactive chat' demo")
@track_command("chat", lambda **kw: {"provider": kw.get("provider"), "is_demo": any([kw.get("demo_1"), kw.get("demo_2"), kw.get("demo_3"), kw.get("demo_chat")])})
def chat(provider: str, model: str, judge_model: str, judge_provider: str, demo_1: bool, demo_2: bool, demo_3: bool, demo_chat: bool):
    """Interactive chat interface for EvalView.

    Ask questions about testing your AI agents in natural language.
    The assistant can help you:

    \b
    - Run test cases
    - Generate new test cases
    - Explain test failures
    - Suggest testing strategies

    Examples:

    \b
      evalview chat                    # Auto-detect provider (prefers Ollama)
      evalview chat --provider ollama  # Use Ollama (free, local)
      evalview chat --provider openai  # Use OpenAI
      evalview chat --demo_1           # "3am panic" demo
      evalview chat --demo_2           # "Instant action" demo
      evalview chat --demo_3           # "Cost explosion" demo
      evalview chat --demo_chat        # "Interactive chat" demo

    Type 'exit' or 'quit' to leave the chat.
    """
    from evalview.chat import run_chat, run_demo

    if demo_1:
        asyncio.run(run_demo(provider=provider, model=model, style=1))
    elif demo_2:
        asyncio.run(run_demo(provider=provider, model=model, style=2))
    elif demo_3:
        asyncio.run(run_demo(provider=provider, model=model, style=3))
    elif demo_chat:
        asyncio.run(run_demo(provider=provider, model=model, style=4))
    else:
        asyncio.run(run_chat(provider=provider, model=model, judge_model=judge_model, judge_provider=judge_provider))


@click.command("trace")
@click.option("--output", "-o", type=click.Path(), help="Save trace to file (JSONL format)")
@click.argument("script", type=click.Path(exists=True))
@click.argument("script_args", nargs=-1)
@track_command("trace")
def trace_cmd(output: Optional[str], script: str, script_args: tuple):
    """Trace LLM calls in any Python script.

    Automatically instruments OpenAI, Anthropic, and Ollama SDK calls
    to capture execution traces without code changes.

    \b
    Examples:
        evalview trace my_agent.py
        evalview trace -o trace.jsonl my_agent.py arg1 arg2
        evalview trace scripts/test.py --verbose

    The trace shows:
        - LLM API calls with token counts and costs
        - Call duration and latency
        - Model and provider information
        - Error details if calls fail
    """
    from evalview.trace_cmd import run_traced_command

    # Build command: python <script> [args...]
    cmd = ["python", script]
    cmd.extend(script_args)

    exit_code, trace_file = run_traced_command(
        command=cmd,
        output_path=output,
        console=console,
    )

    sys.exit(exit_code)
