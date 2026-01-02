"""Interactive chat interface for EvalView.

Provides a conversational interface to run tests, generate test cases,
and explore evaluation results using natural language.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from typing import Optional, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from evalview.core.llm_provider import (
    LLMProvider,
    PROVIDER_CONFIGS,
    is_ollama_running,
    detect_available_providers,
)


SYSTEM_PROMPT = """You are EvalView Assistant - an expert on EvalView, a pytest-style testing framework for AI agents.

## WHAT EVALVIEW DOES
EvalView catches agent regressions before you ship:
- Tool changes (agent used different tools)
- Output changes (response differs from baseline)
- Cost spikes (tokens/$ increased)
- Latency spikes (response time increased)

## AVAILABLE ADAPTERS
| Adapter | Description | Needs Endpoint |
|---------|-------------|----------------|
| http | Generic REST API (default) | Yes |
| langgraph | LangGraph / LangGraph Cloud | Yes |
| crewai | CrewAI multi-agent | Yes |
| openai-assistants | OpenAI Assistants API | No (uses SDK) |
| anthropic / claude | Anthropic Claude API | Yes |
| huggingface / hf | HuggingFace Inference | Yes |
| goose | Block's Goose CLI agent | No (uses CLI) |
| tapescope / streaming | JSONL streaming API | Yes |
| mcp | Model Context Protocol | Yes |

## EXAMPLES IN THE REPO
- examples/goosebench/ - Tests for Block's Goose agent (10 tasks)
- examples/langgraph/agent/ - LangGraph ReAct agent with search + calculator
- examples/crewai/ - CrewAI multi-agent example
- examples/anthropic/ - Claude API example
- examples/openai-assistants/ - OpenAI Assistants example
- examples/huggingface/ - HuggingFace inference example

## HOW TO TEST GOOSE
```command
evalview run examples/goosebench/tasks/
```
Goose doesn't need a server - it runs via CLI. The goose adapter calls `goose run` directly.

## HOW TO TEST LANGGRAPH
1. Start the LangGraph agent:
   cd examples/langgraph/agent && langgraph dev
2. Run tests:
   evalview run examples/langgraph/ --verbose

## YAML TEST CASE SCHEMA
```yaml
name: "Test Name"
adapter: goose  # or http, langgraph, crewai, etc.
endpoint: http://localhost:8000  # if adapter needs it

input:
  query: "Your question here"
  context:
    extensions: ["developer"]  # for goose

expected:
  tools:
    - calculator
    - search
  tool_categories:
    - file_read
    - shell
  output:
    contains: ["expected", "words"]
    not_contains: ["error"]

thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 5000
```

## KEY COMMANDS
```command
evalview demo
```
Shows a demo of regression detection.

```command
evalview quickstart
```
Interactive setup wizard.

```command
evalview run
```
Run tests in tests/test-cases/.

```command
evalview run examples/goosebench/tasks/
```
Run tests from a specific path.

```command
evalview run --diff
```
Compare against golden baseline (detect regressions).

```command
evalview run --verbose
```
Show detailed output.

```command
evalview adapters
```
List all available adapters.

```command
evalview golden save .evalview/results/xxx.json
```
Save a run as baseline for regression detection.

## RULES
1. Put commands in ```command blocks so they can be executed
2. Answer questions using the knowledge above - don't hallucinate
3. For adapter questions, refer to the adapters table
4. For example questions, give the actual path from examples list
5. Keep responses concise but accurate
"""


class ChatSession:
    """Interactive chat session with EvalView assistant."""

    def __init__(
        self,
        provider: LLMProvider,
        model: Optional[str] = None,
        console: Optional[Console] = None,
    ):
        self.provider = provider
        self.model = model or PROVIDER_CONFIGS[provider].default_model
        self.console = console or Console()
        self.history: list[dict] = []
        self.total_tokens = 0
        self.last_tokens = 0

    async def get_response(self, user_message: str) -> str:
        """Get a response from the LLM."""
        self.history.append({"role": "user", "content": user_message})

        if self.provider == LLMProvider.OLLAMA:
            response, tokens = await self._ollama_chat()
        elif self.provider == LLMProvider.OPENAI:
            response, tokens = await self._openai_chat()
        elif self.provider == LLMProvider.ANTHROPIC:
            response, tokens = await self._anthropic_chat()
        else:
            response = f"Provider {self.provider.value} not yet supported for chat."
            tokens = 0

        self.last_tokens = tokens
        self.total_tokens += tokens
        self.history.append({"role": "assistant", "content": response})
        return response

    async def _ollama_chat(self) -> Tuple[str, int]:
        """Chat using Ollama."""
        from openai import AsyncOpenAI

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{ollama_host}/v1",
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )

        tokens = 0
        if response.usage:
            tokens = response.usage.total_tokens

        return response.choices[0].message.content or "", tokens

    async def _openai_chat(self) -> Tuple[str, int]:
        """Chat using OpenAI."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )

        tokens = 0
        if response.usage:
            tokens = response.usage.total_tokens

        return response.choices[0].message.content or "", tokens

    async def _anthropic_chat(self) -> Tuple[str, int]:
        """Chat using Anthropic."""
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = await client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=self.history,
            temperature=0.7,
        )

        tokens = 0
        if response.usage:
            tokens = response.usage.input_tokens + response.usage.output_tokens

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text, tokens


VALID_EVALVIEW_COMMANDS = {
    "demo", "run", "adapters", "quickstart", "list", "init",
    "report", "chat", "connect", "expand", "golden", "judge",
    "record", "trends", "validate-adapter", "skill", "add", "baseline"
}

VALID_RUN_FLAGS = {
    "--pattern", "--verbose", "--no-verbose", "--debug", "--sequential",
    "--track", "--compare-baseline", "--watch", "--summary", "--coverage",
    "--diff", "--strict", "-t", "--test", "-f", "--filter", "--output",
    "--max-workers", "--max-retries", "--retry-delay", "--html-report",
    "--judge-model", "--judge-provider", "--adapter", "--diff-report",
    "--fail-on", "--warn-on", "--help"
}


VALID_DEMO_FLAGS = {"--help"}
VALID_ADAPTERS_FLAGS = {"--help"}
VALID_LIST_FLAGS = {"--help", "--verbose", "-v"}


def validate_command(cmd: str) -> tuple[bool, str]:
    """Validate that a command is a valid evalview command."""
    if not cmd.startswith("evalview"):
        return False, "Not an evalview command"

    parts = cmd.split()
    if len(parts) < 2:
        return True, ""  # Just "evalview" is valid

    subcommand = parts[1]
    if subcommand.startswith("-"):
        # It's a flag like --help
        return True, ""

    if subcommand not in VALID_EVALVIEW_COMMANDS:
        return False, f"Unknown command: {subcommand}. Valid: {', '.join(sorted(VALID_EVALVIEW_COMMANDS))}"

    # Validate flags based on subcommand
    valid_flags = None
    if subcommand == "run":
        valid_flags = VALID_RUN_FLAGS
    elif subcommand == "demo":
        valid_flags = VALID_DEMO_FLAGS
    elif subcommand == "adapters":
        valid_flags = VALID_ADAPTERS_FLAGS
    elif subcommand == "list":
        valid_flags = VALID_LIST_FLAGS

    if valid_flags:
        for part in parts[2:]:
            if part.startswith("-"):
                # Extract just the flag name (before any =)
                flag = part.split("=")[0]
                if flag not in valid_flags:
                    return False, f"Unknown flag '{flag}' for '{subcommand}'. Use: evalview {subcommand} --help"

    return True, ""


def extract_commands(response: str) -> list[str]:
    """Extract executable commands from response."""
    commands = []
    # Match ```command ... ``` blocks
    pattern = r'```command\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    for match in matches:
        cmd = match.strip()
        if cmd.startswith("evalview"):
            commands.append(cmd)
    return commands


def select_provider(console: Console) -> Tuple[LLMProvider, str]:
    """Select which LLM provider to use for chat."""
    available = detect_available_providers()

    # Prefer Ollama if running (free)
    for provider, key in available:
        if provider == LLMProvider.OLLAMA:
            return provider, key

    # Otherwise use first available
    if available:
        provider, key = available[0]
        return provider, key

    # No provider available
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


async def run_chat(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Run the interactive chat interface."""
    console = Console()

    # Select provider
    if provider:
        # Use specified provider
        provider_enum = LLMProvider(provider)
        if provider_enum == LLMProvider.OLLAMA and not is_ollama_running():
            console.print("[red]Ollama is not running. Start with: ollama serve[/red]")
            return
        llm_provider = provider_enum
        api_key = "ollama" if provider_enum == LLMProvider.OLLAMA else os.getenv(PROVIDER_CONFIGS[provider_enum].env_var, "")
        provider_info = f"Using {PROVIDER_CONFIGS[llm_provider].display_name}"
    else:
        llm_provider, api_key = select_provider(console)
        provider_info = f"Using {PROVIDER_CONFIGS[llm_provider].display_name}"

    # Show banner with provider info
    print_banner(console, provider_info)

    # Create session
    session = ChatSession(
        provider=llm_provider,
        model=model,
        console=console,
    )

    # Track stats
    import time
    from rich.live import Live
    from rich.text import Text

    while True:
        try:
            # Get user input (clean prompt)
            console.print()
            user_input = Prompt.ask("[bold green]You[/bold green]")

            if not user_input.strip():
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "help":
                console.print("\n[bold]Tips:[/bold]")
                console.print("  - Ask how to test your agent")
                console.print("  - Ask to generate test cases")
                console.print("  - Ask to run specific tests")
                console.print("  - Ask to explain test failures")
                continue

            if user_input.lower() == "clear":
                session.history = []
                console.print("[dim]Chat history cleared.[/dim]")
                continue

            # Start timing this query
            query_start = time.time()
            response = None

            async def get_response_task():
                nonlocal response
                response = await session.get_response(user_input)

            # Run with live updating timer (per-query)
            import asyncio
            task = asyncio.create_task(get_response_task())

            with Live(console=console, refresh_per_second=4, transient=True) as live:
                while not task.done():
                    query_elapsed = time.time() - query_start
                    live.update(Text(f"  Thinking... {query_elapsed:.1f}s", style="dim"))
                    await asyncio.sleep(0.25)

            await task  # Ensure task is complete

            # Show final stats for this query (one line with separators)
            query_elapsed = time.time() - query_start
            query_tokens = session.last_tokens
            print_separator(console)
            console.print(f"[dim]  {query_elapsed:.1f}s  │  {query_tokens:,} tokens[/dim]")
            print_separator(console)

            # Display response
            console.print()
            console.print("[bold cyan]EvalView[/bold cyan]")
            console.print(Markdown(response))

            # Check for commands to execute
            commands = extract_commands(response)
            for cmd in commands:
                # Validate command before offering to run
                is_valid, error_msg = validate_command(cmd)
                if not is_valid:
                    console.print()
                    console.print(f"[red]Invalid command:[/red] {cmd}")
                    console.print(f"[dim]{error_msg}[/dim]")
                    continue

                console.print()
                console.print(f"[yellow]Run command?[/yellow] [dim]{cmd}[/dim]")
                confirm = Prompt.ask("[dim]y/n[/dim]", default="y")
                if confirm.lower() in ("y", "yes", ""):
                    console.print()
                    result = subprocess.run(cmd, shell=True, cwd=os.getcwd())

        except KeyboardInterrupt:
            console.print("\n\n[dim]Use 'exit' to quit.[/dim]\n")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
            continue


def main():
    """Entry point for chat command."""
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
