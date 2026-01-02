"""Interactive chat interface for EvalView.

Provides a conversational interface to run tests, generate test cases,
and explore evaluation results using natural language.
"""

import asyncio
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
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
            timeout=5
        )
        if result.returncode == 0:
            models = set()
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                if line.strip():
                    # First column is model name
                    model_name = line.split()[0]
                    models.add(model_name)
                    # Also add without tag (e.g., "llama3.1" for "llama3.1:latest")
                    if ":" in model_name:
                        models.add(model_name.split(":")[0])
            return models
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return set()


def get_project_context() -> str:
    """Gather context about the current project for the LLM."""
    context_parts = []

    # Find test cases
    test_dirs = ["tests/test-cases", "tests", "test-cases", "."]
    test_count = 0
    test_locations = []

    for test_dir in test_dirs:
        if os.path.isdir(test_dir):
            yaml_files = glob.glob(f"{test_dir}/**/*.yaml", recursive=True)
            yaml_files += glob.glob(f"{test_dir}/**/*.yml", recursive=True)
            # Filter out config files
            yaml_files = [f for f in yaml_files if "config" not in f.lower()]
            if yaml_files:
                test_count += len(yaml_files)
                test_locations.append(f"{test_dir}/ ({len(yaml_files)} files)")

    if test_count > 0:
        context_parts.append(f"- Found {test_count} test case(s) in: {', '.join(test_locations)}")
    else:
        context_parts.append("- No test cases found yet (use 'evalview init' or 'evalview quickstart')")

    # Check for .evalview directory
    evalview_dir = Path(".evalview")
    if evalview_dir.exists():
        # Check for results
        results_dir = evalview_dir / "results"
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json"))
            if result_files:
                # Get the most recent result
                latest = max(result_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        passed = data.get("passed", 0)
                        failed = data.get("failed", 0)
                        total = data.get("total", passed + failed)
                        context_parts.append(f"- Last run: {passed}/{total} passed, {failed} failed ({latest.name})")
                except (json.JSONDecodeError, KeyError):
                    context_parts.append(f"- Last run: {latest.name}")

        # Check for golden baseline
        golden_dir = evalview_dir / "golden"
        if golden_dir.exists() and list(golden_dir.glob("*.json")):
            context_parts.append("- Golden baseline exists (can use --diff for regression detection)")
        else:
            context_parts.append("- No golden baseline yet (save one with 'evalview golden save')")

        # Check for config
        config_file = evalview_dir / "config.yaml"
        if config_file.exists():
            context_parts.append("- Config file: .evalview/config.yaml")
    else:
        context_parts.append("- EvalView not initialized (run 'evalview init' or 'evalview quickstart')")

    # Check for examples directory
    if os.path.isdir("examples"):
        example_dirs = [d for d in os.listdir("examples") if os.path.isdir(f"examples/{d}")]
        if example_dirs:
            context_parts.append(f"- Example tests available: {', '.join(example_dirs[:5])}")

    return "\n".join(context_parts) if context_parts else "No project context available."


def get_command_key(cmd: str) -> str:
    """Get a key for command permission tracking.

    For 'evalview run examples/foo/' -> 'run'
    For 'evalview list' -> 'list'
    For 'evalview demo' -> 'demo'
    """
    parts = cmd.split()
    if len(parts) < 2:
        return cmd
    return parts[1]  # Return the subcommand


class CommandPermissions:
    """Track which commands the user has allowed to auto-run."""

    def __init__(self):
        self.always_allow: set[str] = set()
        # Pre-allow safe read-only commands
        self.always_allow.update(SAFE_COMMANDS)

    def is_allowed(self, cmd: str) -> bool:
        """Check if command is pre-allowed to run without confirmation."""
        key = get_command_key(cmd)
        return key in self.always_allow

    def allow_always(self, cmd: str) -> None:
        """Mark a command type as always allowed for this session."""
        key = get_command_key(cmd)
        self.always_allow.add(key)

    def get_allowed_list(self) -> list[str]:
        """Get list of always-allowed commands."""
        return sorted(self.always_allow)


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

## EXAMPLES IN THE REPO (use these exact paths)
- examples/goosebench/tasks/ - Tests for Block's Goose agent (10 tasks)
- examples/langgraph/ - LangGraph ReAct agent with search + calculator
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

    # Initialize command permissions
    permissions = CommandPermissions()

    # Show model quality warning for small Ollama models
    if llm_provider == LLMProvider.OLLAMA:
        model_name = model or PROVIDER_CONFIGS[llm_provider].default_model
        if any(small in model_name.lower() for small in SMALL_OLLAMA_MODELS):
            console.print(f"[yellow]Warning:[/yellow] Small model '{model_name}' may give inaccurate suggestions.")
            console.print(f"[dim]For better results, try: /model llama3:70b or /model mixtral[/dim]")
            console.print()

    # Show project context
    context = get_project_context()
    console.print("[bold]Project Status:[/bold]")
    console.print(f"[dim]{context}[/dim]")
    console.print()

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

            if user_input.lower() in ("help", "/help"):
                console.print("\n[bold]Chat Commands:[/bold]")
                console.print("  [cyan]/model[/cyan]         - Switch to a different model")
                console.print("  [cyan]/docs[/cyan]          - Open EvalView documentation")
                console.print("  [cyan]/cli[/cyan]           - Show CLI commands cheatsheet")
                console.print("  [cyan]/permissions[/cyan]   - Show auto-allowed commands")
                console.print("  [cyan]/context[/cyan]       - Show project status")
                console.print("  [cyan]clear[/cyan]          - Clear chat history")
                console.print("  [cyan]exit[/cyan]           - Leave chat")
                console.print("\n[bold]Tips:[/bold]")
                console.print("  - Ask how to test your agent")
                console.print("  - Ask to run specific tests")
                console.print("  - Ask to explain test failures")
                continue

            # /docs command - open documentation
            if user_input.lower() == "/docs":
                import webbrowser
                docs_url = "https://github.com/hidai25/evalview#readme"
                console.print(f"[dim]Opening documentation: {docs_url}[/dim]")
                webbrowser.open(docs_url)
                continue

            # /cli command - show CLI cheatsheet
            if user_input.lower() == "/cli":
                console.print("\n[bold]EvalView CLI Cheatsheet:[/bold]")
                console.print()
                console.print("[bold cyan]Getting Started:[/bold cyan]")
                console.print("  evalview quickstart        # Interactive setup wizard")
                console.print("  evalview init              # Initialize in current directory")
                console.print("  evalview demo              # See regression detection demo")
                console.print()
                console.print("[bold cyan]Running Tests:[/bold cyan]")
                console.print("  evalview run               # Run all tests")
                console.print("  evalview run <path>        # Run tests from specific path")
                console.print("  evalview run --verbose     # Detailed output")
                console.print("  evalview run --diff        # Compare against golden baseline")
                console.print()
                console.print("[bold cyan]Managing Baselines:[/bold cyan]")
                console.print("  evalview golden save <result.json>   # Save as baseline")
                console.print("  evalview golden list                 # List saved baselines")
                console.print("  evalview golden show <name>          # View baseline details")
                console.print()
                console.print("[bold cyan]Other Commands:[/bold cyan]")
                console.print("  evalview adapters          # List available adapters")
                console.print("  evalview list              # List all test cases")
                console.print("  evalview record            # Record agent interactions")
                console.print("  evalview --help            # Full help")
                console.print()
                continue

            # /model command - switch models mid-session
            if user_input.lower().startswith("/model"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    # Show model selection menu
                    console.print(f"\n[bold]Current model:[/bold] {session.model}")
                    console.print(f"[bold]Current provider:[/bold] {llm_provider.value}\n")

                    # Get installed Ollama models
                    installed = get_installed_ollama_models()

                    ollama_models = [
                        ("llama3.1:70b", "Best quality, needs 40GB+ RAM"),
                        ("mixtral", "Great balance, needs 25GB+ RAM"),
                        ("llama3.1:8b", "Good quality, needs 8GB+ RAM"),
                        ("qwen2:7b", "Fast, needs 8GB+ RAM"),
                    ]

                    console.print("[bold cyan]Ollama Models (free, local):[/bold cyan]")
                    for i, (model, desc) in enumerate(ollama_models, 1):
                        # Check if installed
                        model_base = model.split(":")[0]
                        is_installed = model in installed or model_base in installed
                        status = "[green]✓[/green]" if is_installed else "[dim]○[/dim]"
                        console.print(f"  {status} [cyan][{i}][/cyan] {model:<16} - {desc}")

                    if not installed:
                        console.print("  [dim]No models installed. Install: ollama pull llama3.1:8b[/dim]")
                    else:
                        console.print(f"  [dim]Installed: {', '.join(sorted(installed)[:5])}{'...' if len(installed) > 5 else ''}[/dim]")
                    console.print()

                    # Cloud models with API key status
                    has_openai = bool(os.getenv("OPENAI_API_KEY"))
                    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

                    console.print("[bold cyan]Cloud Models:[/bold cyan]")
                    openai_status = "[green]✓[/green]" if has_openai else "[yellow]![/yellow]"
                    anthropic_status = "[green]✓[/green]" if has_anthropic else "[yellow]![/yellow]"

                    console.print(f"  {openai_status} [cyan][5][/cyan] gpt-4o            - OpenAI, best overall")
                    console.print(f"  {openai_status} [cyan][6][/cyan] gpt-4o-mini       - OpenAI, fast & cheap")
                    console.print(f"  {anthropic_status} [cyan][7][/cyan] claude-sonnet-4-20250514  - Anthropic, excellent")
                    console.print(f"  {anthropic_status} [cyan][8][/cyan] claude-3-5-haiku-20241022 - Anthropic, fast")

                    if not has_openai and not has_anthropic:
                        console.print("\n[bold]API Key Setup:[/bold]")
                        console.print("  [dim]export OPENAI_API_KEY=sk-...[/dim]")
                        console.print("  [dim]export ANTHROPIC_API_KEY=sk-ant-...[/dim]")
                        console.print("  [dim]Or add to .env.local file[/dim]")
                    console.print()

                    choice = Prompt.ask("[dim]Select (1-8) or type model name, Enter to cancel[/dim]", default="")

                    if not choice:
                        continue

                    model_map = {
                        "1": ("llama3.1:70b", LLMProvider.OLLAMA),
                        "2": ("mixtral", LLMProvider.OLLAMA),
                        "3": ("llama3.1:8b", LLMProvider.OLLAMA),
                        "4": ("qwen2:7b", LLMProvider.OLLAMA),
                        "5": ("gpt-4o", LLMProvider.OPENAI),
                        "6": ("gpt-4o-mini", LLMProvider.OPENAI),
                        "7": ("claude-sonnet-4-20250514", LLMProvider.ANTHROPIC),
                        "8": ("claude-3-5-haiku-20241022", LLMProvider.ANTHROPIC),
                    }

                    if choice in model_map:
                        new_model, new_provider = model_map[choice]

                        # Check if provider/model is available
                        if new_provider == LLMProvider.OPENAI and not os.getenv("OPENAI_API_KEY"):
                            console.print("[red]Error:[/red] OPENAI_API_KEY not set")
                            console.print("[dim]Run: export OPENAI_API_KEY=sk-...[/dim]")
                        elif new_provider == LLMProvider.ANTHROPIC and not os.getenv("ANTHROPIC_API_KEY"):
                            console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set")
                            console.print("[dim]Run: export ANTHROPIC_API_KEY=sk-ant-...[/dim]")
                        elif new_provider == LLMProvider.OLLAMA:
                            if not is_ollama_running():
                                console.print("[red]Error:[/red] Ollama not running")
                                console.print("[dim]Run: ollama serve[/dim]")
                            else:
                                # Check if model is installed
                                model_base = new_model.split(":")[0]
                                if new_model not in installed and model_base not in installed:
                                    console.print(f"[yellow]Model '{new_model}' not installed.[/yellow]")
                                    console.print(f"[dim]Install it with: ollama pull {new_model}[/dim]")
                                    install = Prompt.ask("[dim]Install now? (y/n)[/dim]", default="y")
                                    if install.lower() in ("y", "yes", ""):
                                        console.print(f"[dim]Running: ollama pull {new_model}...[/dim]")
                                        result = subprocess.run(
                                            ["ollama", "pull", new_model],
                                            capture_output=False  # Show progress
                                        )
                                        if result.returncode == 0:
                                            session.model = new_model
                                            session.provider = new_provider
                                            llm_provider = new_provider
                                            console.print(f"[green]Installed and switched to {new_model}[/green]")
                                        else:
                                            console.print(f"[red]Failed to install {new_model}[/red]")
                                else:
                                    session.model = new_model
                                    session.provider = new_provider
                                    llm_provider = new_provider
                                    console.print(f"[green]Switched to {new_model} ({new_provider.value})[/green]")
                        else:
                            session.model = new_model
                            session.provider = new_provider
                            llm_provider = new_provider
                            console.print(f"[green]Switched to {new_model} ({new_provider.value})[/green]")
                    elif choice:
                        # Direct model name entry
                        session.model = choice
                        console.print(f"[green]Switched to model: {choice}[/green]")
                else:
                    new_model = parts[1].strip()
                    session.model = new_model
                    console.print(f"[green]Switched to model: {new_model}[/green]")
                continue

            # /permissions command - show what's auto-allowed
            if user_input.lower() == "/permissions":
                allowed = permissions.get_allowed_list()
                console.print("\n[bold]Auto-allowed commands:[/bold]")
                for cmd in allowed:
                    console.print(f"  [green]✓[/green] {cmd}")
                console.print("\n[dim]These commands run without asking. Use option [2] to add more.[/dim]")
                continue

            # /context command - show project status
            if user_input.lower() == "/context":
                context = get_project_context()
                console.print("\n[bold]Project Status:[/bold]")
                console.print(f"[dim]{context}[/dim]")
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

                # Check if command is pre-allowed
                should_run = False
                cmd_key = get_command_key(cmd)

                if permissions.is_allowed(cmd):
                    # Auto-run allowed commands
                    console.print()
                    console.print(f"[dim]Auto-running:[/dim] {cmd}")
                    should_run = True
                else:
                    # Ask for permission with 1/2/3 options
                    console.print()
                    console.print(f"[yellow]Run command?[/yellow] [bold]{cmd}[/bold]")
                    console.print(f"  [cyan][1][/cyan] Yes, run once")
                    console.print(f"  [cyan][2][/cyan] Always allow '[bold]{cmd_key}[/bold]' commands")
                    console.print(f"  [cyan][3][/cyan] Skip")
                    choice = Prompt.ask("[dim]Choice[/dim]", default="1")

                    if choice in ("1", "y", "yes", ""):
                        should_run = True
                    elif choice == "2":
                        permissions.allow_always(cmd)
                        console.print(f"[dim]'{cmd_key}' commands will auto-run for this session[/dim]")
                        should_run = True
                    # choice == "3" or anything else means skip

                if should_run:
                    console.print()
                    # Run command and capture output
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=os.getcwd(),
                        capture_output=True,
                        text=True
                    )

                    # Show the output
                    output = result.stdout + result.stderr
                    if output.strip():
                        console.print(output)

                    # Ask LLM to analyze the results
                    if output.strip():
                        console.print()
                        analyze = Prompt.ask("[yellow]Analyze results?[/yellow] [dim]y/n[/dim]", default="y")
                        if analyze.lower() in ("y", "yes", ""):
                            # Truncate output if too long
                            truncated = output[:4000] + "..." if len(output) > 4000 else output
                            analysis_prompt = f"I ran `{cmd}` and got this output:\n\n```\n{truncated}\n```\n\nBriefly summarize the results. Did tests pass or fail? Any issues to address?"

                            # Get analysis with live timer
                            analysis_start = time.time()
                            analysis_response = None

                            async def get_analysis():
                                nonlocal analysis_response
                                analysis_response = await session.get_response(analysis_prompt)

                            analysis_task = asyncio.create_task(get_analysis())

                            with Live(console=console, refresh_per_second=4, transient=True) as live:
                                while not analysis_task.done():
                                    elapsed = time.time() - analysis_start
                                    live.update(Text(f"  Analyzing... {elapsed:.1f}s", style="dim"))
                                    await asyncio.sleep(0.25)

                            await analysis_task

                            # Show analysis
                            analysis_elapsed = time.time() - analysis_start
                            analysis_tokens = session.last_tokens
                            print_separator(console)
                            console.print(f"[dim]  {analysis_elapsed:.1f}s  │  {analysis_tokens:,} tokens[/dim]")
                            print_separator(console)
                            console.print()
                            console.print("[bold cyan]EvalView[/bold cyan]")
                            console.print(Markdown(analysis_response))

        except KeyboardInterrupt:
            console.print("\n\n[dim]Use 'exit' to quit.[/dim]\n")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
            continue


async def run_demo(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Run a scripted demo for marketing videos.

    Uses pre-baked responses for instant, consistent playback.
    Perfect for recording demos - no LLM calls, no waiting.
    """
    import time
    from rich.live import Live
    from rich.text import Text

    console = Console()

    # Show banner
    print_banner(console, "Demo Mode")

    time.sleep(0.5)

    # Pre-baked demo script with static responses
    demo_steps = [
        {
            "user": "What can EvalView do?",
            "response": """EvalView catches **agent regressions** before you ship:

- **Tool changes** - detect when your agent uses different tools
- **Output changes** - catch when responses drift from baseline
- **Cost spikes** - alert on token/$ increases
- **Latency spikes** - monitor response time regressions

Think of it as **pytest for AI agents**.""",
            "tokens": 847,
            "time": 1.2,
        },
        {
            "user": "Show me how to catch a regression",
            "response": """Let me run the regression detection demo:

```command
evalview demo
```

This will compare a current run against a golden baseline and show you exactly what changed.""",
            "tokens": 412,
            "time": 0.8,
            "run_command": "evalview demo",
        },
    ]

    for i, step in enumerate(demo_steps, 1):
        # Show simulated user input with typing effect
        console.print()
        console.print("[bold green]You[/bold green]", end=" ")
        user_text = step["user"]
        for char in user_text:
            console.print(char, end="")
            time.sleep(0.02)  # Fast typing effect
        console.print()

        # Fake "thinking" animation
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            for j in range(int(step["time"] * 10)):
                dots = "." * ((j % 3) + 1)
                live.update(Text(f"  Thinking{dots}", style="dim"))
                time.sleep(0.1)

        # Show stats
        print_separator(console)
        console.print(f"[dim]  {step['time']:.1f}s  │  {step['tokens']:,} tokens[/dim]")
        print_separator(console)

        # Show response
        console.print()
        console.print("[bold cyan]EvalView[/bold cyan]")
        console.print(Markdown(step["response"]))

        # Run command if specified
        if "run_command" in step:
            cmd = step["run_command"]
            console.print()
            console.print(f"[dim]Auto-running:[/dim] {cmd}")
            time.sleep(0.3)
            console.print()

            # Run the actual command
            subprocess.run(cmd, shell=True, cwd=os.getcwd())

        time.sleep(1)

    console.print()
    console.print("[bold green]That's EvalView![/bold green] Catch regressions before your users do.")
    console.print("[dim]Try it: evalview quickstart[/dim]\n")


def main():
    """Entry point for chat command."""
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
