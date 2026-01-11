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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text
from rich.style import Style
from rich.layout import Layout

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition


SLASH_COMMANDS = [
    ("/model", "Switch to a different model"),
    ("/trace", "View execution trace from last run"),
    ("/docs", "Open EvalView documentation"),
    ("/cli", "Show CLI commands cheatsheet"),
    ("/permissions", "Show auto-allowed commands"),
    ("/context", "Show project status"),
    ("/help", "Show help and tips"),
    ("/clear", "Clear chat history"),
    ("/exit", "Leave chat"),
]


def show_slash_menu(console: Console, selected: int = 0) -> Optional[str]:
    """Show slash command dropdown and let user select. Returns selected command or None."""
    import sys
    import tty
    import termios

    def get_key():
        """Get a single keypress."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # Escape sequence
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch2 == '[':
                    if ch3 == 'A': return 'up'
                    if ch3 == 'B': return 'down'
                return 'esc'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    while True:
        # Clear previous menu and redraw
        # Move cursor up by number of commands + 1 for the divider
        for _ in range(len(SLASH_COMMANDS) + 1):
            console.file.write("\033[F\033[K")

        # Draw menu
        console.print("[dim]─── Slash Commands ───[/dim]")
        for i, (cmd, desc) in enumerate(SLASH_COMMANDS):
            if i == selected:
                console.print(f"  [#22d3ee bold]▸ {cmd:<14}[/#22d3ee bold] [dim]{desc}[/dim]")
            else:
                console.print(f"    [dim]{cmd:<14} {desc}[/dim]")

        key = get_key()
        if key == 'up':
            selected = (selected - 1) % len(SLASH_COMMANDS)
        elif key == 'down':
            selected = (selected + 1) % len(SLASH_COMMANDS)
        elif key == '\r' or key == '\n':  # Enter
            return SLASH_COMMANDS[selected][0]
        elif key == '\x1b' or key == 'esc' or key == '\x03':  # Esc or Ctrl+C
            return None
        elif key == '\x7f' or key == '\x08':  # Backspace
            return None


class SlashCommandCompleter(Completer):
    """Autocomplete for slash commands like Claude Code."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only show completions when text starts with /
        if text.startswith("/"):
            for cmd, desc in SLASH_COMMANDS:
                if cmd.lower().startswith(text.lower()):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=cmd,
                        display_meta=desc,
                    )

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

## DEBUGGING WITH /trace
After running tests, users can type `/trace` to see detailed execution traces:
- LLM calls with token counts, costs, latency
- Tool calls with parameters and results
- Hierarchical view of agent execution

When users ask about debugging, test failures, or understanding what happened:
1. Suggest running `/trace` to see the execution details
2. Explain what the trace shows (LLM calls, tools, costs)
3. Help interpret trace output if they share it

Example workflow:
1. User runs tests: `evalview run`
2. Test fails or behaves unexpectedly
3. User types `/trace` to see what happened
4. You help analyze the trace output

## RULES
1. Put commands in ```command blocks so they can be executed
2. Answer questions using the knowledge above - don't hallucinate
3. For adapter questions, refer to the adapters table
4. For example questions, give the actual path from examples list
5. Keep responses concise but accurate
6. When debugging, suggest /trace to see execution details
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

    async def stream_response(self, user_message: str) -> AsyncGenerator[str, None]:
        """Get a response from the LLM via streaming."""
        self.history.append({"role": "user", "content": user_message})

        collected_text = ""
        
        try:
            if self.provider == LLMProvider.OLLAMA:
                stream_gen = self._stream_ollama()
            elif self.provider == LLMProvider.OPENAI:
                stream_gen = self._stream_openai()
            elif self.provider == LLMProvider.ANTHROPIC:
                stream_gen = self._stream_anthropic()
            else:
                yield f"Provider {self.provider.value} not yet supported for chat."
                return

            async for chunk in stream_gen:
                if chunk:
                    collected_text += chunk
                    yield chunk

            # Update tokens estimate (very rough approximation for now as streams differ)
            tokens = len(collected_text) // 4
            self.last_tokens = tokens
            self.total_tokens += tokens
            
            self.history.append({"role": "assistant", "content": collected_text})
            
        except Exception as e:
            error_msg = f"\n\n[Error: {str(e)}]"
            yield error_msg
            self.history.append({"role": "assistant", "content": error_msg})

    async def _stream_ollama(self) -> AsyncGenerator[str, None]:
        """Stream chat using Ollama."""
        from openai import AsyncOpenAI

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{ollama_host}/v1",
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_openai(self) -> AsyncGenerator[str, None]:
        """Stream chat using OpenAI."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_anthropic(self) -> AsyncGenerator[str, None]:
        """Stream chat using Anthropic."""
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        async with client.messages.stream(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=self.history,  # type: ignore[arg-type]
            temperature=0.7,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    
    # Keep old methods as simple aliases for backward compatibility if needed, 
    # but they are not used in the new loop
    async def get_response(self, user_message: str) -> str:
        text = ""
        async for chunk in self.stream_response(user_message):
            text += chunk
        return text


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

    # Initialize prompt_toolkit session with history and slash command completion
    history_file = Path.home() / ".evalview_history"
    # Electric cyan for a cool vibe
    box_color = "#22d3ee"  # Tailwind cyan-400

    # Track if we should show slash menu
    show_slash_dropdown = [False]  # Use list to allow mutation in closure

    # Create key bindings to detect / at start
    kb = KeyBindings()

    @kb.add('/')
    def handle_slash(event):
        """Detect / at start and signal to show menu."""
        buf = event.app.current_buffer
        text_before = buf.document.text_before_cursor
        buf.insert_text('/')
        # If / is at the beginning, signal to show dropdown after prompt exits
        if text_before == '':
            show_slash_dropdown[0] = True
            # Submit immediately to trigger the menu
            buf.validate_and_handle()

    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_file)),
        key_bindings=kb,
        style=PromptStyle.from_dict({
            'prompt': f'{box_color}',
            'rprompt': f'{box_color}',
            'bottom-toolbar': f'noinherit {box_color}',
        })
    )

    while True:
        try:
            # Format current directory for the prompt
            cwd_path = Path.cwd()
            cwd_name = cwd_path.name
            if cwd_path == Path.home():
                cwd_display = "~"
            else:
                cwd_display = f".../{cwd_name}"

            # Show input box frame BEFORE typing
            console.print()
            term_width = console.width or 80

            # Top border with "You" title - electric cyan
            title_text = "─ You "
            dashes_needed = term_width - len(title_text) - 2
            top_border = f"[#22d3ee]╭{title_text}{'─' * dashes_needed}╮[/#22d3ee]"
            console.print(top_border)

            # Prompt inside the "box" - vertical bars on sides
            prompt_html = HTML("<style fg='#22d3ee'>│</style> ")
            rprompt_html = HTML("<style fg='#22d3ee'>│</style>")

            # Bottom border + footer info
            bottom_border = "╰" + "─" * (term_width - 2) + "╯"
            # Footer: path left, model right
            left_info = f"  {cwd_display}"
            right_info = f"{session.model}  /model"
            info_spacing = term_width - len(left_info) - len(right_info)
            info_line = f"{left_info}{' ' * max(info_spacing, 2)}{right_info}"
            bottom_toolbar_html = HTML(
                f"<style fg='#22d3ee'>{bottom_border}</style>\n"
                f"<style fg='#6b7280'>{info_line}</style>"
            )

            try:
                user_input = await prompt_session.prompt_async(
                    prompt_html,
                    rprompt=rprompt_html,
                    bottom_toolbar=bottom_toolbar_html,
                )
            except KeyboardInterrupt:
                # Clear the box frame (1 blank + top + input + bottom + footer = 5 lines)
                for _ in range(5):
                    console.file.write("\033[F\033[K")
                show_slash_dropdown[0] = False
                continue

            # Check if user typed / and we should show the dropdown
            if show_slash_dropdown[0] and user_input == '/':
                show_slash_dropdown[0] = False

                # Clear the prompt area and redraw complete box with /
                for _ in range(5):
                    console.file.write("\033[F\033[K")

                # Redraw the complete box with / inside
                console.print(Panel(
                    "/",
                    title="[bold #22d3ee]You[/bold #22d3ee]",
                    title_align="left",
                    border_style="#22d3ee",
                    padding=(0, 1),
                    expand=True
                ))

                # Show the dropdown menu BELOW the box
                console.print("[dim]─── Slash Commands ───[/dim]")
                for i, (cmd, desc) in enumerate(SLASH_COMMANDS):
                    if i == 0:
                        console.print(f"  [#22d3ee bold]▸ {cmd:<14}[/#22d3ee bold] [dim]{desc}[/dim]")
                    else:
                        console.print(f"    [dim]{cmd:<14} {desc}[/dim]")

                # Let user select
                selected_cmd = show_slash_menu(console, selected=0)

                # Clear everything (box + menu) - box is 3 lines, menu is 9 lines
                total_lines = 3 + 1 + len(SLASH_COMMANDS)  # box + header + commands
                for _ in range(total_lines):
                    console.file.write("\033[F\033[K")

                if selected_cmd:
                    user_input = selected_cmd
                else:
                    # User cancelled, restart input loop
                    continue

            show_slash_dropdown[0] = False

            if not user_input.strip():
                # Clear the empty box
                for _ in range(5):
                    console.file.write("\033[F\033[K")
                continue

            # Clear the incomplete box
            lines_to_clear = 5 + user_input.count('\n')
            for _ in range(lines_to_clear):
                console.file.write("\033[F\033[K")

            # Create the complete Chat Box with content
            console.print(Panel(
                user_input,
                title="[bold #22d3ee]You[/bold #22d3ee]",
                title_align="left",
                border_style="#22d3ee",
                padding=(1, 1),
                expand=True
            ))

            # Footer: path on left, model on right with /model hint
            left_info = f"  {cwd_display}"
            right_info = f"{session.model}"
            hint = "/model"
            spacing = term_width - len(left_info) - len(right_info) - 2
            console.print(f"[dim]{left_info}{' ' * max(spacing, 2)}{right_info}[/dim]")
            console.print(f"[dim]{' ' * (term_width - len(hint) - 2)}{hint}[/dim]")

            if user_input.lower() in ("exit", "quit", "q", "/exit", "/quit"):
                console.print("\n[dim]Goodbye![/dim]")
                break
            
            if user_input.lower() in ("help", "/help"):
                console.print("\n[bold]Chat Commands:[/bold]")
                console.print("  [cyan]/model[/cyan]         - Switch to a different model")
                console.print("  [cyan]/trace[/cyan]         - View execution trace from last run")
                console.print("  [cyan]/trace <name>[/cyan]  - View trace filtered by test name")
                console.print("  [cyan]/docs[/cyan]          - Open EvalView documentation")
                console.print("  [cyan]/cli[/cyan]           - Show CLI commands cheatsheet")
                console.print("  [cyan]/permissions[/cyan]   - Show auto-allowed commands")
                console.print("  [cyan]/context[/cyan]       - Show project status")
                console.print("  [cyan]clear[/cyan]          - Clear chat history")
                console.print("  [cyan]exit[/cyan]           - Leave chat")
                console.print("\n[bold]Debugging:[/bold]")
                console.print("  - Run tests, then use /trace to see LLM calls")
                console.print("  - /trace shows tokens, costs, tool calls")
                console.print("  - Ask \"why did this test fail?\" for AI analysis")
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

            # /trace command - view execution trace
            if user_input.lower().startswith("/trace"):
                parts = user_input.split(maxsplit=1)
                test_filter = parts[1].strip() if len(parts) > 1 else None

                # Find latest results
                results_dir = Path(".evalview/results")
                if not results_dir.exists():
                    console.print("[yellow]No results found. Run some tests first![/yellow]")
                    console.print("[dim]Try: evalview run[/dim]")
                    continue

                result_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not result_files:
                    console.print("[yellow]No results found. Run some tests first![/yellow]")
                    continue

                latest = result_files[0]
                console.print(f"[dim]Loading trace from {latest.name}...[/dim]\n")

                try:
                    from evalview.reporters.json_reporter import JSONReporter
                    from evalview.reporters.trace_reporter import TraceReporter
                    from evalview.core.types import EvaluationResult

                    results_data = JSONReporter.load(str(latest))
                    if not results_data:
                        console.print("[yellow]No results in file[/yellow]")
                        continue

                    results = [EvaluationResult(**data) for data in results_data]

                    # Filter by test name if specified
                    if test_filter:
                        results = [r for r in results if test_filter.lower() in r.test_case.lower()]
                        if not results:
                            console.print(f"[yellow]No tests matching '{test_filter}'[/yellow]")
                            continue

                    reporter = TraceReporter()
                    for result in results:
                        console.print(f"[bold cyan]Test: {result.test_case}[/bold cyan]")
                        console.print(f"[dim]Score: {result.score:.0f} | Pass: {result.passed}[/dim]")
                        console.print()
                        reporter.print_trace_from_result(result)
                        console.print()

                except Exception as e:
                    console.print(f"[red]Error loading trace: {e}[/red]")
                continue

            # /model command - switch models mid-session
            if user_input.lower().startswith("/model"):
                # ... [keep existing model switching logic] ...
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

            if user_input.lower() in ("clear", "/clear"):
                session.history = []
                console.print("[dim]Chat history cleared.[/dim]")
                continue

            
            # Start timing this query
            query_start = time.time()
            full_response = ""

            # Spinner animation
            from rich.spinner import Spinner

            # Use Live to handle the spinner -> stream transition smoothly
            spinner = Spinner("dots", text=" Thinking...", style="cyan")

            with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
                stream_started = False

                async for chunk in session.stream_response(user_input):
                    if not stream_started:
                        # First chunk received: switch from spinner to text stream
                        stream_started = True
                        live.update(Markdown(""))

                    full_response += chunk
                    live.update(Markdown(full_response))

            # Calculate stats
            query_elapsed = time.time() - query_start
            query_tokens = session.last_tokens

            # Stats ABOVE the response (like Claude Code)
            print_separator(console)
            console.print(f"[dim]  {query_elapsed:.1f}s  │  {query_tokens:,} tokens (est)[/dim]")
            print_separator(console)

            # Now print the final response
            console.print()
            console.print(Markdown(full_response))
            console.print()  # Extra spacing before next input

            # Check for commands to execute
            commands = extract_commands(full_response)
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
                    
                    try:
                        choice = await prompt_session.prompt_async(HTML("<dim>Choice (1-3): </dim>"))
                    except KeyboardInterrupt:
                        choice = "3"

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
                    # Use the same spinner style for tool execution
                    with console.status(f"[bold green]Running {cmd}...[/bold green]", spinner="dots"):
                        proc = subprocess.run(
                            cmd,
                            shell=True,
                            cwd=os.getcwd(),
                            capture_output=True,
                            text=True
                        )

                    # Show the output
                    output: str = proc.stdout + proc.stderr
                    if output.strip():
                        # Use a Panel for cleaner output display
                        console.print(Panel(output.strip(), title=f"Output: {cmd}", border_style="dim", expand=False))

                    # Ask LLM to analyze the results
                    if output.strip():
                        console.print()
                        
                        try:
                            analyze = await prompt_session.prompt_async(HTML("<yellow>Analyze results?</yellow> <dim>y/n</dim> "))
                        except KeyboardInterrupt:
                            analyze = "n"
                            
                        if analyze.lower() in ("y", "yes", ""):
                            # Truncate output if too long
                            truncated = output[:4000] + "..." if len(output) > 4000 else output
                            analysis_prompt = f"I ran `{cmd}` and got this output:\n\n```\n{truncated}\n```\n\nBriefly summarize the results. Did tests pass or fail? Any issues to address?"

                            analysis_start = time.time()
                            analysis_full = ""

                            # Stream the analysis with spinner logic
                            analysis_spinner = Spinner("dots", text=" Analyzing...", style="cyan")

                            with Live(analysis_spinner, console=console, refresh_per_second=12, transient=True) as live:
                                stream_started = False
                                async for chunk in session.stream_response(analysis_prompt):
                                    if not stream_started:
                                        stream_started = True
                                        live.update(Markdown(""))
                                    analysis_full += chunk
                                    live.update(Markdown(analysis_full))

                            # Stats ABOVE the response
                            analysis_elapsed = time.time() - analysis_start
                            analysis_tokens = session.last_tokens
                            print_separator(console)
                            console.print(f"[dim]  {analysis_elapsed:.1f}s  │  {analysis_tokens:,} tokens (est)[/dim]")
                            print_separator(console)

                            # Print the response
                            console.print()
                            console.print(Markdown(analysis_full))

        except KeyboardInterrupt:
            console.print("\n\n[dim]Use 'exit' to quit.[/dim]\n")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
            import traceback
            traceback.print_exc()
            continue


async def run_demo(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    style: int = 1,
) -> None:
    """Run a scripted demo for marketing videos.

    Uses pre-baked responses for instant, consistent playback.
    Perfect for recording demos - no LLM calls, no waiting.
    Fully self-contained - no subprocess calls.

    Styles:
        1: "3am panic" - Emotional, relatable crisis scenario
        2: "Instant action" - One-liner straight to demo
        3: "Cost explosion" - Money-focused shock and relief
        4: "Chat UI" - Showcase interactive chat features
    """
    import time
    from rich.live import Live
    from rich.text import Text

    console = Console()

    # Style-specific banners
    banner_subtitles = {
        1: "Demo: The 3am Panic",
        2: "Demo: LangGraph Agent",
        3: "Demo: Cost Explosion",
        4: "Interactive Chat",
    }
    print_banner(console, banner_subtitles.get(style, "Demo Mode"))

    time.sleep(0.5)

    def show_user_input(text: str) -> None:
        """Show simulated user input with typing effect."""
        console.print()
        console.print("[bold green]You[/bold green]", end=" ")
        for char in text:
            console.print(char, end="")
            time.sleep(0.02)
        console.print()

    def show_thinking(duration: float) -> None:
        """Show thinking animation."""
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            for j in range(int(duration * 10)):
                dots = "." * ((j % 3) + 1)
                live.update(Text(f"  Thinking{dots}", style="dim"))
                time.sleep(0.1)

    def show_response(text: str, tokens: int, duration: float) -> None:
        """Show response with stats."""
        print_separator(console)
        console.print(f"[dim]  {duration:.1f}s  │  {tokens:,} tokens[/dim]")
        print_separator(console)
        console.print()
        console.print("[bold cyan]EvalView[/bold cyan]")
        console.print(Markdown(text))

    def show_regression_report(
        results: list[tuple[str, str, str]],
        cost_old: str,
        cost_new: str,
        cost_pct: str,
        latency_old: str,
        latency_new: str,
        latency_pct: str,
    ) -> None:
        """Show inline regression report."""
        console.print()
        console.print("[dim]Running regression check...[/dim]")
        time.sleep(0.3)
        for name, _, _ in results:
            console.print(f"[dim]  Analyzing {name}...[/dim]", end="")
            time.sleep(0.15)
            console.print("[dim] done[/dim]")
        console.print()

        # Report header
        console.print("━" * 68)
        console.print("[bold]                     Regression Report[/bold]")
        console.print("━" * 68)
        console.print()

        # Results
        for name, status, detail in results:
            if status == "PASSED":
                console.print(f"  [green]✓ PASSED[/green]         {name}")
            elif status == "TOOLS_CHANGED":
                console.print(f"  [yellow]⚠ TOOLS_CHANGED[/yellow]  {name:<16} {detail}")
            elif status == "OUTPUT_CHANGED":
                console.print(f"  [blue]~ OUTPUT_CHANGED[/blue] {name:<16} {detail}")
            elif status == "REGRESSION":
                console.print(f"  [red]✗ REGRESSION[/red]     {name:<16} {detail}")

        console.print()
        console.print(f"  Cost:    {cost_old} → {cost_new}  ({cost_pct})  [yellow]⚠[/yellow]")
        console.print(f"  Latency: {latency_old} → {latency_new}  ({latency_pct})  [yellow]⚠[/yellow]")
        console.print()
        console.print("━" * 68)
        console.print("  [red]❌ This would fail CI[/red]")
        console.print("━" * 68)

    # =========================================================================
    # DEMO 1: "3am panic" - Agent broke, what changed? (verbose)
    # =========================================================================
    if style == 1:
        show_user_input("My agent broke in production. Users are complaining. What changed since yesterday?")
        show_thinking(0.8)
        show_response(
            "Don't panic. Let me compare your current agent against yesterday's baseline.",
            523,
            0.8,
        )
        time.sleep(0.3)

        # Verbose test execution - focus on finding the regression
        tests: List[Dict[str, Any]] = [
            {
                "name": "auth-flow",
                "query": "Login with email test@example.com",
                "tools": ["validate_email", "create_session"],
                "answer": "Successfully logged in. Session token: sk-...",
                "status": "PASSED",
                "score": 95,
                "baseline_score": 95,
                "cost": 0.002,
                "latency": 0.8,
            },
            {
                "name": "search-query",
                "query": "Find products matching 'wireless headphones'",
                "tools": ["parse_query", "web_search", "db_search"],
                "answer": "Found 12 products matching your search...",
                "status": "TOOLS_CHANGED",
                "score": 89,
                "baseline_score": 91,
                "cost": 0.004,
                "latency": 1.2,
                "new_tool": "web_search",
            },
            {
                "name": "summarizer",
                "query": "Summarize customer feedback from last week",
                "tools": ["fetch_feedback", "analyze_sentiment"],
                "answer": "Customer feedback was mostly positive...",
                "status": "OUTPUT_CHANGED",
                "score": 82,
                "baseline_score": 88,
                "similarity": 72,
                "cost": 0.003,
                "latency": 0.9,
            },
            {
                "name": "checkout",
                "query": "Process order #12345 with payment",
                "tools": ["validate_cart", "process_payment"],
                "answer": "Error: Unable to process payment method...",
                "status": "REGRESSION",
                "score": 67,
                "baseline_score": 94,
                "cost": 0.005,
                "latency": 1.3,
            },
        ]

        console.print()
        for i, test in enumerate(tests):
            console.print(f"[bold]Test {i+1}/4:[/bold] {test['name']}")
            console.print(f"[dim]  Query:[/dim] \"{test['query']}\"")
            time.sleep(0.15)
            console.print(f"[dim]  Tools:[/dim] {' → '.join(test['tools'])}")
            console.print(f"[dim]  Answer:[/dim] \"{test['answer'][:45]}...\"")

            if test["status"] == "PASSED":
                console.print(f"  [green]✓ PASSED[/green]  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "TOOLS_CHANGED":
                console.print(f"  [yellow]⚠ TOOLS_CHANGED[/yellow]  +{test['new_tool']}  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "OUTPUT_CHANGED":
                console.print(f"  [blue]~ OUTPUT_CHANGED[/blue]  similarity: {test['similarity']}%  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "REGRESSION":
                drop = test['baseline_score'] - test['score']
                console.print(f"  [red]✗ REGRESSION[/red]  score: {test['baseline_score']} → {test['score']} [red](-{drop})[/red]  ${test['cost']:.3f}  {test['latency']}s")
            console.print()
            time.sleep(0.12)

        # Summary
        console.print("━" * 68)
        console.print("[bold]                        Summary[/bold]")
        console.print("━" * 68)
        console.print()
        console.print("  Tests:   [green]1 passed[/green]  [red]1 regression[/red]  [yellow]1 tools changed[/yellow]  [blue]1 output changed[/blue]")
        console.print("  Cost:    $0.014 total (was $0.008)")
        console.print("  Latency: 4.2s total (was 1.1s)")
        console.print()
        console.print("━" * 68)
        console.print("  [red]❌ checkout regressed: 94 → 67 (-27 points)[/red]")
        console.print("━" * 68)

        time.sleep(0.5)
        console.print()
        console.print("[bold green]Found it.[/bold green] The checkout flow broke. Fix it and run `evalview golden update checkout`")
        console.print("[dim]pip install evalview[/dim]")
        console.print()
        console.print("[dim]⭐ Star if this helped → github.com/hidai25/eval-view[/dim]\n")

    # =========================================================================
    # DEMO 2: "LangGraph agent" - Real framework, verbose output
    # =========================================================================
    elif style == 2:
        show_user_input("test my langgraph agent")
        show_thinking(0.5)
        show_response(
            "Running tests against your LangGraph agent on localhost:2024...",
            156,
            0.5,
        )
        time.sleep(0.3)

        # Verbose test execution output
        demo2_tests: List[Dict[str, Any]] = [
            {
                "name": "tavily-search",
                "query": "What is the weather in San Francisco?",
                "tools": ["tavily_search_results_json"],
                "answer": "The weather in San Francisco is currently 62°F...",
                "status": "PASSED",
                "score": 94,
                "cost": 0.003,
                "latency": 1.2,
            },
            {
                "name": "weather-query",
                "query": "Get the forecast for Tokyo this week",
                "tools": ["tavily_search_results_json"],
                "answer": "I don't have access to real-time weather...",
                "status": "FAILED",
                "score": 71,
                "expected_score": 88,
                "cost": 0.004,
                "latency": 0.9,
            },
            {
                "name": "rag-retrieval",
                "query": "Find documents about authentication",
                "tools": ["vector_search", "rerank_documents"],
                "answer": "Found 3 relevant documents about auth...",
                "status": "TOOLS_CHANGED",
                "score": 91,
                "cost": 0.002,
                "latency": 0.8,
                "new_tool": "rerank_documents",
            },
            {
                "name": "summarizer",
                "query": "Summarize the Q3 earnings report",
                "tools": ["tavily_search_results_json"],
                "answer": "Q3 earnings showed 15% revenue growth...",
                "status": "PASSED",
                "score": 96,
                "cost": 0.003,
                "latency": 1.1,
            },
        ]

        console.print()
        for i, test in enumerate(demo2_tests):
            console.print(f"[bold]Test {i+1}/4:[/bold] {test['name']}")
            console.print(f"[dim]  Query:[/dim] \"{test['query']}\"")
            time.sleep(0.2)
            console.print(f"[dim]  Tools:[/dim] {' → '.join(test['tools'])}")
            console.print(f"[dim]  Answer:[/dim] \"{test['answer'][:50]}...\"")

            if test["status"] == "PASSED":
                console.print(f"  [green]✓ PASSED[/green]  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "FAILED":
                console.print(f"  [red]✗ FAILED[/red]  score: {test['expected_score']} → {test['score']} (-{test['expected_score'] - test['score']})  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "TOOLS_CHANGED":
                console.print(f"  [yellow]⚠ TOOLS_CHANGED[/yellow]  +{test['new_tool']}  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            console.print()
            time.sleep(0.15)

        # Summary
        console.print("━" * 68)
        console.print("[bold]                        Summary[/bold]")
        console.print("━" * 68)
        console.print()
        console.print("  Tests:   [green]2 passed[/green]  [red]1 failed[/red]  [yellow]1 changed[/yellow]")
        console.print("  Cost:    $0.012 total")
        console.print("  Latency: 4.0s total (1.0s avg)")
        console.print()
        console.print("━" * 68)
        console.print("  [red]❌ 1 regression detected[/red]")
        console.print("━" * 68)

        time.sleep(0.5)
        console.print()
        console.print("[bold green]Done.[/bold green] Run `evalview golden update weather-query` after fixing.")
        console.print("[dim]pip install evalview[/dim]")
        console.print()
        console.print("[dim]⭐ Star if this helped → github.com/hidai25/eval-view[/dim]\n")

    # =========================================================================
    # DEMO 3: "Cost explosion" - $847 bill shock (verbose, cost-focused)
    # =========================================================================
    elif style == 3:
        show_user_input("My OpenAI bill is $847. Last month it was $12. What happened?")
        show_thinking(0.9)
        show_response(
            "$847 vs $12? That's a **70x spike**. Let me find which tests exploded.",
            634,
            0.9,
        )
        time.sleep(0.3)

        # Verbose test execution - focus on COSTS
        # Math: $12/month → $847/month at 30 runs/month
        # Old: $0.40/run, New: $28.23/run
        demo3_tests: List[Dict[str, Any]] = [
            {
                "name": "auth-flow",
                "query": "Authenticate user with OAuth",
                "tools": ["validate_token", "refresh_session"],
                "answer": "User authenticated successfully...",
                "status": "PASSED",
                "score": 96,
                "cost": 0.10,
                "baseline_cost": 0.10,
                "latency": 0.9,
            },
            {
                "name": "search-query",
                "query": "Search inventory for SKU-12345",
                "tools": ["query_parser", "db_lookup"],
                "answer": "Found 3 items matching SKU-12345...",
                "status": "PASSED",
                "score": 94,
                "cost": 0.10,
                "baseline_cost": 0.10,
                "latency": 1.1,
            },
            {
                "name": "doc-processor",
                "query": "Process and summarize the 50-page contract",
                "tools": ["pdf_extract", "chunk_text", "summarize"],
                "answer": "Contract summary: This agreement covers...",
                "status": "COST_SPIKE",
                "score": 91,
                "cost": 14.02,
                "baseline_cost": 0.10,
                "latency": 23.4,
            },
            {
                "name": "report-gen",
                "query": "Generate quarterly analytics report",
                "tools": ["fetch_metrics", "analyze_trends", "format_report"],
                "answer": "Q3 Report: Revenue up 12%, costs down...",
                "status": "COST_SPIKE",
                "score": 88,
                "cost": 14.01,
                "baseline_cost": 0.10,
                "latency": 19.8,
            },
        ]

        console.print()
        for i, test in enumerate(demo3_tests):
            console.print(f"[bold]Test {i+1}/4:[/bold] {test['name']}")
            console.print(f"[dim]  Query:[/dim] \"{test['query']}\"")
            time.sleep(0.15)
            console.print(f"[dim]  Tools:[/dim] {' → '.join(test['tools'])}")
            console.print(f"[dim]  Answer:[/dim] \"{test['answer'][:40]}...\"")

            if test["status"] == "PASSED":
                console.print(f"  [green]✓ PASSED[/green]  score: {test['score']}  [green]${test['cost']:.2f}[/green]  {test['latency']}s")
            elif test["status"] == "COST_SPIKE":
                cost_increase = test['cost'] / test['baseline_cost']
                console.print(f"  [red]💰 COST SPIKE[/red]  ${test['baseline_cost']:.2f} → [red]${test['cost']:.2f}[/red] ({cost_increase:.0f}x)  {test['latency']}s")
            console.print()
            time.sleep(0.12)

        # Summary - emphasize costs
        # New total: $0.10 + $0.10 + $14.02 + $14.01 = $28.23
        # Old total: $0.10 + $0.10 + $0.10 + $0.10 = $0.40
        console.print("━" * 68)
        console.print("[bold]                        Summary[/bold]")
        console.print("━" * 68)
        console.print()
        console.print("  Tests:   [green]2 passed[/green]  [red]2 cost spikes[/red]")
        console.print()
        console.print("  [bold]Cost breakdown:[/bold]")
        console.print("    auth-flow:     $0.10   [green](no change)[/green]")
        console.print("    search-query:  $0.10   [green](no change)[/green]")
        console.print("    doc-processor: [red]$14.02[/red]  [red](was $0.10 → 140x!)[/red]")
        console.print("    report-gen:    [red]$14.01[/red]  [red](was $0.10 → 140x!)[/red]")
        console.print()
        console.print("  [bold]Total:[/bold] $0.40 → [red]$28.23[/red] per run")
        console.print("  [bold]At 30 runs/month:[/bold] $12 → [red]$847[/red]")
        console.print()
        console.print("━" * 68)
        console.print("  [red]❌ 2 cost explosions detected[/red]")
        console.print("━" * 68)

        time.sleep(0.5)
        console.print()
        console.print("[bold green]Found it.[/bold green] Check doc-processor and report-gen for infinite loops or missing limits.")
        console.print("[dim]pip install evalview[/dim]")
        console.print()
        console.print("[dim]⭐ Star if this helped → github.com/hidai25/eval-view[/dim]\n")

    # =========================================================================
    # DEMO 4: "Chat UI" - Showcase the interactive chat experience
    # =========================================================================
    elif style == 4:
        term_width = console.width or 80

        def show_chat_box(text: str, typing: bool = True) -> None:
            """Show the beautiful chat box with fast typing effect."""
            # Top border
            title_text = "─ You "
            dashes = term_width - len(title_text) - 2
            console.print(f"[#22d3ee]╭{title_text}{'─' * dashes}╮[/#22d3ee]")
            console.print(f"[#22d3ee]│{' ' * (term_width - 2)}│[/#22d3ee]")

            # Type the text - FAST
            if typing:
                console.print(f"[#22d3ee]│[/#22d3ee] ", end="")
                for char in text:
                    console.print(char, end="", highlight=False)
                    time.sleep(0.012)  # Fast typing
                padding = term_width - len(text) - 4
                console.print(f"{' ' * padding}[#22d3ee]│[/#22d3ee]")
            else:
                padding = term_width - len(text) - 4
                console.print(f"[#22d3ee]│[/#22d3ee] {text}{' ' * padding}[#22d3ee]│[/#22d3ee]")

            console.print(f"[#22d3ee]│{' ' * (term_width - 2)}│[/#22d3ee]")
            console.print(f"[#22d3ee]╰{'─' * (term_width - 2)}╯[/#22d3ee]")
            console.print(f"[dim]  .../my-project{' ' * (term_width - 35)}llama3.2[/dim]")
            console.print(f"[dim]{' ' * (term_width - 8)}/model[/dim]")

        def show_slash_dropdown() -> None:
            """Show the slash command dropdown below the box."""
            # Show complete box with / inside
            console.print(Panel(
                "/",
                title="[bold #22d3ee]You[/bold #22d3ee]",
                title_align="left",
                border_style="#22d3ee",
                padding=(0, 1),
                expand=True
            ))
            time.sleep(0.15)

            # Dropdown appears BELOW the box
            commands = [
                ("/model", "Switch to a different model"),
                ("/docs", "Open EvalView documentation"),
                ("/cli", "Show CLI commands cheatsheet"),
                ("/help", "Show help and tips"),
            ]
            console.print("[dim]─── Slash Commands ───[/dim]")
            console.print(f"  [#22d3ee bold]▸ /model        [/#22d3ee bold] [dim]Switch to a different model[/dim]")
            for cmd, desc in commands[1:]:
                console.print(f"    [dim]{cmd:<14} {desc}[/dim]")

            time.sleep(0.8)

        def show_ai_response(text: str, tokens: int, duration: float) -> None:
            """Show AI response with fast streaming effect."""
            print_separator(console)
            console.print(f"[dim]  {duration:.1f}s  │  {tokens:,} tokens[/dim]")
            print_separator(console)
            console.print()

            # Stream the response word by word - FAST
            words = text.split()
            displayed = ""
            with Live(console=console, refresh_per_second=60, transient=False) as live:
                for i, word in enumerate(words):
                    displayed += word + " "
                    live.update(Markdown(displayed))
                    time.sleep(0.015)  # Super fast streaming

        # Scene 1: Show slash commands
        console.print()
        console.print("[dim]Type / to see available commands...[/dim]")
        time.sleep(0.4)
        show_slash_dropdown()

        # Clear and show actual question
        time.sleep(0.3)
        console.print()
        console.print()

        # Scene 2: Ask a question
        show_chat_box("How do I catch regressions before deploying?")
        time.sleep(0.15)
        show_thinking(0.3)

        show_ai_response(
            """Save a **golden baseline** from a working run, then compare future runs against it:

```bash
# 1. Save your current working state
evalview golden save .evalview/results/latest.json

# 2. Make changes to your agent

# 3. Run with --diff to catch regressions
evalview run --diff
```

This catches **tool changes**, **output drift**, **cost spikes**, and **latency issues** before they hit production.""",
            487,
            0.9,
        )

        time.sleep(0.4)

        # Scene 3: Follow-up
        console.print()
        show_chat_box("Run it now")
        time.sleep(0.15)

        # Show command execution
        console.print()
        console.print("[dim]Running:[/dim] evalview run --diff")
        time.sleep(0.15)

        # Quick test results
        quick_results = [
            ("auth-flow", "PASSED", "green"),
            ("search-query", "PASSED", "green"),
            ("checkout", "REGRESSION", "red"),
        ]
        console.print()
        for name, status, color in quick_results:
            time.sleep(0.1)
            icon = "✓" if status == "PASSED" else "✗"
            console.print(f"  [{color}]{icon} {status:<12}[/{color}] {name}")

        console.print()
        console.print("━" * 50)
        console.print("  [red]❌ 1 regression detected - blocked deploy[/red]")
        console.print("━" * 50)

        time.sleep(0.5)
        console.print()
        console.print("[bold #22d3ee]Ask anything. Get answers. Ship with confidence.[/bold #22d3ee]")
        console.print("[dim]pip install evalview && evalview chat[/dim]")
        console.print()
        console.print("[dim]⭐ Star if this helped → github.com/hidai25/eval-view[/dim]\n")


def main():
    """Entry point for chat command."""
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
