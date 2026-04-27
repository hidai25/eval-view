"""Interactive chat interface for EvalView.

Provides a conversational interface to run tests, generate test cases,
and explore evaluation results using natural language.
"""

import asyncio
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from evalview.chat_slash import SLASH_COMMANDS, SlashCommandCompleter, show_slash_menu
from evalview.chat_runtime import (
    CommandPermissions,
    SMALL_OLLAMA_MODELS,
    extract_commands,
    extract_slash_commands,
    get_command_key,
    get_project_context,
    print_banner,
    print_separator,
    select_provider,
    validate_command,
)

from evalview.core.llm_provider import (
    LLMProvider,
    PROVIDER_CONFIGS,
    is_ollama_running,
)


from evalview.chat_prompt import SYSTEM_PROMPT  # noqa: F401  (re-exported for backward compat)
from evalview.chat_session import ChatSession  # noqa: F401  (re-exported for backward compat)
from evalview.chat_demo import run_demo  # noqa: F401  (re-exported for backward compat)
from evalview.chat_commands import (
    handle_adapters,
    handle_cli,
    handle_clear,
    handle_compare,
    handle_context,
    handle_docs,
    handle_help,
    handle_model,
    handle_permissions,
    handle_run,
    handle_skill,
    handle_test,
    handle_trace,
    handle_traces,
)


VALID_EVALVIEW_COMMANDS = {
    "demo", "run", "adapters", "list", "init",
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




async def run_chat(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_provider: Optional[str] = None,
) -> None:
    """Run the interactive chat interface."""
    # Set judge model/provider via env vars if specified (CLI overrides env)
    if judge_provider:
        os.environ["EVAL_PROVIDER"] = judge_provider
    if judge_model:
        from evalview.core.llm_provider import resolve_model_alias
        os.environ["EVAL_MODEL"] = resolve_model_alias(judge_model)

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
            console.print("[dim]For better results, try: /model llama3:70b or /model mixtral[/dim]")
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
        completer=SlashCommandCompleter(),
        key_bindings=kb,
        style=PromptStyle.from_dict({
            'prompt': f'{box_color}',
            'rprompt': f'{box_color}',
            'bottom-toolbar': f'noinherit {box_color}',
        })
    )

    # Telemetry: track chat session metrics
    _chat_message_count = 0
    _chat_slash_commands: Counter = Counter()
    _chat_start_time = time.perf_counter()

    def _send_chat_telemetry():
        """Send chat session telemetry event on exit."""
        try:
            from evalview.telemetry.config import is_telemetry_enabled
            from evalview.telemetry.events import ChatEvent
            from evalview.telemetry.client import get_client

            if not is_telemetry_enabled():
                return
            duration_ms = (time.perf_counter() - _chat_start_time) * 1000
            event = ChatEvent(
                provider=session.provider.value if session.provider else "",
                model=session.model or "",
                message_count=_chat_message_count,
                slash_commands_used=dict(_chat_slash_commands),
                duration_ms=duration_ms,
            )
            get_client().track(event)
        except Exception:
            pass  # Telemetry errors should never break functionality

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

            # Telemetry: count messages and slash commands
            _chat_message_count += 1
            if user_input.strip().startswith("/"):
                _cmd = user_input.strip().split()[0].lower()
                _chat_slash_commands[_cmd] += 1

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
                _send_chat_telemetry()
                break
            
            if user_input.lower() in ("help", "/help"):
                handle_help(console)
                continue

            # /docs command - open documentation
            if user_input.lower() == "/docs":
                handle_docs(console)
                continue

            # /cli command - show CLI cheatsheet
            if user_input.lower() == "/cli":
                handle_cli(console)
                continue

            # /adapters command - list available adapters
            if user_input.lower() == "/adapters":
                handle_adapters(console)
                continue

            # /run command - run a test case
            if user_input.lower().startswith("/run"):
                await handle_run(console, user_input)
                continue

            # /test command - quick ad-hoc test against an adapter
            if user_input.lower().startswith("/test"):
                await handle_test(console, user_input)
                continue

            # /traces command - list and query stored traces
            if user_input.lower().startswith("/traces"):
                handle_traces(console, user_input)
                continue

            # /trace command - trace LLM calls in a Python script
            if user_input.lower().startswith("/trace"):
                handle_trace(console, user_input)
                continue

            # /compare command - compare two test runs
            if user_input.lower().startswith("/compare"):
                handle_compare(console, user_input)
                continue

            # /skill command - test Claude Code skills with real agents
            if user_input.lower().startswith("/skill"):
                handle_skill(console, user_input)
                continue

            # /model command - switch models mid-session
            if user_input.lower().startswith("/model"):
                handle_model(console, session, user_input)
                continue

            # /permissions command - show what's auto-allowed
            if user_input.lower() == "/permissions":
                handle_permissions(console, permissions)
                continue

            # /context command - show project status
            if user_input.lower() == "/context":
                handle_context(console)
                continue

            if user_input.lower() in ("clear", "/clear"):
                handle_clear(console, session)
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
                is_valid, error_msg = validate_command(
                    cmd,
                    VALID_EVALVIEW_COMMANDS,
                    VALID_RUN_FLAGS,
                    VALID_DEMO_FLAGS,
                    VALID_ADAPTERS_FLAGS,
                    VALID_LIST_FLAGS,
                )
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
                    console.print("  [cyan][1][/cyan] Yes, run once")
                    console.print(f"  [cyan][2][/cyan] Always allow '[bold]{cmd_key}[/bold]' commands")
                    console.print("  [cyan][3][/cyan] Skip")
                    
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
                    cmd_output: str = proc.stdout + proc.stderr
                    if cmd_output.strip():
                        # Use a Panel for cleaner output display
                        console.print(Panel(cmd_output.strip(), title=f"Output: {cmd}", border_style="dim", expand=False))

                    # Ask LLM to analyze the results
                    if cmd_output.strip():
                        console.print()
                        
                        try:
                            analyze = await prompt_session.prompt_async(HTML("<yellow>Analyze results?</yellow> <dim>y/n</dim> "))
                        except KeyboardInterrupt:
                            analyze = "n"
                            
                        if analyze.lower() in ("y", "yes", ""):
                            # Truncate output if too long
                            truncated = cmd_output[:4000] + "..." if len(cmd_output) > 4000 else cmd_output
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

            # Check for slash commands in the LLM response
            slash_cmds = extract_slash_commands(full_response)
            for slash_cmd in slash_cmds:
                console.print()
                console.print(f"[yellow]Run command?[/yellow] [bold cyan]{slash_cmd}[/bold cyan]")
                console.print("  [cyan][1][/cyan] Yes, run it")
                console.print("  [cyan][2][/cyan] Skip")

                try:
                    choice = await prompt_session.prompt_async(HTML("<dim>Choice (1-2): </dim>"))
                except KeyboardInterrupt:
                    choice = "2"

                if choice in ("1", "y", "yes", ""):
                    # Inject the slash command to be processed
                    # We'll handle it inline here for simplicity
                    console.print()

                    if slash_cmd.lower().startswith("/adapters"):
                        # Run /adapters inline
                        from evalview.adapters.registry import AdapterRegistry
                        from rich.table import Table

                        adapters = AdapterRegistry.list_adapters()
                        table = Table(title="Available Adapters", show_header=True)
                        table.add_column("Adapter", style="cyan")
                        table.add_column("Description")
                        table.add_column("Tracing", justify="center")

                        descriptions = {
                            "http": "Generic REST API",
                            "langgraph": "LangGraph / LangGraph Cloud",
                            "crewai": "CrewAI multi-agent",
                            "anthropic": "Anthropic Claude API",
                            "claude": "Alias for anthropic",
                            "openai-assistants": "OpenAI Assistants API",
                            "tapescope": "JSONL streaming API",
                            "streaming": "Alias for tapescope",
                            "jsonl": "Alias for tapescope",
                            "huggingface": "HuggingFace Spaces",
                            "hf": "Alias for huggingface",
                            "gradio": "Alias for huggingface",
                            "goose": "Block's Goose CLI agent",
                            "mcp": "Model Context Protocol",
                            "ollama": "Ollama local LLMs",
                        }

                        for name in sorted(adapters.keys()):
                            desc = descriptions.get(name, "Custom adapter")
                            table.add_row(name, desc, "[green]✓[/green]")

                        console.print(table)
                        console.print(f"\n[dim]Total: {len(adapters)} adapters[/dim]")

                    elif slash_cmd.lower().startswith("/test"):
                        # Run /test inline
                        parts = slash_cmd.split(maxsplit=2)
                        if len(parts) >= 3:
                            adapter_type = parts[1].lower()
                            query = parts[2]

                            try:
                                from evalview.adapters.registry import AdapterRegistry
                                from evalview.reporters.trace_reporter import TraceReporter

                                endpoint = os.getenv("EVALVIEW_ENDPOINT", "")
                                default_endpoints = {
                                    "ollama": "http://localhost:11434",
                                    "langgraph": "http://localhost:2024",
                                    "http": "http://localhost:8000",
                                }
                                if not endpoint and adapter_type in default_endpoints:
                                    endpoint = default_endpoints[adapter_type]

                                console.print(f"[bold cyan]Testing with {adapter_type}[/bold cyan]")
                                console.print(f"[dim]Query: {query}[/dim]\n")

                                with console.status("[bold green]Executing...[/bold green]", spinner="dots"):
                                    adapter = AdapterRegistry.create(
                                        adapter_type,
                                        endpoint=endpoint,
                                        timeout=60.0,
                                        verbose=False,
                                    )
                                    trace = await adapter.execute(query)

                                console.print(f"[green]✓ Complete[/green] ({trace.metrics.total_latency:.0f}ms)")
                                if trace.metrics.total_cost:
                                    console.print(f"[dim]Cost: ${trace.metrics.total_cost:.4f}[/dim]")
                                console.print()

                                if trace.trace_context:
                                    reporter = TraceReporter()
                                    reporter.print_trace(trace.trace_context)

                                console.print("\n[bold]Response:[/bold]")
                                test_output = trace.final_output or "(empty)"
                                if len(test_output) > 1000:
                                    test_output = test_output[:1000] + "..."
                                console.print(Panel(test_output, border_style="green"))

                            except ValueError:
                                console.print(f"[red]Unknown adapter: {adapter_type}[/red]")
                                console.print("[dim]Run /adapters to see available adapters[/dim]")
                            except Exception as e:
                                console.print(f"[red]Error: {e}[/red]")
                        else:
                            console.print("[yellow]Usage: /test <adapter> <query>[/yellow]")

                    elif slash_cmd.lower().startswith("/run"):
                        # Run /run inline
                        parts = slash_cmd.split(maxsplit=1)
                        test_filter = parts[1].strip() if len(parts) > 1 else None

                        test_dirs = ["tests/test-cases", "tests", "test-cases", ".evalview/tests", "."]
                        test_files = []
                        for test_dir in test_dirs:
                            if Path(test_dir).exists():
                                test_files.extend(Path(test_dir).glob("*.yaml"))
                                test_files.extend(Path(test_dir).glob("*.yml"))

                        if not test_files:
                            console.print("[yellow]No test cases found.[/yellow]")
                        elif test_filter:
                            test_files = [f for f in test_files if test_filter.lower() in f.stem.lower()]
                            if test_files:
                                test_file = test_files[0]
                                console.print(f"[bold cyan]Running test: {test_file.stem}[/bold cyan]\n")

                                try:
                                    import yaml  # type: ignore[import-untyped]
                                    from evalview.adapters.registry import AdapterRegistry
                                    from evalview.core.types import TestCase
                                    from evalview.evaluators import Evaluator
                                    from evalview.reporters.trace_reporter import TraceReporter

                                    with open(test_file) as test_fh:
                                        test_data = yaml.safe_load(test_fh)

                                    test_case = TestCase(**test_data)
                                    adapter_type = test_case.adapter or "http"
                                    endpoint = test_case.endpoint or ""

                                    with console.status("[bold green]Executing...[/bold green]", spinner="dots"):
                                        run_timeout = (test_case.adapter_config or {}).get("timeout", 30.0)
                                        adapter = AdapterRegistry.create(
                                            adapter_type,
                                            endpoint=endpoint,
                                            timeout=run_timeout,
                                            verbose=False,
                                        )
                                        trace = await adapter.execute(
                                            test_case.input.query,
                                            test_case.input.context,
                                        )

                                    console.print("[green]✓ Execution complete[/green]")
                                    console.print(f"[dim]Latency: {trace.metrics.total_latency:.0f}ms[/dim]")
                                    console.print()

                                    if trace.trace_context:
                                        reporter = TraceReporter()
                                        reporter.print_trace(trace.trace_context)

                                    output_preview = trace.final_output[:500] if trace.final_output else "(empty)"
                                    console.print(Panel(output_preview, title="Agent Response", border_style="green"))

                                    if test_case.expected:
                                        evaluator = Evaluator()
                                        result = await evaluator.evaluate(test_case, trace)
                                        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
                                        console.print(f"\nResult: {status} (Score: {result.score:.0f})")

                                except Exception as e:
                                    console.print(f"[red]Error: {e}[/red]")
                            else:
                                console.print(f"[yellow]No tests matching '{test_filter}'[/yellow]")
                        else:
                            console.print("[bold]Available test cases:[/bold]")
                            for i, f in enumerate(test_files[:10], 1):
                                console.print(f"  [cyan][{i}][/cyan] {f.stem}")

                    elif slash_cmd.lower().startswith("/trace"):
                        # Run /trace inline
                        results_dir = Path(".evalview/results")
                        if results_dir.exists():
                            result_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if result_files:
                                from evalview.reporters.json_reporter import JSONReporter
                                from evalview.reporters.trace_reporter import TraceReporter
                                from evalview.core.types import EvaluationResult

                                latest = result_files[0]
                                console.print(f"[dim]Loading trace from {latest.name}...[/dim]\n")

                                results_data = JSONReporter.load(str(latest))
                                if results_data:
                                    results = [EvaluationResult(**data) for data in results_data]
                                    reporter = TraceReporter()
                                    for result in results[:3]:  # Show first 3
                                        console.print(f"[bold cyan]Test: {result.test_case}[/bold cyan]")
                                        reporter.print_trace_from_result(result)
                                        console.print()
                            else:
                                console.print("[yellow]No results found.[/yellow]")
                        else:
                            console.print("[yellow]No results found. Run some tests first![/yellow]")

                    elif slash_cmd.lower().startswith("/compare"):
                        # Run /compare inline
                        from rich.table import Table
                        results_dir = Path(".evalview/results")

                        if not results_dir.exists():
                            console.print("[yellow]No results found.[/yellow]")
                        else:
                            result_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if len(result_files) < 2:
                                console.print("[yellow]Need at least 2 runs to compare.[/yellow]")
                            else:
                                from evalview.reporters.json_reporter import JSONReporter
                                from evalview.core.types import EvaluationResult

                                file1, file2 = result_files[1], result_files[0]
                                data1 = JSONReporter.load(str(file1))
                                data2 = JSONReporter.load(str(file2))

                                results1 = {r["test_case"]: EvaluationResult(**r) for r in data1} if data1 else {}
                                results2 = {r["test_case"]: EvaluationResult(**r) for r in data2} if data2 else {}

                                console.print(f"\n[bold]Comparing:[/bold] {file1.name} → {file2.name}\n")

                                table = Table(show_header=True)
                                table.add_column("Test", style="cyan")
                                table.add_column("Old", justify="right")
                                table.add_column("New", justify="right")
                                table.add_column("Status")

                                for test in sorted(set(results1.keys()) | set(results2.keys())):
                                    r1, r2 = results1.get(test), results2.get(test)
                                    if r1 and r2:
                                        delta = r2.score - r1.score
                                        if delta < -5:
                                            status = "[red]↓ REGRESSED[/red]"
                                        elif delta > 5:
                                            status = "[green]↑ IMPROVED[/green]"
                                        else:
                                            status = "[dim]— same[/dim]"
                                        table.add_row(test[:25], f"{r1.score:.0f}", f"{r2.score:.0f}", status)
                                    elif r2:
                                        table.add_row(test[:25], "—", f"{r2.score:.0f}", "[cyan]NEW[/cyan]")
                                    elif r1:
                                        table.add_row(test[:25], f"{r1.score:.0f}", "—", "[yellow]REMOVED[/yellow]")

                                console.print(table)

        except KeyboardInterrupt:
            console.print("\n\n[dim]Use 'exit' to quit.[/dim]\n")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            _send_chat_telemetry()
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
            import traceback
            traceback.print_exc()
            continue


def main():
    """Entry point for chat command."""
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
