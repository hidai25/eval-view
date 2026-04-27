"""Slash-command handlers for the interactive chat loop.

Each handler is a free coroutine extracted from chat.py's run_chat loop.
The dispatcher in run_chat owns the matching logic; handlers own the work.
A handler returning normally is equivalent to the original `continue`
inside the loop — control returns to the top of the loop afterwards.

State passed in:
- console: Rich Console for output
- session: ChatSession (its .model and .provider may be mutated by /model)
- permissions: CommandPermissions (mutated by the LLM-response flow, read by /permissions)
- user_input: the raw text the user typed for this iteration
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from evalview.chat_runtime import (
    CommandPermissions,
    get_installed_ollama_models,
)
from evalview.chat_session import ChatSession
from evalview.core.llm_provider import (
    LLMProvider,
    is_ollama_running,
)


# ───────────────────────────────────────────────────────────────────────────
# /docs
# ───────────────────────────────────────────────────────────────────────────


def handle_docs(console: Console) -> None:
    """Open the EvalView documentation in a web browser."""
    import webbrowser
    docs_url = "https://github.com/hidai25/evalview#readme"
    console.print(f"[dim]Opening documentation: {docs_url}[/dim]")
    webbrowser.open(docs_url)


# ───────────────────────────────────────────────────────────────────────────
# /cli
# ───────────────────────────────────────────────────────────────────────────


def handle_cli(console: Console) -> None:
    """Print a quick CLI cheatsheet."""
    console.print("\n[bold]EvalView CLI Cheatsheet:[/bold]")
    console.print()
    console.print("[bold cyan]Getting Started:[/bold cyan]")
    console.print("  evalview demo              # Live regression demo")
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


# ───────────────────────────────────────────────────────────────────────────
# /adapters
# ───────────────────────────────────────────────────────────────────────────


_ADAPTER_DESCRIPTIONS = {
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


def handle_adapters(console: Console) -> None:
    """Render a table of registered adapters with descriptions."""
    from evalview.adapters.registry import AdapterRegistry

    adapters = AdapterRegistry.list_adapters()

    table = Table(title="Available Adapters", show_header=True)
    table.add_column("Adapter", style="cyan")
    table.add_column("Description")
    table.add_column("Tracing", justify="center")

    for name in sorted(adapters.keys()):
        desc = _ADAPTER_DESCRIPTIONS.get(name, "Custom adapter")
        table.add_row(name, desc, "[green]✓[/green]")

    console.print(table)
    console.print(f"\n[dim]Total: {len(adapters)} adapters[/dim]")


# ───────────────────────────────────────────────────────────────────────────
# /run — execute a test case from disk
# ───────────────────────────────────────────────────────────────────────────


async def handle_run(console: Console, user_input: str) -> None:
    """Run a YAML-defined test case end-to-end."""
    parts = user_input.split()

    # Parse flags
    enable_live_trace = False
    test_filter: Optional[str] = None
    run_judge_model: Optional[str] = None
    run_judge_provider: Optional[str] = None

    i = 1
    while i < len(parts):
        part = parts[i]
        if part in ("--trace", "-t"):
            enable_live_trace = True
        elif part == "--judge-model" and i + 1 < len(parts):
            i += 1
            run_judge_model = parts[i]
        elif part == "--judge-provider" and i + 1 < len(parts):
            i += 1
            run_judge_provider = parts[i]
        elif not part.startswith("-"):
            test_filter = part
        i += 1

    # Apply per-run judge overrides to env vars
    if run_judge_provider:
        os.environ["EVAL_PROVIDER"] = run_judge_provider
    if run_judge_model:
        from evalview.core.llm_provider import resolve_model_alias
        os.environ["EVAL_MODEL"] = resolve_model_alias(run_judge_model)

    # Find test cases
    test_dirs = ["tests/test-cases", "tests", "test-cases", ".evalview/tests", "."]
    test_files: list[Path] = []

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            test_files.extend(Path(test_dir).glob("*.yaml"))
            test_files.extend(Path(test_dir).glob("*.yml"))

    if not test_files:
        console.print("[yellow]No test cases found.[/yellow]")
        console.print("[dim]Create one with: evalview init[/dim]")
        return

    # Filter if specified
    if test_filter:
        test_files = [f for f in test_files if test_filter.lower() in f.stem.lower()]
        if not test_files:
            console.print(f"[yellow]No tests matching '{test_filter}'[/yellow]")
            console.print("[dim]Available tests:[/dim]")
            for test_dir in test_dirs:
                if Path(test_dir).exists():
                    for f in Path(test_dir).glob("*.yaml"):
                        console.print(f"  [cyan]{f.stem}[/cyan]")
            return

    # If multiple tests and no filter, show selection
    if len(test_files) > 1 and not test_filter:
        console.print("[bold]Available test cases:[/bold]")
        for i, f in enumerate(test_files[:10], 1):
            console.print(f"  [cyan][{i}][/cyan] {f.stem}")
        if len(test_files) > 10:
            console.print(f"  [dim]... and {len(test_files) - 10} more[/dim]")
        console.print("\n[dim]Usage: /run <test-name>[/dim]")
        return

    # Run the test
    test_file = test_files[0]
    console.print(f"[bold cyan]Running test: {test_file.stem}[/bold cyan]")
    if enable_live_trace:
        console.print("[dim]Live tracing enabled[/dim]")
    console.print()

    try:
        import yaml  # type: ignore[import-untyped]
        from evalview.adapters.registry import AdapterRegistry
        from evalview.core.types import TestCase
        from evalview.evaluators import Evaluator
        from evalview.reporters.trace_reporter import TraceReporter

        # Load test case
        with open(test_file) as tc_file:
            test_data = yaml.safe_load(tc_file)

        test_case = TestCase(**test_data)
        adapter_type = test_case.adapter or "http"
        endpoint = test_case.endpoint or ""

        console.print(f"[dim]Adapter: {adapter_type}[/dim]")
        console.print(f"[dim]Query: {test_case.input.query[:100]}...[/dim]\n")

        # Create live trace reporter if enabled
        live_trace_reporter = None
        if enable_live_trace:
            from evalview.reporters.trace_live_reporter import create_trace_reporter
            live_trace_reporter = create_trace_reporter(console=console)

        # Create adapter
        try:
            timeout = (test_case.adapter_config or {}).get("timeout", 30.0)
            adapter = AdapterRegistry.create(
                adapter_type,
                endpoint=endpoint,
                timeout=timeout,
                verbose=True,
            )
        except Exception as e:
            console.print(f"[red]Failed to create adapter: {e}[/red]")
            if live_trace_reporter:
                live_trace_reporter.close()
            return

        # Execute
        console.print("[dim]Executing...[/dim]")
        trace = await adapter.execute(
            test_case.input.query,
            test_case.input.context,
        )

        # Show live trace if enabled
        if live_trace_reporter and trace:
            live_trace_reporter.report_from_execution_trace(trace, test_case.name)
            live_trace_reporter.close()

        console.print("\n[green]✓ Execution complete[/green]")
        console.print(f"[dim]Latency: {trace.metrics.total_latency:.0f}ms[/dim]")
        if trace.metrics.total_cost:
            console.print(f"[dim]Cost: ${trace.metrics.total_cost:.4f}[/dim]")
        console.print()

        # Show trace (standard view if live trace not enabled)
        if trace.trace_context and not enable_live_trace:
            reporter = TraceReporter()
            reporter.print_trace(trace.trace_context)

        # Show output
        console.print("\n[bold]Output:[/bold]")
        output_preview = trace.final_output[:500] if trace.final_output else "(empty)"
        console.print(Panel(output_preview, title="Agent Response", border_style="green"))

        # Run evaluation if expectations defined
        if test_case.expected:
            console.print("\n[bold]Evaluating...[/bold]")
            evaluator = Evaluator()
            result = await evaluator.evaluate(test_case, trace)

            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(f"\nResult: {status} (Score: {result.score:.0f})")

            if not result.passed and result.evaluations:
                console.print("[dim]Issues:[/dim]")
                tool_eval = result.evaluations.tool_accuracy
                if tool_eval.accuracy < 1.0:
                    issues = []
                    if tool_eval.missing:
                        issues.append(f"missing: {', '.join(tool_eval.missing)}")
                    if tool_eval.unexpected:
                        issues.append(f"unexpected: {', '.join(tool_eval.unexpected)}")
                    console.print(f"  • Tool accuracy: {'; '.join(issues) if issues else f'{tool_eval.accuracy:.0%}'}")
                seq_eval = result.evaluations.sequence_correctness
                if not seq_eval.correct:
                    console.print(f"  • Sequence: {', '.join(seq_eval.violations) if seq_eval.violations else 'incorrect order'}")

    except Exception as e:
        console.print(f"[red]Error running test: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


# ───────────────────────────────────────────────────────────────────────────
# /test — ad-hoc query against an adapter
# ───────────────────────────────────────────────────────────────────────────


_DEFAULT_ENDPOINTS = {
    "ollama": "http://localhost:11434",
    "langgraph": "http://localhost:2024",
    "http": "http://localhost:8000",
}


async def handle_test(console: Console, user_input: str) -> None:
    """Send an ad-hoc query through any adapter (no test case required)."""
    enable_live_trace = False
    test_input = user_input

    if " --trace " in user_input or " -t " in user_input:
        enable_live_trace = True
        test_input = user_input.replace(" --trace ", " ").replace(" -t ", " ")

    parts = test_input.split(maxsplit=2)

    if len(parts) < 3:
        console.print("[bold]Quick Test - Usage:[/bold]")
        console.print("  /test [--trace] <adapter> <query>")
        console.print()
        console.print("[bold]Examples:[/bold]")
        console.print("  /test ollama What is 2+2?")
        console.print("  /test --trace anthropic Explain quantum computing")
        console.print("  /test -t http What's the weather?")
        console.print()
        console.print("[dim]For http/langgraph/crewai, set endpoint first:[/dim]")
        console.print("  export EVALVIEW_ENDPOINT=http://localhost:8000")
        return

    adapter_type = parts[1].lower()
    query = parts[2]

    try:
        from evalview.adapters.registry import AdapterRegistry
        from evalview.reporters.trace_reporter import TraceReporter

        endpoint = os.getenv("EVALVIEW_ENDPOINT", "")
        if not endpoint and adapter_type in _DEFAULT_ENDPOINTS:
            endpoint = _DEFAULT_ENDPOINTS[adapter_type]

        console.print(f"[bold cyan]Testing with {adapter_type}[/bold cyan]")
        if enable_live_trace:
            console.print("[dim]Live tracing enabled[/dim]")
        console.print(f"[dim]Query: {query}[/dim]")
        if endpoint:
            console.print(f"[dim]Endpoint: {endpoint}[/dim]")
        console.print()

        live_trace_reporter = None
        if enable_live_trace:
            from evalview.reporters.trace_live_reporter import create_trace_reporter
            live_trace_reporter = create_trace_reporter(console=console)

        adapter = AdapterRegistry.create(
            adapter_type,
            endpoint=endpoint,
            timeout=60.0,
            verbose=True,
        )

        console.print("[dim]Executing...[/dim]\n")
        trace = await adapter.execute(query)

        if live_trace_reporter and trace:
            live_trace_reporter.report_from_execution_trace(trace, f"test-{adapter_type}")
            live_trace_reporter.close()

        console.print(f"[green]✓ Complete[/green] ({trace.metrics.total_latency:.0f}ms)")
        if trace.metrics.total_cost:
            console.print(f"[dim]Cost: ${trace.metrics.total_cost:.4f}[/dim]")
        console.print()

        if trace.trace_context and not enable_live_trace:
            reporter = TraceReporter()
            reporter.print_trace(trace.trace_context)

        console.print("\n[bold]Response:[/bold]")
        response_output = trace.final_output or "(empty)"
        if len(response_output) > 1000:
            response_output = response_output[:1000] + "..."
        console.print(Panel(response_output, border_style="green"))

    except ValueError:
        console.print(f"[red]Unknown adapter: {adapter_type}[/red]")
        console.print("[dim]Run /adapters to see available adapters[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


# ───────────────────────────────────────────────────────────────────────────
# /traces — query stored trace data
# ───────────────────────────────────────────────────────────────────────────


def handle_traces(console: Console, user_input: str) -> None:
    """List, inspect, export, or summarize stored traces."""
    parts = user_input.split(maxsplit=1)
    subcommand = parts[1].strip() if len(parts) > 1 else None

    try:
        from evalview.storage import TraceDB

        with TraceDB() as db:
            # /traces cost - show cost report
            if subcommand and subcommand.lower() == "cost":
                report = db.get_cost_report(last_days=7)
                totals = report["totals"]
                total_cost = totals.get("total_cost") or 0
                total_calls = totals.get("total_calls") or 0

                console.print("[bold cyan]━━━ Cost Report (Last 7 Days) ━━━[/bold cyan]")
                console.print()

                cost_str = f"${total_cost:.4f}" if total_cost < 0.01 and total_cost > 0 else f"${total_cost:.2f}"
                console.print(f"[bold]Total:[/bold]     {cost_str} across {total_calls:,} LLM calls")
                console.print()

                models = report.get("by_model", [])
                if models:
                    console.print("[bold]By Model:[/bold]")
                    max_cost = max((m.get("total_cost") or 0) for m in models) if models else 1
                    for m in models[:10]:
                        model_name = m.get("model") or "unknown"
                        model_cost = m.get("total_cost") or 0
                        pct = (model_cost / total_cost * 100) if total_cost > 0 else 0
                        mc_str = f"${model_cost:.4f}" if model_cost < 0.01 and model_cost > 0 else f"${model_cost:.2f}"
                        bar_width = 16
                        filled = int((model_cost / max_cost) * bar_width) if max_cost > 0 else 0
                        bar = "█" * filled + "░" * (bar_width - filled)
                        console.print(f"  {model_name:<22} {mc_str:>8}  ({pct:>4.0f}%)  {bar}")
                    console.print()

            # /traces export <id> - export trace to HTML
            elif subcommand and subcommand.lower().startswith("export"):
                export_parts = subcommand.split(maxsplit=1)
                if len(export_parts) < 2:
                    console.print("[bold]Usage:[/bold] /traces export <trace_id>")
                    console.print("[dim]Exports trace to HTML file[/dim]")
                    return

                export_id = export_parts[1].strip()
                trace_data = db.get_trace(export_id)
                if not trace_data:
                    console.print(f"[red]Trace not found: {export_id}[/red]")
                    return

                spans = db.get_trace_spans(export_id)

                try:
                    from evalview.exporters import TraceHTMLExporter
                    exporter = TraceHTMLExporter()
                    output_path = f"trace_{export_id}.html"
                    exporter.export(trace_data, spans, output_path)
                    console.print(f"[green]Exported to: {output_path}[/green]")
                except ImportError:
                    console.print("[red]HTML export requires jinja2. Install with:[/red]")
                    console.print("  pip install evalview[reports]")

            # /traces <id> - show specific trace
            elif subcommand and len(subcommand) >= 4 and not subcommand.startswith("-"):
                trace_data = db.get_trace(subcommand)
                if not trace_data:
                    console.print(f"[red]Trace not found: {subcommand}[/red]")
                    return

                spans = db.get_trace_spans(subcommand)

                console.print("[bold cyan]━━━ Trace Details ━━━[/bold cyan]")
                console.print()
                console.print(f"[bold]Trace ID:[/bold]     {trace_data['run_id']}")
                console.print(f"[bold]Created:[/bold]      {trace_data['created_at'][:19].replace('T', ' ')}")
                if trace_data.get("script_name"):
                    console.print(f"[bold]Script:[/bold]       {trace_data['script_name']}")
                console.print()

                console.print("[bold]Summary:[/bold]")
                console.print(f"  Total calls:    {trace_data.get('total_calls', 0)}")
                tokens = trace_data.get("total_tokens", 0)
                in_tokens = trace_data.get("total_input_tokens", 0)
                out_tokens = trace_data.get("total_output_tokens", 0)
                console.print(f"  Total tokens:   {tokens:,} (in: {in_tokens:,} / out: {out_tokens:,})")
                cost = trace_data.get("total_cost", 0)
                cost_str = f"${cost:.4f}" if cost < 0.01 and cost > 0 else f"${cost:.2f}"
                console.print(f"  Total cost:     {cost_str}")
                console.print()

                if spans:
                    console.print("[bold]LLM Calls:[/bold]")
                    for i, span in enumerate(spans, 1):
                        if span.get("span_type") == "llm":
                            model = span.get("model", "unknown")
                            duration = span.get("duration_ms", 0)
                            dur_str = f"{duration:.0f}ms" if duration < 1000 else f"{duration/1000:.1f}s"
                            span_cost = span.get("cost_usd", 0)
                            span_cost_str = f"${span_cost:.4f}" if span_cost < 0.01 and span_cost > 0 else f"${span_cost:.2f}"
                            status = span.get("status", "success")
                            status_icon = "✓" if status == "success" else "✗"
                            console.print(f"  {i}. {status_icon} {model:<25} {dur_str:>8}  {span_cost_str}")
                console.print()

            # /traces - list recent traces
            else:
                traces_data = db.list_traces(limit=20)

                if not traces_data:
                    console.print("[dim]No traces found.[/dim]")
                    console.print("[dim]Run '/trace <script.py>' to capture traces.[/dim]")
                    return

                console.print("[bold cyan]━━━ Recent Traces ━━━[/bold cyan]")
                console.print()

                for tr in traces_data:
                    created = tr["created_at"][:16].replace("T", " ")
                    cost = tr.get("total_cost", 0)
                    cost_str = f"${cost:.4f}" if cost < 0.01 and cost > 0 else f"${cost:.2f}"
                    script = tr.get("script_name") or "-"
                    console.print(
                        f"[bold]{tr['run_id']}[/bold]  {created}  "
                        f"{tr.get('total_calls', 0)} calls  {cost_str}  [dim]{script}[/dim]"
                    )

                console.print()
                console.print("[dim]Use '/traces <id>' for details, '/traces export <id>' to export HTML, '/traces cost' for cost report[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ───────────────────────────────────────────────────────────────────────────
# /trace — instrument a Python script
# ───────────────────────────────────────────────────────────────────────────


def handle_trace(console: Console, user_input: str) -> None:
    """Run a Python script with LLM-call instrumentation."""
    parts = user_input.split(maxsplit=1)

    if len(parts) < 2:
        console.print("[bold]Trace - Usage:[/bold]")
        console.print("  /trace <script.py> [args...]")
        console.print()
        console.print("[bold]Examples:[/bold]")
        console.print("  /trace my_agent.py")
        console.print("  /trace scripts/test.py --verbose")
        console.print("  /trace agent.py input.json")
        console.print()
        console.print("[dim]Instruments OpenAI, Anthropic, and Ollama SDK calls[/dim]")
        console.print("[dim]Use '/traces' to see past traces[/dim]")
        return

    script_parts = parts[1].strip().split()
    script_path = script_parts[0]
    script_args = script_parts[1:] if len(script_parts) > 1 else []

    if not Path(script_path).exists():
        console.print(f"[red]File not found: {script_path}[/red]")
        return

    try:
        from evalview.trace_cmd import run_traced_command

        console.print()
        trace_command = ["python", script_path]
        trace_command.extend(script_args)

        exit_code, _ = run_traced_command(
            command=trace_command,
            output_path=None,
            console=console,
        )

        if exit_code != 0:
            console.print(f"[yellow]Script exited with code {exit_code}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error tracing script: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


# ───────────────────────────────────────────────────────────────────────────
# /compare — diff two recent test runs
# ───────────────────────────────────────────────────────────────────────────


def handle_compare(console: Console, user_input: str) -> None:
    """Compare two stored EvaluationResult JSON files side by side."""
    parts = user_input.split()

    results_dir = Path(".evalview/results")
    if not results_dir.exists():
        console.print("[yellow]No results found. Run some tests first![/yellow]")
        return

    result_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(result_files) < 2:
        console.print("[yellow]Need at least 2 result files to compare.[/yellow]")
        console.print("[dim]Run tests multiple times to compare.[/dim]")
        return

    try:
        from evalview.reporters.json_reporter import JSONReporter
        from evalview.core.types import EvaluationResult

        # Load the two most recent runs (or specified ones)
        if len(parts) >= 3:
            file1 = results_dir / parts[1] if not parts[1].endswith('.json') else Path(parts[1])
            file2 = results_dir / parts[2] if not parts[2].endswith('.json') else Path(parts[2])
            if not file1.exists():
                file1 = results_dir / f"{parts[1]}.json"
            if not file2.exists():
                file2 = results_dir / f"{parts[2]}.json"
        else:
            file1, file2 = result_files[1], result_files[0]  # older, newer

        if not file1.exists() or not file2.exists():
            console.print("[red]Could not find result files to compare[/red]")
            return

        data1 = JSONReporter.load(str(file1))
        data2 = JSONReporter.load(str(file2))

        results1 = {r["test_case"]: EvaluationResult(**r) for r in data1} if data1 else {}
        results2 = {r["test_case"]: EvaluationResult(**r) for r in data2} if data2 else {}

        console.print("\n[bold]Comparing Results[/bold]")
        console.print(f"[dim]Old: {file1.name}[/dim]")
        console.print(f"[dim]New: {file2.name}[/dim]\n")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Test", style="cyan")
        table.add_column("Old Score", justify="right")
        table.add_column("New Score", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("Old Cost", justify="right")
        table.add_column("New Cost", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("Status")

        all_tests = set(results1.keys()) | set(results2.keys())
        regressions = 0
        improvements = 0

        for test in sorted(all_tests):
            r1 = results1.get(test)
            r2 = results2.get(test)

            if r1 and r2:
                score1 = r1.score
                score2 = r2.score
                score_delta = score2 - score1

                cost1 = r1.trace.metrics.total_cost if r1.trace and r1.trace.metrics else 0
                cost2 = r2.trace.metrics.total_cost if r2.trace and r2.trace.metrics else 0
                cost_delta = cost2 - cost1

                if score_delta < -5:
                    status = "[red]↓ REGRESSED[/red]"
                    regressions += 1
                elif score_delta > 5:
                    status = "[green]↑ IMPROVED[/green]"
                    improvements += 1
                elif not r2.passed and r1.passed:
                    status = "[red]✗ BROKE[/red]"
                    regressions += 1
                elif r2.passed and not r1.passed:
                    status = "[green]✓ FIXED[/green]"
                    improvements += 1
                else:
                    status = "[dim]— same[/dim]"

                score_delta_str = f"{score_delta:+.0f}" if score_delta != 0 else "—"
                if score_delta > 0:
                    score_delta_str = f"[green]{score_delta_str}[/green]"
                elif score_delta < 0:
                    score_delta_str = f"[red]{score_delta_str}[/red]"

                cost_delta_str = f"{cost_delta:+.4f}" if cost_delta != 0 else "—"
                if cost_delta > 0.001:
                    cost_delta_str = f"[red]+${cost_delta:.4f}[/red]"
                elif cost_delta < -0.001:
                    cost_delta_str = f"[green]-${abs(cost_delta):.4f}[/green]"

                table.add_row(
                    test[:30],
                    f"{score1:.0f}",
                    f"{score2:.0f}",
                    score_delta_str,
                    f"${cost1:.4f}",
                    f"${cost2:.4f}",
                    cost_delta_str,
                    status,
                )
            elif r2:
                cost2 = r2.trace.metrics.total_cost if r2.trace and r2.trace.metrics else 0
                table.add_row(
                    test[:30], "—", f"{r2.score:.0f}", "[cyan]NEW[/cyan]",
                    "—", f"${cost2:.4f}", "", "[cyan]+ NEW[/cyan]"
                )
            elif r1:
                cost1 = r1.trace.metrics.total_cost if r1.trace and r1.trace.metrics else 0
                table.add_row(
                    test[:30], f"{r1.score:.0f}", "—", "[yellow]DEL[/yellow]",
                    f"${cost1:.4f}", "—", "", "[yellow]- REMOVED[/yellow]"
                )

        console.print(table)

        console.print()
        if regressions > 0:
            console.print(f"[red]⚠ {regressions} regression(s) detected[/red]")
        if improvements > 0:
            console.print(f"[green]✓ {improvements} improvement(s)[/green]")
        if regressions == 0 and improvements == 0:
            console.print("[dim]No significant changes[/dim]")

    except Exception as e:
        console.print(f"[red]Error comparing: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


# ───────────────────────────────────────────────────────────────────────────
# /skill — invoke `evalview skill` subcommand
# ───────────────────────────────────────────────────────────────────────────


def handle_skill(console: Console, user_input: str) -> None:
    """Shell out to `evalview skill <subcommand>`."""
    parts = user_input.split(maxsplit=1)
    subcommand = parts[1].strip() if len(parts) > 1 else None

    if not subcommand:
        console.print("\n[bold]Skill Testing Commands:[/bold]")
        console.print()
        console.print("  [cyan]/skill test <file.yaml>[/cyan]")
        console.print("    Run skill tests in legacy mode (system prompt + string matching)")
        console.print()
        console.print("  [cyan]/skill test <file.yaml> --agent claude-code[/cyan]")
        console.print("    Run skill tests through Claude Code CLI (recommended)")
        console.print()
        console.print("  [cyan]/skill test <file.yaml> -a claude-code -t ./traces/[/cyan]")
        console.print("    Run with JSONL trace capture for debugging")
        console.print()
        console.print("  [cyan]/skill test <file.yaml> --no-rubric[/cyan]")
        console.print("    Skip Phase 2 rubric evaluation (deterministic only)")
        console.print()
        console.print("  [cyan]/skill validate <SKILL.md>[/cyan]")
        console.print("    Validate a skill file for correct structure")
        console.print()
        console.print("  [cyan]/skill list <directory>[/cyan]")
        console.print("    List all skills in a directory")
        console.print()
        console.print("  [cyan]/skill doctor <directory>[/cyan]")
        console.print("    Diagnose skill issues (token budget, duplicates)")
        console.print()
        console.print("[bold]Available Agents:[/bold]")
        console.print("  claude-code (primary), codex, langgraph, crewai, openai-assistants, custom")
        console.print()
        console.print("[bold]Phase 1 Checks:[/bold]")
        console.print("  tool_calls_contain, files_created, commands_ran, output_contains,")
        console.print("  max_tokens, build_must_pass, smoke_tests, git_clean, no_sudo")
        console.print()
        return

    try:
        sub_parts = subcommand.split()
        sub_cmd = sub_parts[0].lower()

        if sub_cmd == "test":
            skill_cmd = ["evalview", "skill", "test"] + sub_parts[1:]
            console.print(f"\n[dim]Running: {' '.join(skill_cmd)}[/dim]\n")
            skill_result = subprocess.run(skill_cmd, capture_output=False, text=True)
            if skill_result.returncode != 0:
                console.print(f"\n[yellow]Command exited with code {skill_result.returncode}[/yellow]")

        elif sub_cmd == "validate":
            if len(sub_parts) < 2:
                console.print("[yellow]Usage: /skill validate <SKILL.md>[/yellow]")
            else:
                skill_cmd = ["evalview", "skill", "validate"] + sub_parts[1:]
                console.print(f"\n[dim]Running: {' '.join(skill_cmd)}[/dim]\n")
                subprocess.run(skill_cmd, capture_output=False, text=True)

        elif sub_cmd == "list":
            if len(sub_parts) < 2:
                console.print("[yellow]Usage: /skill list <directory>[/yellow]")
            else:
                skill_cmd = ["evalview", "skill", "list"] + sub_parts[1:]
                console.print(f"\n[dim]Running: {' '.join(skill_cmd)}[/dim]\n")
                subprocess.run(skill_cmd, capture_output=False, text=True)

        elif sub_cmd == "doctor":
            if len(sub_parts) < 2:
                console.print("[yellow]Usage: /skill doctor <directory>[/yellow]")
            else:
                skill_cmd = ["evalview", "skill", "doctor"] + sub_parts[1:]
                console.print(f"\n[dim]Running: {' '.join(skill_cmd)}[/dim]\n")
                subprocess.run(skill_cmd, capture_output=False, text=True)

        else:
            console.print(f"[yellow]Unknown skill subcommand: {sub_cmd}[/yellow]")
            console.print("[dim]Available: test, validate, list, doctor[/dim]")

    except Exception as e:
        console.print(f"[red]Error running skill command: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


# ───────────────────────────────────────────────────────────────────────────
# /model — switch the active LLM mid-session
# ───────────────────────────────────────────────────────────────────────────


def handle_model(console: Console, session: ChatSession, user_input: str) -> None:
    """Switch the active model (and provider, when chosen via the menu)."""
    parts = user_input.split(maxsplit=1)
    if len(parts) >= 2:
        new_model = parts[1].strip()
        session.model = new_model
        console.print(f"[green]Switched to model: {new_model}[/green]")
        return

    # Show model selection menu
    console.print(f"\n[bold]Current model:[/bold] {session.model}")
    console.print(f"[bold]Current provider:[/bold] {session.provider.value}\n")

    installed = get_installed_ollama_models()

    ollama_models = [
        ("llama3.1:70b", "Best quality, needs 40GB+ RAM"),
        ("mixtral", "Great balance, needs 25GB+ RAM"),
        ("llama3.1:8b", "Good quality, needs 8GB+ RAM"),
        ("qwen2:7b", "Fast, needs 8GB+ RAM"),
    ]

    console.print("[bold cyan]Ollama Models (free, local):[/bold cyan]")
    for i, (model, desc) in enumerate(ollama_models, 1):
        model_base = model.split(":")[0]
        is_installed = model in installed or model_base in installed
        status = "[green]✓[/green]" if is_installed else "[dim]○[/dim]"
        console.print(f"  {status} [cyan][{i}][/cyan] {model:<16} - {desc}")

    if not installed:
        console.print("  [dim]No models installed. Install: ollama pull llama3.1:8b[/dim]")
    else:
        console.print(f"  [dim]Installed: {', '.join(sorted(installed)[:5])}{'...' if len(installed) > 5 else ''}[/dim]")
    console.print()

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    console.print("[bold cyan]Cloud Models:[/bold cyan]")
    openai_status = "[green]✓[/green]" if has_openai else "[yellow]![/yellow]"
    anthropic_status = "[green]✓[/green]" if has_anthropic else "[yellow]![/yellow]"

    console.print(f"  {openai_status} [cyan][5][/cyan] gpt-5.4-mini      - OpenAI, fast & cheap")
    console.print(f"  {openai_status} [cyan][6][/cyan] gpt-5.4           - OpenAI, best overall")
    console.print(f"  {anthropic_status} [cyan][7][/cyan] claude-sonnet-4-6 - Anthropic, excellent")
    console.print(f"  {anthropic_status} [cyan][8][/cyan] claude-haiku-4-5  - Anthropic, fast")

    if not has_openai and not has_anthropic:
        console.print("\n[bold]API Key Setup:[/bold]")
        console.print("  [dim]export OPENAI_API_KEY=sk-...[/dim]")
        console.print("  [dim]export ANTHROPIC_API_KEY=sk-ant-...[/dim]")
        console.print("  [dim]Or add to .env.local file[/dim]")
    console.print()

    choice = Prompt.ask("[dim]Select (1-8) or type model name, Enter to cancel[/dim]", default="")

    if not choice:
        return

    model_map = {
        "1": ("llama3.1:70b", LLMProvider.OLLAMA),
        "2": ("mixtral", LLMProvider.OLLAMA),
        "3": ("llama3.1:8b", LLMProvider.OLLAMA),
        "4": ("qwen2:7b", LLMProvider.OLLAMA),
        "5": ("gpt-5.4-mini", LLMProvider.OPENAI),
        "6": ("gpt-5.4", LLMProvider.OPENAI),
        "7": ("claude-sonnet-4-6", LLMProvider.ANTHROPIC),
        "8": ("claude-haiku-4-5-20251001", LLMProvider.ANTHROPIC),
    }

    if choice in model_map:
        new_model, new_provider = model_map[choice]

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
                model_base = new_model.split(":")[0]
                if new_model not in installed and model_base not in installed:
                    console.print(f"[yellow]Model '{new_model}' not installed.[/yellow]")
                    console.print(f"[dim]Install it with: ollama pull {new_model}[/dim]")
                    install = Prompt.ask("[dim]Install now? (y/n)[/dim]", default="y")
                    if install.lower() in ("y", "yes", ""):
                        console.print(f"[dim]Running: ollama pull {new_model}...[/dim]")
                        pull_result = subprocess.run(
                            ["ollama", "pull", new_model],
                            capture_output=False,
                        )
                        if pull_result.returncode == 0:
                            session.model = new_model
                            session.provider = new_provider
                            console.print(f"[green]Installed and switched to {new_model}[/green]")
                        else:
                            console.print(f"[red]Failed to install {new_model}[/red]")
                else:
                    session.model = new_model
                    session.provider = new_provider
                    console.print(f"[green]Switched to {new_model} ({new_provider.value})[/green]")
        else:
            session.model = new_model
            session.provider = new_provider
            console.print(f"[green]Switched to {new_model} ({new_provider.value})[/green]")
    elif choice:
        # Direct model name entry
        session.model = choice
        console.print(f"[green]Switched to model: {choice}[/green]")


# ───────────────────────────────────────────────────────────────────────────
# /permissions, /context, /clear — small read-only / state-reset helpers
# ───────────────────────────────────────────────────────────────────────────


def handle_permissions(console: Console, permissions: CommandPermissions) -> None:
    """List all auto-allowed shell commands for this session."""
    allowed = permissions.get_allowed_list()
    console.print("\n[bold]Auto-allowed commands:[/bold]")
    for cmd in allowed:
        console.print(f"  [green]✓[/green] {cmd}")
    console.print("\n[dim]These commands run without asking. Use option [2] to add more.[/dim]")


def handle_context(console: Console) -> None:
    """Show project context (goldens, recent results, agent endpoint)."""
    from evalview.chat_runtime import get_project_context
    context = get_project_context()
    console.print("\n[bold]Project Status:[/bold]")
    console.print(f"[dim]{context}[/dim]")


def handle_clear(console: Console, session: ChatSession) -> None:
    """Reset conversation history without ending the session."""
    session.history = []
    console.print("[dim]Chat history cleared.[/dim]")


def handle_help(console: Console) -> None:
    """Print the in-session help summary."""
    console.print("\n[bold]Chat Commands:[/bold]")
    console.print("  [cyan]/model[/cyan]            - Switch to a different model")
    console.print("  [cyan]/trace <file>[/cyan]     - Trace LLM calls in a Python script")
    console.print("  [cyan]/traces[/cyan]           - List stored traces")
    console.print("  [cyan]/traces <id>[/cyan]      - Show specific trace details")
    console.print("  [cyan]/traces export <id>[/cyan] - Export trace to HTML")
    console.print("  [cyan]/traces cost[/cyan]      - Show cost report")
    console.print("  [cyan]/docs[/cyan]             - Open EvalView documentation")
    console.print("  [cyan]/cli[/cyan]              - Show CLI commands cheatsheet")
    console.print("  [cyan]/permissions[/cyan]      - Show auto-allowed commands")
    console.print("  [cyan]/context[/cyan]          - Show project status")
    console.print("  [cyan]clear[/cyan]             - Clear chat history")
    console.print("  [cyan]exit[/cyan]              - Leave chat")
    console.print("\n[bold]Debugging:[/bold]")
    console.print("  - Add --trace to /run or /test for live tracing")
    console.print("  - Use /trace script.py to trace any Python script")
    console.print("  - Use /traces to see past traces and costs")
    console.print("  - Ask \"why did this test fail?\" for AI analysis")
    console.print("\n[bold]Tips:[/bold]")
    console.print("  - Ask how to test your agent")
    console.print("  - Ask to run specific tests")
    console.print("  - Ask to explain test failures")


__all__ = [
    "handle_adapters",
    "handle_cli",
    "handle_clear",
    "handle_compare",
    "handle_context",
    "handle_docs",
    "handle_help",
    "handle_model",
    "handle_permissions",
    "handle_run",
    "handle_skill",
    "handle_test",
    "handle_trace",
    "handle_traces",
]
