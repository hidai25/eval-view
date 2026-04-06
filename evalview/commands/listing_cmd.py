"""Listing commands — list, adapters, report, view, connect, validate-adapter, record."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

import click
import httpx
import yaml

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@click.command("list", hidden=True)
@click.option("--pattern", default="*.yaml", help="Test case file pattern (default: *.yaml)")
@click.option("--detailed", is_flag=True, help="Show detailed information for each test")
@track_command("list")
def list_cmd(pattern: str, detailed: bool):
    """List all available test cases."""
    asyncio.run(_list_async(pattern, detailed))


async def _list_async(pattern: str, detailed: bool):
    from rich.table import Table
    from evalview.core.loader import TestCaseLoader

    console.print("[blue]Loading test cases...[/blue]\n")

    test_dir = Path("tests/test-cases")
    if not test_dir.exists():
        console.print(f"[yellow]Test directory not found: {test_dir}[/yellow]")
        return

    loader = TestCaseLoader()
    test_cases = loader.load_from_directory(test_dir, pattern=pattern)

    if not test_cases:
        console.print(f"[yellow]No test cases found matching pattern: {pattern}[/yellow]")
        return

    console.print(f"[green]Found {len(test_cases)} test case(s)[/green]\n")

    table = Table(title="Available Test Cases", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="white", no_wrap=False)
    table.add_column("Adapter", style="yellow", justify="center")
    table.add_column("Endpoint", style="dim", no_wrap=False)

    if detailed:
        table.add_column("Description", style="dim", no_wrap=False)

    for test_case in test_cases:
        adapter_name = test_case.adapter or "[dim](from config)[/dim]"
        endpoint = test_case.endpoint or "[dim](from config)[/dim]"

        if detailed:
            description = test_case.description or "[dim]No description[/dim]"
            table.add_row(test_case.name, adapter_name, endpoint, description)
        else:
            table.add_row(test_case.name, adapter_name, endpoint)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

@click.command("adapters", hidden=True)
@track_command("adapters")
def adapters():
    """List all available adapters."""
    from rich.table import Table
    from evalview.adapters.registry import AdapterRegistry

    console.print("[blue]Available Adapters[/blue]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Adapter", style="white")
    table.add_column("Description", style="dim")
    table.add_column("Needs Endpoint", style="yellow", justify="center")

    adapter_info = {
        "http": ("Generic REST API adapter", "Yes"),
        "langgraph": ("LangGraph / LangGraph Cloud", "Yes"),
        "crewai": ("CrewAI multi-agent", "Yes"),
        "openai-assistants": ("OpenAI Assistants API", "No (uses SDK)"),
        "anthropic": ("Anthropic Claude API", "Yes"),
        "claude": ("Alias for anthropic", "Yes"),
        "huggingface": ("HuggingFace Inference", "Yes"),
        "hf": ("Alias for huggingface", "Yes"),
        "gradio": ("Alias for huggingface", "Yes"),
        "goose": ("Block's Goose CLI agent", "No (uses CLI)"),
        "tapescope": ("JSONL streaming API", "Yes"),
        "streaming": ("Alias for tapescope", "Yes"),
        "jsonl": ("Alias for tapescope", "Yes"),
        "mcp": ("Model Context Protocol", "Yes"),
        "mistral": ("Mistral AI API", "No (uses SDK)"),
    }

    for name in sorted(AdapterRegistry.list_names()):
        desc, needs_endpoint = adapter_info.get(name, ("Custom adapter", "Yes"))
        table.add_row(name, desc, needs_endpoint)

    console.print(table)
    console.print(f"\n[dim]Total: {len(AdapterRegistry.list_names())} adapters[/dim]")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@click.command("report", hidden=True)
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--detailed", is_flag=True, help="Show detailed results for each test case")
@click.option("--html", type=click.Path(), help="Generate HTML report to specified path")
@track_command("report", lambda **kw: {"html": bool(kw.get("html"))})
def report(results_file: str, detailed: bool, html: str):
    """Generate report from results file."""
    from evalview.reporters.json_reporter import JSONReporter
    from evalview.reporters.console_reporter import ConsoleReporter

    console.print(f"[blue]Loading results from {results_file}...[/blue]\n")

    results_data = JSONReporter.load(results_file)

    if not results_data:
        console.print("[yellow]No results found in file[/yellow]")
        return

    from evalview.core.types import EvaluationResult
    results = [EvaluationResult(**data) for data in results_data]

    if html:
        try:
            from evalview.visualization import generate_visual_report
            html_path = generate_visual_report(results, output_path=html, auto_open=False)
            console.print(f"[green]✅ HTML report saved to: {html_path}[/green]\n")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not generate HTML report: {e}[/yellow]\n")
        return

    reporter = ConsoleReporter()

    if detailed:
        for result in results:
            reporter.print_detailed(result)
    else:
        reporter.print_summary(results)


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------

def _find_results_file(run_id: str) -> Optional[Path]:
    """Find a results file by run ID or path."""
    if Path(run_id).exists():
        return Path(run_id)

    results_dir = Path(".evalview/results")
    if not results_dir.exists():
        return None

    result_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not result_files:
        return None

    if run_id.lower() == "latest":
        return result_files[0]

    for f in result_files:
        if run_id in f.stem:
            return f

    return None


@click.command("view", hidden=True)
@click.argument("run_id", required=False)
@click.option("-t", "--test", help="Filter by test name (substring match)")
@click.option("--llm-only", is_flag=True, help="Only show LLM call spans")
@click.option("--tools-only", is_flag=True, help="Only show tool call spans")
@click.option("--prompts", is_flag=True, help="Show LLM prompts (truncated)")
@click.option("--completions", is_flag=True, help="Show LLM completions (truncated)")
@click.option("--table", is_flag=True, help="Show span table instead of tree")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--llm-summary", is_flag=True, help="Show LLM call summary with token/cost breakdown")
@track_command("view")
def view(
    run_id: Optional[str],
    test: Optional[str],
    llm_only: bool,
    tools_only: bool,
    prompts: bool,
    completions: bool,
    table: bool,
    output_json: bool,
    llm_summary: bool,
):
    """View execution trace for debugging.

    RUN_ID can be:
      - "latest" (default): View the most recent run
      - A timestamp or partial match of a result file
      - A full path to a results JSON file

    Examples:
        evalview view                    # View latest run
        evalview view latest             # Same as above
        evalview view latest -t "stock"  # Filter by test name
        evalview view abc123 --llm-only  # Show only LLM calls
        evalview view --json             # Output as JSON
        evalview view --llm-summary      # Show LLM token/cost breakdown
    """
    from evalview.reporters.trace_reporter import TraceReporter
    from evalview.reporters.json_reporter import JSONReporter
    from evalview.core.types import EvaluationResult

    if not run_id:
        run_id = "latest"

    results_path = _find_results_file(run_id)
    if not results_path:
        console.print(f"[red]Could not find results for: {run_id}[/red]")
        console.print("[dim]Run 'evalview run' first to generate results[/dim]")
        return

    console.print(f"[blue]Loading results from {results_path}...[/blue]\n")

    results_data = JSONReporter.load(str(results_path))
    if not results_data:
        console.print("[yellow]No results found in file[/yellow]")
        return

    results = [EvaluationResult(**data) for data in results_data]

    if test:
        results = [r for r in results if test.lower() in r.test_case.lower()]
        if not results:
            console.print(f"[yellow]No tests matching '{test}'[/yellow]")
            return

    reporter = TraceReporter()

    for result in results:
        console.print(f"[bold cyan]Test: {result.test_case}[/bold cyan]")
        console.print()

        if output_json:
            from evalview.core.tracing import steps_to_trace_context
            if result.trace.trace_context:
                trace_context = result.trace.trace_context
            else:
                trace_context = steps_to_trace_context(
                    steps=result.trace.steps,
                    session_id=result.trace.session_id,
                    start_time=result.trace.start_time,
                    end_time=result.trace.end_time,
                )
            console.print(reporter.export_json(trace_context))
        elif table:
            from evalview.core.tracing import steps_to_trace_context
            if result.trace.trace_context:
                trace_context = result.trace.trace_context
            else:
                trace_context = steps_to_trace_context(
                    steps=result.trace.steps,
                    session_id=result.trace.session_id,
                    start_time=result.trace.start_time,
                    end_time=result.trace.end_time,
                )
            reporter.print_trace_table(trace_context)
        elif llm_summary:
            from evalview.core.tracing import steps_to_trace_context
            if result.trace.trace_context:
                trace_context = result.trace.trace_context
            else:
                trace_context = steps_to_trace_context(
                    steps=result.trace.steps,
                    session_id=result.trace.session_id,
                    start_time=result.trace.start_time,
                    end_time=result.trace.end_time,
                )
            reporter.print_llm_summary(trace_context)
        else:
            reporter.print_trace_from_result(
                result,
                show_prompts=prompts,
                show_completions=completions,
                llm_only=llm_only,
                tools_only=tools_only,
            )

        console.print()


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------

@click.command("connect", hidden=True)
@click.option("--endpoint", help="Agent endpoint URL to test (optional - will auto-detect common ones)")
@track_command("connect")
def connect(endpoint: str):
    """Test connection to your agent API and auto-configure endpoint."""
    asyncio.run(_connect_async(endpoint))


async def _connect_async(endpoint: Optional[str]):
    """Async implementation of connect command."""
    console.print("[blue]🔍 Testing agent connection...[/blue]\n")

    common_ports = [8000, 2024, 3000, 8080, 5000, 8888, 7860]
    common_patterns = [
        ("langgraph", "LangGraph Cloud", "/ok", "langgraph", "GET"),
        ("langgraph", "LangGraph Cloud", "/info", "langgraph", "GET"),
        ("langgraph", "LangGraph", "/api/chat", "langgraph", "POST"),
        ("langgraph", "LangGraph", "/invoke", "langgraph", "POST"),
        ("http", "LangServe", "/agent", "http", "POST"),
        ("streaming", "LangServe", "/agent/stream", "streaming", "POST"),
        ("streaming", "TapeScope", "/api/unifiedchat", "streaming", "POST"),
        ("crewai", "CrewAI", "/crew", "crewai", "POST"),
        ("http", "FastAPI", "/api/agent", "http", "POST"),
        ("http", "FastAPI", "/chat", "http", "POST"),
    ]

    common_endpoints = []
    for port in common_ports:
        for framework, name, path, adapter, method in common_patterns:
            url = f"http://127.0.0.1:{port}{path}"
            common_endpoints.append((framework, f"{name} (:{port})", url, adapter, method))

    endpoints_to_test = (
        [("http", "Custom", endpoint, "http", "POST")]
        if endpoint
        else common_endpoints
    )

    successful = None
    tested_count = 0

    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning for agent servers...", total=None)

        async with httpx.AsyncClient(timeout=3.0) as client:
            for adapter_type, name, url, default_adapter, method in endpoints_to_test:
                tested_count += 1
                progress.update(task, description=f"Scanning... ({tested_count} endpoints checked)")

                try:
                    if method == "GET":
                        response = await client.get(url)
                    else:
                        response = await client.post(
                            url,
                            json={
                                "query": "test",
                                "message": "test",
                                "messages": [{"role": "user", "content": "test"}],
                            },
                            headers={"Content-Type": "application/json"},
                        )

                    if response.status_code in [200, 201, 422]:
                        content_type = response.headers.get("content-type", "")
                        if not content_type.startswith("application/json"):
                            continue

                        detected_adapter = default_adapter
                        try:
                            data = response.json()
                            if "messages" in data or "thread_id" in data:
                                detected_adapter = "langgraph"
                            elif "tasks" in data or "crew_id" in data or "crew" in data:
                                detected_adapter = "crewai"
                        except Exception:
                            continue

                        successful = (name, url, response, detected_adapter)
                        break

                except (httpx.ConnectError, httpx.TimeoutException, Exception):
                    continue

    console.print()

    if successful:
        name, url, response, detected_adapter = successful
        console.print(f"[green]✅ Successfully connected to {name}![/green]\n")

        console.print("[cyan]Response details:[/cyan]")
        console.print(f"  • Status: {response.status_code}")
        console.print(f"  • Content-Type: {response.headers.get('content-type', 'N/A')}")
        console.print(f"  • Detected adapter: {detected_adapter}")

        try:
            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
                if data and isinstance(data, dict):
                    keys_str = ", ".join(str(k) for k in data.keys())
                    if keys_str:
                        console.print(f"  • Response keys: [{keys_str}]")
        except Exception:
            pass

        console.print()
        if click.confirm("Update .evalview/config.yaml to use this endpoint?", default=True):
            config_path = Path(".evalview/config.yaml")

            if not config_path.exists():
                console.print("[yellow]⚠️  Config file not found. Run 'evalview init' first.[/yellow]")
                return

            with open(config_path) as f:
                config = yaml.safe_load(f)

            config["adapter"] = detected_adapter
            endpoint_url = url
            if detected_adapter == "langgraph" and (url.endswith("/ok") or url.endswith("/info")):
                endpoint_url = url.rsplit("/", 1)[0]
            config["endpoint"] = endpoint_url

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            console.print("[green]✅ Updated config:[/green]")
            console.print(f"  • adapter: {detected_adapter}")
            console.print(f"  • endpoint: {endpoint_url}")
            console.print()
            console.print("[blue]Next steps:[/blue]")
            console.print("  1. Create test cases in tests/test-cases/")
            console.print("  2. Run: evalview run")
        return

    console.print("[red]❌ Could not connect to any agent endpoint.[/red]\n")

    console.print("[cyan]🔍 Scanning for open ports...[/cyan]")
    open_ports = []
    test_ports = [8000, 2024, 3000, 8080, 5000, 8888, 7860, 8501, 7000]

    async with httpx.AsyncClient(timeout=2.0) as client:
        for port in test_ports:
            try:
                response = await client.get(f"http://127.0.0.1:{port}")
                open_ports.append(port)
                console.print(f"  • Port {port}: [green]Open[/green] (HTTP {response.status_code})")
            except Exception:
                pass

    if open_ports:
        console.print()
        console.print(f"[green]Found {len(open_ports)} open port(s)![/green]")
        console.print()

        if click.confirm("Configure connection manually?", default=True):
            custom_port = click.prompt(
                "Port number", type=int, default=open_ports[0] if open_ports else 8000
            )

            console.print("\n[cyan]Common endpoint paths:[/cyan]")
            console.print("  1. /crew         (CrewAI)")
            console.print("  2. /invoke       (LangGraph/LangServe)")
            console.print("  3. /api/chat     (Generic)")
            console.print("  4. Custom path")

            path_choice = click.prompt("Choose (1-4)", type=int, default=1)
            path_map = {1: "/crew", 2: "/invoke", 3: "/api/chat"}

            if path_choice == 4:
                custom_path = click.prompt("Enter custom path", default="/api/chat")
            else:
                custom_path = path_map.get(path_choice, "/api/chat")
            custom_url = f"http://127.0.0.1:{custom_port}{custom_path}"

            console.print(f"\n[cyan]Testing {custom_url}...[/cyan]")

            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        custom_url,
                        json={
                            "query": "test",
                            "message": "test",
                            "messages": [{"role": "user", "content": "test"}],
                        },
                        headers={"Content-Type": "application/json"},
                    )

                    if response.status_code in [200, 201, 422]:
                        console.print("[green]✅ Connected![/green]\n")

                        detected_adapter = "http"
                        try:
                            data = response.json()
                            if "messages" in data or "thread_id" in data:
                                detected_adapter = "langgraph"
                            elif "tasks" in data or "crew_id" in data or "crew" in data:
                                detected_adapter = "crewai"
                        except Exception:
                            pass

                        config_path = Path(".evalview/config.yaml")
                        if config_path.exists():
                            with open(config_path) as f:
                                config = yaml.safe_load(f)

                            config["adapter"] = detected_adapter
                            config["endpoint"] = custom_url

                            with open(config_path, "w") as f:
                                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                            console.print("[green]✅ Config updated:[/green]")
                            console.print(f"  • adapter: {detected_adapter}")
                            console.print(f"  • endpoint: {custom_url}")
                            return
                    else:
                        console.print(f"[red]❌ HTTP {response.status_code}[/red]")
            except Exception as e:
                console.print(f"[red]❌ Failed: {e}[/red]")

    console.print()
    console.print("[yellow]Common issues:[/yellow]")
    console.print("  1. Agent server not running")
    console.print("  2. Non-standard port (check your server logs)")
    console.print("  3. Different endpoint path")
    console.print()
    console.print("[blue]To start an agent:[/blue]")
    console.print("  # LangGraph example:")
    console.print("  cd examples/langgraph/agent && langgraph dev  # port 2024")
    console.print()
    console.print("  # Or the demo agent:")
    console.print("  python demo_agent.py  # port 8000")
    console.print()
    console.print("[blue]Then run:[/blue]")
    console.print("  evalview connect")
    console.print("  # or specify endpoint:")
    console.print("  evalview connect --endpoint http://127.0.0.1:YOUR_PORT/api/chat")


# ---------------------------------------------------------------------------
# validate-adapter
# ---------------------------------------------------------------------------

@click.command("validate-adapter", hidden=True)
@click.option("--endpoint", required=True, help="Endpoint URL to validate")
@click.option(
    "--adapter",
    default="http",
    type=click.Choice(["http", "langgraph", "crewai", "streaming", "tapescope"]),
    help="Adapter type to use (default: http)",
)
@click.option("--query", default="What is 2+2?", help="Test query to send (default: 'What is 2+2?')")
@click.option("--timeout", default=30.0, type=float, help="Request timeout in seconds (default: 30)")
@track_command("validate_adapter", lambda **kw: {"adapter": kw.get("adapter")})
def validate_adapter(endpoint: str, adapter: str, query: str, timeout: float):
    """Validate an adapter endpoint and show detailed response analysis."""
    asyncio.run(_validate_adapter_async(endpoint, adapter, query, timeout))


async def _validate_adapter_async(endpoint: str, adapter_type: str, query: str, timeout: float):
    import json as json_module
    from evalview.adapters.http_adapter import HTTPAdapter
    from evalview.adapters.langgraph_adapter import LangGraphAdapter
    from evalview.adapters.crewai_adapter import CrewAIAdapter
    from evalview.adapters.tapescope_adapter import TapeScopeAdapter

    console.print("[blue]🔍 Validating adapter endpoint...[/blue]\n")
    console.print(f"  Endpoint: {endpoint}")
    console.print(f"  Adapter:  {adapter_type}")
    console.print(f"  Timeout:  {timeout}s")
    console.print(f"  Query:    {query}")
    console.print()

    try:
        test_adapter: Any = None
        if adapter_type == "langgraph":
            test_adapter = LangGraphAdapter(
                endpoint=endpoint, timeout=timeout, verbose=True, allow_private_urls=True,
            )
        elif adapter_type == "crewai":
            test_adapter = CrewAIAdapter(
                endpoint=endpoint, timeout=timeout, verbose=True, allow_private_urls=True,
            )
        elif adapter_type in ["streaming", "tapescope"]:
            test_adapter = TapeScopeAdapter(
                endpoint=endpoint, timeout=timeout, verbose=True, allow_private_urls=True,
            )
        else:
            test_adapter = HTTPAdapter(endpoint=endpoint, timeout=timeout, allow_private_urls=True)

        console.print("[cyan]Executing test query...[/cyan]")

        trace = await test_adapter.execute(query)

        console.print("[green]✅ Adapter validation successful![/green]\n")

        console.print("[bold]Execution Summary:[/bold]")
        console.print(f"  Session ID: {trace.session_id}")
        console.print(f"  Steps captured: {len(trace.steps)}")

        if trace.steps:
            console.print("\n[bold]Tools Used:[/bold]")
            for i, step in enumerate(trace.steps):
                console.print(f"  [{i+1}] {step.tool_name}")
                if step.parameters:
                    params_str = str(step.parameters)[:80]
                    console.print(f"      params: {params_str}{'...' if len(str(step.parameters)) > 80 else ''}")

        console.print("\n[bold]Metrics:[/bold]")
        console.print(f"  Total Cost: ${trace.metrics.total_cost:.4f}")
        console.print(f"  Total Latency: {trace.metrics.total_latency:.0f}ms")
        if trace.metrics.total_tokens:
            console.print(f"  Total Tokens: {trace.metrics.total_tokens.total_tokens}")
            console.print(f"    - Input: {trace.metrics.total_tokens.input_tokens}")
            console.print(f"    - Output: {trace.metrics.total_tokens.output_tokens}")

        console.print("\n[bold]Final Output:[/bold]")
        output_preview = trace.final_output[:500]
        console.print(f"  {output_preview}{'...' if len(trace.final_output) > 500 else ''}")

        if hasattr(test_adapter, "_last_raw_response") and test_adapter._last_raw_response:
            console.print("\n[bold]Raw API Response (first 1000 chars):[/bold]")
            try:
                raw_json = json_module.dumps(test_adapter._last_raw_response, indent=2, default=str)[:1000]
                console.print(f"[dim]{raw_json}[/dim]")
            except Exception:
                console.print(f"[dim]{str(test_adapter._last_raw_response)[:500]}[/dim]")

        warnings = []
        if not trace.steps:
            warnings.append("No tool calls detected - ensure your agent uses tools")
        if trace.metrics.total_cost == 0:
            warnings.append("Cost is 0 - token tracking may not be configured")
        if not trace.metrics.total_tokens:
            warnings.append("No token usage data - check adapter response format")

        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for w in warnings:
                console.print(f"  ⚠️  {w}")

        console.print()

    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]\n")
        console.print("[yellow]Troubleshooting tips:[/yellow]")
        console.print("  1. Check if the agent server is running")
        console.print("  2. Verify the endpoint URL is correct")
        console.print("  3. Try a different adapter type")
        console.print("  4. Increase timeout with --timeout")
        console.print()
        console.print("[dim]For detailed error info, check the server logs.[/dim]")


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

@click.command("record", hidden=True)
@click.option("--query", help="Query to record (non-interactive mode)")
@click.option("--output", help="Output file path (default: auto-generate in tests/test-cases/)")
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Interactive mode - record multiple interactions (default: True)",
)
@click.option("--verbose", is_flag=True, help="Show detailed execution information")
@track_command("record")
def record(query: str, output: str, interactive: bool, verbose: bool):
    """Record agent interactions and generate test cases."""
    asyncio.run(_record_async(query, output, interactive, verbose))


async def _record_async(
    query: Optional[str], output: Optional[str], interactive: bool, verbose: bool
):
    from evalview.recorder import TestCaseRecorder
    from evalview.adapters.http_adapter import HTTPAdapter
    from evalview.adapters.langgraph_adapter import LangGraphAdapter
    from evalview.adapters.crewai_adapter import CrewAIAdapter
    from evalview.adapters.tapescope_adapter import TapeScopeAdapter

    console.print("[blue]🎬 Recording mode started[/blue]")
    console.print("━" * 60)
    console.print()

    config_path = Path(".evalview/config.yaml")
    if not config_path.exists():
        console.print("[red]❌ Config file not found. Run 'evalview init' first.[/red]")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    allow_private_urls = config.get("allow_private_urls", True)
    adapter_type = config.get("adapter", "http")
    adapter: Any = None

    if adapter_type == "langgraph":
        adapter = LangGraphAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            streaming=config.get("streaming", False),
            verbose=verbose,
            model_config=model_config,
            assistant_id=config.get("assistant_id", "agent"),
            allow_private_urls=allow_private_urls,
        )
    elif adapter_type == "crewai":
        adapter = CrewAIAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            verbose=verbose,
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )
    elif adapter_type in ["streaming", "tapescope", "jsonl"]:
        adapter = TapeScopeAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 60.0),
            verbose=verbose,
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )
    else:
        adapter = HTTPAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )

    recorder = TestCaseRecorder(adapter)

    if output:
        output_path = Path(output)
    else:
        test_dir = Path("tests/test-cases")
        test_dir.mkdir(parents=True, exist_ok=True)
        output_path = None

    recorded_cases = []

    if query and not interactive:
        try:
            console.print(f"[dim]📝 Query: {query}[/dim]\n")
            console.print("[dim]🤖 Calling agent...[/dim]", end=" ")

            interaction = await recorder.record_interaction(query)

            console.print("[green]✓[/green]\n")

            console.print("[cyan]📊 Detected:[/cyan]")
            if interaction.trace.steps:
                tools = [s.tool_name for s in interaction.trace.steps if s.tool_name]
                console.print(f"  • Tools: {', '.join(tools)}")
            if interaction.trace.metrics.total_cost:
                console.print(f"  • Cost: ${interaction.trace.metrics.total_cost:.4f}")
            if interaction.trace.metrics.total_latency:
                console.print(f"  • Latency: {interaction.trace.metrics.total_latency:.0f}ms")

            if verbose:
                console.print(f"\n[dim]Output: {interaction.trace.final_output}[/dim]")

            console.print()

            test_case = recorder.generate_test_case(interaction)
            recorded_cases.append((interaction, test_case))

        except Exception as e:
            console.print(f"[red]✗ Failed: {e}[/red]")
            return

    elif interactive:
        console.print(
            "[yellow]💡 Tip: Type 'done' when finished, 'skip' to cancel current recording[/yellow]\n"
        )

        query_num = 1
        while True:
            if not query:
                console.print(
                    f"[bold]📝 Enter query #{query_num} (or 'done' to finish):[/bold] ", end=""
                )
                user_input = input().strip()

                if user_input.lower() == "done":
                    break
                elif user_input.lower() == "skip":
                    continue
                elif not user_input:
                    console.print("[yellow]⚠️  Empty query, skipping[/yellow]\n")
                    continue

                query = user_input

            try:
                console.print()
                console.print("[dim]🤖 Calling agent...[/dim]", end=" ")

                interaction = await recorder.record_interaction(query)

                console.print("[green]✓ Agent response received[/green]\n")

                console.print("[cyan]📊 Detected:[/cyan]")
                if interaction.trace.steps:
                    tools = [s.tool_name for s in interaction.trace.steps if s.tool_name]
                    console.print(f"  • Tools: {', '.join(tools)}")
                else:
                    console.print("  • Tools: None")

                if interaction.trace.metrics.total_cost:
                    console.print(f"  • Cost: ${interaction.trace.metrics.total_cost:.4f}")
                if interaction.trace.metrics.total_latency:
                    console.print(f"  • Latency: {interaction.trace.metrics.total_latency:.0f}ms")

                if verbose:
                    console.print(f"\n[dim]Output: {interaction.trace.final_output}[/dim]")

                console.print()

                test_case = recorder.generate_test_case(interaction)

                console.print(
                    f"[bold]✍️  Test case name [[dim]{test_case.name}[/dim]]:[/bold] ", end=""
                )
                custom_name = input().strip()
                if custom_name:
                    test_case.name = custom_name

                recorded_cases.append((interaction, test_case))

                console.print("[green]✅ Test case saved![/green]\n")

                query_num += 1
                query = None

            except Exception as e:
                console.print(f"[red]✗ Failed: {e}[/red]\n")
                if verbose:
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]\n")

                query = None
                continue
    else:
        console.print("[red]❌ Must provide --query or use --interactive mode[/red]")
        return

    if not recorded_cases:
        console.print("[yellow]⚠️  No test cases recorded[/yellow]")
        return

    console.print()
    console.print("━" * 60)

    saved_files = []
    for interaction, test_case in recorded_cases:
        if output_path and len(recorded_cases) == 1:
            file_path = output_path
        else:
            test_dir = Path("tests/test-cases")
            test_dir.mkdir(parents=True, exist_ok=True)
            file_path = recorder.generate_filename(test_dir)

        recorder.save_to_yaml(test_case, file_path)
        saved_files.append(file_path)

    console.print(f"[green]✅ Recorded {len(recorded_cases)} test case(s)[/green]\n")

    for file_path in saved_files:
        console.print(f"  • {file_path}")

    console.print()
    console.print("[blue]Run with:[/blue] evalview run\n")
