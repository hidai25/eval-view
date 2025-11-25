"""CLI entry point for EvalView."""

import asyncio
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from evalview.core.loader import TestCaseLoader
from evalview.core.pricing import get_model_pricing_info
from evalview.adapters.http_adapter import HTTPAdapter
from evalview.adapters.tapescope_adapter import TapeScopeAdapter
from evalview.adapters.langgraph_adapter import LangGraphAdapter
from evalview.adapters.crewai_adapter import CrewAIAdapter
from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter
from evalview.evaluators.evaluator import Evaluator
from evalview.reporters.json_reporter import JSONReporter
from evalview.reporters.console_reporter import ConsoleReporter

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env.local")

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """EvalView - Testing framework for multi-step AI agents."""
    pass


@main.command()
@click.option(
    "--dir",
    default=".",
    help="Directory to initialize (default: current directory)",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Interactive setup (default: True)",
)
def init(dir: str, interactive: bool):
    """Initialize EvalView in the current directory."""
    console.print("[blue]━━━ EvalView Setup ━━━[/blue]\n")

    base_path = Path(dir)

    # Create directories
    (base_path / ".evalview").mkdir(exist_ok=True)
    (base_path / "tests" / "test-cases").mkdir(parents=True, exist_ok=True)

    # Interactive configuration
    adapter_type = "http"
    endpoint = "http://localhost:3000/api/agent"
    timeout = 30.0
    model_name = "gpt-5-mini"
    custom_pricing = None

    if interactive:
        console.print("[bold]Step 1: API Configuration[/bold]")

        # Ask adapter type
        console.print("\nWhat type of API does your agent use?")
        console.print("  1. Standard REST API (returns complete JSON)")
        console.print("  2. Streaming API (JSONL/Server-Sent Events)")
        adapter_choice = click.prompt("Choice", type=int, default=1)
        adapter_type = "streaming" if adapter_choice == 2 else "http"

        # Ask endpoint
        endpoint = click.prompt("\nAPI endpoint URL", default=endpoint)
        timeout = click.prompt("Timeout (seconds)", type=float, default=timeout)

        console.print("\n[bold]Step 2: Model & Pricing Configuration[/bold]")
        console.print("\nWhich model does your agent use?")
        console.print("  1. gpt-5-mini (recommended for testing)")
        console.print("  2. gpt-5")
        console.print("  3. gpt-5-nano")
        console.print("  4. gpt-4o or gpt-4o-mini")
        console.print("  5. Custom model")

        model_choice = click.prompt("Choice", type=int, default=1)

        model_map = {
            1: "gpt-5-mini",
            2: "gpt-5",
            3: "gpt-5-nano",
            4: "gpt-4o-mini",
        }

        if model_choice == 5:
            model_name = click.prompt("Model name")
        else:
            model_name = model_map.get(model_choice, "gpt-5-mini")

        # Show pricing
        pricing = get_model_pricing_info(model_name)
        console.print(f"\n[cyan]Pricing for {model_name}:[/cyan]")
        console.print(f"  • Input tokens:  ${pricing['input_price_per_1m']:.2f} per 1M tokens")
        console.print(f"  • Output tokens: ${pricing['output_price_per_1m']:.2f} per 1M tokens")
        console.print(f"  • Cached tokens: ${pricing['cached_price_per_1m']:.3f} per 1M tokens")

        # Ask if pricing is correct
        if click.confirm("\nIs this pricing correct for your use case?", default=True):
            console.print("[green]✅ Using standard pricing[/green]")
        else:
            console.print("\n[yellow]Let's set custom pricing:[/yellow]")
            input_price = click.prompt(
                "Input tokens ($ per 1M)", type=float, default=pricing["input_price_per_1m"]
            )
            output_price = click.prompt(
                "Output tokens ($ per 1M)", type=float, default=pricing["output_price_per_1m"]
            )
            cached_price = click.prompt(
                "Cached tokens ($ per 1M)", type=float, default=pricing["cached_price_per_1m"]
            )

            custom_pricing = {
                "input": input_price,
                "output": output_price,
                "cached": cached_price,
            }
            console.print("[green]✅ Custom pricing saved[/green]")

    # Create config file
    config_path = base_path / ".evalview" / "config.yaml"
    if not config_path.exists():
        config_content = f"""# EvalView Configuration
adapter: {adapter_type}
endpoint: {endpoint}
timeout: {timeout}
headers: {{}}

# Model configuration
model:
  name: {model_name}
"""
        if custom_pricing:
            config_content += f"""  pricing:
    input_per_1m: {custom_pricing['input']}
    output_per_1m: {custom_pricing['output']}
    cached_per_1m: {custom_pricing['cached']}
"""
        else:
            config_content += """  # Uses standard OpenAI pricing
  # Override with custom pricing if needed:
  # pricing:
  #   input_per_1m: 0.25
  #   output_per_1m: 2.0
  #   cached_per_1m: 0.025
"""

        config_path.write_text(config_content)
        console.print("\n[green]✅ Created .evalview/config.yaml[/green]")
    else:
        console.print("\n[yellow]⚠️  .evalview/config.yaml already exists[/yellow]")

    # Create example test case
    example_path = base_path / "tests" / "test-cases" / "example.yaml"
    if not example_path.exists():
        example_content = """name: "Example Test Case"
description: "Basic agent test"

input:
  query: "Analyze AAPL stock performance"
  context: {}

expected:
  tools:
    - fetch_stock_data
    - analyze_metrics
  output:
    contains:
      - "revenue"
      - "earnings"
    not_contains:
      - "error"

thresholds:
  min_score: 80
  max_cost: 0.50
  max_latency: 5000
"""
        example_path.write_text(example_content)
        console.print("[green]✅ Created tests/test-cases/example.yaml[/green]")
    else:
        console.print("[yellow]⚠️  tests/test-cases/example.yaml already exists[/yellow]")

    console.print("\n[blue]Next steps:[/blue]")
    console.print("  1. Edit .evalview/config.yaml with your agent endpoint")
    console.print("  2. Write test cases in tests/test-cases/")
    console.print("  3. Run: evalview run\n")


@main.command()
@click.option(
    "--pattern",
    default="*.yaml",
    help="Test case file pattern (default: *.yaml)",
)
@click.option(
    "--test",
    "-t",
    multiple=True,
    help="Specific test name(s) to run (can specify multiple: -t test1 -t test2)",
)
@click.option(
    "--filter",
    "-f",
    help="Filter tests by name pattern (e.g., 'LangGraph*', '*simple*')",
)
@click.option(
    "--output",
    default=".evalview/results",
    help="Output directory for results",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging (shows API requests/responses)",
)
@click.option(
    "--track",
    is_flag=True,
    help="Track results for regression analysis",
)
@click.option(
    "--compare-baseline",
    is_flag=True,
    help="Compare results against baseline and show regressions",
)
def run(
    pattern: str,
    test: tuple,
    filter: str,
    output: str,
    verbose: bool,
    track: bool,
    compare_baseline: bool,
):
    """Run test cases against the agent."""
    asyncio.run(_run_async(pattern, test, filter, output, verbose, track, compare_baseline))


async def _run_async(
    pattern: str,
    test: tuple,
    filter: str,
    output: str,
    verbose: bool,
    track: bool,
    compare_baseline: bool,
):
    """Async implementation of run command."""
    import fnmatch
    from evalview.tracking import RegressionTracker

    if verbose:
        console.print("[dim]🔍 Verbose mode enabled[/dim]\n")

    if track or compare_baseline:
        console.print("[dim]📊 Regression tracking enabled[/dim]\n")

    console.print("[blue]Running test cases...[/blue]\n")

    # Load config
    config_path = Path(".evalview/config.yaml")
    if not config_path.exists():
        console.print("[red]❌ Config file not found. Run 'evalview init' first.[/red]")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract model config
    model_config = config.get("model", {})
    if verbose and model_config:
        console.print(f"[dim]💰 Model: {model_config.get('name', 'gpt-5-mini')}[/dim]")
        if "pricing" in model_config:
            console.print(
                f"[dim]💵 Custom pricing: ${model_config['pricing']['input_per_1m']:.2f} in, ${model_config['pricing']['output_per_1m']:.2f} out[/dim]"
            )

    # SSRF protection config - defaults to True for local development
    # Set to False in production when using untrusted test cases
    allow_private_urls = config.get("allow_private_urls", True)
    if verbose:
        if allow_private_urls:
            console.print("[dim]🔓 SSRF protection: allowing private URLs (local dev mode)[/dim]")
        else:
            console.print("[dim]🔒 SSRF protection: blocking private URLs[/dim]")

    # Initialize adapter based on type
    adapter_type = config.get("adapter", "http")

    if adapter_type == "langgraph":
        adapter = LangGraphAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            streaming=config.get("streaming", False),
            verbose=verbose,
            model_config=model_config,
            assistant_id=config.get("assistant_id", "agent"),  # Cloud API support
            allow_private_urls=allow_private_urls,
        )
    elif adapter_type == "crewai":
        adapter = CrewAIAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 120.0),
            verbose=verbose,
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )
    elif adapter_type == "openai-assistants":
        adapter = OpenAIAssistantsAdapter(
            assistant_id=config.get("assistant_id"),
            timeout=config.get("timeout", 120.0),
            verbose=verbose,
            model_config=model_config,
        )
    elif adapter_type in ["streaming", "tapescope", "jsonl"]:
        # Streaming adapter supports JSONL streaming APIs
        # (tapescope/jsonl are aliases for backward compatibility)
        adapter = TapeScopeAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 60.0),
            verbose=verbose,
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )
    else:
        # HTTP adapter for standard REST APIs
        adapter = HTTPAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )

    # Initialize evaluator
    evaluator = Evaluator(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize tracker if tracking enabled
    tracker = None
    regression_reports = {}
    if track or compare_baseline:
        tracker = RegressionTracker()

    # Load test cases
    test_cases_dir = Path("tests/test-cases")
    if not test_cases_dir.exists():
        console.print("[red]❌ Test cases directory not found: tests/test-cases[/red]")
        return

    test_cases = TestCaseLoader.load_from_directory(test_cases_dir, pattern)

    if not test_cases:
        console.print(f"[yellow]⚠️  No test cases found matching pattern: {pattern}[/yellow]")
        return

    # Filter test cases by name if --test or --filter specified
    if test or filter:
        original_count = len(test_cases)
        filtered_cases = []

        for test_case in test_cases:
            # Check if test name is in the --test list (case-insensitive)
            if test:
                test_name_lower = test_case.name.lower()
                if any(t.lower() == test_name_lower for t in test):
                    filtered_cases.append(test_case)
                    continue

            # Check if test name matches --filter pattern (case-insensitive, fuzzy)
            if filter:
                filter_lower = filter.lower()
                test_name_lower = test_case.name.lower()

                # If filter has wildcards, use pattern matching
                if "*" in filter or "?" in filter:
                    if fnmatch.fnmatch(test_name_lower, filter_lower):
                        filtered_cases.append(test_case)
                        continue
                # Otherwise, do substring match (more user-friendly)
                elif filter_lower in test_name_lower:
                    filtered_cases.append(test_case)
                    continue

        test_cases = filtered_cases

        if not test_cases:
            console.print("[yellow]⚠️  No test cases matched the filter criteria[/yellow]")
            return

        if verbose:
            console.print(f"[dim]Filtered {original_count} → {len(test_cases)} test(s)[/dim]\n")

    console.print(f"Found {len(test_cases)} test case(s)\n")

    # Helper function to get adapter for a test case
    def get_adapter_for_test(test_case):
        """Get adapter for test case - use test-specific if specified, otherwise global."""
        # If test specifies its own adapter, create it
        if test_case.adapter and test_case.endpoint:
            test_adapter_type = test_case.adapter
            test_endpoint = test_case.endpoint
            test_config = test_case.adapter_config or {}

            if verbose:
                console.print(
                    f"[dim]  Using test-specific adapter: {test_adapter_type} @ {test_endpoint}[/dim]"
                )

            # Create adapter based on type
            if test_adapter_type == "langgraph":
                return LangGraphAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 30.0),
                    streaming=test_config.get("streaming", False),
                    verbose=verbose,
                    model_config=model_config,
                    assistant_id=test_config.get("assistant_id", "agent"),
                    allow_private_urls=allow_private_urls,
                )
            elif test_adapter_type == "crewai":
                return CrewAIAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 120.0),
                    verbose=verbose,
                    model_config=model_config,
                    allow_private_urls=allow_private_urls,
                )
            elif test_adapter_type == "openai-assistants":
                return OpenAIAssistantsAdapter(
                    assistant_id=test_config.get("assistant_id"),
                    api_key=test_config.get("api_key"),
                    verbose=verbose,
                    model_config=model_config,
                )
            elif test_adapter_type == "tapescope":
                return TapeScopeAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 120.0),
                    verbose=verbose,
                    model_config=model_config,
                    allow_private_urls=allow_private_urls,
                )
            else:  # Default to HTTP adapter
                return HTTPAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 30.0),
                    model_config=model_config,
                    allow_private_urls=allow_private_urls,
                )

        # Use global adapter
        return adapter

    # Run evaluations
    results = []
    passed = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for test_case in test_cases:
            task = progress.add_task(f"Running {test_case.name}...", total=None)

            try:
                # Get adapter for this test (test-specific or global)
                test_adapter = get_adapter_for_test(test_case)

                # Execute agent
                trace = await test_adapter.execute(test_case.input.query, test_case.input.context)

                # Evaluate
                result = await evaluator.evaluate(test_case, trace)
                results.append(result)

                # Track result and compare to baseline if enabled
                if tracker:
                    if track:
                        tracker.store_result(result)

                    if compare_baseline:
                        regression_report = tracker.compare_to_baseline(result)
                        regression_reports[test_case.name] = regression_report

                if result.passed:
                    passed += 1
                    progress.update(
                        task,
                        description=f"[green]✅ {test_case.name} - PASSED (score: {result.score})[/green]",
                    )
                else:
                    failed += 1
                    progress.update(
                        task,
                        description=f"[red]❌ {test_case.name} - FAILED (score: {result.score})[/red]",
                    )

            except Exception as e:
                import httpx

                failed += 1

                # Provide helpful error messages
                error_msg = str(e)
                if isinstance(e, httpx.ConnectError):
                    error_msg = f"Cannot connect to {config['endpoint']}"
                    console.print(
                        f"\n[red]❌ Connection Error:[/red] Agent server not reachable at {config['endpoint']}"
                    )
                    console.print(
                        "[yellow]💡 Tip:[/yellow] Run 'evalview connect' to test and configure your endpoint\n"
                    )
                elif isinstance(e, httpx.TimeoutException):
                    error_msg = "Request timeout"
                    console.print(
                        f"\n[yellow]⏱️  Timeout:[/yellow] Agent took too long to respond (>{config.get('timeout', 30)}s)"
                    )
                    console.print(
                        "[yellow]💡 Tip:[/yellow] Increase timeout in .evalview/config.yaml or optimize your agent\n"
                    )

                progress.update(
                    task,
                    description=f"[red]❌ {test_case.name} - ERROR: {error_msg}[/red]",
                )

            progress.remove_task(task)

    # Print summary
    console.print()
    reporter = ConsoleReporter()
    reporter.print_summary(results)

    # Print regression analysis if enabled
    if compare_baseline and regression_reports:
        console.print()
        console.print("[bold cyan]📊 Regression Analysis[/bold cyan]")
        console.print("━" * 60)
        console.print()

        any_regressions = False
        for test_name, report in regression_reports.items():
            if report.baseline_score is None:
                continue  # Skip tests without baselines

            # Color code based on severity
            if report.is_regression:
                any_regressions = True
                if report.severity == "critical":
                    status = "[red]🔴 CRITICAL REGRESSION[/red]"
                elif report.severity == "moderate":
                    status = "[yellow]🟡 MODERATE REGRESSION[/yellow]"
                else:
                    status = "[yellow]🟠 MINOR REGRESSION[/yellow]"
            else:
                status = "[green]✅ No regression[/green]"

            console.print(f"[bold]{test_name}[/bold]: {status}")

            # Show score comparison
            if report.score_delta is not None:
                delta_str = f"{report.score_delta:+.1f}"
                percent_str = f"({report.score_delta_percent:+.1f}%)"
                if report.score_delta < 0:
                    console.print(
                        f"  Score: {report.current_score:.1f} [red]↓ {delta_str}[/red] {percent_str} vs baseline {report.baseline_score:.1f}"
                    )
                else:
                    console.print(
                        f"  Score: {report.current_score:.1f} [green]↑ {delta_str}[/green] {percent_str} vs baseline {report.baseline_score:.1f}"
                    )

            # Show cost comparison
            if report.cost_delta is not None and report.cost_delta_percent is not None:
                delta_str = f"${report.cost_delta:+.4f}"
                percent_str = f"({report.cost_delta_percent:+.1f}%)"
                if report.cost_delta_percent > 20:
                    console.print(
                        f"  Cost: ${report.current_cost:.4f} [red]↑ {delta_str}[/red] {percent_str}"
                    )
                else:
                    console.print(f"  Cost: ${report.current_cost:.4f} {delta_str} {percent_str}")

            # Show latency comparison
            if report.latency_delta is not None and report.latency_delta_percent is not None:
                delta_str = f"{report.latency_delta:+.0f}ms"
                percent_str = f"({report.latency_delta_percent:+.1f}%)"
                if report.latency_delta_percent > 30:
                    console.print(
                        f"  Latency: {report.current_latency:.0f}ms [red]↑ {delta_str}[/red] {percent_str}"
                    )
                else:
                    console.print(
                        f"  Latency: {report.current_latency:.0f}ms {delta_str} {percent_str}"
                    )

            # Show specific issues
            if report.is_regression and report.issues:
                console.print(f"  Issues: {', '.join(report.issues)}")

            console.print()

        if any_regressions:
            console.print("[red]⚠️  Regressions detected! Review changes before deploying.[/red]\n")

    # Save results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    JSONReporter.save(results, results_file)

    console.print(f"\n[dim]Results saved to: {results_file}[/dim]\n")

    if track:
        console.print("[dim]📊 Results tracked for regression analysis[/dim]")
        console.print("[dim]   View trends: evalview trends[/dim]")
        console.print("[dim]   Set baseline: evalview baseline set[/dim]\n")


@main.command()
@click.option(
    "--pattern",
    default="*.yaml",
    help="Test case file pattern (default: *.yaml)",
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed information for each test",
)
def list(pattern: str, detailed: bool):
    """List all available test cases."""
    asyncio.run(_list_async(pattern, detailed))


async def _list_async(pattern: str, detailed: bool):
    """Async implementation of list command."""
    from rich.table import Table

    console.print("[blue]Loading test cases...[/blue]\n")

    # Load test cases
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

    # Create table
    table = Table(title="Available Test Cases", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="white", no_wrap=False)
    table.add_column("Adapter", style="yellow", justify="center")
    table.add_column("Endpoint", style="dim", no_wrap=False)

    if detailed:
        table.add_column("Description", style="dim", no_wrap=False)

    # Add rows
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


@main.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed results for each test case",
)
def report(results_file: str, detailed: bool):
    """Generate report from results file."""
    console.print(f"[blue]Loading results from {results_file}...[/blue]\n")

    results_data = JSONReporter.load(results_file)

    if not results_data:
        console.print("[yellow]No results found in file[/yellow]")
        return

    # Convert back to EvaluationResult objects
    from evalview.core.types import EvaluationResult

    results = [EvaluationResult(**data) for data in results_data]

    reporter = ConsoleReporter()

    if detailed:
        for result in results:
            reporter.print_detailed(result)
    else:
        reporter.print_summary(results)


@main.command()
@click.option(
    "--endpoint",
    help="Agent endpoint URL to test (optional - will auto-detect common ones)",
)
def connect(endpoint: str):
    """Test connection to your agent API and auto-configure endpoint."""
    asyncio.run(_connect_async(endpoint))


async def _connect_async(endpoint: Optional[str]):
    """Async implementation of connect command."""
    import httpx

    console.print("[blue]🔍 Testing agent connection...[/blue]\n")

    # Common ports to check
    common_ports = [8000, 2024, 3000, 8080, 5000, 8888, 7860]

    # Common endpoints to try (framework_type, name, path, adapter_type)
    # Will be combined with common_ports
    common_patterns = [
        ("langgraph", "LangGraph", "/api/chat", "langgraph"),
        ("langgraph", "LangGraph", "/invoke", "langgraph"),
        ("langgraph", "LangGraph", "/threads/runs/stream", "langgraph"),  # LangGraph Cloud
        ("http", "LangServe", "/agent", "http"),
        ("streaming", "LangServe", "/agent/stream", "streaming"),
        ("streaming", "TapeScope", "/api/unifiedchat", "streaming"),
        ("crewai", "CrewAI", "/crew", "crewai"),
        ("http", "FastAPI", "/api/agent", "http"),
        ("http", "FastAPI", "/chat", "http"),
    ]

    # Generate all port+path combinations
    common_endpoints = []
    for port in common_ports:
        for framework, name, path, adapter in common_patterns:
            url = f"http://127.0.0.1:{port}{path}"
            common_endpoints.append((framework, f"{name} (:{port})", url, adapter))

    endpoints_to_test = []
    if endpoint:
        # User provided specific endpoint - try to detect adapter type
        endpoints_to_test = [("http", "Custom", endpoint, "http")]
    else:
        # Try common ones
        endpoints_to_test = common_endpoints

    successful = None

    async with httpx.AsyncClient(timeout=5.0) as client:
        for adapter_type, name, url, default_adapter in endpoints_to_test:
            try:
                console.print(f"[dim]Testing {name}: {url}...[/dim]", end=" ")

                # Try a simple POST request
                response = await client.post(
                    url,
                    json={
                        "query": "test",
                        "message": "test",
                        "messages": [{"role": "user", "content": "test"}],
                    },
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code in [
                    200,
                    201,
                    422,
                ]:  # 422 might be validation error but server is running
                    # Try to detect framework from response
                    detected_adapter = default_adapter
                    try:
                        if response.headers.get("content-type", "").startswith("application/json"):
                            data = response.json()
                            # LangGraph detection
                            if "messages" in data or "thread_id" in data:
                                detected_adapter = "langgraph"
                            # CrewAI detection
                            elif "tasks" in data or "crew_id" in data:
                                detected_adapter = "crewai"
                    except Exception:
                        pass

                    console.print("[green]✅ Connected![/green]")
                    successful = (name, url, response, detected_adapter)
                    break
                else:
                    console.print(f"[yellow]❌ HTTP {response.status_code}[/yellow]")

            except httpx.ConnectError:
                console.print("[red]❌ Connection refused[/red]")
            except httpx.TimeoutException:
                console.print("[yellow]❌ Timeout[/yellow]")
            except Exception as e:
                console.print(f"[red]❌ {type(e).__name__}[/red]")

    console.print()

    if successful:
        name, url, response, detected_adapter = successful
        console.print(f"[green]✅ Successfully connected to {name}![/green]\n")

        # Show response info
        console.print("[cyan]Response details:[/cyan]")
        console.print(f"  • Status: {response.status_code}")
        console.print(f"  • Content-Type: {response.headers.get('content-type', 'N/A')}")
        console.print(f"  • Detected adapter: {detected_adapter}")

        # Try to show response preview
        try:
            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
                console.print(f"  • Response keys: {list(data.keys())}")
        except Exception:
            pass

        # Ask if user wants to update config
        console.print()
        if click.confirm("Update .evalview/config.yaml to use this endpoint?", default=True):
            config_path = Path(".evalview/config.yaml")

            if not config_path.exists():
                console.print(
                    "[yellow]⚠️  Config file not found. Run 'evalview init' first.[/yellow]"
                )
                return

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Update config with detected adapter
            config["adapter"] = detected_adapter
            config["endpoint"] = url

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            console.print("[green]✅ Updated config:[/green]")
            console.print(f"  • adapter: {detected_adapter}")
            console.print(f"  • endpoint: {url}")
            console.print()
            console.print("[blue]Next steps:[/blue]")
            console.print("  1. Create test cases in tests/test-cases/")
            console.print("  2. Run: evalview run --verbose")
    else:
        console.print("[red]❌ Could not connect to any agent endpoint.[/red]\n")

        # Try to find open ports
        console.print("[cyan]🔍 Scanning for open ports...[/cyan]")
        open_ports = []
        test_ports = [8000, 2024, 3000, 8080, 5000, 8888, 7860, 8501, 7000]

        async with httpx.AsyncClient(timeout=2.0) as client:
            for port in test_ports:
                try:
                    response = await client.get(f"http://127.0.0.1:{port}")
                    open_ports.append(port)
                    console.print(
                        f"  • Port {port}: [green]Open[/green] (HTTP {response.status_code})"
                    )
                except Exception:
                    pass

        if open_ports:
            console.print()
            console.print(f"[yellow]Found {len(open_ports)} open port(s)![/yellow]")
            console.print("[blue]Try connecting to one of these manually:[/blue]")
            for port in open_ports:
                console.print(f"  evalview connect --endpoint http://127.0.0.1:{port}/api/chat")
            console.print()

            if click.confirm("Do you want to try a custom endpoint?", default=True):
                custom_port = click.prompt(
                    "Enter port number", type=int, default=open_ports[0] if open_ports else 8000
                )
                custom_path = click.prompt("Enter endpoint path", default="/api/chat")
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

                            # Auto-detect adapter
                            detected_adapter = "http"
                            try:
                                data = response.json()
                                if "messages" in data or "thread_id" in data:
                                    detected_adapter = "langgraph"
                                elif "tasks" in data or "crew_id" in data:
                                    detected_adapter = "crewai"
                            except Exception:
                                pass

                            # Update config
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
        console.print("[blue]To start LangGraph agent:[/blue]")
        console.print("  cd /path/to/langgraph-example")
        console.print("  langgraph dev  # Runs on port 2024")
        console.print("  # or")
        console.print("  python main.py")
        console.print()
        console.print("[blue]Then run:[/blue]")
        console.print("  evalview connect")
        console.print("  # or specify endpoint:")
        console.print("  evalview connect --endpoint http://127.0.0.1:YOUR_PORT/api/chat")


@main.command()
@click.option(
    "--query",
    help="Query to record (non-interactive mode)",
)
@click.option(
    "--output",
    help="Output file path (default: auto-generate in tests/test-cases/)",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Interactive mode - record multiple interactions (default: True)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed execution information",
)
def record(query: str, output: str, interactive: bool, verbose: bool):
    """Record agent interactions and generate test cases."""
    asyncio.run(_record_async(query, output, interactive, verbose))


async def _record_async(
    query: Optional[str], output: Optional[str], interactive: bool, verbose: bool
):
    """Async implementation of record command."""
    from evalview.recorder import TestCaseRecorder

    console.print("[blue]🎬 Recording mode started[/blue]")
    console.print("━" * 60)
    console.print()

    # Load config
    config_path = Path(".evalview/config.yaml")
    if not config_path.exists():
        console.print("[red]❌ Config file not found. Run 'evalview init' first.[/red]")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract model config
    model_config = config.get("model", {})

    # SSRF protection config - defaults to True for local development
    allow_private_urls = config.get("allow_private_urls", True)

    # Initialize adapter
    adapter_type = config.get("adapter", "http")

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
        # HTTP adapter for standard REST APIs
        adapter = HTTPAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )

    # Initialize recorder
    recorder = TestCaseRecorder(adapter)

    # Determine output directory
    if output:
        output_path = Path(output)
    else:
        test_dir = Path("tests/test-cases")
        test_dir.mkdir(parents=True, exist_ok=True)
        output_path = None  # Will auto-generate

    recorded_cases = []

    # Non-interactive mode with single query
    if query and not interactive:
        try:
            console.print(f"[dim]📝 Query: {query}[/dim]\n")
            console.print("[dim]🤖 Calling agent...[/dim]", end=" ")

            interaction = await recorder.record_interaction(query)

            console.print("[green]✓[/green]\n")

            # Show detected info
            console.print("[cyan]📊 Detected:[/cyan]")
            if interaction.trace.tool_calls:
                tools = [tc.name for tc in interaction.trace.tool_calls]
                console.print(f"  • Tools: {', '.join(tools)}")
            if interaction.trace.cost:
                console.print(f"  • Cost: ${interaction.trace.cost:.4f}")
            if interaction.trace.latency:
                console.print(f"  • Latency: {interaction.trace.latency:.0f}ms")

            if verbose:
                console.print(f"\n[dim]Output: {interaction.trace.final_output}[/dim]")

            console.print()

            # Generate test case
            test_case = recorder.generate_test_case(interaction)
            recorded_cases.append((interaction, test_case))

        except Exception as e:
            console.print(f"[red]✗ Failed: {e}[/red]")
            return

    # Interactive mode
    elif interactive:
        console.print(
            "[yellow]💡 Tip: Type 'done' when finished, 'skip' to cancel current recording[/yellow]\n"
        )

        query_num = 1
        while True:
            # Get query from user
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

                # Show detected info
                console.print("[cyan]📊 Detected:[/cyan]")
                if interaction.trace.tool_calls:
                    tools = [tc.name for tc in interaction.trace.tool_calls]
                    console.print(f"  • Tools: {', '.join(tools)}")
                else:
                    console.print("  • Tools: None")

                if interaction.trace.cost:
                    console.print(f"  • Cost: ${interaction.trace.cost:.4f}")
                if interaction.trace.latency:
                    console.print(f"  • Latency: {interaction.trace.latency:.0f}ms")

                if verbose:
                    console.print(f"\n[dim]Output: {interaction.trace.final_output}[/dim]")

                console.print()

                # Generate test case
                test_case = recorder.generate_test_case(interaction)

                # Ask for custom name
                console.print(
                    f"[bold]✍️  Test case name [[dim]{test_case.name}[/dim]]:[/bold] ", end=""
                )
                custom_name = input().strip()
                if custom_name:
                    test_case.name = custom_name

                recorded_cases.append((interaction, test_case))

                console.print("[green]✅ Test case saved![/green]\n")

                query_num += 1
                query = None  # Reset for next iteration

            except Exception as e:
                console.print(f"[red]✗ Failed: {e}[/red]\n")
                if verbose:
                    import traceback

                    console.print(f"[dim]{traceback.format_exc()}[/dim]\n")

                query = None  # Reset
                continue
    else:
        console.print("[red]❌ Must provide --query or use --interactive mode[/red]")
        return

    # Save recorded test cases
    if not recorded_cases:
        console.print("[yellow]⚠️  No test cases recorded[/yellow]")
        return

    console.print()
    console.print("━" * 60)

    saved_files = []
    for interaction, test_case in recorded_cases:
        if output_path and len(recorded_cases) == 1:
            # Single file output
            file_path = output_path
        else:
            # Auto-generate filenames
            test_dir = Path("tests/test-cases")
            test_dir.mkdir(parents=True, exist_ok=True)
            file_path = recorder.generate_filename(test_dir)

        recorder.save_to_yaml(test_case, file_path)
        saved_files.append(file_path)

    # Print summary
    console.print(f"[green]✅ Recorded {len(recorded_cases)} test case(s)[/green]\n")

    for file_path in saved_files:
        console.print(f"  • {file_path}")

    console.print()
    console.print("[blue]Run with:[/blue] evalview run\n")


@main.group()
def baseline():
    """Manage test baselines for regression detection."""
    pass


@baseline.command("set")
@click.option(
    "--test",
    help="Specific test name to set baseline for (default: all recent tests)",
)
@click.option(
    "--from-latest",
    is_flag=True,
    help="Set baseline from most recent test run",
)
def baseline_set(test: str, from_latest: bool):
    """Set baseline from recent test results."""
    from evalview.tracking import RegressionTracker

    tracker = RegressionTracker()

    if test:
        # Set baseline for specific test
        if from_latest:
            try:
                tracker.set_baseline_from_latest(test)
                console.print(f"[green]✅ Baseline set for test: {test}[/green]")
            except ValueError as e:
                console.print(f"[red]❌ Error: {e}[/red]")
        else:
            console.print("[yellow]⚠️  Must specify --from-latest or run tests first[/yellow]")
    else:
        # Set baselines for all recent tests
        results = tracker.db.get_recent_results(days=1)
        unique_tests = set(r["test_name"] for r in results)

        if not unique_tests:
            console.print("[yellow]⚠️  No recent test results found. Run tests first.[/yellow]")
            return

        for test_name in unique_tests:
            tracker.set_baseline_from_latest(test_name)

        console.print(f"[green]✅ Baselines set for {len(unique_tests)} test(s)[/green]")


@baseline.command("show")
@click.option(
    "--test",
    help="Specific test name to show baseline for",
)
def baseline_show(test: str):
    """Show current baselines."""
    from evalview.tracking import RegressionTracker
    from rich.table import Table

    tracker = RegressionTracker()

    if test:
        # Show specific baseline
        baseline = tracker.db.get_baseline(test)
        if not baseline:
            console.print(f"[yellow]⚠️  No baseline set for test: {test}[/yellow]")
            return

        console.print(f"\n[bold]Baseline for: {test}[/bold]\n")
        console.print(f"  Score: {baseline['score']:.2f}")
        if baseline.get("cost"):
            console.print(f"  Cost: ${baseline['cost']:.4f}")
        if baseline.get("latency"):
            console.print(f"  Latency: {baseline['latency']:.0f}ms")
        console.print(f"  Created: {baseline['created_at']}")
        if baseline.get("git_commit"):
            console.print(
                f"  Git: {baseline['git_commit']} ({baseline.get('git_branch', 'unknown')})"
            )
        console.print()
    else:
        # Show all baselines
        # Get all unique test names from results
        results = tracker.db.get_recent_results(days=30)
        unique_tests = set(r["test_name"] for r in results)

        table = Table(title="Test Baselines", show_header=True, header_style="bold cyan")
        table.add_column("Test Name", style="white")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Cost", justify="right", style="yellow")
        table.add_column("Latency", justify="right", style="blue")
        table.add_column("Created", style="dim")

        has_baselines = False
        for test_name in sorted(unique_tests):
            baseline = tracker.db.get_baseline(test_name)
            if baseline:
                has_baselines = True
                table.add_row(
                    test_name,
                    f"{baseline['score']:.1f}",
                    f"${baseline.get('cost', 0):.4f}" if baseline.get("cost") else "N/A",
                    f"{baseline.get('latency', 0):.0f}ms" if baseline.get("latency") else "N/A",
                    baseline["created_at"][:10],
                )

        if not has_baselines:
            console.print(
                "[yellow]⚠️  No baselines set. Run 'evalview baseline set' first.[/yellow]"
            )
        else:
            console.print()
            console.print(table)
            console.print()


@baseline.command("clear")
@click.option(
    "--test",
    help="Specific test name to clear baseline for",
)
@click.confirmation_option(prompt="Are you sure you want to clear baselines?")
def baseline_clear(test: str):
    """Clear baselines."""
    from evalview.tracking import RegressionTracker

    tracker = RegressionTracker()

    if test:
        # Clear specific baseline (would need to add this to DB class)
        console.print("[yellow]⚠️  Clear specific baseline not yet implemented[/yellow]")
    else:
        tracker.db.clear_baselines()
        console.print("[green]✅ All baselines cleared[/green]")


@main.command()
@click.option(
    "--days",
    default=30,
    help="Number of days to analyze (default: 30)",
)
@click.option(
    "--test",
    help="Specific test name to show trends for",
)
def trends(days: int, test: str):
    """Show performance trends over time."""
    from evalview.tracking import RegressionTracker
    from rich.table import Table

    tracker = RegressionTracker()

    if test:
        # Show trends for specific test
        stats = tracker.get_statistics(test, days)

        if stats["total_runs"] == 0:
            console.print(f"[yellow]⚠️  No data found for test: {test}[/yellow]")
            return

        console.print(f"\n[bold]Performance Trends: {test}[/bold]")
        console.print(f"Period: Last {days} days\n")

        console.print("[cyan]Test Runs:[/cyan]")
        console.print(f"  Total: {stats['total_runs']}")
        console.print(f"  Passed: {stats['passed_runs']} ({stats['pass_rate']:.1f}%)")
        console.print(f"  Failed: {stats['failed_runs']}")

        if stats["score"]["current"]:
            console.print("\n[cyan]Score:[/cyan]")
            console.print(f"  Current: {stats['score']['current']:.1f}")
            console.print(f"  Average: {stats['score']['avg']:.1f}")
            console.print(f"  Range: {stats['score']['min']:.1f} - {stats['score']['max']:.1f}")

        if stats["cost"]["current"]:
            console.print("\n[cyan]Cost:[/cyan]")
            console.print(f"  Current: ${stats['cost']['current']:.4f}")
            console.print(f"  Average: ${stats['cost']['avg']:.4f}")
            console.print(f"  Range: ${stats['cost']['min']:.4f} - ${stats['cost']['max']:.4f}")

        if stats["latency"]["current"]:
            console.print("\n[cyan]Latency:[/cyan]")
            console.print(f"  Current: {stats['latency']['current']:.0f}ms")
            console.print(f"  Average: {stats['latency']['avg']:.0f}ms")
            console.print(
                f"  Range: {stats['latency']['min']:.0f}ms - {stats['latency']['max']:.0f}ms"
            )

        console.print()

    else:
        # Show overall trends
        daily_trends = tracker.db.get_daily_trends(days)

        if not daily_trends:
            console.print(f"[yellow]⚠️  No trend data available for last {days} days[/yellow]")
            return

        console.print("\n[bold]Overall Performance Trends[/bold]")
        console.print(f"Period: Last {days} days\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Date", style="white")
        table.add_column("Avg Score", justify="right", style="green")
        table.add_column("Avg Cost", justify="right", style="yellow")
        table.add_column("Avg Latency", justify="right", style="blue")
        table.add_column("Tests", justify="center", style="dim")
        table.add_column("Pass Rate", justify="right", style="green")

        for trend in daily_trends[-14:]:  # Show last 14 days
            pass_rate = (
                trend["passed_tests"] / trend["total_tests"] * 100
                if trend["total_tests"] > 0
                else 0
            )

            table.add_row(
                trend["date"],
                f"{trend['avg_score']:.1f}" if trend["avg_score"] else "N/A",
                f"${trend['avg_cost']:.4f}" if trend.get("avg_cost") else "N/A",
                f"{trend['avg_latency']:.0f}ms" if trend.get("avg_latency") else "N/A",
                str(trend["total_tests"]),
                f"{pass_rate:.0f}%",
            )

        console.print(table)
        console.print()


if __name__ == "__main__":
    main()
