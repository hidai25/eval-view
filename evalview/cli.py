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
from evalview.core.pricing import MODEL_PRICING, get_model_pricing_info
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
            input_price = click.prompt("Input tokens ($ per 1M)", type=float, default=pricing['input_price_per_1m'])
            output_price = click.prompt("Output tokens ($ per 1M)", type=float, default=pricing['output_price_per_1m'])
            cached_price = click.prompt("Cached tokens ($ per 1M)", type=float, default=pricing['cached_price_per_1m'])

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
def run(pattern: str, test: tuple, filter: str, output: str, verbose: bool):
    """Run test cases against the agent."""
    asyncio.run(_run_async(pattern, test, filter, output, verbose))


async def _run_async(pattern: str, test: tuple, filter: str, output: str, verbose: bool):
    """Async implementation of run command."""
    import fnmatch

    if verbose:
        console.print("[dim]🔍 Verbose mode enabled[/dim]\n")

    console.print("[blue]Running test cases...[/blue]\n")

    # Load config
    config_path = Path(".evalview/config.yaml")
    if not config_path.exists():
        console.print(
            "[red]❌ Config file not found. Run 'evalview init' first.[/red]"
        )
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract model config
    model_config = config.get("model", {})
    if verbose and model_config:
        console.print(f"[dim]💰 Model: {model_config.get('name', 'gpt-5-mini')}[/dim]")
        if "pricing" in model_config:
            console.print(f"[dim]💵 Custom pricing: ${model_config['pricing']['input_per_1m']:.2f} in, ${model_config['pricing']['output_per_1m']:.2f} out[/dim]")

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
        )
    elif adapter_type == "crewai":
        adapter = CrewAIAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 120.0),
            verbose=verbose,
            model_config=model_config,
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
        )
    else:
        # HTTP adapter for standard REST APIs
        adapter = HTTPAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            model_config=model_config,
        )

    # Initialize evaluator
    evaluator = Evaluator(openai_api_key=os.getenv("OPENAI_API_KEY"))

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
                if '*' in filter or '?' in filter:
                    if fnmatch.fnmatch(test_name_lower, filter_lower):
                        filtered_cases.append(test_case)
                        continue
                # Otherwise, do substring match (more user-friendly)
                elif filter_lower in test_name_lower:
                    filtered_cases.append(test_case)
                    continue

        test_cases = filtered_cases

        if not test_cases:
            console.print(f"[yellow]⚠️  No test cases matched the filter criteria[/yellow]")
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
                console.print(f"[dim]  Using test-specific adapter: {test_adapter_type} @ {test_endpoint}[/dim]")

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
                )
            elif test_adapter_type == "crewai":
                return CrewAIAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 120.0),
                    verbose=verbose,
                    model_config=model_config,
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
                    streaming=test_config.get("streaming", True),
                    verbose=verbose,
                    model_config=model_config,
                )
            else:  # Default to HTTP adapter
                return HTTPAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 30.0),
                    streaming=test_config.get("streaming", False),
                    verbose=verbose,
                    model_config=model_config,
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
                    console.print(f"\n[red]❌ Connection Error:[/red] Agent server not reachable at {config['endpoint']}")
                    console.print("[yellow]💡 Tip:[/yellow] Run 'evalview connect' to test and configure your endpoint\n")
                elif isinstance(e, httpx.TimeoutException):
                    error_msg = "Request timeout"
                    console.print(f"\n[yellow]⏱️  Timeout:[/yellow] Agent took too long to respond (>{config.get('timeout', 30)}s)")
                    console.print("[yellow]💡 Tip:[/yellow] Increase timeout in .evalview/config.yaml or optimize your agent\n")

                progress.update(
                    task,
                    description=f"[red]❌ {test_case.name} - ERROR: {error_msg}[/red]",
                )

            progress.remove_task(task)

    # Print summary
    console.print()
    reporter = ConsoleReporter()
    reporter.print_summary(results)

    # Save results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    JSONReporter.save(results, results_file)

    console.print(f"\n[dim]Results saved to: {results_file}[/dim]\n")


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
                    json={"query": "test", "message": "test", "messages": [{"role": "user", "content": "test"}]},
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code in [200, 201, 422]:  # 422 might be validation error but server is running
                    # Try to detect framework from response
                    detected_adapter = default_adapter
                    try:
                        if response.headers.get('content-type', '').startswith('application/json'):
                            data = response.json()
                            # LangGraph detection
                            if "messages" in data or "thread_id" in data:
                                detected_adapter = "langgraph"
                            # CrewAI detection
                            elif "tasks" in data or "crew_id" in data:
                                detected_adapter = "crewai"
                    except:
                        pass

                    console.print(f"[green]✅ Connected![/green]")
                    successful = (name, url, response, detected_adapter)
                    break
                else:
                    console.print(f"[yellow]❌ HTTP {response.status_code}[/yellow]")

            except httpx.ConnectError:
                console.print(f"[red]❌ Connection refused[/red]")
            except httpx.TimeoutException:
                console.print(f"[yellow]❌ Timeout[/yellow]")
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
            if response.headers.get('content-type', '').startswith('application/json'):
                data = response.json()
                console.print(f"  • Response keys: {list(data.keys())}")
        except:
            pass

        # Ask if user wants to update config
        console.print()
        if click.confirm(f"Update .evalview/config.yaml to use this endpoint?", default=True):
            config_path = Path(".evalview/config.yaml")

            if not config_path.exists():
                console.print("[yellow]⚠️  Config file not found. Run 'evalview init' first.[/yellow]")
                return

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Update config with detected adapter
            config["adapter"] = detected_adapter
            config["endpoint"] = url

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            console.print(f"[green]✅ Updated config:[/green]")
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
                    console.print(f"  • Port {port}: [green]Open[/green] (HTTP {response.status_code})")
                except:
                    pass

        if open_ports:
            console.print()
            console.print(f"[yellow]Found {len(open_ports)} open port(s)![/yellow]")
            console.print("[blue]Try connecting to one of these manually:[/blue]")
            for port in open_ports:
                console.print(f"  evalview connect --endpoint http://127.0.0.1:{port}/api/chat")
            console.print()

            if click.confirm("Do you want to try a custom endpoint?", default=True):
                custom_port = click.prompt("Enter port number", type=int, default=open_ports[0] if open_ports else 8000)
                custom_path = click.prompt("Enter endpoint path", default="/api/chat")
                custom_url = f"http://127.0.0.1:{custom_port}{custom_path}"

                console.print(f"\n[cyan]Testing {custom_url}...[/cyan]")

                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.post(
                            custom_url,
                            json={"query": "test", "message": "test", "messages": [{"role": "user", "content": "test"}]},
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
                            except:
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

                                console.print(f"[green]✅ Config updated:[/green]")
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


async def _record_async(query: Optional[str], output: Optional[str], interactive: bool, verbose: bool):
    """Async implementation of record command."""
    from evalview.recorder import TestCaseRecorder

    console.print("[blue]🎬 Recording mode started[/blue]")
    console.print("━" * 60)
    console.print()

    # Load config
    config_path = Path(".evalview/config.yaml")
    if not config_path.exists():
        console.print(
            "[red]❌ Config file not found. Run 'evalview init' first.[/red]"
        )
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract model config
    model_config = config.get("model", {})

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
        )
    elif adapter_type == "crewai":
        adapter = CrewAIAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            verbose=verbose,
            model_config=model_config,
        )
    elif adapter_type in ["streaming", "tapescope", "jsonl"]:
        adapter = TapeScopeAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 60.0),
            verbose=verbose,
            model_config=model_config,
        )
    else:
        # HTTP adapter for standard REST APIs
        adapter = HTTPAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            model_config=model_config,
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
        console.print("[yellow]💡 Tip: Type 'done' when finished, 'skip' to cancel current recording[/yellow]\n")

        query_num = 1
        while True:
            # Get query from user
            if not query:
                console.print(f"[bold]📝 Enter query #{query_num} (or 'done' to finish):[/bold] ", end="")
                user_input = input().strip()

                if user_input.lower() == 'done':
                    break
                elif user_input.lower() == 'skip':
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
                console.print(f"[bold]✍️  Test case name [[dim]{test_case.name}[/dim]]:[/bold] ", end="")
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


if __name__ == "__main__":
    main()
