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
    "--output",
    default=".evalview/results",
    help="Output directory for results",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging (shows API requests/responses)",
)
def run(pattern: str, output: str, verbose: bool):
    """Run test cases against the agent."""
    asyncio.run(_run_async(pattern, output, verbose))


async def _run_async(pattern: str, output: str, verbose: bool):
    """Async implementation of run command."""
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
    if adapter_type in ["streaming", "tapescope", "jsonl"]:
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

    console.print(f"Found {len(test_cases)} test case(s)\n")

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
                # Execute agent
                trace = await adapter.execute(test_case.input.query, test_case.input.context)

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

    # Common endpoints to try
    common_endpoints = [
        ("LangGraph", "http://localhost:8000/api/chat"),
        ("LangGraph (alt)", "http://localhost:8000/invoke"),
        ("LangServe", "http://localhost:8000/agent"),
        ("TapeScope", "http://localhost:3000/api/unifiedchat"),
        ("Custom FastAPI", "http://localhost:8000/api/agent"),
        ("Custom Express", "http://localhost:3000/api/agent"),
    ]

    endpoints_to_test = []
    if endpoint:
        # User provided specific endpoint
        endpoints_to_test = [("Custom", endpoint)]
    else:
        # Try common ones
        endpoints_to_test = common_endpoints

    successful = None

    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in endpoints_to_test:
            try:
                console.print(f"[dim]Testing {name}: {url}...[/dim]", end=" ")

                # Try a simple POST request
                response = await client.post(
                    url,
                    json={"query": "test", "message": "test"},
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code in [200, 201, 422]:  # 422 might be validation error but server is running
                    console.print(f"[green]✅ Connected![/green]")
                    successful = (name, url, response)
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
        name, url, response = successful
        console.print(f"[green]✅ Successfully connected to {name}![/green]\n")

        # Show response info
        console.print("[cyan]Response details:[/cyan]")
        console.print(f"  • Status: {response.status_code}")
        console.print(f"  • Content-Type: {response.headers.get('content-type', 'N/A')}")

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

            # Determine adapter type
            is_streaming = "stream" in url.lower() or name == "TapeScope"
            adapter_type = "streaming" if is_streaming else "http"

            # Update config
            config["adapter"] = adapter_type
            config["endpoint"] = url

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            console.print(f"[green]✅ Updated config:[/green]")
            console.print(f"  • adapter: {adapter_type}")
            console.print(f"  • endpoint: {url}")
            console.print()
            console.print("[blue]Next steps:[/blue]")
            console.print("  1. Create test cases in tests/test-cases/")
            console.print("  2. Run: evalview run --verbose")
    else:
        console.print("[red]❌ Could not connect to any agent endpoint.[/red]\n")
        console.print("[yellow]Common issues:[/yellow]")
        console.print("  1. Agent server not running")
        console.print("  2. Wrong port number")
        console.print("  3. Firewall blocking connection")
        console.print()
        console.print("[blue]To start LangGraph agent:[/blue]")
        console.print("  cd /path/to/langgraph-example")
        console.print("  python main.py")
        console.print("  # or")
        console.print("  uvicorn main:app --reload --port 8000")
        console.print()
        console.print("[blue]Then run:[/blue]")
        console.print("  evalview connect")
        console.print("  # or specify endpoint:")
        console.print("  evalview connect --endpoint http://localhost:8000/api/chat")


if __name__ == "__main__":
    main()
