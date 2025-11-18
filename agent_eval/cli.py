"""CLI entry point for AgentEval."""

import asyncio
import os
from pathlib import Path
from datetime import datetime
import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent_eval.core.loader import TestCaseLoader
from agent_eval.adapters.http_adapter import HTTPAdapter
from agent_eval.evaluators.evaluator import Evaluator
from agent_eval.reporters.json_reporter import JSONReporter
from agent_eval.reporters.console_reporter import ConsoleReporter

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """AgentEval - Testing framework for multi-step AI agents."""
    pass


@main.command()
@click.option(
    "--dir",
    default=".",
    help="Directory to initialize (default: current directory)",
)
def init(dir: str):
    """Initialize AgentEval in the current directory."""
    console.print("[blue]Initializing AgentEval project...[/blue]\n")

    base_path = Path(dir)

    # Create directories
    (base_path / ".agenteval").mkdir(exist_ok=True)
    (base_path / "tests" / "test-cases").mkdir(parents=True, exist_ok=True)

    # Create config file
    config_path = base_path / ".agenteval" / "config.yaml"
    if not config_path.exists():
        config_content = """# AgentEval Configuration
adapter: http
endpoint: http://localhost:3000/api/agent
timeout: 30.0
headers: {}
"""
        config_path.write_text(config_content)
        console.print("[green]✅ Created .agenteval/config.yaml[/green]")
    else:
        console.print("[yellow]⚠️  .agenteval/config.yaml already exists[/yellow]")

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
    console.print("  1. Edit .agenteval/config.yaml with your agent endpoint")
    console.print("  2. Write test cases in tests/test-cases/")
    console.print("  3. Run: agent-eval run\n")


@main.command()
@click.option(
    "--pattern",
    default="*.yaml",
    help="Test case file pattern (default: *.yaml)",
)
@click.option(
    "--output",
    default=".agenteval/results",
    help="Output directory for results",
)
def run(pattern: str, output: str):
    """Run test cases against the agent."""
    asyncio.run(_run_async(pattern, output))


async def _run_async(pattern: str, output: str):
    """Async implementation of run command."""
    console.print("[blue]Running test cases...[/blue]\n")

    # Load config
    config_path = Path(".agenteval/config.yaml")
    if not config_path.exists():
        console.print(
            "[red]❌ Config file not found. Run 'agent-eval init' first.[/red]"
        )
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize adapter
    adapter = HTTPAdapter(
        endpoint=config["endpoint"],
        headers=config.get("headers", {}),
        timeout=config.get("timeout", 30.0),
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
                failed += 1
                progress.update(
                    task,
                    description=f"[red]❌ {test_case.name} - ERROR: {str(e)}[/red]",
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
    from agent_eval.core.types import EvaluationResult

    results = [EvaluationResult(**data) for data in results_data]

    reporter = ConsoleReporter()

    if detailed:
        for result in results:
            reporter.print_detailed(result)
    else:
        reporter.print_summary(results)


if __name__ == "__main__":
    main()
