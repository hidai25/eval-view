"""Helper functions for skill testing commands.

This module contains extracted helper functions from the CLI skill test commands
to improve maintainability and testability.
"""

import json
from typing import Optional, Tuple, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from evalview.skills.agent_types import (
    AgentType,
    SkillAgentTestSuite,
    SkillAgentTestSuiteResult,
)
from evalview.skills.agent_runner import SkillAgentRunner
from evalview.skills.constants import (
    SCORE_THRESHOLD_HIGH,
    SCORE_THRESHOLD_MEDIUM,
    TRUNCATE_OUTPUT_SHORT,
    TRUNCATE_OUTPUT_MEDIUM,
    TRUNCATE_OUTPUT_LONG,
    PASS_RATE_HIGH,
    PASS_RATE_MEDIUM,
)


def validate_and_parse_agent_type(agent: Optional[str], console: Console) -> Optional[AgentType]:
    """Validate and parse agent type string to enum.

    Args:
        agent: Agent type string to validate
        console: Rich console for error output

    Returns:
        AgentType enum value or None if agent is empty

    Raises:
        SystemExit: If agent type is invalid
    """
    if not agent:
        return None

    try:
        return AgentType(agent)
    except ValueError:
        console.print(f"[red]Error: Unknown agent type: {agent}[/red]")
        console.print(f"[dim]Available types: {', '.join(a.value for a in AgentType)}[/dim]")
        raise SystemExit(1)


def load_test_suite(
    test_file: str,
    agent_type_enum: Optional[AgentType],
    trace_dir: Optional[str],
    no_rubric: bool,
    cwd: Optional[str],
    max_turns: Optional[int],
    verbose: bool,
    model: str,
    console: Console,
) -> Tuple[SkillAgentTestSuite, SkillAgentRunner]:
    """Load test suite from YAML file with configuration.

    Args:
        test_file: Path to YAML test file
        agent_type_enum: Override agent type
        trace_dir: Directory for traces
        no_rubric: Skip rubric evaluation
        cwd: Working directory override
        max_turns: Max conversation turns
        verbose: Show detailed output
        model: Model for rubric evaluation
        console: Rich console for error output

    Returns:
        Tuple of (test_suite, runner)

    Raises:
        SystemExit: If loading fails
    """
    try:
        runner = SkillAgentRunner(
            verbose=verbose,
            skip_rubric=no_rubric,
            trace_dir=trace_dir,
            rubric_model=model,
        )
        suite = runner.load_test_suite(
            yaml_path=test_file,
            agent_type_override=agent_type_enum,
            cwd_override=cwd,
            max_turns_override=max_turns,
        )
        return suite, runner
    except Exception as e:
        console.print(f"[red]Error loading test suite: {e}[/red]")
        raise SystemExit(1)


def print_suite_info(
    suite: SkillAgentTestSuite,
    trace_dir: Optional[str],
    console: Console,
) -> None:
    """Print test suite metadata and configuration.

    Args:
        suite: The loaded test suite
        trace_dir: Trace directory path (optional)
        console: Rich console for output
    """
    console.print(f"  [bold]Suite:[/bold]  {suite.name}")
    console.print(f"  [bold]Skill:[/bold]  [cyan]{suite.skill}[/cyan]")
    console.print(f"  [bold]Agent:[/bold]  [magenta]{suite.agent.type.value}[/magenta]")
    console.print(f"  [bold]Tests:[/bold]  {len(suite.tests)}")
    if trace_dir:
        console.print(f"  [bold]Traces:[/bold] [dim]{trace_dir}[/dim]")
    console.print()


def format_results_as_json(result: SkillAgentTestSuiteResult) -> Dict[str, Any]:
    """Convert test results to JSON-serializable dictionary.

    Args:
        result: Test execution results

    Returns:
        Dictionary ready for JSON serialization
    """
    return {
        "suite_name": result.suite_name,
        "skill_name": result.skill_name,
        "agent_type": result.agent_type.value,
        "passed": result.passed,
        "total_tests": result.total_tests,
        "passed_tests": result.passed_tests,
        "failed_tests": result.failed_tests,
        "pass_rate": result.pass_rate,
        "by_category": {k.value: v for k, v in result.by_category.items()},
        "total_latency_ms": result.total_latency_ms,
        "avg_latency_ms": result.avg_latency_ms,
        "total_tokens": result.total_tokens,
        "results": [
            {
                "test_name": r.test_name,
                "category": r.category.value,
                "passed": r.passed,
                "score": r.score,
                "input": r.input_query,
                "output": r.final_output[:TRUNCATE_OUTPUT_LONG] + "..." if len(r.final_output) > TRUNCATE_OUTPUT_LONG else r.final_output,
                "deterministic": {
                    "passed": r.deterministic.passed,
                    "score": r.deterministic.score,
                    "passed_count": r.deterministic.passed_count,
                    "total_count": r.deterministic.total_count,
                    "failed_checks": [
                        {"name": c.check_name, "message": c.message}
                        for c in r.deterministic.failed_checks
                    ],
                } if r.deterministic else None,
                "rubric": {
                    "passed": r.rubric.passed,
                    "score": r.rubric.score,
                    "rationale": r.rubric.rationale,
                } if r.rubric else None,
                "trace_path": r.trace_path,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in result.results
        ],
    }


def build_results_table(result: SkillAgentTestSuiteResult) -> Table:
    """Build Rich table with test results.

    Args:
        result: Test execution results

    Returns:
        Rich Table object ready for rendering
    """
    table = Table(title="Agent Test Results", show_header=True, header_style="bold cyan")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Test", style="cyan")
    table.add_column("Category", width=10)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Phase 1", justify="center", width=8)
    table.add_column("Phase 2", justify="center", width=8)
    table.add_column("Latency", justify="right", width=10)

    for r in result.results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        score_color = "green" if r.score >= SCORE_THRESHOLD_HIGH else "yellow" if r.score >= SCORE_THRESHOLD_MEDIUM else "red"

        # Phase 1 (deterministic)
        if r.deterministic:
            p1_icon = "[green]✓[/green]" if r.deterministic.passed else "[red]✗[/red]"
        else:
            p1_icon = "[dim]-[/dim]"

        # Phase 2 (rubric)
        if r.rubric:
            p2_icon = "[green]✓[/green]" if r.rubric.passed else "[red]✗[/red]"
        else:
            p2_icon = "[dim]-[/dim]"

        table.add_row(
            status,
            r.test_name,
            r.category.value,
            f"[{score_color}]{r.score:.0f}%[/{score_color}]",
            p1_icon,
            p2_icon,
            f"{r.latency_ms:.0f}ms",
        )

    return table


def print_detailed_test_results(
    result: SkillAgentTestSuiteResult,
    verbose: bool,
    console: Console,
) -> None:
    """Print detailed test results for failed or verbose mode.

    Args:
        result: Test execution results
        verbose: Show all results (True) or only failed (False)
        console: Rich console for output
    """
    # Determine which results to show
    failed_results = [r for r in result.results if not r.passed]
    show_results = result.results if verbose else failed_results

    if not show_results:
        return

    for r in show_results:
        status_icon = "✓" if r.passed else "✗"
        status_color = "green" if r.passed else "red"

        console.print(f"[bold {status_color}]{status_icon} {r.test_name}[/bold {status_color}] [{r.category.value}]")

        # Show query
        console.print("\n[bold]Input:[/bold]")
        query = r.input_query[:TRUNCATE_OUTPUT_SHORT] + "..." if len(r.input_query) > TRUNCATE_OUTPUT_SHORT else r.input_query
        for line in query.split('\n'):
            console.print(f"  [dim]{line}[/dim]")

        # Show response preview
        if verbose or not r.passed:
            console.print("\n[bold]Response:[/bold]")
            output = r.final_output[:TRUNCATE_OUTPUT_MEDIUM] + "..." if len(r.final_output) > TRUNCATE_OUTPUT_MEDIUM else r.final_output
            for line in output.split('\n')[:8]:
                console.print(f"  {line}")
            if len(r.final_output.split('\n')) > 8:
                console.print("  [dim]...[/dim]")

        # Show Phase 1 results
        if r.deterministic:
            console.print("\n[bold]Phase 1 (Deterministic):[/bold]")
            p1_status = "[green]PASSED[/green]" if r.deterministic.passed else "[red]FAILED[/red]"
            console.print(f"  Status: {p1_status} ({r.deterministic.passed_count}/{r.deterministic.total_count} checks)")

            for check in r.deterministic.checks:
                check_icon = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
                console.print(f"  {check_icon} {check.check_name}: {check.message}")

        # Show Phase 2 results
        if r.rubric:
            console.print("\n[bold]Phase 2 (Rubric):[/bold]")
            p2_status = "[green]PASSED[/green]" if r.rubric.passed else "[red]FAILED[/red]"
            console.print(f"  Status: {p2_status} (score: {r.rubric.score:.0f}/{r.rubric.min_score:.0f})")
            console.print(f"  [dim]{r.rubric.rationale[:TRUNCATE_OUTPUT_SHORT]}...[/dim]" if len(r.rubric.rationale) > TRUNCATE_OUTPUT_SHORT else f"  [dim]{r.rubric.rationale}[/dim]")

        # Show trace path
        if r.trace_path:
            console.print(f"\n[dim]Trace: {r.trace_path}[/dim]")

        # Error if any
        if r.error:
            console.print(f"\n[bold red]Error:[/bold red] {r.error}")

        console.print()


def build_summary_panel(
    result: SkillAgentTestSuiteResult,
    elapsed_ms: float,
) -> Panel:
    """Build Rich panel with test summary statistics.

    Args:
        result: Test execution results
        elapsed_ms: Total execution time in milliseconds

    Returns:
        Rich Panel with formatted summary
    """
    pass_rate_color = "green" if result.pass_rate >= PASS_RATE_HIGH else "yellow" if result.pass_rate >= PASS_RATE_MEDIUM else "red"
    status_text = "[green]● All Tests Passed[/green]" if result.passed else "[bold red]● Some Tests Failed[/bold red]"
    border_color = "green" if result.passed else "red"

    # Category breakdown
    category_lines = []
    for cat, stats in result.by_category.items():
        cat_pass_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        cat_color = "green" if cat_pass_rate >= PASS_RATE_HIGH else "yellow" if cat_pass_rate >= PASS_RATE_MEDIUM else "red"
        category_lines.append(
            f"  [bold]{cat.value}:[/bold] [{cat_color}]{stats['passed']}/{stats['total']}[/{cat_color}]"
        )
    category_str = "\n".join(category_lines) if category_lines else "  [dim]No categories[/dim]"

    summary_content = (
        f"  {status_text}\n"
        f"\n"
        f"  [bold]✓ Passed:[/bold]       [green]{result.passed_tests}[/green]\n"
        f"  [bold]✗ Failed:[/bold]       [red]{result.failed_tests}[/red]\n"
        f"  [bold]Pass Rate:[/bold]    [{pass_rate_color}]{result.pass_rate:.0%}[/{pass_rate_color}]\n"
        f"\n"
        f"  [bold]By Category:[/bold]\n{category_str}\n"
        f"\n"
        f"  [bold]Avg Latency:[/bold] {result.avg_latency_ms:.0f}ms\n"
        f"  [bold]Total Tokens:[/bold] {result.total_tokens:,}\n"
        f"  [bold]Total Time:[/bold]  {elapsed_ms:.0f}ms"
    )

    return Panel(summary_content, title="[bold]Agent Test Results[/bold]", border_style=border_color)


def handle_test_completion(
    result: SkillAgentTestSuiteResult,
    test_file: str,
    suite: SkillAgentTestSuite,
    console: Console,
) -> None:
    """Handle test completion with appropriate messaging.

    Args:
        result: Test execution results
        test_file: Path to test file (for re-run command)
        suite: Test suite info (for agent type)
        console: Rich console for output

    Raises:
        SystemExit(1): If tests failed
    """
    if not result.passed:
        console.print()
        console.print("[bold yellow]Agent Skill Test Failed[/bold yellow]")
        console.print()
        console.print("[bold]Next Steps:[/bold]")
        console.print("  1. Review failed checks in Phase 1 (Deterministic)")
        console.print("  2. Check trace files for detailed execution logs")
        console.print("  3. Update skill instructions in SKILL.md")
        console.print(f"  4. Re-run: [dim]evalview skill test {test_file} --agent {suite.agent.type.value}[/dim]")
        console.print()
        raise SystemExit(1)
    else:
        console.print()
        console.print("[bold green]✓ Agent skill tests passed[/bold green]")
        console.print()
