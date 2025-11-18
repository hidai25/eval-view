"""Console reporter for evaluation results."""

from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from agent_eval.core.types import EvaluationResult


class ConsoleReporter:
    """Generates formatted console output for evaluation results."""

    def __init__(self):
        self.console = Console()

    def print_summary(self, results: List[EvaluationResult]) -> None:
        """
        Print summary of evaluation results.

        Args:
            results: List of evaluation results
        """
        if not results:
            self.console.print("[yellow]No results to display[/yellow]")
            return

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        success_rate = (passed / len(results)) * 100 if results else 0

        # Summary table
        table = Table(title="📊 Evaluation Summary", show_header=True)
        table.add_column("Test Case", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Status")
        table.add_column("Cost", justify="right")
        table.add_column("Latency", justify="right")

        for result in results:
            status = "[green]✅ PASSED[/green]" if result.passed else "[red]❌ FAILED[/red]"
            score_color = "green" if result.score >= 80 else "yellow" if result.score >= 60 else "red"

            table.add_row(
                result.test_case,
                f"[{score_color}]{result.score:.1f}[/{score_color}]",
                status,
                f"${result.trace.metrics.total_cost:.4f}",
                f"{result.trace.metrics.total_latency:.0f}ms",
            )

        self.console.print(table)
        self.console.print()

        # Overall stats
        stats_panel = Panel(
            f"[green]✅ Passed: {passed}[/green]\n"
            f"[red]❌ Failed: {failed}[/red]\n"
            f"[blue]📈 Success Rate: {success_rate:.1f}%[/blue]",
            title="Overall Statistics",
            border_style="blue",
        )
        self.console.print(stats_panel)

    def print_detailed(self, result: EvaluationResult) -> None:
        """
        Print detailed evaluation result.

        Args:
            result: Single evaluation result
        """
        self.console.print(f"\n[bold cyan]Test Case: {result.test_case}[/bold cyan]")
        self.console.print(f"Score: {result.score:.1f}/100")
        self.console.print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")

        # Tool accuracy
        tool_eval = result.evaluations.tool_accuracy
        self.console.print(f"\n[bold]Tool Accuracy:[/bold] {tool_eval.accuracy * 100:.1f}%")
        if tool_eval.correct:
            self.console.print(f"  ✅ Correct: {', '.join(tool_eval.correct)}")
        if tool_eval.missing:
            self.console.print(f"  ❌ Missing: {', '.join(tool_eval.missing)}")
        if tool_eval.unexpected:
            self.console.print(f"  ⚠️  Unexpected: {', '.join(tool_eval.unexpected)}")

        # Output quality
        output_eval = result.evaluations.output_quality
        self.console.print(f"\n[bold]Output Quality:[/bold] {output_eval.score:.1f}/100")
        self.console.print(f"  Rationale: {output_eval.rationale}")

        # Costs and latency
        self.console.print(f"\n[bold]Performance:[/bold]")
        self.console.print(f"  Cost: ${result.trace.metrics.total_cost:.4f}")
        self.console.print(f"  Latency: {result.trace.metrics.total_latency:.0f}ms")
