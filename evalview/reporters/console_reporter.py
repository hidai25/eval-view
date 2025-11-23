"""Console reporter for evaluation results."""

from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from evalview.core.types import EvaluationResult


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
        table.add_column("Tokens", justify="right")
        table.add_column("Latency", justify="right")

        for result in results:
            status = "[green]✅ PASSED[/green]" if result.passed else "[red]❌ FAILED[/red]"
            score_color = "green" if result.score >= 80 else "yellow" if result.score >= 60 else "red"

            # Format token usage
            tokens_usage = result.trace.metrics.total_tokens
            if tokens_usage:
                tokens_str = f"{tokens_usage.total_tokens:,}"
                if tokens_usage.cached_tokens > 0:
                    tokens_str += f"\n({tokens_usage.cached_tokens:,} cached)"
            else:
                tokens_str = "N/A"

            table.add_row(
                result.test_case,
                f"[{score_color}]{result.score:.1f}[/{score_color}]",
                status,
                f"${result.trace.metrics.total_cost:.4f}",
                tokens_str,
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

        # Show failure details inline
        for result in results:
            if not result.passed:
                self.console.print(f"\n[bold red]❌ {result.test_case} - Failure Details:[/bold red]")

                # Show why it failed
                issues = []

                # Tool accuracy issues
                tool_eval = result.evaluations.tool_accuracy
                if tool_eval.missing:
                    issues.append(f"  • Missing tools: {', '.join(tool_eval.missing)}")
                if tool_eval.unexpected:
                    issues.append(f"  • Unexpected tools: {', '.join(tool_eval.unexpected)}")

                # Output quality issues
                output_eval = result.evaluations.output_quality
                if output_eval.score < 70:
                    issues.append(f"  • Low output quality: {output_eval.score:.0f}/100")
                    issues.append(f"    Reason: {output_eval.rationale[:100]}...")
                if output_eval.contains_checks.failed:
                    issues.append(f"  • Missing required text: {', '.join(output_eval.contains_checks.failed)}")
                if output_eval.not_contains_checks.failed:
                    issues.append(f"  • Contains forbidden text: {', '.join(output_eval.not_contains_checks.failed)}")

                # Cost issues
                cost_eval = result.evaluations.cost
                if not cost_eval.passed:
                    issues.append(f"  • Cost exceeded: ${cost_eval.total_cost:.4f} > ${cost_eval.threshold:.4f}")

                # Latency issues
                latency_eval = result.evaluations.latency
                if not latency_eval.passed:
                    issues.append(f"  • Latency exceeded: {latency_eval.total_latency:.0f}ms > {latency_eval.threshold:.0f}ms")

                for issue in issues:
                    self.console.print(f"[yellow]{issue}[/yellow]")

                # Show actual vs expected for debugging
                self.console.print(f"\n[dim]  Steps captured: {len(result.trace.steps)}[/dim]")
                if result.trace.steps:
                    self.console.print(f"[dim]  Tools called: {', '.join([s.tool_name for s in result.trace.steps])}[/dim]")

    def print_detailed(self, result: EvaluationResult) -> None:
        """
        Print detailed evaluation result.

        Args:
            result: Single evaluation result
        """
        self.console.print(f"\n[bold cyan]Test Case: {result.test_case}[/bold cyan]")
        self.console.print(f"Score: {result.score:.1f}/100")
        self.console.print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")

        # Show query and response
        if result.input_query:
            self.console.print(f"\n[bold]Query:[/bold]")
            self.console.print(f"  {result.input_query}")

        if result.actual_output:
            self.console.print(f"\n[bold]Response:[/bold]")
            # Truncate long responses
            output = result.actual_output
            if len(output) > 300:
                output = output[:300] + "..."
            self.console.print(f"  {output}")

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

        # Token usage breakdown
        tokens_usage = result.trace.metrics.total_tokens
        if tokens_usage:
            self.console.print(f"  Tokens: {tokens_usage.total_tokens:,} total")
            self.console.print(f"    • Input: {tokens_usage.input_tokens:,}")
            self.console.print(f"    • Output: {tokens_usage.output_tokens:,}")
            if tokens_usage.cached_tokens > 0:
                self.console.print(f"    • Cached: {tokens_usage.cached_tokens:,} (90% discount)")

        self.console.print(f"  Latency: {result.trace.metrics.total_latency:.0f}ms")
