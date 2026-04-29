"""StatisticalReporterMixin — statistical/variance reporting for `ConsoleReporter`.

Inherits-into ConsoleReporter so the parent class stays focused on the core
single-run summary/detail/timeline output. All methods here read `self.console`
(set by the parent `__init__`) but are otherwise self-contained.
"""

from typing import List

from rich.panel import Panel
from rich.table import Table

from evalview.core.types import (
    EvaluationResult,
    FlakinessScore,
    StatisticalEvaluationResult,
    StatisticalMetrics,
)


class StatisticalReporterMixin:
    """Console-rendering helpers for statistical evaluation results."""

    def print_statistical_summary(
        self,
        result: StatisticalEvaluationResult,
        show_individual_runs: bool = False,
    ) -> None:
        """Print a comprehensive statistical evaluation summary.

        Args:
            result: Statistical evaluation result.
            show_individual_runs: Whether to show details of each individual run.
        """
        # Header with pass/fail status
        status_icon = "✅" if result.passed else "❌"
        status_color = "green" if result.passed else "red"
        status_text = "PASSED" if result.passed else "FAILED"

        self.console.print()
        self.console.print(
            f"[bold {status_color}]{status_icon} Statistical Evaluation: {result.test_case}[/bold {status_color}]"
        )
        self.console.print(f"[bold {status_color}]{status_text}[/bold {status_color}]")
        self.console.print()

        # Run summary panel
        pass_rate_color = "green" if result.pass_rate >= 0.8 else "yellow" if result.pass_rate >= 0.5 else "red"

        # pass@k interpretation
        pass_at_k_color = "green" if result.pass_at_k >= 0.95 else "yellow" if result.pass_at_k >= 0.8 else "red"
        pass_at_k_meaning = "usually finds a solution" if result.pass_at_k >= 0.8 else "inconsistent"

        # pass^k interpretation
        pass_power_k_color = "green" if result.pass_power_k >= 0.5 else "yellow" if result.pass_power_k >= 0.2 else "red"
        pass_power_k_meaning = "reliable" if result.pass_power_k >= 0.5 else "needs improvement" if result.pass_power_k >= 0.2 else "unreliable"

        run_summary = (
            f"  [bold]Total Runs:[/bold]     {result.total_runs}\n"
            f"  [bold]Passed:[/bold]         [green]{result.successful_runs}[/green]\n"
            f"  [bold]Failed:[/bold]         [red]{result.failed_runs}[/red]\n"
            f"  [bold]Pass Rate:[/bold]      [{pass_rate_color}]{result.pass_rate:.1%}[/{pass_rate_color}] "
            f"(required: {result.required_pass_rate:.1%})\n"
            f"\n"
            f"  [bold]Reliability Metrics:[/bold]\n"
            f"  [bold]pass@{result.total_runs}:[/bold]       [{pass_at_k_color}]{result.pass_at_k:.1%}[/{pass_at_k_color}] "
            f"[dim]({pass_at_k_meaning})[/dim]\n"
            f"  [bold]pass^{result.total_runs}:[/bold]       [{pass_power_k_color}]{result.pass_power_k:.1%}[/{pass_power_k_color}] "
            f"[dim]({pass_power_k_meaning})[/dim]"
        )
        self.console.print(Panel(run_summary, title="[bold]Run Summary[/bold]", border_style="cyan"))

        # Score statistics table
        self._print_statistics_table(result.score_stats, "Score Statistics", unit="pts")

        # Cost statistics (if available)
        if result.cost_stats:
            self._print_statistics_table(result.cost_stats, "Cost Statistics", unit="$", precision=4)

        # Latency statistics (if available)
        if result.latency_stats:
            self._print_statistics_table(result.latency_stats, "Latency Statistics", unit="ms", precision=0)

        # Flakiness assessment
        self._print_flakiness_panel(result.flakiness)

        # Failure reasons (if any)
        if result.failure_reasons:
            self.console.print()
            self.console.print("[bold red]Failure Reasons:[/bold red]")
            for reason in result.failure_reasons:
                self.console.print(f"  [yellow]• {reason}[/yellow]")

        # Individual run details (optional)
        if show_individual_runs and result.individual_results:
            self._print_individual_runs_table(result.individual_results)

        # Configuration summary
        config = result.variance_config
        self.console.print()
        self.console.print("[dim]Configuration:[/dim]")
        self.console.print(f"  [dim]runs: {config.runs}, pass_rate: {config.pass_rate}, confidence: {config.confidence_level}[/dim]")

    def _print_statistics_table(
        self,
        stats: StatisticalMetrics,
        title: str,
        unit: str = "",
        precision: int = 2,
    ) -> None:
        """Print a formatted statistics table.

        Args:
            stats: Statistical metrics to render.
            title: Table title.
            unit: Unit suffix to display (e.g. "$", "ms", "pts").
            precision: Decimal precision used for formatted values.
        """
        self.console.print()

        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("", width=30)  # Visual indicator column

        def fmt(val: float) -> str:
            if unit == "$":
                return f"${val:.{precision}f}"
            elif unit == "ms":
                return f"{val:.{precision}f}ms"
            else:
                return f"{val:.{precision}f}{unit}"

        # Mean with CI
        ci_str = f"[{stats.confidence_interval_lower:.{precision}f}, {stats.confidence_interval_upper:.{precision}f}]"
        ci_pct = int(stats.confidence_level * 100)
        table.add_row("Mean", fmt(stats.mean), f"[dim]{ci_pct}% CI: {ci_str}[/dim]")

        # Std Dev and Variance
        table.add_row("Std Dev", fmt(stats.std_dev), self._get_variance_indicator(stats.std_dev, stats.mean))
        table.add_row("Variance", f"{stats.variance:.{precision}f}", "")

        # Min/Max range
        range_val = stats.max_value - stats.min_value
        table.add_row("Min", fmt(stats.min_value), "")
        table.add_row("Max", fmt(stats.max_value), f"[dim]range: {range_val:.{precision}f}[/dim]")

        # Percentiles
        table.add_row("Median (P50)", fmt(stats.median), "")
        table.add_row("P25", fmt(stats.percentile_25), "")
        table.add_row("P75", fmt(stats.percentile_75), "")
        table.add_row("P95", fmt(stats.percentile_95), "")

        self.console.print(table)

    def _get_variance_indicator(self, std_dev: float, mean: float) -> str:
        """Get a visual indicator for variance level.

        Args:
            std_dev: Standard deviation for the distribution.
            mean: Mean value for the distribution.

        Returns:
            A Rich markup string describing the variance level.
        """
        if mean == 0:
            return ""

        cv = (std_dev / mean) * 100  # Coefficient of variation
        if cv < 5:
            return "[green]▁▁▁▁▁ Low variance[/green]"
        elif cv < 10:
            return "[green]▂▂▁▁▁ Low variance[/green]"
        elif cv < 20:
            return "[yellow]▄▄▂▁▁ Moderate variance[/yellow]"
        elif cv < 30:
            return "[yellow]▆▆▄▂▁ High variance[/yellow]"
        else:
            return "[red]█████ Very high variance[/red]"

    def _print_flakiness_panel(self, flakiness: FlakinessScore) -> None:
        """Print the flakiness assessment panel.

        Args:
            flakiness: Flakiness score and associated metadata.
        """
        self.console.print()

        # Color based on category
        category_colors = {
            "stable": "green",
            "low_variance": "green",
            "moderate_variance": "yellow",
            "high_variance": "yellow",
            "flaky": "red",
        }
        color = category_colors.get(flakiness.category, "white")

        # Visual flakiness bar
        filled = int(flakiness.score * 10)
        bar = "█" * filled + "░" * (10 - filled)

        content = (
            f"  [bold]Flakiness Score:[/bold] [{color}]{flakiness.score:.2f}[/{color}] [{color}]{bar}[/{color}]\n"
            f"  [bold]Category:[/bold]        [{color}]{flakiness.category}[/{color}]\n"
            f"  [bold]Pass Rate:[/bold]       {flakiness.pass_rate:.1%}\n"
            f"  [bold]Score CV:[/bold]        {flakiness.score_coefficient_of_variation:.1f}%"
        )

        if flakiness.contributing_factors and flakiness.contributing_factors != ["none"]:
            content += "\n\n  [bold]Contributing Factors:[/bold]"
            for factor in flakiness.contributing_factors:
                content += f"\n    [dim]• {factor}[/dim]"

        self.console.print(Panel(content, title="[bold]Flakiness Assessment[/bold]", border_style=color))

    def _print_individual_runs_table(self, results: List[EvaluationResult]) -> None:
        """Print a table showing individual run results.

        Args:
            results: Per-run evaluation results to render.
        """
        self.console.print()

        table = Table(title="Individual Runs", show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=4)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Score", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Tool Acc", justify="right")

        for i, result in enumerate(results, 1):
            status = "[green]✓ Pass[/green]" if result.passed else "[red]✗ Fail[/red]"
            score_color = "green" if result.score >= 80 else "yellow" if result.score >= 60 else "red"

            table.add_row(
                str(i),
                status,
                f"[{score_color}]{result.score:.1f}[/{score_color}]",
                f"${result.trace.metrics.total_cost:.4f}",
                f"{result.trace.metrics.total_latency:.0f}ms",
                f"{result.evaluations.tool_accuracy.accuracy * 100:.0f}%",
            )

        self.console.print(table)

    def print_statistical_comparison(
        self,
        results: List[StatisticalEvaluationResult],
    ) -> None:
        """Print a comparison table of multiple statistical evaluations.

        Args:
            results: Statistical evaluation results to compare.
        """
        if not results:
            self.console.print("[yellow]No results to compare[/yellow]")
            return

        self.console.print()
        self.console.print("[bold]━━━ Statistical Comparison ━━━[/bold]")
        self.console.print()

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Test Case", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Pass Rate", justify="right")
        table.add_column("pass@k", justify="right")
        table.add_column("pass^k", justify="right")
        table.add_column("Mean Score", justify="right")
        table.add_column("Flakiness", justify="center")

        for result in results:
            status = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            pass_color = "green" if result.pass_rate >= 0.8 else "yellow" if result.pass_rate >= 0.5 else "red"
            score_color = "green" if result.score_stats.mean >= 80 else "yellow" if result.score_stats.mean >= 60 else "red"

            # pass@k coloring (high is good - "will it work eventually?")
            pass_at_k_color = "green" if result.pass_at_k >= 0.95 else "yellow" if result.pass_at_k >= 0.8 else "red"

            # pass^k coloring (reliability - "will it work every time?")
            pass_power_k_color = "green" if result.pass_power_k >= 0.5 else "yellow" if result.pass_power_k >= 0.2 else "red"

            flakiness_icons = {
                "stable": "[green]●[/green]",
                "low_variance": "[green]◐[/green]",
                "moderate_variance": "[yellow]◑[/yellow]",
                "high_variance": "[yellow]○[/yellow]",
                "flaky": "[red]◌[/red]",
            }
            flakiness_icon = flakiness_icons.get(result.flakiness.category, "?")

            table.add_row(
                result.test_case,
                status,
                f"[{pass_color}]{result.pass_rate:.1%}[/{pass_color}]",
                f"[{pass_at_k_color}]{result.pass_at_k:.1%}[/{pass_at_k_color}]",
                f"[{pass_power_k_color}]{result.pass_power_k:.1%}[/{pass_power_k_color}]",
                f"[{score_color}]{result.score_stats.mean:.1f}[/{score_color}]",
                f"{flakiness_icon} {result.flakiness.category}",
            )

        self.console.print(table)

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        flaky_count = sum(1 for r in results if r.flakiness.category in ("high_variance", "flaky"))

        self.console.print()
        self.console.print(f"[bold]Summary:[/bold] {passed}/{total} passed, {flaky_count} flaky tests")
        self.console.print()
