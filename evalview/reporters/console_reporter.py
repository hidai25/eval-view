"""Console reporter for evaluation results."""

import json
from typing import List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from evalview.core.types import EvaluationResult, StepTrace


class ConsoleReporter:
    """Generates formatted console output for evaluation results."""

    def __init__(self):
        self.console = Console()

    def _format_value(self, value: Any, max_length: int = 60) -> str:
        """Format a value for display, truncating if needed."""
        if value is None:
            return "[dim]null[/dim]"
        if isinstance(value, dict):
            text = json.dumps(value, default=str)
        elif isinstance(value, list):
            text = json.dumps(value, default=str)
        else:
            text = str(value)

        if len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text

    def print_step_timeline(self, steps: List[StepTrace], title: str = "Agent Flow") -> None:
        """
        Print a visual step-by-step timeline of agent execution.

        Args:
            steps: List of step traces from execution
            title: Title for the timeline panel
        """
        if not steps:
            self.console.print("[dim]No steps captured[/dim]")
            return

        tree = Tree(f"[bold cyan]{title}[/bold cyan]")

        for i, step in enumerate(steps, 1):
            # Status indicator
            if step.success:
                status = "[green]✓[/green]"
                status_style = "green"
            else:
                status = "[red]✗[/red]"
                status_style = "red"

            # Step header with metrics
            latency_ms = step.metrics.latency
            cost = step.metrics.cost

            step_header = Text()
            step_header.append(f"Step {i}: ", style="bold")
            step_header.append(f"{step.tool_name} ", style=f"bold {status_style}")
            step_header.append(status)
            step_header.append(f"  [{latency_ms:.0f}ms", style="dim")
            step_header.append(f" | ${cost:.4f}]", style="dim")

            step_branch = tree.add(step_header)

            # Parameters
            if step.parameters:
                params_text = self._format_value(step.parameters, max_length=80)
                step_branch.add(f"[dim]→ params:[/dim] {params_text}")

            # Output
            if step.output is not None:
                output_text = self._format_value(step.output, max_length=80)
                step_branch.add(f"[dim]← output:[/dim] {output_text}")

            # Error if any
            if step.error:
                step_branch.add(f"[red]! error: {step.error}[/red]")

            # Token usage if available
            if step.metrics.tokens:
                tokens = step.metrics.tokens
                token_str = f"[dim]⚡ tokens: {tokens.total_tokens}"
                if tokens.cached_tokens > 0:
                    token_str += f" ({tokens.cached_tokens} cached)"
                token_str += "[/dim]"
                step_branch.add(token_str)

        self.console.print(tree)
        self.console.print()

    def print_step_table(self, steps: List[StepTrace]) -> None:
        """
        Print a compact table view of step metrics.

        Args:
            steps: List of step traces from execution
        """
        if not steps:
            return

        table = Table(title="Step-by-Step Metrics", show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Tool", style="cyan")
        table.add_column("Status", justify="center", width=6)
        table.add_column("Latency", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Tokens", justify="right")

        for i, step in enumerate(steps, 1):
            status = "[green]✓[/green]" if step.success else "[red]✗[/red]"
            tokens_str = "—"
            if step.metrics.tokens:
                tokens_str = f"{step.metrics.tokens.total_tokens:,}"

            table.add_row(
                str(i),
                step.tool_name,
                status,
                f"{step.metrics.latency:.0f}ms",
                f"${step.metrics.cost:.4f}",
                tokens_str,
            )

        self.console.print(table)
        self.console.print()

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
        table.add_column("Backend", style="magenta")
        table.add_column("Score", justify="right")
        table.add_column("Status")
        table.add_column("Cost", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Latency", justify="right")

        for result in results:
            status = "[green]✅ PASSED[/green]" if result.passed else "[red]❌ FAILED[/red]"
            score_color = (
                "green" if result.score >= 80 else "yellow" if result.score >= 60 else "red"
            )

            # Format token usage
            tokens_usage = result.trace.metrics.total_tokens
            if tokens_usage:
                tokens_str = f"{tokens_usage.total_tokens:,}"
                if tokens_usage.cached_tokens > 0:
                    tokens_str += f"\n({tokens_usage.cached_tokens:,} cached)"
            else:
                tokens_str = "N/A"

            # Get adapter name (capitalize for display)
            adapter_display = (result.adapter_name or "unknown").capitalize()

            table.add_row(
                result.test_case,
                adapter_display,
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

        # Show detailed results for all tests (verbose mode is default)
        for result in results:
            status_icon = "✅" if result.passed else "❌"
            status_color = "green" if result.passed else "red"

            self.console.print(
                f"\n[bold {status_color}]{status_icon} {result.test_case}[/bold {status_color}]"
            )

            # Show query
            if result.input_query:
                self.console.print(f"\n[bold]Query:[/bold]")
                # Wrap long queries
                query = result.input_query[:500] + "..." if len(result.input_query) > 500 else result.input_query
                self.console.print(f"  {query}")

            # Show agent response
            if result.actual_output:
                self.console.print(f"\n[bold]Response:[/bold]")
                # Truncate long responses
                output = result.actual_output[:800] + "..." if len(result.actual_output) > 800 else result.actual_output
                for line in output.split('\n'):
                    self.console.print(f"  {line}")

            # Show evaluation scores
            self.console.print(f"\n[bold]Evaluation Scores:[/bold]")

            tool_eval = result.evaluations.tool_accuracy
            output_eval = result.evaluations.output_quality
            seq_eval = result.evaluations.sequence_correctness

            # Tool accuracy
            tool_status = "✓" if tool_eval.accuracy == 1.0 else "✗"
            self.console.print(f"  Tool Accuracy:    {tool_eval.accuracy*100:.0f}% {tool_status}")

            # Output quality
            output_status = "✓" if output_eval.score >= 70 else "✗"
            self.console.print(f"  Output Quality:   {output_eval.score:.0f}/100 {output_status}")

            # Sequence correctness
            seq_status = "✓" if seq_eval.correct else "✗"
            self.console.print(f"  Sequence:         {'Correct' if seq_eval.correct else 'Incorrect'} {seq_status}")

            # Hallucination check
            if result.evaluations.hallucination:
                hall = result.evaluations.hallucination
                hall_status = "✓" if hall.passed else "✗"
                hall_result = "None detected" if not hall.has_hallucination else f"Detected ({hall.confidence:.0%} confidence)"
                self.console.print(f"  Hallucination:    {hall_result} {hall_status}")

            # Safety check
            if result.evaluations.safety:
                safety = result.evaluations.safety
                safety_status = "✓" if safety.passed else "✗"
                self.console.print(f"  Safety:           {safety.severity.capitalize()} {safety_status}")

            # Show threshold comparison
            min_score = result.min_score if result.min_score is not None else 75
            score_status = "✓" if result.score >= min_score else "✗"
            self.console.print(f"\n  [bold]Overall Score:    {result.score:.1f}/100 (min: {min_score}) {score_status}[/bold]")

            # Show failure reasons if failed
            if not result.passed:
                self.console.print(f"\n[bold red]Failure Reasons:[/bold red]")

                # Score below threshold
                if result.score < min_score:
                    self.console.print(f"[yellow]  • Score {result.score:.1f} < {min_score} (min_score)[/yellow]")

                # Tool issues
                if tool_eval.missing:
                    self.console.print(f"[yellow]  • Missing tools: {', '.join(tool_eval.missing)}[/yellow]")
                if tool_eval.unexpected:
                    self.console.print(f"[yellow]  • Unexpected tools: {', '.join(tool_eval.unexpected)}[/yellow]")
                for hint in tool_eval.hints:
                    self.console.print(f"[yellow]  💡 {hint}[/yellow]")

                # Sequence violations
                if not seq_eval.correct and seq_eval.violations:
                    for violation in seq_eval.violations:
                        self.console.print(f"[yellow]  • Sequence: {violation}[/yellow]")

                # Contains check failures
                if output_eval.contains_checks.failed:
                    self.console.print(f"[yellow]  • Missing required text: {', '.join(output_eval.contains_checks.failed)}[/yellow]")
                if output_eval.not_contains_checks.failed:
                    self.console.print(f"[yellow]  • Contains forbidden text: {', '.join(output_eval.not_contains_checks.failed)}[/yellow]")

                # Cost/latency issues
                if not result.evaluations.cost.passed:
                    cost = result.evaluations.cost
                    self.console.print(f"[yellow]  • Cost exceeded: ${cost.total_cost:.4f} > ${cost.threshold:.4f}[/yellow]")
                if not result.evaluations.latency.passed:
                    lat = result.evaluations.latency
                    self.console.print(f"[yellow]  • Latency exceeded: {lat.total_latency:.0f}ms > {lat.threshold:.0f}ms[/yellow]")

                # Hallucination issues
                if result.evaluations.hallucination and not result.evaluations.hallucination.passed:
                    hall = result.evaluations.hallucination
                    self.console.print(f"[yellow]  • Hallucination ({hall.confidence:.0%} confidence):[/yellow]")
                    details = hall.details[:300] + "..." if len(hall.details) > 300 else hall.details
                    self.console.print(f"[yellow]    {details}[/yellow]")

                # Safety issues
                if result.evaluations.safety and not result.evaluations.safety.passed:
                    safety = result.evaluations.safety
                    self.console.print(f"[yellow]  • Safety issue ({safety.severity}):[/yellow]")
                    if safety.categories_flagged:
                        self.console.print(f"[yellow]    Categories: {', '.join(safety.categories_flagged)}[/yellow]")

                # Output quality rationale
                if output_eval.rationale:
                    self.console.print(f"\n[dim]Output Quality Rationale:[/dim]")
                    rationale = output_eval.rationale[:400] + "..." if len(output_eval.rationale) > 400 else output_eval.rationale
                    self.console.print(f"[dim]  {rationale}[/dim]")

            # Show step-by-step flow
            if result.trace.steps:
                self.console.print()
                self.print_step_timeline(
                    result.trace.steps,
                    title=f"Execution Flow ({len(result.trace.steps)} steps)",
                )

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
            self.console.print("\n[bold]Query:[/bold]")
            self.console.print(f"  {result.input_query}")

        if result.actual_output:
            self.console.print("\n[bold]Response:[/bold]")
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
        # Show helpful hints
        for hint in tool_eval.hints:
            self.console.print(f"  [yellow]💡 {hint}[/yellow]")

        # Sequence correctness
        seq_eval = result.evaluations.sequence_correctness
        seq_status = "[green]✓ Correct[/green]" if seq_eval.correct else "[red]✗ Incorrect[/red]"
        self.console.print(f"\n[bold]Sequence:[/bold] {seq_status}")
        if seq_eval.expected_sequence:
            self.console.print(f"  Expected: {' → '.join(seq_eval.expected_sequence)}")
            self.console.print(f"  Actual:   {' → '.join(seq_eval.actual_sequence)}")
            if seq_eval.violations:
                for violation in seq_eval.violations:
                    self.console.print(f"  [yellow]⚠️  {violation}[/yellow]")

        # Output quality
        output_eval = result.evaluations.output_quality
        self.console.print(f"\n[bold]Output Quality:[/bold] {output_eval.score:.1f}/100")
        self.console.print(f"  Rationale: {output_eval.rationale}")

        # Costs and latency
        self.console.print("\n[bold]Performance:[/bold]")
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

        # Step-by-step execution flow
        if result.trace.steps:
            self.console.print()
            self.print_step_timeline(
                result.trace.steps, title=f"Execution Flow ({len(result.trace.steps)} steps)"
            )
            self.print_step_table(result.trace.steps)
