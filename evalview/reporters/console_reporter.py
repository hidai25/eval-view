"""Console reporter for evaluation results."""

import json
from typing import List, Any, Optional, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from evalview.core.types import EvaluationResult, StepTrace, TestCase


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

        # Overall stats with status indicator
        if failed == 0:
            status = "[green]● All Tests Passed[/green]"
            border = "green"
        else:
            status = "[bold red]● Some Tests Failed[/bold red]"
            border = "red"

        stats_panel = Panel(
            f"  {status}\n"
            f"\n"
            f"  [bold]✅ Passed:[/bold]      [green]{passed}[/green]\n"
            f"  [bold]❌ Failed:[/bold]      [red]{failed}[/red]\n"
            f"  [bold]📈 Success Rate:[/bold] [{'green' if success_rate >= 80 else 'yellow' if success_rate >= 50 else 'red'}]{success_rate:.1f}%[/{'green' if success_rate >= 80 else 'yellow' if success_rate >= 50 else 'red'}]",
            title="[bold]Overall Statistics[/bold]",
            border_style=border,
            padding=(0, 1),
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
                self.console.print("\n[bold]Query:[/bold]")
                # Wrap long queries
                query = result.input_query[:500] + "..." if len(result.input_query) > 500 else result.input_query
                self.console.print(f"  {query}")

            # Show agent response
            if result.actual_output:
                self.console.print("\n[bold]Response:[/bold]")
                # Truncate long responses
                output = result.actual_output[:800] + "..." if len(result.actual_output) > 800 else result.actual_output
                for line in output.split('\n'):
                    self.console.print(f"  {line}")

            # Show evaluation scores
            self.console.print("\n[bold]Evaluation Scores:[/bold]")

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
                self.console.print("\n[bold red]Failure Reasons:[/bold red]")

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
                    self.console.print("\n[dim]Output Quality Rationale:[/dim]")
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

    def print_compact_summary(
        self,
        results: List[EvaluationResult],
        suite_name: Optional[str] = None,
        previous_results: Optional[List[EvaluationResult]] = None,
    ) -> None:
        """
        Print a compact, screenshot-friendly summary of evaluation results.

        Args:
            results: List of evaluation results
            suite_name: Optional name for the test suite
            previous_results: Optional previous run results for delta comparison
        """
        if not results:
            self.console.print("[yellow]No results to display[/yellow]")
            return

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        # Header
        self.console.print()
        self.console.print("[bold]━━━ EvalView Summary ━━━[/bold]")

        # Suite name
        if suite_name:
            self.console.print(f"[dim]Suite:[/dim] {suite_name}")

        # Test counts
        passed_str = f"[green]{passed} passed[/green]"
        failed_str = f"[red]{failed} failed[/red]" if failed > 0 else f"{failed} failed"
        self.console.print(f"[dim]Tests:[/dim] {passed_str}, {failed_str}")

        # Failures section
        failed_results = [r for r in results if not r.passed]
        if failed_results:
            self.console.print()
            self.console.print("[bold]Failures:[/bold]")
            for result in failed_results:
                failure_reason = self._get_compact_failure_reason(result)
                self.console.print(f"  [red]✗[/red] {result.test_case:<30} [dim]{failure_reason}[/dim]")

        # Deltas vs last run
        if previous_results:
            deltas = self._compute_deltas(results, previous_results)
            if deltas:
                self.console.print()
                self.console.print("[bold]Deltas vs last run:[/bold]")

                # Tokens delta
                if deltas.get("tokens_delta") is not None:
                    tokens_pct = deltas["tokens_delta"]
                    arrow = "↑" if tokens_pct > 0 else "↓" if tokens_pct < 0 else "─"
                    color = "red" if tokens_pct > 10 else "green" if tokens_pct < -10 else "yellow"
                    sign = "+" if tokens_pct > 0 else ""
                    self.console.print(f"  [dim]Tokens:[/dim]  [{color}]{sign}{tokens_pct:.0f}%  {arrow}[/{color}]")

                # Latency delta
                if deltas.get("latency_delta") is not None:
                    latency_ms = deltas["latency_delta"]
                    arrow = "↑" if latency_ms > 0 else "↓" if latency_ms < 0 else "─"
                    color = "red" if latency_ms > 100 else "green" if latency_ms < -100 else "yellow"
                    sign = "+" if latency_ms > 0 else ""
                    self.console.print(f"  [dim]Latency:[/dim] [{color}]{sign}{latency_ms:.0f}ms  {arrow}[/{color}]")

                # Cost delta
                if deltas.get("cost_delta") is not None:
                    cost = deltas["cost_delta"]
                    arrow = "↑" if cost > 0 else "↓" if cost < 0 else "─"
                    color = "red" if cost > 0.05 else "green" if cost < -0.05 else "yellow"
                    sign = "+" if cost > 0 else ""
                    self.console.print(f"  [dim]Cost:[/dim]    [{color}]{sign}${abs(cost):.2f}  {arrow}[/{color}]")

        # Regressions warning
        if failed > 0:
            self.console.print()
            self.console.print("[bold yellow]⚠️  Regressions detected[/bold yellow]")
        else:
            self.console.print()
            self.console.print("[bold green]✓ All tests passed[/bold green]")

        self.console.print()

    def _get_compact_failure_reason(self, result: EvaluationResult) -> str:
        """Get a compact, one-line failure reason for display."""
        reasons = []

        # Check tool issues
        tool_eval = result.evaluations.tool_accuracy
        if tool_eval.missing:
            reasons.append(f"missing tool: {tool_eval.missing[0]}")
        elif tool_eval.unexpected:
            reasons.append(f"unexpected tool: {tool_eval.unexpected[0]}")

        # Check cost threshold
        if not result.evaluations.cost.passed:
            cost = result.evaluations.cost
            if cost.threshold and cost.threshold > 0:
                pct = ((cost.total_cost - cost.threshold) / cost.threshold) * 100
                reasons.append(f"cost +{pct:.0f}%")

        # Check latency threshold
        if not result.evaluations.latency.passed:
            lat = result.evaluations.latency
            if lat.threshold and lat.threshold > 0:
                pct = ((lat.total_latency - lat.threshold) / lat.threshold) * 100
                reasons.append(f"latency +{pct:.0f}%")

        # Check hallucination
        if result.evaluations.hallucination and not result.evaluations.hallucination.passed:
            reasons.append("hallucination detected")

        # Check safety
        if result.evaluations.safety and not result.evaluations.safety.passed:
            reasons.append(f"safety: {result.evaluations.safety.severity}")

        # Check score
        min_score = result.min_score if result.min_score is not None else 75
        if result.score < min_score and not reasons:
            reasons.append(f"score {result.score:.0f} < {min_score}")

        return reasons[0] if reasons else "below threshold"

    def _compute_deltas(
        self,
        current: List[EvaluationResult],
        previous: List[EvaluationResult],
    ) -> Dict[str, float]:
        """Compute deltas between current and previous run."""
        deltas = {}

        # Compute totals for current run
        current_tokens = sum(
            r.trace.metrics.total_tokens.total_tokens
            for r in current
            if r.trace.metrics.total_tokens
        )
        current_latency = sum(r.trace.metrics.total_latency for r in current)
        current_cost = sum(r.trace.metrics.total_cost for r in current)

        # Compute totals for previous run
        prev_tokens = sum(
            r.trace.metrics.total_tokens.total_tokens
            for r in previous
            if r.trace.metrics.total_tokens
        )
        prev_latency = sum(r.trace.metrics.total_latency for r in previous)
        prev_cost = sum(r.trace.metrics.total_cost for r in previous)

        # Calculate deltas
        if prev_tokens > 0:
            deltas["tokens_delta"] = ((current_tokens - prev_tokens) / prev_tokens) * 100
        if prev_latency > 0:
            deltas["latency_delta"] = current_latency - prev_latency
        if prev_cost > 0:
            deltas["cost_delta"] = current_cost - prev_cost

        return deltas

    def print_coverage_report(
        self,
        test_cases: List[TestCase],
        results: List[EvaluationResult],
        suite_name: Optional[str] = None,
    ) -> None:
        """
        Print a behavior coverage report.

        Shows coverage across:
        - Tasks: scenarios tested
        - Tools: agent tools exercised
        - Paths: multi-step workflows
        - Eval dimensions: correctness, safety, cost, latency checks

        Args:
            test_cases: List of test case definitions
            results: List of evaluation results
            suite_name: Optional name for the test suite
        """
        if not test_cases:
            self.console.print("[yellow]No test cases to analyze[/yellow]")
            return

        # Header
        self.console.print()
        self.console.print("[bold]━━━ Behavior Coverage ━━━[/bold]")

        if suite_name:
            self.console.print(f"[dim]Suite:[/dim] {suite_name}")
        self.console.print()

        # 1. Tasks Coverage
        total_tasks = len(test_cases)
        executed_tasks = len(results)
        passed_tasks = sum(1 for r in results if r.passed)
        task_pct = (executed_tasks / total_tasks * 100) if total_tasks > 0 else 0

        task_color = "green" if task_pct == 100 else "yellow" if task_pct >= 50 else "red"
        self.console.print(f"[bold]Tasks:[/bold]      [{task_color}]{executed_tasks}/{total_tasks} scenarios ({task_pct:.0f}%)[/{task_color}]")
        if passed_tasks < executed_tasks:
            self.console.print(f"            [dim]({passed_tasks} passing, {executed_tasks - passed_tasks} failing)[/dim]")

        # 2. Tools Coverage
        # Collect all expected tools from test cases
        expected_tools = set()
        for tc in test_cases:
            if tc.expected.tools:
                expected_tools.update(tc.expected.tools)
            if tc.expected.tool_sequence:
                expected_tools.update(tc.expected.tool_sequence)
            if tc.expected.sequence:
                expected_tools.update(tc.expected.sequence)

        # Collect all actually called tools from results
        exercised_tools = set()
        for result in results:
            if result.trace.steps:
                for step in result.trace.steps:
                    if step.tool_name:
                        exercised_tools.add(step.tool_name)

        # Also add tools from evaluations
        for result in results:
            if result.evaluations.tool_accuracy.correct:
                exercised_tools.update(result.evaluations.tool_accuracy.correct)

        # Calculate tool coverage
        if expected_tools:
            tools_covered = expected_tools & exercised_tools
            tool_pct = (len(tools_covered) / len(expected_tools) * 100) if expected_tools else 0
            tool_color = "green" if tool_pct == 100 else "yellow" if tool_pct >= 50 else "red"
            self.console.print(f"[bold]Tools:[/bold]      [{tool_color}]{len(tools_covered)}/{len(expected_tools)} exercised ({tool_pct:.0f}%)[/{tool_color}]")

            # Show missing tools
            missing_tools = expected_tools - exercised_tools
            if missing_tools:
                self.console.print(f"            [dim]missing: {', '.join(sorted(missing_tools))}[/dim]")
        else:
            self.console.print("[bold]Tools:[/bold]      [dim]no tool expectations defined[/dim]")

        # 3. Paths Coverage (multi-step workflows)
        # Count tests with sequence requirements
        sequence_tests = [tc for tc in test_cases if tc.expected.tool_sequence or tc.expected.sequence]
        total_paths = len(sequence_tests)

        if total_paths > 0:
            # Check which sequence tests passed
            sequence_test_names = {tc.name for tc in sequence_tests}
            sequence_results = [r for r in results if r.test_case in sequence_test_names]
            paths_passed = sum(1 for r in sequence_results if r.evaluations.sequence_correctness.correct)

            path_pct = (paths_passed / total_paths * 100) if total_paths > 0 else 0
            path_color = "green" if path_pct == 100 else "yellow" if path_pct >= 50 else "red"
            self.console.print(f"[bold]Paths:[/bold]      [{path_color}]{paths_passed}/{total_paths} multi-step workflows ({path_pct:.0f}%)[/{path_color}]")
        else:
            self.console.print("[bold]Paths:[/bold]      [dim]no sequence tests defined[/dim]")

        # 4. Eval Dimensions
        self.console.print("[bold]Dimensions:[/bold]")

        # Check which dimensions are being tested
        has_tool_check = any(tc.expected.tools or tc.expected.tool_sequence for tc in test_cases)
        has_output_check = any(tc.expected.output for tc in test_cases)
        has_cost_check = any(tc.thresholds.max_cost is not None for tc in test_cases)
        has_latency_check = any(tc.thresholds.max_latency is not None for tc in test_cases)
        has_hallucination_check = any(
            tc.expected.hallucination is not None or (tc.checks is None or tc.checks.hallucination)
            for tc in test_cases
        )
        has_safety_check = any(
            tc.expected.safety is not None or (tc.checks is None or tc.checks.safety)
            for tc in test_cases
        )

        dimensions = []
        if has_tool_check:
            # Check if tool checks pass
            tool_pass = all(r.evaluations.tool_accuracy.accuracy == 1.0 for r in results) if results else False
            dimensions.append(("correctness", tool_pass))
        if has_output_check:
            output_pass = all(r.evaluations.output_quality.score >= 70 for r in results) if results else False
            dimensions.append(("output", output_pass))
        if has_cost_check:
            cost_pass = all(r.evaluations.cost.passed for r in results) if results else False
            dimensions.append(("cost", cost_pass))
        if has_latency_check:
            latency_pass = all(r.evaluations.latency.passed for r in results) if results else False
            dimensions.append(("latency", latency_pass))
        if has_hallucination_check:
            hall_pass = all(
                r.evaluations.hallucination is None or r.evaluations.hallucination.passed
                for r in results
            ) if results else False
            dimensions.append(("hallucination", hall_pass))
        if has_safety_check:
            safety_pass = all(
                r.evaluations.safety is None or r.evaluations.safety.passed
                for r in results
            ) if results else False
            dimensions.append(("safety", safety_pass))

        if dimensions:
            dim_strs = []
            for name, passed in dimensions:
                icon = "[green]✓[/green]" if passed else "[red]✗[/red]"
                dim_strs.append(f"{name} {icon}")
            self.console.print(f"            {', '.join(dim_strs)}")
        else:
            self.console.print("            [dim]no thresholds configured[/dim]")

        # Overall coverage score
        self.console.print()
        coverage_scores = []
        if total_tasks > 0:
            coverage_scores.append(task_pct)
        if expected_tools:
            coverage_scores.append(tool_pct)
        if total_paths > 0:
            coverage_scores.append(path_pct)

        if coverage_scores:
            overall_coverage = sum(coverage_scores) / len(coverage_scores)
            cov_color = "green" if overall_coverage >= 80 else "yellow" if overall_coverage >= 50 else "red"
            self.console.print(f"[bold]Overall:[/bold]    [{cov_color}]{overall_coverage:.0f}% behavior coverage[/{cov_color}]")

        self.console.print()
