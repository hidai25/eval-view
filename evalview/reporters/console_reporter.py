"""Console reporter for evaluation results."""

import json
from typing import List, Any, Optional, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from evalview.core.types import (
    EvaluationResult,
    StepTrace,
    ReasonCode,
)
from evalview.reporters._console_coverage import CoverageReporterMixin
from evalview.reporters._console_statistical import StatisticalReporterMixin


class ConsoleReporter(StatisticalReporterMixin, CoverageReporterMixin):
    """Generates formatted console output for evaluation results."""

    def __init__(self):
        self.console = Console()

    def _format_value(self, value: Any, max_length: int = 60) -> str:
        """Format a value for display, truncating at max_length."""
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

    def _display_reason_codes(self, reason_codes: List[ReasonCode], indent: str = "  ") -> None:
        """Display structured reason codes with icons and formatting.

        Args:
            reason_codes: List of ReasonCode objects to display
            indent: Indentation prefix for each line
        """
        if not reason_codes:
            return

        self.console.print(f"\n{indent}[bold]Failure Reasons:[/bold]")
        for rc in reason_codes:
            # Icon based on severity
            if rc.severity == "error":
                icon = "[red]✗[/red]"
                color = "red"
            elif rc.severity == "warning":
                icon = "[yellow]⚠[/yellow]"
                color = "yellow"
            else:
                icon = "[blue]ℹ[/blue]"
                color = "blue"

            # Display code and message
            self.console.print(f"{indent}{icon} [{color}]{rc.code}:[/{color}] {rc.message}")

            # Display context if present (limit detail)
            if rc.context:
                # Only show key context items, not full dumps
                if "expected_tool" in rc.context:
                    self.console.print(f"{indent}  [dim]Expected: {rc.context['expected_tool']}[/dim]")
                if "actual_tool" in rc.context:
                    self.console.print(f"{indent}  [dim]Actual: {rc.context['actual_tool']}[/dim]")
                if "expected" in rc.context and "actual" in rc.context and isinstance(rc.context["expected"], str):
                    self.console.print(f"{indent}  [dim]Expected: {rc.context['expected']}[/dim]")
                    self.console.print(f"{indent}  [dim]Actual: {rc.context['actual']}[/dim]")

            # Display remediation with helpful icon
            if rc.remediation:
                self.console.print(f"{indent}  [cyan]→ Fix:[/cyan] [dim]{rc.remediation}[/dim]")

    def print_step_timeline(self, steps: List[StepTrace], title: str = "Agent Flow") -> None:
        """Print a visual step-by-step timeline of agent execution.

        Args:
            steps: List of step traces from execution.
            title: Title for the timeline panel.
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
        """Print a compact table view of step metrics.

        Args:
            steps: List of step traces from execution.
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
        """Print a summary of evaluation results.

        Args:
            results: List of evaluation results.
        """
        if not results:
            self.console.print("[yellow]No results to display[/yellow]")
            return

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        success_rate = (passed / len(results)) * 100 if results else 0

        # Check if any results have suite_type set
        has_suite_types = any(r.suite_type for r in results)
        # Check if any results have difficulty set
        has_difficulty = any(r.difficulty for r in results)

        # Summary table
        table = Table(title="📊 Evaluation Summary", show_header=True)
        table.add_column("Test Case", style="cyan")
        if has_suite_types:
            table.add_column("Type", style="dim", width=10)
        if has_difficulty:
            table.add_column("Difficulty", style="dim", width=8)
        table.add_column("Backend", style="magenta")
        table.add_column("Score", justify="right")
        table.add_column("Status")
        table.add_column("Cost", justify="right")
        table.add_column("Latency", justify="right")

        for result in results:
            # For capability tests, failures are expected (hill climbing)
            # For regression tests, failures are critical (safety net)
            if result.suite_type == "capability":
                status = "[green]✅ PASSED[/green]" if result.passed else "[yellow]⚡ CLIMBING[/yellow]"
            elif result.suite_type == "regression":
                status = "[green]✅ PASSED[/green]" if result.passed else "[red]🚨 REGRESSION[/red]"
            else:
                status = "[green]✅ PASSED[/green]" if result.passed else "[red]❌ FAILED[/red]"

            score_color = (
                "green" if result.score >= 80 else "yellow" if result.score >= 60 else "red"
            )

            # Get adapter name (capitalize for display)
            adapter_display = (result.adapter_name or "unknown").capitalize()

            # Suite type display
            suite_display = ""
            if result.suite_type == "capability":
                suite_display = "[blue]capability[/blue]"
            elif result.suite_type == "regression":
                suite_display = "[magenta]regression[/magenta]"

            # Difficulty display with color coding
            difficulty_display = ""
            if result.difficulty:
                difficulty_colors = {
                    "trivial": "dim",
                    "easy": "green",
                    "medium": "yellow",
                    "hard": "red",
                    "expert": "bold red",
                }
                color = difficulty_colors.get(result.difficulty, "white")
                difficulty_display = f"[{color}]{result.difficulty}[/{color}]"

            row = [result.test_case]
            if has_suite_types:
                row.append(suite_display)
            if has_difficulty:
                row.append(difficulty_display)
            _free_adapters = {"opencode", "goose", "ollama"}
            _is_local = (result.adapter_name or "").lower() in _free_adapters
            _cost = result.trace.metrics.total_cost
            cost_display = "[dim]free[/dim]" if _is_local and _cost == 0.0 else f"${_cost:.4f}"
            row.extend([
                adapter_display,
                f"[{score_color}]{result.score:.1f}[/{score_color}]",
                status,
                cost_display,
                f"{result.trace.metrics.total_latency:.0f}ms",
            ])

            table.add_row(*row)

        self.console.print(table)

        # Show score breakdown for tests that aren't a clean pass,
        # so users understand exactly what pulled the score down.
        low_score_notes = []
        for result in results:
            if result.score < 80:
                evals = result.evaluations
                output_score = evals.output_quality.score if evals.output_quality else None
                tool_accuracy = evals.tool_accuracy.accuracy if evals.tool_accuracy else None
                seq_ok = getattr(evals.sequence_correctness, "correct", None) if evals.sequence_correctness else None
                parts = []
                if tool_accuracy is not None:
                    color = "green" if tool_accuracy >= 0.8 else ("yellow" if tool_accuracy >= 0.5 else "red")
                    parts.append(f"tools [{color}]{tool_accuracy*100:.0f}%[/{color}]")
                if output_score is not None:
                    color = "green" if output_score >= 80 else ("yellow" if output_score >= 50 else "red")
                    parts.append(f"output [{color}]{output_score:.0f}/100[/{color}]")
                if seq_ok is not None:
                    parts.append(f"sequence [{'green' if seq_ok else 'red'}]{'✓' if seq_ok else '✗'}[/{'green' if seq_ok else 'red'}]")
                if parts:
                    low_score_notes.append(
                        f"  {result.test_case}: [bold]{result.score:.0f}[/bold] = {' · '.join(parts)}"
                    )

        if low_score_notes:
            self.console.print()
            self.console.print("[dim]Score breakdown:[/dim]")
            for note in low_score_notes:
                self.console.print(note)

        self.console.print()

        # Calculate suite-type breakdowns
        capability_results = [r for r in results if r.suite_type == "capability"]
        regression_results = [r for r in results if r.suite_type == "regression"]
        other_results = [r for r in results if r.suite_type not in ("capability", "regression")]

        capability_passed = sum(1 for r in capability_results if r.passed)
        regression_passed = sum(1 for r in regression_results if r.passed)
        regression_failed = len(regression_results) - regression_passed

        # Overall stats with status indicator
        # Regression failures are critical; capability failures are expected
        if regression_failed > 0:
            status = "[bold red]🚨 Regression Failures Detected[/bold red]"
            border = "red"
        elif failed == 0:
            status = "[green]● All Tests Passed[/green]"
            border = "green"
        else:
            status = "[yellow]● Capability Tests Still Climbing[/yellow]"
            border = "yellow"

        stats_content = f"  {status}\n\n"
        stats_content += f"  [bold]✅ Passed:[/bold]      [green]{passed}[/green]\n"
        stats_content += f"  [bold]❌ Failed:[/bold]      [red]{failed}[/red]\n"
        stats_content += f"  [bold]📈 Success Rate:[/bold] [{'green' if success_rate >= 80 else 'yellow' if success_rate >= 50 else 'red'}]{success_rate:.1f}%[/{'green' if success_rate >= 80 else 'yellow' if success_rate >= 50 else 'red'}]"

        # Add suite type breakdown if applicable
        if has_suite_types:
            stats_content += "\n\n  [bold]By Suite Type:[/bold]"
            if regression_results:
                reg_rate = (regression_passed / len(regression_results) * 100) if regression_results else 0
                reg_color = "green" if reg_rate == 100 else "red"
                stats_content += f"\n  [magenta]Regression:[/magenta]  [{reg_color}]{regression_passed}/{len(regression_results)}[/{reg_color}]"
                if regression_failed > 0:
                    stats_content += f" [red](⚠️  {regression_failed} regressions!)[/red]"
            if capability_results:
                cap_rate = (capability_passed / len(capability_results) * 100) if capability_results else 0
                cap_color = "green" if cap_rate >= 80 else "yellow" if cap_rate >= 50 else "dim"
                stats_content += f"\n  [blue]Capability:[/blue]   [{cap_color}]{capability_passed}/{len(capability_results)}[/{cap_color}] [dim](hill climbing)[/dim]"
            if other_results:
                other_passed = sum(1 for r in other_results if r.passed)
                stats_content += f"\n  [dim]Other:[/dim]        {other_passed}/{len(other_results)}"

        # Add difficulty breakdown if applicable
        if has_difficulty:
            stats_content += "\n\n  [bold]By Difficulty:[/bold]"
            difficulty_levels = ["trivial", "easy", "medium", "hard", "expert"]
            difficulty_colors = {
                "trivial": "dim",
                "easy": "green",
                "medium": "yellow",
                "hard": "red",
                "expert": "bold red",
            }
            for level in difficulty_levels:
                level_results = [r for r in results if r.difficulty == level]
                if level_results:
                    level_passed = sum(1 for r in level_results if r.passed)
                    level_rate = (level_passed / len(level_results) * 100)
                    rate_color = "green" if level_rate >= 80 else "yellow" if level_rate >= 50 else "red"
                    color = difficulty_colors[level]
                    stats_content += f"\n  [{color}]{level.capitalize():8}[/{color}] [{rate_color}]{level_passed}/{len(level_results)}[/{rate_color}] ({level_rate:.0f}%)"

        stats_panel = Panel(
            stats_content,
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

            # Sequence correctness with progress score
            seq_status = "✓" if seq_eval.correct else "✗"
            if seq_eval.correct:
                self.console.print(f"  Sequence:         Correct {seq_status}")
            else:
                progress_pct = seq_eval.progress_score * 100
                self.console.print(f"  Sequence:         {progress_pct:.0f}% complete {seq_status}")

            # Hallucination check
            if result.evaluations.hallucination:
                hall = result.evaluations.hallucination
                hall_status = "✓" if hall.passed else "✗"
                if hall.has_hallucination:
                    hall_result = f"Detected ({hall.confidence:.0%} confidence)"
                else:
                    details = hall.details or ""
                    if "Faithfulness:" in details:
                        faith_line = details.split("\n")[0].replace("Faithfulness: ", "")
                        hall_result = f"Faithfulness {faith_line}"
                    elif "no verifiable" in details.lower():
                        hall_result = "No factual claims to verify"
                    elif "unavailable" in details.lower():
                        hall_result = "Check unavailable (LLM error)"
                    else:
                        hall_result = "None detected"
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

            # Forbidden tool violations (always show when present, pass or fail)
            if result.evaluations.forbidden_tools:
                forbidden_eval = result.evaluations.forbidden_tools
                if not forbidden_eval.passed:
                    self.console.print(
                        "\n[bold red on default]  FORBIDDEN TOOL VIOLATION  [/bold red on default]"
                    )
                    for violation in forbidden_eval.violations:
                        self.console.print(f"  [red]✗[/red] [bold red]{violation}[/bold red] was called but is declared forbidden")
                    self.console.print(
                        "  [dim]This test hard-fails regardless of output quality.[/dim]"
                    )

            # Show failure reasons if failed
            if not result.passed:
                self.console.print("\n[bold red]Failure Reasons:[/bold red]")

                # Forbidden tool violation (already shown above, brief reminder)
                if result.evaluations.forbidden_tools and not result.evaluations.forbidden_tools.passed:
                    violations = result.evaluations.forbidden_tools.violations
                    self.console.print(f"[red]  • Forbidden tools called: {', '.join(violations)}[/red]")

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

    @staticmethod
    def _test_quality_hints(result: "EvaluationResult") -> List[str]:
        """Return suggestions when the test itself looks like the problem, not the agent.

        Args:
            result: Evaluation result to analyze.

        Returns:
            A list of human-readable suggestions for improving the test case.
        """
        hints: List[str] = []
        query = (result.input_query or "").strip()
        output_eval = result.evaluations.output_quality
        tool_eval = result.evaluations.tool_calls

        # 1. Truncated / incomplete query
        _FRAGMENT_ENDINGS = (
            " for", " the", " a", " an", " of", " in", " on",
            " to", " with", " and", " or",
        )
        if len(query) < 15 or len(query.split()) < 3:
            hints.append(
                "Query is very short — add a specific object or intent "
                "(e.g. \"Show me LangSmith pain points\" not \"Search for\")"
            )
        elif query.lower().endswith(_FRAGMENT_ENDINGS):
            hints.append(
                f"Query looks truncated (ends with \"{query.split()[-1]}\") — "
                "complete the search term in your test YAML"
            )

        # 2. Low output quality but agent called the right tools — test expectations are stale
        tool_score = tool_eval.score if tool_eval else 0
        output_score = output_eval.score if output_eval else 100
        if tool_score >= 80 and output_score < 50:
            hints.append(
                "Agent used the right tools but output score is low — "
                "your expected.output.contains strings may be stale or too specific. "
                "Run evalview snapshot to update the baseline."
            )

        # 3. Empty string in contains — always passes, hides real failures
        if output_eval and output_eval.contains_checks:
            checked = getattr(output_eval.contains_checks, "checked", [])
            if any(s == "" for s in checked):
                hints.append(
                    "expected.output.contains has an empty string (\"\") — "
                    "it always passes. Replace with a real phrase your agent outputs."
                )

        return hints

    def print_detailed(self, result: EvaluationResult) -> None:
        """Print a detailed evaluation result.

        Args:
            result: Evaluation result to display.
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

        # Display structured reason codes (enhanced error feedback)
        if hasattr(tool_eval, 'reason_codes') and tool_eval.reason_codes:
            self._display_reason_codes(tool_eval.reason_codes)

        # Sequence correctness with progress score
        seq_eval = result.evaluations.sequence_correctness
        if seq_eval.correct:
            seq_status = "[green]✓ Correct[/green]"
        else:
            progress_pct = seq_eval.progress_score * 100
            seq_status = f"[red]✗ {progress_pct:.0f}% complete[/red]"
        self.console.print(f"\n[bold]Sequence:[/bold] {seq_status}")
        if seq_eval.expected_sequence:
            self.console.print(f"  Expected: {' → '.join(seq_eval.expected_sequence)}")
            self.console.print(f"  Actual:   {' → '.join(seq_eval.actual_sequence)}")
            if seq_eval.violations:
                for violation in seq_eval.violations:
                    self.console.print(f"  [yellow]⚠️  {violation}[/yellow]")

            # Display structured reason codes for sequence violations
            if hasattr(seq_eval, 'reason_codes') and seq_eval.reason_codes:
                self._display_reason_codes(seq_eval.reason_codes)

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
        """Print a compact, screenshot-friendly summary of evaluation results.

        Args:
            results: List of evaluation results.
            suite_name: Optional name for the test suite.
            previous_results: Optional previous run results for delta comparison.
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
        """Get a compact, one-line failure reason for display.

        Args:
            result: Evaluation result to extract a primary failure reason from.

        Returns:
            A single-line string describing the most relevant failure reason.
        """
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
        """Compute aggregate deltas between current and previous runs.

        Args:
            current: Current run results.
            previous: Previous run results.

        Returns:
            Mapping of delta metric names to values. Keys may include:
            - "tokens_delta": Percentage change in total tokens (if previous tokens > 0)
            - "latency_delta": Absolute change in total latency in milliseconds
            - "cost_delta": Absolute change in total cost in USD
        """
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

