"""CoverageReporterMixin — behavior-coverage reporting for `ConsoleReporter`.

Inherits-into ConsoleReporter so the parent class stays focused on the core
single-run summary/detail/timeline output. Reads `self.console` (set by the
parent `__init__`) but is otherwise self-contained.
"""

from typing import List, Optional

from evalview.core.types import EvaluationResult, TestCase


class CoverageReporterMixin:
    """Console-rendering helpers for behavior-coverage reports."""

    def print_coverage_report(
        self,
        test_cases: List[TestCase],
        results: List[EvaluationResult],
        suite_name: Optional[str] = None,
    ) -> None:
        """Print a behavior coverage report.

        Shows coverage across:
        - Tasks: Scenarios tested
        - Tools: Agent tools exercised
        - Paths: Multi-step workflows
        - Eval dimensions: Correctness, safety, cost, latency checks

        Args:
            test_cases: List of test case definitions.
            results: List of evaluation results.
            suite_name: Optional name for the test suite.
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

