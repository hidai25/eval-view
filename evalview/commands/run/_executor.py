"""Single-test executor for the run command.

`execute_single_test` handles:
- Multi-turn conversation execution
- Statistical / pass@k mode (multiple runs)
- Standard single-run execution
- Retry logic with exponential back-off
- Debug trace output
- Regression tracking integration
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from evalview.evaluators.evaluator import Evaluator
    from evalview.core.retry import RetryConfig

logger = logging.getLogger(__name__)


@dataclass
class ExecutorOptions:
    """All shared state needed to execute a single test case."""

    evaluator: "Evaluator"
    retry_config: "RetryConfig"
    global_adapter: Optional[Any]
    model_config: Any
    allow_private_urls: bool
    config: Dict[str, Any]
    verbose: bool
    debug: bool
    track: bool
    compare_baseline: bool
    tracker: Optional[Any]
    regression_reports: Dict[str, Any]
    trace_reporter: Optional[Any]
    statistical_evaluator: Any
    stats_reporter: Any
    no_judge: bool = False


async def execute_single_test(
    test_case: Any,
    options: ExecutorOptions,
    console: Any,
) -> Tuple[bool, Any]:
    """Execute one test case and return ``(passed, EvaluationResult)``.

    Handles multi-turn, statistical, and standard single-run modes transparently.
    Retry and debug output are applied in single-run mode only (statistical mode
    manages its own retry loop).
    """
    import json as json_module
    from evalview.commands.shared import _execute_multi_turn_trace
    from evalview.commands.run._adapters import get_test_adapter
    from evalview.core.retry import with_retry
    from evalview.evaluators.statistical_evaluator import is_statistical_mode

    test_adapter = get_test_adapter(
        test_case,
        options.global_adapter,
        options.model_config,
        options.allow_private_urls,
        options.verbose,
        console,
    )

    context: Dict[str, Any] = dict(test_case.input.context) if test_case.input.context else {}
    if hasattr(test_case, "tools") and test_case.tools:
        context["tools"] = test_case.tools

    async def _execute() -> Any:
        return await test_adapter.execute(test_case.input.query, context)

    # ── Multi-turn ────────────────────────────────────────────────────────────
    if test_case.is_multi_turn:
        if options.verbose:
            console.print(f"[dim]  ↳ multi-turn ({len(test_case.turns)} turns)[/dim]")
        trace = await _execute_multi_turn_trace(test_case, test_adapter)
        adapter_name = getattr(test_adapter, "name", None)
        result = await options.evaluator.evaluate(test_case, trace, adapter_name=adapter_name)
        _track(result, test_case, options)
        return result.passed, result

    # ── Statistical / pass@k ──────────────────────────────────────────────────
    if is_statistical_mode(test_case):
        return await _execute_statistical(test_case, options, _execute, test_adapter, console)

    # ── Standard single-run ───────────────────────────────────────────────────
    retry_cfg = options.retry_config
    if retry_cfg.max_retries > 0:
        retry_result = await with_retry(
            _execute,
            retry_cfg,
            on_retry=(
                lambda attempt, delay, exc: console.print(
                    f"[yellow]  ↻ Retry {attempt}/{retry_cfg.max_retries} for "
                    f"{test_case.name} after {delay:.1f}s ({type(exc).__name__})[/yellow]"
                )
                if options.verbose
                else None
            ),
        )
        if not retry_result.success:
            exc = retry_result.exception
            raise exc if exc is not None else RuntimeError("Test execution failed")
        trace = retry_result.result
    else:
        trace = await _execute()

    if options.trace_reporter:
        options.trace_reporter.report_from_execution_trace(trace, test_case.name)

    if options.debug:
        _print_debug_trace(test_case, test_adapter, trace, json_module, console)

    adapter_name = getattr(test_adapter, "name", None)
    result = await options.evaluator.evaluate(test_case, trace, adapter_name=adapter_name)
    _track(result, test_case, options)
    return result.passed, result


# ── Helpers ───────────────────────────────────────────────────────────────────


def _track(result: Any, test_case: Any, options: ExecutorOptions) -> None:
    """Store result in regression tracker when tracking is enabled."""
    if options.tracker:
        if options.track:
            options.tracker.store_result(result)
        if options.compare_baseline:
            options.regression_reports[test_case.name] = options.tracker.compare_to_baseline(result)


async def _execute_statistical(
    test_case: Any,
    options: ExecutorOptions,
    execute_fn: Any,
    test_adapter: Any,
    console: Any,
) -> Tuple[bool, Any]:
    """Run the test N times and return a statistical aggregate result."""
    from evalview.core.retry import with_retry

    variance_config = test_case.thresholds.variance
    num_runs = variance_config.runs
    console.print(f"\n[cyan]📊 Statistical mode: Running {test_case.name} {num_runs} times...[/cyan]")

    individual_results: List[Any] = []
    for run_idx in range(num_runs):
        try:
            if options.retry_config.max_retries > 0:
                retry_result = await with_retry(
                    execute_fn,
                    options.retry_config,
                    on_retry=lambda attempt, delay, exc: None,
                )
                if not retry_result.success:
                    console.print(f"  [red]Run {run_idx + 1}/{num_runs}: ERROR[/red]")
                    continue
                trace = retry_result.result
            else:
                trace = await execute_fn()

            adapter_name = getattr(test_adapter, "name", None)
            result = await options.evaluator.evaluate(test_case, trace, adapter_name=adapter_name)
            individual_results.append(result)

            status = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            console.print(f"  Run {run_idx + 1}/{num_runs}: {status} score={result.score:.1f}")

        except Exception as exc:
            console.print(f"  [red]Run {run_idx + 1}/{num_runs}: ERROR - {str(exc)[:50]}[/red]")

    if not individual_results:
        raise ValueError(f"All {num_runs} runs failed for {test_case.name}")

    stat_result = options.statistical_evaluator.evaluate_from_results(
        test_case, individual_results, variance_config
    )
    options.stats_reporter.print_statistical_summary(stat_result, show_individual_runs=options.verbose)

    best_result = individual_results[0]
    best_result.passed = stat_result.passed
    best_result.score = stat_result.score_stats.mean
    return stat_result.passed, best_result


def _print_debug_trace(
    test_case: Any,
    test_adapter: Any,
    trace: Any,
    json_module: Any,
    console: Any,
) -> None:
    """Print detailed debug information for a single test execution."""
    console.print(f"\n[cyan]{'─' * 60}[/cyan]")
    console.print(f"[cyan]DEBUG: {test_case.name}[/cyan]")
    console.print(f"[cyan]{'─' * 60}[/cyan]\n")

    if hasattr(test_adapter, "_last_raw_response") and test_adapter._last_raw_response:
        console.print("[bold]Raw API Response:[/bold]")
        try:
            raw_json = json_module.dumps(test_adapter._last_raw_response, indent=2, default=str)[:2000]
            console.print(f"[dim]{raw_json}[/dim]")
            if len(json_module.dumps(test_adapter._last_raw_response, default=str)) > 2000:
                console.print("[dim]... (truncated)[/dim]")
        except Exception:
            console.print(f"[dim]{str(test_adapter._last_raw_response)[:500]}[/dim]")
        console.print()

    console.print("[bold]Parsed ExecutionTrace:[/bold]")
    console.print(f"  Session ID: {trace.session_id}")
    console.print(f"  Duration: {trace.start_time} → {trace.end_time}")
    console.print(f"  Steps: {len(trace.steps)}")
    for i, step in enumerate(trace.steps):
        console.print(f"    [{i + 1}] {step.tool_name}")
        console.print(f"        params: {str(step.parameters)[:100]}")
        console.print(
            f"        metrics: latency={step.metrics.latency:.1f}ms, cost=${step.metrics.cost:.4f}"
        )
        if step.metrics.tokens:
            console.print(
                f"        tokens: in={step.metrics.tokens.input_tokens}, "
                f"out={step.metrics.tokens.output_tokens}"
            )
    console.print(
        f"  Final Output: {trace.final_output[:200]}"
        f"{'...' if len(trace.final_output) > 200 else ''}"
    )
    console.print()
    console.print("[bold]Aggregated Metrics:[/bold]")
    console.print(f"  Total Cost: ${trace.metrics.total_cost:.4f}")
    console.print(f"  Total Latency: {trace.metrics.total_latency:.0f}ms")
    if trace.metrics.total_tokens:
        console.print(
            f"  Total Tokens: in={trace.metrics.total_tokens.input_tokens}, "
            f"out={trace.metrics.total_tokens.output_tokens}, "
            f"cached={trace.metrics.total_tokens.cached_tokens}"
        )
    console.print()
