"""Run command — execute test cases against the agent with full evaluation.

This module is a thin orchestrator. Each phase of the run is delegated to a
focused sub-module:

  _adapters.py   — adapter factory (build_adapter / get_test_adapter)
  _executor.py   — single-test execution (multi-turn, statistical, retry, debug)
  _runner.py     — sequential and parallel test runners with live display
  _reporter.py   — diffs, summaries, HTML reports, exit codes
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from evalview.commands.shared import console, _execute_multi_turn_trace
from evalview.core.llm_provider import get_or_select_provider, save_provider_preference, PROVIDER_CONFIGS
from evalview.core.loader import TestCaseLoader
from evalview.evaluators.evaluator import Evaluator
from evalview.skills.ui_utils import print_evalview_banner
from evalview.telemetry.decorators import track_command, track_run_command

logger = logging.getLogger(__name__)


# ── No-agent guide ─────────────────────────────────────────────────────────────


def _display_no_agent_guide(endpoint: Optional[str] = None) -> None:
    """Minimal prompt shown when no agent is reachable."""
    console.print()
    if endpoint:
        console.print(f"  [yellow]No agent at[/yellow] [bold]{endpoint}[/bold] [yellow]— is it running?[/yellow]")
        console.print()
        console.print("  Start your agent server, then re-run [cyan]evalview run[/cyan].")
    else:
        console.print("  [yellow]No agent configured.[/yellow]")
        console.print()
        console.print("  Point EvalView at your agent in [cyan].evalview/config.yaml[/cyan]:")
        console.print()
        console.print("    [dim]adapter: http[/dim]")
        console.print("    [dim]endpoint: http://localhost:8080/execute[/dim]")
    console.print()
    console.print("  [dim]Need a running agent?[/dim]")
    console.print("  [dim]HTTP:      https://github.com/hidai25/eval-view/blob/main/demo-agent/agent.py[/dim]")
    console.print("  [dim]LangGraph: https://github.com/hidai25/eval-view/tree/main/examples/langgraph[/dim]")
    console.print()
    console.print("  Or see EvalView catch a real regression right now:")
    console.print("  [bold cyan]→ evalview demo[/bold cyan]   [dim](no setup, 30 seconds)[/dim]")
    console.print()


# ── Click command ──────────────────────────────────────────────────────────────


@click.command("run")
@click.argument("path", required=False, default=None)
@click.option("--pattern", default="*.yaml", help="Test case file pattern (default: *.yaml)")
@click.option("--test", "-t", multiple=True, help="Specific test name(s) to run (can specify multiple: -t test1 -t test2)")
@click.option("--filter", "-f", help="Filter tests by name pattern (e.g., 'LangGraph*', '*simple*')")
@click.option("--output", default=".evalview/results", help="Output directory for results")
@click.option("--verbose/--no-verbose", default=True, help="Verbose output with full test details (default: enabled)")
@click.option("--track", is_flag=True, help="Track results for regression analysis")
@click.option("--compare-baseline", is_flag=True, help="Compare results against baseline and show regressions")
@click.option("--debug", is_flag=True, help="Show detailed debug info: raw API response, parsed trace, type conversions")
@click.option("--sequential", is_flag=True, help="Run tests sequentially instead of in parallel (default: parallel)")
@click.option("--max-workers", default=8, type=int, help="Maximum parallel test executions (default: 8)")
@click.option("--max-retries", default=0, type=int, help="Maximum retries for flaky tests (default: 0 = no retries)")
@click.option("--retry-delay", default=1.0, type=float, help="Base delay between retries in seconds (default: 1.0)")
@click.option("--watch", is_flag=True, help="Watch test files and re-run on changes")
@click.option("--html-report", type=click.Path(), help="Generate HTML report to specified path")
@click.option("--summary", is_flag=True, help="Compact output with deltas vs last run and regression detection. Great for CI/CD and sharing.")
@click.option("--coverage", is_flag=True, help="Show behavior coverage report: tasks tested, tools exercised, paths covered, eval dimensions.")
@click.option("--judge-model", type=str, help="Model for LLM-as-judge (e.g., gpt-5, sonnet, llama-70b, gpt-4o). Aliases auto-resolve to full names.")
@click.option(
    "--judge-provider",
    type=click.Choice(["openai", "anthropic", "huggingface", "gemini", "grok", "ollama"]),
    help="Provider for LLM-as-judge evaluation (ollama = free local)",
)
@click.option(
    "--adapter",
    type=click.Choice(["http", "langgraph", "crewai", "anthropic", "openai-assistants", "tapescope", "huggingface", "goose", "ollama", "mcp", "cohere"]),
    help="Override adapter type (e.g., goose, langgraph, mcp). Overrides config file.",
)
@click.option("--diff", is_flag=True, help="Compare against golden baselines. Shows REGRESSION/TOOLS_CHANGED/OUTPUT_CHANGED/PASSED status.")
@click.option("--diff-report", type=click.Path(), help="Generate HTML diff report to specified path (requires --diff)")
@click.option("--fail-on", type=str, default=None, help="Comma-separated statuses that cause exit code 1: REGRESSION, TOOLS_CHANGED, OUTPUT_CHANGED, CONTRACT_DRIFT (default: REGRESSION)")
@click.option("--warn-on", type=str, default=None, help="Comma-separated diff statuses that print warning but exit 0 (default: TOOLS_CHANGED,OUTPUT_CHANGED, or from ci.warn_on in config.yaml)")
@click.option("--strict", is_flag=True, help="Strict mode: fail on any non-PASSED status (equivalent to --fail-on REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED)")
@click.option("--trace", is_flag=True, help="Show live trace output: LLM calls, tool executions, costs, and latency.")
@click.option("--trace-out", type=click.Path(), help="Export trace to JSONL file for debugging or sharing.")
@click.option("--runs", type=int, default=None, help="Run each test N times for statistical evaluation (enables pass@k metrics). Overrides per-test variance config.")
@click.option("--pass-rate", type=float, default=0.8, help="Required pass rate for statistical mode (0.0-1.0, default: 0.8). Only used with --runs.")
@click.option("--difficulty", type=click.Choice(["trivial", "easy", "medium", "hard", "expert"]), default=None, help="Filter tests by difficulty level.")
@click.option("--contracts", is_flag=True, help="Check MCP contracts for interface drift before running tests. Fails fast if external servers changed.")
@click.option("--save-golden", is_flag=True, default=False, help="Save results as golden baseline if all tests pass.")
@click.option("--no-judge", is_flag=True, default=False, help="Skip LLM-as-judge evaluation. Uses deterministic scoring only (string matching + tool assertions). Scores capped at 75. No API key required.")
@click.option("--judge-cache", is_flag=True, default=False, help="Cache LLM judge responses so identical outputs are not re-evaluated. Saves API costs in statistical mode (--runs).")
@click.option("--no-open", is_flag=True, default=False, help="Do not auto-open the HTML report in the browser after the run. Implied when CI=true.")
@track_command("run", lambda **kw: {"adapter": kw.get("adapter") or "auto", "has_path": bool(kw.get("path"))})
def run(
    path: Optional[str],
    pattern: str,
    test: tuple,
    filter: str,
    output: str,
    verbose: bool,
    track: bool,
    compare_baseline: bool,
    debug: bool,
    sequential: bool,
    max_workers: int,
    max_retries: int,
    retry_delay: float,
    watch: bool,
    html_report: str,
    summary: bool,
    coverage: bool,
    judge_model: Optional[str],
    judge_provider: Optional[str],
    adapter: Optional[str],
    diff: bool,
    diff_report: Optional[str],
    fail_on: Optional[str],
    warn_on: Optional[str],
    strict: bool,
    trace: bool,
    trace_out: Optional[str],
    runs: Optional[int],
    pass_rate: float,
    difficulty: Optional[str],
    contracts: bool,
    save_golden: bool,
    no_judge: bool,
    judge_cache: bool,
    no_open: bool,
) -> None:
    """Run test cases against the agent.

    PATH can be a directory containing test cases (e.g., examples/anthropic)
    or a specific test file (e.g., examples/anthropic/test-case.yaml).
    """
    if judge_provider:
        os.environ["EVAL_PROVIDER"] = judge_provider
    if judge_model:
        from evalview.core.llm_provider import resolve_model_alias
        os.environ["EVAL_MODEL"] = resolve_model_alias(judge_model)

    if strict:
        fail_on = "REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED,CONTRACT_DRIFT"
        warn_on = ""

    asyncio.run(_run_async(
        path=path, pattern=pattern, test=test, filter=filter, output=output,
        verbose=verbose, track=track, compare_baseline=compare_baseline, debug=debug,
        sequential=sequential, max_workers=max_workers, max_retries=max_retries,
        retry_delay=retry_delay, watch=watch, html_report=html_report,
        summary=summary, coverage=coverage, adapter_override=adapter,
        diff=diff, diff_report=diff_report, fail_on=fail_on, warn_on=warn_on,
        trace=trace, trace_out=trace_out, runs=runs, pass_rate=pass_rate,
        difficulty_filter=difficulty, contracts=contracts, save_golden=save_golden,
        no_judge=no_judge, judge_cache=judge_cache, no_open=no_open,
    ))


# ── Async orchestrator ─────────────────────────────────────────────────────────


async def _run_async(
    path: Optional[str],
    pattern: str,
    test: tuple,
    filter: str,
    output: str,
    verbose: bool,
    track: bool,
    compare_baseline: bool,
    debug: bool = False,
    sequential: bool = False,
    max_workers: int = 8,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    watch: bool = False,
    html_report: Optional[str] = None,
    summary: bool = False,
    coverage: bool = False,
    adapter_override: Optional[str] = None,
    diff: bool = False,
    diff_report: Optional[str] = None,
    fail_on: Optional[str] = None,
    warn_on: Optional[str] = None,
    trace: bool = False,
    trace_out: Optional[str] = None,
    runs: Optional[int] = None,
    pass_rate: float = 0.8,
    difficulty_filter: Optional[str] = None,
    contracts: bool = False,
    save_golden: bool = False,
    no_judge: bool = False,
    judge_cache: bool = False,
    no_open: bool = False,
) -> None:
    """Async implementation of the run command."""
    import fnmatch
    from dotenv import load_dotenv
    from evalview.tracking import RegressionTracker
    from evalview.core.retry import RetryConfig
    from evalview.core.config import ScoringWeights
    from evalview.evaluators.statistical_evaluator import StatisticalEvaluator
    from evalview.reporters.console_reporter import ConsoleReporter
    from evalview.reporters.trace_live_reporter import create_trace_reporter
    from evalview.commands.run._adapters import build_adapter
    from evalview.commands.run._executor import ExecutorOptions, execute_single_test
    from evalview.commands.run._runner import run_sequential, run_parallel
    from evalview.commands.run._reporter import (
        collect_diffs, display_diff_results, display_regression_analysis,
        save_results, save_golden_if_requested, display_html_reports,
        compute_exit_code, display_trust_frame,
    )

    # ── Environment ──────────────────────────────────────────────────────────
    if path:
        target_dir = Path(path) if Path(path).is_dir() else Path(path).parent
        path_env = target_dir / ".env.local"
        if path_env.exists():
            load_dotenv(dotenv_path=str(path_env), override=True)

    # ── Early config (needed before provider selection) ───────────────────────
    config_path = Path(".evalview/config.yaml")
    if path:
        target_dir = Path(path) if Path(path).is_dir() else Path(path).parent
        path_config = target_dir / ".evalview" / "config.yaml"
        if path_config.exists():
            config_path = path_config

    early_config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            early_config = yaml.safe_load(f) or {}

    print_evalview_banner(console, subtitle="[dim]Catch agent regressions before you ship[/dim]")

    # ── Connectivity check ────────────────────────────────────────────────────
    ec_adapter = (adapter_override or early_config.get("adapter", "http")).lower()
    no_http_adapters = {"openai-assistants", "anthropic", "ollama", "goose", "cohere"}
    ec_endpoint = early_config.get("endpoint") if ec_adapter not in no_http_adapters else None

    if ec_endpoint and ec_adapter not in no_http_adapters:
        import socket as _socket
        from urllib.parse import urlparse as _urlparse
        try:
            parsed = _urlparse(ec_endpoint)
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.settimeout(1.5)
            ok = sock.connect_ex((parsed.hostname or "localhost", parsed.port or 80)) == 0
            sock.close()
            if not ok:
                _display_no_agent_guide(ec_endpoint)
                return
        except Exception:
            _display_no_agent_guide(ec_endpoint)
            return
    elif not ec_endpoint and not early_config and ec_adapter not in no_http_adapters:
        _display_no_agent_guide(None)
        return

    # ── Judge / provider setup ────────────────────────────────────────────────
    judge_cfg = early_config.get("judge", {})
    if judge_cfg:
        if judge_cfg.get("provider"):
            os.environ["EVAL_PROVIDER"] = judge_cfg["provider"]
        if judge_cfg.get("model"):
            from evalview.core.llm_provider import resolve_model_alias
            os.environ["EVAL_MODEL"] = resolve_model_alias(judge_cfg["model"])

    if no_judge:
        console.print(
            "[yellow]⚠  --no-judge: skipping LLM-as-judge. Using deterministic scoring only "
            "(scores capped at 75).[/yellow]\n"
        )
    else:
        result = get_or_select_provider(console)
        if result is None:
            try:
                from evalview.telemetry.client import get_client as _tc
                from evalview.telemetry.events import CommandEvent as _CE
                _tc().track(_CE(
                    command_name="run_failed_early",
                    success=False,
                    properties={"failure_reason": "no_provider_configured", "has_config": bool(early_config)},
                ))
            except Exception:
                pass
            return

        selected_provider, selected_api_key = result

        from evalview.core.llm_provider import LLMProvider, PROVIDER_CONFIGS as _PROVIDER_CONFIGS
        if selected_provider != LLMProvider.OLLAMA:
            try:
                import anthropic as _anthropic
                import openai as _openai

                if selected_provider == LLMProvider.ANTHROPIC:
                    _anthropic.Anthropic(api_key=selected_api_key).models.list()
                elif selected_provider == LLMProvider.OPENAI:
                    _openai.OpenAI(api_key=selected_api_key).models.list()
            except Exception as probe_exc:
                probe_str = str(probe_exc).lower()
                is_auth = any(
                    kw in probe_str
                    for kw in ("authentication", "401", "invalid", "unauthorized", "api key")
                )
                if is_auth:
                    cfg = _PROVIDER_CONFIGS[selected_provider]
                    console.print("\n[bold red]✗ LLM judge authentication failed[/bold red]")
                    console.print(f"  Provider: [bold]{cfg.display_name}[/bold]")
                    console.print(f"  Error:    {str(probe_exc)[:120]}")
                    console.print()
                    console.print("  Fix one of the following:")
                    console.print(f"    1. Set a valid key:   [cyan]export {cfg.env_var}='sk-...'[/cyan]")
                    console.print("    2. Switch provider:   [cyan]evalview run --judge-provider openai[/cyan]")
                    console.print("    3. Skip LLM judge:    [cyan]evalview run --no-judge[/cyan]")
                    console.print()
                    return
                raise

        save_provider_preference(selected_provider)
        if not os.environ.get("EVAL_PROVIDER"):
            os.environ["EVAL_PROVIDER"] = selected_provider.value
        if selected_provider != LLMProvider.OLLAMA:
            provider_cfg = PROVIDER_CONFIGS[selected_provider]
            os.environ[provider_cfg.env_var] = selected_api_key

    # ── Mode announcements ────────────────────────────────────────────────────
    if debug:
        console.print("[dim]🐛 Debug mode enabled - will show raw responses[/dim]\n")
        verbose = True

    if verbose:
        console.print("[dim]🔍 Verbose mode enabled[/dim]\n")

    if track or compare_baseline:
        console.print("[dim]📊 Regression tracking enabled[/dim]\n")

    if sequential:
        console.print("[dim]⏳ Running tests sequentially[/dim]\n")
    else:
        console.print(f"[dim]⚡ Running tests in parallel (max {max_workers} workers)[/dim]\n")

    if max_retries > 0:
        console.print(f"[dim]🔄 Retry enabled: up to {max_retries} retries with {retry_delay}s base delay[/dim]\n")

    # ── Trace reporter ────────────────────────────────────────────────────────
    trace_reporter = None
    if trace or trace_out:
        trace_reporter = create_trace_reporter(console=console, trace_out_path=trace_out)
        if trace:
            console.print("[dim]📡 Trace mode enabled - showing live execution details[/dim]\n")
        if trace_out:
            console.print(f"[dim]📄 Trace output: {trace_out}[/dim]\n")

    # ── Watch mode availability ───────────────────────────────────────────────
    if watch:
        try:
            from evalview.core.watcher import WATCHDOG_AVAILABLE
            if not WATCHDOG_AVAILABLE:
                console.print("[yellow]⚠️  Watch mode requires watchdog. Install with: pip install watchdog[/yellow]")
                console.print("[dim]Falling back to single run mode...[/dim]\n")
                watch = False
            else:
                console.print("[dim]👀 Watch mode enabled - press Ctrl+C to stop[/dim]\n")
        except ImportError:
            console.print("[yellow]⚠️  Watch mode requires watchdog. Install with: pip install watchdog[/yellow]")
            watch = False

    # ── Full config loading ───────────────────────────────────────────────────
    config_path_final: Optional[Path] = None
    if path:
        target_dir = Path(path) if Path(path).is_dir() else Path(path).parent
        path_config = target_dir / ".evalview" / "config.yaml"
        if path_config.exists():
            config_path_final = path_config
            if verbose:
                console.print(f"[dim]📂 Using config from: {path_config}[/dim]")

    if config_path_final is None:
        config_path_final = Path(".evalview/config.yaml")

    config: Dict[str, Any] = {}
    if config_path_final.exists():
        with open(config_path_final) as f:
            config = yaml.safe_load(f) or {}
    elif verbose:
        console.print("[dim]No config file found - will use test case adapter/endpoint if available[/dim]")

    run_endpoint = config.get("endpoint", "")
    run_adapter_type = config.get("adapter", "http")
    if run_endpoint:
        console.print(f"[blue]Running test cases...[/blue]  [dim]→ {run_adapter_type}  {run_endpoint}[/dim]\n")
    else:
        console.print("[blue]Running test cases...[/blue]\n")

    # Apply CI config from config.yaml (CLI flags take precedence)
    ci_cfg = config.get("ci", {})
    if fail_on is None:
        raw = ci_cfg.get("fail_on", ["REGRESSION"])
        fail_on = ",".join(raw) if isinstance(raw, list) else str(raw)
    if warn_on is None:
        raw = ci_cfg.get("warn_on", ["TOOLS_CHANGED", "OUTPUT_CHANGED"])
        warn_on = ",".join(raw) if isinstance(raw, list) else str(raw)

    # ── MCP contract check ────────────────────────────────────────────────────
    contract_drifts: List[Any] = []
    if contracts:
        contract_drifts = await _check_mcp_contracts(fail_on, console)
        if contract_drifts and "CONTRACT_DRIFT" in (fail_on or "").upper():
            console.print("[bold red]Aborting: MCP contract drift detected. Fix contracts before running tests.[/bold red]")
            console.print("[dim]Accept changes: evalview mcp snapshot <endpoint> --name <name>[/dim]\n")
            raise SystemExit(1)

    # ── Model / SSRF / judge config ───────────────────────────────────────────
    model_config = config.get("model", {})
    if verbose and model_config:
        if isinstance(model_config, str):
            console.print(f"[dim]💰 Model: {model_config}[/dim]")
        elif isinstance(model_config, dict):
            console.print(f"[dim]💰 Model: {model_config.get('name', 'gpt-5-mini')}[/dim]")
            if "pricing" in model_config:
                console.print(
                    f"[dim]💵 Custom pricing: ${model_config['pricing']['input_per_1m']:.2f} in, "
                    f"${model_config['pricing']['output_per_1m']:.2f} out[/dim]"
                )

    allow_private_urls = config.get("allow_private_urls", True)
    if verbose:
        lock = "🔓" if allow_private_urls else "🔒"
        label = "allowing private URLs (local dev mode)" if allow_private_urls else "blocking private URLs"
        console.print(f"[dim]{lock} SSRF protection: {label}[/dim]")

    # Judge config from YAML overrides .env.local
    judge_yaml_cfg = config.get("judge", {})
    if judge_yaml_cfg:
        if judge_yaml_cfg.get("provider"):
            os.environ["EVAL_PROVIDER"] = judge_yaml_cfg["provider"]
        if judge_yaml_cfg.get("model"):
            from evalview.core.llm_provider import resolve_model_alias
            os.environ["EVAL_MODEL"] = resolve_model_alias(judge_yaml_cfg["model"])
        if verbose:
            console.print(
                f"[dim]⚖️  Judge: {judge_yaml_cfg.get('provider', 'default')} / "
                f"{judge_yaml_cfg.get('model', 'default')}[/dim]"
            )

    # ── Global adapter ────────────────────────────────────────────────────────
    adapter_type = adapter_override or config.get("adapter", "http")
    global_adapter = None

    if adapter_override and verbose:
        console.print(f"[dim]🔌 Adapter override: {adapter_override}[/dim]")

    has_endpoint = "endpoint" in config
    is_api_adapter = adapter_type in ("openai-assistants", "anthropic", "ollama", "cohere")
    is_cli_adapter = adapter_type == "goose"

    if has_endpoint or is_api_adapter or is_cli_adapter:
        if adapter_type == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            console.print("[red]❌ ANTHROPIC_API_KEY not found in environment.[/red]")
            console.print("[dim]Set it in your .env.local file or export it:[/dim]")
            console.print("[dim]  export ANTHROPIC_API_KEY=sk-ant-...[/dim]")
            return

        global_adapter = build_adapter(adapter_type, None, config, model_config, verbose, allow_private_urls)

        if adapter_type == "goose" and verbose:
            console.print("[dim]🪿 Using Goose CLI adapter[/dim]")

    # ── Evaluator setup ───────────────────────────────────────────────────────
    scoring_weights = None
    if "scoring" in config and "weights" in config["scoring"]:
        try:
            scoring_weights = ScoringWeights(**config["scoring"]["weights"])
            if verbose:
                w = scoring_weights
                console.print(
                    f"[dim]⚖️  Custom weights: tool={w.tool_accuracy}, "
                    f"output={w.output_quality}, sequence={w.sequence_correctness}[/dim]"
                )
        except Exception as exc:
            console.print(f"[yellow]⚠️  Invalid scoring weights in config: {exc}. Using defaults.[/yellow]")

    _judge_cache = None
    if judge_cache and not no_judge:
        from evalview.core.judge_cache import JudgeCache
        _judge_cache = JudgeCache()
        if verbose:
            console.print("[dim]Enabled LLM judge response cache[/dim]")

    evaluator = Evaluator(
        default_weights=scoring_weights,
        skip_llm_judge=no_judge,
        judge_cache=_judge_cache,
    )

    retry_config = RetryConfig(
        max_retries=max_retries,
        base_delay=retry_delay,
        exponential=True,
        jitter=True,
    )

    tracker = None
    regression_reports: Dict[str, Any] = {}
    if track or compare_baseline:
        tracker = RegressionTracker()

    # ── Load test cases ───────────────────────────────────────────────────────
    test_cases = _load_test_cases(path, pattern, verbose, console)
    if test_cases is None:
        return

    # ── Filter: difficulty ────────────────────────────────────────────────────
    if difficulty_filter:
        original = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.difficulty == difficulty_filter]
        if not test_cases:
            console.print(f"[yellow]⚠️  No test cases with difficulty '{difficulty_filter}' found[/yellow]")
            console.print(f"[dim]Original count: {original} tests[/dim]")
            return
        if verbose:
            console.print(f"[dim]🎯 Filtered to {len(test_cases)}/{original} tests with difficulty: {difficulty_filter}[/dim]\n")

    # ── Filter: quality check ─────────────────────────────────────────────────
    test_cases = _apply_quality_filter(test_cases, console)
    if not test_cases:
        console.print("[yellow]⚠️  No runnable tests. Fix test quality issues above and re-run.[/yellow]")
        return

    # ── Variance / statistical mode ───────────────────────────────────────────
    if runs is not None:
        if runs < 2:
            console.print("[red]❌ --runs must be at least 2 for statistical mode[/red]")
            return
        if runs > 100:
            console.print("[red]❌ --runs cannot exceed 100[/red]")
            return

        from evalview.core.types import VarianceConfig
        cli_variance = VarianceConfig(runs=runs, pass_rate=pass_rate)
        for tc in test_cases:
            tc.thresholds.variance = cli_variance
        console.print(f"[cyan]📊 Statistical mode: Running each test {runs} times (pass rate: {pass_rate:.0%})[/cyan]\n")

    # ── Interactive adapter menu (multi-adapter projects) ─────────────────────
    if pattern == "*.yaml" and not test and not filter and sys.stdin.isatty():
        test_cases, html_report = _maybe_show_adapter_menu(test_cases, config, html_report, console)

    # ── Filter: name / glob ───────────────────────────────────────────────────
    if test or filter:
        test_cases = _filter_by_name(test_cases, test, filter, verbose, fnmatch, console)
        if test_cases is None:
            return

    console.print(f"Found {len(test_cases)} test case(s)\n")

    # ── Build executor options ────────────────────────────────────────────────
    statistical_evaluator = StatisticalEvaluator()
    stats_reporter = ConsoleReporter()

    exec_opts = ExecutorOptions(
        evaluator=evaluator,
        retry_config=retry_config,
        global_adapter=global_adapter,
        model_config=model_config,
        allow_private_urls=allow_private_urls,
        config=config,
        verbose=verbose,
        debug=debug,
        track=track,
        compare_baseline=compare_baseline,
        tracker=tracker,
        regression_reports=regression_reports,
        trace_reporter=trace_reporter,
        statistical_evaluator=statistical_evaluator,
        stats_reporter=stats_reporter,
        no_judge=no_judge,
    )

    async def _execute(test_case: Any) -> Any:
        return await execute_single_test(test_case, exec_opts, console)

    # ── Run tests ─────────────────────────────────────────────────────────────
    if sequential:
        results, passed, failed, execution_errors = await run_sequential(test_cases, _execute, console, config)
    else:
        results, passed, failed, execution_errors = await run_parallel(
            test_cases, _execute, max_workers, verbose, console, config
        )

    # ── Judge cache stats ─────────────────────────────────────────────────────
    if _judge_cache is not None:
        cs = _judge_cache.stats()
        if cs["total"] > 0:
            console.print(
                f"  [dim]Judge cache: {cs['hits']} hits / {cs['total']} lookups "
                f"({cs['hit_rate']:.0%} hit rate)[/dim]"
            )

    # ── Summary / coverage / regression analysis ──────────────────────────────
    console.print()
    reporter = ConsoleReporter()
    if summary:
        suite_name = (Path(path).name if Path(path).is_dir() else Path(path).stem) if path else None
        previous = None
        output_dir = Path(output)
        if output_dir.exists():
            from evalview.reporters.json_reporter import JSONReporter
            previous = JSONReporter.get_latest_results(output_dir)
        reporter.print_compact_summary(results, suite_name=suite_name, previous_results=previous)
    else:
        reporter.print_summary(results)

    if coverage:
        suite_name = (Path(path).name if Path(path).is_dir() else Path(path).stem) if path else None
        reporter.print_coverage_report(test_cases, results, suite_name=suite_name)

    display_regression_analysis(regression_reports, console)

    # ── Save results ──────────────────────────────────────────────────────────
    results_file = save_results(results, output, console)
    save_golden_if_requested(save_golden, failed, execution_errors, results, results_file, console)

    # ── Diff display ──────────────────────────────────────────────────────────
    diffs_found: List[Any] = []
    if diff and results:
        diffs_found = collect_diffs(results)
        display_diff_results(diffs_found, results, console)

    # ── HTML reports ──────────────────────────────────────────────────────────
    display_html_reports(html_report, diff_report, diff, diffs_found, results, no_open, watch, console)

    # ── Tracking tip ──────────────────────────────────────────────────────────
    if track:
        console.print("[dim]📊 Results tracked for regression analysis[/dim]")
        console.print("[dim]   View trends: evalview trends[/dim]")
        console.print("[dim]   Set baseline: evalview baseline set[/dim]\n")

    # ── Quick tips ────────────────────────────────────────────────────────────
    if not watch and results:
        if not summary and not coverage:
            console.print("[dim]Quick views:  evalview run --summary | evalview run --coverage[/dim]")
        if diff:
            console.print("[dim]Compare runs: evalview view --run-id <id>[/dim]")
        console.print()

    # ── Guided conversion to snapshot workflow ────────────────────────────────
    if not watch and not diff and results:
        from evalview.core.golden import GoldenStore
        from evalview.core.project_state import ProjectStateStore
        from evalview.core.celebrations import Celebrations

        store = GoldenStore()
        state_store = ProjectStateStore()
        if (
            not store.list_golden()
            and all(r.passed for r in results)
            and not state_store.load().conversion_suggestion_shown
        ):
            Celebrations.conversion_suggestion(len(results))
            state_store.mark_conversion_shown()

    # ── Exit code ─────────────────────────────────────────────────────────────
    exit_code = compute_exit_code(failed, execution_errors, diff, diffs_found, fail_on, warn_on, console)

    # ── Trust-framing summary ─────────────────────────────────────────────────
    if not watch:
        display_trust_frame(passed, failed, execution_errors, results, results_file, console)

    # ── Telemetry ─────────────────────────────────────────────────────────────
    try:
        import time as _time
        track_run_command(
            adapter_type=adapter_type,
            test_count=len(test_cases),
            pass_count=passed,
            fail_count=failed,
            duration_ms=0.0,
            diff_mode=diff,
            watch_mode=watch,
            parallel=not sequential,
        )
    except Exception:
        pass

    # ── Watch mode ────────────────────────────────────────────────────────────
    if watch:
        await _run_watch_mode(
            path=path, pattern=pattern, test=test, filter=filter, output=output,
            verbose=verbose, track=track, compare_baseline=compare_baseline,
            debug=debug, sequential=sequential, max_workers=max_workers,
            max_retries=max_retries, retry_delay=retry_delay,
            html_report=html_report, console=console,
        )
    else:
        if trace_reporter:
            trace_reporter.close()
        if exit_code != 0:
            sys.exit(exit_code)


# ── Phase helpers ──────────────────────────────────────────────────────────────


async def _check_mcp_contracts(fail_on: Optional[str], console: Any) -> List[Any]:
    """Run MCP contract drift checks. Returns list of drifted contracts."""
    from evalview.core.mcp_contract import ContractStore
    from evalview.core.contract_diff import diff_contract, ContractDriftStatus
    from evalview.adapters.mcp_adapter import MCPAdapter as MCPContractAdapter

    store = ContractStore()
    all_contracts = store.list_contracts()
    drifts: List[Any] = []

    if not all_contracts:
        console.print("[dim]--contracts: No contracts found. Create one: evalview mcp snapshot <endpoint> --name <name>[/dim]\n")
        return drifts

    console.print("[cyan]━━━ MCP Contract Check ━━━[/cyan]\n")
    for meta in all_contracts:
        contract = store.load_contract(meta.server_name)
        if not contract:
            continue
        adapter = MCPContractAdapter(endpoint=contract.metadata.endpoint, timeout=30.0)
        try:
            current_tools = await adapter.discover_tools()
        except Exception as exc:
            console.print(f"  [yellow]WARN: {meta.server_name}[/yellow] - could not connect: {exc}")
            continue

        result = diff_contract(contract, current_tools)
        if result.status == ContractDriftStatus.CONTRACT_DRIFT:
            drifts.append(result)
            console.print(f"  [red]CONTRACT_DRIFT: {meta.server_name}[/red] - {result.summary()}")
            for change in result.breaking_changes:
                console.print(f"    [red]{change.kind.value}: {change.tool_name}[/red] - {change.detail}")
        else:
            console.print(f"  [green]PASSED: {meta.server_name}[/green]")

    console.print()
    return drifts


def _load_test_cases(
    path: Optional[str],
    pattern: str,
    verbose: bool,
    console: Any,
) -> Optional[List[Any]]:
    """Load test cases from path / pattern / default directory.

    Returns the list of test cases, or None if a fatal error occurred.
    """
    if path:
        target = Path(path)
        if target.is_file():
            try:
                cases = [TestCaseLoader.load_from_file(target)]
                if verbose:
                    console.print(f"[dim]📄 Loading test case from: {path}[/dim]\n")
                return cases
            except Exception as exc:
                console.print(f"[red]❌ Failed to load test case: {exc}[/red]")
                return None
        elif target.is_dir():
            cases = TestCaseLoader.load_from_directory(target, "*.yaml")
            if verbose:
                console.print(f"[dim]📁 Loading test cases from: {path}[/dim]\n")
            return cases
        else:
            console.print(f"[red]❌ Path not found: {path}[/red]")
            return None

    pattern_path = Path(pattern)
    if pattern_path.is_file():
        try:
            cases = [TestCaseLoader.load_from_file(pattern_path)]
            if verbose:
                console.print(f"[dim]📄 Loading test case from: {pattern}[/dim]\n")
            return cases
        except Exception as exc:
            console.print(f"[red]❌ Failed to load test case: {exc}[/red]")
            return None
    elif pattern_path.is_dir():
        cases = TestCaseLoader.load_from_directory(pattern_path, "*.yaml")
        if verbose:
            console.print(f"[dim]📁 Loading test cases from: {pattern}[/dim]\n")
        return cases

    default_dir = Path("tests/test-cases")
    if not default_dir.exists():
        console.print("[red]❌ Test cases directory not found: tests/test-cases[/red]")
        console.print("[dim]Tip: You can specify a path or file directly:[/dim]")
        console.print("[dim]  evalview run examples/anthropic[/dim]")
        console.print("[dim]  evalview run path/to/test-case.yaml[/dim]")
        return None

    cases = TestCaseLoader.load_from_directory(default_dir, pattern)
    if not cases:
        console.print(f"[yellow]⚠️  No test cases found matching pattern: {pattern}[/yellow]\n")
        console.print("[bold]💡 Create tests by:[/bold]")
        console.print("   • [cyan]evalview record --interactive[/cyan]   (record agent interactions)")
        console.print("   • [cyan]evalview expand <test.yaml>[/cyan]     (generate variations from seed)")
        console.print("   • Or create YAML files manually in tests/test-cases/")
        console.print()
        return None
    return cases


def _apply_quality_filter(test_cases: List[Any], console: Any) -> List[Any]:
    """Warn about low-quality tests and skip auto-generated ones below threshold."""
    from evalview.core.test_quality import score_test_quality, QUALITY_THRESHOLD

    skipped: List[str] = []
    qualified: List[Any] = []

    for tc in test_cases:
        q_score, q_issues = score_test_quality(tc)
        if q_score < QUALITY_THRESHOLD:
            if tc.generated:
                skipped.append(tc.name)
                console.print(f"[yellow]⚠  {tc.name} skipped [generated, quality: {q_score}/100][/yellow]")
                for issue in q_issues:
                    console.print(f"[dim]     • {issue}[/dim]")
            else:
                console.print(
                    f"[yellow]⚠  {tc.name} [quality: {q_score}/100] — score may reflect test issues, "
                    "not agent issues[/yellow]"
                )
                for issue in q_issues:
                    console.print(f"[dim]     • {issue}[/dim]")
                qualified.append(tc)
        else:
            qualified.append(tc)

    if skipped:
        console.print(f"[dim]   {len(skipped)} auto-generated test(s) skipped. Fix and re-run, or rewrite manually.[/dim]\n")

    return qualified


def _maybe_show_adapter_menu(
    test_cases: List[Any],
    config: Dict[str, Any],
    html_report: Optional[str],
    console: Any,
) -> tuple:
    """Show interactive adapter selection menu when multiple adapters are present."""
    import socket

    tests_by_adapter: Dict[str, List[Any]] = {}
    for tc in test_cases:
        key = tc.adapter or config.get("adapter", "http")
        tests_by_adapter.setdefault(key, []).append(tc)

    if len(tests_by_adapter) <= 1:
        return test_cases, html_report

    # Health check endpoints
    def _port_open(endpoint: str) -> bool:
        if not endpoint:
            return False
        try:
            from urllib.parse import urlparse
            parsed = urlparse(endpoint)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            ok = sock.connect_ex((parsed.hostname or "localhost", parsed.port or 80)) == 0
            sock.close()
            return ok
        except Exception:
            return False

    adapter_endpoints: Dict[str, str] = {}
    for name, tests in tests_by_adapter.items():
        for tc in tests:
            if tc.endpoint:
                adapter_endpoints[name] = tc.endpoint
                break
        if name not in adapter_endpoints:
            adapter_endpoints[name] = config.get("endpoint", "")

    console.print("[bold]📋 Test cases found:[/bold]\n")
    menu: List[Any] = []
    for i, (name, tests) in enumerate(tests_by_adapter.items(), 1):
        health = "[green]✅[/green]" if _port_open(adapter_endpoints.get(name, "")) else "[red]❌[/red]"
        endpoint = adapter_endpoints.get(name, "N/A")
        console.print(f"  [{i}] [bold]{name.upper()}[/bold] ({len(tests)} tests) {health}")
        console.print(f"      Endpoint: {endpoint}")
        for tc in tests[:3]:
            console.print(f"        • {tc.name}")
        if len(tests) > 3:
            console.print(f"        • ... and {len(tests) - 3} more")
        console.print()
        menu.append((name, tests))

    console.print(f"  [{len(menu) + 1}] [bold]All tests[/bold] ({len(test_cases)} tests)")
    console.print()

    choice = click.prompt("Which tests to run?", type=int, default=len(menu) + 1)
    if 1 <= choice <= len(menu):
        _, test_cases = menu[choice - 1]
        console.print(f"\n[cyan]Running {menu[choice - 1][0].upper()} tests...[/cyan]")
    else:
        console.print("\n[cyan]Running all tests...[/cyan]")

    console.print("\n[bold]Run mode:[/bold]")
    console.print("  [1] Parallel (faster, default)")
    console.print("  [2] Sequential (easier to follow)")
    run_mode = click.prompt("Select run mode", type=int, default=1)
    if run_mode == 2:
        console.print("[dim]Running tests sequentially...[/dim]\n")
    else:
        console.print("[dim]Running tests in parallel...[/dim]\n")

    cost_model = config.get("model", "gpt-4o-mini")
    console.print(f"[dim]💰 Cost calculated using: {cost_model} pricing[/dim]")
    console.print("[dim]   (Configure in .evalview/config.yaml or test case)[/dim]\n")

    if not html_report:
        if click.confirm("Generate HTML report?", default=True):
            html_report = f".evalview/results/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            console.print(f"[dim]📊 HTML report will be saved to: {html_report}[/dim]\n")

    return test_cases, html_report


def _filter_by_name(
    test_cases: List[Any],
    test: tuple,
    filter: Optional[str],
    verbose: bool,
    fnmatch_module: Any,
    console: Any,
) -> Optional[List[Any]]:
    """Filter test cases by exact name (--test) or glob pattern (--filter)."""
    original_count = len(test_cases)
    filtered: List[Any] = []

    for tc in test_cases:
        name_lower = tc.name.lower()
        if test and any(t.lower() == name_lower for t in test):
            filtered.append(tc)
            continue
        if filter:
            f_lower = filter.lower()
            if "*" in filter or "?" in filter:
                if fnmatch_module.fnmatch(name_lower, f_lower):
                    filtered.append(tc)
            elif f_lower in name_lower:
                filtered.append(tc)

    if not filtered:
        console.print("[yellow]⚠️  No test cases matched the filter criteria[/yellow]")
        return None

    if verbose:
        console.print(f"[dim]Filtered {original_count} → {len(filtered)} test(s)[/dim]\n")
    return filtered


async def _run_watch_mode(
    console: Any,
    **kwargs: Any,
) -> None:
    """Start file watcher and re-run tests on every change."""
    from evalview.core.watcher import TestWatcher

    console.print("[cyan]━" * 60 + "[/cyan]")
    console.print("[cyan]👀 Watching for changes... (Ctrl+C to stop)[/cyan]")
    console.print("[cyan]━" * 60 + "[/cyan]\n")

    run_count = 0

    async def _rerun() -> None:
        nonlocal run_count
        run_count += 1
        console.print(f"\n[blue]━━━ Run #{run_count} ━━━[/blue]\n")
        await _run_async(watch=False, no_open=True, **kwargs)

    watcher = TestWatcher(
        paths=["tests/test-cases", ".evalview"],
        run_callback=_rerun,  # type: ignore[arg-type]
        debounce_seconds=2.0,
    )
    try:
        await watcher.start()
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Watch mode stopped.[/yellow]")
    finally:
        watcher.stop()


# Convenience re-export used by legacy imports
from datetime import datetime
