"""Run command — execute test cases against the agent with full evaluation."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from evalview.adapters.http_adapter import HTTPAdapter
from evalview.adapters.tapescope_adapter import TapeScopeAdapter
from evalview.adapters.langgraph_adapter import LangGraphAdapter
from evalview.adapters.crewai_adapter import CrewAIAdapter
from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter
from evalview.commands.shared import console, _execute_multi_turn_trace
from evalview.core.loader import TestCaseLoader
from evalview.core.llm_provider import (
    get_or_select_provider,
    save_provider_preference,
    PROVIDER_CONFIGS,
    judge_cost_tracker,
)
from evalview.evaluators.evaluator import Evaluator
from evalview.reporters.json_reporter import JSONReporter
from evalview.reporters.console_reporter import ConsoleReporter
from evalview.skills.ui_utils import print_evalview_banner
from evalview.telemetry.decorators import track_command, track_run_command

logger = logging.getLogger(__name__)


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
    # Set judge model/provider via env vars if specified (CLI overrides env)
    if judge_provider:
        os.environ["EVAL_PROVIDER"] = judge_provider
    if judge_model:
        from evalview.core.llm_provider import resolve_model_alias
        os.environ["EVAL_MODEL"] = resolve_model_alias(judge_model)

    # Handle --strict flag (overrides config and CLI)
    if strict:
        fail_on = "REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED,CONTRACT_DRIFT"
        warn_on = ""

    asyncio.run(_run_async(
        path, pattern, test, filter, output, verbose, track, compare_baseline, debug,
        sequential, max_workers, max_retries, retry_delay, watch, html_report, summary, coverage,
        adapter_override=adapter, diff=diff, diff_report=diff_report,
        fail_on=fail_on, warn_on=warn_on, trace=trace, trace_out=trace_out,
        runs=runs, pass_rate=pass_rate, difficulty_filter=difficulty,
        contracts=contracts, save_golden=save_golden,
        no_judge=no_judge, judge_cache=judge_cache,
        no_open=no_open,
    ))


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
    """Async implementation of run command."""
    import fnmatch
    import json as json_module
    from evalview.tracking import RegressionTracker
    from evalview.core.parallel import execute_tests_parallel
    from evalview.core.retry import RetryConfig, with_retry
    from evalview.core.config import ScoringWeights
    from evalview.evaluators.statistical_evaluator import (
        StatisticalEvaluator,
        is_statistical_mode,
    )
    from evalview.reporters.console_reporter import ConsoleReporter
    from evalview.reporters.trace_live_reporter import create_trace_reporter
    from dotenv import load_dotenv

    # Load environment variables from path directory if provided
    if path:
        target_dir = Path(path) if Path(path).is_dir() else Path(path).parent
        path_env = target_dir / ".env.local"
        if path_env.exists():
            load_dotenv(dotenv_path=str(path_env), override=True)

    # Load config EARLY to get judge settings before provider selection
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

    # ── Connectivity check ──────────────────────────────────────────────────
    _ec_adapter = (adapter_override or early_config.get("adapter", "http")).lower()
    _ec_no_http_check = {"openai-assistants", "anthropic", "ollama", "goose", "cohere"}
    _ec_endpoint = early_config.get("endpoint") if _ec_adapter not in _ec_no_http_check else None

    if _ec_endpoint and _ec_adapter not in _ec_no_http_check:
        import socket as _socket
        from urllib.parse import urlparse as _urlparse
        try:
            _p = _urlparse(_ec_endpoint)
            _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            _sock.settimeout(1.5)
            _ok = _sock.connect_ex((_p.hostname or "localhost", _p.port or 80)) == 0
            _sock.close()
            if not _ok:
                _display_no_agent_guide(_ec_endpoint)
                return
        except Exception:
            _display_no_agent_guide(_ec_endpoint)
            return
    elif not _ec_endpoint and not early_config and _ec_adapter not in _ec_no_http_check:
        _display_no_agent_guide(None)
        return

    # Apply judge config from config file BEFORE provider selection
    judge_config = early_config.get("judge", {})
    if judge_config:
        if judge_config.get("provider"):
            os.environ["EVAL_PROVIDER"] = judge_config["provider"]
        if judge_config.get("model"):
            from evalview.core.llm_provider import resolve_model_alias
            os.environ["EVAL_MODEL"] = resolve_model_alias(judge_config["model"])

    # --no-judge: skip provider selection and LLM evaluation entirely
    if no_judge:
        console.print("[yellow]⚠  --no-judge: skipping LLM-as-judge. Using deterministic scoring only (scores capped at 75).[/yellow]\n")
    else:
        # Interactive provider selection for LLM-as-judge
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

        # Validate the API key with a lightweight probe before running any tests.
        from evalview.core.llm_provider import LLMProvider, PROVIDER_CONFIGS as _PROVIDER_CONFIGS
        if selected_provider != LLMProvider.OLLAMA:
            try:
                import anthropic as _anthropic
                import openai as _openai

                if selected_provider == LLMProvider.ANTHROPIC:
                    _anthropic.Anthropic(api_key=selected_api_key).models.list()
                elif selected_provider == LLMProvider.OPENAI:
                    _openai.OpenAI(api_key=selected_api_key).models.list()
            except Exception as _probe_exc:
                _probe_str = str(_probe_exc).lower()
                _is_auth = (
                    "authentication" in _probe_str
                    or "401" in _probe_str
                    or "invalid" in _probe_str
                    or "unauthorized" in _probe_str
                    or "api key" in _probe_str
                )
                if _is_auth:
                    _provider_cfg = _PROVIDER_CONFIGS[selected_provider]
                    console.print("\n[bold red]✗ LLM judge authentication failed[/bold red]")
                    console.print(f"  Provider: [bold]{_provider_cfg.display_name}[/bold]")
                    console.print(f"  Error:    {str(_probe_exc)[:120]}")
                    console.print()
                    console.print("  Fix one of the following:")
                    console.print(f"    1. Set a valid key:   [cyan]export {_provider_cfg.env_var}='sk-...'[/cyan]")
                    console.print("    2. Switch provider:   [cyan]evalview run --judge-provider openai[/cyan]")
                    console.print("    3. Skip LLM judge:    [cyan]evalview run --no-judge[/cyan]")
                    console.print()
                    return
                raise  # non-auth errors propagate normally

        # Save preference for future runs
        save_provider_preference(selected_provider)

        config_for_provider = PROVIDER_CONFIGS[selected_provider]
        if not os.environ.get("EVAL_PROVIDER"):
            os.environ["EVAL_PROVIDER"] = selected_provider.value
        if selected_provider != LLMProvider.OLLAMA:
            os.environ[config_for_provider.env_var] = selected_api_key

    if debug:
        console.print("[dim]🐛 Debug mode enabled - will show raw responses[/dim]\n")
        verbose = True  # Debug implies verbose

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

    # Initialize trace reporter if trace mode enabled
    trace_reporter = None
    if trace or trace_out:
        trace_reporter = create_trace_reporter(
            console=console,
            trace_out_path=trace_out,
        )
        if trace:
            console.print("[dim]📡 Trace mode enabled - showing live execution details[/dim]\n")
        if trace_out:
            console.print(f"[dim]📄 Trace output: {trace_out}[/dim]\n")

    # Handle watch mode
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

    # Load config - check path directory first, then current directory
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

    config_exists = config_path_final.exists()
    if config_exists:
        with open(config_path_final) as f:
            config: Dict[str, Any] = yaml.safe_load(f) or {}
    else:
        config = {}
        if verbose:
            console.print("[dim]No config file found - will use test case adapter/endpoint if available[/dim]")

    _run_endpoint = config.get("endpoint", "")
    _run_adapter = config.get("adapter", "http")
    if _run_endpoint:
        console.print(f"[blue]Running test cases...[/blue]  [dim]→ {_run_adapter}  {_run_endpoint}[/dim]\n")
    else:
        console.print("[blue]Running test cases...[/blue]\n")

    # Apply CI config from config.yaml (if CLI flags not provided)
    ci_config = config.get("ci", {})
    if fail_on is None:
        config_fail_on = ci_config.get("fail_on", ["REGRESSION"])
        if isinstance(config_fail_on, list):
            fail_on = ",".join(config_fail_on)
        else:
            fail_on = str(config_fail_on)
    if warn_on is None:
        config_warn_on = ci_config.get("warn_on", ["TOOLS_CHANGED", "OUTPUT_CHANGED"])
        if isinstance(config_warn_on, list):
            warn_on = ",".join(config_warn_on)
        else:
            warn_on = str(config_warn_on)

    # --- MCP Contract Check (runs before tests, fail fast) ---
    contract_drifts: List[Any] = []
    if contracts:
        from evalview.core.mcp_contract import ContractStore
        from evalview.core.contract_diff import diff_contract, ContractDriftStatus
        from evalview.adapters.mcp_adapter import MCPAdapter as MCPContractAdapter

        contract_store = ContractStore()
        all_contracts = contract_store.list_contracts()

        if all_contracts:
            console.print("[cyan]━━━ MCP Contract Check ━━━[/cyan]\n")

            for meta in all_contracts:
                contract = contract_store.load_contract(meta.server_name)
                if not contract:
                    continue

                mcp_adapter = MCPContractAdapter(endpoint=contract.metadata.endpoint, timeout=30.0)
                try:
                    current_tools = await mcp_adapter.discover_tools()
                except Exception as e:
                    console.print(f"  [yellow]WARN: {meta.server_name}[/yellow] - could not connect: {e}")
                    continue

                contract_result = diff_contract(contract, current_tools)

                if contract_result.status == ContractDriftStatus.CONTRACT_DRIFT:
                    contract_drifts.append(contract_result)
                    console.print(f"  [red]CONTRACT_DRIFT: {meta.server_name}[/red] - {contract_result.summary()}")
                    for change in contract_result.breaking_changes:
                        console.print(f"    [red]{change.kind.value}: {change.tool_name}[/red] - {change.detail}")
                else:
                    console.print(f"  [green]PASSED: {meta.server_name}[/green]")

            console.print()

            if contract_drifts and "CONTRACT_DRIFT" in (fail_on or "").upper():
                console.print("[bold red]Aborting: MCP contract drift detected. Fix contracts before running tests.[/bold red]")
                console.print("[dim]Accept changes: evalview mcp snapshot <endpoint> --name <name>[/dim]\n")
                raise SystemExit(1)
        else:
            console.print("[dim]--contracts: No contracts found. Create one: evalview mcp snapshot <endpoint> --name <name>[/dim]\n")

    # Extract model config (can be string or dict)
    model_config = config.get("model", {})
    if verbose and model_config:
        if isinstance(model_config, str):
            console.print(f"[dim]💰 Model: {model_config}[/dim]")
        elif isinstance(model_config, dict):
            console.print(f"[dim]💰 Model: {model_config.get('name', 'gpt-5-mini')}[/dim]")
            if "pricing" in model_config:
                console.print(
                    f"[dim]💵 Custom pricing: ${model_config['pricing']['input_per_1m']:.2f} in, ${model_config['pricing']['output_per_1m']:.2f} out[/dim]"
                )

    # SSRF protection config
    allow_private_urls = config.get("allow_private_urls", True)
    if verbose:
        if allow_private_urls:
            console.print("[dim]🔓 SSRF protection: allowing private URLs (local dev mode)[/dim]")
        else:
            console.print("[dim]🔒 SSRF protection: blocking private URLs[/dim]")

    # Load judge config from config file (config.yaml overrides .env.local)
    judge_config = config.get("judge", {})
    if judge_config:
        if judge_config.get("provider"):
            os.environ["EVAL_PROVIDER"] = judge_config["provider"]
        if judge_config.get("model"):
            from evalview.core.llm_provider import resolve_model_alias
            os.environ["EVAL_MODEL"] = resolve_model_alias(judge_config["model"])
        if verbose:
            console.print(f"[dim]⚖️  Judge: {judge_config.get('provider', 'default')} / {judge_config.get('model', 'default')}[/dim]")

    # Initialize adapter based on type
    adapter_type = adapter_override if adapter_override else config.get("adapter", "http")
    adapter: Any = None  # Will be None if no config - test cases must provide their own adapter/endpoint

    if adapter_override and verbose:
        console.print(f"[dim]🔌 Adapter override: {adapter_override}[/dim]")

    has_endpoint = "endpoint" in config
    is_api_adapter = adapter_type in ["openai-assistants", "anthropic", "ollama", "cohere"]
    is_cli_adapter = adapter_type in ["goose"]

    if has_endpoint or is_api_adapter or is_cli_adapter:
        if adapter_type == "langgraph":
            adapter = LangGraphAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30.0),
                streaming=config.get("streaming", False),
                verbose=verbose,
                model_config=model_config,
                assistant_id=config.get("assistant_id", "agent"),
                allow_private_urls=allow_private_urls,
            )
        elif adapter_type == "crewai":
            adapter = CrewAIAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 120.0),
                verbose=verbose,
                model_config=model_config,
                allow_private_urls=allow_private_urls,
            )
        elif adapter_type == "openai-assistants":
            adapter = OpenAIAssistantsAdapter(
                assistant_id=config.get("assistant_id"),
                timeout=config.get("timeout", 120.0),
                verbose=verbose,
                model_config=model_config,
            )
        elif adapter_type in ["streaming", "tapescope", "jsonl"]:
            adapter = TapeScopeAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 60.0),
                verbose=verbose,
                model_config=model_config,
                allow_private_urls=allow_private_urls,
            )
        elif adapter_type == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                console.print("[red]❌ ANTHROPIC_API_KEY not found in environment.[/red]")
                console.print("[dim]Set it in your .env.local file or export it:[/dim]")
                console.print("[dim]  export ANTHROPIC_API_KEY=sk-ant-...[/dim]")
                return

            from evalview.adapters.anthropic_adapter import AnthropicAdapter

            anthropic_model = config.get("model", "claude-sonnet-4-5-20250929")
            if isinstance(anthropic_model, dict):
                anthropic_model = anthropic_model.get("name", "claude-sonnet-4-5-20250929")

            adapter = AnthropicAdapter(
                model=anthropic_model,
                tools=config.get("tools", []),
                system_prompt=config.get("system_prompt"),
                max_tokens=config.get("max_tokens", 4096),
                timeout=config.get("timeout", 120.0),
                verbose=verbose,
            )
        elif adapter_type in ["huggingface", "hf", "gradio"]:
            from evalview.adapters.huggingface_adapter import HuggingFaceAdapter

            adapter = HuggingFaceAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 120.0),
                hf_token=os.getenv("HF_TOKEN"),
                function_name=config.get("function_name"),
                verbose=verbose,
                model_config=model_config,
                allow_private_urls=allow_private_urls,
            )
        elif adapter_type == "ollama":
            from evalview.adapters.ollama_adapter import OllamaAdapter

            ollama_model = config.get("model", "llama3.2")
            if isinstance(ollama_model, dict):
                ollama_model = ollama_model.get("name", "llama3.2")

            adapter = OllamaAdapter(
                model=ollama_model,
                endpoint=config.get("endpoint", "http://localhost:11434"),
                timeout=config.get("timeout", 60.0),
                verbose=verbose,
                model_config=model_config,
            )
        elif adapter_type == "goose":
            from evalview.adapters.goose_adapter import GooseAdapter

            adapter = GooseAdapter(
                timeout=config.get("timeout", 300.0),
                cwd=config.get("cwd"),
                extensions=config.get("extensions", ["developer"]),
                provider=config.get("provider"),
                model=config.get("goose_model"),
            )
            if verbose:
                console.print("[dim]🪿 Using Goose CLI adapter[/dim]")
        elif adapter_type == "cohere":
            from evalview.adapters.cohere_adapter import CohereAdapter

            cohere_model = config.get("model")
            if isinstance(cohere_model, dict):
                cohere_model = cohere_model.get("name")
            adapter = CohereAdapter(model=cohere_model)
        else:
            adapter = HTTPAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30.0),
                model_config=model_config,
                allow_private_urls=allow_private_urls,
            )

    # Initialize evaluator with configurable weights
    scoring_weights = None
    if "scoring" in config and "weights" in config["scoring"]:
        try:
            scoring_weights = ScoringWeights(**config["scoring"]["weights"])
            if verbose:
                console.print(f"[dim]⚖️  Custom weights: tool={scoring_weights.tool_accuracy}, output={scoring_weights.output_quality}, sequence={scoring_weights.sequence_correctness}[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Invalid scoring weights in config: {e}. Using defaults.[/yellow]")

    # Build judge cache if requested
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

    # Setup retry config
    retry_config = RetryConfig(
        max_retries=max_retries,
        base_delay=retry_delay,
        exponential=True,
        jitter=True,
    )

    # Initialize tracker if tracking enabled
    tracker = None
    regression_reports: Dict[str, Any] = {}
    if track or compare_baseline:
        tracker = RegressionTracker()

    # Load test cases
    if path:
        target_path = Path(path)
        if target_path.exists() and target_path.is_file():
            try:
                test_cases = [TestCaseLoader.load_from_file(target_path)]
                if verbose:
                    console.print(f"[dim]📄 Loading test case from: {path}[/dim]\n")
            except Exception as e:
                console.print(f"[red]❌ Failed to load test case: {e}[/red]")
                return
        elif target_path.exists() and target_path.is_dir():
            test_cases = TestCaseLoader.load_from_directory(target_path, "*.yaml")
            if verbose:
                console.print(f"[dim]📁 Loading test cases from: {path}[/dim]\n")
        else:
            console.print(f"[red]❌ Path not found: {path}[/red]")
            return
    elif (pattern_path := Path(pattern)).exists() and pattern_path.is_file():
        try:
            test_cases = [TestCaseLoader.load_from_file(pattern_path)]
            if verbose:
                console.print(f"[dim]📄 Loading test case from: {pattern}[/dim]\n")
        except Exception as e:
            console.print(f"[red]❌ Failed to load test case: {e}[/red]")
            return
    elif pattern_path.exists() and pattern_path.is_dir():
        test_cases = TestCaseLoader.load_from_directory(pattern_path, "*.yaml")
        if verbose:
            console.print(f"[dim]📁 Loading test cases from: {pattern}[/dim]\n")
    else:
        test_cases_dir = Path("tests/test-cases")
        if not test_cases_dir.exists():
            console.print("[red]❌ Test cases directory not found: tests/test-cases[/red]")
            console.print("[dim]Tip: You can specify a path or file directly:[/dim]")
            console.print("[dim]  evalview run examples/anthropic[/dim]")
            console.print("[dim]  evalview run path/to/test-case.yaml[/dim]")
            return
        test_cases = TestCaseLoader.load_from_directory(test_cases_dir, pattern)

    if not test_cases:
        console.print(f"[yellow]⚠️  No test cases found matching pattern: {pattern}[/yellow]\n")
        console.print("[bold]💡 Create tests by:[/bold]")
        console.print("   • [cyan]evalview record --interactive[/cyan]   (record agent interactions)")
        console.print("   • [cyan]evalview expand <test.yaml>[/cyan]     (generate variations from seed)")
        console.print("   • Or create YAML files manually in tests/test-cases/")
        console.print()
        console.print("[dim]Example: evalview record → evalview expand recorded-001.yaml --count 50[/dim]")
        return

    # Filter by difficulty if specified
    if difficulty_filter:
        original_count = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.difficulty == difficulty_filter]
        if not test_cases:
            console.print(f"[yellow]⚠️  No test cases with difficulty '{difficulty_filter}' found[/yellow]")
            console.print(f"[dim]Original count: {original_count} tests[/dim]")
            return
        if verbose:
            console.print(f"[dim]🎯 Filtered to {len(test_cases)}/{original_count} tests with difficulty: {difficulty_filter}[/dim]\n")

    # ── Test quality check ───────────────────────────────────────────────────
    from evalview.core.test_quality import score_test_quality, QUALITY_THRESHOLD
    skipped: List[str] = []
    qualified_cases: List[Any] = []
    for tc in test_cases:
        q_score, q_issues = score_test_quality(tc)
        if q_score < QUALITY_THRESHOLD:
            if tc.generated:
                skipped.append(tc.name)
                console.print(f"[yellow]⚠  {tc.name} skipped [generated, quality: {q_score}/100][/yellow]")
                for issue in q_issues:
                    console.print(f"[dim]     • {issue}[/dim]")
            else:
                console.print(f"[yellow]⚠  {tc.name} [quality: {q_score}/100] — score may reflect test issues, not agent issues[/yellow]")
                for issue in q_issues:
                    console.print(f"[dim]     • {issue}[/dim]")
                qualified_cases.append(tc)
        else:
            qualified_cases.append(tc)

    if skipped:
        console.print(f"[dim]   {len(skipped)} auto-generated test(s) skipped. Fix and re-run, or rewrite manually.[/dim]\n")
    test_cases = qualified_cases
    if not test_cases:
        console.print("[yellow]⚠️  No runnable tests. Fix test quality issues above and re-run.[/yellow]")
        return

    # Inject variance config for --runs flag (enables statistical/pass@k mode)
    if runs is not None:
        if runs < 2:
            console.print("[red]❌ --runs must be at least 2 for statistical mode[/red]")
            return
        if runs > 100:
            console.print("[red]❌ --runs cannot exceed 100[/red]")
            return

        from evalview.core.types import VarianceConfig
        cli_variance_config = VarianceConfig(
            runs=runs,
            pass_rate=pass_rate,
        )
        for tc in test_cases:
            tc.thresholds.variance = cli_variance_config

        console.print(f"[cyan]📊 Statistical mode: Running each test {runs} times (pass rate: {pass_rate:.0%})[/cyan]\n")

    # Interactive test selection menu - show when no explicit filter provided
    if pattern == "*.yaml" and not test and not filter and sys.stdin.isatty():
        tests_by_adapter: Dict[str, List[Any]] = {}
        for tc in test_cases:
            adapter_name = tc.adapter or config.get("adapter", "http")
            if adapter_name not in tests_by_adapter:
                tests_by_adapter[adapter_name] = []
            tests_by_adapter[adapter_name].append(tc)

        adapter_endpoints: Dict[str, str] = {}
        for adapter_name, adapter_tests in tests_by_adapter.items():
            for tc in adapter_tests:
                if tc.endpoint:
                    adapter_endpoints[adapter_name] = tc.endpoint
                    break
            if adapter_name not in adapter_endpoints:
                adapter_endpoints[adapter_name] = config.get("endpoint", "")

        def check_health_sync(endpoint: str) -> bool:
            """Quick health check - test if port is open."""
            if not endpoint:
                return False
            try:
                from urllib.parse import urlparse
                import socket
                parsed = urlparse(endpoint)
                host = parsed.hostname or "localhost"
                port = parsed.port or 80
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            except Exception:
                return False

        adapter_health: Dict[str, bool] = {}
        for adapter_name, endpoint in adapter_endpoints.items():
            adapter_health[adapter_name] = check_health_sync(endpoint)

        if len(tests_by_adapter) > 1:
            console.print("[bold]📋 Test cases found:[/bold]\n")

            menu_options: List[Any] = []
            for i, (adapter_name, adapter_tests) in enumerate(tests_by_adapter.items(), 1):
                health_status = "[green]✅[/green]" if adapter_health.get(adapter_name) else "[red]❌[/red]"
                endpoint = adapter_endpoints.get(adapter_name, "N/A")
                console.print(f"  [{i}] [bold]{adapter_name.upper()}[/bold] ({len(adapter_tests)} tests) {health_status}")
                console.print(f"      Endpoint: {endpoint}")
                for tc in adapter_tests[:3]:
                    console.print(f"        • {tc.name}")
                if len(adapter_tests) > 3:
                    console.print(f"        • ... and {len(adapter_tests) - 3} more")
                console.print()
                menu_options.append((adapter_name, adapter_tests))

            console.print(f"  [{len(menu_options) + 1}] [bold]All tests[/bold] ({len(test_cases)} tests)")
            console.print()

            choice = click.prompt(
                "Which tests to run?",
                type=int,
                default=len(menu_options) + 1,
            )

            if 1 <= choice <= len(menu_options):
                selected_adapter, test_cases = menu_options[choice - 1]
                console.print(f"\n[cyan]Running {selected_adapter.upper()} tests...[/cyan]")
            elif choice == len(menu_options) + 1:
                console.print("\n[cyan]Running all tests...[/cyan]")
            else:
                console.print("[yellow]Invalid choice. Running all tests.[/yellow]")

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
                generate_html = click.confirm("Generate HTML report?", default=True)
                if generate_html:
                    html_report = f".evalview/results/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    console.print(f"[dim]📊 HTML report will be saved to: {html_report}[/dim]\n")

    # Filter test cases by name if --test or --filter specified
    if test or filter:
        original_count = len(test_cases)
        filtered_cases: List[Any] = []

        for test_case in test_cases:
            if test:
                test_name_lower = test_case.name.lower()
                if any(t.lower() == test_name_lower for t in test):
                    filtered_cases.append(test_case)
                    continue

            if filter:
                filter_lower = filter.lower()
                test_name_lower = test_case.name.lower()

                if "*" in filter or "?" in filter:
                    if fnmatch.fnmatch(test_name_lower, filter_lower):
                        filtered_cases.append(test_case)
                        continue
                elif filter_lower in test_name_lower:
                    filtered_cases.append(test_case)
                    continue

        test_cases = filtered_cases

        if not test_cases:
            console.print("[yellow]⚠️  No test cases matched the filter criteria[/yellow]")
            return

        if verbose:
            console.print(f"[dim]Filtered {original_count} → {len(test_cases)} test(s)[/dim]\n")

    console.print(f"Found {len(test_cases)} test case(s)\n")

    # Helper function to get adapter for a test case
    def get_adapter_for_test(test_case: Any) -> Any:
        """Get adapter for test case - use test-specific if specified, otherwise global."""
        if test_case.adapter and (test_case.endpoint or test_case.adapter in ["openai-assistants", "goose"]):
            test_adapter_type = test_case.adapter
            test_endpoint = test_case.endpoint
            test_config: Dict[str, Any] = test_case.adapter_config or {}

            if verbose:
                console.print(f"[dim]  Using test-specific adapter: {test_adapter_type} @ {test_endpoint}[/dim]")

            if test_adapter_type == "langgraph":
                return LangGraphAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 30.0),
                    streaming=test_config.get("streaming", False),
                    verbose=verbose,
                    model_config=model_config,
                    assistant_id=test_config.get("assistant_id", "agent"),
                    allow_private_urls=allow_private_urls,
                )
            elif test_adapter_type == "crewai":
                merged_model_config = {**model_config, **test_config}
                return CrewAIAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 120.0),
                    verbose=verbose,
                    model_config=merged_model_config,
                    allow_private_urls=allow_private_urls,
                )
            elif test_adapter_type == "openai-assistants":
                return OpenAIAssistantsAdapter(
                    assistant_id=test_config.get("assistant_id"),
                    timeout=test_config.get("timeout", 120.0),
                    verbose=verbose,
                    model_config=model_config,
                )
            elif test_adapter_type == "tapescope":
                return TapeScopeAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 120.0),
                    verbose=verbose,
                    model_config=model_config,
                    allow_private_urls=allow_private_urls,
                )
            elif test_adapter_type == "mcp":
                from evalview.adapters.mcp_adapter import MCPAdapter
                return MCPAdapter(
                    endpoint=test_endpoint,
                    timeout=test_config.get("timeout", 30.0),
                )
            elif test_adapter_type == "goose":
                from evalview.adapters.goose_adapter import GooseAdapter
                return GooseAdapter(
                    timeout=test_config.get("timeout", 300.0),
                    cwd=test_case.input.context.get("cwd") if test_case.input.context else None,
                    extensions=test_case.input.context.get("extensions") if test_case.input.context else None,
                    provider=test_config.get("provider"),
                    model=test_config.get("model"),
                )
            elif test_adapter_type == "cohere":
                from evalview.adapters.cohere_adapter import CohereAdapter
                cohere_model = model_config.get("name") if isinstance(model_config, dict) else model_config
                return CohereAdapter(model=cohere_model)
            else:
                return HTTPAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 30.0),
                    model_config=model_config,
                    allow_private_urls=allow_private_urls,
                )

        if adapter is None:
            console.print(f"[red]❌ No adapter configured for test: {test_case.name}[/red]")
            console.print("[dim]Either add adapter/endpoint to the test case YAML, or create .evalview/config.yaml[/dim]")
            console.print("[dim]Example in test case:[/dim]")
            console.print("[dim]  adapter: http[/dim]")
            console.print("[dim]  endpoint: http://localhost:8000[/dim]")
            raise ValueError(f"No adapter for test: {test_case.name}")
        return adapter

    # Initialize statistical evaluator and console reporter
    statistical_evaluator = StatisticalEvaluator()
    stats_reporter = ConsoleReporter()

    async def execute_single_test(test_case: Any) -> Any:
        """Execute a single test case with optional retry logic."""
        test_adapter = get_adapter_for_test(test_case)

        context = dict(test_case.input.context) if test_case.input.context else {}
        if hasattr(test_case, 'tools') and test_case.tools:
            context['tools'] = test_case.tools

        async def _execute() -> Any:
            return await test_adapter.execute(test_case.input.query, context)

        # ── Multi-turn execution ─────────────────────────────────────────────
        if test_case.is_multi_turn:
            if verbose:
                console.print(f"[dim]  ↳ multi-turn ({len(test_case.turns)} turns)[/dim]")
            trace = await _execute_multi_turn_trace(test_case, test_adapter)
            adapter_name = getattr(test_adapter, "name", None)
            result = await evaluator.evaluate(test_case, trace, adapter_name=adapter_name)
            if tracker:
                if track:
                    tracker.store_result(result)
                if compare_baseline:
                    regression_reports[test_case.name] = tracker.compare_to_baseline(result)
            return (result.passed, result)

        # Statistical mode
        if is_statistical_mode(test_case):
            variance_config = test_case.thresholds.variance
            num_runs = variance_config.runs
            console.print(f"\n[cyan]📊 Statistical mode: Running {test_case.name} {num_runs} times...[/cyan]")

            individual_results: List[Any] = []
            for run_idx in range(num_runs):
                try:
                    if retry_config.max_retries > 0:
                        retry_result = await with_retry(
                            _execute,
                            retry_config,
                            on_retry=lambda attempt, delay, exc: None,
                        )
                        if not retry_result.success:
                            console.print(f"  [red]Run {run_idx + 1}/{num_runs}: ERROR[/red]")
                            continue
                        trace = retry_result.result
                    else:
                        trace = await _execute()

                    adapter_name = getattr(test_adapter, 'name', None)
                    result = await evaluator.evaluate(test_case, trace, adapter_name=adapter_name)
                    individual_results.append(result)

                    status = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
                    console.print(f"  Run {run_idx + 1}/{num_runs}: {status} score={result.score:.1f}")

                except Exception as e:
                    console.print(f"  [red]Run {run_idx + 1}/{num_runs}: ERROR - {str(e)[:50]}[/red]")

            if not individual_results:
                raise ValueError(f"All {num_runs} runs failed for {test_case.name}")

            stat_result = statistical_evaluator.evaluate_from_results(
                test_case, individual_results, variance_config
            )

            stats_reporter.print_statistical_summary(stat_result, show_individual_runs=verbose)

            best_result = individual_results[0]
            best_result.passed = stat_result.passed
            best_result.score = stat_result.score_stats.mean

            return (stat_result.passed, best_result)

        # Standard single-run execution
        if retry_config.max_retries > 0:
            retry_result = await with_retry(
                _execute,
                retry_config,
                on_retry=lambda attempt, delay, exc: console.print(
                    f"[yellow]  ↻ Retry {attempt}/{retry_config.max_retries} for {test_case.name} after {delay:.1f}s ({type(exc).__name__})[/yellow]"
                ) if verbose else None,
            )
            if not retry_result.success:
                exc = retry_result.exception
                raise exc if exc is not None else RuntimeError("Test execution failed")
            trace = retry_result.result
            if trace_reporter:
                trace_reporter.report_from_execution_trace(trace, test_case.name)
        else:
            trace = await _execute()
            if trace_reporter:
                trace_reporter.report_from_execution_trace(trace, test_case.name)

        # Show debug information if enabled
        if debug:
            console.print(f"\n[cyan]{'─' * 60}[/cyan]")
            console.print(f"[cyan]DEBUG: {test_case.name}[/cyan]")
            console.print(f"[cyan]{'─' * 60}[/cyan]\n")

            if hasattr(test_adapter, '_last_raw_response') and test_adapter._last_raw_response:
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
                console.print(f"    [{i+1}] {step.tool_name}")
                console.print(f"        params: {str(step.parameters)[:100]}")
                console.print(f"        metrics: latency={step.metrics.latency:.1f}ms, cost=${step.metrics.cost:.4f}")
                if step.metrics.tokens:
                    console.print(f"        tokens: in={step.metrics.tokens.input_tokens}, out={step.metrics.tokens.output_tokens}")
            console.print(f"  Final Output: {trace.final_output[:200]}{'...' if len(trace.final_output) > 200 else ''}")
            console.print()
            console.print("[bold]Aggregated Metrics:[/bold]")
            console.print(f"  Total Cost: ${trace.metrics.total_cost:.4f}")
            console.print(f"  Total Latency: {trace.metrics.total_latency:.0f}ms")
            if trace.metrics.total_tokens:
                console.print(f"  Total Tokens: in={trace.metrics.total_tokens.input_tokens}, out={trace.metrics.total_tokens.output_tokens}, cached={trace.metrics.total_tokens.cached_tokens}")
            console.print()

        adapter_name = getattr(test_adapter, 'name', None)
        result = await evaluator.evaluate(test_case, trace, adapter_name=adapter_name)

        if tracker:
            if track:
                tracker.store_result(result)
            if compare_baseline:
                regression_report = tracker.compare_to_baseline(result)
                regression_reports[test_case.name] = regression_report

        return (result.passed, result)

    # Run evaluations
    results: List[Any] = []
    passed = 0
    failed = 0
    execution_errors = 0
    start_time = 0.0

    if sequential:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for test_case in test_cases:
                task = progress.add_task(f"Running {test_case.name}...", total=None)

                try:
                    test_passed, result = await execute_single_test(test_case)
                    results.append(result)

                    if test_passed:
                        passed += 1
                        progress.update(task, description=f"[green]✅ {test_case.name} - PASSED (score: {result.score})[/green]")
                    else:
                        failed += 1
                        progress.update(task, description=f"[red]❌ {test_case.name} - FAILED (score: {result.score})[/red]")

                except Exception as e:
                    import httpx as _httpx
                    execution_errors += 1
                    error_msg = str(e)
                    if isinstance(e, _httpx.ConnectError):
                        error_msg = f"Cannot connect to {config['endpoint']}"
                    elif isinstance(e, _httpx.TimeoutException):
                        error_msg = "Request timeout"
                    progress.update(task, description=f"[red]⚠ {test_case.name} - EXECUTION ERROR: {error_msg}[/red]")

                progress.remove_task(task)
    else:
        def on_start(test_name: str) -> None:
            if verbose:
                console.print(f"[dim]  ▶ Starting: {test_name}[/dim]")

        def on_complete(test_name: str, test_passed: bool, result: Any) -> None:
            nonlocal passed, failed
            if test_passed:
                passed += 1
                console.print(f"[green]✅ {test_name} - PASSED (score: {result.score})[/green]")
            else:
                failed += 1
                console.print(f"[red]❌ {test_name} - FAILED (score: {result.score})[/red]")

        def on_error(test_name: str, exc: Exception) -> None:
            nonlocal execution_errors
            import httpx as _httpx
            execution_errors += 1
            error_msg = str(exc)
            if isinstance(exc, _httpx.ConnectError):
                error_msg = f"Cannot connect to {config['endpoint']}"
            elif isinstance(exc, _httpx.TimeoutException):
                error_msg = "Request timeout"
            console.print(f"[red]⚠ {test_name} - EXECUTION ERROR: {error_msg}[/red]")

        console.print(f"[dim]Executing {len(test_cases)} tests with up to {max_workers} parallel workers...[/dim]\n")

        import time as time_module

        # Reset judge cost tracker for this run
        judge_cost_tracker.reset()

        start_time = time_module.time()
        tests_running: set = set()
        tests_completed = 0

        def format_elapsed() -> str:
            elapsed = time_module.time() - start_time
            mins, secs = divmod(elapsed, 60)
            secs_int = int(secs)
            ms = int((secs - secs_int) * 1000)
            return f"{int(mins):02d}:{secs_int:02d}.{ms:03d}"

        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0

        def get_status_display() -> Panel:
            nonlocal spinner_idx
            elapsed_str = format_elapsed()
            spinner = spinner_frames[spinner_idx % len(spinner_frames)]
            spinner_idx += 1

            running_tests = [*tests_running][:3]
            if running_tests:
                running_lines = "\n".join([f"  [yellow]{spinner}[/yellow] [dim]{t}...[/dim]" for t in running_tests])
            else:
                running_lines = f"  [yellow]{spinner}[/yellow] [dim]Starting tests...[/dim]"

            if failed > 0:
                status = "[bold red]● Running[/bold red]"
            else:
                status = "[green]● Running[/green]"

            judge_cost = judge_cost_tracker.get_summary()

            content = (
                f"  {status}\n"
                f"\n"
                f"  [bold]⏱️  Elapsed:[/bold]    [yellow]{elapsed_str}[/yellow]\n"
                f"  [bold]📋 Progress:[/bold]   {tests_completed}/{len(test_cases)} tests\n"
                f"  [bold]💰 Judge:[/bold]      [dim]{judge_cost}[/dim]\n"
                f"\n"
                f"{running_lines}\n"
                f"\n"
                f"  [green]✓ Passed:[/green] {passed}    [red]✗ Failed:[/red] {failed}"
            )

            border = "red" if failed > 0 else "cyan"
            return Panel(
                content,
                title="[bold]Test Execution[/bold]",
                border_style=border,
                padding=(0, 1),
            )

        def on_start_with_tracking(test_name: str) -> None:
            nonlocal tests_running
            tests_running.add(test_name[:30])
            on_start(test_name)

        def on_complete_with_tracking(test_name: str, test_passed: bool, result: Any) -> None:
            nonlocal tests_running, tests_completed
            tests_running.discard(test_name[:30])
            tests_completed += 1
            on_complete(test_name, test_passed, result)

        def on_error_with_tracking(test_name: str, exc: Exception) -> None:
            nonlocal tests_running, tests_completed
            tests_running.discard(test_name[:30])
            tests_completed += 1
            on_error(test_name, exc)

        if sys.stdin.isatty():
            with Live(get_status_display(), console=console, refresh_per_second=10) as live:
                async def update_display() -> None:
                    while tests_completed < len(test_cases):
                        live.update(get_status_display())
                        await asyncio.sleep(0.1)
                    live.update(get_status_display())

                parallel_task = execute_tests_parallel(
                    test_cases,
                    execute_single_test,
                    max_workers=max_workers,
                    on_start=on_start_with_tracking,
                    on_complete=on_complete_with_tracking,
                    on_error=on_error_with_tracking,
                )
                display_task = update_display()

                parallel_results, _ = await asyncio.gather(parallel_task, display_task, return_exceptions=True)

            final_elapsed = format_elapsed()
            final_judge_cost = judge_cost_tracker.get_summary()
            console.print()
            console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            if execution_errors > 0:
                console.print("[bold cyan]║[/bold cyan]  [bold red]⚠ EXECUTION ERRORS OCCURRED[/bold red]                                  [bold cyan]║[/bold cyan]")
            elif failed == 0:
                console.print("[bold cyan]║[/bold cyan]  [bold green]✓ AGENT HEALTHY[/bold green]                                               [bold cyan]║[/bold cyan]")
            else:
                console.print("[bold cyan]║[/bold cyan]  [bold red]✗ REGRESSION DETECTED[/bold red]                                        [bold cyan]║[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            if execution_errors > 0:
                console.print(f"[bold cyan]║[/bold cyan]  [green]✓ Passed:[/green] {passed:<4}  [red]✗ Failed:[/red] {failed:<4}  [red]⚠ Errors:[/red] {execution_errors:<4}         [bold cyan]║[/bold cyan]")
            else:
                console.print(f"[bold cyan]║[/bold cyan]  [green]✓ Passed:[/green] {passed:<4}  [red]✗ Failed:[/red] {failed:<4}  [dim]Time:[/dim] {final_elapsed}               [bold cyan]║[/bold cyan]")
            console.print(f"[bold cyan]║[/bold cyan]  [dim]💰 Judge cost:[/dim] {final_judge_cost:<45}[bold cyan]║[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
            console.print()
        else:
            parallel_results = await execute_tests_parallel(
                test_cases,
                execute_single_test,
                max_workers=max_workers,
                on_start=on_start,
                on_complete=on_complete,
                on_error=on_error,
            )

        if isinstance(parallel_results, BaseException):
            logger.error(f"parallel_results is an exception: {parallel_results}")
            console.print(f"[red]Error in parallel execution: {parallel_results}[/red]")
        elif parallel_results:
            for pr in list(parallel_results):  # type: ignore[arg-type]
                if pr.success and pr.result:
                    results.append(pr.result)

    # Print judge cache stats if cache was used
    if _judge_cache is not None:
        cs = _judge_cache.stats()
        if cs["total"] > 0:
            console.print(f"  [dim]Judge cache: {cs['hits']} hits / {cs['total']} lookups ({cs['hit_rate']:.0%} hit rate)[/dim]")

    # Print summary
    console.print()
    reporter = ConsoleReporter()
    if summary:
        suite_name = None
        if path:
            suite_name = Path(path).name if Path(path).is_dir() else Path(path).stem

        previous_results = None
        output_dir = Path(output)
        if output_dir.exists():
            previous_results = JSONReporter.get_latest_results(output_dir)

        reporter.print_compact_summary(results, suite_name=suite_name, previous_results=previous_results)
    else:
        reporter.print_summary(results)

    # Print behavior coverage report if enabled
    if coverage:
        suite_name = None
        if path:
            suite_name = Path(path).name if Path(path).is_dir() else Path(path).stem
        reporter.print_coverage_report(test_cases, results, suite_name=suite_name)

    # Print regression analysis if enabled
    if compare_baseline and regression_reports:
        console.print()
        console.print("[bold cyan]📊 Regression Analysis[/bold cyan]")
        console.print("━" * 60)
        console.print()

        any_regressions = False
        for test_name, report in regression_reports.items():
            if report.baseline_score is None:
                continue

            if report.is_regression:
                any_regressions = True
                if report.severity == "critical":
                    status = "[red]🔴 CRITICAL REGRESSION[/red]"
                elif report.severity == "moderate":
                    status = "[yellow]🟡 MODERATE REGRESSION[/yellow]"
                else:
                    status = "[yellow]🟠 MINOR REGRESSION[/yellow]"
            else:
                status = "[green]✅ No regression[/green]"

            console.print(f"[bold]{test_name}[/bold]: {status}")

            if report.score_delta is not None:
                delta_str = f"{report.score_delta:+.1f}"
                percent_str = f"({report.score_delta_percent:+.1f}%)"
                if report.score_delta < 0:
                    console.print(
                        f"  Score: {report.current_score:.1f} [red]↓ {delta_str}[/red] {percent_str} vs baseline {report.baseline_score:.1f}"
                    )
                else:
                    console.print(
                        f"  Score: {report.current_score:.1f} [green]↑ {delta_str}[/green] {percent_str} vs baseline {report.baseline_score:.1f}"
                    )

            if report.cost_delta is not None and report.cost_delta_percent is not None:
                delta_str = f"${report.cost_delta:+.4f}"
                percent_str = f"({report.cost_delta_percent:+.1f}%)"
                if report.cost_delta_percent > 20:
                    console.print(f"  Cost: ${report.current_cost:.4f} [red]↑ {delta_str}[/red] {percent_str}")
                else:
                    console.print(f"  Cost: ${report.current_cost:.4f} {delta_str} {percent_str}")

            if report.latency_delta is not None and report.latency_delta_percent is not None:
                delta_str = f"{report.latency_delta:+.0f}ms"
                percent_str = f"({report.latency_delta_percent:+.1f}%)"
                if report.latency_delta_percent > 30:
                    console.print(f"  Latency: {report.current_latency:.0f}ms [red]↑ {delta_str}[/red] {percent_str}")
                else:
                    console.print(f"  Latency: {report.current_latency:.0f}ms {delta_str} {percent_str}")

            if report.is_regression and report.issues:
                console.print(f"  Issues: {', '.join(report.issues)}")

            console.print()

        if any_regressions:
            console.print("[red]⚠️  Regressions detected! Review changes before deploying.[/red]\n")

    # Save results
    output_dir_path = Path(output)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    results_file = output_dir_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    JSONReporter.save(results, results_file)

    console.print(f"\n[dim]Results saved to: {results_file}[/dim]\n")

    # Auto-save golden baseline if --save-golden and all tests passed cleanly
    if save_golden and failed == 0 and execution_errors == 0 and results:
        try:
            from evalview.core.golden import GoldenStore
            store = GoldenStore()
            saved_count = 0
            for result in results:
                if result and result.score > 0:
                    store.save_golden(result, notes="Auto-saved via --save-golden", source_file=str(results_file))
                    saved_count += 1
            if saved_count > 0:
                console.print(f"[green]Golden baseline saved for {saved_count} test{'s' if saved_count != 1 else ''}.[/green]")
                console.print("[dim]Future runs with --diff will compare against this baseline.[/dim]\n")
        except Exception as e:
            console.print(f"[yellow]Could not save golden baseline: {e}[/yellow]\n")

    # Initialize for diff tracking
    diffs_found: List[Any] = []

    # --- Golden Diff Display ---
    if diff and results:
        from evalview.core.golden import GoldenStore
        from evalview.core.diff import compare_to_golden, DiffStatus

        store = GoldenStore()

        for result in results:
            golden = store.load_golden(result.test_case)
            if golden:
                trace_diff = compare_to_golden(golden, result.trace, result.score)
                if trace_diff.has_differences:
                    diffs_found.append((result.test_case, trace_diff))

        if diffs_found:
            console.print("\n[bold cyan]━━━ Golden Diff Report ━━━[/bold cyan]\n")

            for test_name, trace_diff in diffs_found:
                status = trace_diff.overall_severity
                if status == DiffStatus.REGRESSION:
                    icon = "[red]✗ REGRESSION[/red]"
                elif status == DiffStatus.TOOLS_CHANGED:
                    icon = "[yellow]⚠ TOOLS_CHANGED[/yellow]"
                elif status == DiffStatus.OUTPUT_CHANGED:
                    icon = "[dim]~ OUTPUT_CHANGED[/dim]"
                else:
                    icon = "[green]✓ PASSED[/green]"

                console.print(f"{icon} [bold]{test_name}[/bold]")
                console.print(f"    Summary: {trace_diff.summary()}")

                if trace_diff.tool_diffs:
                    console.print("    [bold]Tool Changes:[/bold]")
                    for td in trace_diff.tool_diffs[:5]:
                        if td.type == "added":
                            console.print(f"      [green]+ {td.actual_tool}[/green] (new step)")
                        elif td.type == "removed":
                            console.print(f"      [red]- {td.golden_tool}[/red] (missing)")
                        elif td.type == "changed":
                            if td.golden_tool == td.actual_tool and td.parameter_diffs:
                                console.print(f"      [yellow]~ {td.golden_tool}[/yellow] (parameters changed)")
                            else:
                                console.print(f"      [yellow]~ {td.golden_tool} -> {td.actual_tool}[/yellow]")

                        if td.parameter_diffs:
                            console.print("        [dim]Parameter differences:[/dim]")
                            for pd in td.parameter_diffs[:10]:
                                if pd.diff_type == "missing":
                                    console.print(f"          [red]- {pd.param_name}[/red]: {pd.golden_value}")
                                elif pd.diff_type == "added":
                                    console.print(f"          [green]+ {pd.param_name}[/green]: {pd.actual_value}")
                                elif pd.diff_type == "type_changed":
                                    console.print(f"          [yellow]~ {pd.param_name}[/yellow]: type changed")
                                    console.print(f"            golden: {type(pd.golden_value).__name__} = {pd.golden_value}")
                                    console.print(f"            actual: {type(pd.actual_value).__name__} = {pd.actual_value}")
                                elif pd.diff_type == "value_changed":
                                    sim_str = ""
                                    if pd.similarity is not None:
                                        sim_pct = int(pd.similarity * 100)
                                        sim_str = f" (similarity: {sim_pct}%)"
                                    console.print(f"          [yellow]~ {pd.param_name}[/yellow]:{sim_str}")
                                    console.print(f"            [dim]golden:[/dim] {pd.golden_value}")
                                    console.print(f"            [dim]actual:[/dim] {pd.actual_value}")

                if abs(trace_diff.score_diff) > 1:
                    direction = "[green]↑[/green]" if trace_diff.score_diff > 0 else "[red]↓[/red]"
                    console.print(f"    Score: {direction} {trace_diff.score_diff:+.1f}")

                console.print()

            regressions = sum(1 for _, d in diffs_found if d.overall_severity == DiffStatus.REGRESSION)
            tools_changed = sum(1 for _, d in diffs_found if d.overall_severity == DiffStatus.TOOLS_CHANGED)
            output_changed = sum(1 for _, d in diffs_found if d.overall_severity == DiffStatus.OUTPUT_CHANGED)

            if regressions > 0:
                console.print(f"[red]✗ {regressions} REGRESSION(s) - score dropped, fix before deploy[/red]")
                console.print()
                console.print("[dim]⭐ EvalView caught this before prod! Star → github.com/hidai25/eval-view[/dim]\n")
            elif tools_changed > 0:
                console.print(f"[yellow]⚠ {tools_changed} TOOLS_CHANGED - agent behavior shifted, review before deploy[/yellow]")
                console.print()
                console.print("[dim]⭐ EvalView caught this! Star → github.com/hidai25/eval-view[/dim]\n")
            elif output_changed > 0:
                console.print(f"[dim]~ {output_changed} OUTPUT_CHANGED - response changed, review before deploy[/dim]\n")
        else:
            goldens = store.list_golden()
            matched = sum(1 for g in goldens if any(r.test_case == g.test_name for r in results))
            if matched > 0:
                console.print(f"[green]✓ PASSED - No differences from golden baseline ({matched} tests compared)[/green]\n")
            elif goldens:
                console.print("[yellow]No golden traces match these tests[/yellow]")
                console.print("[dim]Save one with: evalview golden save " + str(results_file) + "[/dim]\n")
            else:
                console.print("[yellow]No golden traces found[/yellow]")
                console.print("[dim]Create baseline: evalview golden save " + str(results_file) + "[/dim]\n")

    # Generate HTML report if requested
    if html_report and results:
        try:
            from evalview.reporters.html_reporter import HTMLReporter
            html_reporter = HTMLReporter()
            html_path = html_reporter.generate(results, html_report)
            console.print("\n[bold green]📊 HTML Report Generated![/bold green]")
            console.print(f"   [link=file://{Path(html_path).absolute()}]{html_path}[/link]")
            console.print(f"   [dim]Open in browser: open {html_path}[/dim]\n")
        except ImportError as e:
            console.print(f"[yellow]⚠️  Could not generate HTML report: {e}[/yellow]")
            console.print("[dim]Install with: pip install jinja2 plotly[/dim]\n")

    # Generate HTML diff report if requested
    if diff_report and results:
        if not diff:
            console.print("[yellow]⚠️  --diff-report requires --diff flag[/yellow]")
            console.print("[dim]Usage: evalview run --diff --diff-report diff.html[/dim]\n")
        elif diffs_found:
            try:
                from evalview.reporters.html_reporter import DiffReporter
                diff_reporter = DiffReporter()
                diff_path = diff_reporter.generate(
                    diffs=[d for _, d in diffs_found],
                    results=results,
                    output_path=diff_report,
                )
                console.print("\n[bold cyan]📊 Diff Report Generated![/bold cyan]")
                console.print(f"   [link=file://{Path(diff_path).absolute()}]{diff_path}[/link]")
                console.print(f"   [dim]Open in browser: open {diff_path}[/dim]\n")
            except ImportError as e:
                console.print(f"[yellow]⚠️  Could not generate diff report: {e}[/yellow]")
                console.print("[dim]Install with: pip install jinja2[/dim]\n")
        else:
            console.print("[dim]No differences to report - all tests match golden baseline[/dim]\n")

    if track:
        console.print("[dim]📊 Results tracked for regression analysis[/dim]")
        console.print("[dim]   View trends: evalview trends[/dim]")
        console.print("[dim]   Set baseline: evalview baseline set[/dim]\n")

    # Auto-generate and open visual HTML report after every run.
    if not watch and results:
        in_ci = bool(os.environ.get("CI"))
        should_open = not no_open and not in_ci
        try:
            from evalview.visualization import generate_visual_report
            report_path = generate_visual_report(
                results,
                diffs=[d for _, d in diffs_found] if diffs_found else None,
                auto_open=should_open,
                title="EvalView Run Report",
            )
            if should_open:
                console.print(f"\n[bold]📊 Report opened in browser[/bold] [dim]({report_path})[/dim]\n")
            else:
                console.print(f"\n[dim]📊 Report saved: {report_path}[/dim]\n")
        except Exception as _report_err:
            console.print(f"[dim]⚠ Could not generate HTML report: {_report_err}[/dim]")

    # Quick tips
    if not watch and results:
        if not summary and not coverage:
            console.print("[dim]Quick views:  evalview run --summary | evalview run --coverage[/dim]")
        if diff:
            console.print("[dim]Compare runs: evalview view --run-id <id>[/dim]")
        console.print()

    # Guided conversion to snapshot workflow
    if not watch and not diff and results:
        from evalview.core.golden import GoldenStore
        from evalview.core.project_state import ProjectStateStore
        from evalview.core.celebrations import Celebrations

        store = GoldenStore()
        state_store = ProjectStateStore()

        goldens = store.list_golden()
        all_passed_flag = all(r.passed for r in results)

        if not goldens and all_passed_flag and not state_store.load().conversion_suggestion_shown:
            Celebrations.conversion_suggestion(len(results))
            state_store.mark_conversion_shown()

    # --- Exit Code Logic (for CI) ---
    if execution_errors > 0:
        exit_code = 2
    elif failed > 0:
        exit_code = 1
    else:
        exit_code = 0

    # Additional exit code logic for --diff mode
    if diff and diffs_found:
        from evalview.core.diff import DiffStatus

        fail_statuses: set = set()
        warn_statuses: set = set()
        valid_statuses = {"REGRESSION", "TOOLS_CHANGED", "OUTPUT_CHANGED", "PASSED", "CONTRACT_DRIFT"}

        for s in (fail_on or "").upper().split(","):
            s = s.strip()
            if not s:
                continue
            if s in valid_statuses:
                fail_statuses.add(DiffStatus[s])
            else:
                console.print(f"[yellow]Warning: Unknown status '{s}' in --fail-on (valid: {', '.join(valid_statuses)})[/yellow]")

        for s in (warn_on or "").upper().split(","):
            s = s.strip()
            if not s:
                continue
            if s in valid_statuses:
                warn_statuses.add(DiffStatus[s])
            else:
                console.print(f"[yellow]Warning: Unknown status '{s}' in --warn-on (valid: {', '.join(valid_statuses)})[/yellow]")

        fail_count = 0
        warn_count = 0
        status_counts: Dict[Any, int] = {}

        for _, trace_diff in diffs_found:
            diff_status = trace_diff.overall_severity
            status_counts[diff_status] = status_counts.get(diff_status, 0) + 1
            if diff_status in fail_statuses:
                fail_count += 1
            elif diff_status in warn_statuses:
                warn_count += 1

        if fail_count > 0 or warn_count > 0:
            console.print("[bold]━━━ CI Summary ━━━[/bold]")
            for diff_status, count in sorted(status_counts.items(), key=lambda x: x[0].value):
                if diff_status in fail_statuses:
                    console.print(f"  [red]✗ {count} {diff_status.value.upper()}[/red] [dim][FAIL][/dim]")
                elif diff_status in warn_statuses:
                    console.print(f"  [yellow]⚠ {count} {diff_status.value.upper()}[/yellow] [dim][WARN][/dim]")
                else:
                    console.print(f"  [green]✓ {count} {diff_status.value.upper()}[/green]")

            if fail_count > 0:
                exit_code = max(exit_code, 1)
                console.print(f"\n[bold red]Exit: {exit_code}[/bold red] ({fail_count} failure(s) in fail_on set)\n")
            else:
                console.print(f"\n[bold green]Exit: {exit_code}[/bold green] ({warn_count} warning(s) only)\n")

    # Trust-framing summary
    if not watch:
        console.print("[dim]━" * 50 + "[/dim]")
        if execution_errors > 0:
            console.print(f"[bold yellow]{execution_errors} test{'s' if execution_errors != 1 else ''} could not run.[/bold yellow] Check network, timeouts, or agent availability.\n")
        elif failed == 0 and passed > 0:
            console.print(f"[bold green]Agent healthy.[/bold green] {passed}/{passed} checks passed. No regressions detected.\n")
            try:
                from evalview.core.golden import GoldenStore
                store = GoldenStore()
                has_any_golden = any(store.has_golden(r.test_case) for r in results if r)
                if not has_any_golden:
                    console.print("[dim]Tip: Save this as your baseline so future runs detect regressions:[/dim]")
                    console.print(f"[dim]   evalview golden save {results_file}[/dim]\n")
            except Exception:
                pass
        elif failed > 0:
            console.print(f"[bold red]Regression detected in {failed} test{'s' if failed != 1 else ''}.[/bold red] Review changes before shipping.\n")

    # Track run command telemetry
    try:
        import time as time_module
        duration_ms = (time_module.time() - start_time) * 1000 if start_time else 0.0
        track_run_command(
            adapter_type=adapter_type,
            test_count=len(test_cases),
            pass_count=passed,
            fail_count=failed,
            duration_ms=duration_ms,
            diff_mode=diff,
            watch_mode=watch,
            parallel=not sequential,
        )
    except Exception:
        pass

    # Watch mode: re-run tests on file changes
    if watch:
        from evalview.core.watcher import TestWatcher

        console.print("[cyan]━" * 60 + "[/cyan]")
        console.print("[cyan]👀 Watching for changes... (Ctrl+C to stop)[/cyan]")
        console.print("[cyan]━" * 60 + "[/cyan]\n")

        run_count = 1

        async def run_tests_again() -> None:
            nonlocal run_count
            run_count += 1
            console.print(f"\n[blue]━━━ Run #{run_count} ━━━[/blue]\n")

            await _run_async(
                path=path,
                pattern=pattern,
                test=test,
                filter=filter,
                output=output,
                verbose=verbose,
                track=track,
                compare_baseline=compare_baseline,
                debug=debug,
                sequential=sequential,
                max_workers=max_workers,
                max_retries=max_retries,
                retry_delay=retry_delay,
                watch=False,
                html_report=html_report,
                no_open=True,
            )

        watcher = TestWatcher(
            paths=["tests/test-cases", ".evalview"],
            run_callback=run_tests_again,  # type: ignore[arg-type]  # TestWatcher accepts async callbacks despite the sync annotation
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
            if trace_reporter:
                trace_reporter.close()
    else:
        if trace_reporter:
            trace_reporter.close()
        if exit_code != 0:
            sys.exit(exit_code)
