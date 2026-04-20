"""Shared state and helpers used across multiple command modules."""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx
import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv
from rich.console import Console

from evalview.core.adapter_factory import create_adapter
from evalview.core.types import ExecutionTrace, ExecutionMetrics, TokenUsage, TurnTrace

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult, TestCase
    from evalview.core.config import EvalViewConfig
    from evalview.core.diff import TraceDiff
    from evalview.core.golden import GoldenStore, GoldenTrace
    from evalview.core.drift_tracker import DriftTracker
    from evalview.adapters.base import AgentAdapter
    from evalview.core.budget import BudgetTracker

# Load environment variables (.env is the OSS standard, .env.local for overrides)
load_dotenv()
load_dotenv(dotenv_path=".env.local", override=True)

console = Console()


def _set_judge_env(resolved_model: str) -> None:
    """Set EVAL_MODEL and EVAL_PROVIDER env vars, validate API key exists."""
    import os
    import sys
    from evalview.core.llm_configs import PROVIDER_CONFIGS, LLMProvider

    os.environ["EVAL_MODEL"] = resolved_model

    model_lower = resolved_model.lower()
    provider_map = {
        "claude": (LLMProvider.ANTHROPIC, "ANTHROPIC_API_KEY"),
        "gpt-": (LLMProvider.OPENAI, "OPENAI_API_KEY"),
        "o1": (LLMProvider.OPENAI, "OPENAI_API_KEY"),
        "o3": (LLMProvider.OPENAI, "OPENAI_API_KEY"),
        "o4": (LLMProvider.OPENAI, "OPENAI_API_KEY"),
        "o5": (LLMProvider.OPENAI, "OPENAI_API_KEY"),
        "gemini": (LLMProvider.GEMINI, "GEMINI_API_KEY"),
        "grok": (LLMProvider.GROK, "XAI_API_KEY"),
        "deepseek": (LLMProvider.DEEPSEEK, "DEEPSEEK_API_KEY"),
    }

    for prefix, (provider, env_var) in provider_map.items():
        if model_lower.startswith(prefix):
            os.environ["EVAL_PROVIDER"] = provider.value
            if not os.environ.get(env_var):
                config = PROVIDER_CONFIGS[provider]
                console.print(
                    f"\n[red]Missing API key for {config.display_name}.[/red]\n"
                    f"Model [bold]{resolved_model}[/bold] requires [bold]{env_var}[/bold] to be set.\n\n"
                    f"[dim]Get your key at: {config.api_key_url}[/dim]\n"
                    f"[dim]Then: export {env_var}=your-key-here[/dim]\n"
                )
                sys.exit(1)
            break


def _save_judge_to_config(model: str) -> None:
    """Save the chosen judge model to .evalview/config.yaml so it persists."""
    config_path = Path(".evalview/config.yaml")
    data: Dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "judge" not in data:
        data["judge"] = {}
    data["judge"]["model"] = model
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def apply_judge_model(judge_model: Optional[str], interactive: bool = True) -> None:
    """Resolve judge model: --judge flag > config > interactive picker.

    On first use (no judge configured), shows an interactive model picker
    and saves the choice to .evalview/config.yaml. Subsequent runs use
    the saved config silently.

    Args:
        judge_model: Explicit --judge flag value (takes priority).
        interactive: If True, prompt user when no judge is configured.
    """
    import os
    from evalview.core.llm_configs import resolve_model_alias

    # 1. Explicit --judge flag
    if judge_model:
        resolved = resolve_model_alias(judge_model)
        _set_judge_env(resolved)
        return

    # 2. Already set via env var or config
    if os.environ.get("EVAL_MODEL"):
        return

    # 3. Check config.yaml for saved judge preference
    config_path = Path(".evalview/config.yaml")
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        judge_cfg = data.get("judge", {})
        if judge_cfg.get("model"):
            resolved = resolve_model_alias(judge_cfg["model"])
            _set_judge_env(resolved)
            return

    # 4. Interactive picker (first time only, not in CI)
    if not interactive or os.environ.get("CI"):
        return

    import click as _click
    from evalview.core.llm_configs import detect_available_providers
    from evalview.core.pricing import format_pricing_line

    try:
        available = detect_available_providers()
        available_set = {p.provider.value for p in available}
    except Exception:
        available_set = set()

    if not available_set:
        return  # No providers — will fall back to deterministic mode

    choices: List[Tuple[str, str, str]] = []  # (model_id, label, pricing)
    if "openai" in available_set:
        choices.append(("gpt-5.4-mini", "GPT-5.4 Mini", format_pricing_line("gpt-5.4-mini") or ""))
        choices.append(("gpt-5.4", "GPT-5.4", format_pricing_line("gpt-5.4") or ""))
    if "anthropic" in available_set:
        choices.append(("claude-haiku-4-5-20251001", "Claude Haiku 4.5", format_pricing_line("claude-haiku-4-5-20251001") or ""))
        choices.append(("claude-sonnet-4-6", "Claude Sonnet 4.6", format_pricing_line("claude-sonnet-4-6") or ""))
        choices.append(("claude-opus-4-6", "Claude Opus 4.6", format_pricing_line("claude-opus-4-6") or ""))
    if "gemini" in available_set:
        choices.append(("gemini-2.5-flash", "Gemini 2.5 Flash", format_pricing_line("gemini-2.5-flash") or ""))
    if "deepseek" in available_set:
        choices.append(("deepseek-chat", "DeepSeek Chat", format_pricing_line("deepseek-chat") or ""))
    if "ollama" in available_set:
        choices.append(("llama3.2", "Llama 3.2 (Ollama)", "free, local"))

    if not choices:
        return

    console.print("[bold]Which model should judge your agent's output quality?[/bold]\n")
    for i, (model_id, label, pricing) in enumerate(choices, 1):
        rec = "  [dim]<- recommended[/dim]" if i == 1 else ""
        pricing_str = f"  [dim]({pricing})[/dim]" if pricing else ""
        console.print(f"  [cyan]{i}.[/cyan] {label}{pricing_str}{rec}")
    console.print()

    raw = _click.prompt("Choice", default="1", show_default=False).strip()
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            chosen = choices[idx][0]
            _set_judge_env(chosen)
            _save_judge_to_config(chosen)
            console.print(f"[green]Using {choices[idx][1]} as judge.[/green]")
            console.print("[dim]Saved to .evalview/config.yaml — change anytime with --judge[/dim]\n")
            return
    except ValueError:
        pass

    # Invalid input — fall through to auto-detect
    console.print("[dim]Using default judge model.[/dim]\n")


def run_with_spinner(fn: Any, label: str, n_tests: int) -> Any:
    """Run a blocking function with a live spinner and elapsed timer.

    Args:
        fn: Zero-argument callable that returns the result.
        label: Action label (e.g. "Running", "Checking").
        n_tests: Number of tests for the status message.

    Returns:
        Whatever fn() returns.

    Raises:
        Whatever fn() raises — exceptions are captured and re-raised
        on the main thread.
    """
    import time as _time
    import threading
    from rich.live import Live

    result_holder: List[Any] = []
    error_holder: List[BaseException] = []
    start = _time.time()
    done = threading.Event()

    _frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _idx = [0]

    def _render() -> str:
        elapsed = _time.time() - start
        mins, secs = divmod(elapsed, 60)
        ms = int((secs - int(secs)) * 1000)
        ts = f"{int(mins):02d}:{int(secs):02d}.{ms:03d}"
        frame = _frames[_idx[0] % len(_frames)]
        _idx[0] += 1
        return f"  [yellow]{frame}[/yellow] {label} {n_tests} test{'s' if n_tests != 1 else ''}...  [dim]{ts}[/dim]"

    def _run() -> None:
        try:
            result_holder.append(fn())
        except BaseException as exc:
            error_holder.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    with Live(_render(), console=console, refresh_per_second=8, transient=True) as live:
        while not done.wait(timeout=0.125):
            live.update(_render())

    thread.join(timeout=10)

    if error_holder:
        raise error_holder[0]
    return result_holder[0]


def _create_adapter(adapter_type: str, endpoint: str, timeout: float = 30.0, allow_private_urls: bool = True) -> "AgentAdapter":
    """Factory function for creating adapters based on type."""
    return create_adapter(
        adapter_type=adapter_type,
        endpoint=endpoint,
        timeout=timeout,
        allow_private_urls=allow_private_urls,
    )


async def _execute_multi_turn_trace(test_case: TestCase, adapter: AgentAdapter) -> ExecutionTrace:
    """Execute all turns of a multi-turn test and return a merged ExecutionTrace."""
    conversation_history: List[Dict[str, Any]] = []
    all_steps: List[Any] = []
    turn_traces: List[Any] = []
    turn_summaries: List[TurnTrace] = []

    # 1. Iterate through each turn, execute, and collect traces
    if not test_case.turns:
        raise ValueError(f"Test case {test_case.name} has no turns defined.")

    for turn_index, turn in enumerate(test_case.turns):
        turn_context: Dict[str, Any] = dict(turn.context or {})
        if test_case.tools:
            turn_context.setdefault("tools", test_case.tools)
        if conversation_history:
            turn_context["conversation_history"] = list(conversation_history)

        trace = await adapter.execute(turn.query, turn_context)

        # Annotate each step with turn index for better traceability
        for step in trace.steps:
            step.turn_index = turn_index + 1
            step.turn_query = turn.query

        turn_traces.append(trace)
        all_steps.extend(trace.steps)
        turn_summaries.append(
            TurnTrace(
                index=turn_index + 1,
                query=turn.query,
                output=trace.final_output,
                tools=[
                    str(getattr(step, "tool_name", None) or getattr(step, "step_name", None) or "unknown")
                    for step in trace.steps
                ],
                latency_ms=float(getattr(trace.metrics, "total_latency", 0) or 0),
                cost=float(getattr(trace.metrics, "total_cost", 0) or 0),
            )
        )

        conversation_history.append({"role": "user", "content": turn.query})
        conversation_history.append({"role": "assistant", "content": trace.final_output})

    # 2. Merge metrics across turns
    total_cost = sum(t.metrics.total_cost for t in turn_traces)
    total_latency = sum(t.metrics.total_latency for t in turn_traces)

    merged_tokens: Optional[TokenUsage] = None
    if any(t.metrics.total_tokens for t in turn_traces):
        merged_tokens = TokenUsage(
            input_tokens=sum((t.metrics.total_tokens.input_tokens if t.metrics.total_tokens else 0) for t in turn_traces),
            output_tokens=sum((t.metrics.total_tokens.output_tokens if t.metrics.total_tokens else 0) for t in turn_traces),
            cached_tokens=sum((t.metrics.total_tokens.cached_tokens if t.metrics.total_tokens else 0) for t in turn_traces),
        )

    last_trace = turn_traces[-1]
    return ExecutionTrace(
        session_id=str(uuid.uuid4()),
        start_time=turn_traces[0].start_time,
        end_time=last_trace.end_time,
        steps=all_steps,
        final_output=last_trace.final_output,
        metrics=ExecutionMetrics(
            total_cost=total_cost,
            total_latency=total_latency,
            total_tokens=merged_tokens,
        ),
        model_id=last_trace.model_id,
        model_provider=last_trace.model_provider,
        turns=turn_summaries,
    )


def _detect_all_agents() -> List[Dict[str, str]]:
    """Scan common ports and paths for all running agents.

    Returns a list of dicts with 'url', 'port', and 'name' (from /health if available).
    """
    import socket

    ports = [8000, 8080, 8090, 3000, 3001, 5000, 5001, 8888, 8081, 4000]
    execute_paths = ["/execute", "/invoke", "/api/chat", "/api/agent", "/run", "/chat", "/"]
    health_paths = ["/health"]

    open_ports = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                open_ports.append(port)

    if not open_ports:
        return []

    agents: List[Dict[str, str]] = []
    seen_ports: set = set()

    for port in open_ports:
        # Try to get agent name from health endpoint
        agent_name = ""
        for path in health_paths:
            try:
                r = httpx.get(f"http://localhost:{port}{path}", timeout=2.0)
                if r.status_code == 200:
                    data = r.json()
                    # Extract a useful label from health response
                    agent_name = data.get("title", "") or data.get("name", "") or data.get("model", "") or ""
                    if data.get("backend"):
                        agent_name = f"{agent_name} ({data['backend']})" if agent_name else data["backend"]
            except Exception:
                pass

        for path in execute_paths:
            url = f"http://localhost:{port}{path}"
            try:
                r = httpx.post(url, json={"query": "ping"}, timeout=2.0)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        if "output" in data or "response" in data or "message" in data:
                            if port not in seen_ports:
                                seen_ports.add(port)
                                agents.append({"url": url, "port": str(port), "name": agent_name})
                            break
                    except Exception:
                        pass
            except Exception:
                continue

        # Fallback: if health responded but no execute path returned valid data
        # (e.g. agent needs an API key to process requests but the endpoint exists)
        if port not in seen_ports:
            # Try execute paths again — accept any response (even errors) as proof the path exists
            best_path = ""
            for path in execute_paths:
                try:
                    r = httpx.post(f"http://localhost:{port}{path}", json={"query": "ping"}, timeout=2.0)
                    # Any non-404 response means this path exists
                    if r.status_code != 404:
                        best_path = path
                        break
                except Exception:
                    continue

            for path in health_paths:
                try:
                    r = httpx.get(f"http://localhost:{port}{path}", timeout=2.0)
                    if r.status_code == 200:
                        seen_ports.add(port)
                        url = f"http://localhost:{port}{best_path}" if best_path else f"http://localhost:{port}"
                        agents.append({"url": url, "port": str(port), "name": agent_name})
                        break
                except Exception:
                    continue

    return agents


def _detect_agent_endpoint() -> Optional[str]:
    """Scan common ports for running agents. If multiple found, ask user to choose."""
    import click as _click

    agents = _detect_all_agents()

    if not agents:
        return None

    if len(agents) == 1:
        return agents[0]["url"]

    # Multiple agents found — let user choose
    console.print(f"\n[bold]Found {len(agents)} running agents:[/bold]\n")
    for i, agent in enumerate(agents, 1):
        label = agent["name"] or "unknown"
        console.print(f"  [cyan]{i}.[/cyan] {agent['url']}  [dim]{label}[/dim]")
    console.print()

    choice = _click.prompt("Which agent?", default="1", show_default=False).strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(agents):
            return agents[idx]["url"]
    except ValueError:
        pass

    return agents[0]["url"]


def _parse_fail_statuses(fail_on: str) -> set:
    """Parse a comma-separated fail_on string into a set of DiffStatus enums.

    Accepts uppercase strings like 'REGRESSION,TOOLS_CHANGED' and returns
    the corresponding DiffStatus enum members.
    """
    from evalview.core.diff import DiffStatus

    mapping = {s.value.upper(): s for s in DiffStatus}
    return {
        mapping[s.strip().upper()]
        for s in fail_on.split(",")
        if s.strip().upper() in mapping
    }


def _load_config_if_exists() -> Optional["EvalViewConfig"]:
    """Load config from .evalview/config.yaml if it exists."""
    from evalview.core.config import EvalViewConfig

    config_path = Path(".evalview/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            return EvalViewConfig.model_validate(config_data)
    return None


def _build_adapter_for_tc(
    tc: Any,
    config: Optional[Any],
    timeout: float,
) -> Optional[Any]:
    """Build the right adapter for a test case, handling both HTTP and non-HTTP adapters.

    Returns the adapter, or None if the test should be skipped (missing config).
    Raises ValueError for invalid config that should be surfaced to the user.
    """
    adapter_type = tc.adapter or (config.adapter if config else None)
    endpoint = tc.endpoint or (config.endpoint if config else None)

    # These adapters have their own auth/model and don't need an HTTP endpoint
    _no_endpoint_adapters = {"opencode", "goose", "openai-assistants", "mistral", "cohere", "aider"}
    needs_endpoint = adapter_type not in _no_endpoint_adapters if adapter_type else True

    if not adapter_type or (needs_endpoint and not endpoint):
        return None

    allow_private = getattr(config, "allow_private_urls", True) if config else True

    if adapter_type == "opencode":
        from evalview.adapters.opencode_adapter import OpenCodeAdapter
        test_cfg: dict = tc.adapter_config or {}
        ctx = tc.input.context or {}
        return OpenCodeAdapter(
            timeout=test_cfg.get("timeout", timeout),
            model=test_cfg.get("model"),
            cwd=ctx.get("cwd"),
        )

    if adapter_type == "aider":
        from evalview.adapters.aider_adapter import AiderAdapter
        aider_cfg: dict = tc.adapter_config or {}
        aider_ctx = tc.input.context or {}
        return AiderAdapter(
            timeout=aider_cfg.get("timeout", timeout),
            model=aider_cfg.get("model"),
            cwd=aider_ctx.get("cwd"),
            aider_path=aider_cfg.get("aider_path"),
            reset_files=aider_cfg.get("reset_files", True),
        )

    return _create_adapter(adapter_type, endpoint or "", timeout=timeout, allow_private_urls=allow_private)


async def _execute_agent_with_slow_warning(
    tc: "TestCase",
    adapter: "AgentAdapter",
    timeout: float,
    emit_warning: bool = True,
) -> ExecutionTrace:
    """Execute the agent for `tc`, printing a slow-agent warning at 50% of timeout.

    Prints the real measured elapsed time (not `timeout * 0.5`), so the message
    stays accurate if event-loop scheduling slips. The background warning task is
    always cancelled in the finally block.
    """
    warning_task: Optional[asyncio.Task] = None
    if emit_warning and timeout > 0:
        start = time.monotonic()
        warn_after = timeout * 0.5

        async def _warn() -> None:
            await asyncio.sleep(warn_after)
            elapsed = time.monotonic() - start
            console.print(
                f"[yellow]⚠ {tc.name}: Agent is taking a while... "
                f"({elapsed:.1f}s elapsed of {timeout:.1f}s timeout)[/yellow]"
            )

        warning_task = asyncio.create_task(_warn())

    try:
        if tc.is_multi_turn:
            return await _execute_multi_turn_trace(tc, adapter)
        return await adapter.execute(tc.input.query, tc.input.context)
    finally:
        if warning_task is not None:
            warning_task.cancel()
            try:
                await warning_task
            except asyncio.CancelledError:
                pass


def _execute_snapshot_tests(
    test_cases: List["TestCase"],
    config: Optional["EvalViewConfig"],
    timeout: float = 30.0,
    skip_llm_judge: bool = False,
    json_output: bool = False,
) -> List["EvaluationResult"]:
    """Execute tests and evaluate results for snapshot/benchmark commands.

    When json_output=True, per-test console output is suppressed so stdout
    stays clean for JSON consumers.
    """
    from evalview.evaluators.evaluator import Evaluator

    results = []
    evaluator = Evaluator(skip_llm_judge=skip_llm_judge)

    async def _run_one(tc: "TestCase") -> Optional["EvaluationResult"]:
        try:
            adapter = _build_adapter_for_tc(tc, config, timeout)
        except ValueError as e:
            if not json_output:
                console.print(f"[yellow]⚠ Skipping {tc.name}: {e}[/yellow]")
            return None
        if adapter is None:
            if not json_output:
                console.print(f"[yellow]⚠ Skipping {tc.name}: No adapter/endpoint configured[/yellow]")
            return None

        trace = await _execute_agent_with_slow_warning(tc, adapter, timeout)
        return await evaluator.evaluate(tc, trace)

    async def _run_all() -> List[Any]:
        return await asyncio.gather(*[_run_one(tc) for tc in test_cases], return_exceptions=True)

    outcomes = asyncio.run(_run_all())

    for tc, outcome in zip(test_cases, outcomes):
        if isinstance(outcome, BaseException):
            error_str = str(outcome)
            endpoint = tc.endpoint or (config.endpoint if config else None) or ""
            if json_output:
                continue
            if isinstance(outcome, (asyncio.TimeoutError, asyncio.CancelledError)):
                console.print(f"[red]✗ {tc.name}: Async execution failed - {outcome}[/red]")
            else:
                console.print(f"[red]✗ {tc.name}: Failed - {outcome}[/red]")
            # Actionable guidance per failure type
            if "timed out" in error_str.lower() or "timeout" in error_str.lower():
                console.print(f"  [dim]Fix: increase timeout with --timeout 120, or check that your agent at {endpoint} is responsive[/dim]")
            elif "connection" in error_str.lower() or "refused" in error_str.lower():
                console.print(f"  [dim]Fix: make sure your agent is running at {endpoint}[/dim]")
            continue
        if outcome is None:
            continue

        result = outcome
        results.append(result)

        if json_output:
            continue

        if result.passed:
            console.print(f"[green]✓ {tc.name}:[/green] {result.score:.1f}/100")
        else:
            console.print(f"[red]✗ {tc.name}:[/red] {result.score:.1f}/100")
            # Show why the test failed so users know what to fix
            if result.min_score and result.score < result.min_score:
                console.print(f"  [dim]Score {result.score:.1f} < {result.min_score:.1f} minimum[/dim]")
            evals = result.evaluations
            if evals.output_quality.score < 50:
                console.print(f"  [dim]Output quality: {evals.output_quality.score:.0f}/100 — {evals.output_quality.rationale[:120]}[/dim]")
            if evals.hallucination and evals.hallucination.has_hallucination:
                console.print(f"  [dim]Hallucination detected ({evals.hallucination.confidence:.0%} confidence)[/dim]")
            if evals.tool_accuracy.accuracy < 1.0:
                if evals.tool_accuracy.missing:
                    console.print(f"  [dim]Missing tools: {', '.join(evals.tool_accuracy.missing)}[/dim]")
                if evals.tool_accuracy.unexpected:
                    console.print(f"  [dim]Unexpected tools: {', '.join(evals.tool_accuracy.unexpected)}[/dim]")

    return results


def _execute_check_tests(
    test_cases: List["TestCase"],
    config: Optional["EvalViewConfig"],
    json_output: bool,
    semantic_diff: bool = False,
    timeout: float = 30.0,
    skip_llm_judge: bool = False,
    budget_tracker: Optional["BudgetTracker"] = None,
) -> Tuple[List[Tuple[str, "TraceDiff"]], List["EvaluationResult"], "DriftTracker", Dict[str, "GoldenTrace"]]:
    """Execute tests and compare against golden variants.

    Args:
        test_cases: Test cases to run.
        config: EvalView config (adapter, endpoint, thresholds).
        json_output: Suppress non-JSON console output when True.
        semantic_diff: Enable embedding-based semantic similarity (opt-in).
        budget_tracker: Optional budget tracker for mid-run circuit breaking.

    Returns:
        Tuple of (diffs, results, drift_tracker, golden_traces) where
        diffs is [(test_name, TraceDiff)] and golden_traces maps test name
        to the primary GoldenTrace used for comparison.
    """
    from evalview.core.golden import GoldenStore
    from evalview.core.diff import DiffEngine
    from evalview.core.config import DiffConfig
    from evalview.core.drift_tracker import DriftTracker
    from evalview.evaluators.evaluator import Evaluator

    diff_config = config.get_diff_config() if config else DiffConfig()
    # --semantic-diff flag overrides config file setting
    if semantic_diff:
        diff_config = DiffConfig(
            **{**diff_config.model_dump(), "semantic_diff_enabled": True}
        )

    store = GoldenStore()
    diff_engine = DiffEngine(config=diff_config)
    drift_tracker = DriftTracker()
    evaluator = Evaluator(skip_llm_judge=skip_llm_judge)

    results: List["EvaluationResult"] = []
    diffs: List[Tuple[str, "TraceDiff"]] = []
    golden_traces: Dict[str, GoldenTrace] = {}

    if budget_tracker is not None:
        # Sequential execution with budget checking after each test
        async def _run_one_sequential(tc: "TestCase") -> Optional[Tuple["EvaluationResult", "TraceDiff", "GoldenTrace"]]:
            """Run a single test: execute -> evaluate -> diff (async pipeline)."""
            try:
                adapter = _build_adapter_for_tc(tc, config, timeout)
            except ValueError as e:
                if not json_output:
                    console.print(f"[yellow]⚠ Skipping {tc.name}: {e}[/yellow]")
                return None
            if adapter is None:
                return None

            trace = await _execute_agent_with_slow_warning(
                tc, adapter, timeout, emit_warning=not json_output
            )
            result = await evaluator.evaluate(tc, trace)

            golden_variants = store.load_all_golden_variants(tc.name)
            if not golden_variants:
                return None

            diff = await diff_engine.compare_multi_reference_async(
                golden_variants, trace, result.score
            )
            return result, diff, golden_variants[0]

        async def _run_all_with_budget() -> None:
            from evalview.core.budget import BudgetExhausted

            total = len(test_cases)
            completed = 0
            for tc in test_cases:
                try:
                    outcome = await _run_one_sequential(tc)
                except BaseException as exc:
                    if not json_output:
                        if isinstance(exc, (asyncio.TimeoutError, asyncio.CancelledError)):
                            console.print(f"[red]✗ {tc.name}: Async execution timed out — {exc}[/red]")
                        else:
                            console.print(f"[red]✗ {tc.name}: Failed — {exc}[/red]")
                    completed += 1
                    continue

                if outcome is None:
                    completed += 1
                    continue

                result, diff, golden = outcome
                results.append(result)
                diffs.append((tc.name, diff))
                golden_traces[tc.name] = golden
                drift_tracker.record_check(tc.name, diff, result=result)

                # Record cost and check budget
                cost = result.trace.metrics.total_cost
                adapter_type = tc.adapter or (config.adapter if config else "") or ""
                budget_tracker.record_cost(tc.name, cost, adapter=adapter_type)
                completed += 1

                try:
                    budget_tracker.check_budget(completed=completed, total=total)
                except BudgetExhausted:
                    break

        asyncio.run(_run_all_with_budget())
    else:
        # Original concurrent execution (no budget tracking)
        async def _run_one(tc: "TestCase") -> Optional[Tuple["EvaluationResult", "TraceDiff", "GoldenTrace"]]:
            """Run a single test: execute -> evaluate -> diff (async pipeline)."""
            try:
                adapter = _build_adapter_for_tc(tc, config, timeout)
            except ValueError as e:
                if not json_output:
                    console.print(f"[yellow]⚠ Skipping {tc.name}: {e}[/yellow]")
                return None
            if adapter is None:
                return None

            trace = await _execute_agent_with_slow_warning(
                tc, adapter, timeout, emit_warning=not json_output
            )
            result = await evaluator.evaluate(tc, trace)

            golden_variants = store.load_all_golden_variants(tc.name)
            if not golden_variants:
                return None

            # Use async comparison to include semantic diff when enabled
            diff = await diff_engine.compare_multi_reference_async(
                golden_variants, trace, result.score
            )
            return result, diff, golden_variants[0]

        # Run all tests concurrently in a single event loop.
        # return_exceptions=True means exceptions are returned as values (not raised),
        # so one failing test does not cancel the others.
        async def _run_all() -> List:
            return await asyncio.gather(*[_run_one(tc) for tc in test_cases], return_exceptions=True)

        outcomes = asyncio.run(_run_all())

        for tc, outcome in zip(test_cases, outcomes):
            if isinstance(outcome, BaseException):
                if not json_output:
                    if isinstance(outcome, (asyncio.TimeoutError, asyncio.CancelledError)):
                        console.print(f"[red]✗ {tc.name}: Async execution timed out — {outcome}[/red]")
                    else:
                        console.print(f"[red]✗ {tc.name}: Failed — {outcome}[/red]")
                continue
            if outcome is None:
                continue
            result, diff, golden = outcome
            results.append(result)
            diffs.append((tc.name, diff))
            golden_traces[tc.name] = golden
            drift_tracker.record_check(tc.name, diff)

    return diffs, results, drift_tracker, golden_traces


def _analyze_check_diffs(diffs: List[Tuple[str, "TraceDiff"]]) -> Dict[str, Any]:
    """Analyze diffs and return summary statistics.

    Returns:
        Dict with keys: has_regressions, has_tools_changed, has_output_changed, all_passed
    """
    from evalview.core.diff import DiffStatus

    has_regressions = any(d.overall_severity == DiffStatus.REGRESSION for _, d in diffs)
    has_tools_changed = any(d.overall_severity == DiffStatus.TOOLS_CHANGED for _, d in diffs)
    has_output_changed = any(d.overall_severity == DiffStatus.OUTPUT_CHANGED for _, d in diffs)
    all_passed = not has_regressions and not has_tools_changed and not has_output_changed

    return {
        "has_regressions": has_regressions,
        "has_tools_changed": has_tools_changed,
        "has_output_changed": has_output_changed,
        "all_passed": all_passed,
    }


def _cloud_push(saved_test_names: List[str]) -> None:
    """Upload golden baselines for the given tests. Silently skips on error."""
    from evalview.cloud.auth import CloudAuth
    from evalview.cloud.client import CloudClient
    from evalview.core.golden import GoldenStore

    auth = CloudAuth()
    if not auth.is_logged_in():
        return

    store = GoldenStore()

    async def _push() -> None:
        client = CloudClient(auth.get_access_token() or "")
        user_id = auth.get_user_id() or ""
        for test_name in saved_test_names:
            golden = store.load_golden(test_name)
            if golden:
                await client.upload_golden(user_id, test_name, golden.model_dump())

    try:
        asyncio.run(_push())
        console.print("[dim]☁  Synced to cloud[/dim]")
    except Exception:
        if not os.environ.get("EVALVIEW_DEMO"):
            console.print("[dim]⚠  Cloud sync skipped (offline?)[/dim]")


def _cloud_pull(store: "GoldenStore") -> None:
    """Pull missing golden baselines from cloud. Silently skips on error."""
    from evalview.cloud.auth import CloudAuth
    from evalview.cloud.client import CloudClient

    auth = CloudAuth()
    if not auth.is_logged_in():
        return

    async def _pull() -> None:
        client = CloudClient(auth.get_access_token() or "")
        user_id = auth.get_user_id() or ""
        remote_names = await client.list_goldens(user_id)
        for test_name in remote_names:
            if not store.has_golden(test_name):
                data = await client.download_golden(user_id, test_name)
                if data:
                    store.save_golden_from_dict(test_name, data)

    try:
        asyncio.run(_pull())
    except Exception:
        pass  # Silently skip — local goldens still work
