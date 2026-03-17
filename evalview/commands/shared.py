"""Shared state and helpers used across multiple command modules."""
from __future__ import annotations

import asyncio
import os
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

# Load environment variables (.env is the OSS standard, .env.local for overrides)
load_dotenv()
load_dotenv(dotenv_path=".env.local", override=True)

console = Console()


def apply_judge_model(judge_model: Optional[str]) -> None:
    """Resolve --judge flag: set EVAL_MODEL and EVAL_PROVIDER, validate API key.

    Exits with a clear error if the required API key is missing.
    """
    if not judge_model:
        return

    import os
    import sys
    from evalview.core.llm_configs import resolve_model_alias, PROVIDER_CONFIGS, LLMProvider

    resolved = resolve_model_alias(judge_model)
    os.environ["EVAL_MODEL"] = resolved

    # Infer provider from model name and set EVAL_PROVIDER so the right client is used
    model_lower = resolved.lower()
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
                    f"Model [bold]{resolved}[/bold] requires [bold]{env_var}[/bold] to be set.\n\n"
                    f"[dim]Get your key at: {config.api_key_url}[/dim]\n"
                    f"[dim]Then: export {env_var}=your-key-here[/dim]\n"
                )
                sys.exit(1)
            break


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


def _detect_agent_endpoint() -> Optional[str]:
    """Scan common ports and paths for a running agent. Returns URL or None."""
    import socket

    ports = [8090, 8000, 8080, 3000, 3001, 5000, 5001, 8888, 8081, 4000]
    execute_paths = ["/execute", "/invoke", "/api/chat", "/api/agent", "/run", "/chat", "/"]
    health_paths = ["/health"]

    open_ports = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                open_ports.append(port)

    if not open_ports:
        return None

    for port in open_ports:
        for path in execute_paths:
            url = f"http://localhost:{port}{path}"
            try:
                r = httpx.post(url, json={"query": "ping"}, timeout=2.0)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        if "output" in data or "response" in data or "message" in data:
                            return url
                    except Exception:
                        pass
            except Exception:
                continue

        for path in health_paths:
            url = f"http://localhost:{port}{path}"
            try:
                r = httpx.get(url, timeout=2.0)
                if r.status_code == 200:
                    return f"http://localhost:{port}"
            except Exception:
                continue

    return None


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


def _execute_snapshot_tests(
    test_cases: List["TestCase"],
    config: Optional["EvalViewConfig"],
) -> List["EvaluationResult"]:
    """Execute tests and evaluate results for snapshot/benchmark commands."""
    from evalview.evaluators.evaluator import Evaluator

    results = []
    evaluator = Evaluator()

    async def _run_one(tc: "TestCase") -> Optional["EvaluationResult"]:
        adapter_type = tc.adapter or (config.adapter if config else None)
        endpoint = tc.endpoint or (config.endpoint if config else None)

        if not adapter_type or not endpoint:
            console.print(f"[yellow]⚠ Skipping {tc.name}: No adapter/endpoint configured[/yellow]")
            return None

        allow_private = getattr(config, "allow_private_urls", True) if config else True
        try:
            adapter = _create_adapter(adapter_type, endpoint, allow_private_urls=allow_private)
        except ValueError as e:
            console.print(f"[yellow]⚠ Skipping {tc.name}: {e}[/yellow]")
            return None

        if tc.is_multi_turn:
            trace = await _execute_multi_turn_trace(tc, adapter)
        else:
            trace = await adapter.execute(tc.input.query, tc.input.context)
        return await evaluator.evaluate(tc, trace)

    async def _run_all() -> List[Any]:
        return await asyncio.gather(*[_run_one(tc) for tc in test_cases], return_exceptions=True)

    outcomes = asyncio.run(_run_all())

    for tc, outcome in zip(test_cases, outcomes):
        if isinstance(outcome, BaseException):
            if isinstance(outcome, (asyncio.TimeoutError, asyncio.CancelledError)):
                console.print(f"[red]✗ {tc.name}: Async execution failed - {outcome}[/red]")
            else:
                console.print(f"[red]✗ {tc.name}: Failed - {outcome}[/red]")
            continue
        if outcome is None:
            continue

        result = outcome
        results.append(result)

        if result.passed:
            console.print(f"[green]✓ {tc.name}:[/green] {result.score:.1f}/100")
        else:
            console.print(f"[red]✗ {tc.name}:[/red] {result.score:.1f}/100")

    return results


def _execute_check_tests(
    test_cases: List["TestCase"],
    config: Optional["EvalViewConfig"],
    json_output: bool,
    semantic_diff: bool = False,
    timeout: float = 30.0,
) -> Tuple[List[Tuple[str, "TraceDiff"]], List["EvaluationResult"], "DriftTracker", Dict[str, "GoldenTrace"]]:
    """Execute tests and compare against golden variants.

    Args:
        test_cases: Test cases to run.
        config: EvalView config (adapter, endpoint, thresholds).
        json_output: Suppress non-JSON console output when True.
        semantic_diff: Enable embedding-based semantic similarity (opt-in).

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
    evaluator = Evaluator()

    results: List["EvaluationResult"] = []
    diffs: List[Tuple[str, "TraceDiff"]] = []
    golden_traces: Dict[str, GoldenTrace] = {}

    async def _run_one(tc) -> Optional[Tuple["EvaluationResult", "TraceDiff", GoldenTrace]]:
        """Run a single test: execute -> evaluate -> diff (async pipeline)."""
        adapter_type = tc.adapter or (config.adapter if config else None)
        endpoint = tc.endpoint or (config.endpoint if config else None)
        if not adapter_type or not endpoint:
            return None

        allow_private = getattr(config, "allow_private_urls", True) if config else True
        try:
            adapter = _create_adapter(adapter_type, endpoint, timeout=timeout, allow_private_urls=allow_private)
        except ValueError as e:
            if not json_output:
                console.print(f"[yellow]⚠ Skipping {tc.name}: {e}[/yellow]")
            return None

        if tc.is_multi_turn:
            trace = await _execute_multi_turn_trace(tc, adapter)
        else:
            trace = await adapter.execute(tc.input.query, tc.input.context)
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
