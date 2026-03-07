"""Shared state and helpers used across multiple command modules."""
from __future__ import annotations

import asyncio
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx
import yaml
from dotenv import load_dotenv
from rich.console import Console

from evalview.adapters.http_adapter import HTTPAdapter
from evalview.adapters.tapescope_adapter import TapeScopeAdapter
from evalview.adapters.langgraph_adapter import LangGraphAdapter
from evalview.adapters.crewai_adapter import CrewAIAdapter
from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter
from evalview.core.types import ExecutionTrace, ExecutionMetrics, TokenUsage

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult, TestCase
    from evalview.core.config import EvalViewConfig
    from evalview.core.golden import GoldenStore
    from evalview.adapters.base import AgentAdapter

# Load environment variables (.env is the OSS standard, .env.local for overrides)
load_dotenv()
load_dotenv(dotenv_path=".env.local", override=True)

console = Console()


def _create_adapter(adapter_type: str, endpoint: str, timeout: float = 30.0, allow_private_urls: bool = True) -> "AgentAdapter":
    """Factory function for creating adapters based on type."""
    if adapter_type == "cohere":
        from evalview.adapters.cohere_adapter import CohereAdapter
        return CohereAdapter()

    if adapter_type == "mistral":
        from evalview.adapters.mistral_adapter import MistralAdapter
        return MistralAdapter()

    adapter_map = {
        "http": HTTPAdapter,
        "langgraph": LangGraphAdapter,
        "tapescope": TapeScopeAdapter,
        "crewai": CrewAIAdapter,
        "openai": OpenAIAssistantsAdapter,
    }

    adapter_class = adapter_map.get(adapter_type)
    if not adapter_class:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    if adapter_type == "http":
        return adapter_class(endpoint=endpoint, timeout=timeout, allow_private_urls=allow_private_urls)
    return adapter_class(endpoint=endpoint, timeout=timeout)


async def _execute_multi_turn_trace(test_case: Any, adapter: Any) -> ExecutionTrace:
    """Execute all turns of a multi-turn test and return a merged ExecutionTrace."""
    conversation_history: List[Dict[str, Any]] = []
    all_steps: List[Any] = []
    turn_traces: List[Any] = []

    for turn in test_case.turns:
        turn_context: Dict[str, Any] = dict(turn.context or {})
        if test_case.tools:
            turn_context.setdefault("tools", test_case.tools)
        if conversation_history:
            turn_context["conversation_history"] = list(conversation_history)

        trace = await adapter.execute(turn.query, turn_context)
        turn_traces.append(trace)
        all_steps.extend(trace.steps)

        conversation_history.append({"role": "user", "content": turn.query})
        conversation_history.append({"role": "assistant", "content": trace.final_output})

    total_cost = sum(t.metrics.total_cost for t in turn_traces)
    total_latency = sum(t.metrics.total_latency for t in turn_traces)
    merged_tokens: Optional[TokenUsage] = None
    if any(t.metrics.total_tokens for t in turn_traces):
        merged_tokens = TokenUsage(
            input_tokens=sum(
                (t.metrics.total_tokens.input_tokens if t.metrics.total_tokens else 0)
                for t in turn_traces
            ),
            output_tokens=sum(
                (t.metrics.total_tokens.output_tokens if t.metrics.total_tokens else 0)
                for t in turn_traces
            ),
            cached_tokens=sum(
                (t.metrics.total_tokens.cached_tokens if t.metrics.total_tokens else 0)
                for t in turn_traces
            ),
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
    )


def _detect_agent_endpoint() -> Optional[str]:
    """Scan common ports and paths for a running agent. Returns URL or None."""
    import socket

    ports = [8090, 8000, 8080, 3000, 3001, 5000, 5001, 8888, 8081, 4000]
    paths = ["/invoke", "/api/chat", "/api/agent", "/run", "/chat", "/", "/health"]

    open_ports = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                open_ports.append(port)

    if not open_ports:
        return None

    for port in open_ports:
        for path in paths:
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

    return None


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

    for tc in test_cases:
        try:
            adapter_type = tc.adapter or (config.adapter if config else None)
            endpoint = tc.endpoint or (config.endpoint if config else None)

            if not adapter_type or not endpoint:
                console.print(f"[yellow]⚠ Skipping {tc.name}: No adapter/endpoint configured[/yellow]")
                continue

            allow_private = getattr(config, "allow_private_urls", True) if config else True
            try:
                adapter = _create_adapter(adapter_type, endpoint, allow_private_urls=allow_private)
            except ValueError as e:
                console.print(f"[yellow]⚠ Skipping {tc.name}: {e}[/yellow]")
                continue

            async def _run_one_test() -> Any:
                t = await adapter.execute(tc.input.query, tc.input.context)
                return await evaluator.evaluate(tc, t)

            try:
                result = asyncio.run(_run_one_test())
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                console.print(f"[red]✗ {tc.name}: Async execution failed - {e}[/red]")
                continue
            except Exception:
                raise
            results.append(result)

            if result.passed:
                console.print(f"[green]✓ {tc.name}:[/green] {result.score:.1f}/100")
            else:
                console.print(f"[red]✗ {tc.name}:[/red] {result.score:.1f}/100")

        except Exception as e:
            console.print(f"[red]✗ {tc.name}: Failed - {e}[/red]")
            continue

    return results


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
