"""Phase helpers for `evalview run`.

Extracted from `_cmd.py` so the module-level orchestrator there reads as the
high-level flow it is, rather than a wall of utilities. Each helper here
handles one phase of a run:

  _check_mcp_contracts       — MCP contract drift detection
  _load_test_cases           — discover and load test YAMLs from a path
  _apply_quality_filter      — drop low-quality generated tests when flagged
  _print_run_mode_guidance   — header banner with run-mode reminder
  _maybe_show_adapter_menu   — interactive picker when adapter is ambiguous
  _filter_by_name            — narrow tests by `--test` glob
  _filter_by_tags            — narrow tests by `--tags` expression
  _run_watch_mode            — file-watcher re-run loop
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from evalview.core.loader import TestCaseLoader


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
    from evalview.core.project_state import ProjectStateStore

    state_store = ProjectStateStore()

    if path:
        target = Path(path)
        if target.is_file():
            try:
                cases = [TestCaseLoader.load_from_file(target)]
                if verbose:
                    console.print(f"[dim]📄 Loading test case from: {path}[/dim]\n")
                state_store.set_active_test_path(str(target.parent))
                return cases
            except Exception as exc:
                console.print(f"[red]❌ Failed to load test case: {exc}[/red]")
                return None
        elif target.is_dir():
            cases = TestCaseLoader.load_from_directory(target, "*.yaml")
            if verbose:
                console.print(f"[dim]📁 Loading test cases from: {path}[/dim]\n")
            state_store.set_active_test_path(str(target))
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

    active_test_path = state_store.get_active_test_path()
    default_dir = Path(active_test_path) if active_test_path and Path(active_test_path).exists() else Path("tests/test-cases")
    if not default_dir.exists():
        console.print(f"[red]❌ Test cases directory not found: {default_dir}[/red]")
        console.print("[dim]Tip: You can specify a path or file directly:[/dim]")
        console.print("[dim]  evalview run examples/anthropic[/dim]")
        console.print("[dim]  evalview run path/to/test-case.yaml[/dim]")
        return None

    cases = TestCaseLoader.load_from_directory(default_dir, pattern)
    state_store.set_active_test_path(str(default_dir))
    if not cases:
        console.print(f"[yellow]⚠️  No test cases found matching pattern: {pattern}[/yellow]\n")
        console.print("[bold]💡 Create tests by:[/bold]")
        console.print("   • [cyan]evalview record --interactive[/cyan]   (record agent interactions)")
        console.print("   • [cyan]evalview expand <test.yaml>[/cyan]     (generate variations from seed)")
        console.print(f"   • Or create YAML files manually in {default_dir}/")
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


def _print_run_mode_guidance(test_cases: List[Any], console: Any) -> None:
    """Clarify that `run` is direct evaluation, not the main regression flow."""
    generated_count = sum(1 for tc in test_cases if getattr(tc, "generated", False))
    if generated_count == 0:
        return
    console.print(
        "[dim]`evalview run` evaluates the current agent directly. "
        "For the main regression flow, save a baseline with `evalview snapshot` and compare with `evalview check`.[/dim]\n"
    )


def _maybe_show_adapter_menu(
    test_cases: List[Any],
    config: Dict[str, Any],
    html_report: Optional[str],
    console: Any,
) -> tuple:
    """Show interactive adapter selection menu when multiple adapters are present."""
    import socket
    from datetime import datetime

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

    from evalview.core.llm_configs import DEFAULT_FAST_MODEL
    cost_model = config.get("model", DEFAULT_FAST_MODEL)
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


def _filter_by_tags(
    test_cases: List[Any],
    tags: tuple,
) -> tuple[List[Any], list[str]]:
    active_tags: list[str] = []
    for tag in tags:
        value = str(tag).strip().lower()
        if value and value not in active_tags:
            active_tags.append(value)

    if not active_tags:
        return test_cases, []

    filtered = [
        tc for tc in test_cases
        if set(getattr(tc, "tags", []) or []).intersection(active_tags)
    ]
    return filtered, active_tags


async def _run_watch_mode(
    console: Any,
    **kwargs: Any,
) -> None:
    """Start file watcher and re-run tests on every change."""
    # Local import to avoid the circular `_cmd → _helpers → _cmd` chain at
    # module load time. The watch path is the only place a helper calls back
    # into the orchestrator.
    from evalview.commands.run._cmd import _run_async
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
