"""Monitor command - continuous regression detection with webhook alerts."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click

from evalview.commands.shared import (
    _analyze_check_diffs,
    _execute_check_tests,
    _load_config_if_exists,
    _parse_fail_statuses,
    console,
)
from evalview.core.diff import DiffStatus
from evalview.core.noise_tracker import (
    ConfirmationGate,
    detect_coordinated_incident,
    record_cycle_noise,
)
from evalview.telemetry.decorators import track_command


class MonitorError(Exception):
    """Raised when monitor cannot start due to missing prerequisites."""


def _resolve_slack_webhook(
    cli_flag: Optional[str],
    config: Any,
) -> Optional[str]:
    """Resolve Slack webhook URL with clear priority: CLI flag > config > env var."""
    if cli_flag:
        return cli_flag
    if config:
        monitor_cfg = config.get_monitor_config()
        if monitor_cfg.slack_webhook:
            return monitor_cfg.slack_webhook
    return os.environ.get("EVALVIEW_SLACK_WEBHOOK")


def _resolve_discord_webhook(
    cli_flag: Optional[str],
    config: Any,
) -> Optional[str]:
    """Resolve Discord webhook URL with clear priority: CLI flag > config > env var."""
    if cli_flag:
        return cli_flag
    if config:
        monitor_cfg = config.get_monitor_config()
        if monitor_cfg.discord_webhook:
            return monitor_cfg.discord_webhook
    return os.environ.get("EVALVIEW_DISCORD_WEBHOOK")


def _append_history(history_path: Path, record: dict) -> None:
    """Append one JSON record to the JSONL history file.

    Creates parent directories and the file if they do not exist.
    Each call appends a single newline-terminated JSON object.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# Default path for the incidents feed that `evalview autopr` consumes. The
# monitor loop appends one record per confirmed production regression so the
# autopr command can later convert it into a pinned regression test.
DEFAULT_INCIDENTS_PATH = Path(".evalview/incidents.jsonl")


def _build_incident_record(
    test_name: str,
    diff: Any,
    test_case: Any,
    result: Any,
    golden: Any,
    cycle: int,
) -> Dict[str, Any]:
    """Build a single incident record from a confirmed-failing diff.

    The shape of this dict is the contract between `evalview monitor` (which
    writes it) and `evalview.core.regression_synth.synthesize_regression_test`
    (which consumes it). Keep the two in lockstep.
    """
    from evalview.core.regression_synth import truncate_output

    baseline_tools: List[str] = []
    actual_tools: List[str] = []
    baseline_output = ""
    actual_output = ""

    golden_trace = getattr(golden, "trace", None) if golden is not None else None
    if golden_trace is not None:
        baseline_output = getattr(golden_trace, "final_output", "") or ""
        baseline_tools = [
            step.tool_name
            for step in getattr(golden_trace, "steps", []) or []
            if getattr(step, "tool_name", None)
        ]

    result_trace = getattr(result, "trace", None) if result is not None else None
    if result_trace is not None:
        actual_output = getattr(result_trace, "final_output", "") or ""
        actual_tools = [
            step.tool_name
            for step in getattr(result_trace, "steps", []) or []
            if getattr(step, "tool_name", None)
        ]

    query = ""
    if test_case is not None and getattr(test_case, "input", None) is not None:
        query = getattr(test_case.input, "query", "") or ""

    source_file = getattr(test_case, "source_file", None) if test_case is not None else None

    return {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_name": test_name,
        "query": query,
        "cycle": cycle,
        "status": getattr(diff.overall_severity, "value", str(diff.overall_severity)),
        "score_delta": round(getattr(diff, "score_diff", 0.0), 2),
        "baseline_tools": baseline_tools,
        "actual_tools": actual_tools,
        "baseline_output": truncate_output(baseline_output),
        "actual_output": truncate_output(actual_output),
        "model_changed": bool(getattr(diff, "model_changed", False)),
        "golden_model_id": getattr(diff, "golden_model_id", None),
        "actual_model_id": getattr(diff, "actual_model_id", None),
        "source_file": source_file,
    }


def _append_incidents(
    incidents_path: Path,
    alert_diffs: List[Tuple[str, Any]],
    test_cases_by_name: Dict[str, Any],
    results_by_name: Dict[str, Any],
    golden_traces: Dict[str, Any],
    cycle: int,
) -> int:
    """Append incident records for every confirmed-failing diff.

    Returns the number of records written so the caller can log it.
    """
    if not alert_diffs:
        return 0
    incidents_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with incidents_path.open("a", encoding="utf-8") as f:
        for name, diff in alert_diffs:
            record = _build_incident_record(
                test_name=name,
                diff=diff,
                test_case=test_cases_by_name.get(name),
                result=results_by_name.get(name),
                golden=golden_traces.get(name),
                cycle=cycle,
            )
            f.write(json.dumps(record) + "\n")
            written += 1
    return written


def _detect_spikes(
    results: List[Any],
    golden_traces: Dict[str, Any],
    cost_threshold: Optional[float],
    latency_threshold: Optional[float],
) -> List[Dict[str, Any]]:
    """Compare current cost/latency against golden baselines. Returns list of alerts."""
    alerts: List[Dict[str, Any]] = []
    if not cost_threshold and not latency_threshold:
        return alerts

    for r in results:
        name = r.test_case
        golden = golden_traces.get(name)
        if not golden:
            continue

        baseline_cost = golden.trace.metrics.total_cost
        baseline_latency = golden.trace.metrics.total_latency
        current_cost = r.trace.metrics.total_cost
        current_latency = r.trace.metrics.total_latency

        if cost_threshold and baseline_cost > 0 and current_cost / baseline_cost > cost_threshold:
            alerts.append({
                "test_name": name,
                "alert_type": "cost_spike",
                "current": current_cost,
                "baseline": baseline_cost,
                "multiplier": current_cost / baseline_cost,
            })

        if latency_threshold and baseline_latency > 0 and current_latency / baseline_latency > latency_threshold:
            alerts.append({
                "test_name": name,
                "alert_type": "latency_spike",
                "current": current_latency,
                "baseline": baseline_latency,
                "multiplier": current_latency / baseline_latency,
            })

    return alerts


def _build_notifiers(
    slack_webhook: Optional[str],
    discord_webhook: Optional[str],
) -> List[Tuple[str, Any]]:
    """Build enabled webhook notifiers."""
    from evalview.core.discord_notifier import DiscordNotifier
    from evalview.core.slack_notifier import SlackNotifier

    notifiers: List[Tuple[str, Any]] = []
    if slack_webhook:
        notifiers.append(("Slack", SlackNotifier(slack_webhook)))
    if discord_webhook:
        notifiers.append(("Discord", DiscordNotifier(discord_webhook)))
    return notifiers


def _run_monitor_loop(
    test_path: str,
    interval: int,
    slack_webhook: Optional[str],
    discord_webhook: Optional[str],
    fail_on: str,
    timeout: float,
    test_filter: Optional[str],
    config: Any = None,
    history_path: Optional[Path] = None,
    cost_threshold: Optional[float] = None,
    latency_threshold: Optional[float] = None,
    incidents_path: Optional[Path] = None,
) -> None:
    """Main monitor loop. Runs check cycles until Ctrl+C.

    Raises:
        MonitorError: If baselines or test cases are missing.
    """
    from evalview.core.golden import GoldenStore
    from evalview.core.loader import TestCaseLoader
    from evalview.core.messages import (
        get_random_monitor_clean_message,
        get_random_monitor_cycle_message,
        get_random_monitor_start_message,
    )

    notifiers = _build_notifiers(slack_webhook, discord_webhook)

    store = GoldenStore()
    goldens = store.list_golden()
    if not goldens:
        raise MonitorError("No baselines found. Run `evalview snapshot` first.")

    loader = TestCaseLoader()
    test_cases = loader.load_from_directory(Path(test_path))

    if test_filter:
        test_cases = [tc for tc in test_cases if tc.name == test_filter]
        if not test_cases:
            raise MonitorError(f"No test found with name: {test_filter}")

    # Lookup tables used when writing incident records. Built once here so
    # the inner loop doesn't pay an O(n) scan per confirmed failure.
    test_cases_by_name: Dict[str, Any] = {tc.name: tc for tc in test_cases}

    # Tests tagged `gate: strict` in their YAML bypass the confirmation
    # gate — they alert on n=1. Use this for safety-critical behaviors
    # (auth, payments, PII, refunds) where a one-cycle blip is worth
    # investigating immediately rather than waiting for confirmation.
    strict_tests: Set[str] = {
        tc.name
        for tc in test_cases
        if (getattr(tc, "gate", None) or "").lower() == "strict"
    }

    console.print(f"\n[cyan]{get_random_monitor_start_message()}[/cyan]")
    history_hint = f"  |  History: {history_path}" if history_path else ""
    alert_targets = ", ".join(label for label, _ in notifiers) if notifiers else "None"
    strict_hint = (
        f"  |  Strict: {len(strict_tests)}" if strict_tests else ""
    )
    console.print(
        f"[dim]  Tests: {len(test_cases)}  |  Interval: {interval}s  |  "
        f"Alerts: {alert_targets}{strict_hint}{history_hint}[/dim]"
    )
    console.print("[dim]  Press Ctrl+C to stop.[/dim]\n")

    previously_failing: Set[str] = set()
    # Confirmation gate — suppresses n=1 alerts by requiring a failure
    # to persist into a second cycle before paging a human. This is the
    # single highest-leverage noise-reduction lever: it reframes the
    # product's emotional contract so every alert a user sees is one
    # that survived at least two independent runs. Strict tests bypass
    # the gate so safety-critical behaviors still page on n=1.
    gate = ConfirmationGate()
    cycle_count = 0
    total_cost = 0.0
    shutdown = False

    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig: int, frame: Any) -> None:
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)
    fail_statuses = _parse_fail_statuses(fail_on)

    try:
        while not shutdown:
            cycle_count += 1
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            console.print(f"[dim][{now}][/dim] {get_random_monitor_cycle_message()}")

            try:
                diffs, results, _, golden_traces = _execute_check_tests(
                    test_cases, config, json_output=True, timeout=timeout
                )
            except Exception as e:
                console.print(f"[red]  x Cycle {cycle_count} failed: {e}[/red]")
                _sleep_interruptible(interval, lambda: shutdown)
                continue

            cycle_cost = sum(r.trace.metrics.total_cost for r in results)
            total_cost += cycle_cost
            analysis = _analyze_check_diffs(diffs)

            currently_failing: Set[str] = {
                name for name, diff in diffs if diff.overall_severity in fail_statuses
            }

            regressions = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION)
            tools_changed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED)
            output_changed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED)
            passed = len(diffs) - regressions - tools_changed - output_changed

            # Run the cycle's failures through the confirmation gate. Only
            # the `decision.alerts_to_fire` set should reach a notifier.
            # Strict-tagged tests bypass the gate (alert on n=1).
            decision = gate.evaluate(currently_failing, strict=strict_tests)
            record_cycle_noise(decision)

            if decision.strict_immediate:
                names = ", ".join(sorted(decision.strict_immediate))
                console.print(
                    f"[yellow]  Gate: {len(decision.strict_immediate)} "
                    f"strict failure(s) ({names}) — bypassing confirmation, "
                    f"alerting now[/yellow]"
                )
            if decision.self_resolved:
                names = ", ".join(sorted(decision.self_resolved))
                console.print(
                    f"[dim]  Gate: suppressed {len(decision.self_resolved)} "
                    f"unconfirmed failure(s) ({names}) — recovered before alerting[/dim]"
                )
            if decision.pending:
                names = ", ".join(sorted(decision.pending))
                console.print(
                    f"[dim]  Gate: {len(decision.pending)} failure(s) pending "
                    f"confirmation ({names}) — will alert if still failing next cycle[/dim]"
                )

            if not currently_failing:
                cost_part = f"  [dim]${cycle_cost:.4f}[/dim]" if cycle_cost > 0 else ""
                console.print(f"[green]  {get_random_monitor_clean_message()} ({len(diffs)} tests){cost_part}[/green]")

                if previously_failing and notifiers:
                    for label, notifier in notifiers:
                        asyncio.run(notifier.send_recovery_alert(len(diffs)))
                        console.print(f"[dim]  Alert: {label} recovery notification sent[/dim]")
            else:
                parts = []
                if analysis["has_regressions"]:
                    parts.append(f"[red]{regressions} regression{'s' if regressions != 1 else ''}[/red]")
                if analysis["has_tools_changed"]:
                    parts.append(f"[yellow]{tools_changed} tool change{'s' if tools_changed != 1 else ''}[/yellow]")
                if analysis["has_output_changed"]:
                    parts.append(f"[dim]{output_changed} output change{'s' if output_changed != 1 else ''}[/dim]")

                console.print(f"  Warning: {', '.join(parts)}")

                for name, diff in diffs:
                    if diff.overall_severity in fail_statuses:
                        console.print(f"    [red]x {name}[/red] ({diff.overall_severity.value})")

                # Fire alerts only for failures confirmed by the gate —
                # everything that just started failing this cycle waits
                # one cycle before it can page anyone.
                alerts_to_fire = decision.alerts_to_fire
                if alerts_to_fire:
                    alert_diffs = [(n, d) for n, d in diffs if n in alerts_to_fire]
                    # Persist a machine-readable record of every confirmed
                    # failure so `evalview autopr` can later synthesize a
                    # pinned regression test from it. This is the feed that
                    # closes the production-failure → regression-test loop.
                    if incidents_path is not None:
                        results_by_name = {r.test_case: r for r in results}
                        n_written = _append_incidents(
                            incidents_path,
                            alert_diffs,
                            test_cases_by_name,
                            results_by_name,
                            golden_traces,
                            cycle_count,
                        )
                        if n_written:
                            console.print(
                                f"[dim]  Incidents: logged {n_written} "
                                f"to {incidents_path} — run "
                                f"`evalview autopr` to turn into PRs.[/dim]"
                            )
                    if notifiers:
                        # Collapse correlated failures into a single incident
                        # card when they share a common root cause — the
                        # notifier uses `incident.headline` as the summary.
                        incident = detect_coordinated_incident(alert_diffs)
                        for label, notifier in notifiers:
                            asyncio.run(
                                notifier.send_regression_alert(
                                    alert_diffs, analysis, incident=incident
                                )
                            )
                            if incident is not None:
                                console.print(
                                    f"[dim]  Alert: {label} sent 1 incident "
                                    f"({incident.cause}, {len(alert_diffs)} tests)[/dim]"
                                )
                            else:
                                console.print(
                                    f"[dim]  Alert: {label} notified on "
                                    f"{len(alerts_to_fire)} confirmed failure(s)[/dim]"
                                )

            spike_alerts = _detect_spikes(results, golden_traces, cost_threshold, latency_threshold)
            if spike_alerts:
                for a in spike_alerts:
                    if a["alert_type"] == "cost_spike":
                        console.print(
                            f"  [yellow]$ {a['test_name']}: cost spike "
                            f"${a['baseline']:.4f} -> ${a['current']:.4f} ({a['multiplier']:.1f}x)[/yellow]"
                        )
                    else:
                        console.print(
                            f"  [yellow]T {a['test_name']}: latency spike "
                            f"{a['baseline']:.1f}s -> {a['current']:.1f}s ({a['multiplier']:.1f}x)[/yellow]"
                        )
                if notifiers:
                    for label, notifier in notifiers:
                        asyncio.run(notifier.send_cost_latency_alert(spike_alerts))
                        console.print(f"[dim]  Alert: {label} sent {len(spike_alerts)} performance alert(s)[/dim]")

            if history_path is not None:
                record = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "cycle": cycle_count,
                    "total_tests": len(diffs),
                    "passed": passed,
                    "regressions": regressions,
                    "tools_changed": tools_changed,
                    "output_changed": output_changed,
                    "cost": round(cycle_cost, 6),
                    "failing_tests": sorted(currently_failing),
                    "cost_alerts": sum(1 for a in spike_alerts if a["alert_type"] == "cost_spike"),
                    "latency_alerts": sum(1 for a in spike_alerts if a["alert_type"] == "latency_spike"),
                }
                _append_history(history_path, record)

            previously_failing = currently_failing
            _sleep_interruptible(interval, lambda: shutdown)
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    console.print(f"\n[cyan]Monitor stopped after {cycle_count} cycle(s).[/cyan]")
    if total_cost > 0:
        console.print(f"[dim]  Total cost: ${total_cost:.4f}[/dim]")
    if history_path is not None and cycle_count > 0:
        console.print(f"[dim]  History written to: {history_path}[/dim]")
    console.print()


def _run_monitor_dashboard(
    test_path: str,
    interval: int,
    slack_webhook: Optional[str],
    discord_webhook: Optional[str],
    fail_on: str,
    timeout: float,
    test_filter: Optional[str],
    config: Any = None,
    history_path: Optional[Path] = None,
    cost_threshold: Optional[float] = None,
    latency_threshold: Optional[float] = None,
    incidents_path: Optional[Path] = None,
) -> None:
    """Monitor loop with a live-updating Rich dashboard."""
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from evalview.core.golden import GoldenStore
    from evalview.core.loader import TestCaseLoader

    notifiers = _build_notifiers(slack_webhook, discord_webhook)

    store = GoldenStore()
    goldens = store.list_golden()
    if not goldens:
        raise MonitorError("No baselines found. Run `evalview snapshot` first.")

    loader = TestCaseLoader()
    test_cases = loader.load_from_directory(Path(test_path))

    if test_filter:
        test_cases = [tc for tc in test_cases if tc.name == test_filter]
        if not test_cases:
            raise MonitorError(f"No test found with name: {test_filter}")

    # Lookup tables for incident-record construction (see loop variant).
    test_cases_by_name: Dict[str, Any] = {tc.name: tc for tc in test_cases}

    # Strict-tagged tests bypass the confirmation gate — see CLI loop
    # for the rationale.
    strict_tests: Set[str] = {
        tc.name
        for tc in test_cases
        if (getattr(tc, "gate", None) or "").lower() == "strict"
    }

    fail_statuses = _parse_fail_statuses(fail_on)

    status_dot = {
        DiffStatus.PASSED: "[green]o[/green]",
        DiffStatus.TOOLS_CHANGED: "[yellow]o[/yellow]",
        DiffStatus.OUTPUT_CHANGED: "[yellow]o[/yellow]",
        DiffStatus.REGRESSION: "[red]o[/red]",
    }

    previously_failing: Set[str] = set()
    # See `_run_monitor_loop` for the confirmation-gate rationale — same
    # pattern applies to the dashboard variant. Strict tests bypass the
    # gate and alert on n=1.
    gate = ConfirmationGate()
    cycle_count = 0
    total_cost = 0.0
    start_time = time.time()
    last_check_time = ""
    next_check_time = ""
    alerts_sent = 0
    alerts_suppressed = 0
    test_history: Dict[str, List[DiffStatus]] = {}
    current_statuses: Dict[str, DiffStatus] = {}
    checking = False
    error_msg = ""
    shutdown = False

    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig: int, frame: Any) -> None:
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)

    def _build_dashboard() -> Panel:
        uptime_secs = int(time.time() - start_time)
        uptime_m = uptime_secs // 60
        uptime_s = uptime_secs % 60

        header = Text()
        header.append(f"  Cycle: {cycle_count}", style="bold")
        header.append(f"  |  Uptime: {uptime_m}m{uptime_s:02d}s")
        header.append(f"  |  Cost: ${total_cost:.4f}")
        header.append(f"  |  Alerts: {alerts_sent} sent")

        table = Table(show_header=True, header_style="bold", expand=True, padding=(0, 1))
        table.add_column("Test", style="bold", ratio=3)
        table.add_column("Status", ratio=2)
        table.add_column("History", ratio=2, justify="center")

        for tc in test_cases:
            name = tc.name
            status = current_statuses.get(name)

            if status is None:
                status_str = "[dim]pending[/dim]"
            elif status == DiffStatus.PASSED:
                status_str = "[green]PASSED[/green]"
            elif status == DiffStatus.REGRESSION:
                status_str = "[red]REGRESSION[/red]"
            elif status == DiffStatus.TOOLS_CHANGED:
                status_str = "[yellow]TOOLS_CHANGED[/yellow]"
            elif status == DiffStatus.OUTPUT_CHANGED:
                status_str = "[yellow]OUTPUT_CHANGED[/yellow]"
            else:
                status_str = f"[dim]{status.value}[/dim]"

            history = test_history.get(name, [])
            dots = " ".join(status_dot.get(s, "[dim].[/dim]") for s in history[-5:])
            if not dots:
                dots = "[dim]. . . . .[/dim]"

            table.add_row(name, status_str, dots)

        footer = Text()
        if checking:
            footer.append("  Checking...", style="cyan")
        elif error_msg:
            footer.append(f"  Error: {error_msg}", style="red")
        else:
            footer.append(f"  Last: {last_check_time}", style="dim")
            footer.append(f"  |  Next: {next_check_time}", style="dim")
        footer.append("  |  Press Ctrl+C to stop", style="dim")

        from rich.console import Group

        content = Group(header, "", table, "", footer)
        return Panel(content, title="EvalView Monitor", border_style="blue")

    with Live(_build_dashboard(), console=console, refresh_per_second=1) as live:
        while not shutdown:
            cycle_count += 1
            checking = True
            last_check_time = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            live.update(_build_dashboard())

            try:
                diffs, results, _, golden_traces = _execute_check_tests(
                    test_cases, config, json_output=True, timeout=timeout
                )
                error_msg = ""
            except Exception as e:
                error_msg = str(e)[:60]
                checking = False
                live.update(_build_dashboard())
                _sleep_interruptible(interval, lambda: shutdown)
                continue

            checking = False
            cycle_cost = sum(r.trace.metrics.total_cost for r in results)
            total_cost += cycle_cost

            for name, diff in diffs:
                current_statuses[name] = diff.overall_severity
                if name not in test_history:
                    test_history[name] = []
                test_history[name].append(diff.overall_severity)

            currently_failing: Set[str] = {
                name for name, diff in diffs if diff.overall_severity in fail_statuses
            }

            # Confirmation gate: suppress n=1 alerts and record noise stats.
            decision = gate.evaluate(currently_failing, strict=strict_tests)
            record_cycle_noise(decision)
            alerts_suppressed += len(decision.self_resolved)

            if not currently_failing and previously_failing and notifiers:
                for label, notifier in notifiers:
                    asyncio.run(notifier.send_recovery_alert(len(diffs)))
                    alerts_sent += 1

            alerts_to_fire = decision.alerts_to_fire
            if alerts_to_fire:
                alert_diffs = [(n, d) for n, d in diffs if n in alerts_to_fire]
                if incidents_path is not None:
                    results_by_name = {r.test_case: r for r in results}
                    _append_incidents(
                        incidents_path,
                        alert_diffs,
                        test_cases_by_name,
                        results_by_name,
                        golden_traces,
                        cycle_count,
                    )
                if notifiers:
                    analysis = _analyze_check_diffs(diffs)
                    incident = detect_coordinated_incident(alert_diffs)
                    for label, notifier in notifiers:
                        asyncio.run(
                            notifier.send_regression_alert(
                                alert_diffs, analysis, incident=incident
                            )
                        )
                        alerts_sent += 1

            spike_alerts = _detect_spikes(results, golden_traces, cost_threshold, latency_threshold)
            if spike_alerts and notifiers:
                for label, notifier in notifiers:
                    asyncio.run(notifier.send_cost_latency_alert(spike_alerts))
                    alerts_sent += 1

            if history_path is not None:
                regressions = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION)
                tools_changed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED)
                output_changed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED)
                passed = len(diffs) - regressions - tools_changed - output_changed
                record = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "cycle": cycle_count,
                    "total_tests": len(diffs),
                    "passed": passed,
                    "regressions": regressions,
                    "tools_changed": tools_changed,
                    "output_changed": output_changed,
                    "cost": round(cycle_cost, 6),
                    "failing_tests": sorted(currently_failing),
                }
                _append_history(history_path, record)

            previously_failing = currently_failing
            next_time = datetime.now(timezone.utc).timestamp() + interval
            next_check_time = datetime.fromtimestamp(next_time, tz=timezone.utc).strftime("%H:%M:%S UTC")
            live.update(_build_dashboard())
            _sleep_interruptible(interval, lambda: shutdown)

    signal.signal(signal.SIGINT, original_sigint)
    console.print(f"\n[cyan]Monitor stopped after {cycle_count} cycle(s).[/cyan]")
    if total_cost > 0:
        console.print(f"[dim]  Total cost: ${total_cost:.4f}[/dim]")
    console.print()


def _sleep_interruptible(seconds: int, should_stop: Any) -> None:
    """Sleep in 1-second ticks so Ctrl+C is responsive."""
    for _ in range(seconds):
        if should_stop():
            break
        time.sleep(1)


@click.command("monitor")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--interval", "-i", type=int, default=None, help="Seconds between checks (default: 300)")
@click.option("--slack-webhook", default=None, help="Slack webhook URL for alerts")
@click.option("--discord-webhook", default=None, help="Discord webhook URL for alerts")
@click.option("--fail-on", default=None, help="Comma-separated statuses that trigger alerts (default: REGRESSION)")
@click.option("--timeout", type=float, default=None, help="Timeout per test in seconds (default: 30)")
@click.option("--test", "-t", "test_filter", default=None, help="Monitor only this specific test")
@click.option(
    "--history",
    "history_path",
    default=None,
    type=click.Path(),
    help="Append each cycle's results to a JSONL file",
)
@click.option("--alert-cost-spike", "cost_spike", type=float, default=None, help="Alert when cost exceeds baseline by this multiplier (e.g. 2.0)")
@click.option("--alert-latency-spike", "latency_spike", type=float, default=None, help="Alert when latency exceeds baseline by this multiplier (e.g. 3.0)")
@click.option(
    "--incidents",
    "incidents_path_opt",
    default=None,
    type=click.Path(),
    is_flag=False,
    flag_value=str(DEFAULT_INCIDENTS_PATH),
    help=(
        "Append every confirmed regression to this JSONL file "
        "(default: .evalview/incidents.jsonl when flag used without a value). "
        "`evalview autopr` reads this to auto-generate regression test PRs. "
        "Pass --no-incidents to disable."
    ),
)
@click.option(
    "--no-incidents",
    "no_incidents",
    is_flag=True,
    default=False,
    help="Disable incident logging even if configured.",
)
@click.option("--dashboard", is_flag=True, help="Live-updating terminal dashboard instead of scrolling logs")
@track_command("monitor")
def monitor(
    test_path: str,
    interval: Optional[int],
    slack_webhook: Optional[str],
    discord_webhook: Optional[str],
    fail_on: Optional[str],
    timeout: Optional[float],
    test_filter: Optional[str],
    history_path: Optional[str],
    cost_spike: Optional[float],
    latency_spike: Optional[float],
    incidents_path_opt: Optional[str],
    no_incidents: bool,
    dashboard: bool = False,
) -> None:
    """Continuously check for regressions with optional webhook alerts.

    Runs evalview check in a loop, alerting you when regressions appear
    and notifying when they are resolved. Designed for production monitoring.

    \b
    Examples:
        evalview monitor                                # Check every 5 min
        evalview monitor --interval 60                  # Check every minute
        evalview monitor --slack-webhook https://...    # Alert to Slack
        evalview monitor --discord-webhook https://...  # Alert to Discord
        evalview monitor --test "weather-lookup"        # Monitor one test
        evalview monitor --fail-on REGRESSION,TOOLS_CHANGED
        evalview monitor --history monitor_log.jsonl    # Persist cycle history
        evalview monitor --alert-cost-spike 2.0         # Alert if cost doubles
        evalview monitor --alert-latency-spike 3.0      # Alert if latency triples
        evalview monitor --incidents                    # Log confirmed failures for `evalview autopr`

    \b
    Configuration (config.yaml):
        monitor:
          interval: 300
          slack_webhook: https://hooks.slack.com/services/...
          discord_webhook: https://discord.com/api/webhooks/...
          fail_on: [REGRESSION]
          cost_threshold: 2.0
          latency_threshold: 3.0

    \b
    Environment variables:
        EVALVIEW_SLACK_WEBHOOK     Slack webhook URL (fallback)
        EVALVIEW_DISCORD_WEBHOOK   Discord webhook URL (fallback)
    """
    from evalview.core.config import apply_judge_config

    config = _load_config_if_exists()
    apply_judge_config(config)
    monitor_cfg = config.get_monitor_config() if config else None

    resolved_interval = interval or (monitor_cfg.interval if monitor_cfg else 300)
    resolved_fail_on = fail_on or (",".join(monitor_cfg.fail_on) if monitor_cfg else "REGRESSION")
    resolved_timeout = timeout or (monitor_cfg.timeout if monitor_cfg else 30.0)

    if resolved_interval < 10:
        click.echo("Error: --interval must be at least 10 seconds.", err=True)
        sys.exit(1)

    if resolved_timeout <= 0:
        click.echo("Error: --timeout must be a positive number.", err=True)
        sys.exit(1)

    resolved_history = Path(history_path) if history_path else None
    resolved_cost_threshold = cost_spike or (monitor_cfg.cost_threshold if monitor_cfg else None)
    resolved_latency_threshold = latency_spike or (monitor_cfg.latency_threshold if monitor_cfg else None)
    resolved_slack_webhook = _resolve_slack_webhook(slack_webhook, config)
    resolved_discord_webhook = _resolve_discord_webhook(discord_webhook, config)

    # Incident logging resolves in this order:
    #   --no-incidents            -> disabled
    #   --incidents [PATH]        -> explicit CLI path (or default when bare)
    #   monitor.incidents_path    -> config value if present
    #   monitor.incidents_enabled -> default path when explicitly enabled
    # When none of the above is set we leave the feature off so existing
    # users see no behavior change until they opt in.
    if no_incidents:
        resolved_incidents: Optional[Path] = None
    elif incidents_path_opt:
        resolved_incidents = Path(incidents_path_opt)
    elif monitor_cfg is not None and getattr(monitor_cfg, "incidents_path", None):
        resolved_incidents = Path(monitor_cfg.incidents_path)
    elif monitor_cfg is not None and getattr(monitor_cfg, "incidents_enabled", False):
        resolved_incidents = DEFAULT_INCIDENTS_PATH
    else:
        resolved_incidents = None

    try:
        if dashboard:
            _run_monitor_dashboard(
                test_path=test_path,
                interval=resolved_interval,
                slack_webhook=resolved_slack_webhook,
                discord_webhook=resolved_discord_webhook,
                fail_on=resolved_fail_on,
                timeout=resolved_timeout,
                test_filter=test_filter,
                config=config,
                history_path=resolved_history,
                cost_threshold=resolved_cost_threshold,
                latency_threshold=resolved_latency_threshold,
                incidents_path=resolved_incidents,
            )
        else:
            _run_monitor_loop(
                test_path=test_path,
                interval=resolved_interval,
                slack_webhook=resolved_slack_webhook,
                discord_webhook=resolved_discord_webhook,
                fail_on=resolved_fail_on,
                timeout=resolved_timeout,
                test_filter=test_filter,
                config=config,
                history_path=resolved_history,
                cost_threshold=resolved_cost_threshold,
                latency_threshold=resolved_latency_threshold,
                incidents_path=resolved_incidents,
            )
    except MonitorError as e:
        console.print(f"[red]ERROR {e}[/red]")
        sys.exit(1)
