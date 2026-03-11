"""Monitor command — continuous regression detection with Slack alerts."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Set

import click

from evalview.commands.shared import console, _load_config_if_exists
from evalview.commands.check_cmd import (
    _execute_check_tests,
    _analyze_check_diffs,
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


def _run_monitor_loop(
    test_path: str,
    interval: int,
    slack_webhook: Optional[str],
    fail_on: str,
    timeout: float,
    test_filter: Optional[str],
) -> None:
    """Main monitor loop. Runs check cycles until Ctrl+C.

    Raises:
        MonitorError: If baselines or test cases are missing.
    """
    from evalview.core.loader import TestCaseLoader
    from evalview.core.golden import GoldenStore
    from evalview.core.messages import (
        get_random_monitor_start_message,
        get_random_monitor_cycle_message,
        get_random_monitor_clean_message,
    )
    from evalview.core.slack_notifier import SlackNotifier
    from evalview.core.config import apply_judge_config

    config = _load_config_if_exists()
    apply_judge_config(config)

    webhook_url = _resolve_slack_webhook(slack_webhook, config)
    notifier = SlackNotifier(webhook_url) if webhook_url else None

    # Verify snapshots exist
    store = GoldenStore()
    goldens = store.list_golden()
    if not goldens:
        raise MonitorError("No baselines found. Run `evalview snapshot` first.")

    # Load test cases
    loader = TestCaseLoader()
    test_cases = loader.load_from_directory(Path(test_path))

    if test_filter:
        test_cases = [tc for tc in test_cases if tc.name == test_filter]
        if not test_cases:
            raise MonitorError(f"No test found with name: {test_filter}")

    console.print(f"\n[cyan]{get_random_monitor_start_message()}[/cyan]")
    console.print(f"[dim]  Tests: {len(test_cases)}  |  Interval: {interval}s  |  Slack: {'✓' if notifier else '—'}[/dim]")
    console.print(f"[dim]  Press Ctrl+C to stop.[/dim]\n")

    # Track state across cycles
    previously_failing: Set[str] = set()
    cycle_count = 0
    total_cost = 0.0
    shutdown = False

    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig: int, frame: Any) -> None:
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)

    fail_statuses = set(s.strip().upper() for s in fail_on.split(","))

    try:
        while not shutdown:
            cycle_count += 1
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            console.print(f"[dim][{now}][/dim] {get_random_monitor_cycle_message()}")

            try:
                diffs, results, _, _ = _execute_check_tests(
                    test_cases, config, json_output=True, timeout=timeout
                )
            except Exception as e:
                console.print(f"[red]  ✗ Cycle {cycle_count} failed: {e}[/red]")
                _sleep_interruptible(interval, lambda: shutdown)
                continue

            # Track cumulative cost
            cycle_cost = sum(r.trace.metrics.total_cost for r in results)
            total_cost += cycle_cost

            analysis = _analyze_check_diffs(diffs)

            # Determine currently failing tests (only those matching fail_on)
            currently_failing: Set[str] = set()
            for name, diff in diffs:
                if diff.overall_severity.value.upper() in fail_statuses:
                    currently_failing.add(name)

            if not currently_failing:
                cost_part = f"  [dim]${cycle_cost:.4f}[/dim]" if cycle_cost > 0 else ""
                console.print(f"[green]  {get_random_monitor_clean_message()} ({len(diffs)} tests){cost_part}[/green]")

                # Send recovery if we were failing before
                if previously_failing and notifier:
                    asyncio.run(notifier.send_recovery_alert(len(diffs)))
                    console.print("[dim]  📤 Slack: recovery notification sent[/dim]")
            else:
                # Show summary using analysis dict
                parts = []
                if analysis["has_regressions"]:
                    count = sum(1 for _, d in diffs if d.overall_severity.value == "REGRESSION")
                    parts.append(f"[red]{count} regression{'s' if count > 1 else ''}[/red]")
                if analysis["has_tools_changed"]:
                    count = sum(1 for _, d in diffs if d.overall_severity.value == "TOOLS_CHANGED")
                    parts.append(f"[yellow]{count} tool change{'s' if count > 1 else ''}[/yellow]")
                if analysis["has_output_changed"]:
                    count = sum(1 for _, d in diffs if d.overall_severity.value == "OUTPUT_CHANGED")
                    parts.append(f"[dim]{count} output change{'s' if count > 1 else ''}[/dim]")

                console.print(f"  ⚠  {', '.join(parts)}")

                for name, diff in diffs:
                    if diff.overall_severity.value.upper() in fail_statuses:
                        console.print(f"    [red]✗ {name}[/red] ({diff.overall_severity.value})")

                # Only alert on NEW failures (avoid spamming)
                new_failures = currently_failing - previously_failing
                if new_failures and notifier:
                    alert_diffs = [(n, d) for n, d in diffs if n in currently_failing]
                    asyncio.run(notifier.send_regression_alert(alert_diffs, analysis))
                    console.print(f"[dim]  📤 Slack: alerted on {len(new_failures)} new failure(s)[/dim]")

            previously_failing = currently_failing

            _sleep_interruptible(interval, lambda: shutdown)
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    # Summary on exit
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
@click.option("--fail-on", default=None, help="Comma-separated statuses that trigger alerts (default: REGRESSION)")
@click.option("--timeout", type=float, default=None, help="Timeout per test in seconds (default: 30)")
@click.option("--test", "-t", "test_filter", default=None, help="Monitor only this specific test")
@track_command("monitor")
def monitor(
    test_path: str,
    interval: Optional[int],
    slack_webhook: Optional[str],
    fail_on: Optional[str],
    timeout: Optional[float],
    test_filter: Optional[str],
) -> None:
    """Continuously check for regressions with optional Slack alerts.

    Runs evalview check in a loop, alerting you when regressions appear
    and notifying when they're resolved. Designed for production monitoring.

    \b
    Examples:
        evalview monitor                                # Check every 5 min
        evalview monitor --interval 60                  # Check every minute
        evalview monitor --slack-webhook https://...    # Alert to Slack
        evalview monitor --test "weather-lookup"        # Monitor one test
        evalview monitor --fail-on REGRESSION,TOOLS_CHANGED

    \b
    Configuration (config.yaml):
        monitor:
          interval: 300
          slack_webhook: https://hooks.slack.com/services/...
          fail_on: [REGRESSION]

    \b
    Environment variables:
        EVALVIEW_SLACK_WEBHOOK    Slack webhook URL (fallback)
    """
    # Resolve defaults from config file
    config = _load_config_if_exists()
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

    try:
        _run_monitor_loop(
            test_path=test_path,
            interval=resolved_interval,
            slack_webhook=_resolve_slack_webhook(slack_webhook, config),
            fail_on=resolved_fail_on,
            timeout=resolved_timeout,
            test_filter=test_filter,
        )
    except MonitorError as e:
        console.print(f"[red]❌ {e}[/red]")
        sys.exit(1)
