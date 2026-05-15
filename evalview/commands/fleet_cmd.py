"""`evalview fleet` — cross-instance rollup of monitor history files.

A canary instance, a prod instance, a dev's laptop, three regional pods —
they each write their own ``monitor --history`` JSONL. The single-history
commands (``since``, ``progress``, ``drift``) tell you what happened in
one of them. ``fleet`` tells you what's happening across all of them.

    evalview fleet --dir .evalview/history/
    evalview fleet --history monitor-eu.jsonl --history monitor-us.jsonl
    evalview fleet --json > fleet.json
    evalview fleet --anomalies-only      # only show pods deviating from mean
    evalview fleet --require-clean       # CI gate

Pure analytics. No network, no LLM.
"""
from __future__ import annotations

import json
import sys
from typing import Tuple

import click

from evalview.commands.shared import console
from evalview.core.fleet import (
    DEFAULT_ANOMALY_SIGMA,
    DEFAULT_TEST_IMPACT_PCT,
    FleetReport,
    build_fleet_report,
    discover_history_files,
)
from evalview.telemetry.decorators import track_command


def _render(report: FleetReport, anomalies_only: bool) -> None:
    """Pretty-print the report to the console.

    Layout mirrors the slack-digest aesthetic: one hero number, one
    concern, one action. We keep individual instance rows short so the
    table fits in a terminal even with 20+ pods.
    """
    n = len(report.instances)
    if n == 0:
        console.print("[dim]No history files matched. Pass --history FILE or "
                      "--dir DIR pointing at JSONLs written by "
                      "`evalview monitor --history`.[/dim]")
        return

    console.print()
    console.print("[bold]EvalView Fleet[/bold]")
    pct = report.fleet_pass_rate * 100
    color = "green" if pct >= 95 else "yellow" if pct >= 80 else "red"
    console.print(
        f"  Instances: [cyan]{n}[/cyan]  "
        f"|  Cycles: [cyan]{report.fleet_cycles}[/cyan]  "
        f"|  Pass rate: [{color}]{pct:.1f}%[/{color}]  "
        f"|  Cost: [cyan]${report.fleet_cost:.4f}[/cyan]"
    )
    if report.fleet_regressions:
        console.print(
            f"  [red]Regressions across fleet: {report.fleet_regressions}[/red]"
        )

    # Anomalies first — these are the "stop the line" rows. Surface them
    # *before* the full table so they don't get lost in a 20-pod fleet.
    if report.anomalies:
        console.print()
        console.print("[bold yellow]Anomalies[/bold yellow]  "
                      f"[dim](≥ {report.anomaly_sigma:.1f}σ from fleet mean)[/dim]")
        for a in report.anomalies:
            arrow = "↓" if a.direction == "below" else "↑"
            console.print(
                f"  {arrow} [cyan]{a.instance}[/cyan]  "
                f"pass {a.pass_rate * 100:.1f}%  "
                f"({a.sigma_distance:+.1f}σ vs fleet mean {a.fleet_mean * 100:.1f}%)"
            )

    if anomalies_only:
        return

    # Fleet-wide failures: a test failing across many pods means fixing
    # one pod won't help. This is the "is it everywhere?" callout.
    if report.fleet_wide_failures:
        console.print()
        console.print(
            "[bold]Fleet-wide failing tests[/bold]  "
            f"[dim](≥ {report.test_impact_pct * 100:.0f}% of instances)[/dim]"
        )
        for f in report.fleet_wide_failures:
            console.print(
                f"  • [red]{f.test_name}[/red]  "
                f"({f.impact_pct * 100:.0f}% — "
                f"{', '.join(f.affected_instances[:4])}"
                f"{', …' if len(f.affected_instances) > 4 else ''})"
            )

    console.print()
    console.print("[bold]Per-instance[/bold]")
    # Sort: anomalies bubble up via the report's own pass_rate sort.
    for s in report.instances:
        rate = s.pass_rate * 100
        bar = "▇" * min(int(rate / 10), 10) + "·" * max(0, 10 - int(rate / 10))
        color = "green" if rate >= 95 else "yellow" if rate >= 80 else "red"
        failing_part = (
            f"  [dim]{len(s.failing_tests)} failing[/dim]" if s.failing_tests else ""
        )
        console.print(
            f"  [{color}]{bar}[/{color}]  "
            f"[cyan]{s.instance:<20.20}[/cyan]  "
            f"{rate:5.1f}%  "
            f"[dim]{s.cycles:>4} cyc[/dim]  "
            f"[dim]${s.cost:.4f}[/dim]"
            f"{failing_part}"
        )


def _verdict(report: FleetReport) -> Tuple[str, int]:
    """Return ``(label, exit_code)`` for ``--require-clean``.

    Clean = no anomalies, no fleet-wide failures, no regressions. We're
    intentionally strict here: ``--require-clean`` is for the CI gate
    use case where any cross-instance signal should block.
    """
    if report.fleet_regressions or report.anomalies or report.fleet_wide_failures:
        return "fleet has cross-instance signal — investigate", 1
    return "fleet is clean across all instances", 0


@click.command("fleet")
@click.option(
    "--history",
    "history_paths",
    multiple=True,
    type=click.Path(),
    help="Path to a monitor history JSONL. Repeat for multiple instances. "
         "Globs are expanded. Combine with --dir.",
)
@click.option(
    "--dir",
    "directories",
    multiple=True,
    type=click.Path(),
    help="Scan this directory for *.jsonl files (non-recursive). "
         "Repeat for multiple roots.",
)
@click.option(
    "--anomaly-sigma",
    type=float,
    default=DEFAULT_ANOMALY_SIGMA,
    show_default=True,
    help="Z-score threshold for flagging an instance as anomalous.",
)
@click.option(
    "--test-impact-pct",
    type=float,
    default=DEFAULT_TEST_IMPACT_PCT,
    show_default=True,
    help="Fraction of instances a test must fail in to count as fleet-wide.",
)
@click.option(
    "--anomalies-only",
    is_flag=True,
    help="Skip the per-instance and fleet-wide tables; print only anomalies.",
)
@click.option(
    "--require-clean",
    is_flag=True,
    help="Exit 1 if the fleet has any regression, anomaly, or fleet-wide "
         "failure. Useful in CI to block deploys when canary diverges.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit a machine-readable JSON report to stdout.",
)
@track_command("fleet")
def fleet_cmd(
    history_paths: Tuple[str, ...],
    directories: Tuple[str, ...],
    anomaly_sigma: float,
    test_impact_pct: float,
    anomalies_only: bool,
    require_clean: bool,
    json_output: bool,
) -> None:
    """Roll up multiple monitor history files into one fleet view.

    Each ``--history FILE`` (or each ``*.jsonl`` in a ``--dir``) is one
    monitor session — typically one agent instance or one region. The
    command produces a fleet summary, per-instance breakdown, anomaly
    callouts (instances ≥ 2σ off the fleet mean), and fleet-wide
    failures (tests failing in ≥ 40% of instances).

    \b
    Examples:
        evalview fleet --dir .evalview/history/
        evalview fleet --history monitor-eu.jsonl --history monitor-us.jsonl
        evalview fleet --json > fleet.json
        evalview fleet --anomalies-only
        evalview fleet --require-clean   # CI gate

    History files are produced by `evalview monitor --history FILE.jsonl`.
    """
    files = discover_history_files(history_paths, directories)
    report = build_fleet_report(
        files,
        anomaly_sigma=anomaly_sigma,
        test_impact_pct=test_impact_pct,
    )

    if json_output:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        _render(report, anomalies_only=anomalies_only)

    if require_clean:
        label, code = _verdict(report)
        if not json_output:
            color = "green" if code == 0 else "red"
            console.print()
            console.print(f"  [{color}]{label}[/{color}]")
        sys.exit(code)
