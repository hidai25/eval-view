"""`evalview drift` — per-test drift visualization + incident markers.

Where the Week 1 DriftTracker detects drift mathematically (OLS slope
below a threshold), this command visualizes it: sparklines per test so
a human can eyeball which tests are trending down, which are oscillating,
and which are rock-solid.

Incident markers ("!" above a sample) flag checks where the test flipped
status — e.g. passed → regression — so you can see "oh, the drift started
right after that prompt change on Tuesday."

Usage:
    evalview drift                        # all tests, last 20 samples
    evalview drift my-test                # one test, full history
    evalview drift --last 7d              # only samples from the last week
    evalview drift --worst                # sort by steepest decline first
    evalview drift --json                 # machine-readable
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from evalview.commands.shared import console
from evalview.commands.since_cmd import _load_history
from evalview.core.trends import compute_slope
from evalview.telemetry.decorators import track_command


_HISTORY_PATH = Path(".evalview") / "history.jsonl"

# Default sparkline width — also the default per-test sample cap so
# visualization stays legible. Override with --last for a time filter.
_DEFAULT_SPARK_WIDTH = 20

# Thresholds for drift severity. Same defaults as DriftTracker so the
# CLI and the verdict layer agree on what "drifting" means.
_SLOPE_WARN = -0.01    # mild downward trend
_SLOPE_BAD = -0.02     # concerning — matches verdict flip threshold


_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float]) -> str:
    """Render a unicode sparkline for a series of values.

    Normalizes to the min/max of the series. All-equal values render as
    mid-height bars so the user still sees the shape (vs. showing zeros
    which look broken).
    """
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return _SPARK_GLYPHS[4] * len(values)
    out = ""
    for v in values:
        idx = int((v - lo) / span * (len(_SPARK_GLYPHS) - 1))
        out += _SPARK_GLYPHS[idx]
    return out


def _status_transitions(
    series: List[Tuple[str, str]],
) -> List[int]:
    """Return indices where the test's status flipped between samples.

    `series` is a list of (ts, status) tuples in chronological order.
    Used for incident markers in the rendered sparkline — a `!` above
    the index where a regression first appeared.
    """
    out: List[int] = []
    for i in range(1, len(series)):
        if series[i][1] != series[i - 1][1]:
            out.append(i)
    return out


def _parse_last(last: Optional[str]) -> Optional[timedelta]:
    """Parse `--last 7d | 24h | 30m` into a timedelta. None means no limit."""
    if not last:
        return None
    raw = last.strip().lower()
    for suffix, factor in (("d", 86400), ("h", 3600), ("m", 60)):
        if raw.endswith(suffix):
            try:
                n = int(raw[: -len(suffix)].strip())
                return timedelta(seconds=n * factor)
            except ValueError:
                break
    return None


def _filter_by_window(
    entries: List[Dict[str, Any]],
    window: Optional[timedelta],
) -> List[Dict[str, Any]]:
    if window is None:
        return list(entries)
    cutoff = datetime.now(timezone.utc) - window
    out: List[Dict[str, Any]] = []
    for e in entries:
        ts_raw = str(e.get("ts") or "")
        try:
            dt = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt >= cutoff:
            out.append(e)
    return out


def _per_test_series(
    entries: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group entries by test name, preserving chronological order."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        name = str(e.get("test") or "")
        if not name:
            continue
        out.setdefault(name, []).append(e)
    return out


def _classify(slope: float, sample_count: int) -> Tuple[str, str]:
    """Classify a drift slope into a color + label.

    Returns (color, label) where color is a Rich style and label is a
    short human word. Labels match drift_tracker.classify_drift so the
    CLI and the verdict layer share vocabulary.
    """
    if sample_count < 3:
        return ("dim", "insufficient_history")
    if slope <= _SLOPE_BAD:
        return ("red", "declining")
    if slope <= _SLOPE_WARN:
        return ("yellow", "soft decline")
    if slope >= abs(_SLOPE_WARN):
        return ("green", "improving")
    return ("dim", "stable")


# ───────────────────────── rendering ─────────────────────────


def _build_rows(
    per_test: Dict[str, List[Dict[str, Any]]],
    sort_worst: bool,
    sample_cap: int,
) -> List[Dict[str, Any]]:
    """Return renderable rows sorted by test name or by worst-drift-first."""
    rows: List[Dict[str, Any]] = []
    for name, series in per_test.items():
        # Keep only the last N samples so a long history doesn't
        # dominate the sparkline (slope is still computed on the
        # retained window).
        trimmed = series[-sample_cap:]
        sims = [
            float(e.get("output_similarity") or 0.0)
            for e in trimmed
            if e.get("output_similarity") is not None
        ]
        if not sims:
            continue
        slope = compute_slope(sims)
        color, label = _classify(slope, len(sims))

        status_series = [
            (str(e.get("ts") or ""), str(e.get("status") or ""))
            for e in trimmed
        ]
        incidents = _status_transitions(status_series)

        rows.append(
            {
                "test": name,
                "samples": len(sims),
                "slope": slope,
                "color": color,
                "label": label,
                "spark": _sparkline(sims),
                "incidents": incidents,
                "first": sims[0],
                "last": sims[-1],
            }
        )

    if sort_worst:
        rows.sort(key=lambda r: r["slope"])
    else:
        rows.sort(key=lambda r: r["test"])
    return rows


def _incident_markers(spark: str, incidents: List[int]) -> str:
    """Return a marker row aligned with the sparkline.

    A `!` glyph appears under (actually above — prints on the preceding
    line) each sample where the status changed. Empty otherwise.
    """
    if not incidents:
        return " " * len(spark)
    chars = [" "] * len(spark)
    for i in incidents:
        if 0 <= i < len(chars):
            chars[i] = "!"
    return "".join(chars)


def _render(rows: List[Dict[str, Any]]) -> None:
    from rich.table import Table

    if not rows:
        console.print(
            "[yellow]No drift history yet.[/yellow] Run "
            "[bold]evalview check[/bold] a few times to collect samples."
        )
        return

    table = Table(
        title=f"Drift per test ({len(rows)} tracked)",
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Test", no_wrap=True)
    table.add_column("Trend", no_wrap=True)
    table.add_column("Samples", justify="right")
    table.add_column("Slope", justify="right")
    table.add_column("First → Last", no_wrap=True)
    table.add_column("Status", no_wrap=True)

    for row in rows:
        spark_cell = f"[{row['color']}]{row['spark']}[/{row['color']}]"
        if row["incidents"]:
            # Show markers above the sparkline via a two-line cell.
            markers = _incident_markers(row["spark"], row["incidents"])
            spark_cell = f"[red]{markers}[/red]\n{spark_cell}"

        slope_pct = row["slope"] * 100
        slope_str = f"{slope_pct:+.1f}%"

        table.add_row(
            row["test"],
            spark_cell,
            str(row["samples"]),
            slope_str,
            f"{row['first']:.0%} → {row['last']:.0%}",
            f"[{row['color']}]{row['label']}[/{row['color']}]",
        )

    console.print(table)

    # ── Summary line: one thing to look at first ──
    worst = min(rows, key=lambda r: r["slope"])
    if worst["slope"] <= _SLOPE_WARN and worst["samples"] >= 3:
        console.print()
        console.print(
            f"[yellow]Most concerning:[/yellow] [bold]{worst['test']}[/bold] "
            f"— slope {worst['slope'] * 100:+.1f}% per check over "
            f"{worst['samples']} samples"
        )
        console.print(f"  [cyan]→ evalview replay {worst['test']} --trace[/cyan]")
    elif all(r["slope"] > _SLOPE_WARN for r in rows):
        console.print()
        console.print("[green]No concerning drift detected.[/green]")
    console.print()


# ───────────────────────── command ─────────────────────────


@click.command("drift")
@click.argument("test_name", required=False, default=None)
@click.option("--last", "last", default=None,
              help="Time window: 7d | 24h | 30m (default: all history).")
@click.option("--worst", is_flag=True,
              help="Sort tests by steepest decline first.")
@click.option("--json", "json_output", is_flag=True,
              help="Emit machine-readable JSON.")
@click.option("--width", type=int, default=_DEFAULT_SPARK_WIDTH,
              help="Sparkline width (and sample cap per test).")
@track_command("drift")
def drift_cmd(
    test_name: Optional[str],
    last: Optional[str],
    worst: bool,
    json_output: bool,
    width: int,
) -> None:
    """Per-test drift sparklines with incident markers.

    Reads .evalview/history.jsonl (populated by every `evalview check`)
    and renders a sparkline per test alongside its OLS slope.

    Tests above the "concerning" threshold are colored red, soft
    declines yellow, improvements green.
    """
    entries = _load_history(_HISTORY_PATH)
    if not entries:
        if json_output:
            click.echo(json.dumps({"rows": []}))
            return
        console.print(
            "[yellow]No drift history yet.[/yellow] Run "
            "[bold]evalview check[/bold] first."
        )
        return

    window = _parse_last(last)
    entries = _filter_by_window(entries, window)

    per_test = _per_test_series(entries)
    if test_name:
        per_test = {k: v for k, v in per_test.items() if k == test_name}
        if not per_test:
            console.print(f"[red]No drift history for test '{test_name}'[/red]")
            return

    rows = _build_rows(per_test, sort_worst=worst, sample_cap=max(3, width))

    if json_output:
        click.echo(json.dumps({"rows": rows}, default=str, indent=2))
        return

    _render(rows)
