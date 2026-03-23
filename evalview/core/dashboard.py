"""Visual dashboard components for check command output.

Pure rendering functions that take data and return Rich renderables.
No I/O, no side effects — all data is passed in as arguments.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from rich.panel import Panel

# Unicode block characters for sparkline rendering (8 levels)
_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float], low: float = 0.0, high: float = 1.0) -> str:
    """Convert a list of values into a Unicode sparkline string.

    Args:
        values: Data points to visualize.
        low: Minimum value for scaling (default 0.0).
        high: Maximum value for scaling (default 1.0).

    Returns:
        String of Unicode block characters representing the trend.
    """
    if not values:
        return ""
    span = high - low if high != low else 1.0
    return "".join(
        _SPARK_CHARS[min(7, max(0, int((v - low) / span * 8)))]
        for v in values
    )


def render_scorecard(
    passed: int,
    tools_changed: int,
    output_changed: int,
    regressions: int,
    execution_failures: int,
    current_streak: int,
    longest_streak: int,
    health_pct: float,
) -> Panel:
    """Build the dashboard scorecard as a Rich Panel.

    Shows a colored bar, summary stats, and health gauge in a single panel.

    Args:
        passed: Number of tests that passed.
        tools_changed: Number of tests with tool changes.
        output_changed: Number of tests with output changes.
        regressions: Number of regressions detected.
        execution_failures: Number of tests that failed to execute.
        current_streak: Current consecutive clean check streak.
        longest_streak: Best streak ever achieved.
        health_pct: Health percentage (0-100).

    Returns:
        Rich Panel containing the scorecard.
    """
    total = passed + tools_changed + output_changed + regressions + execution_failures
    bar_width = 40

    # Build colored bar
    if total == 0:
        bar = "[dim]" + "░" * bar_width + "[/dim]"
    else:
        g = int(bar_width * passed / total)
        y = int(bar_width * (tools_changed + output_changed) / total)
        r = int(bar_width * (regressions + execution_failures) / total)
        # Distribute remainder to largest segment
        remainder = bar_width - g - y - r
        if g >= y and g >= r:
            g += remainder
        elif y >= r:
            y += remainder
        else:
            r += remainder
        bar = ""
        if g > 0:
            bar += "[green]" + "█" * g + "[/green]"
        if y > 0:
            bar += "[yellow]" + "█" * y + "[/yellow]"
        if r > 0:
            bar += "[red]" + "█" * r + "[/red]"

    # Stats line
    parts = []
    if passed:
        parts.append(f"[green]{passed} passed[/green]")
    changed = tools_changed + output_changed
    if changed:
        parts.append(f"[yellow]{changed} changed[/yellow]")
    if regressions:
        parts.append(f"[red]{regressions} {'regression' if regressions == 1 else 'regressions'}[/red]")
    if execution_failures:
        parts.append(f"[red]{execution_failures} {'failure' if execution_failures == 1 else 'failures'}[/red]")

    stats_line = " · ".join(parts) if parts else "[dim]No tests compared[/dim]"

    # Streak info
    streak_part = ""
    if current_streak > 0 or longest_streak > 0:
        streak_parts = []
        if current_streak > 0:
            streak_parts.append(f"Streak: {current_streak}")
        if longest_streak > 0 and longest_streak != current_streak:
            streak_parts.append(f"Best: {longest_streak}")
        streak_part = "  |  " + " · ".join(streak_parts)

    # Health gauge (10 circles)
    filled = max(0, min(10, int(health_pct / 10)))
    empty = 10 - filled
    if health_pct >= 80:
        gauge_color = "green"
    elif health_pct >= 50:
        gauge_color = "yellow"
    else:
        gauge_color = "red"

    filled_circles = "●" * filled
    empty_circles = "○" * empty
    gauge = f"[{gauge_color}]{filled_circles}[/{gauge_color}][dim]{empty_circles}[/dim]"
    health_line = f"Health: [{gauge_color}]{health_pct:.0f}%[/{gauge_color}]  {gauge}"

    # Assemble panel content
    content = f"  {bar}\n\n  {stats_line}{streak_part}\n  {health_line}"

    # Panel border color matches health
    return Panel(
        content,
        title="Check Dashboard",
        border_style=gauge_color,
        padding=(1, 1),
    )


def render_sparklines(
    test_trends: Dict[str, List[float]],
    overall_pass_trend: List[float],
) -> Optional[str]:
    """Build sparkline trend display using Unicode block characters.

    Args:
        test_trends: Map of test_name -> list of output_similarity values
                     (oldest first, up to last 10).
        overall_pass_trend: Overall pass rate per cycle (oldest first).

    Returns:
        Formatted string for console.print(), or None if no data.
    """
    if not test_trends and not overall_pass_trend:
        return None

    lines = ["[bold]Trends[/bold] [dim](last 10 checks)[/dim]\n"]

    # Per-test output similarity trends
    for name, values in test_trends.items():
        spark = _sparkline(values)
        if not spark:
            continue

        # Trend direction
        if len(values) >= 2:
            first, last = values[0], values[-1]
            first_pct = f"{first:.2f}"
            last_pct = f"{last:.2f}"
            if last > first + 0.02:
                direction = f"[green]{first_pct} → {last_pct} ↑[/green]"
            elif last < first - 0.02:
                direction = f"[red]{first_pct} → {last_pct} ↓[/red]"
            else:
                direction = f"[dim]{first_pct} → {last_pct} =[/dim]"
        else:
            direction = f"[dim]{values[0]:.2f}[/dim]"

        # Pad test name for alignment
        padded_name = name[:24].ljust(24)
        lines.append(f"  [cyan]{padded_name}[/cyan]  {spark}  {direction}")

    # Overall pass rate trend
    if overall_pass_trend and len(overall_pass_trend) >= 2:
        spark = _sparkline(overall_pass_trend)
        first_pct = f"{overall_pass_trend[0] * 100:.0f}%"
        last_pct = f"{overall_pass_trend[-1] * 100:.0f}%"
        padded = "Overall pass rate".ljust(24)
        lines.append(f"\n  [bold]{padded}[/bold]  {spark}  {first_pct} → {last_pct}")

    return "\n".join(lines) if len(lines) > 1 else None


def render_confidence_label(
    confidence_pct: float,
    confidence_label: str,
) -> str:
    """Format a confidence annotation for a test verdict line.

    Args:
        confidence_pct: Confidence percentage (0-99).
        confidence_label: One of "high", "medium", "low", "insufficient_history".

    Returns:
        Rich-formatted string describing the confidence level.
    """
    if confidence_label == "insufficient_history":
        return "[dim](insufficient history)[/dim]"
    elif confidence_label == "high":
        return f"[bold]({confidence_pct:.0f}% confidence — outside normal variance)[/bold]"
    elif confidence_label == "medium":
        return f"[yellow]({confidence_pct:.0f}% confidence — borderline signal)[/yellow]"
    else:  # low
        return f"[dim]({confidence_pct:.0f}% confidence — likely noise)[/dim]"


def render_smart_accept_suggestion(
    test_name: str,
    score_improved: bool,
    tools_changed: bool,
    baseline_tools: List[str],
    current_tools: List[str],
    baseline_score: float,
    current_score: float,
) -> Optional[Panel]:
    """Render a suggestion to accept an intentional-looking change.

    Returns a Panel with the suggestion if the change looks intentional
    (score improved or stayed stable), or None if the change looks like
    a genuine regression.

    Args:
        test_name: Name of the test.
        score_improved: Whether the score went up.
        tools_changed: Whether tools changed.
        baseline_tools: Tool sequence from the baseline.
        current_tools: Tool sequence from the current run.
        baseline_score: Score from the baseline.
        current_score: Score from the current run.

    Returns:
        Rich Panel with accept suggestion, or None.
    """
    score_diff = current_score - baseline_score

    # Only suggest accepting if score improved or stayed roughly the same
    if score_diff < -2.0:
        return None  # Score dropped significantly — doesn't look intentional

    # Build diff preview
    parts = []

    if tools_changed and baseline_tools != current_tools:
        bl = ", ".join(baseline_tools) if baseline_tools else "(none)"
        cl = ", ".join(current_tools) if current_tools else "(none)"
        parts.append(f"  Tools:  [dim]{bl}[/dim]  →  [bold]{cl}[/bold]")

    sign = "+" if score_diff > 0 else ""
    score_color = "green" if score_diff >= 0 else "yellow"
    parts.append(
        f"  Score:  {baseline_score:.0f} → {current_score:.0f}  "
        f"[{score_color}]({sign}{score_diff:.0f})[/{score_color}]"
    )

    # Build suggestion
    quoted = f'"{test_name}"' if " " in test_name else test_name
    hint = "Score improved — this looks intentional." if score_improved else "Score is stable — this may be intentional."

    content = (
        f"[green]{hint}[/green]\n\n"
        + "\n".join(parts)
        + f"\n\n  Accept:   [bold]evalview snapshot --test {quoted}[/bold]"
        + f"\n  Preview:  [dim]evalview snapshot --test {quoted} --preview[/dim]"
    )

    return Panel(
        content,
        title="💡 Accept this change?",
        border_style="green" if score_improved else "yellow",
        padding=(0, 1),
    )
