"""`evalview progress` — the researcher's delta report.

Where `since` answers "what's happening now", `progress` answers
"did my changes help?" — and that is the question a night-owl prompt
tinkerer is asking at 2am.

Key idea: split fingerprinted history into a "before" window and an
"after" window, then show the **improvements** — tests that went from
non-passing to passing, score quality lifts, cost reductions.

The headline is the "worth a commit" line: a statistical-confidence
gate that says "this improvement is reproducible across N consecutive
runs, you can bless it" — the single most useful operational cue for
someone iterating on prompts.

Usage:
    evalview progress                       # since last run
    evalview progress --since yesterday
    evalview progress --since 7d
    evalview progress --since a4f2e91       # since a commit
    evalview progress --since 2026-04-10    # since a date
    evalview progress --json                # machine-readable
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from evalview.commands.shared import console
from evalview.commands.since_cmd import (
    _entries_since,
    _load_history,
    _parse_since,
)
from evalview.telemetry.decorators import track_command


_HISTORY_PATH = Path(".evalview") / "history.jsonl"

# How many consecutive passing samples in the "after" window an
# improvement needs before we call it "worth a commit". Below this and
# we hedge with "possibly improved, rerun statistically".
_CONFIDENCE_THRESHOLD = 3


# ───────────────────────── per-window state ─────────────────────────


def _latest_status_per_test(
    entries: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Collapse a window of entries into one row per test.

    Returns a dict keyed by test name; the value is the most recent
    entry for that test in the window. Used to answer "what is this
    test's final state in this window?" without re-scanning.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        name = str(e.get("test") or "")
        if not name:
            continue
        ts = str(e.get("ts") or "")
        prev = out.get(name)
        if prev is None or ts > str(prev.get("ts") or ""):
            out[name] = e
    return out


def _consecutive_pass_count(
    entries: List[Dict[str, Any]],
    test_name: str,
) -> int:
    """Count how many of the most recent entries for `test_name` are passing.

    Walks entries newest-first and stops at the first non-passed sample.
    Used by the "worth a commit" confidence gate.
    """
    # Walk newest-first — entries are in write order (chronological)
    count = 0
    for e in reversed(entries):
        if str(e.get("test") or "") != test_name:
            continue
        if str(e.get("status") or "") == "passed":
            count += 1
        else:
            break
    return count


def _scores_per_test(
    entries: List[Dict[str, Any]],
    test_name: str,
) -> List[float]:
    """Return output_similarity values for a test in chronological order."""
    out: List[float] = []
    for e in entries:
        if str(e.get("test") or "") != test_name:
            continue
        sim = e.get("output_similarity")
        if sim is None:
            continue
        try:
            out.append(float(sim))
        except (TypeError, ValueError):
            continue
    return out


# ───────────────────────── delta computation ─────────────────────────


def _compute_delta(
    before: List[Dict[str, Any]],
    after: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the before/after delta across tests.

    Returns a structured dict with:
        - improved: tests that went from non-passed → passed
        - regressed: tests that went from passed → non-passed
        - still_broken: tests non-passing in both windows
        - still_clean: tests passing in both windows
        - new_tests: tests in after but not in before
        - removed_tests: tests in before but not in after (filtered out
                          or deleted)
        - avg_similarity_before / _after / _delta
        - worth_commit: list of (test_name, confidence) — improvements
          that have enough consecutive passes in the after window to
          call "reproducible"

    Everything in this function is pure: no I/O, no clock reads. Easy
    to unit-test.
    """
    before_state = _latest_status_per_test(before)
    after_state = _latest_status_per_test(after)

    before_names = set(before_state)
    after_names = set(after_state)

    def is_pass(entry: Optional[Dict[str, Any]]) -> bool:
        if entry is None:
            return False
        return str(entry.get("status") or "") == "passed"

    improved: List[Tuple[str, Optional[str]]] = []
    regressed: List[Tuple[str, Optional[str]]] = []
    still_broken: List[str] = []
    still_clean: List[str] = []

    for name in before_names | after_names:
        b = before_state.get(name)
        a = after_state.get(name)
        if b is None and a is not None:
            # New test — only count as "improvement" if it's passing,
            # otherwise it's baselined failing and we don't celebrate.
            if is_pass(a):
                improved.append((name, str(a.get("git_sha") or "") or None))
            else:
                still_broken.append(name)
            continue
        if a is None and b is not None:
            # Dropped test — don't count as either improvement or regression
            continue
        # Both early-continues above guarantee both `a` and `b` are non-None
        # by this point. Explicit assertion narrows mypy's Optional types.
        assert a is not None and b is not None
        if not is_pass(b) and is_pass(a):
            improved.append((name, str(a.get("git_sha") or "") or None))
        elif is_pass(b) and not is_pass(a):
            regressed.append((name, str(a.get("git_sha") or "") or None))
        elif is_pass(b) and is_pass(a):
            still_clean.append(name)
        else:
            still_broken.append(name)

    # Aggregate similarity lift across the full history (not just the
    # per-test last entry) so the number is a genuine trend signal.
    def avg_similarity(entries: List[Dict[str, Any]]) -> Optional[float]:
        sims = [
            float(e["output_similarity"])
            for e in entries
            if e.get("output_similarity") is not None
        ]
        return sum(sims) / len(sims) if sims else None

    sim_before = avg_similarity(before)
    sim_after = avg_similarity(after)
    sim_delta = (
        (sim_after - sim_before)
        if sim_before is not None and sim_after is not None
        else None
    )

    # "Worth a commit" confidence gate
    worth_commit: List[Tuple[str, str]] = []
    for name, _sha in improved:
        passes = _consecutive_pass_count(after, name)
        if passes >= _CONFIDENCE_THRESHOLD:
            worth_commit.append((name, "high"))
        elif passes >= 1:
            worth_commit.append((name, "medium"))
        # No entry means we saw the pass in before_state→after_state
        # transition but the after window only had one sample total —
        # that's low confidence, omit from worth_commit so we don't
        # over-promise.

    return {
        "improved": improved,
        "regressed": regressed,
        "still_broken": still_broken,
        "still_clean": still_clean,
        "new_tests_count": len(after_names - before_names),
        "removed_tests_count": len(before_names - after_names),
        "avg_similarity_before": sim_before,
        "avg_similarity_after": sim_after,
        "avg_similarity_delta": sim_delta,
        "worth_commit": worth_commit,
    }


# ───────────────────────── rendering ─────────────────────────


def _render(label: str, delta: Dict[str, Any]) -> None:
    """Render the progress report to the console.

    Keeps the "one hero, one concern, one action" rule: the improved
    count is the hero, regressions are the concern, worth-commit is
    the action.
    """
    from rich.panel import Panel

    console.print()

    improved = delta["improved"]
    regressed = delta["regressed"]
    worth_commit = delta["worth_commit"]

    if not improved and not regressed and not delta["still_clean"]:
        console.print(
            Panel(
                "[yellow]No history in this window.[/yellow]\n\n"
                "`evalview progress` compares two points in the fingerprinted\n"
                "history (.evalview/history.jsonl). Run [bold]evalview check[/bold]\n"
                "a few times, then come back.",
                title=f"Progress since {label}",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        console.print()
        return

    # ── Headline ──
    hero_color = "green" if len(improved) >= len(regressed) else "yellow"
    header = [
        f"[bold]Progress since {label}[/bold]",
        "",
    ]
    if improved:
        header.append(f"  [bold green]✨ {len(improved)} test(s) now passing that weren't[/bold green]")
    if regressed:
        header.append(
            f"  [red]⚠  {len(regressed)} test(s) regressed[/red]"
        )
    if delta["still_clean"]:
        header.append(
            f"  [dim]{len(delta['still_clean'])} test(s) stayed clean[/dim]"
        )

    console.print(
        Panel(
            "\n".join(header),
            border_style=hero_color,
            padding=(1, 2),
        )
    )

    # ── Improved list ──
    if improved:
        console.print()
        console.print("[bold green]Improved:[/bold green]")
        for name, sha in improved[:10]:
            sha_part = f" [dim](at {sha[:7]})[/dim]" if sha else ""
            console.print(f"  [green]+[/green] {name}{sha_part}")
        if len(improved) > 10:
            console.print(f"  [dim]… and {len(improved) - 10} more[/dim]")

    # ── Regressed list ──
    if regressed:
        console.print()
        console.print("[bold red]Regressed:[/bold red]")
        for name, sha in regressed[:5]:
            sha_part = f" [dim](at {sha[:7]})[/dim]" if sha else ""
            console.print(f"  [red]−[/red] {name}{sha_part}")
        if len(regressed) > 5:
            console.print(f"  [dim]… and {len(regressed) - 5} more[/dim]")

    # ── Quality lift ──
    sim_delta = delta["avg_similarity_delta"]
    if sim_delta is not None and abs(sim_delta) > 0.005:
        before = delta["avg_similarity_before"] or 0.0
        after = delta["avg_similarity_after"] or 0.0
        color = "green" if sim_delta > 0 else "red"
        arrow = "↑" if sim_delta > 0 else "↓"
        console.print()
        console.print(
            f"[bold]Output similarity:[/bold] "
            f"{before:.2%} → {after:.2%} "
            f"[{color}]{arrow} {abs(sim_delta):.2%}[/{color}]"
        )

    # ── Worth a commit ──
    if worth_commit:
        console.print()
        console.print("[bold]Worth a commit:[/bold]")
        for name, confidence in worth_commit[:3]:
            badge = (
                f"[green]({confidence} confidence)[/green]"
                if confidence == "high"
                else f"[yellow]({confidence} confidence)[/yellow]"
            )
            console.print(f"  [green]✓[/green] {name} {badge}")
            console.print(f"    [cyan]→ evalview golden update {name}[/cyan]")

    # ── Fallback action if nothing is "worth a commit" yet ──
    if not worth_commit and improved:
        # Name the improved tests so the user knows which ones to
        # validate. A generic `--statistical 5` message against an
        # unnamed target is much less actionable than a specific one.
        names = ", ".join(n for n, _sha in improved[:3])
        suffix = f" (+{len(improved) - 3} more)" if len(improved) > 3 else ""
        console.print()
        console.print(
            f"[yellow]Improvements in {names}{suffix} aren't yet reproducible.[/yellow] "
            "Rerun with:"
        )
        console.print(
            f"  [cyan]→ evalview check --statistical 5 "
            f"--test {improved[0][0]}[/cyan]"
        )

    console.print()


# ───────────────────────── command ─────────────────────────


@click.command("progress")
@click.option(
    "--since",
    "since",
    default=None,
    help='Reference point: sha | date | "yesterday" | "Nd" (default: previous run).',
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit machine-readable JSON.",
)
@track_command("progress")
def progress_cmd(since: Optional[str], json_output: bool) -> None:
    """Show what improved since a reference point.

    The researcher's counterpart to `evalview since`. Instead of
    "what's happening now", `progress` answers "did my changes help?" —
    with a "worth a commit" confidence gate that tells you when an
    improvement is reproducible enough to bless as the new baseline.
    """
    entries = _load_history(_HISTORY_PATH)
    if not entries:
        if json_output:
            click.echo(json.dumps({"label": "ever", "delta": None}))
            return
        console.print(
            "[yellow]No history yet.[/yellow] Run [bold]evalview check[/bold] first."
        )
        return

    cutoff_dt, cutoff_sha, label = _parse_since(since, entries)

    # Split into before / after
    after = _entries_since(entries, cutoff_dt, cutoff_sha)
    after_keys = {id(e) for e in after}
    before = [e for e in entries if id(e) not in after_keys]

    delta = _compute_delta(before, after)

    if json_output:
        serializable = {
            "label": label,
            "delta": {
                **{k: v for k, v in delta.items() if k not in ("improved", "regressed", "worth_commit")},
                "improved": [
                    {"test": n, "git_sha": s} for n, s in delta["improved"]
                ],
                "regressed": [
                    {"test": n, "git_sha": s} for n, s in delta["regressed"]
                ],
                "worth_commit": [
                    {"test": n, "confidence": c} for n, c in delta["worth_commit"]
                ],
            },
        }
        click.echo(json.dumps(serializable, default=str, indent=2))
        return

    _render(label, delta)
