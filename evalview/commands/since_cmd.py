"""`evalview since` — the daily habit brief.

Answers "what's changed since I last looked?" in under 2 seconds.

Profiles it serves:
  - Daily shipper: defaults to "since yesterday" if last check was today,
    otherwise "since last run"
  - Night owl: "since last run" works regardless of clock time
  - Solo indie: no team noise, just what changed here
  - Researcher: pair with --since <sha> to see what moved between commits

Usage examples:
    evalview since                        # since your last check
    evalview since --since yesterday      # last 24h
    evalview since --since 7d             # last 7 days
    evalview since --since 2026-04-10     # since a date
    evalview since --since a4f2e91        # since a git SHA (reads fingerprints)
    evalview since --json                 # machine-readable for cloud / CI

Design rules:
  - Under 2 seconds on a cold filesystem — no subprocess, no network
  - One hero number (pass rate), one drift concern, one action
  - Never crashes on missing history; a fresh project gets a nudge, not a stack trace
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


_HISTORY_PATH = Path(".evalview") / "history.jsonl"


# ───────────────────────── history loading ─────────────────────────


def _load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return entries


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


# ───────────────────────── `--since` parsing ─────────────────────────


def _parse_since(
    since: Optional[str],
    entries: List[Dict[str, Any]],
) -> Tuple[Optional[datetime], Optional[str], str]:
    """Resolve `--since` into a (cutoff_datetime, cutoff_sha, label).

    Accepts:
        - None                   → "since your previous check run"
        - "yesterday"            → 24 hours ago
        - "Nd" / "N days"        → N days ago
        - ISO date "2026-04-10"  → that date at 00:00
        - git SHA (7+ hex chars) → first entry with that sha
        - "last"                 → same as None (previous run)

    Returns a tuple `(cutoff_dt, cutoff_sha, label)` where either
    `cutoff_dt` or `cutoff_sha` is set (or both None, meaning "all
    history"). `label` is the human string for the brief header.

    This function never raises — an unrecognized value silently falls
    back to "previous run" and logs nothing. The brief is meant to be
    cheap and forgiving, not pedantic.
    """
    if since is None or since.strip() == "" or since.strip().lower() == "last":
        cutoff = _cutoff_of_previous_run(entries)
        if cutoff is None:
            return (None, None, "ever")
        return (cutoff, None, "your last check")

    raw = since.strip().lower()

    if raw == "yesterday":
        return (
            datetime.now(timezone.utc) - timedelta(days=1),
            None,
            "yesterday",
        )

    # "Nd" or "N days" — integer days lookback
    for suffix in ("d", " days", " day"):
        if raw.endswith(suffix):
            try:
                n = int(raw[: -len(suffix)].strip())
                return (
                    datetime.now(timezone.utc) - timedelta(days=n),
                    None,
                    f"{n} day{'s' if n != 1 else ''} ago",
                )
            except ValueError:
                break

    # ISO date
    try:
        dt = datetime.fromisoformat(since.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (dt, None, since.strip())
    except ValueError:
        pass

    # git SHA (hex, 7+ chars) — look up the first entry that matches
    cleaned = since.strip()
    if len(cleaned) >= 7 and all(c in "0123456789abcdef" for c in cleaned.lower()):
        for e in entries:
            sha = str(e.get("git_sha") or "").lower()
            if sha and cleaned.lower().startswith(sha[:7]):
                return (None, sha, f"commit {sha}")

    # Unknown → fall back to previous run, but keep the raw label so the
    # user sees we heard them.
    cutoff = _cutoff_of_previous_run(entries)
    return (cutoff, None, since.strip())


def _cutoff_of_previous_run(entries: List[Dict[str, Any]]) -> Optional[datetime]:
    """Find the timestamp that separates the previous run from this one.

    "Previous run" = the most recent *distinct* run in the history. Runs
    are distinguished by git_sha + minute-level timestamp (same logic as
    `evalview log`). The cutoff is the latest timestamp in the previous
    run so "since last" includes anything newer.
    """
    if not entries:
        return None

    def run_key(e: Dict[str, Any]) -> Tuple[str, str, str]:
        sha = str(e.get("git_sha") or "")
        prompt = str(e.get("prompt_hash") or "")
        ts = str(e.get("ts") or "")
        if sha or prompt:
            ts_bucket = ts[:16]
        else:
            ts_bucket = ts[:19]
        return (sha, prompt, ts_bucket)

    distinct_runs: Dict[Tuple[str, str, str], datetime] = {}
    for e in entries:
        ts = _parse_ts(e.get("ts"))
        if ts is None:
            continue
        key = run_key(e)
        if key not in distinct_runs or ts > distinct_runs[key]:
            distinct_runs[key] = ts

    if len(distinct_runs) < 2:
        return None
    sorted_runs = sorted(distinct_runs.values(), reverse=True)
    # Return the max timestamp of the 2nd-most-recent run — entries
    # strictly newer than this belong to the current run.
    return sorted_runs[1]


# ───────────────────────── aggregation ─────────────────────────


def _entries_since(
    entries: List[Dict[str, Any]],
    cutoff_dt: Optional[datetime],
    cutoff_sha: Optional[str],
) -> List[Dict[str, Any]]:
    if cutoff_sha:
        seen_sha = False
        out: List[Dict[str, Any]] = []
        # Walk from oldest to newest, keep entries AFTER first seeing the sha
        for e in entries:
            sha = str(e.get("git_sha") or "").lower()
            if seen_sha:
                out.append(e)
            elif sha.startswith(cutoff_sha[:7]):
                seen_sha = True
        return out

    if cutoff_dt is None:
        return list(entries)

    out = []
    for e in entries:
        ts = _parse_ts(e.get("ts"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts > cutoff_dt:
            out.append(e)
    return out


def _summarize(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a slice of history into the brief's headline numbers."""
    if not entries:
        return {
            "total": 0,
            "passed": 0,
            "regression": 0,
            "tools_changed": 0,
            "output_changed": 0,
            "pass_rate": None,
            "avg_score_diff": 0.0,
            "models": [],
            "tests_improved": [],
            "tests_regressed": [],
        }

    total = len(entries)
    by_status: Dict[str, int] = {}
    for e in entries:
        s = str(e.get("status") or "")
        by_status[s] = by_status.get(s, 0) + 1

    passed = by_status.get("passed", 0)
    regression = by_status.get("regression", 0)
    tools_changed = by_status.get("tools_changed", 0)
    output_changed = by_status.get("output_changed", 0)
    pass_rate = (passed / total) if total > 0 else None

    score_diffs = [float(e.get("score_diff", 0.0) or 0.0) for e in entries]
    avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0

    models = sorted({
        str(e.get("model_id") or "") for e in entries if e.get("model_id")
    })

    # Track per-test moves (improved = score up, regressed = score down)
    per_test: Dict[str, float] = {}
    for e in entries:
        name = str(e.get("test") or "")
        if not name:
            continue
        per_test[name] = per_test.get(name, 0.0) + float(e.get("score_diff", 0.0) or 0.0)
    improved = sorted(
        [(n, d) for n, d in per_test.items() if d > 1.0],
        key=lambda x: -x[1],
    )
    regressed = sorted(
        [(n, d) for n, d in per_test.items() if d < -1.0],
        key=lambda x: x[1],
    )

    return {
        "total": total,
        "passed": passed,
        "regression": regression,
        "tools_changed": tools_changed,
        "output_changed": output_changed,
        "pass_rate": pass_rate,
        "avg_score_diff": avg_score_diff,
        "models": models,
        "tests_improved": [n for n, _ in improved[:3]],
        "tests_regressed": [n for n, _ in regressed[:3]],
    }


def _sparkline(values: List[float], width: int = 8) -> str:
    """Tiny unicode sparkline for drift visualization.

    Maps values into 8 bar-glyphs. Returns the most recent `width`
    samples. For fewer samples it right-aligns with blanks.
    """
    glyphs = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo
    if span == 0:
        return glyphs[4] * len(recent)
    out = ""
    for v in recent:
        idx = int((v - lo) / span * (len(glyphs) - 1))
        out += glyphs[idx]
    return out


def _detect_drifting_tests(
    entries_in_window: List[Dict[str, Any]],
    all_entries: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """Return per-test (name, sparkline) for tests in the window.

    Uses the full history (not just the window) so a sparkline has
    enough samples to be meaningful even when the window is narrow.
    Sorted by "most recently worse" so the drifting ones bubble up.
    """
    window_names = {str(e.get("test") or "") for e in entries_in_window}
    window_names.discard("")

    per_test_series: Dict[str, List[float]] = {}
    for e in all_entries:
        name = str(e.get("test") or "")
        if name not in window_names:
            continue
        sim = e.get("output_similarity")
        if sim is None:
            continue
        per_test_series.setdefault(name, []).append(float(sim))

    # Drop tests with <3 samples — nothing meaningful to show
    out: List[Tuple[str, float, str]] = []
    for name, series in per_test_series.items():
        if len(series) < 3:
            continue
        spark = _sparkline(series)
        # Rank by "recency-weighted decline": prefer tests whose recent
        # samples are below their max.
        decline = max(series) - series[-1]
        out.append((name, decline, spark))

    out.sort(key=lambda x: -x[1])
    return [(name, spark) for name, _d, spark in out[:5]]


# ───────────────────────── rendering ─────────────────────────


def _render_brief(
    label: str,
    window: Dict[str, Any],
    drift_rows: List[Tuple[str, str]],
    stale_quarantine: List[Dict[str, Any]],
    inactive_days: Optional[int],
) -> None:
    """Render the since-brief to the console.

    Kept tight on purpose. The single most important design rule here:
    one hero number, one concern, one action. Anything more and users
    stop reading it in the morning.
    """
    from rich.panel import Panel

    console.print()

    # ── Headline ──
    if window["total"] == 0:
        console.print(
            Panel(
                "[yellow]No runs in this window.[/yellow]\n\n"
                "Run [bold]evalview check[/bold] to record a baseline —\n"
                "the since-brief needs at least one prior run to compare against.",
                title=f"Since {label}",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        console.print()
        return

    pass_rate = window["pass_rate"] or 0.0
    hero = f"[bold green]{int(round(pass_rate * 100))}%[/bold green]"
    if pass_rate >= 0.95:
        vibe = "green"
    elif pass_rate >= 0.80:
        vibe = "yellow"
    else:
        vibe = "red"

    header_lines = [
        f"[bold]Since {label}[/bold]",
        "",
        f"  {hero}  pass rate across [bold]{window['total']}[/bold] runs",
    ]
    if window["regression"]:
        header_lines.append(
            f"  [red]❌ {window['regression']} regression(s)[/red]"
        )
    if window["tools_changed"] or window["output_changed"]:
        soft = window["tools_changed"] + window["output_changed"]
        header_lines.append(f"  [yellow]⚠  {soft} soft change(s)[/yellow]")
    if window["tests_improved"]:
        header_lines.append(
            f"  [green]✨ improved: {', '.join(window['tests_improved'])}[/green]"
        )

    console.print(
        Panel(
            "\n".join(header_lines),
            border_style=vibe,
            padding=(1, 2),
        )
    )

    # ── Drift sparklines ──
    if drift_rows:
        console.print()
        console.print("[bold]Drift sparklines[/bold] [dim](most-declining first)[/dim]")
        for name, spark in drift_rows:
            console.print(f"  [cyan]{spark}[/cyan]  {name}")

    # ── Stale quarantine alert ──
    if stale_quarantine:
        console.print()
        console.print(
            "[yellow]⏰ Stale quarantine:[/yellow] "
            f"{len(stale_quarantine)} entr{'y' if len(stale_quarantine) == 1 else 'ies'} "
            "overdue for review"
        )
        for entry in stale_quarantine[:3]:
            owner = entry.get("owner") or "<unknown>"
            age = entry.get("age_days")
            age_str = f"{age}d" if age is not None else "?"
            console.print(f"  [dim]• {entry['test_name']} — {owner} — {age_str}[/dim]")

    # ── Inactivity nudge ──
    if inactive_days is not None and inactive_days >= 2:
        console.print()
        console.print(
            f"[yellow]⏳ You haven't run `evalview check` in {inactive_days} days.[/yellow]"
        )

    # ── Single next action ──
    console.print()
    if window["regression"]:
        rtests = window.get("tests_regressed") or []
        target = rtests[0] if rtests else None
        if target:
            console.print("[bold]One thing to look at first:[/bold]")
            console.print(f"  [cyan]→ evalview replay {target} --trace[/cyan]")
        else:
            console.print(
                "[bold]One thing to look at first:[/bold] [cyan]evalview check --fail-on REGRESSION[/cyan]"
            )
    elif drift_rows:
        first_name = drift_rows[0][0]
        console.print("[bold]One thing to look at first:[/bold]")
        console.print(f"  [cyan]→ evalview replay {first_name}[/cyan]")
    elif stale_quarantine:
        console.print(
            "[bold]One thing to look at first:[/bold] "
            "[cyan]evalview quarantine list --stale-only[/cyan]"
        )
    else:
        console.print("[green]Nothing to look at. Your agent is stable.[/green]")

    console.print()


# ───────────────────────── command ─────────────────────────


@click.command("since")
@click.option("--since", "since", default=None,
              help='Time window: "yesterday" | "Nd" | ISO date | git SHA (default: last run).')
@click.option("--json", "json_output", is_flag=True,
              help="Emit machine-readable JSON instead of a panel.")
@track_command("since")
def since_cmd(since: Optional[str], json_output: bool) -> None:
    """Show what's changed since your last run (or a custom window).

    The daily habit anchor: run it first thing in the morning, run it
    before a big merge, run it when you come back from a weekend. Output
    is designed to fit on one screen in under 2 seconds with one hero
    number, one concern, and one action.
    """
    entries = _load_history(_HISTORY_PATH)

    cutoff_dt, cutoff_sha, label = _parse_since(since, entries)
    window_entries = _entries_since(entries, cutoff_dt, cutoff_sha)
    window = _summarize(window_entries)
    drift_rows = _detect_drifting_tests(window_entries, entries)

    # Stale quarantine — load lazily so `since` doesn't crash on a bad YAML.
    stale_quarantine: List[Dict[str, Any]] = []
    try:
        from evalview.core.quarantine import QuarantineStore
        store = QuarantineStore()
        for e in store.list_stale():
            stale_quarantine.append(e.to_dict())
    except Exception:
        pass

    # Inactivity — only surface if the last check was a while ago
    inactive_days: Optional[int] = None
    try:
        from evalview.core.project_state import ProjectStateStore
        inactive_days = ProjectStateStore().days_since_last_check()
    except Exception:
        pass

    if json_output:
        click.echo(
            json.dumps(
                {
                    "label": label,
                    "window": window,
                    "drift": [
                        {"test": name, "sparkline": spark}
                        for name, spark in drift_rows
                    ],
                    "stale_quarantine": stale_quarantine,
                    "inactive_days": inactive_days,
                },
                default=str,
                indent=2,
            )
        )
        return

    _render_brief(
        label=label,
        window=window,
        drift_rows=drift_rows,
        stale_quarantine=stale_quarantine,
        inactive_days=inactive_days,
    )
