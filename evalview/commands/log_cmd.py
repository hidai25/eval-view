"""`evalview log` — `git log` for eval runs.

Reads .evalview/history.jsonl and shows a reverse-chronological summary
grouped by run (tests checked at the same timestamp cluster into one run).

Each line answers: when, which agent version, how many tests, what verdict,
what changed vs the previous run.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


_HISTORY_PATH = Path(".evalview") / "history.jsonl"


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


def _group_runs(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group per-test entries into runs.

    A "run" is a cluster of entries sharing the same version fingerprint
    *and* a tight timestamp bucket. The grouping key is:

        (git_sha, prompt_hash, ts_bucket)

    with `ts_bucket` at minute precision when the entry has a git_sha
    (commits don't change mid-run, so a minute is safe) and at second
    precision when both git_sha and prompt_hash are missing (no version
    anchor — fall back to tighter time windows so back-to-back runs in a
    non-git directory don't collapse into a single row).
    """
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        ts = str(e.get("ts") or "")
        sha = str(e.get("git_sha") or "")
        prompt = str(e.get("prompt_hash") or "")
        # Minute precision when we have a version anchor; second otherwise.
        # The ISO format is "YYYY-MM-DDTHH:MM:SS…" — slice at 16 for minute,
        # 19 for second.
        if sha or prompt:
            ts_bucket = ts[:16]
        else:
            ts_bucket = ts[:19]
        key = (sha, prompt, ts_bucket)
        buckets[key].append(e)

    runs: List[Dict[str, Any]] = []
    for (sha, _prompt, _ts_bucket), items in buckets.items():
        latest_ts = max(str(i.get("ts") or "") for i in items)
        statuses = [str(i.get("status") or "") for i in items]
        n_pass = sum(1 for s in statuses if s == "passed")
        n_regression = sum(1 for s in statuses if s == "regression")
        n_tools_changed = sum(1 for s in statuses if s == "tools_changed")
        n_output_changed = sum(1 for s in statuses if s == "output_changed")

        score_diffs = [
            float(i.get("score_diff", 0.0) or 0.0) for i in items
        ]
        avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0

        models = {str(i.get("model_id") or "") for i in items}
        models.discard("")
        users = {str(i.get("user") or "") for i in items}
        users.discard("")
        prompt_hashes = {str(i.get("prompt_hash") or "") for i in items}
        prompt_hashes.discard("")

        runs.append({
            "ts": latest_ts,
            "git_sha": sha or None,
            "prompt_hash": next(iter(prompt_hashes), None),
            "user": next(iter(users), None),
            "model": next(iter(models), None),
            "test_count": len(items),
            "pass": n_pass,
            "regression": n_regression,
            "tools_changed": n_tools_changed,
            "output_changed": n_output_changed,
            "avg_score_diff": avg_score_diff,
        })

    runs.sort(key=lambda r: str(r["ts"]), reverse=True)
    return runs


def _format_ts(raw: str) -> str:
    if not raw:
        return "unknown"
    try:
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return raw[:16]


def _verdict_for_run(run: Dict[str, Any]) -> Tuple[str, str]:
    if run["regression"] > 0:
        return ("BLOCK", "red")
    if run["tools_changed"] > 0 or run["output_changed"] > 0:
        return ("INVESTIGATE", "yellow")
    return ("CLEAN", "green")


@click.command("log")
@click.option("-n", "--limit", default=20, show_default=True,
              help="Maximum number of runs to show.")
@click.option("--json", "json_output", is_flag=True,
              help="Emit machine-readable JSON instead of a table.")
@track_command("log")
def log_cmd(limit: int, json_output: bool) -> None:
    """Show recent check runs — `git log` for agent evaluations.

    Each row is one run, with:
        • timestamp
        • short git SHA (if available)
        • test count + pass/fail breakdown
        • verdict (CLEAN / INVESTIGATE / BLOCK)
        • average score delta vs the run's baseline

    Source: .evalview/history.jsonl, populated by every `evalview check`.
    """
    entries = _load_history(_HISTORY_PATH)
    if not entries:
        if json_output:
            click.echo(json.dumps({"runs": []}))
            return
        console.print(
            "[yellow]No history yet.[/yellow] "
            "Run [bold]evalview check[/bold] to start recording."
        )
        return

    runs = _group_runs(entries)[:limit]

    if json_output:
        click.echo(json.dumps({"runs": runs}, default=str))
        return

    from rich.table import Table

    table = Table(
        title=f"Recent eval runs (newest first, showing {len(runs)})",
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("When", style="dim", no_wrap=True)
    table.add_column("Rev", style="cyan", no_wrap=True)
    table.add_column("Tests", justify="right")
    table.add_column("Pass", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Δ score", justify="right")
    table.add_column("Verdict", no_wrap=True)
    table.add_column("Model", style="dim")

    for run in runs:
        verdict_text, color = _verdict_for_run(run)
        failures = (
            run["regression"] + run["tools_changed"] + run["output_changed"]
        )
        score_delta = run["avg_score_diff"]
        if score_delta > 0.05:
            score_str = f"[green]+{score_delta:.2f}[/green]"
        elif score_delta < -0.05:
            score_str = f"[red]{score_delta:.2f}[/red]"
        else:
            score_str = f"{score_delta:+.2f}"

        table.add_row(
            _format_ts(str(run["ts"])),
            run["git_sha"] or "—",
            str(run["test_count"]),
            str(run["pass"]),
            str(failures) if failures else "[dim]0[/dim]",
            score_str,
            f"[{color}]{verdict_text}[/{color}]",
            (run["model"] or "—")[:24],
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]Tip: `evalview check` appends each run; "
        "entries older than ~10k lines are pruned automatically.[/dim]"
    )
