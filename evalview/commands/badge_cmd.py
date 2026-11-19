"""Badge command — generate a shields.io-compatible status badge."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command

# Badge JSON lives here by default
_DEFAULT_BADGE_PATH = ".evalview/badge.json"

# Status → badge config
_STATUS_CONFIG: Dict[str, Dict[str, Any]] = {
    "passed": {"label": "evalview", "message": "passing", "color": "brightgreen"},
    "tools_changed": {"label": "evalview", "message": "tools changed", "color": "yellow"},
    "output_changed": {"label": "evalview", "message": "output changed", "color": "yellow"},
    "regression": {"label": "evalview", "message": "regression", "color": "red"},
    "no_baselines": {"label": "evalview", "message": "no baselines", "color": "lightgrey"},
    "incomplete": {"label": "evalview", "message": "incomplete", "color": "red"},
    "error": {"label": "evalview", "message": "error", "color": "lightgrey"},
}


@click.command("badge")
@click.option(
    "--output", "-o",
    default=_DEFAULT_BADGE_PATH,
    show_default=True,
    help="Path to write the badge JSON file.",
)
@click.option(
    "--check/--no-check",
    "run_check",
    default=True,
    help="Run a check first to get fresh status (default: True).",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Use quick mode for the check (no LLM judge, $0).",
)
@click.option(
    "--test-dir",
    default="tests",
    show_default=True,
    help="Path to test cases directory.",
)
@track_command("badge")
def badge(output: str, run_check: bool, quick: bool, test_dir: str) -> None:
    """Generate a shields.io-compatible badge for your README.

    Writes a JSON endpoint file that shields.io can read. Host it anywhere
    publicly accessible, then add the badge to your README:

    \b
    Usage:
        evalview badge                    # Run check + write badge JSON
        evalview badge --no-check         # Badge from last check (no API calls)
        evalview badge --quick            # Fast check, no LLM judge

    \b
    Then commit the badge file and add to your README (replace USER/REPO):
        ![EvalView](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/USER/REPO/main/.evalview/badge.json&style=flat)

    The badge auto-updates every time `evalview check` runs.
    """
    badge_data = _get_badge_data(run_check, quick, test_dir)

    # Write badge JSON
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(badge_data, indent=2) + "\n")

    console.print(f"[green]Badge written to:[/green] {out_path}")
    console.print()

    # Show preview
    msg = badge_data["message"]
    color = badge_data["color"]
    color_code = {
        "brightgreen": "green",
        "yellow": "yellow",
        "red": "red",
        "lightgrey": "dim",
    }.get(color, "white")
    console.print(f"  [{color_code}]evalview | {msg}[/{color_code}]")
    console.print()

    # Show usage hint
    console.print("[dim]Commit the badge file, then add to your README (replace USER/REPO):[/dim]")
    console.print("  ![EvalView](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/USER/REPO/main/.evalview/badge.json&style=flat)")
    console.print()


def _get_badge_data(run_check: bool, quick: bool, test_dir: str) -> dict:
    """Get badge data, optionally running a fresh check first."""
    if run_check:
        return _badge_from_check(quick, test_dir)
    return _badge_from_last_result()


def _badge_from_check(quick: bool, test_dir: str) -> dict:
    """Run a gate check and return badge data."""
    from evalview.api import gate

    try:
        result = gate(test_dir=test_dir, quick=quick)
    except Exception as e:
        console.print(f"[red]Check failed:[/red] {e}")
        return _STATUS_CONFIG["error"].copy() | {"schemaVersion": 1}

    s = result.summary
    return _badge_for_counts(
        compared=s.total, requested=s.requested_tests, unchanged=s.unchanged,
        regressions=s.regressions, incomplete=not result.passed and "error" in result.raw_json,
    )


def _badge_for_counts(compared: int, requested: int, unchanged: int, regressions: int, incomplete: bool) -> dict:
    if incomplete or compared < requested:
        cfg = _STATUS_CONFIG["incomplete"].copy()
        if requested:
            cfg["message"] = f"incomplete: {compared}/{requested} compared"
        if regressions:
            cfg["message"] += f", {regressions} regression{'s' if regressions != 1 else ''}"
    elif regressions:
        cfg = _STATUS_CONFIG["regression"].copy()
        cfg["message"] = f"{regressions} regression{'s' if regressions != 1 else ''}"
    elif compared:
        cfg = _STATUS_CONFIG["passed" if unchanged == compared else "tools_changed"].copy()
        cfg["message"] = f"{unchanged}/{compared} passing"
    else:
        cfg = _STATUS_CONFIG["no_baselines"].copy()
    return {**cfg, "schemaVersion": 1}


def _badge_from_last_result() -> dict:
    """Read badge data from the most recent check result."""
    results_dir = Path(".evalview/results")
    if not results_dir.exists():
        return _STATUS_CONFIG["no_baselines"].copy() | {"schemaVersion": 1}

    json_files = sorted(
        results_dir.glob("*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not json_files:
        return _STATUS_CONFIG["no_baselines"].copy() | {"schemaVersion": 1}

    try:
        with open(json_files[0]) as f:
            data = json.load(f)

        # Handle both check format (dict with summary) and run format (list)
        if isinstance(data, list):
            total = len(data)
            passed = sum(1 for r in data if r.get("passed", False))
            failed = total - passed
            if failed > 0:
                cfg = _STATUS_CONFIG["regression"].copy()
                cfg["message"] = f"{failed} failed"
            elif total > 0:
                cfg = _STATUS_CONFIG["passed"].copy()
                cfg["message"] = f"{total}/{total} passing"
            else:
                return _STATUS_CONFIG["no_baselines"].copy() | {"schemaVersion": 1}
            cfg["schemaVersion"] = 1
            return cfg

        summary = data.get("summary", {})
        total = summary.get("total_tests", summary.get("total", 0))
        regressions = summary.get("regressions", 0)
        changed = regressions + summary.get("tools_changed", 0) + summary.get("output_changed", 0)
        unchanged = summary.get("unchanged", summary.get("passed", total - changed))

        return _badge_for_counts(
            compared=total,
            requested=summary.get("requested_tests", total + summary.get("execution_failures", 0)),
            unchanged=unchanged,
            regressions=regressions,
            incomplete=data.get("status") == "incomplete" or bool(data.get("error")),
        )
    except (json.JSONDecodeError, KeyError):
        return _STATUS_CONFIG["error"].copy() | {"schemaVersion": 1}


def update_badge_after_check(diffs: list, total_tests: int, incomplete: bool = False) -> None:
    """Auto-update badge JSON after a check. Called from check_cmd if badge file exists.

    Only writes if the badge file already exists (opt-in — user must run
    ``evalview badge`` once to create it).
    """
    from evalview.core.diff import DiffStatus

    regressions = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION)
    passed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED)
    update_badge_counts(len(diffs), total_tests, passed, regressions, incomplete)


def update_badge_counts(compared: int, requested: int, unchanged: int, regressions: int, incomplete: bool) -> None:
    badge_path = Path(_DEFAULT_BADGE_PATH)
    if badge_path.exists():
        cfg = _badge_for_counts(compared, requested, unchanged, regressions, incomplete)
        badge_path.write_text(json.dumps(cfg, indent=2) + "\n")
