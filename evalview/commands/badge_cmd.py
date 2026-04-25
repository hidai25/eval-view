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

    if not result.diffs:
        return _STATUS_CONFIG["no_baselines"].copy() | {"schemaVersion": 1}

    s = result.summary
    total = s.total

    if s.regressions > 0:
        cfg = _STATUS_CONFIG["regression"].copy()
        cfg["message"] = f"{s.regressions} regression{'s' if s.regressions != 1 else ''}"
    elif s.tools_changed > 0 or s.output_changed > 0:
        changed = s.tools_changed + s.output_changed
        cfg = _STATUS_CONFIG["tools_changed"].copy()
        cfg["message"] = f"{total - changed}/{total} passing"
    else:
        cfg = _STATUS_CONFIG["passed"].copy()
        cfg["message"] = f"{total}/{total} passing"

    cfg["schemaVersion"] = 1
    return cfg


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
        tools_changed = summary.get("tools_changed", 0)
        output_changed = summary.get("output_changed", 0)
        unchanged = summary.get("unchanged", summary.get("passed", 0))

        if regressions > 0:
            cfg = _STATUS_CONFIG["regression"].copy()
            cfg["message"] = f"{regressions} regression{'s' if regressions != 1 else ''}"
        elif tools_changed > 0 or output_changed > 0:
            cfg = _STATUS_CONFIG["tools_changed"].copy()
            cfg["message"] = f"{unchanged}/{total} passing"
        elif total > 0:
            cfg = _STATUS_CONFIG["passed"].copy()
            cfg["message"] = f"{total}/{total} passing"
        else:
            return _STATUS_CONFIG["no_baselines"].copy() | {"schemaVersion": 1}

        cfg["schemaVersion"] = 1
        return cfg
    except (json.JSONDecodeError, KeyError):
        return _STATUS_CONFIG["error"].copy() | {"schemaVersion": 1}


def update_badge_after_check(diffs: list, total_tests: int) -> None:
    """Auto-update badge JSON after a check. Called from check_cmd if badge file exists.

    Only writes if the badge file already exists (opt-in — user must run
    ``evalview badge`` once to create it).
    """
    badge_path = Path(_DEFAULT_BADGE_PATH)
    if not badge_path.exists():
        return

    from evalview.core.diff import DiffStatus

    regressions = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION)
    tools_changed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED)
    output_changed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED)
    passed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED)

    if regressions > 0:
        cfg = _STATUS_CONFIG["regression"].copy()
        cfg["message"] = f"{regressions} regression{'s' if regressions != 1 else ''}"
    elif tools_changed > 0 or output_changed > 0:
        cfg = _STATUS_CONFIG["tools_changed"].copy()
        cfg["message"] = f"{passed}/{total_tests} passing"
    elif total_tests > 0:
        cfg = _STATUS_CONFIG["passed"].copy()
        cfg["message"] = f"{total_tests}/{total_tests} passing"
    else:
        return

    cfg["schemaVersion"] = 1
    badge_path.write_text(json.dumps(cfg, indent=2) + "\n")
