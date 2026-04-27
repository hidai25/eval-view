"""`evalview diff` - compare two result-file runs."""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


def _load_result_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON in {path}: {exc}[/red]")
        sys.exit(1)
    except OSError as exc:
        console.print(f"[red]Could not read {path}: {exc}[/red]")
        sys.exit(1)

    if not isinstance(data, list):
        console.print(
            f"[red]Invalid result file {path}: expected a list of test-case results[/red]"
        )
        sys.exit(1)

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            console.print(f"[red]Invalid result file {path}: item {index} is not an object[/red]")
            sys.exit(1)
        if not isinstance(item.get("test_case"), str) or not item.get("test_case"):
            console.print(
                f"[red]Invalid result file {path}: item {index} is missing test_case[/red]"
            )
            sys.exit(1)

    return data


def _by_test_case(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(case["test_case"]): case for case in results}


def _tool_sequence(case: Dict[str, Any]) -> List[str]:
    evaluations = case.get("evaluations")
    if not isinstance(evaluations, dict):
        return []
    sequence = evaluations.get("sequence_correctness")
    if not isinstance(sequence, dict):
        return []
    actual = sequence.get("actual_sequence")
    if not isinstance(actual, list):
        return []
    return [str(item) for item in actual]


def _nested_number(
    case: Dict[str, Any],
    evaluation: str,
    fields: Tuple[str, ...],
) -> Optional[float]:
    evaluations = case.get("evaluations")
    if not isinstance(evaluations, dict):
        return None
    section = evaluations.get(evaluation)
    if not isinstance(section, dict):
        return None
    for field in fields:
        value = section.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _cost(case: Dict[str, Any]) -> Optional[float]:
    return _nested_number(case, "cost", ("total_cost",))


def _latency_ms(case: Dict[str, Any]) -> Optional[float]:
    total_ms = _nested_number(case, "latency", ("total_ms", "total_latency_ms", "total_latency"))
    if total_ms is not None:
        return total_ms

    seconds = _nested_number(case, "latency", ("total_seconds", "seconds"))
    if seconds is not None:
        return seconds * 1000

    return None


def _output_similarity(before: Dict[str, Any], after: Dict[str, Any]) -> float:
    before_output = before.get("actual_output") or ""
    after_output = after.get("actual_output") or ""
    return difflib.SequenceMatcher(None, str(before_output), str(after_output)).ratio() * 100


def _compare_cases(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    before_names = set(before)
    after_names = set(after)
    compared: List[Dict[str, Any]] = []

    for name in sorted(before_names & after_names):
        before_case = before[name]
        after_case = after[name]
        before_tools = _tool_sequence(before_case)
        after_tools = _tool_sequence(after_case)

        before_cost = _cost(before_case)
        after_cost = _cost(after_case)
        before_latency = _latency_ms(before_case)
        after_latency = _latency_ms(after_case)

        compared.append(
            {
                "test_case": name,
                "tool_sequence": {
                    "before": before_tools,
                    "after": after_tools,
                    "changed": before_tools != after_tools,
                },
                "output_similarity": _output_similarity(before_case, after_case),
                "cost_before": before_cost,
                "cost_after": after_cost,
                "cost_delta": (
                    after_cost - before_cost
                    if before_cost is not None and after_cost is not None
                    else None
                ),
                "latency_before_ms": before_latency,
                "latency_after_ms": after_latency,
                "latency_delta_ms": (
                    after_latency - before_latency
                    if before_latency is not None and after_latency is not None
                    else None
                ),
            }
        )

    return {
        "added": sorted(after_names - before_names),
        "removed": sorted(before_names - after_names),
        "compared": compared,
    }


def _format_sequence(sequence: List[str]) -> str:
    return "[" + ", ".join(sequence) + "]"


def _format_tool_sequence(row: Dict[str, Any]) -> str:
    sequence = row["tool_sequence"]
    if not sequence["changed"]:
        return "[dim]unchanged[/dim]"
    return f"{_format_sequence(sequence['before'])} -> {_format_sequence(sequence['after'])}"


def _format_similarity(value: float) -> str:
    rendered = f"{value:.1f}%"
    if value == 100 or value >= 95:
        return f"[dim]{rendered}[/dim]"
    if value >= 80:
        return f"[yellow]{rendered}[/yellow]"
    return f"[red]{rendered}[/red]"


def _format_money(value: float) -> str:
    return f"${value:.4f}"


def _format_delta_value(delta: float, formatter: Callable[[float], str]) -> str:
    if delta > 0:
        return f"+{formatter(delta)}"
    if delta < 0:
        return f"-{formatter(abs(delta))}"
    return formatter(0)


def _format_cost(row: Dict[str, Any]) -> str:
    before = row["cost_before"]
    after = row["cost_after"]
    delta = row["cost_delta"]
    if before is None or after is None or delta is None:
        return "[dim]n/a[/dim]"

    rendered = (
        f"{_format_money(before)} → {_format_money(after)} "
        f"({_format_delta_value(delta, _format_money)})"
    )
    if delta < 0:
        return f"[green]{rendered}[/green]"
    if delta > 0:
        return f"[red]{rendered}[/red]"
    return f"[dim]{rendered}[/dim]"


def _format_ms(value: float) -> str:
    return f"{value:.1f}ms"


def _format_latency(row: Dict[str, Any]) -> str:
    before = row["latency_before_ms"]
    after = row["latency_after_ms"]
    delta = row["latency_delta_ms"]
    if before is None or after is None or delta is None:
        return "[dim]n/a[/dim]"

    rendered = (
        f"{_format_ms(before)} → {_format_ms(after)} "
        f"({_format_delta_value(delta, _format_ms)})"
    )
    if delta < 0:
        return f"[green]{rendered}[/green]"
    if delta > 0:
        return f"[red]{rendered}[/red]"
    return f"[dim]{rendered}[/dim]"


def _render_diff(payload: Dict[str, Any]) -> None:
    from rich.table import Table

    table = Table(header_style="bold cyan")
    table.add_column("Test Case")
    table.add_column("Tool Sequence")
    table.add_column("Output Sim", justify="right")
    table.add_column("Cost")
    table.add_column("Latency")

    for row in payload["compared"]:
        table.add_row(
            row["test_case"],
            _format_tool_sequence(row),
            _format_similarity(row["output_similarity"]),
            _format_cost(row),
            _format_latency(row),
        )

    console.print(table)

    if payload["added"]:
        console.print("\n[green]Added[/green]")
        for name in payload["added"]:
            console.print(f"  - {name}")

    if payload["removed"]:
        console.print("\n[red]Removed[/red]")
        for name in payload["removed"]:
            console.print(f"  - {name}")


@click.command(name="diff", help="Pretty-print what changed between two result files")
@click.argument("file1", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.argument("file2", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
@track_command("diff")
def diff_cmd(file1: Path, file2: Path, as_json: bool) -> None:
    before = _by_test_case(_load_result_file(file1))
    after = _by_test_case(_load_result_file(file2))
    payload = _compare_cases(before, after)

    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return

    _render_diff(payload)
