"""Snapshot command — run tests and save passing results as baseline."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import click
import yaml

from evalview.commands.shared import (
    console,
    _execute_snapshot_tests,
    _load_config_if_exists,
    _cloud_push,
)
from evalview.telemetry.decorators import track_command

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult


def _save_snapshot_results(
    results: List["EvaluationResult"],
    notes: Optional[str],
    variant: Optional[str] = None,
) -> int:
    """Save passing test results as golden baselines.

    Returns:
        Number of tests successfully saved
    """
    from evalview.core.golden import GoldenStore

    store = GoldenStore()

    # Filter to passing results
    passing = [r for r in results if r.passed]

    if not passing:
        console.print("\n[yellow]No passing tests to snapshot.[/yellow]")
        console.print("[dim]Fix failing tests first, then run evalview snapshot again.[/dim]\n")
        return 0

    # Save passing results as golden
    console.print()
    saved_count = 0
    saved_names = []
    for result in passing:
        try:
            store.save_golden(result, notes=notes, variant_name=variant)
            variant_label = f" (variant: {variant})" if variant else ""
            console.print(f"[green]✓ Snapshotted:[/green] {result.test_case}{variant_label}")
            saved_count += 1
            saved_names.append(result.test_case)
        except Exception as e:
            console.print(f"[red]❌ Failed to save {result.test_case}: {e}[/red]")

    # Silent cloud push — never blocks or fails the snapshot
    if saved_names:
        _cloud_push(saved_names)

    return saved_count


def _is_generated_draft(test_case) -> bool:
    meta = test_case.meta or {}
    return meta.get("generated_by") == "evalview generate" and meta.get("review_status", "draft") != "approved"


def _approve_generated_tests(test_cases: List) -> None:
    """Mark generated draft tests as approved in their YAML source files."""
    approved_at = datetime.now(timezone.utc).isoformat()
    for test_case in test_cases:
        source_file = getattr(test_case, "source_file", None)
        if not source_file:
            continue

        path = Path(source_file)
        if not path.exists():
            continue

        original = path.read_text(encoding="utf-8")
        header_lines = []
        body_lines = original.splitlines()
        while body_lines and body_lines[0].startswith("#"):
            header_lines.append(body_lines.pop(0))
        if body_lines and body_lines[0] == "":
            header_lines.append(body_lines.pop(0))

        data = yaml.safe_load("\n".join(body_lines)) or {}
        meta = dict(data.get("meta") or {})
        meta["review_status"] = "approved"
        meta["approved_at"] = approved_at
        data["meta"] = meta

        serialized = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
        prefix = "\n".join(header_lines)
        if prefix:
            serialized = prefix + "\n" + serialized
        path.write_text(serialized, encoding="utf-8")


def _summarize_mixed_targets(test_cases: List, config) -> tuple[list[str], list[str]]:
    """Return distinct endpoints and adapters represented in the selected tests."""
    config_endpoint = getattr(config, "endpoint", None) if config else None
    config_adapter = getattr(config, "adapter", None) if config else None

    endpoints = sorted(
        {
            endpoint
            for endpoint in ((tc.endpoint or config_endpoint) for tc in test_cases)
            if endpoint
        }
    )
    adapters = sorted(
        {
            adapter
            for adapter in ((tc.adapter or config_adapter) for tc in test_cases)
            if adapter
        }
    )
    return endpoints, adapters


def _group_tests_by_target(test_cases: List, config) -> Dict[tuple[str, str], list[str]]:
    """Group tests by their effective adapter/endpoint target."""
    config_endpoint = getattr(config, "endpoint", None) if config else None
    config_adapter = getattr(config, "adapter", None) if config else None

    groups: Dict[tuple[str, str], list[str]] = {}
    for test_case in test_cases:
        adapter = test_case.adapter or config_adapter or "<unknown-adapter>"
        endpoint = test_case.endpoint or config_endpoint or "<unknown-endpoint>"
        source = getattr(test_case, "source_file", None)
        label = Path(source).name if source else test_case.name
        groups.setdefault((adapter, endpoint), []).append(label)
    return groups


@click.command("snapshot")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--notes", "-n", help="Notes about this snapshot")
@click.option("--test", "-t", help="Snapshot only this specific test (by name)")
@click.option("--variant", help="Save as a named variant for non-deterministic agents (max 5 per test)")
@click.option("--approve-generated", is_flag=True, help="Approve generated draft tests before snapshotting them.")
@track_command("snapshot")
def snapshot(test_path: str, notes: str, test: str, variant: str, approve_generated: bool):
    """Run tests and snapshot passing results as baseline.

    This is the simple workflow: snapshot → check → fix → snapshot.

    TEST_PATH is the directory containing test cases (default: tests/).

    Examples:
        evalview snapshot                         # Snapshot all passing tests
        evalview snapshot --test "my-test"        # Snapshot one test only
        evalview snapshot --notes "v2.0"          # Add notes to snapshot
        evalview snapshot --variant variant1      # Save as alternate acceptable behavior
    """
    from evalview.core.loader import TestCaseLoader
    from evalview.core.project_state import ProjectStateStore
    from evalview.core.celebrations import Celebrations
    from evalview.core.messages import get_random_snapshot_message
    from evalview.skills.ui_utils import print_evalview_banner

    print_evalview_banner(console, subtitle="[dim]Catch agent regressions before you ship[/dim]")

    # Initialize stores
    state_store = ProjectStateStore()

    # Check if this is the first snapshot ever
    is_first = state_store.is_first_snapshot()

    console.print(f"\n[cyan]▶ {get_random_snapshot_message()}[/cyan]\n")

    # Load test cases
    loader = TestCaseLoader()
    try:
        test_cases = loader.load_from_directory(Path(test_path))
    except Exception as e:
        console.print(f"[red]❌ Failed to load test cases: {e}[/red]\n")
        Celebrations.no_tests_found()
        return

    if not test_cases:
        Celebrations.no_tests_found()
        return

    # Filter to specific test if requested
    if test:
        test_cases = [tc for tc in test_cases if tc.name == test]
        if not test_cases:
            console.print(f"[red]❌ No test found with name: {test}[/red]\n")
            return

    draft_generated = [tc for tc in test_cases if _is_generated_draft(tc)]
    if draft_generated and not approve_generated:
        console.print("[yellow]Generated draft tests require approval before snapshotting.[/yellow]")
        for test_case in draft_generated[:8]:
            console.print(f"  • {test_case.name}")
        console.print("\n[dim]Review the generated YAML, then run: evalview snapshot "
                      f"{test_path} --approve-generated[/dim]\n")
        return
    if draft_generated and approve_generated:
        _approve_generated_tests(draft_generated)
        for test_case in draft_generated:
            if test_case.meta is None:
                test_case.meta = {}
            test_case.meta["review_status"] = "approved"
        console.print(f"[green]✓ Approved {len(draft_generated)} generated test(s)[/green]\n")

    # Run tests
    console.print(f"[cyan]Running {len(test_cases)} test(s)...[/cyan]\n")

    # Load config
    config = _load_config_if_exists()

    # Apply judge config from config.yaml (env vars / CLI flags take priority)
    from evalview.core.config import apply_judge_config
    apply_judge_config(config)

    endpoints, adapters = _summarize_mixed_targets(test_cases, config)
    target_groups = _group_tests_by_target(test_cases, config)

    # Execute tests
    results = _execute_snapshot_tests(test_cases, config)
    failed_count = len(test_cases) - len(results)

    # Save passing results as golden
    saved_count = _save_snapshot_results(results, notes, variant=variant)

    if saved_count == 0:
        return

    if failed_count > 0:
        console.print(
            f"\n[yellow]Only {saved_count} of {len(test_cases)} selected test(s) were snapshotted.[/yellow]"
        )
        console.print("[dim]EvalView saves baselines only for passing tests.[/dim]")
        if len(endpoints) > 1 or len(adapters) > 1:
            console.print("[yellow]This test selection mixes multiple endpoints or adapters.[/yellow]")
            if endpoints:
                console.print(f"[dim]Endpoints: {', '.join(endpoints)}[/dim]")
            if adapters:
                console.print(f"[dim]Adapters: {', '.join(adapters)}[/dim]")
            if len(target_groups) > 1:
                console.print("[dim]Tests grouped by target:[/dim]")
                for (adapter, endpoint), files in sorted(target_groups.items()):
                    listed = ", ".join(files[:4])
                    if len(files) > 4:
                        listed += f", +{len(files) - 4} more"
                    console.print(f"[dim]  • {adapter} @ {endpoint}: {listed}[/dim]")
            console.print(
                "[dim]To clean this up:[/dim]"
            )
            console.print(
                "[dim]  1. Run evalview init if .evalview/config.yaml still points at an old agent.[/dim]"
            )
            console.print(
                "[dim]  2. Move or delete tests that target other adapters/endpoints before snapshotting.[/dim]"
            )
            console.print(
                "[dim]  3. Or snapshot a clean subfolder only, for example: evalview snapshot tests/current-agent[/dim]"
            )
        else:
            console.print("[dim]Fix the failing tests, then rerun evalview snapshot for a complete baseline.[/dim]")

    # Update project state
    state_store.update_snapshot(test_count=saved_count)

    # Celebrate!
    if is_first:
        Celebrations.first_snapshot(saved_count)
    else:
        console.print(f"\n[green]Baseline updated: {saved_count} test(s)[/green]")
        console.print("[dim]Run: evalview check[/dim]\n")
