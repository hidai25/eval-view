"""Snapshot command — run tests and save passing results as baseline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import click
import yaml  # type: ignore[import-untyped]

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
    quiet: bool = False,
) -> Dict[str, Path]:
    """Save passing test results as golden baselines.

    Returns:
        Mapping of test_case name to the saved golden file path. Only
        tests that were actually written appear in the result — tests
        that raised during save are omitted so callers can report
        accurate per-test status.
    """
    from evalview.core.golden import GoldenStore

    store = GoldenStore()

    # Filter to passing results
    passing = [r for r in results if r.passed]

    if not passing:
        if not quiet:
            console.print("\n[yellow]No passing tests to snapshot.[/yellow]")
            timed_out = [r for r in results if not r.passed and "timeout" in str(getattr(r, "actual_output", "") or "").lower()]
            low_score = [r for r in results if not r.passed and r not in timed_out]
            if timed_out:
                console.print(f"[dim]  {len(timed_out)} test(s) timed out → try: evalview snapshot --timeout 120[/dim]")
            if low_score:
                console.print(f"[dim]  {len(low_score)} test(s) scored below threshold → run evalview run for detailed failure reasons[/dim]")
            if not timed_out and not low_score:
                console.print("[dim]  Run evalview run to see detailed failure reasons, then fix and retry.[/dim]")
            console.print()
        return {}

    # Save passing results as golden
    if not quiet:
        console.print()
    saved: Dict[str, Path] = {}
    for result in passing:
        try:
            path = store.save_golden(result, notes=notes, variant_name=variant)
            variant_label = f" (variant: {variant})" if variant else ""
            if not quiet:
                console.print(f"[green]✓ Snapshotted:[/green] {result.test_case}{variant_label}")
            # save_golden returns a Path on success; fall back to the
            # deterministic path helper if an older implementation returns None.
            saved[result.test_case] = path if path is not None else store._get_golden_path(result.test_case, variant)
        except Exception as e:
            if not quiet:
                console.print(f"[red]❌ Failed to save {result.test_case}: {e}[/red]")

    # Silent cloud push — never blocks or fails the snapshot
    if saved:
        _cloud_push(list(saved.keys()))

    return saved


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


def _generated_snapshot_paths(test_path: str, test_cases: List) -> list[str]:
    """Suggest the narrowest useful path(s) for approving generated drafts."""
    roots = {
        str(Path(getattr(test_case, "source_file")).parent)
        for test_case in test_cases
        if getattr(test_case, "source_file", None)
    }
    if not roots:
        return [test_path]

    resolved: list[str] = []
    for root in sorted(roots):
        try:
            resolved.append(str(Path(root).relative_to(Path.cwd())))
        except ValueError:
            resolved.append(root)
    return resolved


def _resolve_default_test_path(test_path: str) -> str:
    """Use the active onboarding/generation folder when the user omitted a path."""
    if test_path != "tests":
        return test_path
    from evalview.core.project_state import ProjectStateStore

    active = ProjectStateStore().get_active_test_path()
    if active and Path(active).exists():
        return active
    return test_path


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


@click.group(invoke_without_command=True)
@click.option("--path", "test_path", default="tests", type=click.Path(exists=True), help="Directory containing test cases (default: tests/).")
@click.option("--notes", "-n", help="Notes about this snapshot")
@click.option("--test", "-t", help="Snapshot only this specific test (by name)")
@click.option("--variant", help="Save as a named variant for non-deterministic agents (max 5 per test)")
@click.option("--approve-generated", is_flag=True, help="Approve generated draft tests before snapshotting them.")
@click.option("--reset", is_flag=True, help="Delete all existing baselines before capturing new ones.")
@click.option("--judge", "judge_model", default=None, help="Judge model for scoring (e.g. gpt-5.4-mini, sonnet, deepseek-chat).")
@click.option("--no-judge", "no_judge", is_flag=True, default=False, help="Skip LLM-as-judge evaluation. Uses deterministic scoring only (scores capped at 75). No API key required.")
@click.option("--timeout", default=30.0, type=float, help="Timeout in seconds per test (default: 30).")
@click.option("--preview", is_flag=True, help="Show what would change without saving. Dry-run mode for snapshot.")
@click.option("--json", "json_output", is_flag=True, help="Emit a JSON payload on stdout for CI. Suppresses Rich output, auto-approves generated drafts, and skips the dashboard prompt.")
@track_command("snapshot")
@click.pass_context
def snapshot(ctx: click.Context, test_path: str, notes: str, test: str, variant: str, approve_generated: bool, reset: bool, judge_model: Optional[str], no_judge: bool, timeout: float, preview: bool, json_output: bool):
    """Run tests and snapshot passing results as baseline.

    This is the simple workflow: snapshot → check → fix → snapshot.

    \b
    Subcommands:
      snapshot list              List all saved baselines
      snapshot show <name>       View baseline details
      snapshot delete <name>     Remove a baseline

    \b
    Examples:
        evalview snapshot                         # Snapshot all passing tests
        evalview snapshot --test "my-test"        # Snapshot one test only
        evalview snapshot --notes "v2.0"          # Add notes to snapshot
        evalview snapshot --variant variant1      # Save as alternate acceptable behavior
        evalview snapshot --reset                 # Clear old baselines and start fresh
    """
    if ctx.invoked_subcommand is not None:
        return
    from evalview.core.loader import TestCaseLoader
    from evalview.core.project_state import ProjectStateStore
    from evalview.core.celebrations import Celebrations
    from evalview.core.messages import get_random_snapshot_message
    from evalview.skills.ui_utils import print_evalview_banner

    # --preview and --json collide: preview emits human-readable diff output,
    # --json promises a parseable payload. Fail fast rather than silently
    # drop one of them.
    if json_output and preview:
        print(json.dumps(
            {"error": "--preview cannot be combined with --json"}, indent=2
        ))
        ctx.exit(2)

    if not json_output:
        print_evalview_banner(console, subtitle="[dim]Catch agent regressions before you ship[/dim]")

    # Initialize stores
    from evalview.core.golden import GoldenStore

    state_store = ProjectStateStore()
    test_path = _resolve_default_test_path(test_path)

    # Reset: delete all existing baselines before capturing new ones
    if reset:
        golden_store = GoldenStore()
        existing = golden_store.list_golden()
        if existing:
            golden_dir = golden_store.golden_dir
            if golden_dir.exists():
                import shutil
                shutil.rmtree(golden_dir)
                golden_dir.mkdir(parents=True, exist_ok=True)
            if not json_output:
                console.print(f"[yellow]Cleared {len(existing)} existing baseline(s).[/yellow]\n")
        else:
            if not json_output:
                console.print("[dim]No existing baselines to clear.[/dim]\n")

    # Check if this is the first snapshot ever
    is_first = state_store.is_first_snapshot()

    if not json_output:
        console.print(f"\n[cyan]▶ {get_random_snapshot_message()}[/cyan]\n")

    # Load test cases
    loader = TestCaseLoader()
    try:
        test_cases = loader.load_from_directory(Path(test_path))
    except Exception as e:
        if not json_output:
            console.print(f"[red]❌ Failed to load test cases: {e}[/red]\n")
            Celebrations.no_tests_found()
        else:
            print(json.dumps({"error": str(e)}, indent=2))
        return

    if not test_cases:
        if json_output:
            print(json.dumps({"error": "no tests found"}, indent=2))
        else:
            Celebrations.no_tests_found()
        return

    # Filter to specific test if requested
    if test:
        test_cases = [tc for tc in test_cases if tc.name == test]
        if not test_cases:
            if json_output:
                print(json.dumps({"error": f"no test found with name: {test}"}, indent=2))
            else:
                console.print(f"[red]❌ No test found with name: {test}[/red]\n")
            return

    draft_generated = [tc for tc in test_cases if _is_generated_draft(tc)]
    if draft_generated and not approve_generated:
        if json_output:
            approve_generated = True
        else:
            console.print(
                f"[yellow]{len(draft_generated)} generated draft test(s) need approval:[/yellow]"
            )
            for test_case in draft_generated[:8]:
                source = Path(getattr(test_case, "source_file", test_case.name)).name
                query = getattr(getattr(test_case, "input", None), "query", "") or ""
                query_preview = query[:60] + ("..." if len(query) > 60 else "")
                console.print(f"  • {test_case.name} [dim]({source}: {query_preview})[/dim]")
            if len(draft_generated) > 8:
                console.print(f"  [dim]... and {len(draft_generated) - 8} more[/dim]")
            console.print()

            if click.confirm("Approve these drafts and snapshot?", default=False):
                approve_generated = True
            else:
                console.print("[dim]Skipped. You can review the YAML files and re-run when ready.[/dim]\n")
                return
    if draft_generated and approve_generated:
        _approve_generated_tests(draft_generated)
        for test_case in draft_generated:
            if test_case.meta is None:
                test_case.meta = {}
            test_case.meta["review_status"] = "approved"
        if not json_output:
            console.print(f"[green]✓ Approved {len(draft_generated)} generated test(s)[/green]\n")
            console.print("[dim]Approval marks the YAML as reviewed. The tests still need to pass before a baseline is saved.[/dim]\n")

    # Load config
    config = _load_config_if_exists()

    # Apply judge config: --judge flag > env vars > config.yaml
    from evalview.commands.shared import apply_judge_model
    apply_judge_model(judge_model)
    from evalview.core.config import apply_judge_config
    apply_judge_config(config)

    endpoints, adapters = _summarize_mixed_targets(test_cases, config)
    target_groups = _group_tests_by_target(test_cases, config)

    # Execute tests. In JSON mode we skip the live spinner — it writes Rich
    # frames to the same console stream as our JSON payload and would make
    # stdout unparseable.
    if json_output:
        results = _execute_snapshot_tests(
            test_cases, config, timeout=timeout, skip_llm_judge=no_judge, json_output=True
        )
    else:
        from evalview.commands.shared import run_with_spinner
        results = run_with_spinner(
            lambda: _execute_snapshot_tests(test_cases, config, timeout=timeout, skip_llm_judge=no_judge),
            "Snapshotting",
            len(test_cases),
        )
    failed_count = len(test_cases) - len(results)

    # Preview mode: show what would change without saving
    if preview:
        from evalview.core.golden import GoldenStore as _PreviewStore

        preview_store = _PreviewStore()
        console.print("\n[bold]Snapshot Preview[/bold] [dim](no changes saved)[/dim]\n")

        for result in results:
            if not result.passed:
                console.print(f"  [dim]-- {result.test_case}: would skip (not passing)[/dim]")
                continue

            golden = preview_store.load_golden(result.test_case)
            if golden:
                baseline_tools_str = " → ".join(golden.tool_sequence) if golden.tool_sequence else "(none)"
                current_tools_str = " → ".join(
                    str(getattr(s, "tool_name", None) or getattr(s, "step_name", "?"))
                    for s in (result.trace.steps or [])
                ) or "(none)"
                score_change = result.score - golden.metadata.score
                sign = "+" if score_change > 0 else ""
                score_color = "green" if score_change >= 0 else "red"

                console.print(f"  [cyan]{result.test_case}[/cyan]")
                console.print(f"    Baseline: [{baseline_tools_str}]")
                console.print(f"    New:      [{current_tools_str}]")
                console.print(
                    f"    Score:    {golden.metadata.score:.0f} → {result.score:.0f} "
                    f"[{score_color}]({sign}{score_change:.0f})[/{score_color}]"
                )
                console.print()
            else:
                console.print(
                    f"  [green]+[/green] {result.test_case}: "
                    f"[green]new baseline[/green] (score: {result.score:.0f})"
                )
                console.print()

        console.print("[dim]No baselines were modified. Remove --preview to save.[/dim]\n")
        return

    # Save passing results as golden
    saved_paths = _save_snapshot_results(results, notes, variant=variant, quiet=json_output)
    saved_count = len(saved_paths)

    # JSON output mode
    if json_output:
        snapshot_data = {
            "snapshot": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_path": test_path,
                "variant": variant,
                "notes": notes,
                "total_tests": len(test_cases),
                "passing": len([r for r in results if r.passed]),
                "failed": failed_count,
                "saved": saved_count,
            },
            "tests": [
                {
                    "name": result.test_case,
                    "passed": result.passed,
                    "score": result.score,
                    "saved": result.test_case in saved_paths,
                    "golden_file": str(saved_paths[result.test_case])
                    if result.test_case in saved_paths
                    else None,
                }
                for result in results
            ],
        }
        print(json.dumps(snapshot_data, indent=2))
        state_store.update_snapshot(test_count=saved_count)
        return

    if saved_count == 0:
        if draft_generated and approve_generated:
            console.print(
                "[dim]These approved generated drafts still failed. Review the YAML expectations or regenerate with broader coverage before snapshotting again.[/dim]\n"
            )
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

    # Offer to open the dashboard so users can inspect what was baselined
    if results and not bool(__import__("os").environ.get("CI")):
        if click.confirm("Open dashboard to review baseline details?", default=True):
            try:
                from evalview.visualization import generate_visual_report

                path = generate_visual_report(
                    results=results,
                    auto_open=True,
                    title="EvalView Snapshot Report",
                )
                console.print(f"[dim]Report: {path}[/dim]\n")
            except Exception as e:
                console.print(f"[dim]Could not generate report: {e}[/dim]\n")


# ── Subcommands: list, show, delete ──────────────────────────────────────────


@snapshot.command("list")
@track_command("snapshot_list")
def snapshot_list():
    """List all saved baselines."""
    from evalview.core.golden import GoldenStore

    store = GoldenStore()
    goldens = store.list_golden_with_variants()

    if not goldens:
        console.print("\n[yellow]No baselines found.[/yellow]")
        console.print("[dim]Run: evalview snapshot[/dim]\n")
        return

    console.print("\n[bold]Saved Baselines[/bold]\n")

    single_count = 0
    multi_count = 0
    for item in sorted(goldens, key=lambda x: x["metadata"].test_name):
        g = item["metadata"]
        variant_count = item["variant_count"]

        # Check if multi-turn by loading the golden trace
        golden = store.load_golden(g.test_name)
        turns = golden.per_turn_tool_sequences if golden else None
        n_turns = len(turns) if turns else 0

        tags: List[str] = []
        if variant_count > 1:
            tags.append(f"{variant_count} variants")
        if n_turns >= 2:
            tags.append(f"{n_turns} turns")
            multi_count += 1
        else:
            single_count += 1

        n_tools = len(golden.tool_sequence) if golden else 0
        tag_str = f"  [dim]({', '.join(tags)})[/dim]" if tags else ""

        console.print(f"  [cyan]{g.test_name}[/cyan]{tag_str}")
        detail_parts = [f"{g.score:.0f}/100", g.blessed_at.strftime('%Y-%m-%d %H:%M')]
        if n_tools:
            tool_str = " → ".join(golden.tool_sequence[:4]) if golden else ""
            if n_tools > 4:
                tool_str += f" (+{n_tools - 4})"
            detail_parts.append(tool_str)
        console.print(f"    [dim]{' | '.join(detail_parts)}[/dim]")

    console.print()
    parts = []
    if single_count:
        parts.append(f"{single_count} single-turn")
    if multi_count:
        parts.append(f"{multi_count} multi-turn")
    console.print(f"[dim]{len(goldens)} baseline(s): {', '.join(parts)}[/dim]\n")

    console.print(f"\n[dim]{len(goldens)} baseline(s)[/dim]\n")


@snapshot.command("show")
@click.argument("test_name")
@track_command("snapshot_show")
def snapshot_show(test_name: str):
    """View details of a saved baseline."""
    from evalview.core.golden import GoldenStore
    from rich.panel import Panel

    store = GoldenStore()
    golden = store.load_golden(test_name)

    if not golden:
        console.print(f"\n[yellow]No baseline found for: {test_name}[/yellow]\n")
        return

    console.print(f"\n[bold]{test_name}[/bold]\n")
    console.print(f"  Score:    {golden.metadata.score:.1f}")
    console.print(f"  Captured: {golden.metadata.blessed_at.strftime('%Y-%m-%d %H:%M')}")
    if golden.metadata.model_id:
        provider = golden.metadata.model_provider or ""
        model = f"{provider}/{golden.metadata.model_id}" if provider else golden.metadata.model_id
        console.print(f"  Model:   {model}")
    if golden.metadata.notes:
        console.print(f"  Notes:   {golden.metadata.notes}")
    console.print()

    if golden.tool_sequence:
        console.print("[bold]Tool Sequence:[/bold]")
        for i, tool in enumerate(golden.tool_sequence, 1):
            console.print(f"  {i}. {tool}")
        console.print()

    preview = golden.trace.final_output[:500]
    if len(golden.trace.final_output) > 500:
        preview += "..."
    console.print("[bold]Output:[/bold]")
    console.print(Panel(preview, border_style="dim"))
    console.print()


@snapshot.command("delete")
@click.argument("test_name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@track_command("snapshot_delete")
def snapshot_delete(test_name: str, force: bool):
    """Remove a saved baseline."""
    from evalview.core.golden import GoldenStore

    store = GoldenStore()

    if not store.has_golden(test_name):
        console.print(f"\n[yellow]No baseline found for: {test_name}[/yellow]\n")
        return

    if not force:
        if not click.confirm(f"Delete baseline for '{test_name}'?", default=False):
            console.print("[dim]Cancelled[/dim]")
            return

    store.delete_golden(test_name)
    console.print(f"\n[green]Deleted: {test_name}[/green]\n")
