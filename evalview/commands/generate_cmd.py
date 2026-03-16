"""Generate command — draft a regression suite by probing an agent."""
from __future__ import annotations

import threading
import time
from pathlib import Path
import yaml  # type: ignore[import-untyped]

import click

from evalview.commands.shared import _detect_agent_endpoint, _load_config_if_exists, console
from evalview.core.adapter_factory import create_adapter
from evalview.core.project_state import ProjectStateStore
from evalview.telemetry.decorators import track_command
from evalview.test_generation import AgentTestGenerator, load_seed_prompts, run_generation


def _print_generate_failure_guidance(
    *,
    endpoint: str | None,
    agent_url: str | None,
    from_log: str | None,
) -> None:
    """Print actionable guidance when live probing produced no draft tests."""
    if from_log:
        return

    detected_endpoint = _detect_agent_endpoint()
    if detected_endpoint and endpoint and detected_endpoint != endpoint and not agent_url:
        console.print()
        console.print(
            f"[yellow]A different local agent is running at {detected_endpoint}.[/yellow]"
        )
        console.print(
            f"[dim]Your current config still points at {endpoint}. "
            "Run one of these:[/dim]"
        )
        console.print("[dim]  • evalview init[/dim]")
        console.print(f"[dim]  • evalview generate --agent {detected_endpoint}[/dim]")
        return

    if endpoint:
        console.print()
        console.print("[dim]Next steps:[/dim]")
        console.print("[dim]  • Start the agent at the endpoint above, then rerun evalview generate[/dim]")
        console.print("[dim]  • If your config is stale, run evalview init to refresh .evalview/config.yaml[/dim]")


def _print_test_summary_table(tests: list) -> None:
    """Print a compact summary table of all generated tests."""
    from rich.table import Table

    table = Table(title="Generated Tests", show_lines=False, pad_edge=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Query", max_width=50)
    table.add_column("Behavior", style="cyan", width=14)
    table.add_column("Tools", style="green", max_width=40)
    table.add_column("Source", style="dim", width=18)

    for i, test in enumerate(tests, 1):
        query = test.input.query[:48] + ("..." if len(test.input.query) > 48 else "")
        meta = test.meta or {}
        behavior = str(meta.get("behavior_class", "unknown")).replace("_", " ")
        tools = " -> ".join(test.expected.tools[:3]) if test.expected.tools else "-"
        source = str(meta.get("prompt_source", "unknown"))
        table.add_row(str(i), query, behavior, tools, source)

    console.print()
    console.print(table)


def _print_test_yaml_inline(tests: list, generator: "AgentTestGenerator") -> None:
    """Print full YAML for every generated test so users can review before approving."""
    console.print()
    console.print("[bold]Full Test YAML[/bold]")
    for i, test in enumerate(tests, 1):
        meta = test.meta or {}
        behavior = str(meta.get("behavior_class", "unknown")).replace("_", " ")
        tools_label = " -> ".join(test.expected.tools[:4]) if test.expected.tools else "no tools"
        console.print(f"\n[bold cyan]--- Test {i}/{len(tests)}: {behavior} ({tools_label}) ---[/bold cyan]")
        payload = test.model_dump(exclude_none=True)
        yaml_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
        console.print(yaml_text.rstrip())
    console.print()


def _print_generated_test_preview(output_dir: Path, max_files: int = 2) -> None:
    """Print generated YAML inline so users can inspect drafts without context-switching."""
    yaml_files = sorted([path for path in output_dir.glob("*.yaml") if path.is_file()])
    if not yaml_files:
        return

    console.print()
    console.print("[bold]Generated Test Preview[/bold]")
    for path in yaml_files[:max_files]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        meta = data.get("meta") or {}
        behavior = str(meta.get("behavior_class") or "unknown").replace("_", " ")
        prompt_source = str(meta.get("prompt_source") or "unknown").replace("_", " ")
        turns = data.get("turns") or []
        turn_label = f"{len(turns)} turns" if turns else "single turn"
        console.print(f"[dim]{path}[/dim]")
        console.print(f"[dim]Behavior: {behavior} | {turn_label} | source: {prompt_source}[/dim]")
        console.print(path.read_text(encoding="utf-8").rstrip())
        console.print()
    if len(yaml_files) > max_files:
        console.print(f"[dim]+ {len(yaml_files) - max_files} more generated test file(s)[/dim]")


@click.command("generate")
@click.option("--agent", "agent_url", help="Agent endpoint URL. Defaults to config or auto-detect.")
@click.option("--adapter", "adapter_type", default=None, help="Adapter type (default: config or http).")
@click.option("--budget", default=None, type=click.IntRange(1, 100), help="Number of probe runs. If omitted, you'll be asked interactively.")
@click.option("--out", "out_dir", default="tests/generated", help="Output directory for generated tests.")
@click.option("--seed", "seed_path", help="Path to newline-delimited seed prompts.")
@click.option("--from-log", "from_log", type=click.Path(exists=True), help="Generate from an existing log file instead of live probing.")
@click.option(
    "--log-format",
    "log_format",
    default="auto",
    type=click.Choice(["auto", "jsonl", "openai", "evalview"]),
    help="Log format when using --from-log.",
)
@click.option("--include-tools", help="Comma-separated tool names to focus on.")
@click.option("--exclude-tools", help="Comma-separated tool names to avoid.")
@click.option("--timeout", default=120.0, type=float, help="Probe timeout in seconds.")
@click.option("--allow-private-urls", is_flag=True, help="Allow private/local agent URLs.")
@click.option("--allow-live-side-effects", is_flag=True, help="Allow prompts that may trigger side-effecting tools.")
@click.option("--keep-old", is_flag=True, help="Keep existing generated drafts instead of replacing the output folder.")
@click.option("--dry-run", is_flag=True, help="Preview generation without writing files.")
@click.option("--no-synthesize", is_flag=True, help="Skip LLM-powered prompt synthesis (use heuristic prompts only).")
@track_command("generate")
def generate(
    agent_url: str | None,
    adapter_type: str | None,
    budget: int,
    out_dir: str,
    seed_path: str | None,
    from_log: str | None,
    log_format: str,
    include_tools: str | None,
    exclude_tools: str | None,
    timeout: float,
    allow_private_urls: bool,
    allow_live_side_effects: bool,
    keep_old: bool,
    dry_run: bool,
    no_synthesize: bool,
) -> None:
    """Generate a draft regression suite from live agent probing.

    Examples:
        evalview generate --agent http://localhost:8000
        evalview generate --budget 20 --seed prompts.txt
        evalview generate --dry-run
    """
    config = _load_config_if_exists()

    # Interactive budget selection when not explicitly provided
    if budget is None and from_log is None:
        console.print("[bold]How many tests to generate?[/bold]\n")
        console.print("  [cyan]1.[/cyan] Quick    (~4 tests,  ~2 min)   [dim]← recommended[/dim]")
        console.print("  [cyan]2.[/cyan] Standard (~8 tests,  ~5 min)")
        console.print("  [cyan]3.[/cyan] Thorough (~20 tests, ~12 min)")
        console.print()
        choice = click.prompt("Choice", default="1", show_default=False).strip()
        budget_map = {"1": 4, "2": 8, "3": 20}
        if choice in budget_map:
            budget = budget_map[choice]
        else:
            # Treat as a direct number
            try:
                budget = max(1, min(100, int(choice)))
            except ValueError:
                budget = 4
        console.print()
    elif budget is None:
        budget = 4

    needs_live_agent = from_log is None
    endpoint = (
        agent_url
        or (config.endpoint if config else None)
        or (_detect_agent_endpoint() if needs_live_agent else None)
    )
    resolved_adapter = adapter_type or (config.adapter if config else "http")

    if needs_live_agent and not endpoint:
        console.print("[red]✗ No agent endpoint configured or detected.[/red]")
        console.print("[dim]Pass --agent http://localhost:8000 or run evalview init first.[/dim]")
        raise click.Abort()

    if budget < 3:
        console.print("[yellow]⚠ Budget below 3 will produce very weak coverage.[/yellow]")

    if from_log and seed_path:
        console.print("[yellow]Ignoring --seed because --from-log was provided.[/yellow]")

    seed_prompts = [] if from_log else load_seed_prompts(seed_path)
    included = [item.strip() for item in include_tools.split(",")] if include_tools else []
    excluded = [item.strip() for item in exclude_tools.split(",")] if exclude_tools else []
    adapter = None
    if needs_live_agent or resolved_adapter == "mcp":
        adapter = create_adapter(
            adapter_type=resolved_adapter,
            endpoint=endpoint or "",
            timeout=timeout,
            allow_private_urls=allow_private_urls or bool(getattr(config, "allow_private_urls", False)),
        )

    console.print("[bold cyan]Generating draft suite[/bold cyan]")
    console.print(f"[dim]Adapter:[/dim] {resolved_adapter}")
    if endpoint:
        console.print(f"[dim]Endpoint:[/dim] {endpoint}")
    if from_log:
        console.print(f"[dim]Source:[/dim] log file ({from_log})")
    else:
        console.print(f"[dim]Probe budget:[/dim] {budget}")
        if seed_prompts:
            console.print(f"[dim]Seed prompts:[/dim] {len(seed_prompts)}")
    if included:
        console.print(f"[dim]Include tools:[/dim] {', '.join(included)}")
    if excluded:
        console.print(f"[dim]Exclude tools:[/dim] {', '.join(excluded)}")
    if not allow_live_side_effects:
        console.print("[dim]Side effects:[/dim] safe mode")
    console.print()

    # Shared state for the probe progress — uses a background thread to
    # keep the elapsed timer ticking while asyncio.run() blocks the main thread.
    _gen_start = time.time()
    _gen_state = {"phase": "", "completed": 0, "total": 0, "stop": False}

    def _format_gen_elapsed() -> str:
        elapsed = time.time() - _gen_start
        mins, secs = divmod(elapsed, 60)
        return f"{int(mins):02d}:{int(secs):02d}"

    def _timer_thread() -> None:
        """Background thread that reprints the status line every second."""
        while not _gen_state["stop"]:
            n = _gen_state["completed"]
            t = _gen_state["total"] or "?"
            phase = _gen_state["phase"] or "Starting..."
            # \r + clear line + reprint — works in most terminals
            line = f"\r\033[K  ⏱  {_format_gen_elapsed()}  [{n}/{t}]  {phase}"
            try:
                console.file.write(line)
                console.file.flush()
            except Exception:
                pass
            time.sleep(1)
        # Clear the timer line when done
        try:
            console.file.write("\r\033[K")
            console.file.flush()
        except Exception:
            pass

    _timer = threading.Thread(target=_timer_thread, daemon=True)

    def _on_probe(num: int, total: int, query: str, status: str, tools: list) -> None:
        _gen_state["total"] = total
        if status == "info":
            _gen_state["phase"] = query
            return
        # Clear timer line before printing result
        try:
            console.file.write("\r\033[K")
            console.file.flush()
        except Exception:
            pass
        _gen_state["completed"] = num
        if status == "fail":
            console.print(f"[dim]  [red]✗[/red] [{num}/{total}] {query} [timeout][/dim]")
        elif tools:
            console.print(f"[dim]  [green]✓[/green] [{num}/{total}] {query} → {', '.join(tools[:3])}[/dim]")
        else:
            console.print(f"[dim]  [green]✓[/green] [{num}/{total}] {query}[/dim]")
        if num < total:
            _gen_state["phase"] = f"Probing [{num + 1}/{total}]..."
        else:
            _gen_state["phase"] = "Building tests..."

    if from_log:
        from evalview.importers.log_importer import parse_log_file

        generator = AgentTestGenerator(
            adapter=adapter,
            endpoint=endpoint or "",
            adapter_type=resolved_adapter,
            include_tools=included,
            exclude_tools=excluded,
            allow_live_side_effects=allow_live_side_effects,
            project_root=Path.cwd(),
        )
        entries = parse_log_file(Path(from_log), fmt=log_format, max_entries=budget)
        result = generator.generate_from_log_entries(entries)
    else:
        _gen_state["total"] = budget
        _gen_state["phase"] = f"Probing [1/{budget}]..."
        _timer.start()

        result = run_generation(
            adapter=adapter,
            endpoint=endpoint or "",
            adapter_type=resolved_adapter,
            budget=budget,
            seed_prompts=seed_prompts,
            include_tools=included,
            exclude_tools=excluded,
            allow_live_side_effects=allow_live_side_effects,
            project_root=Path.cwd(),
            synthesize=not no_synthesize,
            on_probe_complete=_on_probe,
        )

    _gen_state["stop"] = True
    if _timer.is_alive():
        _timer.join(timeout=2)

    if not result.tests:
        console.print("[yellow]⚠ No draft tests were generated.[/yellow]")
        if result.failures:
            console.print("[dim]Probe failures:[/dim]")
            for failure in result.failures[:5]:
                console.print(f"[dim]  • {failure}[/dim]")
        _print_generate_failure_guidance(
            endpoint=endpoint,
            agent_url=agent_url,
            from_log=from_log,
        )
        raise click.Abort()

    generator = AgentTestGenerator(
        adapter=adapter,
        endpoint=endpoint or "",
        adapter_type=resolved_adapter,
        include_tools=included,
        exclude_tools=excluded,
        allow_live_side_effects=allow_live_side_effects,
        project_root=Path.cwd(),
    )
    output_dir = Path(out_dir)

    # ── Show all tests for review before writing ──────────────────────────
    _print_test_summary_table(result.tests)
    _print_test_yaml_inline(result.tests, generator)

    if result.failures:
        console.print(f"[yellow]⚠ {len(result.failures)} probe(s) failed (timeout / error):[/yellow]")
        for failure in result.failures[:5]:
            console.print(f"[dim]  • {failure[:120]}[/dim]")
        if len(result.failures) > 5:
            console.print(f"[dim]  + {len(result.failures) - 5} more[/dim]")
        console.print()

    if dry_run:
        console.print(f"[green]✓ Would generate {len(result.tests)} draft tests[/green]")
    else:
        # ── Ask for approval ──────────────────────────────────────────────
        approved = click.confirm(
            f"Save these {len(result.tests)} tests to {out_dir}?",
            default=True,
        )
        if not approved:
            console.print("[dim]Discarded. Re-run with --seed or --budget to adjust.[/dim]")
            raise click.Abort()

        replacing_existing = output_dir.exists() and any(output_dir.iterdir())
        generated_yaml, handwritten_yaml = generator.classify_output_dir(output_dir)
        full_replace_confirmed = False
        if handwritten_yaml and not keep_old:
            keep_handwritten = click.confirm(
                f"{output_dir} contains {len(handwritten_yaml)} hand-written YAML test(s). "
                "Keep them and replace only EvalView-generated drafts?",
                default=True,
            )
            if not keep_handwritten:
                generator._replace_all_yaml_suite(output_dir)
                full_replace_confirmed = True
                replacing_existing = False

        written = generator.write_suite(
            result,
            output_dir,
            replace_existing=not keep_old and not full_replace_confirmed,
        )
        ProjectStateStore().set_active_test_path(out_dir)
        console.print(f"[green]✓ Saved {len(result.tests)} tests[/green] [dim]({_format_gen_elapsed()} elapsed)[/dim]")
        console.print(f"[dim]Output:[/dim] {output_dir}")
        console.print(f"[dim]Files written:[/dim] {len(written)}")
        if full_replace_confirmed:
            console.print("[dim]Replaced all YAML drafts in this folder, including hand-written tests.[/dim]")
        elif replacing_existing and not keep_old:
            console.print("[dim]Replaced previous generated drafts in this folder.[/dim]")

    # Explain clustering if probes > tests
    if result.probes_run > len(result.tests) and len(result.tests) > 0:
        console.print(
            f"[dim]{result.probes_run} probes → {len(result.tests)} tests "
            f"(duplicate behavior paths were merged)[/dim]"
        )

    covered = result.report.get("covered", {})
    discovery = result.report.get("discovery", {})
    console.print()
    if discovery.get("count"):
        console.print(f"[bold]Discovery[/bold]\n  tools: {discovery['count']}")
        console.print()
    console.print("[bold]Coverage[/bold]")
    console.print(f"  tool paths: {covered.get('tool_paths', 0)}")
    console.print(f"  direct answers: {covered.get('direct_answers', 0)}")
    console.print(f"  clarifications: {covered.get('clarifications', 0)}")
    console.print(f"  multi-turn: {covered.get('multi_turn', 0)}")
    console.print(f"  refusals: {covered.get('refusals', 0)}")
    console.print(f"  error paths: {covered.get('error_paths', 0)}")

    tools_seen = result.report.get("tools_seen", {})
    if tools_seen:
        console.print()
        console.print("[bold]Tools seen[/bold]")
        for tool_name, count in tools_seen.items():
            console.print(f"  {tool_name}: {count}")

    prompt_sources = result.report.get("prompt_sources", {})
    if prompt_sources:
        console.print()
        console.print("[bold]Prompt sources[/bold]")
        for source, count in prompt_sources.items():
            console.print(f"  {source}: {count}")

    synthesis_info = result.report.get("prompt_synthesis", {})
    if synthesis_info.get("count", 0) > 0:
        console.print()
        console.print(
            f"[bold]Prompt synthesis[/bold]: {synthesis_info['count']} "
            "domain-specific prompts via LLM"
        )

    gaps = result.report.get("gaps", [])
    if gaps:
        console.print()
        console.print("[bold]Gaps[/bold]")
        for gap in gaps:
            console.print(f"  • {gap}")

    console.print()
    if dry_run:
        console.print("[dim]Re-run without --dry-run to write tests.[/dim]")
    else:
        console.print(f"[dim]Next: review {output_dir}, then run evalview snapshot {out_dir}[/dim]")
