"""Generate command — draft a regression suite by probing an agent."""
from __future__ import annotations

from pathlib import Path

import click

from evalview.commands.shared import _detect_agent_endpoint, _load_config_if_exists, console
from evalview.core.adapter_factory import create_adapter
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


@click.command("generate")
@click.option("--agent", "agent_url", help="Agent endpoint URL. Defaults to config or auto-detect.")
@click.option("--adapter", "adapter_type", default=None, help="Adapter type (default: config or http).")
@click.option("--budget", default=20, type=click.IntRange(1, 100), help="Maximum number of probe runs.")
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
@click.option("--timeout", default=30.0, type=float, help="Probe timeout in seconds.")
@click.option("--allow-private-urls", is_flag=True, help="Allow private/local agent URLs.")
@click.option("--allow-live-side-effects", is_flag=True, help="Allow prompts that may trigger side-effecting tools.")
@click.option("--dry-run", is_flag=True, help="Preview generation without writing files.")
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
    dry_run: bool,
) -> None:
    """Generate a draft regression suite from live agent probing.

    Examples:
        evalview generate --agent http://localhost:8000
        evalview generate --budget 40 --seed prompts.txt
        evalview generate --dry-run
    """
    config = _load_config_if_exists()

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

    if from_log:
        from evalview.importers.log_importer import parse_log_file

        generator = AgentTestGenerator(
            adapter=adapter,
            endpoint=endpoint or "",
            adapter_type=resolved_adapter,
            include_tools=included,
            exclude_tools=excluded,
            allow_live_side_effects=allow_live_side_effects,
        )
        entries = parse_log_file(Path(from_log), fmt=log_format, max_entries=budget)
        result = generator.generate_from_log_entries(entries)
    else:
        result = run_generation(
            adapter=adapter,
            endpoint=endpoint or "",
            adapter_type=resolved_adapter,
            budget=budget,
            seed_prompts=seed_prompts,
            include_tools=included,
            exclude_tools=excluded,
            allow_live_side_effects=allow_live_side_effects,
        )

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
    )
    output_dir = Path(out_dir)

    if dry_run:
        console.print(f"[green]✓ Would generate {len(result.tests)} draft tests[/green]")
    else:
        written = generator.write_suite(result, output_dir)
        console.print(f"[green]✓ Generated {len(result.tests)} draft tests[/green]")
        console.print(f"[dim]Output:[/dim] {output_dir}")
        console.print(f"[dim]Files written:[/dim] {len(written)}")

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
