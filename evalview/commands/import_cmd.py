"""Import command — convert production logs into test cases."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("import")
@click.argument("log_file", type=click.Path(exists=True))
@click.option(
    "--format", "fmt",
    default="auto",
    type=click.Choice(["auto", "jsonl", "openai", "evalview"]),
    help="Log format (default: auto-detect)",
)
@click.option(
    "--output-dir", "output_dir",
    default="tests/imported",
    show_default=True,
    help="Directory for generated test YAML files",
)
@click.option(
    "--max", "max_entries",
    default=50,
    show_default=True,
    help="Maximum number of log entries to import",
)
@click.option(
    "--prefix",
    default="imported",
    show_default=True,
    help="Filename prefix for generated test cases",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview what would be imported without writing files",
)
@track_command("import")
def import_logs(
    log_file: str,
    fmt: str,
    output_dir: str,
    max_entries: int,
    prefix: str,
    dry_run: bool,
) -> None:
    """Convert production logs into EvalView test cases.

    Reads a log file and generates one YAML test case per entry.
    Supports JSONL, OpenAI chat-completion logs, and EvalView capture format.

    \b
    Examples:
        evalview import prod.jsonl
        evalview import traces.jsonl --format openai --output-dir tests/prod
        evalview import logs.jsonl --max 100 --dry-run
    """
    from evalview.importers.log_importer import parse_log_file, detect_format, entries_to_yaml

    path = Path(log_file)

    detected = fmt if fmt != "auto" else detect_format(path)
    if detected == "unknown":
        console.print(
            f"\n[yellow]⚠ Could not detect log format for {path.name}.[/yellow]\n"
            "Specify with [bold]--format jsonl|openai|evalview[/bold]\n"
        )
        sys.exit(1)

    fmt_label = f"[cyan]{detected}[/cyan]" + (" [dim](auto-detected)[/dim]" if fmt == "auto" else "")
    console.print(f"\n[cyan]◈ Importing {path.name}[/cyan]  format: {fmt_label}\n")

    entries = parse_log_file(path, fmt=detected, max_entries=max_entries)

    if not entries:
        console.print("[yellow]No entries found in log file.[/yellow]\n")
        sys.exit(1)

    # Preview
    from rich.table import Table
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Query", min_width=45)
    table.add_column("Tools", style="cyan")
    table.add_column("Has Output", justify="center", width=10)

    for i, e in enumerate(entries[:10], 1):
        tools_str = ", ".join(e.tool_calls) if e.tool_calls else "[dim]—[/dim]"
        has_out = "[green]✓[/green]" if e.output else "[dim]—[/dim]"
        query_preview = e.query[:60] + "…" if len(e.query) > 60 else e.query
        table.add_row(str(i), query_preview, tools_str, has_out)

    if len(entries) > 10:
        table.add_row("[dim]…[/dim]", f"[dim]+{len(entries) - 10} more[/dim]", "", "")

    console.print(table)
    console.print()

    if dry_run:
        console.print(f"[dim]Dry run — {len(entries)} test cases would be written to {output_dir}/[/dim]\n")
        return

    out = Path(output_dir)
    written = entries_to_yaml(entries, out, name_prefix=prefix)

    console.print(f"[green]✓ {len(written)} test cases written to {out}/[/green]")
    console.print(f"[dim]Next: evalview snapshot   (capture baseline)[/dim]\n")
