"""Add command — copy a test pattern template into the project."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("add")
@click.argument("pattern", required=False)
@click.option("--tool", help="Tool name to use in the test")
@click.option("--query", help="Query to use in the test")
@click.option("--list", "list_patterns", is_flag=True, help="List available patterns")
@click.option("--output", "-o", help="Output file path (default: tests/<pattern>.yaml)")
@track_command("add")
def add(pattern: Optional[str], tool: Optional[str], query: Optional[str], list_patterns: bool, output: Optional[str]):
    """Add a test pattern to your project.

    Examples:
        evalview add                           # List available patterns
        evalview add tool-not-called           # Copy pattern to tests/
        evalview add cost-budget --output my-test.yaml
        evalview add tool-not-called --tool get_weather --query "What's the weather?"
    """
    # Find templates directory
    templates_dir = Path(__file__).parent.parent / "templates" / "patterns"

    if not templates_dir.exists():
        console.print("[red]Error: Templates directory not found[/red]")
        return

    # List available patterns
    available_patterns = [f.stem for f in templates_dir.glob("*.yaml")]

    if list_patterns or not pattern:
        console.print("\n[bold cyan]Available Test Patterns[/bold cyan]\n")

        for p in sorted(available_patterns):
            # Read description from file
            pattern_file = templates_dir / f"{p}.yaml"
            with open(pattern_file) as f:
                content = f.read()
                # Extract first comment line as description
                lines = content.split("\n")
                desc = ""
                for line in lines:
                    if line.startswith("# Pattern:"):
                        desc = line.replace("# Pattern:", "").strip()
                        break
                    elif line.startswith("#") and not line.startswith("# "):
                        continue
                    elif line.startswith("# ") and "Common failure" not in line and "Customize" not in line:
                        desc = line.replace("# ", "").strip()
                        if desc:
                            break

            console.print(f"  [green]{p}[/green]")
            if desc:
                console.print(f"    [dim]{desc}[/dim]")

        console.print("\n[dim]Usage: evalview add <pattern-name>[/dim]")
        console.print("[dim]       evalview add <pattern-name> --tool my_tool --query \"My query\"[/dim]\n")
        return

    # Check if pattern exists
    if pattern not in available_patterns:
        console.print(f"[red]Error: Pattern '{pattern}' not found[/red]")
        console.print(f"[dim]Available: {', '.join(available_patterns)}[/dim]")
        return

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        # Create tests directory if needed
        tests_dir = Path("tests")
        tests_dir.mkdir(exist_ok=True)
        output_path = tests_dir / f"{pattern}.yaml"

    # Check if file exists
    if output_path.exists():
        if not click.confirm(f"File {output_path} already exists. Overwrite?"):
            console.print("[yellow]Aborted[/yellow]")
            return

    # Read template
    template_path = templates_dir / f"{pattern}.yaml"
    with open(template_path) as f:
        content = f.read()

    # Apply substitutions if provided
    if tool:
        # Replace tool names in common patterns
        content = content.replace("calculator", tool)
        content = content.replace("retriever", tool)

    if query:
        # Replace query strings
        content = re.sub(r'query: "[^"]*"', f'query: "{query}"', content)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    console.print(f"\n[green]✓[/green] Created [cyan]{output_path}[/cyan]")

    # Show what was created
    console.print(f"\n[dim]━━━ {output_path} ━━━[/dim]")
    # Show first 20 lines
    lines = content.split("\n")[:20]
    for line in lines:
        if line.startswith("#"):
            console.print(f"[dim]{line}[/dim]")
        else:
            console.print(line)
    if len(content.split("\n")) > 20:
        console.print("[dim]...[/dim]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"  1. Edit [cyan]{output_path}[/cyan] to match your agent")
    console.print(f"  2. Run: [green]evalview run {output_path}[/green]\n")
