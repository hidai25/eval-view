"""Feedback command — open a pre-filled GitHub issue or discussion."""

import platform
import webbrowser
from urllib.parse import quote

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("evalview")
except Exception:
    _VERSION = "unknown"


def _build_url(category: str, title: str, body: str) -> str:
    """Build a GitHub new-issue URL with pre-filled fields."""
    base = "https://github.com/hidai25/eval-view/issues/new"
    labels = {"bug": "bug", "feature": "enhancement", "question": "question"}.get(
        category, ""
    )
    params = f"title={quote(title)}&body={quote(body)}"
    if labels:
        params += f"&labels={quote(labels)}"
    return f"{base}?{params}"


def _env_block() -> str:
    """Collect environment info for the issue body."""
    import sys

    lines = [
        f"- EvalView: {_VERSION}",
        f"- Python: {sys.version.split()[0]}",
        f"- OS: {platform.system()} {platform.release()}",
    ]
    return "\n".join(lines)


@click.command("feedback")
@click.option(
    "--bug", "category", flag_value="bug", help="Report a bug.",
)
@click.option(
    "--feature", "category", flag_value="feature", help="Request a feature.",
)
@click.option(
    "--question", "category", flag_value="question", help="Ask a question.",
)
@track_command("feedback")
def feedback(category: str) -> None:
    """Send feedback, report a bug, or request a feature.

    Opens a pre-filled GitHub issue in your browser.
    """
    if not category:
        category = click.prompt(
            "What kind of feedback?",
            type=click.Choice(["bug", "feature", "question"], case_sensitive=False),
        )

    prompts = {
        "bug": "Briefly describe the bug",
        "feature": "Briefly describe the feature",
        "question": "What's your question",
    }

    title = click.prompt(prompts[category])

    templates = {
        "bug": (
            "## What happened\n\n"
            "{title}\n\n"
            "## Steps to reproduce\n\n"
            "1. \n2. \n3. \n\n"
            "## Expected behavior\n\n\n\n"
            "## Environment\n\n{env}\n"
        ),
        "feature": (
            "## What\n\n"
            "{title}\n\n"
            "## Why\n\n\n\n"
            "## Environment\n\n{env}\n"
        ),
        "question": (
            "{title}\n\n"
            "## Context\n\n\n\n"
            "## Environment\n\n{env}\n"
        ),
    }

    body = templates[category].format(title=title, env=_env_block())
    url = _build_url(category, f"[{category}] {title}", body)

    console.print()
    console.print(f"[green]Opening GitHub...[/green]")
    webbrowser.open(url)
    console.print("[dim]If the browser didn't open, copy this URL:[/dim]")
    console.print(f"[dim]{url}[/dim]")
    console.print()
