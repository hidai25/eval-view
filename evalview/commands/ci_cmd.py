"""CI/CD integration commands."""
from __future__ import annotations

import os
import sys
from typing import Optional

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.group()
def ci():
    """CI/CD integration commands.

    Commands for integrating EvalView with CI/CD pipelines.

    \b
    Examples:
        evalview ci comment              # Post results as PR comment
        evalview ci comment --dry-run    # Preview comment without posting
    """
    pass


@ci.command("comment")
@click.option(
    "--results",
    "-r",
    type=click.Path(exists=True),
    help="Path to results JSON file (default: latest in .evalview/results/)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print comment to stdout instead of posting to PR",
)
@click.option(
    "--update/--no-update",
    default=True,
    help="Update existing comment instead of creating new one (default: True)",
)
@track_command("ci_comment", lambda **kw: {"dry_run": kw.get("dry_run")})
def ci_comment(results: Optional[str], dry_run: bool, update: bool):
    """Post test results as a PR comment.

    Automatically detects PR context from GitHub Actions environment.
    Uses the `gh` CLI to post comments (pre-installed in GitHub Actions).

    \b
    Add to your workflow:
        - name: Post PR comment
          if: github.event_name == 'pull_request'
          run: evalview ci comment
          env:
            GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    """
    import json as json_module
    from evalview.ci.comment import (
        load_latest_results,
        generate_pr_comment,
        post_pr_comment,
        update_or_create_comment,
    )

    # Load results
    if results:
        with open(results) as f:
            data = json_module.load(f)
    else:
        data = load_latest_results()

    if not data:
        console.print("[red]No results found.[/red]")
        console.print("[dim]Run 'evalview run' first, or specify --results path.[/dim]")
        sys.exit(1)

    # Handle both list and dict formats
    if type(data).__name__ == "list":
        results_list = data
    elif type(data).__name__ == "dict" and "results" in data:
        results_list = data["results"]
    else:
        results_list = [data]

    # Check for diff results
    diff_results = None
    if type(data).__name__ == "dict" and "diff_results" in data:
        diff_results = data["diff_results"]

    # Get run URL from environment
    run_url = None
    github_server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    github_repo = os.environ.get("GITHUB_REPOSITORY", "")
    github_run_id = os.environ.get("GITHUB_RUN_ID", "")
    if github_repo and github_run_id:
        run_url = f"{github_server}/{github_repo}/actions/runs/{github_run_id}"

    # Generate comment
    comment = generate_pr_comment(results_list, diff_results, run_url)

    if dry_run:
        console.print("[cyan]━━━ PR Comment Preview ━━━[/cyan]\n")
        console.print(comment)
        console.print()
        return

    # Post comment
    if update:
        success = update_or_create_comment(comment)
    else:
        success = post_pr_comment(comment)

    if success:
        console.print("[green]✓ Posted PR comment[/green]")
    else:
        # Not in PR context or gh CLI not available - just print
        console.print("[yellow]Not in PR context or gh CLI not available.[/yellow]")
        console.print("[dim]Comment preview:[/dim]\n")
        console.print(comment)
