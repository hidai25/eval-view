"""autopr command — turn production incidents into regression tests + PRs.

This is the "glue" command that closes the loop:

    evalview monitor  ──▶  .evalview/incidents.jsonl  ──▶  evalview autopr  ──▶  PR

``evalview monitor`` appends one incident record per confirmed regression to
``.evalview/incidents.jsonl``.  ``evalview autopr`` reads those records,
synthesizes a regression test YAML for each new incident under
``tests/regressions/``, commits them on a branch, and (if ``--open-pr``) uses
the ``gh`` CLI to open a pull request.

Design goals:

- **Local-first.** Default mode writes files and prints ``git`` commands. No
  network required.
- **Idempotent.** Skips incidents whose synthesized test already exists.
- **Honest exit codes.** Exits 0 only when at least one new test was written
  (``--require-new``) or always-0 for dry runs.
- **Safe.** Never force-pushes, never rewrites history, never auto-merges.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import yaml  # type: ignore[import-untyped]

from evalview.commands.shared import console
from evalview.core.regression_synth import (
    SynthesisError,
    incident_slug,
    synthesize_regression_test,
)
from evalview.telemetry.decorators import track_command


DEFAULT_INCIDENTS_PATH = Path(".evalview/incidents.jsonl")
DEFAULT_TESTS_DIR = Path("tests/regressions")
DEFAULT_BRANCH_PREFIX = "evalview/autopr"


class AutoprError(Exception):
    """Raised when autopr cannot complete a requested action."""


def load_incidents(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL incidents file into a list of dicts.

    Malformed lines are logged and skipped — one bad row shouldn't poison the
    whole run.  Returns an empty list when the file is missing: autopr is
    designed to be wired into a recurring workflow, and "no incidents this
    cycle" is a perfectly normal, non-error outcome.
    """
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                console.print(
                    f"[yellow]  Warning: skipping malformed incident on line {i} "
                    f"of {path}: {e}[/yellow]"
                )
    return records


def _existing_slugs(tests_dir: Path) -> set[str]:
    """Collect incident slugs already present in ``tests_dir``.

    Looks at ``meta.incident.slug`` in each YAML file — that's how the
    synthesizer marks which incident produced which test.
    """
    slugs: set[str] = set()
    if not tests_dir.exists():
        return slugs
    for yaml_file in tests_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        meta = (data or {}).get("meta") or {}
        incident = meta.get("incident") or {}
        slug = incident.get("slug")
        if slug:
            slugs.add(slug)
    return slugs


def write_regression_test(
    incident: Dict[str, Any],
    tests_dir: Path,
    min_score: float,
) -> Tuple[Path, Dict[str, Any]]:
    """Synthesize a regression test for an incident and write it to disk.

    Returns the path and the dict that was written.
    """
    test = synthesize_regression_test(incident, min_score=min_score)
    slug = incident_slug(incident)
    tests_dir.mkdir(parents=True, exist_ok=True)
    path = tests_dir / f"{slug}.yaml"
    # ``yaml.safe_dump`` with sort_keys=False preserves the field order that
    # the synthesizer chose — a human reviewing the test reads name first,
    # then description, then input/expected.
    path.write_text(
        yaml.safe_dump(test, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path, test


def _run_git(args: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a git subcommand and capture output for error reporting."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _ensure_git_available() -> None:
    if shutil.which("git") is None:
        raise AutoprError(
            "git not found on PATH — autopr needs git to commit and push "
            "regression tests. Install git or run with --no-commit."
        )


def _ensure_clean_index(repo: Path) -> None:
    """Refuse to touch the working tree if the user has uncommitted changes.

    This is a safety guard: autopr is going to create a branch, add files,
    commit, and push. Doing that on top of a dirty index is how you lose
    in-progress work.
    """
    status = _run_git(["status", "--porcelain"], cwd=repo)
    if status.returncode != 0:
        raise AutoprError(
            f"git status failed: {status.stderr.strip() or status.stdout.strip()}"
        )
    dirty = [
        line for line in status.stdout.splitlines()
        if line and not line.startswith("??")
    ]
    if dirty:
        raise AutoprError(
            "working tree has uncommitted changes — commit or stash them "
            "before running `evalview autopr --commit` (see `git status`)."
        )


def _branch_name(prefix: str, stamp: Optional[str] = None) -> str:
    ts = stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}/{ts}"


def _commit_message(written: List[Tuple[Path, Dict[str, Any]]]) -> str:
    """Build a commit message that describes every new regression test."""
    header = (
        f"regression tests: auto-generated from {len(written)} "
        f"production incident{'s' if len(written) != 1 else ''}"
    )
    lines = [header, ""]
    lines.append(
        "Synthesized by `evalview autopr` from .evalview/incidents.jsonl."
    )
    lines.append("")
    for path, test in written:
        meta = (test.get("meta") or {}).get("incident") or {}
        source = meta.get("source_test", "?")
        status = meta.get("status", "?")
        delta = meta.get("score_delta")
        delta_str = f" ({delta:+.1f})" if isinstance(delta, (int, float)) else ""
        lines.append(f"- {path}: {source} [{status}]{delta_str}")
    lines.append("")
    lines.append("Review assertions before merging — generated tests pin")
    lines.append("the observed failing output and tool sequence.")
    return "\n".join(lines)


def _commit_and_push(
    repo: Path,
    branch: str,
    written: List[Tuple[Path, Dict[str, Any]]],
    push: bool,
) -> None:
    _ensure_git_available()
    _ensure_clean_index(repo)

    create = _run_git(["checkout", "-b", branch], cwd=repo)
    if create.returncode != 0:
        raise AutoprError(
            f"failed to create branch {branch}: "
            f"{create.stderr.strip() or create.stdout.strip()}"
        )

    add_args = ["add", "--"] + [str(p) for p, _ in written]
    added = _run_git(add_args, cwd=repo)
    if added.returncode != 0:
        raise AutoprError(
            f"git add failed: {added.stderr.strip() or added.stdout.strip()}"
        )

    commit = _run_git(
        ["commit", "-m", _commit_message(written)],
        cwd=repo,
    )
    if commit.returncode != 0:
        raise AutoprError(
            f"git commit failed: {commit.stderr.strip() or commit.stdout.strip()}"
        )

    if not push:
        return

    pushed = _run_git(["push", "-u", "origin", branch], cwd=repo)
    if pushed.returncode != 0:
        raise AutoprError(
            f"git push failed: {pushed.stderr.strip() or pushed.stdout.strip()}"
        )


def _open_pr_with_gh(
    branch: str,
    written: List[Tuple[Path, Dict[str, Any]]],
) -> Optional[str]:
    """Use the ``gh`` CLI to open a PR. Returns the PR URL on success.

    Missing ``gh`` is not an error — the command just prints manual
    instructions and continues.  The point of autopr is to *help*, not force
    a particular GitHub toolchain.
    """
    if shutil.which("gh") is None:
        console.print(
            "[yellow]  gh CLI not found — skipping automatic PR creation.[/yellow]"
        )
        console.print(
            f"[dim]  Create the PR manually:  gh pr create --head {branch}"
            f"  (or open the compare URL in a browser)[/dim]"
        )
        return None

    title = f"EvalView autopr: {len(written)} regression test(s) from prod incidents"
    body_lines = [
        "## Summary",
        "",
        f"- Adds {len(written)} regression test(s) auto-generated from",
        "  `.evalview/incidents.jsonl` by `evalview autopr`.",
        "- Each test pins the observed bad output and tool sequence so",
        "  the same production failure cannot recur.",
        "",
        "## Incidents covered",
        "",
    ]
    for path, test in written:
        meta = (test.get("meta") or {}).get("incident") or {}
        source = meta.get("source_test", "?")
        status = meta.get("status", "?")
        body_lines.append(f"- `{path.name}` — `{source}` ({status})")
    body_lines.extend([
        "",
        "## Review checklist",
        "",
        "- [ ] Assertions match the root cause (check `not_contains` / `forbidden_tools`)",
        "- [ ] `min_score` is appropriate for the severity",
        "- [ ] Test runs green when the fix is applied",
        "",
        "Generated by `evalview autopr`.",
    ])
    body = "\n".join(body_lines)

    proc = subprocess.run(
        ["gh", "pr", "create", "--head", branch, "--title", title, "--body", body],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        console.print(
            f"[yellow]  gh pr create failed: "
            f"{proc.stderr.strip() or proc.stdout.strip()}[/yellow]"
        )
        return None
    url = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return url or None


def _render_summary(
    written: List[Tuple[Path, Dict[str, Any]]],
    skipped: int,
) -> None:
    if not written and skipped == 0:
        console.print(
            "[dim]No incidents to process. "
            "Run `evalview monitor` to populate .evalview/incidents.jsonl.[/dim]"
        )
        return

    if written:
        console.print(
            f"[green]  Wrote {len(written)} regression test"
            f"{'s' if len(written) != 1 else ''}:[/green]"
        )
        for path, test in written:
            meta = (test.get("meta") or {}).get("incident") or {}
            source = meta.get("source_test", "?")
            console.print(f"    [cyan]{path}[/cyan]  [dim](from {source})[/dim]")

    if skipped:
        console.print(
            f"[dim]  Skipped {skipped} incident(s) — "
            f"regression tests already exist.[/dim]"
        )


@click.command("autopr")
@click.option(
    "--from",
    "incidents_path",
    default=str(DEFAULT_INCIDENTS_PATH),
    type=click.Path(),
    help="Path to incidents JSONL (written by `evalview monitor`).",
)
@click.option(
    "--tests-dir",
    default=str(DEFAULT_TESTS_DIR),
    type=click.Path(),
    help="Directory where regression test YAMLs are written.",
)
@click.option(
    "--min-score",
    type=float,
    default=90.0,
    help="min_score threshold applied to synthesized regression tests.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would be written without touching disk or git.",
)
@click.option(
    "--commit",
    is_flag=True,
    help="Create a branch, commit the new tests, and optionally push.",
)
@click.option(
    "--push/--no-push",
    default=True,
    help="Push the commit to origin (default: yes when --commit is set).",
)
@click.option(
    "--open-pr",
    is_flag=True,
    help="Use `gh pr create` to open a PR (implies --commit --push).",
)
@click.option(
    "--branch-prefix",
    default=DEFAULT_BRANCH_PREFIX,
    help="Branch name prefix (default: evalview/autopr).",
)
@click.option(
    "--require-new",
    is_flag=True,
    help="Exit 1 if no new regression tests were generated "
         "(useful in CI when you want the workflow to fail when there's "
         "nothing to do).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Only process the N most recent incidents.",
)
@track_command("autopr")
def autopr(
    incidents_path: str,
    tests_dir: str,
    min_score: float,
    dry_run: bool,
    commit: bool,
    push: bool,
    open_pr: bool,
    branch_prefix: str,
    require_new: bool,
    limit: Optional[int],
) -> None:
    """Turn production incidents into regression tests and open a PR.

    This is the closing glue of the EvalView loop: `evalview monitor` logs
    every confirmed production regression to `.evalview/incidents.jsonl`,
    and `evalview autopr` reads that file, synthesizes one regression test
    per incident under `tests/regressions/`, and (optionally) opens a PR.

    \b
    Typical usage:
        evalview autopr --dry-run            # see what would be written
        evalview autopr                      # write files only
        evalview autopr --commit             # write + commit on a new branch
        evalview autopr --open-pr            # write + commit + push + gh PR

    \b
    In CI:
        # .github/workflows/autopr.yml — see examples/github-workflows/
        - run: evalview autopr --open-pr --require-new

    The synthesizer is deterministic and local — no LLM, no network. See
    `evalview/core/regression_synth.py` for the full schema.
    """
    repo = Path.cwd()
    incidents_file = Path(incidents_path)
    tests_path = Path(tests_dir)

    incidents = load_incidents(incidents_file)
    if limit is not None and limit >= 0:
        incidents = incidents[-limit:]

    if not incidents:
        console.print(
            f"[dim]No incidents found at {incidents_file}. "
            f"Run `evalview monitor --history ...` to produce some.[/dim]"
        )
        if require_new:
            sys.exit(1)
        return

    existing = _existing_slugs(tests_path)
    written: List[Tuple[Path, Dict[str, Any]]] = []
    skipped = 0

    for incident in incidents:
        try:
            slug = incident_slug(incident)
        except Exception as e:  # pragma: no cover - defensive
            console.print(f"[yellow]  Skipping malformed incident: {e}[/yellow]")
            continue

        if slug in existing:
            skipped += 1
            continue

        try:
            test = synthesize_regression_test(incident, min_score=min_score)
        except SynthesisError as e:
            console.print(
                f"[yellow]  Skipping incident {slug}: {e}[/yellow]"
            )
            continue

        target = tests_path / f"{slug}.yaml"
        if dry_run:
            console.print(f"[dim]  Would write {target}[/dim]")
            written.append((target, test))
            existing.add(slug)
            continue

        tests_path.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(test, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append((target, test))
        existing.add(slug)

    _render_summary(written, skipped)

    if dry_run:
        if require_new and not written:
            sys.exit(1)
        return

    if not written:
        if require_new:
            sys.exit(1)
        return

    # git/gh side-effects only if explicitly requested.
    if open_pr:
        commit = True
        push = True

    if commit:
        branch = _branch_name(branch_prefix)
        try:
            _commit_and_push(repo, branch, written, push=push)
        except AutoprError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(2)
        console.print(
            f"[green]  Committed {len(written)} test(s) on branch "
            f"[cyan]{branch}[/cyan][/green]"
        )
        if push and open_pr:
            url = _open_pr_with_gh(branch, written)
            if url:
                console.print(f"[green]  PR opened: [cyan]{url}[/cyan][/green]")
    else:
        # Friendly hint for the manual path.
        console.print()
        console.print("[dim]Next steps:[/dim]")
        console.print(
            f"[dim]  git checkout -b evalview/autopr && "
            f"git add {tests_path} && git commit -m 'regression tests from "
            f"prod incidents' && git push -u origin HEAD[/dim]"
        )
        console.print(
            "[dim]  Or rerun with --open-pr to do the above automatically.[/dim]"
        )

    if require_new and not written:
        sys.exit(1)
