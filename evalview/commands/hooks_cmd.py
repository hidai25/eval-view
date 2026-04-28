"""Git hook management commands."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


_HOOK_BEGIN = "# ── BEGIN evalview-managed block ──────────────────────────────────────"
_HOOK_END   = "# ── END evalview-managed block ────────────────────────────────────────"

_HOOK_SCRIPT_TEMPLATE = """\
{begin}
# Installed by: evalview install-hooks
# Do not edit this block — use 'evalview uninstall-hooks' to remove it.
EVALVIEW_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {{ echo "[evalview] Not inside a git repo — skipping."; exit 0; }}
EVALVIEW_GOLDEN_DIR="$EVALVIEW_ROOT/.evalview/golden"
if ! find "$EVALVIEW_GOLDEN_DIR" -maxdepth 1 -name "*.golden.json" 2>/dev/null | grep -q .; then
    echo "[evalview] No baseline found — skipping regression check."
    exit 0
fi
echo "[evalview] Running regression check..."
"{python}" -m evalview check --fail-on REGRESSION
EVALVIEW_EXIT=$?
if [ $EVALVIEW_EXIT -ne 0 ]; then
    echo "[evalview] Push blocked: regression detected. Run 'evalview check' for details."
    exit 1
fi
{end}
"""


def _find_git_hooks_dir(git_dir: Optional[str]) -> Optional[Path]:
    """Return the git hooks directory, walking up from cwd if not specified.

    Args:
        git_dir: Explicit path to the ``.git`` directory, or ``None`` to
            auto-detect by walking up from the current working directory.

    Returns:
        Path to the hooks directory inside the found ``.git`` dir, or
        ``None`` if no git repository could be located.
    """
    if git_dir:
        return Path(git_dir) / "hooks"
    search = Path.cwd()
    for candidate_dir in [search, *search.parents]:
        dot_git = candidate_dir / ".git"
        if dot_git.is_dir():
            return dot_git / "hooks"
        # git worktrees store .git as a file pointing to the real git dir
        if dot_git.is_file():
            for line in dot_git.read_text(encoding="utf-8").splitlines():
                if line.startswith("gitdir:"):
                    real_git = Path(line[len("gitdir:"):].strip())
                    if not real_git.is_absolute():
                        real_git = candidate_dir / real_git
                    # Worktrees share hooks with the main repo
                    main_git = real_git
                    while (main_git / "commondir").exists():
                        common = (main_git / "commondir").read_text(encoding="utf-8").strip()
                        main_git = (main_git / common).resolve()
                    return main_git / "hooks"
    return None


@click.command("install-hooks")
@click.option(
    "--hook",
    "hook_name",
    default="pre-push",
    type=click.Choice(["pre-push", "pre-commit"]),
    show_default=True,
    help="Which git hook to install the regression check into.",
)
@click.option(
    "--git-dir",
    "git_dir",
    default=None,
    help="Path to your .git directory (auto-detected if not set).",
)
@track_command("install_hooks")
def install_hooks(hook_name: str, git_dir: Optional[str]) -> None:
    """Install a git hook that runs 'evalview check' automatically.

    Injects a managed block into your pre-push (or pre-commit) hook so
    every push is guarded against regressions — no manual habit required.

    The hook is idempotent: running install-hooks again on a repo that
    already has the block is safe and does nothing.

    If no golden baselines exist yet the hook exits silently, so it never
    blocks a push on a fresh clone.

    \b
    Examples:
        evalview install-hooks                 # default: pre-push
        evalview install-hooks --hook pre-commit
        evalview install-hooks --git-dir /path/to/.git
    """
    import platform
    import stat

    hooks_dir = _find_git_hooks_dir(git_dir)
    if hooks_dir is None:
        console.print(
            "[red]✗ No git repository found.[/red] "
            "Run this command from inside a git repo, or pass --git-dir."
        )
        raise SystemExit(1)

    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / hook_name

    managed_block = _HOOK_SCRIPT_TEMPLATE.format(
        begin=_HOOK_BEGIN,
        end=_HOOK_END,
        python=sys.executable,
    )

    # ── Read existing hook (or start with shebang) ──────────────────────────
    if hook_path.exists():
        existing = hook_path.read_text(encoding="utf-8")
        if _HOOK_BEGIN in existing:
            console.print(
                f"[yellow]⚠ evalview block already present in {hook_name}.[/yellow] "
                "Nothing changed."
            )
            return
        new_content = existing.rstrip("\n") + "\n\n" + managed_block
    else:
        new_content = "#!/usr/bin/env bash\n\n" + managed_block

    hook_path.write_text(new_content, encoding="utf-8")

    # ── Make executable (POSIX only) ─────────────────────────────────────────
    if platform.system() != "Windows":
        current = hook_path.stat().st_mode
        hook_path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    console.print(
        f"[green]✓ Installed evalview regression check into [bold]{hook_name}[/bold].[/green]"
    )
    console.print(
        f"  Hook path: [dim]{hook_path}[/dim]"
    )
    console.print(
        "  Every push will now run [bold]evalview check[/bold] automatically. "
        "To remove, run [bold]evalview uninstall-hooks[/bold]."
    )


@click.command("uninstall-hooks")
@click.option(
    "--hook",
    "hook_name",
    default="pre-push",
    type=click.Choice(["pre-push", "pre-commit"]),
    show_default=True,
    help="Which git hook to remove the evalview block from.",
)
@click.option(
    "--git-dir",
    "git_dir",
    default=None,
    help="Path to your .git directory (auto-detected if not set).",
)
@track_command("uninstall_hooks")
def uninstall_hooks(hook_name: str, git_dir: Optional[str]) -> None:
    """Remove the evalview-managed block from a git hook.

    Strips only the block that was added by 'evalview install-hooks',
    leaving any other content in the hook file untouched.  If the hook
    file becomes empty after removal (i.e. it contained only the evalview
    block and the shebang), the file is deleted entirely.

    \b
    Examples:
        evalview uninstall-hooks                 # default: pre-push
        evalview uninstall-hooks --hook pre-commit
    """
    hooks_dir = _find_git_hooks_dir(git_dir)
    if hooks_dir is None:
        console.print(
            "[red]✗ No git repository found.[/red] "
            "Run this command from inside a git repo, or pass --git-dir."
        )
        raise SystemExit(1)

    hook_path = hooks_dir / hook_name

    if not hook_path.exists():
        console.print(f"[dim]No {hook_name} hook found — nothing to do.[/dim]")
        return

    existing = hook_path.read_text(encoding="utf-8")

    if _HOOK_BEGIN not in existing:
        console.print(
            f"[dim]No evalview block found in {hook_name} — nothing to do.[/dim]"
        )
        return

    # Strip the managed block (and any blank lines immediately before it)
    lines = existing.splitlines(keepends=True)
    out: List[str] = []
    inside_block = False
    for line in lines:
        if line.rstrip("\r\n") == _HOOK_BEGIN:
            inside_block = True
            # Also remove the trailing blank line we may have added above the block
            while out and out[-1].strip() == "":
                out.pop()
            continue
        if line.rstrip("\r\n") == _HOOK_END:
            inside_block = False
            continue
        if not inside_block:
            out.append(line)

    stripped = "".join(out).rstrip("\n")

    # If only the shebang (or nothing) remains, delete the file
    meaningful = [line for line in stripped.splitlines() if line.strip() and not line.startswith("#!")]
    if not meaningful:
        hook_path.unlink()
        console.print(
            f"[green]✓ Removed evalview block from [bold]{hook_name}[/bold] "
            "(hook file was otherwise empty and has been deleted).[/green]"
        )
    else:
        mode = hook_path.stat().st_mode
        hook_path.write_text(stripped + "\n", encoding="utf-8")
        hook_path.chmod(mode)  # restore permissions (write() may reset via umask)
        console.print(
            f"[green]✓ Removed evalview block from [bold]{hook_name}[/bold].[/green]"
        )
        console.print(f"  Other hook content preserved. Hook path: [dim]{hook_path}[/dim]")
