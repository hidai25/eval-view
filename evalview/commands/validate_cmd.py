"""Validate command -- lint test YAML/TOML files without running agents."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import yaml
from pydantic import ValidationError

from evalview.commands.shared import console
from evalview.core.loader import CONFIG_FILE_PATTERNS
from evalview.core.types import TestCase
from evalview.telemetry.decorators import track_command

# Resolve a TOML loader once at import time. Python 3.11+ ships tomllib in the
# stdlib; older versions need the optional `tomli` package. If neither is
# available we fall back gracefully: TOML files become parse-errors with a
# clear message instead of a confusing ModuleNotFoundError, and they are still
# collected so the user sees them in --json output.
try:
    import tomllib as _toml_loader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - py<3.11
    try:
        import tomli as _toml_loader  # type: ignore[import-not-found, no-redef]
    except ImportError:
        _toml_loader = None  # type: ignore[assignment]

# File extensions recognized as test cases. TOML is only collected when a
# loader is available so we don't surface false "missing dependency" errors
# for unrelated TOML in the directory (e.g. pyproject.toml).
_TEST_EXTENSIONS: Tuple[str, ...] = (".yaml", ".yml") + ((".toml",) if _toml_loader is not None else ())


@click.command("validate")
@click.argument("path", type=click.Path(exists=True))
@click.option("--json", "output_json", is_flag=True, help="Output machine-readable JSON")
@track_command("validate", lambda **kw: {"output_json": kw.get("output_json")})
def validate(path: str, output_json: bool) -> None:
    """Lint test YAML/TOML files for schema errors without running any agent.

    Validates each test file against the TestCase Pydantic schema and reports
    every error across every file (not just the first). Exits 1 if any file
    fails validation. No API key or agent required.

    Examples:
        evalview validate tests/
        evalview validate tests/single_test.yaml
        evalview validate tests/ --json
    """
    path_obj = Path(path)
    files = _collect_test_files(path_obj)

    if not files:
        if output_json:
            click.echo(
                json.dumps(
                    {
                        "valid": True,
                        "files_checked": 0,
                        "files_with_errors": 0,
                        "results": [],
                    }
                )
            )
        else:
            console.print(f"[yellow]No YAML/TOML test files found in {path}[/yellow]")
        sys.exit(0)

    start = time.time()
    results: List[Dict[str, Any]] = []
    has_errors = False

    for fp in files:
        file_errors = _validate_file(fp)
        if file_errors:
            has_errors = True
        results.append(
            {
                "file": str(fp),
                "valid": not file_errors,
                "errors": file_errors,
            }
        )

    elapsed_ms = (time.time() - start) * 1000

    if output_json:
        click.echo(
            json.dumps(
                {
                    "valid": not has_errors,
                    "files_checked": len(results),
                    "files_with_errors": sum(1 for r in results if not r["valid"]),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "results": results,
                },
                indent=2,
            )
        )
    else:
        _render_human_output(results, elapsed_ms)

    sys.exit(1 if has_errors else 0)


def _collect_test_files(path_obj: Path) -> List[Path]:
    if path_obj.is_file():
        return [path_obj] if path_obj.suffix.lower() in _TEST_EXTENSIONS else []
    if not path_obj.is_dir():
        return []
    files: List[Path] = []
    for ext in _TEST_EXTENSIONS:
        files.extend(path_obj.rglob(f"*{ext}"))
    return sorted(f for f in files if f.is_file() and f.name.lower() not in CONFIG_FILE_PATTERNS)


def _validate_file(file_path: Path) -> List[Dict[str, Any]]:
    """Return list of error dicts (empty list = valid)."""
    try:
        if file_path.suffix.lower() == ".toml":
            if _toml_loader is None:
                return [
                    {
                        "field": None,
                        "message": (
                            "TOML support requires Python 3.11+ stdlib tomllib "
                            "or the 'tomli' package. Install with: pip install tomli"
                        ),
                        "type": "missing_dependency",
                    }
                ]
            with open(file_path, "rb") as f:
                data = _toml_loader.load(f)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        return [{"field": None, "message": f"YAML parse error: {exc}", "type": "parse"}]
    # TOMLDecodeError (both stdlib tomllib and the tomli backport) subclasses
    # ValueError; UnicodeDecodeError likewise — so this pair covers parse and
    # encoding failures without swallowing programming bugs.
    except (OSError, ValueError) as exc:
        return [{"field": None, "message": f"Parse error: {exc}", "type": "parse"}]

    if not isinstance(data, dict):
        return [
            {
                "field": None,
                "message": f"Top-level must be a mapping, got {type(data).__name__}",
                "type": "structure",
            }
        ]

    try:
        TestCase(**data)
        return []
    except ValidationError as exc:
        return [
            {
                "field": ".".join(str(loc) for loc in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]
    except (KeyError, TypeError, ValueError) as exc:
        # TestCase pre-validators can raise these before Pydantic returns
        # a ValidationError (e.g. malformed multi-turn lacking `query`).
        # Convert to a validation error so the lint continues across files
        # rather than aborting the whole run.
        return [
            {
                "field": None,
                "message": f"Schema construction error: {exc}",
                "type": "schema_error",
            }
        ]


def _render_human_output(results: List[Dict[str, Any]], elapsed_ms: float) -> None:
    valid_count = sum(1 for r in results if r["valid"])
    error_count = len(results) - valid_count

    for r in results:
        if r["valid"]:
            console.print(f"[green]OK[/green]   {r['file']}")
        else:
            console.print(f"[red]FAIL[/red] {r['file']}")
            for err in r["errors"]:
                if err.get("field"):
                    console.print(f"       [dim]{err['field']}:[/dim] {err['message']}")
                else:
                    console.print(f"       {err['message']}")

    console.print()
    if error_count == 0:
        console.print(
            f"[green]{valid_count} file(s) valid[/green]  [dim]({elapsed_ms:.1f}ms)[/dim]"
        )
    else:
        console.print(
            f"[red]{error_count} file(s) failed validation[/red], "
            f"{valid_count} ok  [dim]({elapsed_ms:.1f}ms)[/dim]"
        )
