"""freshness command — detect production-query coverage gaps in the eval suite.

Where ``evalview autopr`` closes the loop from a *failing* production trace to
a regression test, ``evalview freshness`` closes the loop from *drifted
traffic* to a new capability test — even when nothing has failed yet.

    evalview monitor  ──▶  .evalview/incidents.jsonl  ─┐
                                                       ├──▶  freshness ──▶ tests/coverage/*.yaml
    (optional)  --from prod-queries.jsonl  ────────────┘

Design goals:

- **Pure & local.** No network, no LLM, no embeddings — Jaccard token overlap.
  Matches the deterministic contract of ``regression_synth``.
- **Idempotent.** Skips clusters whose proposed test already exists on disk
  (by ``meta.coverage.slug``).
- **Honest defaults.** Reports the gap; only writes files when ``--propose``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import yaml  # type: ignore[import-untyped]

from evalview.commands.autopr_cmd import load_incidents
from evalview.commands.shared import console
from evalview.core.freshness import (
    DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_MIN_CLUSTER_SIZE,
    FreshnessReport,
    QueryCluster,
    build_freshness_report,
    coverage_slug,
    extract_queries_from_records,
    synthesize_coverage_test,
)
from evalview.telemetry.decorators import track_command


DEFAULT_INCIDENTS_PATH = Path(".evalview/incidents.jsonl")
DEFAULT_COVERAGE_DIR = Path("tests/coverage")
DEFAULT_TESTS_DIR = Path("tests")


def _load_jsonl_queries(path: Path, field_name: str = "query") -> List[str]:
    """Read a JSONL file and pull out ``field_name`` from each record.

    Malformed lines are logged and skipped — the freshness command is meant
    to ingest noisy production logs, and one bad row should not block the
    whole report.
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
                    f"[yellow]  Warning: skipping malformed record on line {i} "
                    f"of {path}: {e}[/yellow]"
                )
    return extract_queries_from_records(records, field_name=field_name)


def _load_suite_queries(tests_dir: Path) -> List[str]:
    """Load every test case's input.query from ``tests_dir``.

    Uses the canonical loader so we get the same answer the rest of EvalView
    would. Multi-turn tests contribute their first-turn query (which is what
    the loader pre-populates into ``input``).
    """
    from evalview.core.loader import TestCaseLoader

    if not tests_dir.exists():
        return []

    loader = TestCaseLoader()
    try:
        cases = loader.load_from_directory(tests_dir)
    except Exception as e:  # pragma: no cover - defensive
        console.print(
            f"[yellow]  Warning: failed to load suite from {tests_dir}: {e}[/yellow]"
        )
        return []

    out: List[str] = []
    for tc in cases:
        if tc.input is not None and isinstance(tc.input.query, str):
            q = tc.input.query.strip()
            if q:
                out.append(q)
    return out


def _existing_slugs(coverage_dir: Path) -> set[str]:
    """Collect coverage slugs already present in ``coverage_dir``.

    Mirrors ``autopr_cmd._existing_slugs`` — uses ``meta.coverage.slug`` as
    the idempotency key.
    """
    slugs: set[str] = set()
    if not coverage_dir.exists():
        return slugs
    for yaml_file in coverage_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        meta = (data or {}).get("meta") or {}
        coverage = meta.get("coverage") or {}
        slug = coverage.get("slug")
        if slug:
            slugs.add(slug)
    return slugs


def _render_report(report: FreshnessReport, coverage_dir: Path) -> None:
    """Pretty-print the freshness report to the console."""
    coverage = report.coverage
    console.print()
    console.print("[bold]EvalView Freshness Report[/bold]")
    console.print(
        f"  Production queries: [cyan]{report.prod_size}[/cyan]  "
        f"|  Suite queries: [cyan]{report.suite_size}[/cyan]"
    )

    if coverage.total == 0:
        console.print("[dim]  No production queries to score. "
                      "Run `evalview monitor --incidents` to populate "
                      "`.evalview/incidents.jsonl`.[/dim]")
        return

    pct = coverage.coverage_pct
    color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
    console.print(
        f"  Coverage: [{color}]{pct:.1f}%[/{color}]  "
        f"([green]{coverage.covered}[/green] covered, "
        f"[yellow]{coverage.uncovered}[/yellow] uncovered, "
        f"threshold {coverage.threshold:.2f})"
    )

    if not report.clusters:
        if coverage.uncovered:
            console.print(
                f"[dim]  {coverage.uncovered} uncovered queries, but none clustered "
                f"above size {report.min_cluster_size} — likely one-offs.[/dim]"
            )
        else:
            console.print("[green]  No coverage gaps detected.[/green]")
        return

    console.print(
        f"\n  [bold]{len(report.clusters)} coverage gap"
        f"{'s' if len(report.clusters) != 1 else ''} detected:[/bold]"
    )
    for i, cluster in enumerate(report.clusters, start=1):
        slug = coverage_slug(cluster)
        pct_of_uncovered = (
            100.0 * cluster.size / coverage.uncovered if coverage.uncovered else 0.0
        )
        console.print(
            f"\n  [bold cyan]Gap {i}[/bold cyan]  "
            f"({cluster.size} queries, {pct_of_uncovered:.0f}% of uncovered, "
            f"avg similarity {cluster.avg_intra_similarity:.2f})"
        )
        console.print(f"    Representative: [italic]\"{cluster.representative}\"[/italic]")
        examples = cluster.examples()
        # Skip the representative when listing extra examples.
        extras = [e for e in examples if e != cluster.representative][:3]
        if extras:
            console.print("    Examples:")
            for e in extras:
                console.print(f"      • [dim]{e}[/dim]")
        console.print(
            f"    [dim]→ Stub would be: {coverage_dir / (slug + '.yaml')}[/dim]"
        )


def _write_proposed_stubs(
    clusters: List[QueryCluster],
    coverage_dir: Path,
    min_score: float,
    dry_run: bool,
) -> Tuple[List[Tuple[Path, Dict[str, Any]]], int]:
    """Write one YAML stub per cluster. Returns (written, skipped_existing)."""
    written: List[Tuple[Path, Dict[str, Any]]] = []
    skipped = 0

    existing = _existing_slugs(coverage_dir)
    for cluster in clusters:
        slug = coverage_slug(cluster)
        if slug in existing:
            skipped += 1
            continue
        test = synthesize_coverage_test(cluster, min_score=min_score)
        target = coverage_dir / f"{slug}.yaml"
        if dry_run:
            console.print(f"[dim]  Would write {target}[/dim]")
            written.append((target, test))
            existing.add(slug)
            continue
        coverage_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(test, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append((target, test))
        existing.add(slug)
    return written, skipped


@click.command("freshness")
@click.option(
    "--from",
    "incidents_path",
    default=str(DEFAULT_INCIDENTS_PATH),
    type=click.Path(),
    help="Path to incidents JSONL (default: .evalview/incidents.jsonl).",
)
@click.option(
    "--from-log",
    "extra_log_path",
    default=None,
    type=click.Path(),
    help="Optional additional JSONL of production records (any with a 'query' field).",
)
@click.option(
    "--query-field",
    default="query",
    help="JSON field name to read query strings from (default: query).",
)
@click.option(
    "--tests-dir",
    default=str(DEFAULT_TESTS_DIR),
    type=click.Path(),
    help="Directory of existing test cases to score against (default: tests).",
)
@click.option(
    "--coverage-dir",
    default=str(DEFAULT_COVERAGE_DIR),
    type=click.Path(),
    help="Where --propose writes coverage-gap stubs (default: tests/coverage).",
)
@click.option(
    "--threshold",
    type=float,
    default=DEFAULT_COVERAGE_THRESHOLD,
    show_default=True,
    help="Min Jaccard similarity for a production query to count as covered.",
)
@click.option(
    "--cluster-threshold",
    type=float,
    default=DEFAULT_CLUSTER_THRESHOLD,
    show_default=True,
    help="Min Jaccard similarity for two uncovered queries to cluster together.",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=DEFAULT_MIN_CLUSTER_SIZE,
    show_default=True,
    help="Drop clusters smaller than this (likely one-offs, not patterns).",
)
@click.option(
    "--min-score",
    type=float,
    default=70.0,
    show_default=True,
    help="min_score threshold applied to synthesized coverage stubs.",
)
@click.option(
    "--propose",
    is_flag=True,
    help="Write one YAML stub per coverage gap to --coverage-dir.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="With --propose, print what would be written without touching disk.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit a machine-readable JSON report to stdout instead of rich text.",
)
@click.option(
    "--require-gaps",
    is_flag=True,
    help="Exit 1 when at least one cluster ≥ --min-cluster-size is detected "
         "(useful in CI to surface drift).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Only consider the N most recent production records.",
)
@track_command("freshness")
def freshness(
    incidents_path: str,
    extra_log_path: Optional[str],
    query_field: str,
    tests_dir: str,
    coverage_dir: str,
    threshold: float,
    cluster_threshold: float,
    min_cluster_size: int,
    min_score: float,
    propose: bool,
    dry_run: bool,
    json_output: bool,
    require_gaps: bool,
    limit: Optional[int],
) -> None:
    """Score production-query coverage of the eval suite and propose gap tests.

    EvalView already turns *failing* incidents into regression tests via
    `evalview autopr`. ``freshness`` does the complementary job: it surfaces
    *unfailing* production queries that the suite doesn't represent — the
    silent drift that lets evals look healthy while real traffic moves on.

    \b
    Typical usage:
        evalview freshness                            # see coverage + gaps
        evalview freshness --propose                  # also write stubs
        evalview freshness --propose --dry-run        # preview stubs
        evalview freshness --json > freshness.json    # machine-readable
        evalview freshness --require-gaps             # CI gate

    \b
    Inputs:
        --from           JSONL whose records carry a 'query' field. Defaults
                         to .evalview/incidents.jsonl (written by
                         `evalview monitor --incidents`).
        --from-log       Additional JSONL (e.g. your own production traffic
                         log). Concatenated with --from.

    The scoring is pure Jaccard token overlap — no embeddings, no network,
    no LLM. Deterministic across runs, safe to commit the output of.
    """
    incidents_file = Path(incidents_path)
    extra_log = Path(extra_log_path) if extra_log_path else None
    tests_path = Path(tests_dir)
    coverage_path = Path(coverage_dir)

    # Pull queries from incidents.jsonl + optional extra log.
    incident_records = load_incidents(incidents_file)
    prod_queries = extract_queries_from_records(
        incident_records, field_name=query_field
    )
    if extra_log is not None:
        prod_queries.extend(_load_jsonl_queries(extra_log, field_name=query_field))

    if limit is not None and limit >= 0:
        prod_queries = prod_queries[-limit:]

    suite_queries = _load_suite_queries(tests_path)

    report = build_freshness_report(
        prod_queries,
        suite_queries,
        coverage_threshold=threshold,
        cluster_threshold=cluster_threshold,
        min_cluster_size=min_cluster_size,
    )

    if json_output:
        click.echo(json.dumps(report.to_dict(), indent=2, sort_keys=False))
    else:
        _render_report(report, coverage_path)

    written: List[Tuple[Path, Dict[str, Any]]] = []
    skipped = 0
    if propose and report.clusters:
        written, skipped = _write_proposed_stubs(
            list(report.clusters),
            coverage_path,
            min_score=min_score,
            dry_run=dry_run,
        )
        if not json_output:
            if written:
                action = "Would write" if dry_run else "Wrote"
                console.print(
                    f"\n[green]  {action} {len(written)} coverage stub"
                    f"{'s' if len(written) != 1 else ''} to "
                    f"[cyan]{coverage_path}[/cyan][/green]"
                )
                if not dry_run:
                    console.print(
                        "[dim]  Next: review the stubs, then run "
                        "`evalview snapshot` to capture current behavior as a baseline.[/dim]"
                    )
            if skipped:
                console.print(
                    f"[dim]  Skipped {skipped} cluster(s) — coverage stubs already exist.[/dim]"
                )

    elif propose and not report.clusters and not json_output:
        console.print(
            "[dim]  Nothing to propose — no qualifying coverage gaps found.[/dim]"
        )

    if require_gaps and report.clusters:
        sys.exit(1)
