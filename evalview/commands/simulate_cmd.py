"""``evalview simulate`` — run a test against mocked tools.

Closes the "no pre-deployment simulation" gap from the April 2026
agent-eval complaints. Users declare mocks in the test YAML under a
``mocks:`` section and run::

    evalview simulate tests/my-test.yaml
    evalview simulate tests/ --test my-test --variants 5 --seed 7

The engine lives in :mod:`evalview.core.simulation`; this module is
only the CLI glue — argument parsing, adapter wiring, and output
rendering (human-readable summary or JSON for CI).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click

from evalview.core.adapter_factory import create_adapter_from_config
from evalview.core.config import EvalViewConfig
from evalview.core.loader import TestCaseLoader
from evalview.core.simulation import Simulator
from evalview.core.types import MockSpec, TestCase
from evalview.telemetry.decorators import track_command

logger = logging.getLogger(__name__)


def _load_config() -> Optional[EvalViewConfig]:
    """Best-effort config load — returns None when no config file exists."""
    from evalview.commands.shared import _load_config_if_exists

    return _load_config_if_exists()


def _resolve_test_cases(
    test_path: str, test_filter: Optional[str]
) -> List[TestCase]:
    """Load every test below ``test_path`` (file or dir) and optionally filter."""
    loader = TestCaseLoader()
    path = Path(test_path)
    if path.is_file():
        cases = [loader.load_from_file(path)]
    else:
        cases = loader.load_from_directory(path)

    if test_filter:
        cases = [c for c in cases if c.name == test_filter]
    return cases


def _format_human(
    test_name: str,
    seed: int,
    variants: int,
    mocks_applied: List[dict],
    variant_outcomes: List[dict],
    branches: List[dict],
) -> str:
    """Readable terminal summary. JSON path skips this entirely."""
    lines: List[str] = []
    lines.append(f"▶ {test_name}  (seed={seed}, variants={variants})")
    if mocks_applied:
        lines.append("  Mocks applied:")
        for m in mocks_applied:
            lines.append(f"    · {m['kind']}:{m['matcher']} ×{m['count']}")
    else:
        lines.append("  Mocks applied: none matched")

    lines.append("  Variants:")
    if variant_outcomes:
        for v in variant_outcomes:
            cost = f"${v['total_cost']:.4f}"
            lat = f"{v['total_latency_ms']:.0f}ms"
            lines.append(
                f"    · #{v['variant_index']} branch={v['branch_id']} "
                f"{cost} {lat}"
            )
    else:
        lines.append("    · (single run)")

    if branches:
        lines.append("  Branches:")
        for b in branches:
            path = " → ".join(b["decision_path"]) or "(no tool calls)"
            lines.append(f"    · {b['branch_id']}: {path}")

    return "\n".join(lines)


async def _run_one(
    tc: TestCase,
    config: Optional[EvalViewConfig],
    variants: int,
    seed_override: Optional[int],
) -> dict:
    """Run a single test case and return a serializable summary dict."""
    if tc.mocks is None:
        spec = MockSpec()
    else:
        spec = tc.mocks

    if seed_override is not None:
        spec = spec.model_copy(update={"seed": seed_override})

    # Build adapter. Per-test overrides win over config defaults.
    adapter_type = tc.adapter or (config.adapter if config else None)
    endpoint = tc.endpoint or (config.endpoint if config else "")
    if not adapter_type:
        raise click.ClickException(
            f"Test '{tc.name}' has no adapter. Set 'adapter:' on the test or in "
            ".evalview/config.yaml."
        )
    run_config = EvalViewConfig.model_validate(
        {**(config.model_dump() if config else {}), "adapter": adapter_type, "endpoint": endpoint}
    )
    adapter = create_adapter_from_config(run_config)

    sim = Simulator(adapter, spec)
    if variants > 1:
        traces, result = await sim.run_variants(tc, variants=variants)
    else:
        _, result = await sim.run(tc)
        traces = []

    return {
        "test_name": tc.name,
        "run_type": "simulation",
        "simulation": result.model_dump(),
        "trace_count": len(traces) or 1,
    }


@click.command("simulate")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--test", "-t", "test_filter", default=None, help="Run only this test by name.")
@click.option("--seed", type=int, default=None, help="Override the seed declared in YAML.")
@click.option("--variants", type=int, default=1, help="Run N deterministic replays (default: 1).")
@click.option("--json", "json_output", is_flag=True, default=False, help="Emit JSON summary for CI.")
@track_command("simulate")
def simulate(
    test_path: str,
    test_filter: Optional[str],
    seed: Optional[int],
    variants: int,
    json_output: bool,
) -> None:
    """Run tests hermetically against declared mocks.

    Mocks are declared under the top-level ``mocks:`` key in each test
    YAML. Tool calls that match ``tool_mocks`` return the mock value
    instead of hitting ``tool_executor``; unmatched calls fall through
    unless ``strict: true`` is set.

    Examples:
        evalview simulate tests/
        evalview simulate tests/my-test.yaml --variants 10
        evalview simulate tests/ --test flight-search --seed 42 --json
    """
    if variants < 1:
        raise click.ClickException("--variants must be >= 1")

    cases = _resolve_test_cases(test_path, test_filter)
    if not cases:
        raise click.ClickException(
            f"No tests found at {test_path}"
            + (f" matching '{test_filter}'" if test_filter else "")
        )

    config = _load_config()

    summaries: List[dict] = []
    for tc in cases:
        try:
            summary = asyncio.run(_run_one(tc, config, variants, seed))
        except Exception as exc:
            summaries.append({
                "test_name": tc.name,
                "run_type": "simulation",
                "error": str(exc),
            })
            if not json_output:
                click.echo(f"✗ {tc.name}: {exc}", err=True)
            continue
        summaries.append(summary)
        if not json_output:
            sim = summary["simulation"]
            click.echo(
                _format_human(
                    summary["test_name"],
                    sim["seed"],
                    variants,
                    sim["mocks_applied"],
                    sim["variant_outcomes"],
                    sim["branches_explored"],
                )
            )

    if json_output:
        click.echo(json.dumps({"results": summaries}, indent=2))

    if any("error" in s for s in summaries):
        sys.exit(1)


__all__ = ["simulate"]
