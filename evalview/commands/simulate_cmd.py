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
from evalview.core.cassette import (
    DEFAULT_CASSETTE_DIR,
    cassette_path_for,
    load_cassette,
    save_cassette,
)
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


def _capability_flag(value: bool) -> str:
    return "✓" if value else "✗"


def _format_human(
    test_name: str,
    seed: int,
    variants: int,
    mocks_applied: List[dict],
    variant_outcomes: List[dict],
    branches: List[dict],
    adapter_capability: Optional[dict] = None,
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

    if adapter_capability:
        cap_str = (
            f"tools={_capability_flag(adapter_capability.get('tools'))} "
            f"responses={_capability_flag(adapter_capability.get('responses'))} "
            f"http={_capability_flag(adapter_capability.get('http'))}"
        )
        if not any(adapter_capability.values()):
            cap_str += "  ⚠ adapter has no interception seam — run is hitting live services"
        lines.append(f"  Adapter capability: {cap_str}")

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
    *,
    record: bool = False,
    replay: bool = False,
    allow_live: bool = False,
    cassette_dir: Path = DEFAULT_CASSETTE_DIR,
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

    cassette_path = cassette_path_for(tc.name, cassette_dir)
    replay_cassette = None
    if replay:
        if not cassette_path.exists():
            raise click.ClickException(
                f"--replay set but no cassette found at {cassette_path}. "
                f"Run with --record first."
            )
        replay_cassette = load_cassette(cassette_path)

    sim = Simulator(adapter, spec)
    if variants > 1:
        traces, result = await sim.run_variants(
            tc,
            variants=variants,
            replay_cassette=replay_cassette,
            record=record,
            allow_live=allow_live,
        )
    else:
        _, result = await sim.run(
            tc,
            replay_cassette=replay_cassette,
            record=record,
            allow_live=allow_live,
        )
        traces = []

    cassette_info: Optional[dict] = None
    if record and result.recorded_cassette is not None:
        save_cassette(result.recorded_cassette, cassette_path)
        cassette_info = {
            "path": str(cassette_path),
            "interactions": len(result.recorded_cassette.interactions),
        }
    elif replay_cassette is not None:
        cassette_info = {
            "path": str(cassette_path),
            "interactions": len(replay_cassette.interactions),
            "mode": "replay",
        }

    return {
        "test_name": tc.name,
        "run_type": "simulation",
        "simulation": result.model_dump(),
        "trace_count": len(traces) or 1,
        "cassette": cassette_info,
    }


@click.command("simulate")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--test", "-t", "test_filter", default=None, help="Run only this test by name.")
@click.option("--seed", type=int, default=None, help="Override the seed declared in YAML.")
@click.option("--variants", type=int, default=1, help="Run N deterministic replays (default: 1).")
@click.option("--json", "json_output", is_flag=True, default=False, help="Emit JSON summary for CI.")
@click.option(
    "--record",
    is_flag=True,
    default=False,
    help="Capture every real tool call into .evalview/cassettes/<test>.json for later replay.",
)
@click.option(
    "--replay",
    is_flag=True,
    default=False,
    help="Serve tool calls from the cassette at .evalview/cassettes/<test>.json (hermetic).",
)
@click.option(
    "--cassette-dir",
    type=click.Path(),
    default=str(DEFAULT_CASSETTE_DIR),
    show_default=True,
    help="Override the directory used to find/store cassettes.",
)
@click.option(
    "--allow-live",
    is_flag=True,
    default=False,
    help=(
        "Suppress the warning when the adapter has no interception seam. "
        "By default an uninterceptable adapter logs a warning; with this "
        "flag it logs INFO instead. Hermetic modes (--record, --replay, "
        "or mocks.strict=true) still raise."
    ),
)
@track_command("simulate")
def simulate(
    test_path: str,
    test_filter: Optional[str],
    seed: Optional[int],
    variants: int,
    json_output: bool,
    record: bool,
    replay: bool,
    cassette_dir: str,
    allow_live: bool,
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
    if record and replay:
        raise click.ClickException("--record and --replay are mutually exclusive.")
    cassette_dir_path = Path(cassette_dir)

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
            summary = asyncio.run(_run_one(
                tc, config, variants, seed,
                record=record, replay=replay,
                allow_live=allow_live,
                cassette_dir=cassette_dir_path,
            ))
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
                    adapter_capability=sim.get("adapter_capability"),
                )
            )
            if summary.get("cassette"):
                ci = summary["cassette"]
                verb = "Replayed" if ci.get("mode") == "replay" else "Recorded"
                click.echo(f"  {verb} cassette: {ci['path']} ({ci['interactions']} interactions)")

    if json_output:
        click.echo(json.dumps({"results": summaries}, indent=2))

    if any("error" in s for s in summaries):
        sys.exit(1)


__all__ = ["simulate"]
