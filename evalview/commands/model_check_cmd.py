"""`evalview model-check` — closed-model behavioral drift detection.

Runs a small, fixed canary suite directly against a provider (v1: Anthropic
only). Each prompt is scored by a pure structural scorer
(``tool_choice`` / ``json_schema`` / ``refusal`` / ``exact_match``), so
there is NO LLM judge dependency in v1 and therefore no calibration problem.

Each invocation produces a snapshot. Drift comparisons use a two-anchor
model:

- **reference**  — the first-ever (or user-pinned) snapshot; never auto-
                   updated, so gradual drift is detectable.
- **latest prior** — the most recent snapshot before this run.

Provider fingerprint signal strength is honestly labeled. Anthropic does
not currently expose a per-response fingerprint, so the signal is
"behavior-only" (weak). OpenAI's ``system_fingerprint`` will be wired in
v1.1 and labeled "strong". See ``docs/MODEL_CHECK.md``.

Architecture notes:
  - Provider calls go through ``core.model_provider_runner``, NOT through
    the agent adapter abstraction. Canary runs do not use tool loops or
    goldens; the agent adapter shape is the wrong fit and would couple
    drift signal stability to changes in the agent test path.
  - Sampling is pinned at temperature=0.0, top_p=1.0. Snapshots refuse
    to compare across different sampling configs.
  - The command returns exit 0 on no drift, 1 on any drift detected, 2
    on usage / configuration errors.
  - Classification and rendering are in separate modules
    (``core.drift_classifier`` and ``commands.model_check_render``) so
    they can be reused from CI integrations without importing Click.
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import click

from evalview.benchmarks.canary import PUBLIC_SUITE_PATH
from evalview.commands.model_check_render import (
    build_json_payload,
    render_comparison,
    render_header,
    render_next_steps,
)
from evalview.commands.shared import console
from evalview.core.budget import BudgetExhausted
from evalview.core.canary_suite import (
    CanaryPrompt,
    CanarySuite,
    CanarySuiteError,
    load_canary_suite,
)
from evalview.core.drift_classifier import (
    DEFAULT_MEDIUM_FLIP_COUNT,
    DEFAULT_MEDIUM_FLIP_RATIO,
    DEFAULT_WEAK_DRIFT_DELTA,
    classify,
)
from evalview.core.drift_kind import DriftKind
from evalview.core.model_check_scoring import ScoreResult, score_prompt
from evalview.core.model_provider_runner import (
    CompletionResult,
    ProviderError,
    SUPPORTED_PROVIDERS,
    detect_provider,
    run_completion,
)
from evalview.core.model_snapshots import (
    ModelCheckPromptResult,
    ModelSnapshot,
    ModelSnapshotMetadata,
    ModelSnapshotStore,
    SnapshotSuiteMismatchError,
)
from evalview.core.pricing import calculate_cost, get_model_pricing_info
from evalview.telemetry.decorators import track_command

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


# Cost ESTIMATION (used for --dry-run and pre-flight budget check) assumes
# a typical canary prompt uses roughly this many tokens on each side of the
# API boundary. Deliberately generous so the estimate over-quotes rather
# than surprising the user. ACTUAL cost recorded after the run uses real
# token counts from each API response.
_EST_INPUT_TOKENS_PER_CALL = 400
_EST_OUTPUT_TOKENS_PER_CALL = 300

# Sampling is pinned for v1. These constants are surfaced in snapshot
# metadata and used for the suite-compatibility check, so older snapshots
# will refuse to compare if these values change in the future.
_PINNED_TEMPERATURE = 0.0
_PINNED_TOP_P = 1.0
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TIMEOUT_SECONDS = 60.0

# Default concurrency limit. Each prompt is independent so they can run
# concurrently, but we cap parallelism to avoid overwhelming the provider
# with simultaneous requests (rate limits, connection pools, etc.).
_DEFAULT_CONCURRENCY = 4


# --------------------------------------------------------------------------- #
# Provider resolution
# --------------------------------------------------------------------------- #


def _resolve_provider(model_id: str, explicit: Optional[str]) -> str:
    """Resolve the provider from --provider or by inference from --model.

    Fails loudly on unknown providers; silent fallback would risk routing
    one model id to the wrong API and producing meaningless drift output.
    """
    if explicit:
        provider = explicit.strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise click.UsageError(
                f"Provider '{explicit}' is not supported in v1. "
                f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        return provider

    inferred = detect_provider(model_id)
    if inferred is None:
        raise click.UsageError(
            f"Could not infer provider from model id {model_id!r}. "
            f"Pass --provider explicitly. "
            f"Supported in v1: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return inferred


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


@dataclass
class _SuiteRunOutcome:
    """Aggregated run output. All cost numbers are derived from real token usage."""

    results: List[ModelCheckPromptResult]
    total_cost_usd: float
    fingerprint: Optional[str]
    fingerprint_confidence: str


async def _run_one_prompt(
    *,
    prompt: CanaryPrompt,
    provider: str,
    model: str,
    runs_per_prompt: int,
    max_tokens: int,
    timeout: float,
) -> Tuple[ModelCheckPromptResult, float, Optional[str], str]:
    """Execute one prompt N times, score each run, return aggregate + cost.

    Each run uses the pinned sampling configuration. Per-run failures
    propagate as ProviderError so the caller can decide whether to abort
    the whole suite (default) or skip and continue.

    Returns:
        (result, total_cost_usd, fingerprint, fingerprint_confidence)
    """
    per_run: List[bool] = []
    latencies: List[float] = []
    cost = 0.0
    last_fp: Optional[str] = None
    last_fp_conf: str = "none"

    for _ in range(runs_per_prompt):
        completion: CompletionResult = await run_completion(
            provider,
            model,
            prompt.prompt,
            temperature=_PINNED_TEMPERATURE,
            top_p=_PINNED_TOP_P,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        try:
            score: ScoreResult = score_prompt(
                prompt.scorer,
                response=completion.text,
                expected=prompt.expected,
            )
        except ValueError as exc:
            # Suite YAML misconfiguration — surface loudly, never silently fail.
            raise click.UsageError(f"Prompt '{prompt.id}': {exc}") from exc

        per_run.append(score.passed)
        latencies.append(completion.latency_ms)
        cost += calculate_cost(
            model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        last_fp = completion.fingerprint
        last_fp_conf = completion.fingerprint_confidence

    pass_rate = sum(1 for p in per_run if p) / len(per_run)
    result = ModelCheckPromptResult(
        prompt_id=prompt.id,
        category=prompt.category,
        pass_rate=pass_rate,
        n_runs=runs_per_prompt,
        per_run_passed=per_run,
        latency_ms_mean=statistics.fmean(latencies) if latencies else None,
        latency_ms_stdev=(
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        ),
    )
    return result, cost, last_fp, last_fp_conf


async def _run_suite(
    *,
    suite: CanarySuite,
    provider: str,
    model: str,
    runs_per_prompt: int,
    max_tokens: int,
    timeout: float,
    budget_usd: float,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> _SuiteRunOutcome:
    """Run every prompt N times, with bounded concurrency and budget checks.

    Prompts are dispatched concurrently up to ``concurrency`` (controlled
    by an ``asyncio.Semaphore``). After each prompt completes and before
    the next one starts, the accumulated cost is checked against the budget
    *inside* the semaphore. This guarantees that the check always sees
    the latest cost and that at most ``concurrency`` prompts can be
    in-flight past any given check.

    If the budget is exceeded, ``BudgetExhausted`` is raised. Because
    ``asyncio.gather`` propagates the first exception and cancels pending
    tasks, prompts still waiting for the semaphore are never started. A
    prompt that has started is never cancelled mid-run — partial results
    would corrupt the snapshot.
    """
    results: List[Tuple[ModelCheckPromptResult, float, Optional[str], str]] = []
    total_cost = 0.0
    completed_count = 0
    cost_lock = asyncio.Lock()

    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_budget(prompt: CanaryPrompt):
        nonlocal total_cost, completed_count

        async with semaphore:
            # Budget gate: checked INSIDE the semaphore so it always sees
            # the accumulated cost from all previously completed prompts.
            # If checked outside, all tasks would read total_cost=0
            # simultaneously and bypass the check.
            async with cost_lock:
                if total_cost >= budget_usd:
                    raise BudgetExhausted(
                        spent=total_cost,
                        limit=budget_usd,
                        completed=completed_count,
                        total=len(suite.prompts),
                    )

            result, cost, fp, fp_conf = await _run_one_prompt(
                prompt=prompt,
                provider=provider,
                model=model,
                runs_per_prompt=runs_per_prompt,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            async with cost_lock:
                total_cost += cost
                completed_count += 1

            return result, cost, fp, fp_conf

    # asyncio.gather preserves input order so results stay deterministic.
    # return_exceptions=False (default) means the first BudgetExhausted
    # propagates immediately and cancels tasks still waiting on the semaphore.
    raw_results = await asyncio.gather(*[_run_with_budget(p) for p in suite.prompts])

    all_results: List[ModelCheckPromptResult] = []
    final_cost = 0.0
    fingerprint: Optional[str] = None
    fp_confidence: str = "none"
    for result, cost, fp, fp_conf in raw_results:
        all_results.append(result)
        final_cost += cost
        fingerprint = fp
        fp_confidence = fp_conf

    return _SuiteRunOutcome(
        results=all_results,
        total_cost_usd=final_cost,
        fingerprint=fingerprint,
        fingerprint_confidence=fp_confidence,
    )


# --------------------------------------------------------------------------- #
# Cost estimation
# --------------------------------------------------------------------------- #


def _estimate_cost_usd(model_id: str, n_calls: int) -> float:
    """Rough cost estimate based on pricing.py tables.

    Deliberately conservative (rounds up) so --dry-run never undersells.
    """
    pricing = get_model_pricing_info(model_id)
    per_call = (
        _EST_INPUT_TOKENS_PER_CALL * pricing["input_price_per_token"]
        + _EST_OUTPUT_TOKENS_PER_CALL * pricing["output_price_per_token"]
    )
    return round(per_call * n_calls, 4)


# --------------------------------------------------------------------------- #
# Command
# --------------------------------------------------------------------------- #


# Exit codes used by this command. Documented so CI integrations can rely
# on them.
EXIT_OK = 0
EXIT_DRIFT_DETECTED = 1
EXIT_USAGE_ERROR = 2


@click.command("model-check")
@click.option("--model", required=True, help="Model id (e.g. claude-opus-4-5-20251101).")
@click.option(
    "--provider",
    default=None,
    help="Provider name. Auto-detected from the model id when omitted.",
)
@click.option(
    "--suite",
    "suite_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to a custom canary suite YAML. Defaults to the bundled public canary.",
)
@click.option("--runs", "runs_per_prompt", default=3, show_default=True, type=int)
@click.option("--budget", default=2.00, show_default=True, type=float, help="Maximum USD spend.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print a cost estimate and exit without calling the provider.",
)
@click.option(
    "--pin",
    is_flag=True,
    help="Pin the resulting snapshot as the new reference for this model.",
)
@click.option(
    "--reset-reference",
    is_flag=True,
    help="Delete the existing reference before the run so this snapshot becomes the new baseline.",
)
@click.option("--out", "out_path", default=None, type=click.Path(path_type=Path))
@click.option(
    "--no-save",
    is_flag=True,
    help="Do not persist the snapshot to disk (useful for ad-hoc testing).",
)
@click.option("--json", "json_output", is_flag=True, help="Emit a JSON payload instead of human output.")
@click.option(
    "--keep",
    "keep_last",
    default=50,
    show_default=True,
    type=int,
    help="Number of timestamped snapshots to retain per model during pruning.",
)
@click.option(
    "--concurrency",
    default=_DEFAULT_CONCURRENCY,
    show_default=True,
    type=int,
    help="Max concurrent prompt calls to the provider.",
)
@click.option(
    "--drift-threshold",
    default=None,
    type=float,
    help=(
        "Override the minimum per-prompt pass-rate delta to count as drift "
        f"(default: {DEFAULT_WEAK_DRIFT_DELTA})."
    ),
)
@click.option(
    "--medium-flip-count",
    default=None,
    type=int,
    help=(
        "Override the minimum prompt flips for MEDIUM confidence on small suites "
        f"(default: {DEFAULT_MEDIUM_FLIP_COUNT}). For suites >20 prompts, "
        f"the threshold scales automatically to {DEFAULT_MEDIUM_FLIP_RATIO:.0%} of suite size."
    ),
)
@track_command("model_check")
def model_check(
    model: str,
    provider: Optional[str],
    suite_path: Optional[Path],
    runs_per_prompt: int,
    budget: float,
    dry_run: bool,
    pin: bool,
    reset_reference: bool,
    out_path: Optional[Path],
    no_save: bool,
    json_output: bool,
    keep_last: int,
    concurrency: int,
    drift_threshold: Optional[float],
    medium_flip_count: Optional[int],
) -> None:
    """Detect behavioral drift in a closed model against a fixed canary suite.

    \b
    Runs a small set of structural prompts (tool selection, JSON schema,
    refusal behavior, exact match) against the model with pinned
    temperature=0, then compares results to two anchors:

    \b
      • reference snapshot — first run ever, or a user-pinned one. Never
                             auto-updates, so gradual drift is detectable.
      • previous snapshot  — most recent prior run. Day-over-day delta.

    Drift is classified as NONE / WEAK / MEDIUM / STRONG depending on how
    many prompts flipped pass↔fail and whether the provider exposes a
    fingerprint change. v1 supports Anthropic; OpenAI ships in v1.1.

    \b
    Classification thresholds scale automatically for large suites (>20
    prompts). Override with --drift-threshold and --medium-flip-count.

    No LLM judge is used in v1, so there is no calibration requirement.

    \b
    Examples:
      evalview model-check --model claude-opus-4-5-20251101 --dry-run
      evalview model-check --model claude-opus-4-5-20251101
      evalview model-check --model claude-opus-4-5-20251101 --pin
      evalview model-check --model claude-opus-4-5-20251101 --json
      evalview model-check --model claude-opus-4-5-20251101 --keep 100

    See docs/MODEL_CHECK.md for the per-provider signal strength table.

    \b
    IMPORTANT — Anthropic limitation (v1):
      Anthropic does not expose a per-response fingerprint. Drift
      detection is behavior-only (weak signal). STRONG classifications
      are not possible until OpenAI support ships in v1.1.
    """
    # --- Load suite -------------------------------------------------------
    suite_file = suite_path or PUBLIC_SUITE_PATH
    try:
        suite = load_canary_suite(suite_file)
    except CanarySuiteError as exc:
        console.print(f"[red]Failed to load canary suite:[/red] {exc}")
        sys.exit(EXIT_USAGE_ERROR)

    # --- Resolve provider -------------------------------------------------
    try:
        provider_resolved = _resolve_provider(model, provider)
    except click.UsageError as exc:
        console.print(f"[red]{exc.message}[/red]")
        sys.exit(EXIT_USAGE_ERROR)

    n_calls = len(suite.prompts) * runs_per_prompt
    estimated_cost = _estimate_cost_usd(model, n_calls)

    # --- Dry-run path: print estimate and exit -----------------------------
    if dry_run:
        console.print()
        console.print("[bold]Would run:[/bold] " + model)
        console.print(
            f"  Suite:           {suite.suite_name} {suite.version} "
            f"({len(suite.prompts)} prompts × {runs_per_prompt} runs = {n_calls} calls)"
        )
        console.print(f"  Provider:        {provider_resolved}")
        console.print(f"  Sampling:        temperature={_PINNED_TEMPERATURE} top_p={_PINNED_TOP_P}")
        console.print(f"  Estimated cost:  ${estimated_cost:.4f}")
        console.print(f"  Budget cap:      ${budget:.2f}")
        console.print(f"  Concurrency:     {concurrency}")
        console.print()
        console.print("[dim]Re-run without --dry-run to execute.[/dim]")
        console.print()
        return

    if estimated_cost > budget:
        console.print(
            f"[red]Estimated cost ${estimated_cost:.4f} exceeds --budget ${budget:.2f}.[/red] "
            f"Run with --dry-run to confirm, or raise --budget."
        )
        sys.exit(EXIT_USAGE_ERROR)

    # --- Load store + handle reference management ------------------------
    store = ModelSnapshotStore()
    if reset_reference:
        if store.reset_reference(model):
            console.print(f"[yellow]Reference for {model} was deleted.[/yellow]")

    reference_before = store.load_reference(model)

    # JSON mode must produce clean machine-readable output. Human progress
    # messages go to the rich console (stderr-friendly).
    if not json_output:
        console.print(
            f"[dim]Running {len(suite.prompts)} prompts × {runs_per_prompt} runs "
            f"against {model} (concurrency={concurrency})…[/dim]"
        )

    # --- Run the suite ---------------------------------------------------
    try:
        outcome = asyncio.run(
            _run_suite(
                suite=suite,
                provider=provider_resolved,
                model=model,
                runs_per_prompt=runs_per_prompt,
                max_tokens=_DEFAULT_MAX_TOKENS,
                timeout=_DEFAULT_TIMEOUT_SECONDS,
                budget_usd=budget,
                concurrency=concurrency,
            )
        )
    except ProviderError as exc:
        console.print(f"[red]Provider error:[/red] {exc}")
        sys.exit(EXIT_USAGE_ERROR)
    except BudgetExhausted as exc:
        console.print(
            f"[yellow]Budget exhausted after {exc.completed}/{exc.total} prompts. "
            f"Spent ${exc.spent:.4f} of ${exc.limit:.2f} budget.[/yellow]"
        )
        sys.exit(EXIT_USAGE_ERROR)
    except click.UsageError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        console.print(f"[red]model-check failed during execution:[/red] {exc}")
        sys.exit(EXIT_USAGE_ERROR)

    # --- Build snapshot ---------------------------------------------------
    snapshot = ModelSnapshot(
        metadata=ModelSnapshotMetadata(
            model_id=model,
            provider=provider_resolved,
            snapshot_at=datetime.now(timezone.utc),
            suite_name=suite.suite_name,
            suite_version=suite.version,
            suite_hash=suite.suite_hash,
            temperature=_PINNED_TEMPERATURE,
            top_p=_PINNED_TOP_P,
            runs_per_prompt=runs_per_prompt,
            provider_fingerprint=outcome.fingerprint,
            fingerprint_confidence=outcome.fingerprint_confidence,
            cost_total_usd=outcome.total_cost_usd,
            evalview_version=_get_evalview_version(),
        ),
        results=outcome.results,
    )

    # --- Persist ---------------------------------------------------------
    saved_path: Optional[Path] = None
    if not no_save:
        try:
            saved_path = store.save_snapshot(snapshot)
        except Exception as exc:
            console.print(f"[red]Failed to save snapshot:[/red] {exc}")
            sys.exit(EXIT_USAGE_ERROR)

        if pin:
            store.pin_reference(model, snapshot)
            console.print(
                f"[green]Pinned current run as the new reference for {model}.[/green]"
            )

        # Prune old snapshots, respecting --keep.
        pruned = store.prune(model, keep_last=keep_last)
        if pruned > 0:
            logger.info("Pruned %d old snapshots for %s (kept %d)", pruned, model, keep_last)

    # --- Comparisons ------------------------------------------------------
    # Excluding the just-saved path guarantees "previous" means *before now*.
    previous = store.load_latest(model, exclude=saved_path)

    # Reference was captured before save so auto-pin on first run still
    # produces a meaningful "vs reference: none" output.
    reference = reference_before

    try:
        if reference is not None:
            ModelSnapshotStore.assert_comparable(snapshot, reference)
        if previous is not None:
            ModelSnapshotStore.assert_comparable(snapshot, previous)
    except SnapshotSuiteMismatchError as exc:
        console.print(f"[yellow]Skipping comparison: {exc}[/yellow]")
        reference = None
        previous = None

    # Apply user-overridden or default thresholds.
    classify_kwargs = {}
    if drift_threshold is not None:
        classify_kwargs["weak_drift_delta"] = drift_threshold
    if medium_flip_count is not None:
        classify_kwargs["medium_flip_count"] = medium_flip_count

    vs_reference = classify(snapshot, reference, **classify_kwargs)
    vs_previous = classify(snapshot, previous, **classify_kwargs)

    # --- Output ----------------------------------------------------------
    if json_output:
        payload = build_json_payload(
            snapshot, suite, vs_reference, vs_previous, reference, previous
        )
        click.echo(json.dumps(payload, indent=2, default=str))
    else:
        render_header(snapshot, suite, outcome.total_cost_usd)
        render_comparison("vs reference", vs_reference, reference, snapshot)
        render_comparison("vs previous", vs_previous, previous, snapshot)
        render_next_steps(
            model,
            has_drift=(
                vs_reference.kind != DriftKind.NONE
                or vs_previous.kind != DriftKind.NONE
            ),
        )

    if out_path is not None:
        try:
            out_path.write_text(
                json.dumps(
                    build_json_payload(
                        snapshot, suite, vs_reference, vs_previous, reference, previous
                    ),
                    indent=2,
                    default=str,
                )
            )
        except OSError as exc:
            console.print(f"[red]Failed to write --out file {out_path}:[/red] {exc}")

    # --- Exit code -------------------------------------------------------
    has_any_drift = (
        vs_reference.kind != DriftKind.NONE or vs_previous.kind != DriftKind.NONE
    )
    sys.exit(EXIT_DRIFT_DETECTED if has_any_drift else EXIT_OK)


def _get_evalview_version() -> Optional[str]:
    """Best-effort version lookup for snapshot metadata."""
    try:
        from importlib.metadata import version

        return version("evalview")
    except Exception:  # pragma: no cover
        return None


__all__ = ["model_check"]
