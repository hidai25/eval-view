"""`evalview model-check` — closed-model behavioral drift detection.

Runs a small, fixed canary suite directly against a provider (Anthropic,
OpenAI, ...) with no user agent in the loop. Each prompt is scored by a
pure structural scorer (tool choice / JSON schema / refusal / regex), so
there is NO LLM judge dependency in v1 and no calibration problem.

Each invocation produces a snapshot. Drift comparisons use a three-anchor
model:

- **reference**  — the first-ever (or user-pinned) snapshot; never auto-
                   updated, so gradual drift is detectable.
- **latest prior** — the most recent snapshot before this run.
- **trend**     — OLS slope from core/drift_tracker over the full history.

Provider fingerprint signal strength is honestly labeled (STRONG for
OpenAI ``system_fingerprint``, WEAK for providers that only echo the
requested model id). See ``docs/MODEL_CHECK.md``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from evalview.benchmarks.canary import PUBLIC_SUITE_PATH
from evalview.commands.shared import console
from evalview.core.canary_suite import (
    CanaryPrompt,
    CanarySuite,
    CanarySuiteError,
    load_canary_suite,
)
from evalview.core.drift_kind import DriftConfidence, DriftKind
from evalview.core.model_check_scoring import ScoreResult, score_prompt
from evalview.core.model_snapshots import (
    ModelCheckPromptResult,
    ModelSnapshot,
    ModelSnapshotMetadata,
    ModelSnapshotStore,
    SnapshotSuiteMismatchError,
)
from evalview.core.pricing import get_model_pricing_info
from evalview.telemetry.decorators import track_command

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


# Cost estimation assumes a typical canary prompt uses roughly this many
# tokens on each side of the API boundary. Deliberately generous so the
# estimate errs on the side of over-quoting rather than surprising the user.
_EST_INPUT_TOKENS_PER_CALL = 400
_EST_OUTPUT_TOKENS_PER_CALL = 300

# Drift classification thresholds. Module-level so they can be tuned without
# touching the logic below.
_WEAK_DRIFT_DELTA = 0.01     # any pass-rate change beyond this is noted
_MEDIUM_DRIFT_FLIP_COUNT = 2  # two or more flipped prompts → medium confidence


# Provider name normalization: what the adapter / env considers native.
_KNOWN_PROVIDERS = ("anthropic", "openai")


# --------------------------------------------------------------------------- #
# Data classes (command-local)
# --------------------------------------------------------------------------- #


@dataclass
class _PromptDelta:
    """Per-prompt comparison between two snapshots."""

    prompt_id: str
    category: str
    current_rate: float
    other_rate: float
    flipped: bool

    @property
    def delta(self) -> float:
        return self.current_rate - self.other_rate


@dataclass
class _Classification:
    """Outcome of comparing a current snapshot against one other snapshot."""

    kind: DriftKind
    confidence: Optional[DriftConfidence]
    drift_count: int
    flipped_ids: List[str]
    pass_rate_delta: float
    deltas: List[_PromptDelta] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Provider adapter plumbing
# --------------------------------------------------------------------------- #


def _infer_provider(model_id: str, explicit: Optional[str]) -> str:
    """Guess the provider from the model id when --provider is omitted.

    Errors clearly when the caller's choice is unsupported so typos don't
    silently resolve to the wrong adapter.
    """
    if explicit:
        provider = explicit.strip().lower()
        if provider not in _KNOWN_PROVIDERS:
            raise click.UsageError(
                f"Unsupported provider '{explicit}'. "
                f"Supported in v1: {', '.join(_KNOWN_PROVIDERS)}."
            )
        return provider

    lowered = model_id.lower()
    if lowered.startswith("claude") or "anthropic" in lowered:
        return "anthropic"
    if lowered.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    raise click.UsageError(
        f"Could not infer provider from model id {model_id!r}. "
        f"Pass --provider explicitly ({', '.join(_KNOWN_PROVIDERS)})."
    )


def _check_api_key(provider: str) -> None:
    """Fail fast with a clear message before any adapter import is attempted."""
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }[provider]
    if not os.environ.get(env_var):
        raise click.UsageError(
            f"{env_var} is not set. Export it before running model-check."
        )


def _build_adapter(provider: str, model_id: str, *, timeout: float = 120.0):
    """Construct the raw provider adapter. Import is local to keep cold start fast."""
    if provider == "anthropic":
        from evalview.adapters.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(model=model_id, timeout=timeout)
    if provider == "openai":
        # Use the OpenAI Assistants adapter; "endpoint" is the model id here.
        from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter

        return OpenAIAssistantsAdapter(endpoint=model_id, timeout=timeout)
    raise click.UsageError(f"Unsupported provider: {provider}")


def _fingerprint_strength(provider: str) -> str:
    """Honest labeling of how much signal we actually get from each provider."""
    if provider == "openai":
        return "strong"  # system_fingerprint is per-response
    return "weak"  # Anthropic and friends: requested model id only


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


async def _run_single(
    adapter: Any,
    prompt: CanaryPrompt,
) -> Tuple[ScoreResult, float]:
    """Execute one prompt once against the adapter and score it.

    Returns the score result plus the measured latency in milliseconds.
    Adapter-level exceptions are caught and reported as a failed score so
    a single transient API error does not abort the whole run.
    """
    start = datetime.now(timezone.utc)
    try:
        trace = await adapter.execute(prompt.prompt)
    except Exception as exc:  # pragma: no cover - depends on provider error shape
        logger.warning("Adapter error on prompt %s: %s", prompt.id, exc)
        return ScoreResult(False, f"adapter error: {exc}"), 0.0

    latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000.0

    response = getattr(trace, "final_output", "") or ""
    tool_calls = [
        step.tool_name
        for step in getattr(trace, "steps", []) or []
        if getattr(step, "tool_name", None)
    ]

    try:
        result = score_prompt(
            prompt.scorer,
            response=response,
            tool_calls=tool_calls,
            expected=prompt.expected,
        )
    except ValueError as exc:
        # Suite-level misconfiguration — surface it loudly, never silently fail.
        raise click.UsageError(f"Prompt '{prompt.id}': {exc}") from exc

    return result, latency_ms


async def _run_prompt_with_retries(
    adapter: Any,
    prompt: CanaryPrompt,
    runs_per_prompt: int,
) -> ModelCheckPromptResult:
    """Run a prompt N times, aggregate pass rate and latency."""
    passes: List[bool] = []
    latencies: List[float] = []
    for _ in range(runs_per_prompt):
        result, latency_ms = await _run_single(adapter, prompt)
        passes.append(result.passed)
        latencies.append(latency_ms)

    pass_rate = sum(1 for p in passes if p) / len(passes)
    mean_latency = statistics.mean(latencies) if latencies else None
    stdev_latency = statistics.stdev(latencies) if len(latencies) > 1 else None

    return ModelCheckPromptResult(
        prompt_id=prompt.id,
        category=prompt.category,
        pass_rate=pass_rate,
        n_runs=runs_per_prompt,
        per_run_passed=passes,
        latency_ms_mean=mean_latency,
        latency_ms_stdev=stdev_latency,
    )


async def _run_suite(
    adapter: Any,
    suite: CanarySuite,
    runs_per_prompt: int,
    progress_cb=None,
) -> List[ModelCheckPromptResult]:
    results: List[ModelCheckPromptResult] = []
    for prompt in suite.prompts:
        result = await _run_prompt_with_retries(adapter, prompt, runs_per_prompt)
        results.append(result)
        if progress_cb is not None:
            progress_cb(result)
    return results


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
# Classification
# --------------------------------------------------------------------------- #


def _classify(current: ModelSnapshot, other: Optional[ModelSnapshot]) -> _Classification:
    """Compare current vs another snapshot and decide drift kind/confidence."""
    if other is None:
        return _Classification(
            kind=DriftKind.NONE,
            confidence=None,
            drift_count=0,
            flipped_ids=[],
            pass_rate_delta=0.0,
        )

    by_id_other = {r.prompt_id: r for r in other.results}

    deltas: List[_PromptDelta] = []
    drift_count = 0
    flipped_ids: List[str] = []

    for r in current.results:
        prior = by_id_other.get(r.prompt_id)
        if prior is None:
            # New prompt — not a drift signal, but record it with zero delta.
            deltas.append(
                _PromptDelta(
                    prompt_id=r.prompt_id,
                    category=r.category,
                    current_rate=r.pass_rate,
                    other_rate=r.pass_rate,
                    flipped=False,
                )
            )
            continue
        delta = r.pass_rate - prior.pass_rate
        flipped = r.passed != prior.passed
        if abs(delta) > _WEAK_DRIFT_DELTA:
            drift_count += 1
        if flipped:
            flipped_ids.append(r.prompt_id)
        deltas.append(
            _PromptDelta(
                prompt_id=r.prompt_id,
                category=r.category,
                current_rate=r.pass_rate,
                other_rate=prior.pass_rate,
                flipped=flipped,
            )
        )

    pass_rate_delta = current.overall_pass_rate - other.overall_pass_rate

    # Provider fingerprint is strong ground-truth signal when present.
    fp_now = current.metadata.provider_fingerprint
    fp_other = other.metadata.provider_fingerprint
    fingerprint_changed = (
        fp_now is not None
        and fp_other is not None
        and fp_now != fp_other
        and current.metadata.fingerprint_confidence == "strong"
    )

    if fingerprint_changed:
        kind = DriftKind.MODEL
        confidence = DriftConfidence.STRONG
    elif len(flipped_ids) >= _MEDIUM_DRIFT_FLIP_COUNT:
        kind = DriftKind.MODEL
        confidence = DriftConfidence.MEDIUM
    elif drift_count > 0 or flipped_ids:
        kind = DriftKind.MODEL
        confidence = DriftConfidence.WEAK
    else:
        kind = DriftKind.NONE
        confidence = None

    return _Classification(
        kind=kind,
        confidence=confidence,
        drift_count=drift_count,
        flipped_ids=flipped_ids,
        pass_rate_delta=pass_rate_delta,
        deltas=deltas,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt_drift(cls: _Classification) -> str:
    if cls.kind == DriftKind.NONE:
        return "NONE"
    conf = cls.confidence.value if cls.confidence else "unknown"
    return f"{cls.kind.value.upper()} ({conf} confidence)"


def _render_header(snapshot: ModelSnapshot, suite: CanarySuite, cost: float) -> None:
    md = snapshot.metadata
    fp_label = md.provider_fingerprint or "(none)"
    fp_strength = md.fingerprint_confidence
    strength_hint = {
        "strong": "per-response fingerprint",
        "weak": "behavior-only — provider does not expose per-response fingerprint",
    }.get(fp_strength, fp_strength)

    console.print()
    console.print("[bold]EvalView model-check[/bold]")
    console.print(f"  Model:        {md.model_id}")
    console.print(f"  Provider:     {md.provider}")
    console.print(
        f"  Suite:        {suite.suite_name} {suite.version} "
        f"({len(suite.prompts)} prompts, {suite.suite_hash[:19]}…)"
    )
    console.print(f"  Runs/prompt:  {md.runs_per_prompt}")
    console.print(f"  Temperature:  {md.temperature}")
    console.print(f"  Fingerprint:  {fp_label} [{fp_strength} — {strength_hint}]")
    console.print(f"  Cost:         ${cost:.4f}")
    console.print()


def _render_comparison(
    title: str,
    cls: _Classification,
    other: Optional[ModelSnapshot],
    current: ModelSnapshot,
) -> None:
    if other is None:
        console.print(f"[dim]{title}: no prior snapshot — this run is the baseline.[/dim]")
        console.print()
        return

    age = current.metadata.snapshot_at - other.metadata.snapshot_at
    days = max(int(age.total_seconds() // 86400), 0)
    other_ts = other.metadata.snapshot_at.strftime("%Y-%m-%d")
    console.print(f"[bold]{title}[/bold] ({other_ts}, {days}d ago)")

    drift_label = _fmt_drift(cls)
    drift_color = {
        DriftKind.NONE: "green",
        DriftKind.MODEL: "yellow",
        DriftKind.CONTRACT: "yellow",
        DriftKind.BEHAVIORAL: "cyan",
    }.get(cls.kind, "white")
    console.print(f"  Drift:      [{drift_color}]{drift_label}[/{drift_color}]")
    console.print(
        f"  Pass rate:  {other.passed_count}/{other.total_count} → "
        f"{current.passed_count}/{current.total_count} "
        f"({cls.pass_rate_delta:+.1%})"
    )
    if cls.flipped_ids:
        console.print(f"  Flipped:    {', '.join(cls.flipped_ids)}")
    console.print()


def _render_next_steps(model_id: str, has_drift: bool) -> None:
    console.print("[dim]Next steps:[/dim]")
    if has_drift:
        console.print(
            f"[dim]  • Accept as new reference: "
            f"evalview model-check --model {model_id} --pin[/dim]"
        )
    console.print(
        f"[dim]  • Reset baseline:          "
        f"evalview model-check --model {model_id} --reset-reference[/dim]"
    )
    console.print(
        "[dim]  • Full JSON output:        "
        "add --json to any invocation[/dim]"
    )
    console.print()


# --------------------------------------------------------------------------- #
# JSON output
# --------------------------------------------------------------------------- #


def _build_json_payload(
    snapshot: ModelSnapshot,
    suite: CanarySuite,
    vs_reference: _Classification,
    vs_previous: _Classification,
    reference: Optional[ModelSnapshot],
    previous: Optional[ModelSnapshot],
) -> Dict[str, Any]:
    def _cls_dict(cls: _Classification, other: Optional[ModelSnapshot]) -> Dict[str, Any]:
        return {
            "drift_kind": cls.kind.value,
            "drift_confidence": cls.confidence.value if cls.confidence else None,
            "pass_rate_delta": cls.pass_rate_delta,
            "drift_count": cls.drift_count,
            "flipped_prompts": cls.flipped_ids,
            "other_snapshot_at": other.metadata.snapshot_at.isoformat() if other else None,
        }

    return {
        "schema_version": 1,
        "snapshot": json.loads(snapshot.model_dump_json()),
        "suite": {
            "name": suite.suite_name,
            "version": suite.version,
            "hash": suite.suite_hash,
            "prompt_count": len(suite.prompts),
        },
        "vs_reference": _cls_dict(vs_reference, reference),
        "vs_previous": _cls_dict(vs_previous, previous),
    }


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
) -> None:
    """Detect behavioral drift in a closed model against a fixed canary suite.

    v1 runs structural-only prompts (tool choice, JSON schema, refusal,
    regex) against Anthropic or OpenAI and compares the result against
    two anchors: the pinned reference (never auto-updates) and the most
    recent prior snapshot. Drift is classified as one of NONE / WEAK /
    MEDIUM / STRONG depending on how many prompts flipped direction and
    whether the provider exposed a fingerprint change.

    No LLM judge is used, so there is no calibration requirement.
    """
    if runs_per_prompt < 1:
        raise click.UsageError("--runs must be >= 1")

    # --- Load suite -------------------------------------------------------
    suite_file = suite_path or PUBLIC_SUITE_PATH
    try:
        suite = load_canary_suite(suite_file)
    except CanarySuiteError as exc:
        console.print(f"[red]Failed to load canary suite:[/red] {exc}")
        sys.exit(EXIT_USAGE_ERROR)

    # --- Resolve provider -------------------------------------------------
    try:
        provider_resolved = _infer_provider(model, provider)
    except click.UsageError as exc:
        console.print(f"[red]{exc.message}[/red]")
        sys.exit(EXIT_USAGE_ERROR)

    n_calls = len(suite.prompts) * runs_per_prompt
    estimated_cost = _estimate_cost_usd(model, n_calls)

    if estimated_cost > budget and not dry_run:
        console.print(
            f"[red]Estimated cost ${estimated_cost:.4f} exceeds --budget ${budget:.2f}.[/red] "
            f"Run with --dry-run to confirm or raise --budget."
        )
        sys.exit(EXIT_USAGE_ERROR)

    if dry_run:
        console.print()
        console.print("[bold]Would run:[/bold] " + model)
        console.print(
            f"  Suite:           {suite.suite_name} {suite.version} "
            f"({len(suite.prompts)} prompts × {runs_per_prompt} runs = {n_calls} calls)"
        )
        console.print(f"  Provider:        {provider_resolved}")
        console.print(f"  Estimated cost:  ${estimated_cost:.4f}")
        console.print(f"  Budget cap:      ${budget:.2f}")
        console.print()
        console.print("[dim]Re-run without --dry-run to execute.[/dim]")
        console.print()
        return

    _check_api_key(provider_resolved)

    # --- Load store + handle reference management ------------------------
    store = ModelSnapshotStore()
    if reset_reference:
        existed = store.reset_reference(model)
        if existed:
            console.print(f"[yellow]Reference for {model} was deleted.[/yellow]")

    reference_before = store.load_reference(model)
    previous_before = store.load_latest(model)

    # --- Run the suite ---------------------------------------------------
    adapter = _build_adapter(provider_resolved, model)

    # JSON mode must produce clean machine-readable output. Human progress
    # messages go to stderr so JSON mode stays pure on stdout.
    if not json_output:
        console.print(
            f"[dim]Running {len(suite.prompts)} prompts × {runs_per_prompt} runs "
            f"against {model}…[/dim]"
        )

    try:
        results = asyncio.run(_run_suite(adapter, suite, runs_per_prompt))
    except Exception as exc:
        console.print(f"[red]model-check failed during execution:[/red] {exc}")
        sys.exit(EXIT_USAGE_ERROR)

    # --- Build snapshot ---------------------------------------------------
    metadata = ModelSnapshotMetadata(
        model_id=model,
        provider=provider_resolved,
        snapshot_at=datetime.now(timezone.utc),
        suite_name=suite.suite_name,
        suite_version=suite.version,
        suite_hash=suite.suite_hash,
        temperature=0.0,
        top_p=1.0,
        runs_per_prompt=runs_per_prompt,
        provider_fingerprint=model,  # best we can do for weak-fp providers
        fingerprint_confidence=_fingerprint_strength(provider_resolved),
        cost_total_usd=estimated_cost,  # true cost tracking lands in v1.1
    )
    snapshot = ModelSnapshot(metadata=metadata, results=results)

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

    # --- Comparisons ------------------------------------------------------
    # Excluding the just-saved path guarantees "previous" means *before now*.
    previous = (
        store.load_latest(model, exclude=saved_path)
        if saved_path is not None
        else previous_before
    )

    # Reference was captured before save so auto-pin on first run still
    # produces a meaningful "vs reference: none" output.
    reference = reference_before

    try:
        if reference is not None:
            ModelSnapshotStore.assert_comparable(snapshot, reference)
        if previous is not None:
            ModelSnapshotStore.assert_comparable(snapshot, previous)
    except SnapshotSuiteMismatchError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(EXIT_USAGE_ERROR)

    vs_reference = _classify(snapshot, reference)
    vs_previous = _classify(snapshot, previous)

    # --- Output ----------------------------------------------------------
    if json_output:
        payload = _build_json_payload(
            snapshot, suite, vs_reference, vs_previous, reference, previous
        )
        click.echo(json.dumps(payload, indent=2, default=str))
    else:
        _render_header(snapshot, suite, estimated_cost)
        _render_comparison("vs reference", vs_reference, reference, snapshot)
        _render_comparison("vs previous", vs_previous, previous, snapshot)
        _render_next_steps(
            model,
            has_drift=(
                vs_reference.kind != DriftKind.NONE
                or vs_previous.kind != DriftKind.NONE
            ),
        )

    if out_path is not None:
        out_path.write_text(
            json.dumps(
                _build_json_payload(
                    snapshot, suite, vs_reference, vs_previous, reference, previous
                ),
                indent=2,
                default=str,
            )
        )

    # --- Exit code -------------------------------------------------------
    has_any_drift = (
        vs_reference.kind != DriftKind.NONE or vs_previous.kind != DriftKind.NONE
    )
    sys.exit(EXIT_DRIFT_DETECTED if has_any_drift else EXIT_OK)


__all__ = ["model_check"]
