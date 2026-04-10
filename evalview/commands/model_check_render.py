"""Rendering helpers for `evalview model-check` CLI output.

Extracted from the command module so rendering logic is testable in
isolation and the command module stays focused on orchestration.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from evalview.commands.shared import console
from evalview.core.canary_suite import CanarySuite
from evalview.core.drift_classifier import Classification
from evalview.core.drift_kind import DriftKind
from evalview.core.model_snapshots import ModelSnapshot


# --------------------------------------------------------------------------- #
# Human-readable rendering
# --------------------------------------------------------------------------- #


def fmt_drift(cls: Classification) -> str:
    if cls.kind == DriftKind.NONE:
        return "NONE"
    conf = cls.confidence.value if cls.confidence else "unknown"
    return f"{cls.kind.value.upper()} ({conf} confidence)"


def render_header(snapshot: ModelSnapshot, suite: CanarySuite, cost: float) -> None:
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


def render_comparison(
    title: str,
    cls: Classification,
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

    drift_label = fmt_drift(cls)
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


def render_next_steps(model_id: str, has_drift: bool) -> None:
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


def build_json_payload(
    snapshot: ModelSnapshot,
    suite: CanarySuite,
    vs_reference: Classification,
    vs_previous: Classification,
    reference: Optional[ModelSnapshot],
    previous: Optional[ModelSnapshot],
) -> Dict[str, Any]:
    def _cls_dict(cls: Classification, other: Optional[ModelSnapshot]) -> Dict[str, Any]:
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


__all__ = [
    "build_json_payload",
    "fmt_drift",
    "render_comparison",
    "render_header",
    "render_next_steps",
]
