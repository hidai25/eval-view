"""Drift classification for `evalview model-check`.

Compares two model snapshots and classifies any behavioral drift by kind
(``DriftKind``) and confidence (``DriftConfidence``).  Extracted from the
command module so classification logic is reusable from CI integrations
or a programmatic API without importing Click.

Thresholds can be overridden at call-time. When a ``suite_size`` is
provided, the medium-confidence flip threshold scales proportionally
so large custom suites don't fire MEDIUM on a single noisy prompt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from evalview.core.drift_kind import DriftConfidence, DriftKind
from evalview.core.model_snapshots import ModelSnapshot


# --------------------------------------------------------------------------- #
# Default thresholds (module-level so they're importable as constants)
# --------------------------------------------------------------------------- #

DEFAULT_WEAK_DRIFT_DELTA = 0.01
"""Any per-prompt pass-rate change beyond this is noted as drift."""

DEFAULT_MEDIUM_FLIP_COUNT = 2
"""Minimum prompt flips for MEDIUM confidence (for suites <= 20 prompts)."""

DEFAULT_MEDIUM_FLIP_RATIO = 0.10
"""When suite has >20 prompts, MEDIUM requires this fraction of prompts to flip."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class PromptDelta:
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
class Classification:
    """Outcome of comparing a current snapshot against one other snapshot."""

    kind: DriftKind
    confidence: Optional[DriftConfidence]
    drift_count: int
    flipped_ids: List[str]
    pass_rate_delta: float
    deltas: List[PromptDelta] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Classification logic
# --------------------------------------------------------------------------- #


def _effective_medium_flip_count(
    suite_size: int,
    medium_flip_count: int,
    medium_flip_ratio: float,
) -> int:
    """Compute the effective flip-count threshold for MEDIUM confidence.

    For small suites (<= 20 prompts), uses the fixed ``medium_flip_count``.
    For larger suites, uses ``ceil(suite_size * medium_flip_ratio)`` so that
    a 50-prompt suite needs ~5 flips for MEDIUM rather than 2.
    """
    if suite_size <= 20:
        return medium_flip_count
    return max(medium_flip_count, math.ceil(suite_size * medium_flip_ratio))


def classify(
    current: ModelSnapshot,
    other: Optional[ModelSnapshot],
    *,
    weak_drift_delta: float = DEFAULT_WEAK_DRIFT_DELTA,
    medium_flip_count: int = DEFAULT_MEDIUM_FLIP_COUNT,
    medium_flip_ratio: float = DEFAULT_MEDIUM_FLIP_RATIO,
) -> Classification:
    """Compare *current* vs *other* snapshot and decide drift kind/confidence.

    Args:
        current: the snapshot just produced.
        other: the snapshot to compare against (reference or previous).
            If ``None``, returns ``DriftKind.NONE`` — there's nothing to
            compare against.
        weak_drift_delta: minimum per-prompt pass-rate delta to count as
            a drift signal.
        medium_flip_count: fixed flip threshold for MEDIUM on small suites.
        medium_flip_ratio: fractional flip threshold for MEDIUM on large
            suites (>20 prompts).
    """
    if other is None:
        return Classification(
            kind=DriftKind.NONE,
            confidence=None,
            drift_count=0,
            flipped_ids=[],
            pass_rate_delta=0.0,
        )

    by_id_other = {r.prompt_id: r for r in other.results}

    deltas: List[PromptDelta] = []
    drift_count = 0
    flipped_ids: List[str] = []

    for r in current.results:
        prior = by_id_other.get(r.prompt_id)
        if prior is None:
            # New prompt — not a drift signal, but record it with zero delta.
            deltas.append(
                PromptDelta(
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
        if abs(delta) > weak_drift_delta:
            drift_count += 1
        if flipped:
            flipped_ids.append(r.prompt_id)
        deltas.append(
            PromptDelta(
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
        and other.metadata.fingerprint_confidence == "strong"
    )

    # Scale the medium threshold for large suites.
    effective_medium = _effective_medium_flip_count(
        suite_size=len(current.results),
        medium_flip_count=medium_flip_count,
        medium_flip_ratio=medium_flip_ratio,
    )

    if fingerprint_changed:
        kind = DriftKind.MODEL
        confidence = DriftConfidence.STRONG
    elif len(flipped_ids) >= effective_medium:
        kind = DriftKind.MODEL
        confidence = DriftConfidence.MEDIUM
    elif drift_count > 0 or flipped_ids:
        kind = DriftKind.MODEL
        confidence = DriftConfidence.WEAK
    else:
        kind = DriftKind.NONE
        confidence = None

    return Classification(
        kind=kind,
        confidence=confidence,
        drift_count=drift_count,
        flipped_ids=flipped_ids,
        pass_rate_delta=pass_rate_delta,
        deltas=deltas,
    )


__all__ = [
    "Classification",
    "DEFAULT_MEDIUM_FLIP_COUNT",
    "DEFAULT_MEDIUM_FLIP_RATIO",
    "DEFAULT_WEAK_DRIFT_DELTA",
    "PromptDelta",
    "classify",
]
