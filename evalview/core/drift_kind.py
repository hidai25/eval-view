"""Unified drift taxonomy across EvalView.

`DriftStatus` answers "did the test pass, change, or regress?" — the *what*.
`DriftKind` answers "if it changed, what kind of drift was it?" — the *why*.

These two dimensions are orthogonal. A test can be `DiffStatus.OUTPUT_CHANGED`
with `DriftKind.MODEL` (the closed model drifted), `DriftKind.CONTRACT` (the
external MCP server changed its API), or `DriftKind.BEHAVIORAL` (same model,
same tools, different execution path).

Separating the two keeps `DiffStatus` stable while allowing multiple drift
sources to share one taxonomy. Adding a new drift source in the future only
requires a new `DriftKind` value.
"""
from __future__ import annotations

from enum import Enum


class DriftKind(Enum):
    """What kind of drift was detected, independent of pass/fail status.

    NONE        — no drift, or drift detection was not performed
    MODEL       — the underlying LLM behavior changed (silent model update,
                  provider change, fingerprint change)
    CONTRACT    — an external MCP server's tool schema changed under the agent
    BEHAVIORAL  — same model and tools, but a different execution path
    """

    NONE = "none"
    MODEL = "model"
    CONTRACT = "contract"
    BEHAVIORAL = "behavioral"


class DriftConfidence(Enum):
    """How confident the classifier is that observed drift is real.

    STRONG  — provider-level fingerprint change confirmed (ground truth)
    MEDIUM  — multiple prompts flipped direction; likely real behavioral change
    WEAK    — pass rate moved but nothing flipped; possibly sampling noise
    """

    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


__all__ = ["DriftKind", "DriftConfidence"]
