"""Shared observability signal extraction for EvalView.

Provides a single source of truth for extracting and summarizing behavioral
anomalies, trust scores, and coherence issues from evaluation results.

Every surface that displays or transmits observability data should call
``extract_observability_summary()`` rather than re-implementing the
extraction logic.

The ``AnomalyReportDict``, ``TrustReportDict``, and ``CoherenceReportDict``
TypedDicts define the canonical schema for observability reports.  All
``to_dict()`` methods on the report dataclasses conform to these schemas,
and all consumers should type-hint against them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Pydantic on Python < 3.12 requires typing_extensions.TypedDict for field
# validation; typing.TypedDict triggers PydanticUserError. typing_extensions
# backports also cover Required/NotRequired and total=False semantics the
# same way across all versions.
from typing_extensions import NotRequired, TypedDict


# ── Report schemas (TypedDict) ───────────────────────────────────────────
# These define the dict shape produced by to_dict() on each report class.
# Consumers (templates, CI comments, cloud push) should type-hint against
# these rather than using Dict[str, Any].


class AnomalyEntryDict(TypedDict):
    """Schema for a single anomaly entry in the anomaly report."""
    pattern: str
    severity: str
    description: str
    step_indices: List[int]
    tool_name: Optional[str]
    evidence: Dict[str, Any]


class AnomalyReportDict(TypedDict):
    """Schema for the anomaly_report field on EvaluationResult."""
    anomalies: List[AnomalyEntryDict]
    total_steps: int
    unique_tools: int
    error_count: int
    summary: str


class TrustFlagDict(TypedDict):
    """Schema for a single gaming flag in the trust report.

    ``check`` / ``severity`` / ``description`` / ``evidence`` are always
    emitted by ``GamingFlag.to_dict()``. ``hint`` is :class:`NotRequired`
    — it carries a one-sentence recommendation for a human reviewer
    ("quarantine this test", "verify the tool list passed to the
    agent") and is emitted only when the originating check defines one.
    Cloud surfaces it verbatim in the TrustIndicator popover so the UX
    doesn't have to invent meaning that belongs on the OSS side.
    """
    check: str
    severity: str
    description: str
    evidence: Dict[str, Any]
    hint: NotRequired[str]


class TrustReportDict(TypedDict):
    """Schema for the trust_report field on EvaluationResult."""
    flags: List[TrustFlagDict]
    trust_score: float
    summary: str


class CoherenceIssueDict(TypedDict):
    """Schema for a single coherence issue in the coherence report."""
    category: str
    severity: str
    turn_index: int
    reference_turn: Optional[int]
    description: str
    evidence: Dict[str, Any]


class CoherenceReportDict(TypedDict):
    """Schema for the coherence_report field on EvaluationResult."""
    issues: List[CoherenceIssueDict]
    total_turns: int
    coherence_score: float
    summary: str

# ── Thresholds (single source of truth) ──────────────────────────────────

#: Trust score below this threshold is considered "low trust"
LOW_TRUST_THRESHOLD: float = 0.8


# ── Summary dataclass ──────────��─────────────────────────────────────────


#: Schema version for the verdict/cloud/MCP observability payload.
#: Bump on any breaking shape change so downstream consumers can gate on it.
OBSERVABILITY_SCHEMA_VERSION: str = "1"


@dataclass
class ObservabilitySummary:
    """Aggregated observability signals across a set of evaluation results."""

    anomaly_count: int = 0
    anomaly_tests: List[str] = field(default_factory=list)

    low_trust_count: int = 0
    low_trust_tests: List[str] = field(default_factory=list)

    coherence_issue_count: int = 0
    coherence_tests: List[str] = field(default_factory=list)

    #: Suite-level anti-gaming flags (e.g. all tests scoring perfect, suspiciously
    #: uniform latency across the batch). Each entry matches TrustFlagDict shape.
    batch_hardening_flags: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_signals(self) -> bool:
        return (
            self.anomaly_count > 0
            or self.low_trust_count > 0
            or self.coherence_issue_count > 0
            or len(self.batch_hardening_flags) > 0
        )

    def to_payload(self) -> Dict[str, Any]:
        """Return the observability body (without the ``observability`` wrapper).

        Signals are reported as aggregate counts with a stable ``schema_version``.
        Callers decide whether to wrap (e.g. as ``{"observability": payload}``)
        or merge into an existing envelope. Returns an empty dict when no
        signals are present, so callers can do ``if payload: ...``.
        """
        body: Dict[str, Any] = {"schema_version": OBSERVABILITY_SCHEMA_VERSION}
        if self.anomaly_count:
            body["anomalies"] = {
                "count": self.anomaly_count,
                "tests": self.anomaly_tests[:10],
            }
        if self.low_trust_count:
            body["low_trust"] = {
                "count": self.low_trust_count,
                "tests": self.low_trust_tests[:10],
            }
        if self.coherence_issue_count:
            body["coherence"] = {
                "count": self.coherence_issue_count,
                "tests": self.coherence_tests[:10],
            }
        if self.batch_hardening_flags:
            body["batch_hardening"] = {
                "flag_count": len(self.batch_hardening_flags),
                "flags": self.batch_hardening_flags,
            }
        # Nothing signaled beyond the version stamp — caller likely wants to skip.
        if len(body) == 1:
            return {}
        return body

    # Backwards-compatible alias — some callers (CI comment renderer) still
    # merge signal keys into the verdict payload directly. Returns the same
    # shape as ``to_payload`` but wrapped under the ``observability`` key.
    def to_verdict_payload(self) -> Dict[str, Any]:
        body = self.to_payload()
        return {"observability": body} if body else {}


# ── Extraction helper ���───────────────────────────���───────────────────────


def extract_observability_summary(
    results: Optional[List[Any]],
) -> ObservabilitySummary:
    """Extract observability signals from a list of EvaluationResult objects.

    Runs per-test aggregation (anomalies, low-trust, coherence) and also the
    suite-level batch anti-gaming check (all-perfect, uniform-latency).

    Safe to call with None, empty list, or results that lack the new fields.
    """
    summary = ObservabilitySummary()
    if not results:
        return summary

    batch_inputs: List[Dict[str, Any]] = []

    for r in results:
        test_name = getattr(r, "test_case", "?")
        try:
            ar = getattr(r, "anomaly_report", None)
            if isinstance(ar, dict) and ar.get("anomalies"):
                summary.anomaly_count += 1
                summary.anomaly_tests.append(test_name)

            tr = getattr(r, "trust_report", None)
            if isinstance(tr, dict):
                trust_score = float(tr.get("trust_score", 1.0))
                if trust_score < LOW_TRUST_THRESHOLD:
                    summary.low_trust_count += 1
                    summary.low_trust_tests.append(test_name)

            cr = getattr(r, "coherence_report", None)
            if isinstance(cr, dict) and cr.get("issues"):
                summary.coherence_issue_count += 1
                summary.coherence_tests.append(test_name)

            score = getattr(r, "score", None)
            if isinstance(score, (int, float)):
                entry: Dict[str, Any] = {"score": float(score)}
                # Latency lives on result.evaluations.latency.total_latency (seconds).
                # Convert to ms for batch check; skip if unmeasured (0 or missing).
                try:
                    total_latency_s = r.evaluations.latency.total_latency
                    if total_latency_s and total_latency_s > 0:
                        entry["latency_ms"] = float(total_latency_s) * 1000.0
                except AttributeError:
                    pass
                batch_inputs.append(entry)
        except (TypeError, ValueError, AttributeError):
            continue

    # Suite-level batch anti-gaming check (all-perfect, uniform-latency).
    # Imported lazily to avoid a circular import at module load time.
    if batch_inputs:
        try:
            from evalview.core.benchmark_hardening import check_gaming_batch
            batch_report = check_gaming_batch(batch_inputs)
            if batch_report.flags:
                summary.batch_hardening_flags = [
                    f.to_dict() for f in batch_report.flags
                ]
        except Exception:  # noqa: BLE001 — observability must never break the caller
            pass

    return summary
