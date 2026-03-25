"""Auto-heal engine for evalview check --heal.

Bounded remediation for clearly safe cases:
- RETRY flaky failures (pure output/score drift, no tool changes)
- PROPOSE_VARIANT when retry fails but score is acceptable
- Hard-escalate everything else (forbidden tools, structural changes, etc.)

Full audit trail written to .evalview/healing/.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from evalview.core.diff import DiffStatus, TraceDiff, DiffEngine
from evalview.core.golden import GoldenStore, GoldenTrace
from evalview.core.types import EvaluationResult, TestCase

if TYPE_CHECKING:
    from evalview.adapters.base import AgentAdapter
    from evalview.evaluators.evaluator import Evaluator

logger = logging.getLogger(__name__)

# --- Thresholds (tune these, not logic) ---
MIN_VARIANT_SCORE: float = 70.0
MAX_COST_MULTIPLIER: float = 2.0
MAX_LATENCY_MULTIPLIER: float = 3.0
MAX_AUTO_VARIANTS: int = 3
MODEL_UPDATE_RETRY_ALL: bool = True


# --- Enums ---

class HealingAction(str, Enum):
    NO_ACTION = "no_action"
    RETRY = "retry"
    PROPOSE_VARIANT = "propose_variant"
    FLAG_REVIEW = "flag_review"
    BLOCKED = "blocked"


class HealingTrigger(str, Enum):
    """What caused the healing action — structured, not string-matched."""
    NONDETERMINISM = "nondeterminism"
    MODEL_UPDATE = "model_update"
    FORBIDDEN_TOOL = "forbidden_tool"
    STRUCTURAL_CHANGE = "structural_change"
    PARAM_CHANGE = "param_change"
    COST_SPIKE = "cost_spike"
    LATENCY_SPIKE = "latency_spike"
    SCORE_IMPROVEMENT = "score_improvement"
    OTHER = "other"


# --- Models ---

class HealingDiagnosis(BaseModel):
    """Structured diagnosis — returned by diagnose(). Feeds audit log."""
    action: HealingAction
    trigger: HealingTrigger = HealingTrigger.OTHER
    reason: str
    root_cause_category: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HealingResult(BaseModel):
    """Result of attempting to heal a single test."""
    test_name: str
    original_status: str  # DiffStatus.value
    diagnosis: HealingDiagnosis
    attempted: bool = False
    healed: bool
    proposed: bool = False
    final_status: str  # DiffStatus.value
    original_score: Optional[float] = None
    baseline_score: Optional[float] = None
    retry_score: Optional[float] = None
    retry_status: Optional[str] = None
    baseline_model: Optional[str] = None
    actual_model: Optional[str] = None
    variant_saved: Optional[str] = None
    variant_path: Optional[str] = None


class ModelUpdateSummary(BaseModel):
    """Populated when model_changed=True detected on any test."""
    golden_model: str
    actual_model: str
    affected_count: int
    healed_count: int
    failed_count: int


class HealingSummary(BaseModel):
    """Summary of all healing actions for a check run."""
    results: List[HealingResult]
    total_healed: int
    total_proposed: int
    total_review: int
    total_blocked: int
    attempted_count: int = 0
    unresolved_count: int = 0
    failed_count: int = 0
    policy_version: str = "v1"
    thresholds: Dict[str, float] = Field(default_factory=dict)
    model_update: Optional[ModelUpdateSummary] = None
    audit_path: Optional[str] = None


# --- Engine ---

class HealingEngine:
    """Diagnose failures and attempt bounded remediation."""

    def __init__(self, golden_store: GoldenStore, evaluator: "Evaluator"):
        self._store = golden_store
        self._evaluator = evaluator

    def diagnose(
        self,
        diff: TraceDiff,
        result: EvaluationResult,
        test_case: TestCase,
        golden: GoldenTrace,
    ) -> HealingDiagnosis:
        """Pure, deterministic, zero side effects, zero cost."""

        # 1. PASSED -> NO_ACTION
        if diff.overall_severity == DiffStatus.PASSED:
            return HealingDiagnosis(
                action=HealingAction.NO_ACTION,
                trigger=HealingTrigger.OTHER,
                reason="passed",
            )

        # 2. Forbidden tool -> BLOCKED
        ft_eval = result.evaluations.forbidden_tools
        if ft_eval and not ft_eval.passed:
            return HealingDiagnosis(
                action=HealingAction.BLOCKED,
                trigger=HealingTrigger.FORBIDDEN_TOOL,
                reason=f"forbidden tool called: {', '.join(ft_eval.violations)}",
                details={"violations": ft_eval.violations},
            )

        # 3. Tool added/removed/reordered -> FLAG_REVIEW
        structural = [
            td for td in diff.tool_diffs
            if td.type in ("added", "removed", "reordered")
        ]
        structural += [
            td for td in diff.tool_diffs
            if td.type == "changed" and td.golden_tool != td.actual_tool
        ]
        if structural:
            tool_changes = [
                f"{td.type}: {td.golden_tool or '?'} -> {td.actual_tool or '?'}"
                for td in structural
            ]
            return HealingDiagnosis(
                action=HealingAction.FLAG_REVIEW,
                trigger=HealingTrigger.STRUCTURAL_CHANGE,
                reason=f"tool {'changes' if len(structural) > 1 else 'change'} detected — review needed",
                details={"tool_changes": tool_changes},
            )

        # 4. Cost spike -> FLAG_REVIEW
        golden_cost = golden.trace.metrics.total_cost
        if golden_cost > 0 and result.trace.metrics.total_cost > golden_cost * MAX_COST_MULTIPLIER:
            return HealingDiagnosis(
                action=HealingAction.FLAG_REVIEW,
                trigger=HealingTrigger.COST_SPIKE,
                reason=f"cost spike: ${result.trace.metrics.total_cost:.4f} > {MAX_COST_MULTIPLIER}x baseline ${golden_cost:.4f}",
                details={
                    "golden_cost": golden_cost,
                    "actual_cost": result.trace.metrics.total_cost,
                    "multiplier": result.trace.metrics.total_cost / golden_cost if golden_cost > 0 else 0,
                },
            )

        # 4b. Latency spike -> FLAG_REVIEW
        golden_latency = golden.trace.metrics.total_latency
        if golden_latency > 0 and result.trace.metrics.total_latency > golden_latency * MAX_LATENCY_MULTIPLIER:
            return HealingDiagnosis(
                action=HealingAction.FLAG_REVIEW,
                trigger=HealingTrigger.LATENCY_SPIKE,
                reason=f"latency spike: {result.trace.metrics.total_latency:.0f}ms > {MAX_LATENCY_MULTIPLIER}x baseline {golden_latency:.0f}ms",
                details={
                    "golden_latency": golden_latency,
                    "actual_latency": result.trace.metrics.total_latency,
                    "multiplier": result.trace.metrics.total_latency / golden_latency if golden_latency > 0 else 0,
                },
            )

        # 5. Score went UP -> FLAG_REVIEW (candidate improvement, not auto-accept)
        if diff.score_diff > 0:
            return HealingDiagnosis(
                action=HealingAction.FLAG_REVIEW,
                trigger=HealingTrigger.SCORE_IMPROVEMENT,
                reason="candidate improvement — run `evalview snapshot` to accept",
                details={"score_delta": diff.score_diff},
            )

        # 6. Model changed -> RETRY with MODEL_UPDATE trigger
        if diff.model_changed:
            return HealingDiagnosis(
                action=HealingAction.RETRY,
                trigger=HealingTrigger.MODEL_UPDATE,
                reason=f"model update drift ({diff.actual_model_id})",
                details={
                    "golden_model": diff.golden_model_id,
                    "actual_model": diff.actual_model_id,
                },
            )

        # 7. Parameter changes (same tool, different args) -> FLAG_REVIEW
        if diff.tool_diffs:
            return HealingDiagnosis(
                action=HealingAction.FLAG_REVIEW,
                trigger=HealingTrigger.PARAM_CHANGE,
                reason="parameter changes detected — review needed",
                details={"param_diffs": len(diff.tool_diffs)},
            )

        # 8. Output drift / score drop with NO tool diffs -> RETRY
        return HealingDiagnosis(
            action=HealingAction.RETRY,
            trigger=HealingTrigger.NONDETERMINISM,
            reason="suspected non-determinism — retrying",
        )

    async def heal_test(
        self,
        diff: TraceDiff,
        result: EvaluationResult,
        test_case: TestCase,
        golden_variants: List[GoldenTrace],
        adapter: "AgentAdapter",
        diff_engine: DiffEngine,
    ) -> HealingResult:
        """Diagnose -> retry if appropriate -> propose variant if retry fails."""

        golden = golden_variants[0]
        diagnosis = self.diagnose(diff, result, test_case, golden)

        if diagnosis.action != HealingAction.RETRY:
            return HealingResult(
                test_name=test_case.name,
                original_status=diff.overall_severity.value,
                diagnosis=diagnosis,
                attempted=False,
                healed=False,
                proposed=(diagnosis.action == HealingAction.PROPOSE_VARIANT),
                final_status=diff.overall_severity.value,
                original_score=result.score,
                baseline_score=golden.metadata.score,
                baseline_model=golden.metadata.model_id,
                actual_model=getattr(result.trace, "model_id", None),
            )

        # --- RETRY ---
        if test_case.is_multi_turn:
            from evalview.commands.shared import _execute_multi_turn_trace
            retry_trace = await _execute_multi_turn_trace(test_case, adapter)
        else:
            retry_trace = await adapter.execute(
                test_case.input.query, test_case.input.context
            )

        retry_result = await self._evaluator.evaluate(test_case, retry_trace)
        retry_diff = await diff_engine.compare_multi_reference_async(
            golden_variants, retry_trace, retry_result.score
        )

        # Retry passed?
        if retry_diff.overall_severity == DiffStatus.PASSED:
            return HealingResult(
                test_name=test_case.name,
                original_status=diff.overall_severity.value,
                diagnosis=HealingDiagnosis(
                    action=HealingAction.RETRY,
                    trigger=diagnosis.trigger,
                    reason=f"retried — {diagnosis.trigger.value}",
                    details=diagnosis.details,
                ),
                attempted=True,
                healed=True,
                proposed=False,
                final_status=DiffStatus.PASSED.value,
                original_score=result.score,
                baseline_score=golden.metadata.score,
                retry_score=retry_result.score,
                retry_status=DiffStatus.PASSED.value,
                baseline_model=golden.metadata.model_id,
                actual_model=getattr(retry_result.trace, "model_id", None),
            )

        # Retry failed — can we propose a variant?
        no_structural_changes = not any(
            td.type in ("added", "removed")
            or (td.type == "changed" and td.golden_tool != td.actual_tool)
            for td in retry_diff.tool_diffs
        )
        total_variants = self._store.count_variants(test_case.name)
        has_default_variant = self._store.load_golden(test_case.name) is not None
        named_variant_count = max(0, total_variants - (1 if has_default_variant else 0))

        if (
            retry_result.score >= MIN_VARIANT_SCORE
            and no_structural_changes
            and named_variant_count < MAX_AUTO_VARIANTS
        ):
            h = hashlib.md5(retry_trace.final_output.encode()).hexdigest()[:4]
            variant_name = f"auto_heal_{h}"
            saved_path = self._store.save_golden(
                retry_result,
                notes=f"Auto-heal candidate variant (score {retry_result.score:.1f})",
                variant_name=variant_name,
            )
            return HealingResult(
                test_name=test_case.name,
                original_status=diff.overall_severity.value,
                diagnosis=HealingDiagnosis(
                    action=HealingAction.PROPOSE_VARIANT,
                    trigger=diagnosis.trigger,
                    reason=f"saved candidate variant {variant_name} (score {retry_result.score:.1f})",
                    details={
                        **diagnosis.details,
                        "retry_status": retry_diff.overall_severity.value,
                        "named_variant_count_before": named_variant_count,
                        "max_auto_variants": MAX_AUTO_VARIANTS,
                    },
                ),
                attempted=True,
                healed=False,
                proposed=True,
                final_status=retry_diff.overall_severity.value,
                original_score=result.score,
                baseline_score=golden.metadata.score,
                retry_score=retry_result.score,
                retry_status=retry_diff.overall_severity.value,
                baseline_model=golden.metadata.model_id,
                actual_model=getattr(retry_result.trace, "model_id", None),
                variant_saved=variant_name,
                variant_path=str(saved_path),
            )

        # Retry failed, can't variant -> escalate
        retry_details = {
            **diagnosis.details,
            "retry_score": retry_result.score,
            "retry_status": retry_diff.overall_severity.value,
            "named_variant_count_before": named_variant_count,
            "max_auto_variants": MAX_AUTO_VARIANTS,
            "min_variant_score": MIN_VARIANT_SCORE,
            "variant_blocked_by": [],
        }
        if retry_result.score < MIN_VARIANT_SCORE:
            retry_details["variant_blocked_by"].append("score_below_threshold")
        if not no_structural_changes:
            retry_details["variant_blocked_by"].append("structural_change")
        if named_variant_count >= MAX_AUTO_VARIANTS:
            retry_details["variant_blocked_by"].append("variant_capacity_reached")
        return HealingResult(
            test_name=test_case.name,
            original_status=diff.overall_severity.value,
            diagnosis=HealingDiagnosis(
                action=HealingAction.FLAG_REVIEW,
                trigger=diagnosis.trigger,
                reason="retry failed and could not be proposed as a variant",
                details=retry_details,
            ),
            attempted=True,
            healed=False,
            proposed=False,
            final_status=retry_diff.overall_severity.value,
            original_score=result.score,
            baseline_score=golden.metadata.score,
            retry_score=retry_result.score,
            retry_status=retry_diff.overall_severity.value,
            baseline_model=golden.metadata.model_id,
            actual_model=getattr(retry_result.trace, "model_id", None),
        )


# --- Audit log ---

def save_audit_log(summary: HealingSummary) -> str:
    """Write .evalview/healing/<ISO-timestamp>.json.

    Only call when --heal produced at least one non-PASSED case.
    Returns the file path.
    """
    healing_dir = Path(".evalview") / "healing"
    healing_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    path = healing_dir / f"{ts}.json"

    with open(path, "w") as f:
        f.write(summary.model_dump_json(indent=2))

    logger.info(f"Healing audit log saved: {path}")
    return str(path)
