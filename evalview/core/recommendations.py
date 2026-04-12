"""
Improvement Recommendation Engine.

Generates specific, actionable recommendations based on failure patterns.
Complements root_cause.py (which identifies what changed) by suggesting
what to do about it — specific enough to act on, not generic advice.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Recommendation:
    """Actionable suggestion for fixing a regression.

    Decision-grade fields:
        - `likely_cause`: short hypothesis about WHY this happened, used by the
          verdict panel and PR comment as the "likely cause (confidence)" line.
        - `severity`: "high" | "medium" | "low" — how urgent this is, distinct
          from `confidence` (how sure we are of the diagnosis).
        - `suggested_commands`: runnable `evalview …` commands the user can
          copy-paste. Each rec emits zero or more. This is the bridge from
          "good advice" to "operational leverage."
    """

    action: str       # e.g. "Tighten tool description for search_db"
    confidence: str   # "high" | "medium" | "low"
    category: str     # "prompt" | "tool" | "model" | "routing" | "guardrail" | "config"
    detail: str       # 1-2 sentence explanation
    likely_cause: str = ""
    severity: str = "medium"
    suggested_commands: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "category": self.category,
            "detail": self.detail,
            "likely_cause": self.likely_cause,
            "severity": self.severity,
            "suggested_commands": list(self.suggested_commands),
        }


def recommend(
    *,
    status: str,
    score_delta: float = 0.0,
    output_similarity: Optional[float] = None,
    tool_changes: int = 0,
    model_changed: bool = False,
    diff_json: Optional[Dict[str, Any]] = None,
    tools_added: Optional[List[str]] = None,
    tools_removed: Optional[List[str]] = None,
) -> List[Recommendation]:
    """Generate recommendations from observed failure signals."""
    recs: List[Recommendation] = []
    dj = diff_json or {}
    added = tools_added or dj.get("tools_added") or []
    removed = tools_removed or dj.get("tools_removed") or []

    if status == "passed":
        return recs

    # ── Tool additions / removals ──
    if added or removed:
        tool_names = ", ".join(added + removed)
        parts: List[str] = []
        if added:
            parts.append(f"started using {', '.join(added)}")
        if removed:
            parts.append(f"stopped using {', '.join(removed)}")
        recs.append(Recommendation(
            action=f"Review tool descriptions for: {tool_names}",
            confidence="high",
            category="tool",
            detail=(
                f"The agent {' and '.join(parts)}. "
                "Check if tool descriptions clearly specify when each tool should be used."
            ),
            likely_cause=(
                "Tool selection changed — usually a prompt or tool-description edit"
                " nudged the model toward a different execution path."
            ),
            severity="high",
            suggested_commands=[
                "evalview replay --trace",
                "evalview golden update <test>   # if the new path is correct",
            ],
        ))

    if tool_changes > 3:
        recs.append(Recommendation(
            action="Add step-level assertions for critical tool sequences",
            confidence="high",
            category="tool",
            detail=(
                f"{tool_changes} tool calls differ from baseline — the agent's reasoning "
                "path changed significantly. Add expected_tools assertions to lock down "
                "the critical path."
            ),
            likely_cause=(
                f"Large tool-sequence delta ({tool_changes} changes) — reasoning path"
                " shifted beyond normal variance."
            ),
            severity="high",
            suggested_commands=[
                "evalview replay --trace",
                "evalview check --statistical 5   # confirm it's not flake",
            ],
        ))

    # ── Score + output quality ──
    sim = output_similarity
    if score_delta < -5 and sim is not None and sim < 0.5:
        recs.append(Recommendation(
            action="Check if the prompt lost critical instructions",
            confidence="high",
            category="prompt",
            detail=(
                f"Output is only {int(sim * 100)}% similar to baseline and quality "
                f"dropped {abs(score_delta):.1f} points. This usually means the prompt "
                "is missing key context, examples, or constraints that guided the agent."
            ),
            likely_cause=(
                f"Large divergence (sim={int(sim * 100)}%, score -{abs(score_delta):.1f}):"
                " prompt likely lost critical context or constraints."
            ),
            severity="high",
            suggested_commands=[
                "git diff HEAD~1 -- prompts/   # what changed in the prompt?",
                "evalview replay <test>",
            ],
        ))
    elif score_delta < -3 and sim is not None and sim > 0.8:
        recs.append(Recommendation(
            action="Check if examples or few-shot formatting changed",
            confidence="medium",
            category="prompt",
            detail=(
                f"Output is {int(sim * 100)}% similar (phrasing held) but quality "
                f"dropped {abs(score_delta):.1f} points. This pattern often means "
                "few-shot examples, formatting instructions, or output schema changed subtly."
            ),
            likely_cause=(
                "High textual similarity but lower quality — usually a subtle"
                " few-shot or schema-formatting change."
            ),
            severity="medium",
            suggested_commands=[
                "evalview replay <test>",
            ],
        ))

    # ── Model drift ──
    if model_changed:
        recs.append(Recommendation(
            action="Pin your model version to avoid upstream drift",
            confidence="high",
            category="model",
            detail=(
                "The model version changed between runs. Pin the exact model ID "
                "(e.g. gpt-4o-2024-08-06 instead of gpt-4o) to prevent silent "
                "behavior changes from provider updates."
            ),
            likely_cause=(
                "Provider silently updated the model between runs."
                " This is drift you can't see in your own repo."
            ),
            severity="high",
            suggested_commands=[
                "evalview check --statistical 5   # confirm variance before pinning",
            ],
        ))

    # ── Hallucination ──
    hall_score = dj.get("hallucination_score")
    if hall_score is not None and hall_score > 0.5:
        recs.append(Recommendation(
            action="Add grounding constraints or retrieval verification",
            confidence="high",
            category="guardrail",
            detail=(
                f"Hallucination confidence is {int(hall_score * 100)}%. Add explicit "
                'grounding instructions ("Only use information from the provided '
                'context") or verify tool results before generating the final answer.'
            ),
            likely_cause=(
                "Model is generating content unsupported by the tool results"
                " it received."
            ),
            severity="high",
            suggested_commands=[
                "evalview replay <test>   # inspect tool outputs vs final answer",
            ],
        ))
    elif hall_score is not None and hall_score > 0.2:
        recs.append(Recommendation(
            action="Review agent output for ungrounded claims",
            confidence="medium",
            category="guardrail",
            detail=(
                f"Possible hallucination detected ({int(hall_score * 100)}% confidence). "
                "Check if the agent's claims are supported by tool results."
            ),
            likely_cause="Borderline hallucination signal — worth a spot-check.",
            severity="medium",
            suggested_commands=["evalview replay <test>"],
        ))

    # ── Cost spike ──
    cost_delta = dj.get("cost_delta") or dj.get("cost_diff")
    if cost_delta is not None and cost_delta > 0.01:
        recs.append(Recommendation(
            action="Trim context or add max_tokens to control cost",
            confidence="medium",
            category="config",
            detail=(
                f"Cost increased by ${cost_delta:.4f}. Consider adding max_tokens, "
                "trimming conversation history, or reducing retrieval chunk size."
            ),
            likely_cause=(
                f"Cost grew by ${cost_delta:.4f} per run — usually larger context"
                " windows, longer tool loops, or a more expensive model."
            ),
            severity="medium",
            suggested_commands=[
                "evalview check --statistical 3   # confirm the spike is real",
            ],
        ))

    # ── Safety ──
    safety_score = dj.get("safety_score")
    if safety_score is not None and safety_score < 0.5:
        recs.append(Recommendation(
            action="Add safety guardrails to the system prompt",
            confidence="high",
            category="guardrail",
            detail=(
                f"Safety score is {int(safety_score * 100)}%. Add explicit safety "
                "instructions and output filtering to prevent harmful content."
            ),
            likely_cause="Safety evaluator flagged the output.",
            severity="high",
            suggested_commands=["evalview replay <test>"],
        ))

    # ── PII ──
    if dj.get("pii_detected"):
        recs.append(Recommendation(
            action="Add PII filtering to the output pipeline",
            confidence="high",
            category="guardrail",
            detail=(
                "PII was detected in the agent's output. Add a post-processing step "
                "to scrub or redact sensitive data before returning to the user."
            ),
            likely_cause="PII detector matched the agent's output.",
            severity="high",
            suggested_commands=["evalview replay <test>"],
        ))

    # ── Fallback ──
    if not recs and status in ("regression", "tools_changed", "output_changed"):
        if sim is not None and sim < 0.7:
            recs.append(Recommendation(
                action="Compare the full output diff to identify what changed",
                confidence="low",
                category="prompt",
                detail=(
                    f"Output is {int(sim * 100)}% similar with no clear tool or model "
                    "trigger. Review the full diff for prompt formatting issues, "
                    "context truncation, or retrieval changes."
                ),
                likely_cause="Unknown — signals don't point to a single cause.",
                severity="medium",
                suggested_commands=[
                    "evalview replay <test>",
                    "evalview check --statistical 5",
                ],
            ))
        else:
            recs.append(Recommendation(
                action="Re-run the test to check for non-determinism",
                confidence="low",
                category="config",
                detail=(
                    "No specific failure signal identified. Re-run with --statistical "
                    "to check if this is LLM variance rather than a real regression."
                ),
                likely_cause="Possible LLM variance rather than a real change.",
                severity="low",
                suggested_commands=[
                    "evalview check --statistical 5",
                ],
            ))

    return recs


def recommend_from_trace_diff(diff: Any) -> List[Recommendation]:
    """Generate recommendations from a TraceDiff object (CLI integration)."""
    tools_added: List[str] = []
    tools_removed: List[str] = []
    tool_changes = 0

    if hasattr(diff, "tool_diffs") and diff.tool_diffs:
        tool_changes = len(diff.tool_diffs)
        for td in diff.tool_diffs:
            if td.type == "added" and td.actual_tool:
                tools_added.append(td.actual_tool)
            elif td.type == "removed" and td.golden_tool:
                tools_removed.append(td.golden_tool)

    output_similarity = None
    if hasattr(diff, "output_diff") and diff.output_diff:
        output_similarity = diff.output_diff.similarity

    score_delta = getattr(diff, "score_diff", 0.0)
    model_changed = getattr(diff, "model_changed", False)

    status = "passed"
    severity = getattr(diff, "overall_severity", None)
    if severity:
        status = severity.value if hasattr(severity, "value") else str(severity)

    return recommend(
        status=status,
        score_delta=score_delta,
        output_similarity=output_similarity,
        tool_changes=tool_changes,
        model_changed=model_changed,
        tools_added=tools_added,
        tools_removed=tools_removed,
    )
