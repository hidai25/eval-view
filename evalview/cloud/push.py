"""Best-effort result push to EvalView Cloud.

NOT truly fire-and-forget: push_result() blocks the CLI synchronously
via asyncio.run() for up to ~15s (3 retries x 15s timeout, with backoff).
This is acceptable because:
  - Successful pushes complete in <1s (single POST)
  - Retries only trigger on transient failures (rare)
  - The CLI has already displayed all results before push runs
  - Auth/billing errors fail immediately (no retry)
"""

import asyncio
import hashlib
import logging
import os
import subprocess
from typing import Any, Dict, Optional

import httpx

from evalview.core.types import SCHEMA_VERSION

logger = logging.getLogger(__name__)

CLOUD_API_URL = os.environ.get(
    "EVALVIEW_CLOUD_URL", "http://localhost:3000/api/v1"
)


def _get_api_token() -> Optional[str]:
    """Resolve API token from env or config."""
    token = os.environ.get("EVALVIEW_API_TOKEN")
    if token:
        return token
    try:
        from evalview.commands.shared import _load_config_if_exists
        config = _load_config_if_exists()
        return getattr(getattr(config, "cloud", None), "api_token", None) if config else None
    except Exception:
        return None


def _get_git_context() -> Dict[str, Any]:
    """Best-effort git metadata. Never fails."""
    ctx: Dict[str, Any] = {}
    try:
        ctx["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()[:40]
    except Exception:
        pass
    try:
        ctx["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    for env_var in ("GITHUB_PR_NUMBER", "CI_MERGE_REQUEST_IID", "PULL_REQUEST_NUMBER"):
        val = os.environ.get(env_var)
        if val:
            try:
                ctx["git_pr"] = int(val)
            except ValueError:
                pass
            break
    return ctx


def _build_diff_json(d: Any) -> Optional[Dict[str, Any]]:
    """Extract rich diff data from a TestDiff for cloud display.

    Pulls tool diffs, root cause, healing status, and output comparison
    from the raw TraceDiff so the cloud dashboard can show behavioral
    analysis without recomputing.
    """
    try:
        raw = getattr(d, "raw", None)
        if raw is None:
            return None

        result: Dict[str, Any] = {}

        # Tool diffs
        tool_diffs = getattr(raw, "tool_diffs", [])
        if tool_diffs:
            tools_added = []
            tools_removed = []
            tools_changed_list = []
            for td in tool_diffs:
                if td.type == "missing":
                    tools_removed.append(td.golden_tool or "unknown")
                elif td.type == "extra":
                    tools_added.append(td.actual_tool or "unknown")
                elif td.type == "mismatch":
                    tools_changed_list.append({
                        "position": td.position,
                        "expected": td.golden_tool,
                        "actual": td.actual_tool,
                        "params_changed": len(td.parameter_diffs) if td.parameter_diffs else 0,
                    })
            if tools_added:
                result["tools_added"] = tools_added
            if tools_removed:
                result["tools_removed"] = tools_removed
            if tools_changed_list:
                result["tools_changed"] = tools_changed_list

        # Output diff
        output_diff = getattr(raw, "output_diff", None)
        if output_diff:
            result["output_similarity"] = getattr(output_diff, "similarity", None)
            semantic_sim = getattr(output_diff, "semantic_similarity", None)
            if semantic_sim is not None:
                result["semantic_similarity"] = semantic_sim

        # Model/runtime info
        if getattr(raw, "model_changed", False):
            result["model_changed"] = True
            result["golden_model"] = getattr(raw, "golden_model_id", None)
            result["actual_model"] = getattr(raw, "actual_model_id", None)

        # Root cause (if computed)
        root_cause = getattr(raw, "root_cause", None)
        if root_cause is not None:
            rc_dict = root_cause.to_dict() if hasattr(root_cause, "to_dict") else None
            if rc_dict:
                result["root_cause"] = rc_dict

        # Evaluator scores (judge, hallucination, PII, safety)
        scores = getattr(raw, "evaluator_scores", None) or getattr(raw, "scores", None)
        if scores and isinstance(scores, dict):
            result["evaluator_scores"] = scores

        # Judge info
        judge_info = getattr(raw, "judge_info", None)
        if judge_info and isinstance(judge_info, dict):
            result["judge"] = judge_info
        else:
            judge_model = getattr(raw, "judge_model", None)
            if judge_model:
                result["judge"] = {"model": judge_model}

        # Hallucination score
        hallucination = getattr(raw, "hallucination_score", None)
        if hallucination is not None:
            result["hallucination_score"] = hallucination

        # PII detection
        pii = getattr(raw, "pii_detected", None)
        if pii is not None:
            result["pii_detected"] = pii

        # Safety score
        safety = getattr(raw, "safety_score", None)
        if safety is not None:
            result["safety_score"] = safety

        # Healing status
        healed = getattr(d, "healed", None) or getattr(raw, "healed", None)
        if healed is not None:
            result["healed"] = healed
        healing_status = getattr(d, "healing_status", None) or getattr(raw, "healing_status", None)
        if healing_status:
            result["healing_status"] = healing_status

        return result if result else None
    except Exception:
        return None


async def _push_async(payload: Dict[str, Any], token: str) -> Optional[str]:
    """Push with 3 retries, exponential backoff. Returns dashboard URL or None."""
    return await _push_to_url(f"{CLOUD_API_URL}/results", payload, token)


def push_comparison(results: Any, query: str, threshold: float = 0.8) -> Optional[str]:
    """Push compare_models() results to EvalView Cloud. Best-effort, blocking.

    Args:
        results: List[ModelResult] returned by compare_models().
        query: The query that was evaluated.
        threshold: The pass/fail threshold used.

    Returns:
        Dashboard URL if successful, None otherwise.
    """
    token = _get_api_token()
    if not token:
        return None

    try:
        git = _get_git_context()
        source = "ci" if os.environ.get("CI") else "cli"

        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        best = sorted_results[0].model if sorted_results else None

        payload = {
            "query": query,
            "models": [r.model for r in results],
            "results": [
                {
                    "model": r.model,
                    "output": (r.output or "")[:4096],
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                    "passed": r.passed,
                    "error": r.error,
                    "metadata": r.metadata or {},
                }
                for r in results
            ],
            "best_model": best,
            "threshold": threshold,
            "source": source,
            **git,
        }

        url = f"{CLOUD_API_URL}/comparisons"
        return asyncio.run(_push_to_url(url, payload, token))
    except Exception as e:
        logger.debug("Cloud comparison push failed: %s", e)
        return None


async def _push_to_url(url: str, payload: Dict[str, Any], token: str) -> Optional[str]:
    """Generic push with 3 retries, exponential backoff.

    Sends ``X-EvalView-Schema`` so the cloud can branch on wire-format
    version. Clouds that only understand v1 should ignore the header
    and safely drop fields they don't recognize.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-EvalView-Schema": str(SCHEMA_VERSION),
    }
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in (200, 201):
                    return resp.json().get("dashboard_url")
                if resp.status_code in (401, 402, 403):
                    logger.debug("Cloud push auth/billing failed: %s", resp.status_code)
                    return None
                logger.debug("Cloud push failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.debug("Cloud push error (attempt %d): %s", attempt + 1, e)
        if attempt < 2:
            await asyncio.sleep(2 ** attempt)
    return None


def push_result(gate_result: Any) -> Optional[str]:
    """Push a GateResult to cloud. Best-effort, blocking.

    Runs synchronously after check results are displayed.
    Returns dashboard URL if successful, None otherwise.
    Never raises. Blocks for at most ~15s (3 retries with backoff).
    """
    token = _get_api_token()
    if not token:
        return None

    try:
        git = _get_git_context()
        source = "ci" if os.environ.get("CI") else "cli"

        # Discriminator lets cloud route simulation runs to the
        # /runs/[id]/simulation tab without guessing from the body.
        # Cloud's Zod accepts "standard" | "simulation"; we send
        # "standard" (the older "check" alias still validates via a
        # cloud-side compatibility shim, but new pushes use the
        # canonical name).
        run_type = "simulation" if gate_result.raw_json.get("simulation") else "standard"

        payload = {
            "run_id": gate_result.raw_json.get("run_id", hashlib.md5(
                str(gate_result.raw_json).encode()
            ).hexdigest()[:8]),
            "run_type": run_type,
            "schema_version": SCHEMA_VERSION,
            "status": gate_result.status.value,
            "source": source,
            **git,
            "summary": {
                "total": gate_result.summary.total,
                "unchanged": gate_result.summary.unchanged,
                "regressions": gate_result.summary.regressions,
                "tools_changed": gate_result.summary.tools_changed,
                "output_changed": gate_result.summary.output_changed,
            },
            "total_cost": gate_result.raw_json.get("total_cost", 0),
            "total_latency_ms": gate_result.raw_json.get("total_latency_ms", 0),
            "diffs": [
                {
                    "test_name": d.test_name,
                    "status": d.status.value,
                    "score_delta": d.score_delta,
                    "output_similarity": d.output_similarity,
                    "tool_changes": d.tool_changes,
                    "model_changed": d.model_changed,
                    "diff_json": _build_diff_json(d),
                }
                for d in gate_result.diffs
            ],
            "result_json": gate_result.raw_json,
        }

        # Observability — sent in three complementary shapes so cloud
        # populates both the per-test detail panels and the aggregate
        # columns on the runs row:
        #
        #   1. Per-test arrays at the top level. These keys
        #      (`behavioral_anomalies`, `trust_scores`,
        #      `coherence_analysis`) are what cloud's /api/v1/results
        #      route reads to fill test_diffs.anomaly_report,
        #      trust_score, coherence_score, coherence_report. Names
        #      come from check_display.py's --json output; we forward
        #      them verbatim out of raw_json.
        #   2. Aggregate count blocks at the top level
        #      (`low_trust_tests`, `coherence_issues`) so cloud's
        #      runs.low_trust_count / coherence_issue_count columns
        #      populate without re-walking the per-test arrays.
        #   3. The legacy `observability` envelope is still attached
        #      for older cloud routes / digest renderers that read it.
        for key in ("behavioral_anomalies", "trust_scores", "coherence_analysis"):
            value = gate_result.raw_json.get(key)
            if value:
                payload[key] = value

        obs = gate_result.observability
        if obs.low_trust_count:
            payload["low_trust_tests"] = {
                "count": obs.low_trust_count,
                "tests": obs.low_trust_tests[:10],
            }
        if obs.coherence_issue_count:
            payload["coherence_issues"] = {
                "count": obs.coherence_issue_count,
                "tests": obs.coherence_tests[:10],
            }

        obs_payload = obs.to_payload()
        if obs_payload:
            payload["observability"] = obs_payload

        # Schema v2 fields. Cloud stores these in dedicated tables
        # (simulations, rationale_events). Both optional — agents and
        # adapters that don't capture them simply omit the keys.
        simulation = gate_result.raw_json.get("simulation")
        if simulation:
            payload["simulation"] = simulation
        rationale_events = gate_result.raw_json.get("rationale_events")
        if rationale_events:
            payload["rationale_events"] = rationale_events

        return asyncio.run(_push_async(payload, token))
    except Exception as e:
        logger.debug("Cloud push failed: %s", e)
        return None
