"""PR comment generation for CI integration.

Generates clean, scannable PR comments with:
- Overall status (PASSED / REGRESSION / TOOLS_CHANGED / OUTPUT_CHANGED)
- Summary metrics (tests, pass rate, cost, latency)
- Top changes when using diff mode
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime


def load_latest_results(results_dir: str = ".evalview/results") -> Optional[Dict[str, Any]]:
    """Load the most recent results file."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    json_files = sorted(results_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not json_files:
        return None

    with open(json_files[0]) as f:
        return json.load(f)


def get_status_emoji(status: str) -> str:
    """Get emoji for diff status."""
    return {
        "passed": "\u2705",  # ✅
        "regression": "\u274c",  # ❌
        "tools_changed": "\u26a0\ufe0f",  # ⚠️
        "output_changed": "\u26a0\ufe0f",  # ⚠️
    }.get(status.lower(), "\u2753")  # ❓


def get_status_label(status: str) -> str:
    """Get human-readable label for status."""
    return {
        "passed": "PASSED",
        "regression": "REGRESSION",
        "tools_changed": "TOOLS CHANGED",
        "output_changed": "OUTPUT CHANGED",
    }.get(status.lower(), status.upper())


def format_cost(cost: float) -> str:
    """Format cost value."""
    if cost == 0:
        return "$0.00"
    elif cost < 0.01:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def format_latency(ms: float) -> str:
    """Format latency in human-readable form."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    else:
        return f"{ms/1000:.1f}s"


def format_delta(current: float, baseline: float, is_cost: bool = False) -> str:
    """Format delta with arrow."""
    if baseline == 0:
        return ""

    delta = current - baseline
    pct = (delta / baseline) * 100 if baseline != 0 else 0

    if abs(pct) < 1:
        return ""

    arrow = "\u2191" if delta > 0 else "\u2193"  # ↑ or ↓
    sign = "+" if delta > 0 else ""

    # For cost, up is bad. For others, context dependent
    if is_cost:
        color = "red" if delta > 0 else "green"
    else:
        color = ""

    return f" ({sign}{pct:.0f}%{arrow})"


def generate_pr_comment(
    results: List[Dict[str, Any]],
    diff_results: Optional[List[Dict[str, Any]]] = None,
    run_url: Optional[str] = None,
) -> str:
    """Generate markdown PR comment from results.

    Args:
        results: List of EvaluationResult dicts
        diff_results: Optional list of TraceDiff dicts (from --diff mode)
        run_url: Optional link to the GitHub Actions run

    Returns:
        Markdown string for PR comment
    """
    if not results:
        return "## EvalView Results\n\nNo test results found."

    # Calculate summary stats
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    # Calculate totals
    total_cost = sum(r.get("trace", {}).get("metrics", {}).get("total_cost", 0) for r in results)
    total_latency = sum(r.get("trace", {}).get("metrics", {}).get("total_latency", 0) for r in results)
    avg_score = sum(r.get("score", 0) for r in results) / total if total > 0 else 0

    # Determine overall status
    if diff_results:
        # Use diff status
        statuses = [d.get("overall_severity", "passed") for d in diff_results]
        if "regression" in statuses:
            overall_status = "regression"
        elif "tools_changed" in statuses:
            overall_status = "tools_changed"
        elif "output_changed" in statuses:
            overall_status = "output_changed"
        else:
            overall_status = "passed"
    else:
        overall_status = "passed" if failed == 0 else "regression"

    emoji = get_status_emoji(overall_status)
    label = get_status_label(overall_status)

    # Build comment
    lines = []

    # Header with status
    lines.append(f"## {emoji} EvalView: {label}")
    lines.append("")

    # Summary table
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Tests | {passed}/{total} passed ({pass_rate:.0f}%) |")
    lines.append(f"| Avg Score | {avg_score:.1f}/100 |")
    lines.append(f"| Total Cost | {format_cost(total_cost)} |")
    lines.append(f"| Total Latency | {format_latency(total_latency)} |")
    lines.append("")

    # Diff details (if available)
    if diff_results:
        changes = [d for d in diff_results if d.get("has_differences", False)]
        if changes:
            lines.append("### Changes from Baseline")
            lines.append("")

            # Show top 5 changes
            for diff in changes[:5]:
                test_name = diff.get("test_name", "Unknown")
                status = diff.get("overall_severity", "passed")
                status_emoji = get_status_emoji(status)

                summary_parts = []

                # Score change
                score_diff = diff.get("score_diff", 0)
                if abs(score_diff) > 1:
                    direction = "+" if score_diff > 0 else ""
                    summary_parts.append(f"score {direction}{score_diff:.1f}")

                # Tool changes
                tool_diffs = diff.get("tool_diffs", [])
                if tool_diffs:
                    summary_parts.append(f"{len(tool_diffs)} tool change(s)")

                # Latency change
                latency_diff = diff.get("latency_diff", 0)
                if abs(latency_diff) > 100:  # >100ms change
                    direction = "+" if latency_diff > 0 else ""
                    summary_parts.append(f"latency {direction}{latency_diff:.0f}ms")

                summary = ", ".join(summary_parts) if summary_parts else "minor changes"
                lines.append(f"- {status_emoji} **{test_name}**: {summary}")

            if len(changes) > 5:
                lines.append(f"- ... and {len(changes) - 5} more")

            lines.append("")

    # Failed tests (if no diff mode)
    if not diff_results and failed > 0:
        lines.append("### Failed Tests")
        lines.append("")

        failed_tests = [r for r in results if not r.get("passed", False)]
        for r in failed_tests[:5]:
            name = r.get("test_case", "Unknown")
            score = r.get("score", 0)
            min_score = r.get("min_score", 70)
            lines.append(f"- \u274c **{name}**: score {score:.1f} (min: {min_score})")

        if len(failed_tests) > 5:
            lines.append(f"- ... and {len(failed_tests) - 5} more")

        lines.append("")

    # Footer
    if run_url:
        lines.append(f"[View full report]({run_url})")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by [EvalView](https://github.com/hidai25/eval-view)*")

    return "\n".join(lines)


def post_pr_comment(comment: str, pr_number: Optional[int] = None) -> bool:
    """Post comment to PR using gh CLI.

    Args:
        comment: Markdown comment to post
        pr_number: PR number (auto-detected from GITHUB_REF if not provided)

    Returns:
        True if comment was posted successfully
    """
    # Get PR number from environment if not provided
    if pr_number is None:
        github_ref = os.environ.get("GITHUB_REF", "")
        # refs/pull/123/merge -> 123
        if "/pull/" in github_ref:
            try:
                pr_number = int(github_ref.split("/pull/")[1].split("/")[0])
            except (IndexError, ValueError):
                return False
        else:
            # Not a PR context
            return False

    # Check if gh CLI is available
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    # Post comment
    try:
        result = subprocess.run(
            ["gh", "pr", "comment", str(pr_number), "--body", comment],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def update_or_create_comment(comment: str, pr_number: Optional[int] = None) -> bool:
    """Update existing EvalView comment or create new one.

    This prevents comment spam by updating the existing comment.

    Args:
        comment: Markdown comment to post
        pr_number: PR number (auto-detected if not provided)

    Returns:
        True if comment was posted/updated successfully
    """
    # Get PR number from environment if not provided
    if pr_number is None:
        github_ref = os.environ.get("GITHUB_REF", "")
        if "/pull/" in github_ref:
            try:
                pr_number = int(github_ref.split("/pull/")[1].split("/")[0])
            except (IndexError, ValueError):
                return False
        else:
            return False

    # Try to find existing comment
    try:
        result = subprocess.run(
            ["gh", "pr", "view", str(pr_number), "--json", "comments"],
            capture_output=True,
            text=True,
            check=True,
        )
        comments_data = json.loads(result.stdout)
        comments = comments_data.get("comments", [])

        # Find our comment (look for the signature)
        evalview_comment = None
        for c in comments:
            if "*Generated by [EvalView]" in c.get("body", ""):
                evalview_comment = c
                break

        if evalview_comment:
            # Update existing comment
            comment_url = evalview_comment.get("url", "")
            if comment_url:
                # Extract comment ID from URL
                # https://github.com/owner/repo/pull/123#issuecomment-456
                if "#issuecomment-" in comment_url:
                    comment_id = comment_url.split("#issuecomment-")[1]
                    subprocess.run(
                        ["gh", "api", "-X", "PATCH",
                         f"/repos/:owner/:repo/issues/comments/{comment_id}",
                         "-f", f"body={comment}"],
                        capture_output=True,
                        check=True,
                    )
                    return True

        # No existing comment, create new one
        return post_pr_comment(comment, pr_number)

    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        # Fall back to creating new comment
        return post_pr_comment(comment, pr_number)
