"""Slack webhook notifier for EvalView monitor alerts."""

from typing import Any, Dict, List, Tuple
import logging

import httpx

logger = logging.getLogger(__name__)


class SlackNotifier:
    """Send regression alerts to Slack via incoming webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send_regression_alert(
        self,
        diffs: List[Tuple[str, Any]],
        analysis: Dict[str, Any],
    ) -> bool:
        """Send a regression alert to Slack.

        Args:
            diffs: List of (test_name, TraceDiff) tuples.
            analysis: Output of _analyze_check_diffs.

        Returns:
            True if the message was sent successfully.
        """
        from evalview.core.diff import DiffStatus

        # Build the list of failing tests
        failing = []
        for name, diff in diffs:
            if diff.overall_severity == DiffStatus.REGRESSION:
                score_part = f" (score {diff.score_diff:+.1f})" if diff.score_diff is not None else ""
                failing.append(f":red_circle: *{name}* — REGRESSION{score_part}")
            elif diff.overall_severity == DiffStatus.TOOLS_CHANGED:
                failing.append(f":large_orange_circle: *{name}* — TOOLS_CHANGED")
            elif diff.overall_severity == DiffStatus.OUTPUT_CHANGED:
                failing.append(f":white_circle: *{name}* — OUTPUT_CHANGED")

        if not failing:
            return True  # Nothing to report

        total = len(diffs)
        passed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED)

        text = (
            f":warning: *EvalView Monitor — Regression Detected*\n\n"
            f"{passed}/{total} tests passing\n\n"
            + "\n".join(failing)
            + "\n\n_Run `evalview check` for full details._"
        )

        payload = {"text": text}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.webhook_url, json=payload)
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.warning("Slack notification failed: %s", e)
            return False

    async def send_recovery_alert(self, total_tests: int) -> bool:
        """Send a recovery notification when all tests pass again."""
        payload = {
            "text": (
                f":white_check_mark: *EvalView Monitor — All Clear*\n\n"
                f"All {total_tests} tests passing. Regression resolved."
            )
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.webhook_url, json=payload)
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.warning("Slack notification failed: %s", e)
            return False
