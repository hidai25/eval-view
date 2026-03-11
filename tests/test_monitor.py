"""Tests for the monitor command and Slack notifier."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from evalview.commands.monitor_cmd import (
    MonitorError,
    _resolve_slack_webhook,
    _run_monitor_loop,
    _sleep_interruptible,
)
from evalview.core.slack_notifier import SlackNotifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(base: Path, adapter: str = "http", endpoint: str = "http://example.com") -> None:
    config_dir = base / ".evalview"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.dump({"adapter": adapter, "endpoint": endpoint}))


def _make_fake_trace():
    from evalview.core.types import ExecutionTrace, ExecutionMetrics
    return ExecutionTrace(
        session_id="s1",
        start_time=datetime.now(),
        end_time=datetime.now(),
        steps=[],
        final_output="The answer is 42.",
        metrics=ExecutionMetrics(total_cost=0.01, total_latency=100.0),
    )


def _make_fake_result(test_name: str, score: float = 90.0):
    from evalview.core.types import EvaluationResult
    result = MagicMock(spec=EvaluationResult)
    result.test_case = test_name
    result.score = score
    result.passed = score >= 70
    result.trace = _make_fake_trace()
    return result


def _write_golden(base: Path, test_name: str) -> None:
    from evalview.core.golden import GoldenStore
    store = GoldenStore(base_path=base)
    fake_result = MagicMock()
    fake_result.test_case = test_name
    fake_result.score = 90.0
    fake_result.passed = True
    fake_result.trace = _make_fake_trace()
    store.save_golden(fake_result)


def _write_test_yaml(test_dir: Path, name: str) -> None:
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / f"{name}.yaml").write_text(
        yaml.dump({
            "name": name,
            "input": {"query": "hello"},
            "expected": {"tools": []},
            "thresholds": {"min_score": 0},
        })
    )


def _make_diff(status: str, score_diff: float = 0.0):
    """Create a minimal TraceDiff with the given status.

    Accepts both upper and lowercase status strings (e.g. "REGRESSION" or "regression").
    """
    from evalview.core.diff import TraceDiff, DiffStatus, OutputDiff

    severity = DiffStatus(status.lower())
    return TraceDiff(
        test_name="test",
        has_differences=(severity != DiffStatus.PASSED),
        tool_diffs=[],
        output_diff=OutputDiff(
            similarity=1.0 if severity == DiffStatus.PASSED else 0.5,
            golden_preview="",
            actual_preview="",
            diff_lines=[],
            severity=severity,
        ),
        score_diff=score_diff,
        latency_diff=0.0,
        overall_severity=severity,
    )


# ---------------------------------------------------------------------------
# Webhook resolution
# ---------------------------------------------------------------------------

class TestResolveSlackWebhook:
    """Test webhook URL priority: CLI flag > config > env var."""

    def test_cli_flag_wins(self):
        config = MagicMock()
        config.get_monitor_config.return_value.slack_webhook = "https://config-url"
        result = _resolve_slack_webhook("https://cli-url", config)
        assert result == "https://cli-url"

    def test_config_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("EVALVIEW_SLACK_WEBHOOK", "https://env-url")
        config = MagicMock()
        config.get_monitor_config.return_value.slack_webhook = "https://config-url"
        result = _resolve_slack_webhook(None, config)
        assert result == "https://config-url"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("EVALVIEW_SLACK_WEBHOOK", "https://env-url")
        result = _resolve_slack_webhook(None, None)
        assert result == "https://env-url"

    def test_returns_none_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("EVALVIEW_SLACK_WEBHOOK", raising=False)
        result = _resolve_slack_webhook(None, None)
        assert result is None

    def test_config_with_no_webhook_falls_through(self, monkeypatch):
        monkeypatch.setenv("EVALVIEW_SLACK_WEBHOOK", "https://env-url")
        config = MagicMock()
        config.get_monitor_config.return_value.slack_webhook = None
        result = _resolve_slack_webhook(None, config)
        assert result == "https://env-url"


# ---------------------------------------------------------------------------
# MonitorError on missing prerequisites
# ---------------------------------------------------------------------------

class TestMonitorPrerequisites:
    """Monitor should raise MonitorError when baselines or tests are missing."""

    def test_no_baselines_raises_error(self, tmp_path, monkeypatch):
        _write_config(tmp_path)
        _write_test_yaml(tmp_path / "tests", "my-test")
        # No golden written
        monkeypatch.chdir(tmp_path)

        with pytest.raises(MonitorError, match="No baselines found"):
            _run_monitor_loop(
                test_path="tests",
                interval=10,
                slack_webhook=None,
                fail_on="REGRESSION",
                timeout=30.0,
                test_filter=None,
            )

    def test_missing_test_filter_raises_error(self, tmp_path, monkeypatch):
        _write_config(tmp_path)
        _write_test_yaml(tmp_path / "tests", "my-test")
        _write_golden(tmp_path, "my-test")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(MonitorError, match="No test found"):
            _run_monitor_loop(
                test_path="tests",
                interval=10,
                slack_webhook=None,
                fail_on="REGRESSION",
                timeout=30.0,
                test_filter="nonexistent-test",
            )


# ---------------------------------------------------------------------------
# Sleep interruptible
# ---------------------------------------------------------------------------

class TestSleepInterruptible:
    """Test the interruptible sleep utility."""

    def test_stops_early_when_flag_set(self):
        """Should return in ~0s when should_stop returns True immediately."""
        import time
        start = time.monotonic()
        _sleep_interruptible(100, lambda: True)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0  # Should be near-instant

    def test_sleeps_for_duration(self):
        """Should sleep for approximately the given duration."""
        import time
        start = time.monotonic()
        _sleep_interruptible(2, lambda: False)
        elapsed = time.monotonic() - start
        assert elapsed >= 1.5  # Allow some slack


# ---------------------------------------------------------------------------
# Slack Notifier
# ---------------------------------------------------------------------------

class TestSlackNotifier:
    """Test Slack notification formatting and delivery."""

    def test_regression_alert_includes_test_names(self):
        from evalview.core.diff import DiffStatus
        notifier = SlackNotifier("https://hooks.slack.com/test")

        diff_reg = _make_diff("REGRESSION", score_diff=-15.0)
        diff_tool = _make_diff("TOOLS_CHANGED")
        diffs = [("auth-flow", diff_reg), ("search-api", diff_tool)]
        analysis = {"has_regressions": True, "has_tools_changed": True}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = asyncio.run(notifier.send_regression_alert(diffs, analysis))

        assert result is True
        # Verify the posted payload contains test names
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "auth-flow" in payload["text"]
        assert "search-api" in payload["text"]
        assert "REGRESSION" in payload["text"]

    def test_regression_alert_handles_none_score_diff(self):
        """score_diff=None should not crash the formatter."""
        notifier = SlackNotifier("https://hooks.slack.com/test")

        diff = _make_diff("REGRESSION")
        diff.score_diff = None
        diffs = [("my-test", diff)]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            # Should not raise
            result = asyncio.run(notifier.send_regression_alert(diffs, {}))

        assert result is True

    def test_recovery_alert_message(self):
        notifier = SlackNotifier("https://hooks.slack.com/test")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = asyncio.run(notifier.send_recovery_alert(5))

        assert result is True
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "All Clear" in payload["text"]
        assert "5" in payload["text"]

    def test_empty_diffs_returns_true(self):
        """No failing tests should return True without sending."""
        notifier = SlackNotifier("https://hooks.slack.com/test")
        diff = _make_diff("PASSED")
        result = asyncio.run(notifier.send_regression_alert([("ok", diff)], {}))
        assert result is True

    def test_network_failure_returns_false(self):
        """Network errors should be caught and return False."""
        notifier = SlackNotifier("https://hooks.slack.com/test")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Network down"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = asyncio.run(notifier.send_recovery_alert(3))

        assert result is False


# ---------------------------------------------------------------------------
# MonitorConfig
# ---------------------------------------------------------------------------

class TestMonitorConfig:
    """Test MonitorConfig defaults and validation."""

    def test_defaults(self):
        from evalview.core.config import MonitorConfig
        cfg = MonitorConfig()
        assert cfg.interval == 300
        assert cfg.slack_webhook is None
        assert cfg.fail_on == ["REGRESSION"]
        assert cfg.timeout == 30.0

    def test_custom_values(self):
        from evalview.core.config import MonitorConfig
        cfg = MonitorConfig(interval=60, timeout=120.0, fail_on=["REGRESSION", "TOOLS_CHANGED"])
        assert cfg.interval == 60
        assert cfg.timeout == 120.0
        assert len(cfg.fail_on) == 2

    def test_minimum_interval(self):
        from evalview.core.config import MonitorConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MonitorConfig(interval=5)  # Below minimum of 10

    def test_evalview_config_has_monitor(self):
        from evalview.core.config import EvalViewConfig, MonitorConfig
        config = EvalViewConfig(
            adapter="http",
            endpoint="http://example.com",
            monitor=MonitorConfig(interval=60),
        )
        assert config.get_monitor_config().interval == 60

    def test_evalview_config_monitor_defaults(self):
        from evalview.core.config import EvalViewConfig
        config = EvalViewConfig(adapter="http", endpoint="http://example.com")
        cfg = config.get_monitor_config()
        assert cfg.interval == 300  # Default


# ---------------------------------------------------------------------------
# Monitor messages
# ---------------------------------------------------------------------------

class TestMonitorMessages:
    """Test monitor-specific message lists."""

    def test_start_messages_exist(self):
        from evalview.core.messages import MONITOR_START_MESSAGES, get_random_monitor_start_message
        assert len(MONITOR_START_MESSAGES) >= 3
        msg = get_random_monitor_start_message()
        assert msg in MONITOR_START_MESSAGES

    def test_cycle_messages_exist(self):
        from evalview.core.messages import MONITOR_CYCLE_MESSAGES, get_random_monitor_cycle_message
        assert len(MONITOR_CYCLE_MESSAGES) >= 3
        msg = get_random_monitor_cycle_message()
        assert msg in MONITOR_CYCLE_MESSAGES

    def test_clean_messages_exist(self):
        from evalview.core.messages import MONITOR_CLEAN_MESSAGES, get_random_monitor_clean_message
        assert len(MONITOR_CLEAN_MESSAGES) >= 3
        msg = get_random_monitor_clean_message()
        assert msg in MONITOR_CLEAN_MESSAGES

    def test_messages_rotate(self):
        from evalview.core.messages import get_random_monitor_cycle_message
        seen = set()
        for _ in range(100):
            seen.add(get_random_monitor_cycle_message())
        assert len(seen) >= 3
