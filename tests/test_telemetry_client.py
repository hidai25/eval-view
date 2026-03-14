"""Tests for telemetry client quiet behavior."""

from __future__ import annotations

import logging

from evalview.telemetry.client import _silence_telemetry_network_loggers


def test_silence_telemetry_network_loggers_sets_critical_level():
    _silence_telemetry_network_loggers()

    assert logging.getLogger("posthog").level == logging.CRITICAL
    assert logging.getLogger("backoff").level == logging.CRITICAL
    assert logging.getLogger("urllib3.connectionpool").level == logging.CRITICAL
