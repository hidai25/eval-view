"""Tests for telemetry client quiet behavior."""

from __future__ import annotations

import logging

from evalview.telemetry.client import _silence_telemetry_network_loggers


def test_silence_telemetry_network_loggers_sets_critical_level():
    _silence_telemetry_network_loggers()

    assert logging.getLogger("posthog").level == logging.CRITICAL
    assert logging.getLogger("backoff").level == logging.CRITICAL
    assert logging.getLogger("urllib3.connectionpool").level == logging.CRITICAL


def test_default_payload_omits_user_content(monkeypatch):
    import json
    from evalview.telemetry.client import _usage_properties
    from evalview.telemetry.events import CommandEvent, SkillTestGenerationFailedEvent

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REPOSITORY", "secret/private")
    failure = SkillTestGenerationFailedEvent(skill_name="secret skill", error_message="secret prompt")
    command = CommandEvent(command_name="run", properties={"adapter": "secret/adapter", "has_path": True})
    props = _usage_properties(command)
    assert props["project_ci"] is False
    assert props["properties"] == {"adapter": "other", "has_path": True}
    assert "secret" not in json.dumps([props, _usage_properties(failure)])


def test_command_exit_records_outcome_and_resolved_properties(monkeypatch):
    import pytest
    from unittest.mock import Mock
    from evalview.telemetry import decorators

    client = Mock()
    monkeypatch.setattr(decorators, "get_client", lambda: client)
    monkeypatch.setattr(decorators, "is_telemetry_enabled", lambda: True)

    @decorators.track_command("check")
    def run_check():
        decorators.record_command_properties(comparison_complete=True, changed=True)
        raise SystemExit(1)

    with pytest.raises(SystemExit):
        run_check()
    event = client.track.call_args.args[0]
    assert event.success is False
    assert event.properties == {"comparison_complete": True, "changed": True}
