"""Tests for run-command onboarding guidance."""

from __future__ import annotations


def test_no_agent_guide_points_to_init_instead_of_stale_examples():
    from evalview.commands.run._cmd import _display_no_agent_guide
    from evalview.commands.run._cmd import console

    with console.capture() as capture:
        _display_no_agent_guide("http://localhost:8090/execute")

    output = capture.get()
    assert "evalview init" in output
    assert "generated-from-init" in output
    assert "demo-agent/agent.py" not in output
