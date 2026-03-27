"""Tests for behavior-tag filtering in evalview run."""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner


def test_run_passes_tag_filters_into_async_entrypoint(monkeypatch, tmp_path):
    """CLI should forward repeatable --tag options into the async run path."""
    from evalview.commands.run._cmd import run

    monkeypatch.chdir(tmp_path)
    captured = {}

    async def _fake_run_async(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("evalview.commands.run._cmd._run_async", _fake_run_async)

    runner = CliRunner()
    result = runner.invoke(run, ["tests", "--tag", "tool_use", "--tag", "retrieval", "--dry-run"])

    assert result.exit_code == 0
    assert captured["tags"] == ("tool_use", "retrieval")
    assert captured["dry_run"] is True


def test_filter_by_tags_matches_any_requested_behavior():
    """Tag filtering should use OR semantics and keep normalized matches."""
    from evalview.commands.run._cmd import _filter_by_tags

    test_cases = [
        SimpleNamespace(name="tool-path", tags=["tool_use", "retrieval"]),
        SimpleNamespace(name="memory-path", tags=["memory"]),
        SimpleNamespace(name="untagged", tags=[]),
    ]

    filtered, active_tags = _filter_by_tags(test_cases, ("tool_use", "memory"))

    assert active_tags == ["tool_use", "memory"]
    assert [tc.name for tc in filtered] == ["tool-path", "memory-path"]
