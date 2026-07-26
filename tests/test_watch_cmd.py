"""Tests for `evalview watch` and `evalview check --watch`.

The regression these pin: the watch loop runs under ``asyncio.run()``, so a
triggered check must go through ``gate_async``. Calling the synchronous
``gate()`` there raises "asyncio.run() cannot be called from a running event
loop", which ``_run_check`` swallows into a red "Check failed" line — the
banner and the initial check still look healthy, so the breakage is invisible
until you actually save a file.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from evalview.commands.watch_cmd import _run_check_async, run_watch, watch


def _fake_result():
    """A GateResult-shaped stub with no diffs (the early-return path)."""
    return SimpleNamespace(diffs=[], passed=True, summary=None)


class TestWatchAsyncCheck:
    """`_run_check_async` must be usable from inside a running event loop."""

    async def test_runs_inside_running_event_loop(self):
        """The regression: this raised RuntimeError via the sync gate()."""
        called = {}

        async def fake_gate_async(**kwargs):
            called.update(kwargs)
            return _fake_result()

        with patch("evalview.api.gate_async", fake_gate_async):
            # Directly awaited, so we are provably inside a running loop —
            # the exact condition that broke the real watch loop.
            assert asyncio.get_running_loop() is not None
            await _run_check_async(
                test_dir="tests",
                test_name=None,
                quick=True,
                fail_statuses=set(),
                sound=False,
                trigger_path="some/file.py",
            )

        assert called["test_dir"] == "tests"
        assert called["quick"] is True

    async def test_gate_failure_is_reported_not_raised(self):
        """A failing check must not tear down the watch loop."""

        async def boom(**kwargs):
            raise RuntimeError("agent unreachable")

        with patch("evalview.api.gate_async", boom):
            # Must return normally so the loop survives to the next change.
            await _run_check_async(
                test_dir="tests",
                test_name=None,
                quick=True,
                fail_statuses=set(),
                sound=False,
                trigger_path="some/file.py",
            )


class TestCheckWatchFlag:
    """`check --watch` delegates to the shared watcher."""

    def test_delegates_to_run_watch(self, tmp_path):
        from evalview.commands.check_cmd import check

        (tmp_path / "tests").mkdir()

        with patch("evalview.commands.watch_cmd.run_watch") as mock_watch:
            CliRunner().invoke(check, [str(tmp_path / "tests"), "--watch", "--no-judge"])

        assert mock_watch.called
        kwargs = mock_watch.call_args.kwargs
        # --no-judge maps onto the watcher's quick mode.
        assert kwargs["quick"] is True
        assert kwargs["fail_on"] == "REGRESSION"

    def test_strict_maps_to_all_statuses(self, tmp_path):
        """--strict must fail on the same statuses a one-shot check does."""
        from evalview.commands.check_cmd import check

        (tmp_path / "tests").mkdir()

        with patch("evalview.commands.watch_cmd.run_watch") as mock_watch:
            CliRunner().invoke(check, [str(tmp_path / "tests"), "--watch", "--strict"])

        assert mock_watch.call_args.kwargs["fail_on"] == (
            "REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED"
        )

    @pytest.mark.parametrize(
        "flag,extra",
        [
            ("--json", []),
            ("--heal", []),
            ("--dry-run", []),
            ("--explain", []),
            ("--report", ["out.html"]),
            ("--budget", ["1.0"]),
            ("--statistical", ["5"]),
        ],
    )
    def test_rejects_incompatible_flags(self, tmp_path, flag, extra):
        """Refusing is deliberate — silently dropping a typed flag is worse."""
        from evalview.commands.check_cmd import check

        (tmp_path / "tests").mkdir()

        with patch("evalview.commands.watch_cmd.run_watch") as mock_watch:
            result = CliRunner().invoke(
                check, [str(tmp_path / "tests"), "--watch", flag, *extra]
            )

        assert result.exit_code == 1
        assert flag in result.output
        assert not mock_watch.called

    def test_missing_test_dir_exits(self, tmp_path):
        """run_watch exits rather than starting a watcher on nothing."""
        with pytest.raises(SystemExit):
            run_watch(
                watch_paths=["."],
                test_dir=str(tmp_path / "does-not-exist"),
                test_name=None,
                quick=True,
                fail_on="REGRESSION",
                sound=False,
                interval=2.0,
            )


class TestWatchCommandUnchanged:
    """The standalone `watch` command still works after the extraction."""

    def test_watch_delegates_to_run_watch(self, tmp_path):
        (tmp_path / "tests").mkdir()

        with patch("evalview.commands.watch_cmd.run_watch") as mock_watch:
            CliRunner().invoke(watch, ["--test-dir", str(tmp_path / "tests"), "--quick"])

        assert mock_watch.called
        kwargs = mock_watch.call_args.kwargs
        assert kwargs["quick"] is True
        assert kwargs["watch_paths"] == ["."]
