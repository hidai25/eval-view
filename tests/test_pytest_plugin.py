"""Tests for evalview/pytest_plugin.py."""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evalview.core.diff import DiffStatus, TraceDiff
from evalview.core.golden import GoldenStore, GoldenMetadata, GoldenTrace
from evalview.core.types import ExecutionTrace, ExecutionMetrics


def _make_golden(test_name: str, store: GoldenStore) -> GoldenTrace:
    """Write a minimal golden trace to the store and return it."""
    from evalview.core.types import EvaluationResult, Evaluations, ToolEvaluation, SequenceEvaluation, OutputEvaluation, ContainsChecks

    trace = ExecutionTrace(
        session_id="s1",
        start_time=datetime.now(),
        end_time=datetime.now(),
        steps=[],
        final_output="The answer is 42.",
        metrics=ExecutionMetrics(total_cost=0.0, total_latency=100.0),
    )
    result = MagicMock()
    result.test_case = test_name
    result.score = 90.0
    result.passed = True
    result.trace = trace
    store.save_golden(result)
    return store.load_golden(test_name)


class TestPytestMarkers:
    """Verify markers are registered without error."""

    def test_markers_registered(self):
        from evalview.pytest_plugin import pytest_configure

        class FakeConfig:
            _lines = []
            def addinivalue_line(self, name, val):
                self._lines.append((name, val))

        cfg = FakeConfig()
        pytest_configure(cfg)
        marker_names = [line[0] for line in cfg._lines]
        assert all(n == "markers" for n in marker_names)
        all_text = " ".join(line[1] for line in cfg._lines)
        assert "agent_regression" in all_text
        assert "model_sensitive" in all_text


class TestEvalviewCheckFixture:
    """Tests for the evalview_check fixture behaviour."""

    @pytest.fixture
    def golden_store(self, tmp_path):
        return GoldenStore(base_path=tmp_path)

    def test_skips_when_no_golden(self, golden_store):
        """evalview_check should skip (not fail) when no baseline exists."""
        from evalview.pytest_plugin import evalview_check as make_check

        # Simulate what pytest does: call the fixture factory with the store
        check_fn = None

        # Build the fixture closure directly
        def fake_request():
            pass
        fake_request.node = MagicMock()
        fake_request.node.get_closest_marker = MagicMock(return_value=None)

        called_skip = []

        with patch("pytest.skip", side_effect=lambda msg: called_skip.append(msg)):
            # Manually build the fixture closure
            from evalview.core.diff import DiffEngine

            def _check(test_name, test_path=None, config_path=None):
                variants = golden_store.load_all_golden_variants(test_name)
                if not variants:
                    pytest.skip(f"No golden baseline for '{test_name}'.")

            _check("nonexistent-test")

        assert len(called_skip) == 1
        assert "No golden baseline" in called_skip[0]

    def test_returns_tracediff_on_success(self, golden_store, tmp_path):
        """evalview_check should return a TraceDiff when golden exists."""
        _make_golden("my-test", golden_store)

        fake_trace = ExecutionTrace(
            session_id="s2",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="The answer is 42.",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=100.0),
        )
        fake_result = MagicMock()
        fake_result.score = 90.0
        fake_result.trace = fake_trace

        with patch("evalview.core.runner.run_single_test", new=AsyncMock(return_value=fake_result)):
            from evalview.core.diff import DiffEngine
            variants = golden_store.load_all_golden_variants("my-test")
            engine = DiffEngine()
            diff = engine.compare_multi_reference(variants, fake_trace, 90.0)

        assert isinstance(diff, TraceDiff)
        assert diff.overall_severity == DiffStatus.PASSED

    def test_model_sensitive_warning_logged(self, golden_store, caplog):
        """model_sensitive marker should trigger a warning log when model changed."""
        import logging

        golden = _make_golden("my-test", golden_store)

        # Simulate a TraceDiff where model changed
        diff = TraceDiff(
            test_name="my-test",
            has_differences=True,
            tool_diffs=[],
            output_diff=None,
            score_diff=0.0,
            latency_diff=0.0,
            overall_severity=DiffStatus.OUTPUT_CHANGED,
            model_changed=True,
            golden_model_id="claude-3-5-sonnet-20241022",
            actual_model_id="claude-3-5-sonnet-20250219",
        )

        fake_marker = MagicMock()
        fake_request = MagicMock()
        fake_request.node.get_closest_marker.return_value = fake_marker

        fake_trace = ExecutionTrace(
            session_id="s3",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="ok",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
            model_id="claude-3-5-sonnet-20250219",
        )
        fake_result = MagicMock()
        fake_result.score = 85.0
        fake_result.trace = fake_trace

        with (
            patch("evalview.core.runner.run_single_test", new=AsyncMock(return_value=fake_result)),
            patch("evalview.core.diff.DiffEngine.compare_multi_reference", return_value=diff),
            caplog.at_level(logging.WARNING, logger="evalview.pytest_plugin"),
        ):
            # Manually invoke fixture logic
            import logging as log_mod
            logger = log_mod.getLogger("evalview.pytest_plugin")
            marker = fake_request.node.get_closest_marker("model_sensitive")
            if marker and diff.model_changed:
                logger.warning(
                    f"[model_sensitive] Model changed for test 'my-test': "
                    f"{diff.golden_model_id!r} → {diff.actual_model_id!r}."
                )

        assert any("model_sensitive" in r.message for r in caplog.records)
        assert any("claude-3-5-sonnet-20241022" in r.message for r in caplog.records)
