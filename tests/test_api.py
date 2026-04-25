"""Tests for evalview.api — the programmatic Python API."""
from unittest.mock import MagicMock


from evalview.api import (
    gate,
    GateResult,
    GateSummary,
    TestDiff,
    DiffStatus,
    _worst_status,
    _build_gate_result,
)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestWorstStatus:
    def test_empty_diffs_returns_passed(self):
        assert _worst_status([]) == DiffStatus.PASSED

    def test_single_regression(self):
        diff = MagicMock()
        diff.overall_severity = DiffStatus.REGRESSION
        assert _worst_status([("t", diff)]) == DiffStatus.REGRESSION

    def test_regression_beats_tools_changed(self):
        d1 = MagicMock()
        d1.overall_severity = DiffStatus.TOOLS_CHANGED
        d2 = MagicMock()
        d2.overall_severity = DiffStatus.REGRESSION
        assert _worst_status([("a", d1), ("b", d2)]) == DiffStatus.REGRESSION

    def test_tools_changed_beats_output_changed(self):
        d1 = MagicMock()
        d1.overall_severity = DiffStatus.OUTPUT_CHANGED
        d2 = MagicMock()
        d2.overall_severity = DiffStatus.TOOLS_CHANGED
        assert _worst_status([("a", d1), ("b", d2)]) == DiffStatus.TOOLS_CHANGED

    def test_all_passed(self):
        d1 = MagicMock()
        d1.overall_severity = DiffStatus.PASSED
        d2 = MagicMock()
        d2.overall_severity = DiffStatus.PASSED
        assert _worst_status([("a", d1), ("b", d2)]) == DiffStatus.PASSED


class TestBuildGateResult:
    def _make_diff(self, status, score_diff=0.0, tool_diffs=None):
        d = MagicMock()
        d.overall_severity = status
        d.score_diff = score_diff
        d.output_diff = MagicMock()
        d.output_diff.similarity = 0.95
        d.output_diff.semantic_similarity = None
        d.tool_diffs = tool_diffs or []
        d.model_changed = False
        return d

    def test_all_passed_returns_passed(self):
        d = self._make_diff(DiffStatus.PASSED)
        result = _build_gate_result(
            [("test-a", d)], total_tests=1, fail_on={DiffStatus.REGRESSION}
        )
        assert result.passed is True
        assert result.exit_code == 0
        assert result.status == DiffStatus.PASSED
        assert result.summary.total == 1
        assert result.summary.unchanged == 1

    def test_regression_fails(self):
        d = self._make_diff(DiffStatus.REGRESSION, score_diff=-10.0)
        result = _build_gate_result(
            [("test-a", d)], total_tests=1, fail_on={DiffStatus.REGRESSION}
        )
        assert result.passed is False
        assert result.exit_code == 1
        assert result.summary.regressions == 1

    def test_tools_changed_passes_when_not_in_fail_on(self):
        d = self._make_diff(DiffStatus.TOOLS_CHANGED)
        result = _build_gate_result(
            [("test-a", d)], total_tests=1, fail_on={DiffStatus.REGRESSION}
        )
        assert result.passed is True  # TOOLS_CHANGED not in fail_on

    def test_tools_changed_fails_when_in_fail_on(self):
        d = self._make_diff(DiffStatus.TOOLS_CHANGED)
        result = _build_gate_result(
            [("test-a", d)],
            total_tests=1,
            fail_on={DiffStatus.REGRESSION, DiffStatus.TOOLS_CHANGED},
        )
        assert result.passed is False

    def test_execution_failures_counted(self):
        d = self._make_diff(DiffStatus.PASSED)
        result = _build_gate_result(
            [("test-a", d)], total_tests=3, fail_on={DiffStatus.REGRESSION}
        )
        assert result.summary.execution_failures == 2

    def test_raw_json_populated(self):
        d = self._make_diff(DiffStatus.PASSED)
        result = _build_gate_result(
            [("test-a", d)], total_tests=1, fail_on={DiffStatus.REGRESSION}
        )
        assert "summary" in result.raw_json
        assert "diffs" in result.raw_json

    def test_test_diff_passed_property(self):
        d = self._make_diff(DiffStatus.PASSED)
        result = _build_gate_result(
            [("test-a", d)], total_tests=1, fail_on={DiffStatus.REGRESSION}
        )
        assert result.diffs[0].passed is True

        d2 = self._make_diff(DiffStatus.REGRESSION)
        result2 = _build_gate_result(
            [("test-b", d2)], total_tests=1, fail_on={DiffStatus.REGRESSION}
        )
        assert result2.diffs[0].passed is False


# ---------------------------------------------------------------------------
# Integration tests for gate()
# ---------------------------------------------------------------------------


class TestGate:
    def test_missing_test_dir_returns_passed(self):
        result = gate(test_dir="/nonexistent/path/that/does/not/exist")
        assert result.passed is True
        assert result.summary.total == 0
        assert "error" in result.raw_json

    def test_empty_test_dir(self, tmp_path):
        result = gate(test_dir=str(tmp_path))
        assert result.passed is True
        assert result.summary.total == 0

    def test_no_matching_test_name(self, tmp_path):
        (tmp_path / "sample.yaml").write_text(
            "name: sample\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
            encoding="utf-8",
        )
        result = gate(test_dir=str(tmp_path), test_name="nonexistent")
        assert result.passed is True
        assert result.summary.total == 0

    def test_gate_with_mocked_execution(self, tmp_path, monkeypatch):
        """gate() should return structured results from the execution pipeline."""
        (tmp_path / "sample.yaml").write_text(
            "name: sample\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
            encoding="utf-8",
        )

        mock_diff = MagicMock()
        mock_diff.overall_severity = DiffStatus.PASSED
        mock_diff.score_diff = 0.0
        mock_diff.output_diff = MagicMock()
        mock_diff.output_diff.similarity = 1.0
        mock_diff.output_diff.semantic_similarity = None
        mock_diff.tool_diffs = []
        mock_diff.model_changed = False

        mock_result = MagicMock()
        mock_result.passed = True

        def fake_execute(test_cases, config, json_output, semantic_diff=False, timeout=30.0, skip_llm_judge=False):
            return [("sample", mock_diff)], [mock_result], MagicMock(), {}

        monkeypatch.setattr("evalview.commands.shared._execute_check_tests", fake_execute)

        result = gate(test_dir=str(tmp_path))
        assert isinstance(result, GateResult)
        assert result.passed is True
        assert result.summary.total == 1
        assert result.diffs[0].test_name == "sample"

    def test_quick_mode_skips_judge(self, tmp_path, monkeypatch):
        """gate(quick=True) should pass skip_llm_judge=True to the execution pipeline."""
        (tmp_path / "sample.yaml").write_text(
            "name: sample\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
            encoding="utf-8",
        )

        captured = {}

        def fake_execute(test_cases, config, json_output, semantic_diff=False, timeout=30.0, skip_llm_judge=False):
            captured["skip_llm_judge"] = skip_llm_judge
            captured["semantic_diff"] = semantic_diff
            mock_diff = MagicMock()
            mock_diff.overall_severity = DiffStatus.PASSED
            mock_diff.score_diff = 0.0
            mock_diff.output_diff = None
            mock_diff.tool_diffs = []
            mock_diff.model_changed = False
            return [("sample", mock_diff)], [MagicMock()], MagicMock(), {}

        monkeypatch.setattr("evalview.commands.shared._execute_check_tests", fake_execute)

        gate(test_dir=str(tmp_path), quick=True)
        assert captured["skip_llm_judge"] is True
        assert captured["semantic_diff"] is False


# ---------------------------------------------------------------------------
# GateResult / TestDiff type tests
# ---------------------------------------------------------------------------


class TestTypes:
    def test_gate_summary_defaults(self):
        gs = GateSummary()
        assert gs.total == 0
        assert gs.regressions == 0
        assert gs.execution_failures == 0

    def test_gate_result_fields(self):
        gr = GateResult(
            passed=True,
            exit_code=0,
            status=DiffStatus.PASSED,
            summary=GateSummary(total=1, unchanged=1),
            diffs=[],
        )
        assert gr.passed is True
        assert gr.raw_json == {}

    def test_test_diff_passed_property(self):
        td = TestDiff(
            test_name="t",
            status=DiffStatus.PASSED,
            score_delta=0,
            output_similarity=1.0,
            semantic_similarity=None,
            tool_changes=0,
            model_changed=False,
            raw=None,
        )
        assert td.passed is True

    def test_test_diff_not_passed(self):
        td = TestDiff(
            test_name="t",
            status=DiffStatus.REGRESSION,
            score_delta=-5.0,
            output_similarity=0.5,
            semantic_similarity=None,
            tool_changes=2,
            model_changed=False,
            raw=None,
        )
        assert td.passed is False

    def test_top_level_imports(self):
        from evalview import gate as g, DiffStatus as D, GateResult as G
        assert g is gate
        assert D is DiffStatus
        assert G is GateResult
