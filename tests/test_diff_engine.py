"""Unit tests for diff engine and parameter diffing."""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pytest

from evalview.core.diff import DiffEngine, ParameterDiff, DiffStatus
from evalview.core.golden import GoldenStore, GoldenTrace, GoldenMetadata
from evalview.core.config import DiffConfig
from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    ExecutionMetrics,
    StepMetrics
)


class TestParameterDiff:
    """Test parameter diffing functionality."""

    @pytest.fixture
    def diff_engine(self):
        """Create a diff engine for testing."""
        return DiffEngine()

    def test_parameter_diff_missing(self, diff_engine):
        """Test detection of missing parameters."""
        golden_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"query": "test", "max_results": 10},
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        actual_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"query": "test"},  # max_results missing
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        diffs = diff_engine._compare_tool_parameters(golden_step, actual_step)

        assert len(diffs) == 1
        assert diffs[0].param_name == "max_results"
        assert diffs[0].diff_type == "missing"
        assert diffs[0].golden_value == 10
        assert diffs[0].actual_value is None

    def test_parameter_diff_added(self, diff_engine):
        """Test detection of added parameters."""
        golden_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"query": "test"},
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        actual_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"query": "test", "limit": 5},  # limit added
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        diffs = diff_engine._compare_tool_parameters(golden_step, actual_step)

        assert len(diffs) == 1
        assert diffs[0].param_name == "limit"
        assert diffs[0].diff_type == "added"
        assert diffs[0].golden_value is None
        assert diffs[0].actual_value == 5

    def test_parameter_diff_value_changed(self, diff_engine):
        """Test detection of value changes."""
        golden_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"query": "test query", "max_results": 10},
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        actual_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"query": "different query", "max_results": 20},
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        diffs = diff_engine._compare_tool_parameters(golden_step, actual_step)

        assert len(diffs) == 2

        # Check query diff
        query_diff = next(d for d in diffs if d.param_name == "query")
        assert query_diff.diff_type == "value_changed"
        assert query_diff.similarity is not None
        assert 0 <= query_diff.similarity <= 1

        # Check max_results diff
        max_results_diff = next(d for d in diffs if d.param_name == "max_results")
        assert max_results_diff.diff_type == "value_changed"
        assert max_results_diff.golden_value == 10
        assert max_results_diff.actual_value == 20

    def test_parameter_diff_type_changed(self, diff_engine):
        """Test detection of type changes."""
        golden_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"max_results": "10"},  # String
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        actual_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"max_results": 10},  # Integer
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        diffs = diff_engine._compare_tool_parameters(golden_step, actual_step)

        assert len(diffs) == 1
        assert diffs[0].diff_type == "type_changed"
        assert diffs[0].golden_value == "10"
        assert diffs[0].actual_value == 10

    def test_parameter_diff_string_similarity(self, diff_engine):
        """Test string similarity calculation."""
        golden_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"query": "AAPL stock price"},
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        actual_step = StepTrace(
            step_id="1",
            step_name="search",
            tool_name="search",
            parameters={"query": "AAPL current price"},
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        diffs = diff_engine._compare_tool_parameters(golden_step, actual_step)

        assert len(diffs) == 1
        assert diffs[0].similarity is not None
        assert diffs[0].similarity > 0.5  # Should be fairly similar

    def test_parameter_diff_no_parameters(self, diff_engine):
        """Test handling of steps with no parameters."""
        golden_step = StepTrace(
            step_id="1",
            step_name="action",
            tool_name="action",
            parameters={},  # Empty dict instead of None
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        actual_step = StepTrace(
            step_id="1",
            step_name="action",
            tool_name="action",
            parameters={},  # Empty dict instead of None
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        diffs = diff_engine._compare_tool_parameters(golden_step, actual_step)

        assert len(diffs) == 0


class TestMultiReferenceComparison:
    """Test multi-reference golden comparison."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def diff_engine(self):
        """Create a diff engine."""
        return DiffEngine()

    def test_compare_multi_reference_best_match(self, diff_engine, temp_dir):
        """Test that best matching variant is selected."""
        # Create 3 different golden traces
        trace1 = ExecutionTrace(
            session_id="test",
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="search",
                    tool_name="search",
                    parameters={"query": "exact match"},
                    output="result1",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0, latency=0)
                )
            ],
            final_output="Output 1",
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        trace2 = ExecutionTrace(
            session_id="test",
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="search",
                    tool_name="search",
                    parameters={"query": "different query"},
                    output="result2",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0, latency=0)
                )
            ],
            final_output="Output 2",
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        # Actual trace matches trace1
        actual = ExecutionTrace(
            session_id="test",
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="search",
                    tool_name="search",
                    parameters={"query": "exact match"},
                    output="result1",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0, latency=0)
                )
            ],
            final_output="Output 1",
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        golden1 = GoldenTrace(
            metadata=GoldenMetadata(test_name="test", blessed_at=datetime.now(), score=90.0),
            trace=trace1,
            tool_sequence=["search"],
            output_hash="hash1"
        )

        golden2 = GoldenTrace(
            metadata=GoldenMetadata(test_name="test", blessed_at=datetime.now(), score=90.0),
            trace=trace2,
            tool_sequence=["search"],
            output_hash="hash2"
        )

        # Compare
        diff = diff_engine.compare_multi_reference([golden1, golden2], actual, 90.0)

        # Should match golden1 (variant 0 = default)
        assert diff.overall_severity == DiffStatus.PASSED
        assert diff.matched_variant == "default"

    def test_compare_multi_reference_empty_list_errors(self, diff_engine):
        """Test that empty golden list raises error."""
        trace = ExecutionTrace(
            session_id="test",
            steps=[],
            final_output="Output",
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        with pytest.raises(ValueError, match="At least one golden variant"):
            diff_engine.compare_multi_reference([], trace, 90.0)

    def test_compare_multi_reference_severity_ranking(self, diff_engine):
        """Test that best match is chosen by severity ranking."""
        # Golden 1: tools different (TOOLS_CHANGED)
        trace1 = ExecutionTrace(
            session_id="test",
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="analyze",
                    tool_name="analyze",
                    parameters={},
                    output="result",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0, latency=0)
                )
            ],
            final_output="Output",
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        # Golden 2: exact match (PASSED)
        trace2 = ExecutionTrace(
            session_id="test",
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="search",
                    tool_name="search",
                    parameters={},
                    output="result",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0, latency=0)
                )
            ],
            final_output="Output",
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        # Actual matches trace2
        actual = ExecutionTrace(
            session_id="test",
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="search",
                    tool_name="search",
                    parameters={},
                    output="result",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0, latency=0)
                )
            ],
            final_output="Output",
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        golden1 = GoldenTrace(
            metadata=GoldenMetadata(test_name="test", blessed_at=datetime.now(), score=90.0),
            trace=trace1,
            tool_sequence=["analyze"],
            output_hash="hash1"
        )

        golden2 = GoldenTrace(
            metadata=GoldenMetadata(test_name="test", blessed_at=datetime.now(), score=90.0),
            trace=trace2,
            tool_sequence=["search"],
            output_hash="hash2"
        )

        # Should prefer golden2 (PASSED > TOOLS_CHANGED)
        diff = diff_engine.compare_multi_reference([golden1, golden2], actual, 90.0)

        assert diff.overall_severity == DiffStatus.PASSED
        assert diff.matched_variant == "variant_1"  # Second in list


class TestDiffConfig:
    """Test configurable diff thresholds."""

    def test_default_config(self):
        """Test that default config has reasonable values."""
        config = DiffConfig()

        assert config.tool_similarity_threshold == 0.8
        assert config.output_similarity_threshold == 0.9
        assert config.score_regression_threshold == 5.0
        assert config.ignore_whitespace is True
        assert config.ignore_case_in_output is False

    def test_custom_config(self):
        """Test creating engine with custom config."""
        config = DiffConfig(
            tool_similarity_threshold=0.85,
            output_similarity_threshold=0.95,
            score_regression_threshold=3.0
        )

        # Create engine with custom config (config is used internally)
        engine = DiffEngine(config=config)

        # Verify engine was created successfully
        assert engine is not None
        assert isinstance(engine, DiffEngine)

    def test_config_validation(self):
        """Test that config validates thresholds."""
        # Should accept valid values
        config = DiffConfig(tool_similarity_threshold=0.5)
        assert config.tool_similarity_threshold == 0.5

        # Should reject invalid values (outside 0-1 range)
        with pytest.raises(Exception):  # Pydantic validation error
            DiffConfig(tool_similarity_threshold=1.5)


class TestEdgeCases:
    """Test edge cases in diff engine."""

    @pytest.fixture
    def diff_engine(self):
        """Create a diff engine."""
        return DiffEngine()

    def test_empty_traces(self, diff_engine):
        """Test diffing empty traces."""
        golden = GoldenTrace(
            metadata=GoldenMetadata(test_name="test", blessed_at=datetime.now(), score=90.0),
            trace=ExecutionTrace(
                session_id="test",
                steps=[],
                final_output="",
                metrics=ExecutionMetrics(total_cost=0, total_latency=0),
                start_time=datetime.now(),
                end_time=datetime.now()
            ),
            tool_sequence=[],
            output_hash=""
        )

        actual = ExecutionTrace(
            session_id="test",
            steps=[],
            final_output="",
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        diff = diff_engine.compare(golden, actual, 90.0)

        assert diff.overall_severity == DiffStatus.PASSED

    def test_empty_vs_nonempty_parameters(self, diff_engine):
        """Test diffing empty vs non-empty parameters."""
        golden_step = StepTrace(
            step_id="1",
            step_name="action",
            tool_name="action",
            parameters={},  # Empty
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        actual_step = StepTrace(
            step_id="1",
            step_name="action",
            tool_name="action",
            parameters={"new_param": "value"},  # Has parameter
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        diffs = diff_engine._compare_tool_parameters(golden_step, actual_step)

        # Should detect added parameter
        assert len(diffs) == 1
        assert diffs[0].diff_type == "added"

    def test_nested_parameter_structures(self, diff_engine):
        """Test diffing nested parameter structures."""
        golden_step = StepTrace(
            step_id="1",
            step_name="action",
            tool_name="action",
            parameters={"config": {"nested": "value"}},
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        actual_step = StepTrace(
            step_id="1",
            step_name="action",
            tool_name="action",
            parameters={"config": {"nested": "different"}},
            output="result",
            success=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metrics=StepMetrics(cost=0, latency=0)
        )

        diffs = diff_engine._compare_tool_parameters(golden_step, actual_step)

        # Should detect value change
        assert len(diffs) == 1
        assert diffs[0].diff_type == "value_changed"
