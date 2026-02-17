"""Unit tests for golden trace storage and variant management."""

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pytest

from evalview.core.golden import GoldenStore, GoldenTrace, GoldenMetadata
from evalview.core.types import (
    EvaluationResult,
    ExecutionTrace,
    StepTrace,
    ExecutionMetrics,
    StepMetrics,
    Evaluations,
    ToolEvaluation,
    SequenceEvaluation,
    OutputEvaluation,
    CostEvaluation,
    LatencyEvaluation,
    ContainsChecks
)


def create_sample_evaluations():
    """Create a sample Evaluations object for testing."""
    return Evaluations(
        tool_accuracy=ToolEvaluation(accuracy=1.0, correct=["search"], missing=[], unexpected=[]),
        sequence_correctness=SequenceEvaluation(correct=True, expected_sequence=["search"], actual_sequence=["search"], violations=[], progress_score=1.0),
        output_quality=OutputEvaluation(
            score=85.0,
            rationale="Good output",
            contains_checks=ContainsChecks(passed=[], failed=[]),
            not_contains_checks=ContainsChecks(passed=[], failed=[])
        ),
        cost=CostEvaluation(total_cost=0.01, threshold=0.10, passed=True),
        latency=LatencyEvaluation(total_latency=100.0, threshold=5000.0, passed=True)
    )


class TestGoldenStore:
    """Test basic golden store operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sample_result(self):
        """Create a sample evaluation result for testing."""
        trace = ExecutionTrace(
            session_id="test-session",
            steps=[
                StepTrace(
                    step_id="step-1",
                    step_name="search",
                    tool_name="search",
                    parameters={"query": "test"},
                    output="result",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0.01, latency=50)
                )
            ],
            final_output="Final result",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=100),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        return EvaluationResult(
            test_case="test-example",
            trace=trace,
            score=85.0,
            passed=True,
            evaluations=create_sample_evaluations(),
            timestamp=datetime.now()
        )

    def test_save_and_load_golden(self, temp_dir, sample_result):
        """Test saving and loading a golden trace."""
        store = GoldenStore(temp_dir)

        # Save
        path = store.save_golden(sample_result, notes="Test baseline")

        assert path.exists()
        assert path.name == "test-example.golden.json"

        # Load
        golden = store.load_golden("test-example")

        assert golden is not None
        assert golden.metadata.test_name == "test-example"
        assert golden.metadata.notes == "Test baseline"
        assert golden.metadata.score == 85.0
        assert len(golden.trace.steps) == 1
        assert golden.trace.steps[0].tool_name == "search"

    def test_has_golden(self, temp_dir, sample_result):
        """Test checking if golden exists."""
        store = GoldenStore(temp_dir)

        assert not store.has_golden("test-example")

        store.save_golden(sample_result)

        assert store.has_golden("test-example")

    def test_delete_golden(self, temp_dir, sample_result):
        """Test deleting a golden trace."""
        store = GoldenStore(temp_dir)

        store.save_golden(sample_result)
        assert store.has_golden("test-example")

        result = store.delete_golden("test-example")

        assert result is True
        assert not store.has_golden("test-example")

    def test_delete_nonexistent_golden(self, temp_dir):
        """Test deleting a golden that doesn't exist."""
        store = GoldenStore(temp_dir)

        result = store.delete_golden("nonexistent")

        assert result is False

    def test_list_golden(self, temp_dir, sample_result):
        """Test listing all golden traces."""
        store = GoldenStore(temp_dir)

        # Empty list initially
        assert store.list_golden() == []

        # Save some goldens
        store.save_golden(sample_result)

        result2 = EvaluationResult(
            test_case="test-example-2",
            trace=sample_result.trace,
            score=90.0,
            passed=True,
            evaluations=create_sample_evaluations(),
            timestamp=datetime.now()
        )
        store.save_golden(result2)

        # List should have both
        goldens = store.list_golden()

        assert len(goldens) == 2
        assert any(g.test_name == "test-example" for g in goldens)
        assert any(g.test_name == "test-example-2" for g in goldens)


class TestGoldenVariants:
    """Test multi-reference golden variant management."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sample_result(self):
        """Create a sample evaluation result for testing."""
        trace = ExecutionTrace(
            session_id="test-session",
            steps=[
                StepTrace(
                    step_id="step-1",
                    step_name="search",
                    tool_name="search",
                    parameters={"query": "test"},
                    output="result",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0.01, latency=50)
                )
            ],
            final_output="Final result",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=100),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        return EvaluationResult(
            test_case="test-multi",
            trace=trace,
            score=85.0,
            passed=True,
            evaluations=create_sample_evaluations(),
            timestamp=datetime.now()
        )

    def test_save_variant(self, temp_dir, sample_result):
        """Test saving a golden variant."""
        store = GoldenStore(temp_dir)

        # Save default
        store.save_golden(sample_result)

        # Save variant
        path = store.save_golden(sample_result, variant_name="variant1")

        assert path.exists()
        assert path.name == "test-multi.variant_variant1.golden.json"

    def test_load_all_variants(self, temp_dir, sample_result):
        """Test loading all variants for a test."""
        store = GoldenStore(temp_dir)

        # Save default + 2 variants
        store.save_golden(sample_result)
        store.save_golden(sample_result, variant_name="variant1")
        store.save_golden(sample_result, variant_name="variant2")

        # Load all
        variants = store.load_all_golden_variants("test-multi")

        assert len(variants) == 3

    def test_count_variants(self, temp_dir, sample_result):
        """Test counting variants."""
        store = GoldenStore(temp_dir)

        assert store.count_variants("test-multi") == 0

        store.save_golden(sample_result)
        assert store.count_variants("test-multi") == 1

        store.save_golden(sample_result, variant_name="variant1")
        assert store.count_variants("test-multi") == 2

        store.save_golden(sample_result, variant_name="variant2")
        assert store.count_variants("test-multi") == 3

    def test_max_5_variants(self, temp_dir, sample_result):
        """Test that max 5 variants are enforced."""
        store = GoldenStore(temp_dir)

        # Save default + 4 variants (5 total)
        store.save_golden(sample_result)
        for i in range(4):
            store.save_golden(sample_result, variant_name=f"variant{i}")

        assert store.count_variants("test-multi") == 5

        # Try to save 6th variant (should fail)
        with pytest.raises(ValueError, match="Maximum 5 variants"):
            store.save_golden(sample_result, variant_name="variant_extra")

    def test_overwrite_existing_variant(self, temp_dir, sample_result):
        """Test that overwriting existing variant is allowed."""
        store = GoldenStore(temp_dir)

        # Save default + 4 variants (5 total)
        store.save_golden(sample_result, notes="Original")
        for i in range(4):
            store.save_golden(sample_result, variant_name=f"variant{i}")

        assert store.count_variants("test-multi") == 5

        # Overwrite default (should succeed, not count as new)
        store.save_golden(sample_result, notes="Updated")

        golden = store.load_golden("test-multi")
        assert golden.metadata.notes == "Updated"
        assert store.count_variants("test-multi") == 5  # Still 5

    def test_delete_variant(self, temp_dir, sample_result):
        """Test deleting a specific variant."""
        store = GoldenStore(temp_dir)

        store.save_golden(sample_result)
        store.save_golden(sample_result, variant_name="variant1")
        store.save_golden(sample_result, variant_name="variant2")

        assert store.count_variants("test-multi") == 3

        # Delete variant1
        result = store.delete_golden("test-multi", variant_name="variant1")

        assert result is True
        assert store.count_variants("test-multi") == 2

        # Remaining should be default and variant2
        variants = store.load_all_golden_variants("test-multi")
        assert len(variants) == 2

    def test_list_with_variant_counts(self, temp_dir, sample_result):
        """Test listing goldens with variant counts."""
        store = GoldenStore(temp_dir)

        # Create test-multi with 3 variants
        store.save_golden(sample_result)  # test-multi
        store.save_golden(sample_result, variant_name="v1")
        store.save_golden(sample_result, variant_name="v2")

        # Create another test with 1 variant
        result2 = EvaluationResult(
            test_case="test-single",
            trace=sample_result.trace,
            score=90.0,
            passed=True,
            evaluations=create_sample_evaluations(),
            timestamp=datetime.now()
        )
        store.save_golden(result2)

        # List with counts
        goldens = store.list_golden_with_variants()

        assert len(goldens) == 2

        multi = next(g for g in goldens if g["metadata"].test_name == "test-multi")
        assert multi["variant_count"] == 3

        single = next(g for g in goldens if g["metadata"].test_name == "test-single")
        assert single["variant_count"] == 1


class TestPathSanitization:
    """Test path sanitization and security."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sample_result(self):
        """Create a sample evaluation result for testing."""
        trace = ExecutionTrace(
            session_id="test-session",
            steps=[],
            final_output="Final result",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=100),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        return EvaluationResult(
            test_case="placeholder",  # Will be overridden
            trace=trace,
            score=85.0,
            passed=True,
            evaluations=create_sample_evaluations(),
            timestamp=datetime.now()
        )

    def test_special_characters_sanitized(self, temp_dir, sample_result):
        """Test that special characters are sanitized in filenames."""
        store = GoldenStore(temp_dir)

        sample_result.test_case = "test/with/slashes"
        path = store.save_golden(sample_result)

        # Slashes should be replaced with underscores
        assert "/" not in path.name
        assert path.name == "test_with_slashes.golden.json"

    def test_path_traversal_prevented(self, temp_dir, sample_result):
        """Test that path traversal attempts are prevented."""
        store = GoldenStore(temp_dir)

        sample_result.test_case = "../../../etc/passwd"
        path = store.save_golden(sample_result)

        # Should be sanitized and stay in golden_dir
        assert path.parent == store.golden_dir
        assert ".." not in path.name
        assert "/" not in path.name

    def test_dots_removed_from_name(self, temp_dir, sample_result):
        """Test that dots (except extension) are removed to prevent traversal."""
        store = GoldenStore(temp_dir)

        sample_result.test_case = "test.with.dots"
        path = store.save_golden(sample_result)

        # Dots should be replaced except for the extension
        assert path.name == "test_with_dots.golden.json"

    def test_variant_name_sanitized(self, temp_dir, sample_result):
        """Test that variant names are also sanitized."""
        store = GoldenStore(temp_dir)

        sample_result.test_case = "test"
        path = store.save_golden(sample_result, variant_name="../evil")

        # Variant name should be sanitized
        assert path.parent == store.golden_dir
        assert path.name == "test.variant____evil.golden.json"


class TestGoldenEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_load_nonexistent_returns_none(self, temp_dir):
        """Test that loading nonexistent golden returns None."""
        store = GoldenStore(temp_dir)

        golden = store.load_golden("nonexistent")

        assert golden is None

    def test_load_corrupted_json(self, temp_dir):
        """Test that corrupted JSON is handled gracefully."""
        store = GoldenStore(temp_dir)

        # Create corrupted file
        store.golden_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = store.golden_dir / "corrupted.golden.json"
        corrupted_path.write_text("not valid json {]}")

        # list_golden should skip it with warning
        goldens = store.list_golden()

        assert len(goldens) == 0  # Should not crash

    def test_empty_test_name(self, temp_dir):
        """Test handling of empty test name."""
        store = GoldenStore(temp_dir)

        path = store._get_golden_path("")

        # Should still generate a valid path
        assert path.name == ".golden.json"

    def test_very_long_test_name(self, temp_dir):
        """Test handling of very long test names."""
        store = GoldenStore(temp_dir)

        long_name = "a" * 500
        path = store._get_golden_path(long_name)

        # Should generate a valid (if long) filename
        assert path.name == f"{'a' * 500}.golden.json"

    def test_unicode_test_name(self, temp_dir):
        """Test handling of Unicode characters in test names."""
        store = GoldenStore(temp_dir)

        path = store._get_golden_path("test-🔥-emoji")

        # Unicode should be replaced with underscores
        assert "🔥" not in path.name
        assert path.name == "test-_-emoji.golden.json"

    def test_load_all_variants_empty_dir(self, temp_dir):
        """Test loading variants when golden dir doesn't exist."""
        store = GoldenStore(temp_dir)

        # Don't create golden_dir
        variants = store.load_all_golden_variants("test")

        assert variants == []

    def test_tool_sequence_extraction(self, temp_dir):
        """Test that tool sequence is correctly extracted from trace."""
        store = GoldenStore(temp_dir)

        trace = ExecutionTrace(
            session_id="test",
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="search",
                    tool_name="search",
                    parameters={},
                    output="",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0, latency=0)
                ),
                StepTrace(
                    step_id="2",
                    step_name="analyze",
                    tool_name="analyze",
                    parameters={},
                    output="",
                    success=True,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    metrics=StepMetrics(cost=0, latency=0)
                )
            ],
            final_output="result",
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        result = EvaluationResult(
            test_case="test",
            trace=trace,
            score=85.0,
            passed=True,
            evaluations=create_sample_evaluations(),
            timestamp=datetime.now()
        )

        store.save_golden(result)
        golden = store.load_golden("test")

        assert golden.tool_sequence == ["search", "analyze"]
