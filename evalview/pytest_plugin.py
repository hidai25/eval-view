"""pytest plugin for EvalView agent regression testing.

Provides fixtures and markers so developers can write agent regression
tests alongside their regular pytest test suite — no separate CLI workflow
needed.

This plugin registers automatically when evalview is installed (via the
pytest11 entry point in pyproject.toml). No conftest.py setup required.

Usage:
    # In any test file
    def test_weather_agent_stable(evalview_check):
        diff = evalview_check("weather-lookup")
        assert diff.overall_severity.value == "passed", diff.summary()

    def test_tool_sequence_unchanged(evalview_check):
        diff = evalview_check("tool-use-test")
        assert not diff.tool_diffs, f"Tool changes: {diff.summary()}"

    @pytest.mark.agent_regression
    @pytest.mark.model_sensitive
    def test_json_output_format(evalview_check):
        diff = evalview_check("structured-output")
        # model_sensitive marker surfaces a warning if model_id changed
        assert diff.overall_severity.value in ("passed", "output_changed")

Requirements:
    - .evalview/config.yaml defining your agent endpoint
    - At least one snapshot saved: run 'evalview snapshot'
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import pytest

from evalview.core.diff import DiffEngine, DiffStatus, TraceDiff
from evalview.core.golden import GoldenStore

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """Register EvalView custom markers."""
    config.addinivalue_line(
        "markers",
        "agent_regression: mark test as an agent regression test "
        "(requires an evalview golden baseline — run 'evalview snapshot' first)",
    )
    config.addinivalue_line(
        "markers",
        "model_sensitive: mark test as sensitive to underlying model version changes "
        "(emits a warning in the test output when model_id differs from baseline)",
    )


@pytest.fixture(scope="session")
def evalview_golden_store() -> GoldenStore:
    """Session-scoped GoldenStore pointed at the current working directory."""
    return GoldenStore(base_path=Path("."))


@pytest.fixture
def evalview_check(evalview_golden_store: GoldenStore, request: pytest.FixtureRequest):
    """Run a named test and diff the result against its golden baseline.

    Skips (does not fail) the test if no baseline exists — run
    'evalview snapshot' first to capture one.

    The returned TraceDiff contains:
        - overall_severity: DiffStatus (PASSED, OUTPUT_CHANGED, TOOLS_CHANGED, REGRESSION)
        - tool_diffs: list of ToolDiff for changed/added/removed tools
        - output_diff: OutputDiff with similarity score and unified diff lines
        - summary(): human-readable change summary

    Args (passed when calling the fixture):
        test_name (str): Name of the test case (must match YAML ``name:`` field).
        test_path (Path, optional): Directory with YAML test files. Defaults to "tests/".
        config_path (Path, optional): Config file path. Defaults to ".evalview/config.yaml".

    Returns:
        TraceDiff

    Example:
        def test_search_stable(evalview_check):
            diff = evalview_check("search-test")
            assert diff.overall_severity.value == "passed", diff.summary()

        def test_model_sensitive(evalview_check, request):
            # @pytest.mark.model_sensitive is applied via marker
            diff = evalview_check("reasoning-test")
            assert not diff.tool_diffs
    """

    def _check(
        test_name: str,
        test_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ) -> TraceDiff:
        # Skip if no baseline
        variants = evalview_golden_store.load_all_golden_variants(test_name)
        if not variants:
            pytest.skip(
                f"No golden baseline for '{test_name}'. "
                "Run 'evalview snapshot' first to capture one."
            )

        from evalview.core.runner import run_single_test

        result = asyncio.run(run_single_test(test_name, test_path, config_path))
        engine = DiffEngine()
        diff = engine.compare_multi_reference(variants, result.trace, result.score)

        # Surface model-change warning if the test is marked model_sensitive
        marker = request.node.get_closest_marker("model_sensitive")
        if marker and getattr(diff, "model_changed", False):
            golden_model = getattr(diff, "golden_model_id", None)
            actual_model = getattr(diff, "actual_model_id", None)
            logger.warning(
                f"[model_sensitive] Model changed for test '{test_name}': "
                f"{golden_model!r} → {actual_model!r}. "
                "Review output changes carefully — this may be model drift."
            )

        return diff

    return _check


@pytest.fixture
def evalview_snapshot(evalview_golden_store: GoldenStore):
    """Run a test and save the result as the golden baseline.

    Useful in session-scoped setup fixtures or for programmatically updating
    baselines from Python code (e.g., before a major refactor).

    Args (passed when calling the fixture):
        test_name (str): Name of the test case.
        notes (str, optional): Human-readable description for this snapshot.
        test_path (Path, optional): Directory with YAML test files. Defaults to "tests/".
        config_path (Path, optional): Config file path. Defaults to ".evalview/config.yaml".

    Returns:
        EvaluationResult for the captured run.

    Example:
        @pytest.fixture(scope="session", autouse=True)
        def capture_baseline(evalview_snapshot):
            # Capture fresh baselines before any regression tests run
            evalview_snapshot("search-test", notes="session setup baseline")
            evalview_snapshot("reasoning-test", notes="session setup baseline")
    """

    def _snapshot(
        test_name: str,
        notes: Optional[str] = None,
        test_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        from evalview.core.runner import run_single_test

        result = asyncio.run(run_single_test(test_name, test_path, config_path))
        if result.passed:
            evalview_golden_store.save_golden(result, notes=notes)
        return result

    return _snapshot
