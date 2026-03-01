"""Single-test execution helpers for EvalView.

Provides run_single_test() and check_single_test() as reusable async
functions, decoupled from the CLI. Used by:

  - The pytest plugin (evalview/pytest_plugin.py)
  - Programmatic integrations (e.g., notebook evaluation)

The CLI (_execute_snapshot_tests / _execute_check_tests in cli.py) contains
the full production implementation with Rich console output, telemetry, and
streak tracking. This module provides a clean, minimal interface for the
common single-test case without those side effects.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import yaml  # type: ignore[import-untyped]

from evalview.core.config import EvalViewConfig
from evalview.core.diff import DiffEngine, TraceDiff
from evalview.core.golden import GoldenStore
from evalview.core.types import EvaluationResult, ExecutionTrace
from evalview.evaluators.evaluator import Evaluator

logger = logging.getLogger(__name__)


def _load_config(config_path: Optional[Path] = None) -> Optional[EvalViewConfig]:
    """Load EvalView config from YAML file.

    Args:
        config_path: Path to config file. Defaults to .evalview/config.yaml.

    Returns:
        Parsed EvalViewConfig or None if file not found.
    """
    path = config_path or Path(".evalview/config.yaml")
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    return EvalViewConfig.model_validate(data)


def _create_adapter(config: EvalViewConfig):
    """Create an agent adapter from config.

    Mirrors the _create_adapter() factory in cli.py. Kept in sync manually —
    if you add an adapter to the CLI factory, add it here too.

    Args:
        config: EvalViewConfig with adapter type and endpoint.

    Returns:
        AgentAdapter instance.

    Raises:
        ValueError: If adapter type is unknown.
    """
    from evalview.adapters.crewai_adapter import CrewAIAdapter
    from evalview.adapters.http_adapter import HTTPAdapter
    from evalview.adapters.langgraph_adapter import LangGraphAdapter
    from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter
    from evalview.adapters.tapescope_adapter import TapeScopeAdapter

    # Keep in sync with cli.py:_create_adapter().
    # Note: AnthropicAdapter is not included here because it is used
    # programmatically (not via .evalview/config.yaml). Use adapter: http
    # with a compatible endpoint for Anthropic-backed agents in config files.
    adapter_map = {
        "http": HTTPAdapter,
        "langgraph": LangGraphAdapter,
        "tapescope": TapeScopeAdapter,
        "crewai": CrewAIAdapter,
        "openai": OpenAIAssistantsAdapter,
    }

    adapter_class = adapter_map.get(config.adapter)
    if not adapter_class:
        raise ValueError(
            f"Unknown adapter type: '{config.adapter}'. "
            f"Supported: {', '.join(adapter_map.keys())}"
        )

    timeout = getattr(config, "timeout", 30.0)
    allow_private = getattr(config, "allow_private_urls", True)

    if config.adapter == "http":
        return adapter_class(
            endpoint=config.endpoint,
            timeout=timeout,
            allow_private_urls=allow_private,
        )
    return adapter_class(endpoint=config.endpoint, timeout=timeout)


async def run_single_test(
    test_name: str,
    test_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> EvaluationResult:
    """Execute a single named test case and return the evaluation result.

    Args:
        test_name: Name of the test case to run (must match YAML ``name:`` field).
        test_path: Directory containing YAML test files. Defaults to "tests/".
        config_path: Path to .evalview/config.yaml. Defaults to auto-discovery
                     (.evalview/config.yaml in the current directory).

    Returns:
        EvaluationResult with score, trace, and per-evaluator details.

    Raises:
        FileNotFoundError: If the test case is not found in test_path.
        ValueError: If config is missing or adapter type is unknown.
        RuntimeError: If agent execution fails.

    Example:
        result = asyncio.run(run_single_test("weather-lookup"))
        print(f"Score: {result.score}")
    """
    from evalview.core.loader import TestCaseLoader

    config = _load_config(config_path)
    if config is None:
        raise ValueError(
            "No .evalview/config.yaml found. "
            "Run 'evalview init' to create one, or pass config_path explicitly."
        )

    # Load all test cases from the directory and find the one we want
    path = test_path or Path("tests")
    loader = TestCaseLoader()
    test_cases = loader.load_from_directory(str(path))
    tc = next((t for t in test_cases if t.name == test_name), None)
    if tc is None:
        available = [t.name for t in test_cases]
        raise FileNotFoundError(
            f"Test case '{test_name}' not found in {path}. "
            f"Available tests: {available}"
        )

    # Allow per-test-case adapter/endpoint to override the config default
    adapter_type = tc.adapter or config.adapter
    endpoint = tc.endpoint or config.endpoint
    run_config = EvalViewConfig.model_validate(
        {**config.model_dump(), "adapter": adapter_type, "endpoint": endpoint}
    )

    adapter = _create_adapter(run_config)
    evaluator = Evaluator()

    trace: ExecutionTrace = await adapter.execute(tc.input.query, tc.input.context)
    result: EvaluationResult = await evaluator.evaluate(tc, trace)
    return result


async def check_single_test(
    test_name: str,
    test_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Tuple[EvaluationResult, TraceDiff]:
    """Run a test and diff the result against its golden baseline.

    Args:
        test_name: Name of the test to run and check.
        test_path: Directory containing YAML test files. Defaults to "tests/".
        config_path: Path to .evalview/config.yaml. Defaults to auto-discovery.

    Returns:
        Tuple of (EvaluationResult, TraceDiff). The TraceDiff contains
        overall_severity, tool_diffs, output_diff, and a summary().

    Raises:
        FileNotFoundError: If the test case or golden baseline is not found.
        ValueError: If config is missing or adapter type is unknown.

    Example:
        result, diff = asyncio.run(check_single_test("weather-lookup"))
        assert diff.overall_severity.value == "passed", diff.summary()
    """
    result = await run_single_test(test_name, test_path, config_path)

    store = GoldenStore()
    variants = store.load_all_golden_variants(test_name)
    if not variants:
        raise FileNotFoundError(
            f"No golden baseline found for '{test_name}'. "
            "Run 'evalview snapshot' first to capture a baseline."
        )

    engine = DiffEngine()
    diff = engine.compare_multi_reference(variants, result.trace, result.score)
    return result, diff
