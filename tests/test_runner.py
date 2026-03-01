"""Tests for evalview/core/runner.py."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


def _write_config(base: Path, adapter: str = "http", endpoint: str = "http://localhost:8000") -> Path:
    config_dir = base / ".evalview"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump({"adapter": adapter, "endpoint": endpoint}))
    return config_path


def _write_test_case(test_dir: Path, name: str = "my-test", query: str = "hello") -> None:
    test_dir.mkdir(parents=True, exist_ok=True)
    # TestCase requires `expected` and `thresholds` fields
    (test_dir / f"{name}.yaml").write_text(
        yaml.dump({
            "name": name,
            "input": {"query": query},
            "expected": {"tools": []},
            "thresholds": {"min_score": 0},
        })
    )


class TestLoadConfig:
    def test_returns_none_when_missing(self, tmp_path):
        from evalview.core.runner import _load_config
        result = _load_config(tmp_path / "nonexistent.yaml")
        assert result is None

    def test_loads_valid_config(self, tmp_path):
        from evalview.core.runner import _load_config
        path = _write_config(tmp_path)
        config = _load_config(path)
        assert config is not None
        assert config.adapter == "http"
        assert config.endpoint == "http://localhost:8000"


class TestCreateAdapter:
    def test_creates_http_adapter(self, tmp_path):
        from evalview.core.runner import _create_adapter, _load_config
        _write_config(tmp_path, adapter="http")
        config = _load_config(tmp_path / ".evalview" / "config.yaml")
        adapter = _create_adapter(config)
        from evalview.adapters.http_adapter import HTTPAdapter
        assert isinstance(adapter, HTTPAdapter)

    def test_raises_for_unknown_adapter(self, tmp_path):
        from evalview.core.runner import _create_adapter, _load_config
        _write_config(tmp_path, adapter="nonexistent")
        config = _load_config(tmp_path / ".evalview" / "config.yaml")
        with pytest.raises(ValueError, match="Unknown adapter"):
            _create_adapter(config)

    def test_adapter_map_includes_tapescope(self, tmp_path):
        """Regression: tapescope was previously missing from runner's adapter map."""
        from evalview.core.runner import _create_adapter, _load_config
        _write_config(tmp_path, adapter="tapescope", endpoint="http://example.com/api")
        config = _load_config(tmp_path / ".evalview" / "config.yaml")
        try:
            adapter = _create_adapter(config)
            from evalview.adapters.tapescope_adapter import TapeScopeAdapter
            assert isinstance(adapter, TapeScopeAdapter)
        except ImportError:
            pytest.skip("tapescope package not installed")
        except ValueError as e:
            pytest.fail(f"tapescope missing from adapter_map: {e}")

    def test_unknown_adapter_raises(self, tmp_path):
        """anthropic is not in adapter_map (programmatic-only) — raises ValueError."""
        from evalview.core.runner import _create_adapter, _load_config
        _write_config(tmp_path, adapter="anthropic")
        config = _load_config(tmp_path / ".evalview" / "config.yaml")
        with pytest.raises(ValueError, match="Unknown adapter"):
            _create_adapter(config)


class TestRunSingleTest:
    @pytest.fixture
    def project(self, tmp_path):
        _write_config(tmp_path)
        _write_test_case(tmp_path / "tests")
        return tmp_path

    @pytest.mark.asyncio
    async def test_raises_when_no_config(self, tmp_path):
        from evalview.core.runner import run_single_test
        with pytest.raises(ValueError, match="config.yaml"):
            await run_single_test("any-test", config_path=tmp_path / "missing.yaml")

    @pytest.mark.asyncio
    async def test_raises_when_test_not_found(self, project):
        from evalview.core.runner import run_single_test
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            await run_single_test(
                "nonexistent",
                test_path=project / "tests",
                config_path=project / ".evalview" / "config.yaml",
            )

    @pytest.mark.asyncio
    async def test_runs_test_and_returns_result(self, project):
        from evalview.core.runner import run_single_test
        from evalview.core.types import (
            EvaluationResult,
            ExecutionTrace,
            ExecutionMetrics,
        )
        from datetime import datetime

        fake_trace = ExecutionTrace(
            session_id="s1",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="The answer is 42.",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=100.0),
        )

        fake_result = MagicMock(spec=EvaluationResult)
        fake_result.score = 90.0
        fake_result.passed = True
        fake_result.trace = fake_trace

        mock_adapter = MagicMock()
        mock_adapter.execute = AsyncMock(return_value=fake_trace)

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(return_value=fake_result)

        with (
            patch("evalview.core.runner._create_adapter", return_value=mock_adapter),
            patch("evalview.core.runner.Evaluator", return_value=mock_evaluator),
        ):
            result = await run_single_test(
                "my-test",
                test_path=project / "tests",
                config_path=project / ".evalview" / "config.yaml",
            )

        assert result is fake_result
        mock_adapter.execute.assert_called_once_with("hello", None)


class TestCheckSingleTest:
    @pytest.fixture
    def project(self, tmp_path):
        _write_config(tmp_path)
        _write_test_case(tmp_path / "tests")
        return tmp_path

    @pytest.mark.asyncio
    async def test_raises_when_no_golden(self, project):
        from evalview.core.runner import check_single_test
        from evalview.core.types import (
            EvaluationResult,
            ExecutionTrace,
            ExecutionMetrics,
        )
        from datetime import datetime

        fake_trace = ExecutionTrace(
            session_id="s1",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="ok",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )
        fake_result = MagicMock(spec=EvaluationResult)
        fake_result.score = 90.0
        fake_result.trace = fake_trace

        mock_adapter = MagicMock()
        mock_adapter.execute = AsyncMock(return_value=fake_trace)
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(return_value=fake_result)

        with (
            patch("evalview.core.runner._create_adapter", return_value=mock_adapter),
            patch("evalview.core.runner.Evaluator", return_value=mock_evaluator),
        ):
            with pytest.raises(FileNotFoundError, match="No golden baseline"):
                await check_single_test(
                    "my-test",
                    test_path=project / "tests",
                    config_path=project / ".evalview" / "config.yaml",
                )
