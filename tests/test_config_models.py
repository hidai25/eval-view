"""Tests for evalview.core.config models."""

import os
import pytest
from pydantic import ValidationError

from evalview.core.config import (
    ScoringWeights,
    DiffConfig,
    RetryConfig,
    CIConfig,
    JudgeConfig,
    MonitorConfig,
    EvalViewConfig,
    ScoringConfig,
    apply_judge_config,
    DEFAULT_WEIGHTS,
)


class TestScoringWeights:
    """Tests for ScoringWeights validation."""

    def test_default_weights_sum_to_one(self):
        w = ScoringWeights()
        assert abs(w.tool_accuracy + w.output_quality + w.sequence_correctness - 1.0) < 0.001

    def test_custom_weights_sum_to_one(self):
        w = ScoringWeights(tool_accuracy=0.5, output_quality=0.3, sequence_correctness=0.2)
        assert w.tool_accuracy == 0.5

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            ScoringWeights(tool_accuracy=0.5, output_quality=0.5, sequence_correctness=0.5)

    def test_negative_weight_raises(self):
        with pytest.raises(ValidationError):
            ScoringWeights(tool_accuracy=-0.1, output_quality=0.6, sequence_correctness=0.5)

    def test_weight_above_one_raises(self):
        with pytest.raises(ValidationError):
            ScoringWeights(tool_accuracy=1.5, output_quality=0.0, sequence_correctness=0.0)

    def test_to_dict(self):
        w = ScoringWeights()
        d = w.to_dict()
        assert set(d.keys()) == {"tool_accuracy", "output_quality", "sequence_correctness"}
        assert all(isinstance(v, float) for v in d.values())

    def test_all_zero_raises(self):
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            ScoringWeights(tool_accuracy=0.0, output_quality=0.0, sequence_correctness=0.0)


class TestDiffConfig:
    """Tests for DiffConfig defaults and validation."""

    def test_defaults(self):
        c = DiffConfig()
        assert c.tool_similarity_threshold == 0.8
        assert c.output_similarity_threshold == 0.9
        assert c.score_regression_threshold == 5.0
        assert c.ignore_whitespace is True
        assert c.ignore_case_in_output is False
        assert c.semantic_diff_enabled is None

    def test_custom_thresholds(self):
        c = DiffConfig(tool_similarity_threshold=0.95, output_similarity_threshold=0.5)
        assert c.tool_similarity_threshold == 0.95
        assert c.output_similarity_threshold == 0.5

    def test_threshold_bounds(self):
        with pytest.raises(ValidationError):
            DiffConfig(tool_similarity_threshold=1.5)
        with pytest.raises(ValidationError):
            DiffConfig(output_similarity_threshold=-0.1)

    def test_semantic_diff_explicit_false(self):
        c = DiffConfig(semantic_diff_enabled=False)
        assert c.semantic_diff_enabled is False

    def test_semantic_similarity_weight_bounds(self):
        c = DiffConfig(semantic_similarity_weight=0.0)
        assert c.semantic_similarity_weight == 0.0
        c = DiffConfig(semantic_similarity_weight=1.0)
        assert c.semantic_similarity_weight == 1.0
        with pytest.raises(ValidationError):
            DiffConfig(semantic_similarity_weight=1.5)


class TestRetryConfig:
    """Tests for RetryConfig validation."""

    def test_defaults(self):
        c = RetryConfig()
        assert c.max_retries == 0
        assert c.base_delay == 1.0
        assert c.exponential is True
        assert c.jitter is True

    def test_bounds(self):
        with pytest.raises(ValidationError):
            RetryConfig(max_retries=-1)
        with pytest.raises(ValidationError):
            RetryConfig(max_retries=11)
        with pytest.raises(ValidationError):
            RetryConfig(base_delay=0.01)


class TestCIConfig:
    """Tests for CIConfig."""

    def test_defaults(self):
        c = CIConfig()
        assert c.fail_on == ["REGRESSION"]
        assert "TOOLS_CHANGED" in c.warn_on

    def test_custom_fail_on(self):
        c = CIConfig(fail_on=["REGRESSION", "TOOLS_CHANGED"])
        assert len(c.fail_on) == 2


class TestJudgeConfig:
    """Tests for JudgeConfig."""

    def test_defaults_are_none(self):
        c = JudgeConfig()
        assert c.provider is None
        assert c.model is None

    def test_custom_values(self):
        c = JudgeConfig(provider="anthropic", model="sonnet")
        assert c.provider == "anthropic"
        assert c.model == "sonnet"


class TestMonitorConfig:
    """Tests for MonitorConfig."""

    def test_defaults(self):
        c = MonitorConfig()
        assert c.interval == 300
        assert c.slack_webhook is None
        assert c.discord_webhook is None
        assert c.fail_on == ["REGRESSION"]
        assert c.timeout == 30.0

    def test_interval_lower_bound(self):
        with pytest.raises(ValidationError):
            MonitorConfig(interval=5)

    def test_timeout_lower_bound(self):
        with pytest.raises(ValidationError):
            MonitorConfig(timeout=0.5)


class TestEvalViewConfig:
    """Tests for the full EvalViewConfig model."""

    def test_minimal_config(self):
        c = EvalViewConfig(adapter="http", endpoint="http://localhost:8000/invoke")
        assert c.adapter == "http"
        assert c.timeout == 30.0
        assert c.allow_private_urls is True

    def test_get_scoring_weights_default(self):
        c = EvalViewConfig(adapter="http", endpoint="http://x")
        w = c.get_scoring_weights()
        assert isinstance(w, ScoringWeights)

    def test_get_scoring_weights_custom(self):
        c = EvalViewConfig(
            adapter="http",
            endpoint="http://x",
            scoring=ScoringConfig(weights=ScoringWeights(
                tool_accuracy=0.4, output_quality=0.4, sequence_correctness=0.2
            )),
        )
        assert c.get_scoring_weights().tool_accuracy == 0.4

    def test_get_diff_config_default(self):
        c = EvalViewConfig(adapter="http", endpoint="http://x")
        d = c.get_diff_config()
        assert isinstance(d, DiffConfig)
        assert d.score_regression_threshold == 5.0

    def test_get_monitor_config_default(self):
        c = EvalViewConfig(adapter="http", endpoint="http://x")
        m = c.get_monitor_config()
        assert m.interval == 300

    def test_get_judge_config_none(self):
        c = EvalViewConfig(adapter="http", endpoint="http://x")
        assert c.get_judge_config() is None

    def test_budget_optional(self):
        c = EvalViewConfig(adapter="http", endpoint="http://x", budget=1.5)
        assert c.budget == 1.5


class TestApplyJudgeConfig:
    """Tests for apply_judge_config."""

    def test_none_config_noop(self):
        apply_judge_config(None)  # Should not raise

    def test_no_judge_section_noop(self):
        c = EvalViewConfig(adapter="http", endpoint="http://x")
        apply_judge_config(c)  # Should not raise

    def test_sets_env_vars(self, monkeypatch):
        monkeypatch.delenv("EVAL_PROVIDER", raising=False)
        monkeypatch.delenv("EVAL_MODEL", raising=False)
        c = EvalViewConfig(
            adapter="http",
            endpoint="http://x",
            judge=JudgeConfig(provider="anthropic", model="sonnet"),
        )
        apply_judge_config(c)
        assert os.environ.get("EVAL_PROVIDER") == "anthropic"
        # Model should be resolved via alias
        assert os.environ.get("EVAL_MODEL") is not None

    def test_existing_env_takes_priority(self, monkeypatch):
        monkeypatch.setenv("EVAL_PROVIDER", "openai")
        c = EvalViewConfig(
            adapter="http",
            endpoint="http://x",
            judge=JudgeConfig(provider="anthropic"),
        )
        apply_judge_config(c)
        assert os.environ["EVAL_PROVIDER"] == "openai"  # Not overwritten


class TestDefaultWeights:
    """Test the module-level DEFAULT_WEIGHTS constant."""

    def test_is_scoring_weights_instance(self):
        assert isinstance(DEFAULT_WEIGHTS, ScoringWeights)

    def test_sums_to_one(self):
        total = DEFAULT_WEIGHTS.tool_accuracy + DEFAULT_WEIGHTS.output_quality + DEFAULT_WEIGHTS.sequence_correctness
        assert abs(total - 1.0) < 0.001
