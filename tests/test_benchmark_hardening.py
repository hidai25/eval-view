"""Tests for benchmark hardening / anti-gaming checks."""

from datetime import datetime

import pytest

from evalview.core.benchmark_hardening import (
    GamingCheck,
    GamingFlag,
    FlagSeverity,
    check_gaming,
    check_gaming_batch,
    _check_suspiciously_fast,
    _check_config_leakage,
    _check_score_without_work,
    _check_too_perfect,
    _check_abnormal_file_access,
    _compute_trust_score,
)
from evalview.core.types import (
    ExecutionTrace,
    ExecutionMetrics,
    StepTrace,
    StepMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    tool: str,
    params: dict = None,
    output: str = "ok",
) -> StepTrace:
    return StepTrace(
        step_id=f"s-{tool}",
        step_name=tool,
        tool_name=tool,
        parameters=params or {},
        output=output,
        success=True,
        metrics=StepMetrics(latency=100, cost=0.01),
    )


_SENTINEL = object()


def _trace(
    steps=_SENTINEL,
    latency: float = 5000.0,
    cost: float = 0.1,
) -> ExecutionTrace:
    if steps is _SENTINEL:
        steps = [_step("search"), _step("analyze"), _step("respond")]
    return ExecutionTrace(
        session_id="test",
        start_time=datetime(2025, 1, 1),
        end_time=datetime(2025, 1, 1, 0, 1),
        steps=steps,
        final_output="done",
        metrics=ExecutionMetrics(total_cost=cost, total_latency=latency),
    )


# ---------------------------------------------------------------------------
# Suspiciously fast
# ---------------------------------------------------------------------------


class TestSuspiciouslyFast:
    def test_normal_latency_no_flag(self):
        trace = _trace(latency=5000)
        assert _check_suspiciously_fast(trace) == []

    def test_fast_latency_flagged(self):
        trace = _trace(latency=100)
        flags = _check_suspiciously_fast(trace)
        assert len(flags) == 1
        assert flags[0].check == GamingCheck.SUSPICIOUSLY_FAST

    def test_very_fast_is_suspicious(self):
        trace = _trace(latency=50)  # Under half the threshold
        flags = _check_suspiciously_fast(trace)
        assert flags[0].severity == FlagSeverity.SUSPICIOUS

    def test_zero_latency_skipped(self):
        """Latency=0 means not measured, should not flag."""
        trace = _trace(latency=0)
        assert _check_suspiciously_fast(trace) == []


# ---------------------------------------------------------------------------
# Config leakage
# ---------------------------------------------------------------------------


class TestConfigLeakage:
    def test_no_leakage_normal_trace(self):
        trace = _trace()
        assert _check_config_leakage(trace) == []

    def test_leakage_in_params(self):
        steps = [
            _step("read_file", {"path": "/app/.evalview/golden/test.json"}),
        ]
        trace = _trace(steps=steps)
        flags = _check_config_leakage(trace)
        assert len(flags) == 1
        assert flags[0].check == GamingCheck.CONFIG_LEAKAGE

    def test_leakage_ground_truth(self):
        steps = [
            _step("read_file", {"path": "/data/ground_truth.json"}),
        ]
        trace = _trace(steps=steps)
        flags = _check_config_leakage(trace)
        assert len(flags) == 1

    def test_leakage_in_output(self):
        steps = [
            _step("bash", {"cmd": "cat file.txt"}, output="evalview.yaml config found"),
        ]
        trace = _trace(steps=steps)
        flags = _check_config_leakage(trace)
        assert len(flags) == 1

    def test_multiple_leakage_is_critical(self):
        steps = [
            _step("read", {"path": ".evalview/golden/test.json"}),
            _step("read", {"path": "/data/answer_key.txt"}),
        ]
        trace = _trace(steps=steps)
        flags = _check_config_leakage(trace)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Score without work
# ---------------------------------------------------------------------------


class TestScoreWithoutWork:
    def test_high_score_with_tools_ok(self):
        trace = _trace()  # 3 steps
        assert _check_score_without_work(trace, score=95) == []

    def test_high_score_no_tools_flagged(self):
        trace = _trace(steps=[])
        flags = _check_score_without_work(trace, score=95)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.CRITICAL

    def test_moderate_score_no_tools_less_severe(self):
        trace = _trace(steps=[])
        flags = _check_score_without_work(trace, score=82)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.INFO

    def test_low_score_no_tools_no_flag(self):
        trace = _trace(steps=[])
        assert _check_score_without_work(trace, score=50) == []


# ---------------------------------------------------------------------------
# Too perfect
# ---------------------------------------------------------------------------


class TestTooPerfect:
    def test_normal_score_no_flag(self):
        trace = _trace()
        assert _check_too_perfect(trace, score=85) == []

    def test_perfect_score_flagged(self):
        trace = _trace()
        flags = _check_too_perfect(trace, score=100)
        assert len(flags) == 1
        assert flags[0].check == GamingCheck.TOO_PERFECT

    def test_perfect_fast_light_is_critical(self):
        trace = _trace(steps=[_step("x")], latency=100)
        flags = _check_too_perfect(trace, score=100)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Abnormal file access
# ---------------------------------------------------------------------------


class TestAbnormalFileAccess:
    def test_normal_params_no_flag(self):
        trace = _trace()
        assert _check_abnormal_file_access(trace) == []

    def test_suspicious_extension(self):
        steps = [_step("read", {"path": "answers.golden"})]
        trace = _trace(steps=steps)
        flags = _check_abnormal_file_access(trace)
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestCheckGaming:
    def test_clean_result(self):
        trace = _trace()
        report = check_gaming(trace, score=80)
        assert not report.has_flags
        assert report.trust_score == 1.0

    def test_multiple_flags_reduce_trust(self):
        trace = _trace(steps=[], latency=50)
        report = check_gaming(trace, score=100)
        assert report.has_flags
        assert report.trust_score < 1.0

    def test_report_summary(self):
        trace = _trace()
        report = check_gaming(trace, score=80)
        assert "No gaming signals" in report.summary()

    def test_report_to_dict(self):
        trace = _trace()
        report = check_gaming(trace, score=80)
        d = report.to_dict()
        assert "trust_score" in d
        assert "flags" in d


# ---------------------------------------------------------------------------
# Reviewer hint field — must be omitted when None and present when set.
# Cloud's TrustIndicator popover renders this verbatim (see TrustFlagDict
# in evalview.core.observability), so the on-the-wire shape matters.
# ---------------------------------------------------------------------------


class TestGamingFlagHint:
    def test_to_dict_omits_hint_when_none(self):
        flag = GamingFlag(
            check=GamingCheck.SUSPICIOUSLY_FAST,
            severity=FlagSeverity.INFO,
            description="x",
        )
        assert "hint" not in flag.to_dict()

    def test_to_dict_includes_hint_when_set(self):
        flag = GamingFlag(
            check=GamingCheck.CONFIG_LEAKAGE,
            severity=FlagSeverity.CRITICAL,
            description="x",
            hint="quarantine this test",
        )
        d = flag.to_dict()
        assert d["hint"] == "quarantine this test"

    def test_config_leakage_check_emits_hint(self):
        steps = [_step("read", {"path": ".evalview/golden/test.json"})]
        flags = _check_config_leakage(_trace(steps=steps))
        assert len(flags) == 1 and flags[0].hint
        # The hint must be actionable, not a restatement of the description.
        assert flags[0].hint != flags[0].description

    def test_score_without_work_check_emits_hint(self):
        flags = _check_score_without_work(_trace(steps=[]), score=95)
        assert len(flags) == 1 and flags[0].hint
        assert flags[0].hint != flags[0].description


# ---------------------------------------------------------------------------
# Batch checks
# ---------------------------------------------------------------------------


class TestBatchChecks:
    def test_empty_batch(self):
        report = check_gaming_batch([])
        assert not report.has_flags

    def test_all_perfect_flagged(self):
        results = [{"score": 100, "trace": _trace()} for _ in range(5)]
        report = check_gaming_batch(results)
        assert report.has_flags
        assert any(f.severity == FlagSeverity.CRITICAL for f in report.flags)

    def test_mixed_scores_ok(self):
        results = [
            {"score": 100},
            {"score": 85},
            {"score": 72},
        ]
        report = check_gaming_batch(results)
        assert not report.has_flags


# ---------------------------------------------------------------------------
# Suspiciously fast — boundary cases
# ---------------------------------------------------------------------------


class TestSuspiciouslyFastBoundary:
    def test_info_severity_between_half_and_full(self):
        """Latency=400 is above 250 (half of 500) but below 500 threshold → INFO."""
        trace = _trace(latency=400)
        flags = _check_suspiciously_fast(trace)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.INFO

    def test_zero_steps_not_flagged(self):
        """Fast trace with zero steps should not be flagged."""
        trace = _trace(steps=[], latency=100)
        flags = _check_suspiciously_fast(trace)
        assert flags == []


# ---------------------------------------------------------------------------
# Too perfect — latency=0 and many-tool cases
# ---------------------------------------------------------------------------


class TestTooPerfectLatencyZero:
    def test_unmeasured_latency_not_fast(self):
        """score=100, latency=0, 1 step: should NOT be CRITICAL (latency=0 = unmeasured)."""
        trace = _trace(steps=[_step("x")], latency=0)
        flags = _check_too_perfect(trace, score=100)
        assert len(flags) == 1
        # latency=0 means unmeasured, so is_fast should be False
        # With 1 step (is_light=True but not is_fast), severity should be SUSPICIOUS
        assert flags[0].severity != FlagSeverity.CRITICAL

    def test_perfect_with_many_tools_high_latency_is_info(self):
        """score=100, latency=5000, 10 steps: should be INFO severity."""
        steps = [_step(f"tool_{i}") for i in range(10)]
        trace = _trace(steps=steps, latency=5000)
        flags = _check_too_perfect(trace, score=100)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.INFO


# ---------------------------------------------------------------------------
# Abnormal file access — substring fix
# ---------------------------------------------------------------------------


class TestAbnormalFileAccessBoundary:
    def test_evalview_path_not_matched(self):
        """Params containing '.evalview/' should NOT match '.eval' extension."""
        steps = [_step("read", {"path": "/app/.evalview/config.yaml"})]
        trace = _trace(steps=steps)
        flags = _check_abnormal_file_access(trace)
        # ".evalview" should NOT match ".eval" because the regex requires
        # a non-alphanumeric char or end-of-string after the extension
        assert flags == []


# ---------------------------------------------------------------------------
# Trust score clamping
# ---------------------------------------------------------------------------


class TestTrustScoreClamping:
    def test_trust_never_below_zero(self):
        """Many flags should clamp trust to 0.0."""
        flags = [
            GamingFlag(check=GamingCheck.CONFIG_LEAKAGE, severity=FlagSeverity.CRITICAL, description="a"),
            GamingFlag(check=GamingCheck.TOO_PERFECT, severity=FlagSeverity.CRITICAL, description="b"),
            GamingFlag(check=GamingCheck.SCORE_WITHOUT_WORK, severity=FlagSeverity.CRITICAL, description="c"),
            GamingFlag(check=GamingCheck.SUSPICIOUSLY_FAST, severity=FlagSeverity.CRITICAL, description="d"),
        ]
        trust = _compute_trust_score(flags)
        assert trust == 0.0

    def test_trust_exact_values(self):
        """Single CRITICAL flag: 1.0 - 0.3 = 0.7."""
        flags = [
            GamingFlag(check=GamingCheck.CONFIG_LEAKAGE, severity=FlagSeverity.CRITICAL, description="x"),
        ]
        trust = _compute_trust_score(flags)
        assert trust == 0.7


# ---------------------------------------------------------------------------
# Batch — partial perfect
# ---------------------------------------------------------------------------


class TestBatchPartialPerfect:
    def test_eighty_percent_perfect_flagged(self):
        """5 results, 4 perfect, 1 imperfect → should flag SUSPICIOUS."""
        results = [
            {"score": 100},
            {"score": 100},
            {"score": 100},
            {"score": 100},
            {"score": 70},
        ]
        report = check_gaming_batch(results)
        assert report.has_flags
        assert any(f.severity == FlagSeverity.SUSPICIOUS for f in report.flags)

    def test_below_eighty_percent_ok(self):
        """5 results, 3 perfect, 2 imperfect → should NOT flag."""
        results = [
            {"score": 100},
            {"score": 100},
            {"score": 100},
            {"score": 70},
            {"score": 65},
        ]
        report = check_gaming_batch(results)
        assert not report.has_flags


# ---------------------------------------------------------------------------
# Batch timing similarity
# ---------------------------------------------------------------------------


class TestBatchTimingSimilarity:
    def test_identical_latencies_flagged(self):
        """All tests completing in identical time should be flagged."""
        results = [
            {"score": 80, "latency_ms": 1000},
            {"score": 85, "latency_ms": 1000},
            {"score": 80, "latency_ms": 1000},
        ]
        report = check_gaming_batch(results)
        timing_flags = [f for f in report.flags if f.check == GamingCheck.SUSPICIOUSLY_FAST]
        assert len(timing_flags) == 1
        assert timing_flags[0].severity == FlagSeverity.SUSPICIOUS

    def test_varied_latencies_not_flagged(self):
        """Normal latency variance should not be flagged."""
        results = [
            {"score": 80, "latency_ms": 1000},
            {"score": 85, "latency_ms": 3500},
            {"score": 80, "latency_ms": 2200},
        ]
        report = check_gaming_batch(results)
        timing_flags = [f for f in report.flags if f.check == GamingCheck.SUSPICIOUSLY_FAST]
        assert timing_flags == []

    def test_missing_latencies_skipped(self):
        """Batch with no latency_ms fields should not crash or flag."""
        results = [
            {"score": 80},
            {"score": 85},
            {"score": 80},
        ]
        report = check_gaming_batch(results)
        timing_flags = [f for f in report.flags if f.check == GamingCheck.SUSPICIOUSLY_FAST]
        assert timing_flags == []


# ---------------------------------------------------------------------------
# Exhaustive leakage pattern tests
# ---------------------------------------------------------------------------


class TestAllLeakagePatterns:
    """Ensure every LEAKAGE_PATTERNS regex actually matches."""

    @pytest.mark.parametrize("path,pattern_desc", [
        ("/app/.evalview/golden/test.json", ".evalview/"),
        ("/app/evalview.yaml", "evalview.yaml"),
        ("/app/evalview.yml", "evalview.yml"),
        ("/data/golden/baseline.json", "golden/"),
        ("/data/baseline/expected.json", "baseline/"),
        ("/data/ground_truth.json", "ground_truth"),
        ("/data/ground-truth.json", "ground-truth"),
        ("/data/groundtruth.json", "groundtruth"),
        ("/keys/answer_key.txt", "answer_key"),
        ("/keys/answer-key.txt", "answer-key"),
        ("/keys/answerkey.txt", "answerkey"),
        ("/out/expected_output.json", "expected_output"),
        ("/out/expected-output.json", "expected-output"),
        ("/cfg/test_config.yaml", "test_config"),
        ("/cfg/test-config.yaml", "test-config"),
        ("/app/.eval/results.json", ".eval/"),
        ("/app/eval_results.json", "eval_results"),
        ("/app/eval-results.json", "eval-results"),
        ("/rubrics/scoring_rubric.md", "scoring_rubric"),
        ("/rubrics/scoring-rubric.md", "scoring-rubric"),
        ("/prompts/judge_prompt.txt", "judge_prompt"),
        ("/prompts/judge-prompt.txt", "judge-prompt"),
    ])
    def test_leakage_pattern_matches(self, path, pattern_desc):
        steps = [_step("read_file", {"path": path})]
        trace = _trace(steps=steps)
        flags = _check_config_leakage(trace)
        assert len(flags) == 1, f"Expected leakage flag for '{pattern_desc}' in path '{path}'"


# ---------------------------------------------------------------------------
# Exhaustive suspicious extension tests
# ---------------------------------------------------------------------------


class TestAllSuspiciousExtensions:
    """Ensure every SUSPICIOUS_EXTENSIONS entry is detected."""

    @pytest.mark.parametrize("ext", [
        ".eval", ".golden", ".baseline", ".answer",
        ".rubric", ".scoring", ".judge",
    ])
    def test_extension_flagged(self, ext):
        filename = f"answers{ext}"
        steps = [_step("read", {"path": f"/data/{filename}"})]
        trace = _trace(steps=steps)
        flags = _check_abnormal_file_access(trace)
        assert len(flags) == 1, f"Expected flag for extension '{ext}'"
