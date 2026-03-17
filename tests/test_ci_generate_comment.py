"""Tests for PR comment generation including alerts, collapsing, and job summary."""

from __future__ import annotations

import json
import os
import tempfile

from click.testing import CliRunner


def _generate_report() -> dict:
    return {
        "report_version": 1,
        "source": "logs",
        "probes_run": 12,
        "tests_generated": 5,
        "discovery": {
            "count": 2,
            "tools": [
                {"name": "weather_api", "description": "Get the weather"},
                {"name": "calculator", "description": "Perform arithmetic"},
            ],
        },
        "behavior_signatures": {
            "tool_path:weather_api": 2,
            "tool_path:calculator": 1,
            "refusal": 1,
        },
        "covered": {
            "tool_paths": 2,
            "direct_answers": 1,
            "clarifications": 1,
            "multi_turn": 1,
            "refusals": 1,
            "error_paths": 0,
        },
        "draft_tests": [
            {
                "name": "Weather Test",
                "signature": "tool_path:weather_api",
                "rationale": "Observed weather tool path",
            },
            {
                "name": "Refusal Test",
                "signature": "refusal",
                "rationale": "Observed refusal path",
            },
        ],
        "gaps": [
            "No error-path behavior observed.",
            "Discovered but not exercised: calculator",
        ],
        "changes_since_last_generation": {
            "new_signatures": ["refusal"],
            "resolved_signatures": [],
            "new_tools": ["weather_api"],
            "resolved_gaps": ["No clarification path observed."],
            "new_gaps": ["No error-path behavior observed."],
            "tests_generated_delta": 1,
        },
    }


def _check_data_with_spikes() -> dict:
    """Check data with cost/latency spikes and model change."""
    return {
        "summary": {
            "total_tests": 5,
            "unchanged": 3,
            "regressions": 1,
            "tools_changed": 1,
            "output_changed": 0,
            "model_changed": True,
        },
        "diffs": [
            {
                "test_name": "search-flow",
                "status": "regression",
                "score_delta": -15.0,
                "tool_diffs": [{"type": "removed", "tool": "search_api"}],
                "output_similarity": 0.72,
                "current_cost": 0.05,
                "baseline_cost": 0.02,
                "current_latency": 3000,
                "baseline_latency": 1500,
                "baseline_model": "gpt-5.4",
                "current_model": "gpt-5.4-mini",
            },
            {
                "test_name": "create-flow",
                "status": "tools_changed",
                "score_delta": 0,
                "tool_diffs": [{"type": "added", "tool": "validator"}],
                "output_similarity": 0.98,
                "current_cost": 0.03,
                "baseline_cost": 0.02,
                "current_latency": 2000,
                "baseline_latency": 1000,
            },
            {
                "test_name": "list-flow",
                "status": "passed",
                "score_delta": 0,
            },
            {
                "test_name": "delete-flow",
                "status": "passed",
                "score_delta": 0,
            },
            {
                "test_name": "update-flow",
                "status": "passed",
                "score_delta": 0,
            },
        ],
    }


def _check_data_many_changes() -> dict:
    """Check data with >5 changes to test collapsible overflow."""
    diffs = []
    for i in range(8):
        diffs.append({
            "test_name": f"test-{i}",
            "status": "regression",
            "score_delta": -5.0,
            "tool_diffs": [{"type": "removed", "tool": f"tool_{i}"}],
            "output_similarity": 0.80,
        })
    return {
        "summary": {
            "total_tests": 8,
            "unchanged": 0,
            "regressions": 8,
            "tools_changed": 0,
            "output_changed": 0,
        },
        "diffs": diffs,
    }


def test_generate_suite_pr_comment_contains_review_workflow():
    """Generated-suite comments should summarize coverage and next steps."""
    from evalview.ci.comment import generate_suite_pr_comment

    comment = generate_suite_pr_comment(_generate_report(), "https://example.com/run/123")

    assert "EvalView Generate" in comment
    assert "Draft Test(s)" in comment
    assert "Discovered Tools" in comment
    assert "Changes Since Last Generation" in comment
    assert "Coverage Gaps" in comment
    assert "snapshot --approve-generated" in comment
    assert "weather_api" in comment


def test_ci_comment_detects_generate_report_format(tmp_path):
    """ci comment should render generated-suite reports via the new formatter."""
    from evalview.commands.ci_cmd import ci_comment

    report_path = tmp_path / "generated.report.json"
    report_path.write_text(json.dumps(_generate_report()), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(ci_comment, ["--results", str(report_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "EvalView Generate" in result.output
    assert "Coverage Gaps" in result.output


def test_check_comment_includes_alerts():
    """Check comments should show cost/latency spike alerts and model change."""
    from evalview.ci.comment import generate_check_pr_comment

    comment = generate_check_pr_comment(_check_data_with_spikes())

    assert "Cost spike" in comment
    assert "Latency spike" in comment
    assert "Model changed" in comment or "Model Changed" in comment
    assert "gpt-5.4" in comment
    assert "gpt-5.4-mini" in comment


def test_check_comment_model_change_in_table():
    """Model change should appear in the summary table."""
    from evalview.ci.comment import generate_check_pr_comment

    comment = generate_check_pr_comment(_check_data_with_spikes())

    # Should have model change in summary table
    assert "Model Changed" in comment
    assert "gpt-5.4" in comment


def test_check_comment_collapsible_overflow():
    """Comments with >5 changes should use <details> collapse."""
    from evalview.ci.comment import generate_check_pr_comment

    comment = generate_check_pr_comment(_check_data_many_changes())

    assert "<details>" in comment
    assert "</details>" in comment
    assert "more change(s)" in comment
    # First 5 should be visible
    assert "test-0" in comment
    assert "test-4" in comment
    # Remaining 3 should be in collapsed section
    assert "test-7" in comment


def test_check_comment_no_alerts_when_clean():
    """Clean check should not show alerts section."""
    from evalview.ci.comment import generate_check_pr_comment

    clean_data = {
        "summary": {
            "total_tests": 3,
            "unchanged": 3,
            "regressions": 0,
            "tools_changed": 0,
            "output_changed": 0,
        },
        "diffs": [],
    }
    comment = generate_check_pr_comment(clean_data)

    assert "Alerts" not in comment
    assert "Cost spike" not in comment
    assert "Latency spike" not in comment


def test_run_comment_collapsible_failed_tests():
    """Run comments with many failures should use <details> collapse."""
    from evalview.ci.comment import generate_pr_comment

    results = [
        {"test_case": f"test-{i}", "score": 30.0, "min_score": 70, "passed": False}
        for i in range(8)
    ]
    comment = generate_pr_comment(results)

    assert "<details>" in comment
    assert "more failure(s)" in comment


def test_write_job_summary():
    """write_job_summary should append to GITHUB_STEP_SUMMARY file."""
    from evalview.ci.comment import write_job_summary

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        summary_path = f.name

    try:
        os.environ["GITHUB_STEP_SUMMARY"] = summary_path
        result = write_job_summary("## Test Summary\nAll passed!")
        assert result is True

        with open(summary_path, encoding="utf-8") as f:
            content = f.read()
        assert "Test Summary" in content
        assert "All passed!" in content
    finally:
        os.environ.pop("GITHUB_STEP_SUMMARY", None)
        os.unlink(summary_path)


def test_write_job_summary_not_in_actions():
    """write_job_summary should return False when not in GitHub Actions."""
    from evalview.ci.comment import write_job_summary

    os.environ.pop("GITHUB_STEP_SUMMARY", None)
    result = write_job_summary("## Test")
    assert result is False


def test_cost_spike_detection():
    """Cost spike should be detected when increase exceeds threshold."""
    from evalview.ci.comment import _detect_cost_spike

    spike = _detect_cost_spike(_check_data_with_spikes())
    assert spike is not None
    current, baseline = spike
    assert current > baseline


def test_latency_spike_detection():
    """Latency spike should be detected when increase exceeds threshold."""
    from evalview.ci.comment import _detect_latency_spike

    spike = _detect_latency_spike(_check_data_with_spikes())
    assert spike is not None
    current, baseline = spike
    assert current > baseline


def test_no_spike_when_stable():
    """No spike should be detected when metrics are stable."""
    from evalview.ci.comment import _detect_cost_spike, _detect_latency_spike

    stable_data = {
        "diffs": [
            {
                "test_name": "test-1",
                "current_cost": 0.02,
                "baseline_cost": 0.02,
                "current_latency": 1000,
                "baseline_latency": 1000,
            },
        ],
    }
    assert _detect_cost_spike(stable_data) is None
    assert _detect_latency_spike(stable_data) is None


def test_model_change_detection():
    """Model change should be detected with old/new names."""
    from evalview.ci.comment import _detect_model_change

    result = _detect_model_change(_check_data_with_spikes())
    assert result is not None
    assert "gpt-5.4" in result
    assert "gpt-5.4-mini" in result


def test_model_change_not_detected_when_same():
    """No model change when summary says no change."""
    from evalview.ci.comment import _detect_model_change

    data = {"summary": {}, "diffs": []}
    assert _detect_model_change(data) is None
