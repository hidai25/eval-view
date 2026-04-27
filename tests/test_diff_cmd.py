from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from click.testing import CliRunner

from evalview.commands.diff_cmd import diff_cmd


def _case(
    name: str,
    *,
    output: str = "same output",
    tools: Optional[List[str]] = None,
    cost: float = 0.001,
    latency_ms: float = 100.0,
) -> Dict[str, Any]:
    # Mirror the real EvaluationResult shape: `evaluations.latency.total_latency`
    # is the canonical field, in milliseconds (set in adapters via
    # `(end - start).total_seconds() * 1000`). Using the real field name in the
    # fixture ensures these tests would catch a regression in `_latency_ms`'s
    # field-name list.
    return {
        "test_case": name,
        "passed": True,
        "score": 90.0,
        "evaluations": {
            "sequence_correctness": {"actual_sequence": tools or [], "expected_sequence": []},
            "cost": {"total_cost": cost, "passed": True},
            "latency": {"total_latency": latency_ms, "passed": True},
        },
        "actual_output": output,
    }


def _write_results(path: Path, cases: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(cases), encoding="utf-8")


def test_diff_happy_path_renders_table_added_and_removed(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_results(
        before,
        [
            _case("same", output="hello", tools=["calculator"], cost=0.001, latency_ms=120),
            _case("changed", output="alpha beta", tools=["calculator"], cost=0.002, latency_ms=200),
            _case("removed", output="gone"),
        ],
    )
    _write_results(
        after,
        [
            _case("same", output="hello", tools=["calculator"], cost=0.001, latency_ms=120),
            _case(
                "changed",
                output="alpha beta gamma",
                tools=["calculator", "search"],
                cost=0.0015,
                latency_ms=180,
            ),
            _case("added", output="new"),
        ],
    )

    result = CliRunner().invoke(diff_cmd, [str(before), str(after)])

    assert result.exit_code == 0
    assert "Test Case" in result.output
    assert "Tool Sequence" in result.output
    assert "Output Sim" in result.output
    assert "Cost" in result.output
    assert "Latency" in result.output
    assert "same" in result.output
    assert "changed" in result.output
    assert "Added" in result.output
    assert "added" in result.output
    assert "Removed" in result.output
    assert "removed" in result.output


def test_diff_json_outputs_machine_readable_payload(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_results(
        before,
        [
            _case("same", output="hello", tools=["calculator"], cost=0.001, latency_ms=100),
            _case("changed", output="abc", tools=["calculator"], cost=0.002, latency_ms=100),
            _case("removed"),
        ],
    )
    _write_results(
        after,
        [
            _case("same", output="hello", tools=["calculator"], cost=0.001, latency_ms=100),
            _case(
                "changed",
                output="abcd",
                tools=["calculator", "search"],
                cost=0.001,
                latency_ms=125,
            ),
            _case("added"),
        ],
    )

    result = CliRunner().invoke(diff_cmd, [str(before), str(after), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["added"] == ["added"]
    assert payload["removed"] == ["removed"]
    assert [row["test_case"] for row in payload["compared"]] == ["changed", "same"]

    changed = payload["compared"][0]
    assert changed["tool_sequence"] == {
        "before": ["calculator"],
        "after": ["calculator", "search"],
        "changed": True,
    }
    assert changed["output_similarity"] < 100
    assert round(changed["cost_delta"], 6) == -0.001
    assert changed["latency_delta_ms"] == 25.0


def test_diff_missing_file_exits_1(tmp_path: Path) -> None:
    existing = tmp_path / "existing.json"
    _write_results(existing, [])

    result = CliRunner().invoke(diff_cmd, [str(existing), str(tmp_path / "missing.json")])

    assert result.exit_code == 1
    assert "File not found" in result.output


def test_diff_invalid_json_exits_1(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text("not json", encoding="utf-8")
    _write_results(after, [])

    result = CliRunner().invoke(diff_cmd, [str(before), str(after)])

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output


def test_diff_json_object_instead_of_list_exits_1(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps({"test_case": "not-a-list"}), encoding="utf-8")
    _write_results(after, [])

    result = CliRunner().invoke(diff_cmd, [str(before), str(after)])

    assert result.exit_code == 1
    assert "expected a list" in result.output


# ---------------------------------------------------------------------------
# Branches the original PR's tests didn't exercise: the seconds-based latency
# fallback, malformed items at the per-item level, and graceful degradation
# when the entire `evaluations` section is missing.
# ---------------------------------------------------------------------------


def test_diff_latency_total_seconds_fallback_is_converted_to_ms(tmp_path: Path) -> None:
    # Some external result formats (and older EvalView versions) expose
    # latency as `total_seconds`. _latency_ms must convert to ms so deltas
    # are comparable across formats.
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    seconds_case = {
        "test_case": "x",
        "passed": True,
        "evaluations": {
            "sequence_correctness": {"actual_sequence": [], "expected_sequence": []},
            "cost": {"total_cost": 0.001, "passed": True},
            "latency": {"total_seconds": 1.5, "passed": True},  # 1500 ms
        },
        "actual_output": "hi",
    }
    seconds_case_after = {**seconds_case, "evaluations": {**seconds_case["evaluations"]}}
    seconds_case_after["evaluations"] = {**seconds_case["evaluations"]}
    seconds_case_after["evaluations"]["latency"] = {"total_seconds": 2.0, "passed": True}

    _write_results(before, [seconds_case])
    _write_results(after, [seconds_case_after])

    result = CliRunner().invoke(diff_cmd, [str(before), str(after), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    row = payload["compared"][0]
    assert row["latency_before_ms"] == 1500.0
    assert row["latency_after_ms"] == 2000.0
    assert row["latency_delta_ms"] == 500.0


def test_diff_missing_evaluations_section_degrades_gracefully(tmp_path: Path) -> None:
    # A result file that lacks the `evaluations` block entirely (e.g. an
    # early-failure crash result) must still diff cleanly: the row appears
    # with n/a for tool sequence / cost / latency, no exception.
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    skeleton = {"test_case": "ghost", "passed": False, "actual_output": "x"}
    _write_results(before, [skeleton])
    _write_results(after, [skeleton])

    result = CliRunner().invoke(diff_cmd, [str(before), str(after), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    row = payload["compared"][0]
    assert row["tool_sequence"]["before"] == []
    assert row["tool_sequence"]["after"] == []
    assert row["cost_before"] is None
    assert row["latency_before_ms"] is None
    assert row["cost_delta"] is None
    assert row["latency_delta_ms"] is None


def test_diff_item_missing_test_case_exits_1(tmp_path: Path) -> None:
    # _load_result_file iterates per-item to validate `test_case`; this
    # exercises that loop (vs. the top-level shape checks above).
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_results(before, [{"passed": True, "actual_output": "no-name"}])
    _write_results(after, [])

    result = CliRunner().invoke(diff_cmd, [str(before), str(after)])

    assert result.exit_code == 1
    assert "missing test_case" in result.output
