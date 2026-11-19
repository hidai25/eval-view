"""Run the composite action's check step against representative CLI results."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.skipif(
    not all(shutil.which(command) for command in ("bash", "jq", "bc")),
    reason="The composite action uses bash, jq and bc",
)
@pytest.mark.parametrize(
    "payload, cli_exit, status, step_exit, failed",
    [
        ("", 0, "incomplete", 1, 0),
        ({"error": "No tests"}, 1, "incomplete", 1, 0),
        ({"unchanged": 1}, 0, "passed", 0, 0),
        ({"unchanged": 1}, 1, "incomplete", 1, 0),
        ({"unchanged": 1, "requested_tests": 2, "execution_failures": 1}, 1, "incomplete", 1, 1),
        ({"regressions": 1}, 1, "regression", 1, 1),
        ({"tools_changed": 1}, 0, "tools_changed", 0, 1),
    ],
)
def test_check_action_outputs(tmp_path, payload, cli_exit, status, step_exit, failed):
    if isinstance(payload, dict) and "error" not in payload:
        counts = dict(total_tests=1, unchanged=0, regressions=0, tools_changed=0, output_changed=0)
        payload = {"summary": {**counts, **payload}}
    (tmp_path / "result").write_text(json.dumps(payload) if isinstance(payload, dict) else payload)
    cli = tmp_path / "evalview"
    cli.write_text('#!/bin/sh\ncat result\nexit "$MOCK_EXIT"\n')
    cli.chmod(0o755)
    action = yaml.safe_load((Path(__file__).resolve().parents[1] / "action.yml").read_text())
    step = next(step for step in action["runs"]["steps"] if step.get("id") == "run-tests")
    output = tmp_path / "output"
    (tmp_path / "evalview-report.html").write_text("Previous successful check")
    env = {
        **os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "GITHUB_OUTPUT": str(output), "INPUT_MODE": "check", "MOCK_EXIT": str(cli_exit),
        "INPUT_GENERATE_REPORT": "true",
    }
    subprocess.run(["bash", "-e", "-c", step["run"]], cwd=tmp_path, env=env, check=True)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["status"] == status
    assert int(values["exit-code"]) == step_exit
    assert int(values["failed-tests"]) == failed
    assert "report-file" not in values


def test_incomplete_check_comment_reports_the_requested_count():
    from evalview.ci.comment import generate_check_pr_comment

    comment = generate_check_pr_comment({
        "status": "incomplete",
        "summary": {"total_tests": 1, "requested_tests": 2, "unchanged": 1, "execution_failures": 1},
    })
    assert "INCOMPLETE" in comment
    assert "1/2 unchanged" in comment
