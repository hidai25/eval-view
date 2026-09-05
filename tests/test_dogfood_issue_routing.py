"""Execute the shared issue router against a mocked GitHub client, without network."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROUTER = Path(__file__).resolve().parents[1] / "dogfood/update-issue.cjs"
TITLES = {"core": "🐕 Core dogfood is failing", "live": "🐕 Live provider dogfood is failing"}
NODE_HARNESS = """
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const calls = [];
const issues = {};
for (const name of ['createLabel', 'listForRepo', 'createComment', 'update', 'create']) {
  issues[name] = async args => { calls.push({name, args}); return {}; };
}
const github = {rest: {issues}, paginate: async (method, args) => {
  calls.push({name: 'listForRepo', args});
  return input.issues;
}};
const core = {info: message => calls.push({name: 'info', message})};
const context = {repo: {owner: 'example', repo: 'evalview'}, runId: 42,
  eventName: input.event || 'push', ref: input.ref || 'refs/heads/main',
  payload: input.payload || {}};
require(process.argv[1])({github, context, core}).then(
  () => console.log(JSON.stringify({calls})),
  error => console.log(JSON.stringify({calls, error: error.message})),
);
"""


def run_router(tmp_path, *, scope="core", failed="", has_failures="false", issues=None, **context):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute the GitHub Actions issue helper")
    result = subprocess.run(
        [node, "-e", NODE_HARNESS, str(ROUTER)],
        input=json.dumps({"issues": issues or [], **context}),
        env={**os.environ, "DOGFOOD_SCOPE": scope, "FAILED_CHECKS": failed,
             "HAS_FAILURES": has_failures, "GITHUB_SERVER_URL": "https://github.com"},
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def scoped_issue(scope, number):
    return {"number": number, "title": TITLES[scope], "labels": [{"name": f"dogfood-{scope}"}]}


@pytest.mark.parametrize("scope", ["core", "live"])
def test_green_only_closes_its_exact_title_and_label(tmp_path, scope):
    other_scope = "live" if scope == "core" else "core"
    issues = [
        {"number": 264, "title": "🐕 Daily dogfood is failing", "labels": [{"name": "dogfood"}]},
        scoped_issue(other_scope, 300),
        {"number": 301, "title": TITLES[scope], "labels": [{"name": "dogfood"}]},
        scoped_issue(scope, 302),
    ]
    result = run_router(tmp_path, scope=scope, issues=issues)
    updates = [call for call in result["calls"] if call["name"] == "update"]
    assert [call["args"]["issue_number"] for call in updates] == [302]
    comment = next(call["args"]["body"] for call in result["calls"] if call["name"] == "createComment")
    assert ("deterministic core checks" if scope == "core" else "live provider checks") in comment
    if scope == "core":
        assert "does not establish live-provider recovery" in comment


@pytest.mark.parametrize("scope,failed", [("core", "pytest"), ("live", "provider")])
def test_failure_creates_separate_scoped_issue_and_artifact_reference(tmp_path, scope, failed):
    result = run_router(tmp_path, scope=scope, failed=failed, has_failures="true")
    created = next(call["args"] for call in result["calls"] if call["name"] == "create")
    assert created["title"] == TITLES[scope]
    assert created["labels"] == ["dogfood", f"dogfood-{scope}"]
    assert f"{scope}-dogfood-evidence-42" in created["body"]
    assert not any(call["name"] == "update" for call in result["calls"])


def test_missing_aggregation_never_closes_existing_incident(tmp_path):
    result = run_router(tmp_path, has_failures="", issues=[scoped_issue("core", 302)])
    assert not any(call["name"] == "update" for call in result["calls"])
    comment = next(call["args"]["body"] for call in result["calls"] if call["name"] == "createComment")
    assert "workflow-setup" in comment


def test_live_failure_cannot_leak_into_core_evidence(tmp_path):
    (tmp_path / "provider-output.txt").write_text("LIVE PROVIDER EVIDENCE")
    result = run_router(tmp_path, failed="provider", has_failures="true")
    created = next(call["args"] for call in result["calls"] if call["name"] == "create")
    assert "workflow-setup" in created["body"]
    assert "LIVE PROVIDER EVIDENCE" not in created["body"]


@pytest.mark.parametrize("context", [
    {"event": "pull_request", "ref": "refs/pull/1/merge"},
    {"event": "pull_request_target"},
    {"event": "workflow_dispatch", "ref": "refs/heads/feature"},
    {"payload": {"pull_request": {"number": 1}}},
])
def test_pr_and_branch_runs_make_no_github_calls(tmp_path, context):
    result = run_router(tmp_path, has_failures="true", failed="pytest", **context)
    assert all(call["name"] == "info" for call in result["calls"])


def test_unknown_scope_makes_no_github_calls(tmp_path):
    result = run_router(tmp_path, scope="")
    assert "DOGFOOD_SCOPE" in result["error"]
    assert result["calls"] == []
