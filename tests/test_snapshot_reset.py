"""A reset may replace existing baselines only after a complete, passing snapshot."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from click.testing import CliRunner

from evalview.commands import snapshot_cmd
from evalview.core.golden import GoldenStore
from evalview.evaluators.evaluator import Evaluator


@pytest.fixture
def project(monkeypatch, tmp_path, minimal_test_case, empty_trace):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CI", "1")
    tests = Path("tests")
    tests.mkdir()
    names = ["one", "two"]
    for name in names:
        (tests / f"{name}.yaml").write_text(
            f"name: {name}\ninput: {{query: test}}\nexpected: {{}}\nthresholds: {{min_score: 0}}\n"
        )
    minimal_test_case.thresholds.max_cost = 1.0
    minimal_test_case.thresholds.max_latency = 1000.0
    result = asyncio.run(Evaluator(skip_llm_judge=True).evaluate(minimal_test_case, empty_trace))
    results = [result.model_copy(update={"test_case": name, "passed": True}) for name in names]
    store = GoldenStore()
    store.save_golden(results[0], notes="original")
    store.save_golden(results[0], variant_name="older")
    (store.golden_dir / ".gitkeep").touch()
    return store, results


def contents(store):
    return {p.name: p.read_bytes() for p in store.golden_dir.iterdir()}


def invoke_reset(*args):
    return CliRunner().invoke(snapshot_cmd.snapshot, ["--reset", "--no-judge", *args])


def test_preview_reset_rejected_before_execution_or_writes(project, monkeypatch):
    store, _ = project
    original = contents(store)
    monkeypatch.setattr(snapshot_cmd, "_execute_snapshot_tests", lambda *a, **kw: pytest.fail())
    result = invoke_reset("--preview")
    assert result.exit_code == 2
    assert "--preview cannot be combined with --reset" in result.output
    assert contents(store) == original


@pytest.mark.parametrize("failure", ["empty", "missing", "failed", "write", "invalid_file"])
def test_unsuccessful_reset_preserves_every_original(project, monkeypatch, failure):
    store, results = project
    original = contents(store)
    if failure == "empty":
        results = []
    if failure == "missing":
        results.pop()
    if failure == "failed":
        results[1] = results[1].model_copy(update={"passed": False})
    if failure in {"write", "invalid_file"}:
        save = GoldenStore.save_golden

        def fail_second(self, result, **kwargs):
            if result.test_case == "two":
                if failure == "write":
                    raise OSError("disk full")
                path = save(self, result, **kwargs)
                path.write_text("{}")
                return path
            return save(self, result, **kwargs)

        monkeypatch.setattr(GoldenStore, "save_golden", fail_second)
    monkeypatch.setattr(snapshot_cmd, "_execute_snapshot_tests", lambda *a, **kw: results)
    monkeypatch.setattr(snapshot_cmd, "_cloud_push", lambda *a: pytest.fail())
    outcome = invoke_reset("--json")
    assert outcome.exit_code == 1, outcome.output
    assert "reset failed" in json.loads(outcome.output)["error"]
    assert contents(store) == original
    assert not Path(".evalview/state.json").exists()


def test_reset_cancelled_at_draft_prompt_preserves_originals(project, monkeypatch):
    store, _ = project
    original = contents(store)
    draft = Path("tests/one.yaml")
    draft.write_text(draft.read_text() + "meta: {generated_by: evalview generate, review_status: draft}\n")
    monkeypatch.setattr(snapshot_cmd, "_execute_snapshot_tests", lambda *a, **kw: pytest.fail())
    outcome = CliRunner().invoke(snapshot_cmd.snapshot, ["--reset", "--no-judge"], input="n\n")
    assert outcome.exit_code == 0
    assert "Skipped" in outcome.output
    assert contents(store) == original


def test_interrupted_directory_replacement_restores_original(project, monkeypatch):
    store, results = project
    original = contents(store)
    rename = Path.rename

    def interrupt_after_backup(self, target):
        if target.name == "previous":
            rename(self, target)
            raise KeyboardInterrupt
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", interrupt_after_backup)
    with pytest.raises(KeyboardInterrupt):
        store.replace_all(results, notes="new", variant_name=None)
    assert contents(store) == original


@pytest.mark.parametrize("empty_live_directory", [False, True])
def test_new_process_recovers_baselines_after_abrupt_exit(project, empty_live_directory):
    store, results = project
    original = contents(store)
    Path("result.json").write_text(results[0].model_dump_json())
    script = """
import os
from pathlib import Path
from evalview.core.golden import GoldenStore
from evalview.core.types import EvaluationResult
store = GoldenStore()
rename = Path.rename
def terminate_after_backup(self, target):
    moved = rename(self, target)
    if self == store.golden_dir:
        os._exit(37)
    return moved
Path.rename = terminate_after_backup
store.replace_all([EvaluationResult.model_validate_json(Path("result.json").read_text())], "new", None)
"""
    env = os.environ.copy()
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([repo, env.get("PYTHONPATH", "")])
    child = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, timeout=15)
    assert child.returncode == 37, child.stderr
    assert not store.golden_dir.exists()
    if empty_live_directory:
        store.golden_dir.mkdir()
    reopened = GoldenStore()
    assert contents(reopened) == original
    assert len(reopened.load_all_golden_variants("one")) == 2


def test_writer_waits_for_reset_publication(project, monkeypatch):
    store, results = project
    backed_up = threading.Event()
    release_reset = threading.Event()
    writer_started = threading.Event()
    rename = Path.rename

    def pause_after_backup(self, target):
        path = rename(self, target)
        if self == store.golden_dir:
            backed_up.set()
            assert release_reset.wait(timeout=5)
        return path

    def competing_write():
        writer_started.set()
        GoldenStore().save_golden(results[0], notes="concurrent", variant_name="additional")

    monkeypatch.setattr(Path, "rename", pause_after_backup)
    with ThreadPoolExecutor(2) as pool:
        reset = pool.submit(store.replace_all, results, "new", None)
        assert backed_up.wait(timeout=5)
        writer = pool.submit(competing_write)
        try:
            assert writer_started.wait(timeout=5)
            assert not writer.done()
        finally:
            release_reset.set()
        reset.result(timeout=5)
        writer.result(timeout=5)
    assert set(contents(store)) == {
        "one.golden.json", "two.golden.json", "one.variant_additional.golden.json", ".gitkeep"
    }


def test_complete_reset_replaces_old_variants_and_reports_real_paths(project, monkeypatch):
    store, results = project
    monkeypatch.setattr(snapshot_cmd, "_execute_snapshot_tests", lambda *a, **kw: results)
    outcome = invoke_reset("--notes", "new", "--variant", "current", "--json")
    assert outcome.exit_code == 0, outcome.output
    data = json.loads(outcome.output)
    assert data["snapshot"]["saved"] == 2
    assert set(contents(store)) == {
        "one.variant_current.golden.json", "two.variant_current.golden.json", ".gitkeep"
    }
    for item in data["tests"]:
        metadata = json.loads(Path(item["golden_file"]).read_text())["metadata"]
        assert metadata["test_name"] == item["name"]
        assert metadata["notes"] == "new"
