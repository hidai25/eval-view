"""Unit tests for core/canary_suite.py and the bundled canary YAMLs."""
from __future__ import annotations

from pathlib import Path

import pytest

from evalview.benchmarks.canary import HELD_OUT_SUITE_PATH, PUBLIC_SUITE_PATH
from evalview.core.canary_suite import (
    CanarySuiteError,
    hash_suite_bytes,
    load_canary_suite,
)


# --------------------------------------------------------------------------- #
# Bundled suites
# --------------------------------------------------------------------------- #


def test_public_suite_loads_and_has_15_prompts():
    suite = load_canary_suite(PUBLIC_SUITE_PATH)
    assert suite.suite_name == "canary"
    assert suite.version == "v1.public"
    assert len(suite.prompts) == 15
    # Every prompt has a unique id and a known scorer.
    ids = [p.id for p in suite.prompts]
    assert len(ids) == len(set(ids))


def test_public_suite_category_distribution():
    suite = load_canary_suite(PUBLIC_SUITE_PATH)
    by_scorer: dict[str, int] = {}
    for p in suite.prompts:
        by_scorer[p.scorer] = by_scorer.get(p.scorer, 0) + 1
    # Locked in by design in the plan (§5.2).
    assert by_scorer == {
        "tool_choice": 5,
        "json_schema": 4,
        "refusal": 3,
        "exact_match": 3,
    }


def test_held_out_suite_loads_and_has_5_prompts():
    suite = load_canary_suite(HELD_OUT_SUITE_PATH)
    assert suite.version == "v1.held-out"
    assert len(suite.prompts) == 5


def test_public_and_held_out_hashes_differ():
    public = load_canary_suite(PUBLIC_SUITE_PATH)
    held_out = load_canary_suite(HELD_OUT_SUITE_PATH)
    assert public.suite_hash != held_out.suite_hash


def test_hash_is_deterministic_and_prefixed():
    public1 = load_canary_suite(PUBLIC_SUITE_PATH)
    public2 = load_canary_suite(PUBLIC_SUITE_PATH)
    assert public1.suite_hash == public2.suite_hash
    assert public1.suite_hash.startswith("sha256:")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(content)
    return path


def test_missing_file_raises_clear_error(tmp_path: Path):
    with pytest.raises(CanarySuiteError, match="not found"):
        load_canary_suite(tmp_path / "missing.yaml")


def test_invalid_yaml_raises(tmp_path: Path):
    path = _write(tmp_path, "prompts: [ unclosed")
    with pytest.raises(CanarySuiteError, match="Invalid YAML"):
        load_canary_suite(path)


def test_missing_suite_name_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """
version: v1
prompts:
  - id: x
    scorer: exact_match
    prompt: hello
    expected: {pattern: hello}
""",
    )
    with pytest.raises(CanarySuiteError, match="suite_name"):
        load_canary_suite(path)


def test_unknown_scorer_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """
suite_name: x
version: v1
prompts:
  - id: bad
    scorer: magic_judge
    prompt: hello
    expected: {}
""",
    )
    with pytest.raises(CanarySuiteError, match="unknown scorer"):
        load_canary_suite(path)


def test_duplicate_prompt_id_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """
suite_name: x
version: v1
prompts:
  - id: dup
    scorer: exact_match
    prompt: a
    expected: {pattern: a}
  - id: dup
    scorer: exact_match
    prompt: b
    expected: {pattern: b}
""",
    )
    with pytest.raises(CanarySuiteError, match="Duplicate prompt id"):
        load_canary_suite(path)


def test_missing_prompt_text_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """
suite_name: x
version: v1
prompts:
  - id: noprompt
    scorer: exact_match
    expected: {pattern: a}
""",
    )
    with pytest.raises(CanarySuiteError, match="prompt"):
        load_canary_suite(path)


def test_empty_prompts_list_raises(tmp_path: Path):
    path = _write(tmp_path, "suite_name: x\nversion: v1\nprompts: []\n")
    with pytest.raises(CanarySuiteError, match="non-empty"):
        load_canary_suite(path)


# --------------------------------------------------------------------------- #
# Hash helper
# --------------------------------------------------------------------------- #


def test_hash_suite_bytes_is_stable():
    raw = b"suite_name: x\nversion: v1\nprompts: []\n"
    assert hash_suite_bytes(raw) == hash_suite_bytes(raw)
    assert hash_suite_bytes(raw).startswith("sha256:")
    assert hash_suite_bytes(raw) != hash_suite_bytes(raw + b"\n")
