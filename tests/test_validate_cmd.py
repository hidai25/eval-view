"""Tests for the `evalview validate` command."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from evalview.commands.validate_cmd import _toml_loader, _validate_file, validate

requires_toml = pytest.mark.skipif(
    _toml_loader is None,
    reason="needs Python 3.11+ stdlib tomllib or the tomli backport",
)


def _good_yaml() -> str:
    return (
        "name: test_case\n"
        "input:\n"
        "  query: hello\n"
        "expected:\n"
        "  tools: []\n"
        "thresholds:\n"
        "  min_score: 0\n"
    )


def test_validate_dir_all_valid_exits_zero(tmp_path):
    (tmp_path / "ok.yaml").write_text(_good_yaml())
    runner = CliRunner()
    result = runner.invoke(validate, [str(tmp_path)])
    assert result.exit_code == 0


def test_validate_invalid_exits_one(tmp_path):
    (tmp_path / "bad.yaml").write_text("query: hello\n")
    runner = CliRunner()
    result = runner.invoke(validate, [str(tmp_path)])
    assert result.exit_code == 1


def test_validate_json_output_is_parseable(tmp_path):
    (tmp_path / "bad.yaml").write_text("query: hello\n")
    runner = CliRunner()
    result = runner.invoke(validate, [str(tmp_path), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["valid"] is False
    assert payload["files_checked"] == 1
    assert payload["files_with_errors"] == 1
    assert payload["results"][0]["errors"]


def test_validate_reports_all_errors_across_files(tmp_path):
    (tmp_path / "bad1.yaml").write_text("query: hello\n")
    (tmp_path / "bad2.yaml").write_text("query: world\n")
    (tmp_path / "ok.yaml").write_text(_good_yaml())
    runner = CliRunner()
    result = runner.invoke(validate, [str(tmp_path), "--json"])
    payload = json.loads(result.output)
    assert payload["files_checked"] == 3
    assert payload["files_with_errors"] == 2


def test_validate_skips_config_files(tmp_path):
    (tmp_path / "config.yaml").write_text("foo: bar\n")
    (tmp_path / "ok.yaml").write_text(_good_yaml())
    runner = CliRunner()
    result = runner.invoke(validate, [str(tmp_path), "--json"])
    payload = json.loads(result.output)
    assert payload["files_checked"] == 1


def test_validate_yaml_parse_error_reports_filename(tmp_path):
    f = tmp_path / "broken.yaml"
    f.write_text("foo: [unclosed\n")
    runner = CliRunner()
    result = runner.invoke(validate, [str(tmp_path), "--json"])
    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["results"][0]["errors"][0]["type"] == "parse"
    assert "broken.yaml" in payload["results"][0]["file"]


def test_validate_single_file(tmp_path):
    f = tmp_path / "ok.yaml"
    f.write_text(_good_yaml())
    runner = CliRunner()
    result = runner.invoke(validate, [str(f)])
    assert result.exit_code == 0


def test_validate_no_files_exits_zero(tmp_path):
    runner = CliRunner()
    result = runner.invoke(validate, [str(tmp_path)])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Coverage for branches that the original PR's test suite did not exercise:
# TOML happy/sad path, the "top-level must be a mapping" structure error,
# the schema_error fallback for TestCase pre-validators, and the dead-code
# missing-dependency message (kept defensively even though file collection
# currently shields it from production reach).
# ---------------------------------------------------------------------------


@requires_toml
def test_validate_toml_valid_file(tmp_path):
    f = tmp_path / "ok.toml"
    f.write_text(
        'name = "test_case"\n'
        "[input]\n"
        'query = "hello"\n'
        "[expected]\n"
        "tools = []\n"
        "[thresholds]\n"
        "min_score = 0\n"
    )
    runner = CliRunner()
    result = runner.invoke(validate, [str(f)])
    assert result.exit_code == 0


@requires_toml
def test_validate_toml_schema_error(tmp_path):
    # No `name` field — TestCase requires it.
    f = tmp_path / "bad.toml"
    f.write_text("[input]\nquery = 'hello'\n")
    runner = CliRunner()
    result = runner.invoke(validate, [str(f), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["results"][0]["errors"]


def test_validate_top_level_not_mapping(tmp_path):
    # YAML list at top level — TestCase expects a mapping.
    f = tmp_path / "list.yaml"
    f.write_text("- a\n- b\n")
    runner = CliRunner()
    result = runner.invoke(validate, [str(f), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["results"][0]["errors"][0]["type"] == "structure"


def test_validate_schema_construction_error(tmp_path):
    # Multi-turn pre-validator (_populate_input_from_first_turn) raises
    # KeyError when turns[0] lacks `query`. This must be caught and reported
    # as schema_error rather than aborting the whole validate run.
    f = tmp_path / "bad_multiturn.yaml"
    f.write_text(
        "name: bad\n"
        "turns:\n"
        "  - context: missing-query-field\n"
        "  - query: second\n"
    )
    runner = CliRunner()
    result = runner.invoke(validate, [str(f), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["results"][0]["errors"][0]["type"] == "schema_error"


def test_validate_file_missing_toml_loader_message(tmp_path, monkeypatch):
    # Direct call to _validate_file: when no TOML loader is available, the
    # caller (a future refactor or a hand-built path) gets a clear, actionable
    # message rather than a confusing AttributeError on `_toml_loader.load`.
    monkeypatch.setattr("evalview.commands.validate_cmd._toml_loader", None)
    f = tmp_path / "x.toml"
    f.write_text("name = 'x'\n")
    errors = _validate_file(f)
    assert len(errors) == 1
    assert errors[0]["type"] == "missing_dependency"
    assert "tomllib" in errors[0]["message"] or "tomli" in errors[0]["message"]
