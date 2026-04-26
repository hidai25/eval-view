"""Tests for the `evalview validate` command."""

from __future__ import annotations

import json

from click.testing import CliRunner

from evalview.commands.validate_cmd import validate


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
