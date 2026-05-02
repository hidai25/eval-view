"""Tests for CSV log importer (issue #94)."""
from __future__ import annotations

from pathlib import Path

from evalview.importers.log_importer import (
    detect_format,
    parse_csv,
    parse_log_file,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_detect_format_uses_csv_extension(tmp_path):
    path = _write(tmp_path / "traces.csv", "query,output\nhi,hello\n")
    assert detect_format(path) == "csv"


def test_detect_format_falls_back_to_csv_for_recognised_header(tmp_path):
    # No .csv extension but the header is clearly tabular
    path = _write(tmp_path / "traces.log", "query,output,tools\nhi,hello,a\n")
    assert detect_format(path) == "csv"


def test_detect_format_returns_unknown_for_arbitrary_text(tmp_path):
    path = _write(tmp_path / "junk.log", "this is not structured\n")
    assert detect_format(path) == "unknown"


def test_parse_csv_happy_path(tmp_path):
    path = _write(
        tmp_path / "traces.csv",
        'query,output,tools\n'
        'What is the weather in SF?,SF is 63 and sunny.,weather_api\n'
        'Reset the production database.,I can\'t help with that.,\n',
    )
    entries = parse_csv(path)
    assert len(entries) == 2
    assert entries[0].query == "What is the weather in SF?"
    assert entries[0].output == "SF is 63 and sunny."
    assert entries[0].tool_calls == ["weather_api"]
    assert entries[1].tool_calls == []


def test_parse_csv_accepts_alias_columns(tmp_path):
    path = _write(
        tmp_path / "traces.csv",
        "input,response,actions\n"
        "Hello,Hi back,echo\n",
    )
    entries = parse_csv(path)
    assert len(entries) == 1
    assert entries[0].query == "Hello"
    assert entries[0].output == "Hi back"
    assert entries[0].tool_calls == ["echo"]


def test_parse_csv_tool_cell_separators(tmp_path):
    import csv as _csv

    path = tmp_path / "traces.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["query", "tools"])
        writer.writerow(["a", "x, y, z"])
        writer.writerow(["b", "x;y;z"])
        writer.writerow(["c", "x|y|z"])
        writer.writerow(["d", '["x", "y"]'])

    entries = parse_csv(path)
    assert [e.tool_calls for e in entries] == [
        ["x", "y", "z"],
        ["x", "y", "z"],
        ["x", "y", "z"],
        ["x", "y"],
    ]


def test_parse_csv_skips_empty_query_with_warning(tmp_path):
    path = _write(
        tmp_path / "traces.csv",
        "query,output\n"
        ",no query here\n"
        "real query,real output\n",
    )
    warnings: list[str] = []
    entries = parse_csv(path, warn=warnings.append)
    assert len(entries) == 1
    assert entries[0].query == "real query"
    assert any("empty query" in w for w in warnings)


def test_parse_csv_warns_on_missing_query_column(tmp_path):
    path = _write(
        tmp_path / "traces.csv",
        "id,foo\n1,bar\n",
    )
    warnings: list[str] = []
    entries = parse_csv(path, warn=warnings.append)
    assert entries == []
    assert any("missing a query column" in w for w in warnings)


def test_parse_csv_respects_max_entries(tmp_path):
    rows = "query\n" + "\n".join(f"q{i}" for i in range(10)) + "\n"
    path = _write(tmp_path / "traces.csv", rows)
    entries = parse_csv(path, max_entries=3)
    assert len(entries) == 3
    assert [e.query for e in entries] == ["q0", "q1", "q2"]


def test_parse_log_file_dispatches_to_csv(tmp_path):
    path = _write(
        tmp_path / "traces.csv",
        "query,output\nhello,world\n",
    )
    entries = parse_log_file(path)
    assert len(entries) == 1
    assert entries[0].query == "hello"
    assert entries[0].output == "world"


def test_parse_log_file_explicit_csv_format(tmp_path):
    # File without .csv extension; force fmt='csv'
    path = _write(tmp_path / "traces.txt", "query\nhello\n")
    entries = parse_log_file(path, fmt="csv")
    assert len(entries) == 1
    assert entries[0].query == "hello"
