"""Production log importer — converts existing logs to EvalView test cases.

Supports four formats (auto-detected):
  - JSONL       each line: {"input": "...", "output": "...", "tools": [...]}
  - OpenAI      each line: {"messages": [...], "choices": [...]}
  - EvalView    capture proxy format: {"request": {...}, "response": {...}}
  - CSV         header row with `query`, optional `output`, optional `tools`
                (tools as comma-separated within the cell, semicolon-separated,
                or pipe-separated)

Usage::
    from evalview.importers.log_importer import parse_log_file, entries_to_yaml
    entries = parse_log_file(Path("prod.jsonl"))
    paths   = entries_to_yaml(entries, Path("tests/imported/"))
"""
from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class LogEntry:
    """A single parsed log entry, format-agnostic."""
    query: str
    output: str = ""
    tool_calls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Format detection ───────────────────────────────────────────────────────────

def detect_format(path: Path) -> str:
    """Sniff the first non-empty line to identify the log format.

    Returns one of: "openai" | "evalview" | "jsonl" | "csv" | "unknown"
    """
    if path.suffix.lower() == ".csv":
        return "csv"

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                # Fall back to CSV detection for non-JSON files: a header that
                # starts with a recognised column name is enough to dispatch.
                lower = stripped.lower()
                if "," in stripped and lower.split(",")[0].strip() in {
                    "query", "input", "prompt", "question", "user_message",
                }:
                    return "csv"
                return "unknown"

            if "messages" in obj and isinstance(obj.get("messages"), list):
                return "openai"
            if "request" in obj and "response" in obj:
                return "evalview"
            if any(k in obj for k in ("input", "query", "prompt", "user_message", "question")):
                return "jsonl"
            # Fall through to JSONL for any other JSON object
            return "jsonl"

    return "unknown"


# ── Format parsers ─────────────────────────────────────────────────────────────

def _extract_tool_names(tool_data: Any) -> List[str]:
    """Normalise tool call data from any common representation."""
    if not tool_data:
        return []
    names: List[str] = []
    items = tool_data if isinstance(tool_data, list) else [tool_data]
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("tool")
                or item.get("tool_name")
                or (item.get("function") or {}).get("name")
            )
            if name:
                names.append(str(name))
    return names


def parse_jsonl(path: Path, max_entries: int = 200) -> List[LogEntry]:
    """Parse JSONL — each line is a JSON object with flexible field names."""
    entries: List[LogEntry] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            if len(entries) >= max_entries:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            query = (
                obj.get("input") or obj.get("query") or obj.get("prompt")
                or obj.get("user_message") or obj.get("question")
                or obj.get("user_input") or ""
            )
            output = (
                obj.get("output") or obj.get("response") or obj.get("answer")
                or obj.get("assistant_message") or obj.get("result") or ""
            )
            tools = _extract_tool_names(
                obj.get("tools") or obj.get("tool_calls")
                or obj.get("tool_use") or obj.get("actions") or []
            )

            if not query:
                continue

            entries.append(LogEntry(
                query=str(query),
                output=str(output),
                tool_calls=tools,
                metadata={"source_line": lineno},
            ))
    return entries


def parse_openai(path: Path, max_entries: int = 200) -> List[LogEntry]:
    """Parse OpenAI chat completion log format (one completion per line)."""
    entries: List[LogEntry] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            if len(entries) >= max_entries:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            messages: List[Dict[str, Any]] = obj.get("messages", [])

            # Last user message is the query
            query = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    query = content if isinstance(content, str) else str(content)
                    break
            if not query:
                continue

            # Output + tool calls from assistant choice or message
            output = ""
            tool_calls_raw: Any = []
            choices = obj.get("choices", [])
            if choices:
                assistant_msg = choices[0].get("message", {})
                output = assistant_msg.get("content") or ""
                tool_calls_raw = assistant_msg.get("tool_calls", [])
            else:
                for msg in messages:
                    if msg.get("role") == "assistant":
                        output = msg.get("content") or ""
                        tool_calls_raw = msg.get("tool_calls", [])

            entries.append(LogEntry(
                query=query,
                output=str(output),
                tool_calls=_extract_tool_names(tool_calls_raw),
            ))
    return entries


def parse_evalview_capture(path: Path, max_entries: int = 200) -> List[LogEntry]:
    """Parse EvalView capture proxy log format."""
    entries: List[LogEntry] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            if len(entries) >= max_entries:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            req = obj.get("request", {})
            resp = obj.get("response", {})
            query = req.get("query") or req.get("input") or req.get("message") or ""
            output = resp.get("output") or resp.get("response") or resp.get("content") or ""
            tools = _extract_tool_names(resp.get("tool_calls") or resp.get("tools") or [])

            if not query:
                continue

            entries.append(LogEntry(
                query=str(query),
                output=str(output),
                tool_calls=tools,
            ))
    return entries


_CSV_QUERY_KEYS = ("query", "input", "prompt", "question", "user_message", "user_input")
_CSV_OUTPUT_KEYS = ("output", "response", "answer", "assistant_message", "result")
_CSV_TOOL_KEYS = ("tools", "tool_calls", "tool_use", "actions")


def _split_csv_tools(value: str) -> List[str]:
    """Split a CSV tool cell into tool names. Accepts comma, semicolon, pipe."""
    if not value:
        return []
    raw = value.strip()
    if not raw:
        return []
    # JSON list embedded in the cell, e.g. ["weather_api", "geocode"]
    if raw.startswith("[") and raw.endswith("]"):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
    # Otherwise treat as a separator-delimited list
    parts = re.split(r"[;,|]", raw)
    return [p.strip() for p in parts if p.strip()]


def parse_csv(
    path: Path,
    max_entries: int = 200,
    *,
    warn: Optional[Callable[[str], None]] = None,
) -> List[LogEntry]:
    """Parse a CSV log file.

    The header row identifies columns. The first column matched against
    ``_CSV_QUERY_KEYS`` becomes the query (required). ``_CSV_OUTPUT_KEYS``
    and ``_CSV_TOOL_KEYS`` are optional. Tool cells may be JSON-list,
    comma-, semicolon-, or pipe-separated.

    Malformed rows (missing query, unparseable tools cell) are skipped and
    surfaced via ``warn(message)``; if ``warn`` is None, messages go to
    stderr so they appear during ``evalview generate``.
    """
    if warn is None:
        def warn(message: str) -> None:
            print(f"warn: {message}", file=sys.stderr)

    entries: List[LogEntry] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        try:
            reader = csv.DictReader(f)
        except csv.Error as exc:
            warn(f"{path}: failed to open as CSV ({exc}); skipping")
            return entries

        if reader.fieldnames is None:
            warn(f"{path}: CSV has no header row; skipping")
            return entries

        # Map CSV columns to our canonical fields. The first matching column
        # wins so users can keep secondary columns for their own metadata.
        normalized = {name: name.strip().lower() for name in reader.fieldnames if name}
        query_col = next(
            (name for name, low in normalized.items() if low in _CSV_QUERY_KEYS),
            None,
        )
        if query_col is None:
            warn(
                f"{path}: CSV header is missing a query column "
                f"(expected one of: {', '.join(_CSV_QUERY_KEYS)}); skipping"
            )
            return entries
        output_col = next(
            (name for name, low in normalized.items() if low in _CSV_OUTPUT_KEYS),
            None,
        )
        tool_col = next(
            (name for name, low in normalized.items() if low in _CSV_TOOL_KEYS),
            None,
        )

        for lineno, row in enumerate(reader, start=2):  # header is line 1
            if len(entries) >= max_entries:
                break
            query = (row.get(query_col) or "").strip() if query_col else ""
            if not query:
                warn(f"{path}:{lineno}: row has empty query; skipped")
                continue
            output = (row.get(output_col) or "").strip() if output_col else ""
            tools = _split_csv_tools(row.get(tool_col) or "") if tool_col else []
            entries.append(LogEntry(
                query=query,
                output=output,
                tool_calls=tools,
                metadata={"source_line": lineno},
            ))
    return entries


def parse_log_file(
    path: Path,
    fmt: str = "auto",
    max_entries: int = 200,
) -> List[LogEntry]:
    """Parse a log file, auto-detecting format when fmt='auto'."""
    if fmt == "auto":
        fmt = detect_format(path)

    if fmt == "openai":
        return parse_openai(path, max_entries)
    if fmt == "evalview":
        return parse_evalview_capture(path, max_entries)
    if fmt == "csv":
        return parse_csv(path, max_entries)
    return parse_jsonl(path, max_entries)


# ── YAML serialisation ─────────────────────────────────────────────────────────

def _slugify(text: str, max_len: int = 40) -> str:
    """Convert arbitrary text to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text[:max_len])
    slug = re.sub(r"\s+", "-", slug).strip("-").lower()
    return slug or "test"


def entries_to_yaml(
    entries: List[LogEntry],
    output_dir: Path,
    name_prefix: str = "imported",
) -> List[Path]:
    """Write each LogEntry as an EvalView test case YAML file.

    Returns the list of paths written.
    """
    import yaml  # type: ignore[import-untyped]

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for i, entry in enumerate(entries, 1):
        slug = _slugify(entry.query)
        name = f"{name_prefix}-{i:03d}-{slug}"[:64]

        test: Dict[str, Any] = {
            "name": name,
            "description": f"Imported from log — {entry.query[:80]}",
            "input": {"query": entry.query},
            "expected": {},
            "thresholds": {"min_score": 70},
        }

        if entry.tool_calls:
            test["expected"]["tools"] = entry.tool_calls

        # Use factual numbers from the output as lightweight contains checks.
        # Avoids brittle exact-match assertions while still giving the evaluator
        # a hint about what a correct answer looks like.
        if entry.output and len(entry.output) > 20:
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", entry.output)[:2]
            if numbers:
                test["expected"]["output"] = {"contains": numbers}

        if not test["expected"]:
            test["expected"] = {}

        out_path = output_dir / f"{name}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(test, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        written.append(out_path)

    return written
