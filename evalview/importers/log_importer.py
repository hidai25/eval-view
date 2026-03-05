"""Production log importer — converts existing logs to EvalView test cases.

Supports three formats (auto-detected):
  - JSONL       each line: {"input": "...", "output": "...", "tools": [...]}
  - OpenAI      each line: {"messages": [...], "choices": [...]}
  - EvalView    capture proxy format: {"request": {...}, "response": {...}}

Usage::
    from evalview.importers.log_importer import parse_log_file, entries_to_yaml
    entries = parse_log_file(Path("prod.jsonl"))
    paths   = entries_to_yaml(entries, Path("tests/imported/"))
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    Returns one of: "openai" | "evalview" | "jsonl" | "unknown"
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
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
