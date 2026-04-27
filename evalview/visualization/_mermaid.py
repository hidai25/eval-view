"""Mermaid sequence-diagram generation for the visual report.

Rendered client-side by Mermaid.js (loaded from CDN by the report HTML).
Labels are sanitized so they can't contain characters Mermaid treats as
syntax (->, --, etc.).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult


def _safe_mermaid(s: str) -> str:
    """Strip everything except safe alphanumeric + basic punctuation for Mermaid labels."""
    s = s.replace("\n", " ").replace("\r", "")
    s = re.sub(r'[^\w\s\.\-_/=:,]', '', s)
    s = s[:28].strip()
    return (s + '...') if len(s) == 28 else s or '...'


def _strip_markdown(text: str) -> str:
    """Remove common markdown symbols for clean display in HTML."""
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text, flags=re.DOTALL)  # bold/italic
    text = re.sub(r'`(.+?)`', r'\1', text, flags=re.DOTALL)               # inline code
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)            # headings
    return text


def _mermaid_from_steps(steps: List[Any], query: str = "", output: str = "") -> str:
    """Core Mermaid sequence diagram builder from a steps list."""
    if not steps:
        return "sequenceDiagram\n    Note over Agent: Direct response — no tools used"

    lines = ["sequenceDiagram"]
    lines.append("    participant User")
    lines.append("    participant Agent")

    seen_tools: Dict[str, str] = {}
    for step in steps:
        tool: str = str(getattr(step, "tool_name", None) or getattr(step, "step_name", None) or "unknown")
        if tool not in seen_tools:
            alias = f"T{len(seen_tools)}"
            seen_tools[tool] = alias
            short = (tool[:31] + "…") if len(tool) > 32 else tool
            lines.append(f"    participant {alias} as {short}")

    short_query = _safe_mermaid((query[:40] + "…") if len(query) > 40 else query) if query else "..."
    lines.append(f"    User->>Agent: {short_query}")

    current_turn = None

    for step in steps:
        step_turn = getattr(step, "turn_index", None)

        # Add a turn separator when the turn index changes
        if step_turn is not None and step_turn != current_turn:
            step_query = getattr(step, "turn_query", "") or ""
            safe_query = _safe_mermaid((step_query[:57] + "...") if len(step_query) > 60 else step_query)
            if safe_query:
                lines.append(f"    Note over User,Agent: Turn {step_turn} - {safe_query}")
            else:
                lines.append(f"    Note over User,Agent: Turn {step_turn}")
            current_turn = step_turn

        tool = str(getattr(step, "tool_name", None) or getattr(step, "step_name", None) or "unknown")
        alias = seen_tools.get(tool, tool)
        params = getattr(step, "parameters", {}) or {}
        param_str = ", ".join(f"{k}={str(v)[:20]}" for k, v in list(params.items())[:2])
        if len(params) > 2:
            param_str += "…"
        success = getattr(step, "success", True)
        arrow = "->>" if success else "-x"
        lines.append(f"    Agent{arrow}{alias}: {_safe_mermaid(param_str or tool)}")
        out = getattr(step, "output", None)
        out_str = str(out)[:30] if out is not None else "ok"
        lines.append(f"    {alias}-->Agent: {_safe_mermaid(out_str)}")

    short_out = _safe_mermaid((output[:40] + "…") if len(output) > 40 else output) if output else "..."
    lines.append(f"    Agent-->>User: {short_out}")

    return "\n".join(lines)


def _mermaid_trace(result: "EvaluationResult") -> str:
    """Convert an EvaluationResult into a Mermaid sequence diagram."""
    steps = []
    try:
        steps = result.trace.steps or []
    except AttributeError:
        pass
    query: str = str(getattr(result, "input_query", "") or "")
    output: str = str(getattr(result, "actual_output", "") or "")
    return _mermaid_from_steps(steps, query, output)
