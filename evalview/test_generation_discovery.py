"""Tool-schema discovery and small free-function helpers for test generation.

These were tucked at the bottom of test_generation.py and don't depend on
AgentTestGenerator state — extracting them gives the main module a smaller
surface and makes them straightforward to unit test in isolation.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionMetrics,
    ExecutionTrace,
    StepMetrics,
    StepTrace,
)
from evalview.importers.log_importer import LogEntry


def _normalize_name(value: str) -> str:
    """Strip non-alphanumerics and lowercase — for tool-name comparisons."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _utc_now_iso() -> str:
    """ISO-8601 timestamp in UTC for report metadata."""
    return datetime.now(timezone.utc).isoformat()


def _trace_from_log_entry(entry: LogEntry) -> Any:
    """Build a minimal trace-like object from a log entry for synthesis."""
    now = datetime.now()
    steps = [
        StepTrace(
            step_id=f"step-{index + 1}",
            step_name=f"Imported {tool_name}",
            tool_name=tool_name,
            parameters={},
            output="",
            success=True,
            metrics=StepMetrics(latency=0.0, cost=0.0),
        )
        for index, tool_name in enumerate(entry.tool_calls)
    ]
    return ExecutionTrace(
        session_id=f"imported-{abs(hash(entry.query)) % 100000}",
        start_time=now,
        end_time=now,
        steps=steps,
        final_output=entry.output or "",
        metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
    )


async def discover_tool_schemas(
    adapter: Optional[AgentAdapter],
    adapter_type: str,
    endpoint: str,
) -> List[Dict[str, Any]]:
    """Discover tool metadata for probe planning when the adapter supports it."""
    try:
        if adapter is not None and hasattr(adapter, "discover_tools"):
            tools = await adapter.discover_tools()
            return _normalize_discovered_tools(tools)
        if adapter_type == "http" and endpoint:
            return await _discover_http_tools(endpoint)
    except Exception:
        return []
    return []


def _normalize_discovered_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for tool in tools:
        normalized.append(
            {
                "name": tool.get("name", ""),
                "description": tool.get("description", "") or tool.get("summary", ""),
                "inputSchema": tool.get("inputSchema") or tool.get("parameters") or {},
            }
        )
    return [tool for tool in normalized if tool["name"]]


async def _discover_http_tools(endpoint: str) -> List[Dict[str, Any]]:
    candidates = []
    parsed = urlsplit(endpoint)
    base = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    for path in ("/openapi.json", "/swagger.json", "/docs/openapi.json"):
        candidates.append(f"{base}{path}")

    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in candidates:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                tools = _extract_openapi_tools(data)
                if tools:
                    return tools
            except Exception:
                continue
    return []


def _extract_openapi_tools(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths = schema.get("paths", {})
    discovered: List[Dict[str, Any]] = []
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue
            name = operation.get("operationId") or f"{method.upper()} {path}"
            description = operation.get("summary") or operation.get("description") or ""
            properties: Dict[str, Any] = {}
            required: List[str] = []

            for parameter in operation.get("parameters", []):
                if not isinstance(parameter, dict):
                    continue
                param_name = parameter.get("name")
                if not param_name:
                    continue
                properties[param_name] = parameter.get("schema", {"type": "string"})
                if parameter.get("required"):
                    required.append(param_name)

            request_schema = (
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            if isinstance(request_schema, dict):
                body_props = request_schema.get("properties", {})
                properties.update(body_props)
                required.extend(request_schema.get("required", []))

            discovered.append(
                {
                    "name": name,
                    "description": description,
                    "inputSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": sorted(set(required)),
                    },
                }
            )
    return discovered
