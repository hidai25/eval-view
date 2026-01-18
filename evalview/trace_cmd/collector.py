"""Trace collector for writing span data to a file.

This module is used by the patcher to collect trace data from instrumented
SDK calls and write them to a JSONL file for later processing.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading

__all__ = ["TraceCollector", "get_collector"]

# Global collector instance (set via env var path)
_collector: Optional["TraceCollector"] = None
_lock = threading.Lock()


class TraceCollector:
    """Collects trace spans and writes them to a JSONL file.

    Thread-safe for use in multi-threaded applications.
    """

    def __init__(self, output_path: str):
        """Initialize the collector.

        Args:
            output_path: Path to write JSONL trace data
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._span_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0

        # Write trace start record
        self._write_record({
            "type": "trace_start",
            "trace_spec_version": "1.0",
            "source": "trace_cmd",
            "started_at": datetime.now().isoformat(),
            "pid": os.getpid(),
        })

    def record_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        cost: float,
        finish_reason: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record an LLM API call.

        Args:
            provider: SDK provider (openai, anthropic)
            model: Model identifier
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            duration_ms: Call duration in milliseconds
            cost: Estimated cost in USD
            finish_reason: Reason for completion
            error: Error message if call failed
        """
        self._span_count += 1
        self._total_tokens += input_tokens + output_tokens
        self._total_cost += cost

        self._write_record({
            "type": "span",
            "span_type": "llm",
            "span_id": f"span_{self._span_count:04d}",
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_ms": duration_ms,
            "cost_usd": cost,
            "finish_reason": finish_reason,
            "status": "error" if error else "success",
            "error_message": error,
            "timestamp": datetime.now().isoformat(),
        })

    def _write_record(self, record: Dict[str, Any]) -> None:
        """Write a record to the output file (thread-safe)."""
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()

    def close(self) -> Dict[str, Any]:
        """Close the collector and return summary stats.

        Returns:
            Summary dictionary with totals
        """
        total_time_ms = (time.time() - self._start_time) * 1000

        summary = {
            "type": "trace_end",
            "ended_at": datetime.now().isoformat(),
            "total_llm_calls": self._span_count,
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "total_time_ms": total_time_ms,
        }

        self._write_record(summary)
        self._file.close()

        return summary


def get_collector() -> Optional[TraceCollector]:
    """Get or create the global trace collector.

    Uses EVALVIEW_TRACE_OUTPUT env var to determine output path.
    Returns None if env var not set (tracing not enabled).
    """
    global _collector

    output_path = os.environ.get("EVALVIEW_TRACE_OUTPUT")
    if not output_path:
        return None

    with _lock:
        if _collector is None:
            _collector = TraceCollector(output_path)

    return _collector


def close_collector() -> Optional[Dict[str, Any]]:
    """Close the global collector and return summary.

    Returns:
        Summary stats or None if no collector active
    """
    global _collector

    with _lock:
        if _collector is not None:
            summary = _collector.close()
            _collector = None
            return summary

    return None
