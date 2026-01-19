"""SQLite database for trace storage.

Schema:
    traces: id, run_id, created_at, total_cost, total_tokens, total_latency_ms,
            source, script_name, status, summary_json
    spans: id, trace_id, span_id, span_type, provider, model, input_tokens,
           output_tokens, duration_ms, cost_usd, status, error_message, timestamp
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["TraceDB"]

# Default database location
DEFAULT_DB_PATH = ".evalview/traces.db"

# Schema version for migrations
SCHEMA_VERSION = 1

CREATE_TABLES_SQL = """
-- Traces table
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    total_cost REAL DEFAULT 0.0,
    total_tokens INTEGER DEFAULT 0,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_latency_ms REAL DEFAULT 0.0,
    total_calls INTEGER DEFAULT 0,
    source TEXT NOT NULL,
    script_name TEXT,
    status TEXT DEFAULT 'completed',
    summary_json TEXT
);

-- Spans table
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id INTEGER NOT NULL,
    span_id TEXT NOT NULL,
    span_type TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    duration_ms REAL DEFAULT 0.0,
    cost_usd REAL DEFAULT 0.0,
    finish_reason TEXT,
    status TEXT DEFAULT 'success',
    error_message TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (trace_id) REFERENCES traces(id) ON DELETE CASCADE
);

-- Schema version table
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at);
CREATE INDEX IF NOT EXISTS idx_traces_source ON traces(source);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_model ON spans(model);
"""


class TraceDB:
    """SQLite database for storing and querying traces."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database. Defaults to .evalview/traces.db
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self._conn.cursor()
        cursor.executescript(CREATE_TABLES_SQL)

        # Check/set schema version
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,)
            )

        self._conn.commit()

    def save_trace(
        self,
        source: str,
        script_name: Optional[str] = None,
        spans: Optional[List[Dict[str, Any]]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a trace with its spans.

        Args:
            source: Source of trace ('eval' or 'trace_cmd')
            script_name: Name of script being traced
            spans: List of span dictionaries
            summary: Summary statistics

        Returns:
            The run_id of the saved trace
        """
        run_id = uuid.uuid4().hex[:8]  # 8 hex chars = 4 billion combinations
        now = datetime.now().isoformat()
        spans = spans or []
        summary = summary or {}

        # Calculate totals from spans (single pass)
        total_cost = summary.get("total_cost_usd", 0.0)
        total_tokens = summary.get("total_tokens", 0)
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = summary.get("total_time_ms", 0.0)
        total_calls = 0

        for span in spans:
            if span.get("span_type") == "llm":
                total_calls += 1
                total_input_tokens += span.get("input_tokens", 0)
                total_output_tokens += span.get("output_tokens", 0)
                if total_cost == 0:
                    total_cost += span.get("cost_usd", 0.0)
                if total_latency_ms == 0:
                    total_latency_ms += span.get("duration_ms", 0.0)

        if total_tokens == 0:
            total_tokens = total_input_tokens + total_output_tokens

        cursor = self._conn.cursor()

        # Insert trace
        cursor.execute(
            """
            INSERT INTO traces (
                run_id, created_at, total_cost, total_tokens,
                total_input_tokens, total_output_tokens, total_latency_ms,
                total_calls, source, script_name, status, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, now, total_cost, total_tokens,
                total_input_tokens, total_output_tokens, total_latency_ms,
                total_calls, source, script_name, "completed",
                json.dumps(summary) if summary else None
            )
        )
        trace_id = cursor.lastrowid

        # Insert spans
        for span in spans:
            if span.get("type") == "span":
                cursor.execute(
                    """
                    INSERT INTO spans (
                        trace_id, span_id, span_type, provider, model,
                        input_tokens, output_tokens, duration_ms, cost_usd,
                        finish_reason, status, error_message, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace_id,
                        span.get("span_id", ""),
                        span.get("span_type", "unknown"),
                        span.get("provider"),
                        span.get("model"),
                        span.get("input_tokens", 0),
                        span.get("output_tokens", 0),
                        span.get("duration_ms", 0.0),
                        span.get("cost_usd", 0.0),
                        span.get("finish_reason"),
                        span.get("status", "success"),
                        span.get("error_message"),
                        span.get("timestamp", now),
                    )
                )

        self._conn.commit()
        return run_id

    def list_traces(
        self,
        last_hours: Optional[int] = None,
        last_days: Optional[int] = None,
        source: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List traces with optional filters.

        Args:
            last_hours: Filter to traces from last N hours
            last_days: Filter to traces from last N days
            source: Filter by source ('eval' or 'trace_cmd')
            limit: Maximum number of traces to return

        Returns:
            List of trace dictionaries
        """
        query = "SELECT * FROM traces WHERE 1=1"
        params: List[Any] = []

        if last_hours is not None:
            cutoff = (datetime.now() - timedelta(hours=last_hours)).isoformat()
            query += " AND created_at >= ?"
            params.append(cutoff)
        elif last_days is not None:
            cutoff = (datetime.now() - timedelta(days=last_days)).isoformat()
            query += " AND created_at >= ?"
            params.append(cutoff)

        if source is not None:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.cursor()
        cursor.execute(query, params)

        return [dict(row) for row in cursor.fetchall()]

    def get_trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace by run_id.

        Args:
            run_id: The trace run_id

        Returns:
            Trace dictionary or None if not found
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM traces WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_trace_spans(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all spans for a trace.

        Args:
            run_id: The trace run_id

        Returns:
            List of span dictionaries
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT s.* FROM spans s
            JOIN traces t ON s.trace_id = t.id
            WHERE t.run_id = ?
            ORDER BY s.id
            """,
            (run_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_cost_report(
        self,
        last_days: int = 7,
    ) -> Dict[str, Any]:
        """Get cost report for a time period.

        Args:
            last_days: Number of days to include

        Returns:
            Report dictionary with totals and breakdowns
        """
        cutoff = (datetime.now() - timedelta(days=last_days)).isoformat()
        cursor = self._conn.cursor()

        # Total stats
        cursor.execute(
            """
            SELECT
                COUNT(*) as trace_count,
                SUM(total_calls) as total_calls,
                SUM(total_cost) as total_cost,
                SUM(total_tokens) as total_tokens,
                SUM(total_input_tokens) as total_input_tokens,
                SUM(total_output_tokens) as total_output_tokens
            FROM traces
            WHERE created_at >= ?
            """,
            (cutoff,)
        )
        totals = dict(cursor.fetchone())

        # By model
        cursor.execute(
            """
            SELECT
                model,
                COUNT(*) as call_count,
                SUM(cost_usd) as total_cost,
                SUM(input_tokens + output_tokens) as total_tokens
            FROM spans s
            JOIN traces t ON s.trace_id = t.id
            WHERE t.created_at >= ? AND s.span_type = 'llm'
            GROUP BY model
            ORDER BY total_cost DESC
            """,
            (cutoff,)
        )
        by_model = [dict(row) for row in cursor.fetchall()]

        # By day
        cursor.execute(
            """
            SELECT
                DATE(created_at) as day,
                SUM(total_cost) as total_cost,
                SUM(total_calls) as total_calls
            FROM traces
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY day
            """,
            (cutoff,)
        )
        by_day = [dict(row) for row in cursor.fetchall()]

        return {
            "period_days": last_days,
            "totals": totals,
            "by_model": by_model,
            "by_day": by_day,
        }

    def delete_trace(self, run_id: str) -> bool:
        """Delete a trace and its spans.

        Args:
            run_id: The trace run_id

        Returns:
            True if deleted, False if not found
        """
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM traces WHERE run_id = ?", (run_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()

    def __enter__(self) -> "TraceDB":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
