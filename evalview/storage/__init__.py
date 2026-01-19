"""Local trace storage using SQLite.

Provides persistent storage for traces from both `evalview run` and
`evalview trace` commands, enabling historical queries and cost reports.

Usage:
    from evalview.storage import TraceDB

    db = TraceDB()
    traces = db.list_traces(last_hours=24)
    db.close()
"""

from evalview.storage.database import TraceDB

__all__ = ["TraceDB"]
