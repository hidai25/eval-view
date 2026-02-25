"""Cache for LLM judge responses to avoid redundant API calls.

When running tests in statistical mode (--runs N), the same agent output
may be evaluated multiple times by the LLM judge. This module caches
judge responses keyed on the full evaluation context — test name, query,
output text, and all criteria — so identical evaluations are served from
cache instead of making duplicate API calls.
"""

import hashlib
import logging
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class JudgeCache:
    """In-memory cache with optional SQLite persistence for LLM judge results.

    Args:
        enabled: Whether caching is active (default True).
        persist_path: Path to a SQLite file for cross-session persistence.
                      When None, cache is in-memory only.
        ttl: Time-to-live in seconds. 0 means entries never expire.
             Default is 86400 (24 hours).
    """

    def __init__(
        self,
        enabled: bool = True,
        persist_path: Optional[str] = None,
        ttl: int = 86400,
    ):
        self.enabled = enabled
        self.persist_path = persist_path
        self.ttl = ttl

        # In-memory cache: key -> (timestamp, value)
        self._memory: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        # Stats
        self.hits = 0
        self.misses = 0

        # Initialise SQLite if persistence is requested
        if self.persist_path:
            self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Look up a cached judge result.

        Returns the cached dict on hit, or None on miss / expiry.
        """
        if not self.enabled:
            return None

        # Try memory first
        entry = self._memory.get(key)
        if entry is not None:
            ts, value = entry
            if self._is_valid(ts):
                self.hits += 1
                return value
            else:
                del self._memory[key]

        # Fall back to SQLite
        if self.persist_path:
            row = self._db_get(key)
            if row is not None:
                ts, value = row
                if self._is_valid(ts):
                    # Promote to memory
                    self._memory[key] = (ts, value)
                    self.hits += 1
                    return value
                else:
                    self._db_delete(key)

        self.misses += 1
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        """Store a judge result in the cache."""
        if not self.enabled:
            return

        ts = time.time()
        self._memory[key] = (ts, value)

        if self.persist_path:
            self._db_put(key, ts, value)

    def stats(self) -> Dict[str, Union[int, float]]:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(self.hits / total, 2) if total else 0.0,
            "entries": len(self._memory),
        }

    # ------------------------------------------------------------------
    # Cache key builder
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(
        test_name: str,
        query: str,
        output_text: str,
        contains: Optional[list] = None,
        not_contains: Optional[list] = None,
    ) -> str:
        """Build a deterministic cache key from the full evaluation context.

        Includes test name, query, output, and all criteria fields to
        prevent collisions between different test cases with the same output.
        """
        parts = [
            f"name:{test_name}",
            f"query:{query}",
            f"output:{output_text}",
            f"contains:{','.join(sorted(contains or []))}",
            f"not_contains:{','.join(sorted(not_contains or []))}",
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # TTL helper
    # ------------------------------------------------------------------

    def _is_valid(self, timestamp: float) -> bool:
        if self.ttl == 0:
            return True
        return (time.time() - timestamp) < self.ttl

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS judge_cache (
                    key TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    score REAL NOT NULL,
                    rationale TEXT NOT NULL
                )"""
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.persist_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _db_get(self, key: str) -> Optional[tuple]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT timestamp, score, rationale FROM judge_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return (row[0], {"score": row[1], "rationale": row[2]})

    def _db_put(self, key: str, ts: float, value: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO judge_cache (key, timestamp, score, rationale) VALUES (?, ?, ?, ?)",
                (key, ts, value.get("score", 0), value.get("rationale", "")),
            )

    def _db_delete(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM judge_cache WHERE key = ?", (key,))
