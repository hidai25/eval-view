"""Cache layer for LLM judge responses.

Avoids redundant API calls when the same agent output is evaluated
multiple times during statistical test runs (e.g., 10 repeated runs).

Cache key is a SHA-256 hash of (output_text, evaluation_criteria) so
identical evaluations are served from cache on subsequent runs.
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JudgeCache:
    """Hash-based cache for LLM-as-judge evaluation results.

    Supports in-memory caching for single sessions and optional SQLite
    persistence across sessions.

    Args:
        persist_path: Path to SQLite database file. If None, cache is
            in-memory only and lost when the process exits.
        ttl_seconds: Time-to-live for cache entries in seconds.
            Defaults to 86400 (24 hours). Set to 0 to disable expiry.
        enabled: Whether caching is active. When False, all lookups
            return None and stores are no-ops.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        ttl_seconds: int = 86400,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._db: Optional[sqlite3.Connection] = None
        self._hits = 0
        self._misses = 0

        if persist_path and enabled:
            db_path = Path(persist_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(db_path))
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self._db.commit()

    @staticmethod
    def _make_key(output_text: str, criteria: str, test_case_id: str = "") -> str:
        """Create a deterministic cache key from evaluation inputs.

        Args:
            output_text: The agent output being evaluated.
            criteria: Evaluation criteria (e.g., expected contains list).
            test_case_id: Optional test case identifier to scope cache
                entries so different tests don't share results.

        Returns:
            A hex SHA-256 digest string.
        """
        raw = json.dumps(
            {"output": output_text, "criteria": criteria, "test_case": test_case_id},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, output_text: str, criteria: str, test_case_id: str = "") -> Optional[Dict[str, Any]]:
        """Look up a cached judge result.

        Returns:
            The cached result dict, or None on cache miss.
        """
        if not self.enabled:
            return None

        key = self._make_key(output_text, criteria, test_case_id)
        now = time.time()

        # Check in-memory first
        if key in self._memory:
            entry = self._memory[key]
            if self.ttl_seconds == 0 or (now - entry["created_at"]) < self.ttl_seconds:
                self._hits += 1
                return entry["result"]
            else:
                del self._memory[key]

        # Check SQLite
        if self._db is not None:
            row = self._db.execute(
                "SELECT result_json, created_at FROM judge_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                result_json, created_at = row
                if self.ttl_seconds == 0 or (now - created_at) < self.ttl_seconds:
                    result = json.loads(result_json)
                    self._memory[key] = {"result": result, "created_at": created_at}
                    self._hits += 1
                    return result
                else:
                    self._db.execute(
                        "DELETE FROM judge_cache WHERE cache_key = ?", (key,)
                    )
                    self._db.commit()

        self._misses += 1
        return None

    def put(self, output_text: str, criteria: str, result: Dict[str, Any], test_case_id: str = "") -> None:
        """Store a judge result in the cache."""
        if not self.enabled:
            return

        key = self._make_key(output_text, criteria, test_case_id)
        now = time.time()

        self._memory[key] = {"result": result, "created_at": now}

        if self._db is not None:
            self._db.execute(
                """
                INSERT OR REPLACE INTO judge_cache (cache_key, result_json, created_at)
                VALUES (?, ?, ?)
                """,
                (key, json.dumps(result), now),
            )
            self._db.commit()

    @property
    def stats(self) -> Dict[str, int]:
        """Return cache hit/miss statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": self._hits + self._misses,
            "entries": len(self._memory),
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory.clear()
        if self._db is not None:
            self._db.execute("DELETE FROM judge_cache")
            self._db.commit()

    def close(self) -> None:
        """Close the SQLite connection if open."""
        if self._db is not None:
            self._db.close()
            self._db = None
