"""Database operations for regression tracking."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class TrackingDatabase:
    """SQLite database for tracking test results over time."""

    def __init__(self, db_path: Path = Path(".evalview/tracking.db")):
        """
        Initialize tracking database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Test results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    score REAL NOT NULL,
                    passed INTEGER NOT NULL,
                    cost REAL,
                    latency REAL,
                    git_commit TEXT,
                    git_branch TEXT,
                    tool_accuracy REAL,
                    output_quality REAL,
                    sequence_correct INTEGER,
                    hallucination_detected INTEGER,
                    safety_passed INTEGER,
                    metadata TEXT,
                    UNIQUE(test_name, timestamp)
                )
            """
            )

            # Baselines table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS baselines (
                    test_name TEXT PRIMARY KEY,
                    score REAL NOT NULL,
                    cost REAL,
                    latency REAL,
                    tool_accuracy REAL,
                    output_quality REAL,
                    created_at DATETIME NOT NULL,
                    git_commit TEXT,
                    git_branch TEXT,
                    metadata TEXT
                )
            """
            )

            # Daily trends table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_trends (
                    date DATE PRIMARY KEY,
                    avg_score REAL NOT NULL,
                    avg_cost REAL,
                    avg_latency REAL,
                    total_tests INTEGER NOT NULL,
                    passed_tests INTEGER NOT NULL,
                    failed_tests INTEGER NOT NULL,
                    metadata TEXT
                )
            """
            )

            # Create indexes
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_test_results_name
                ON test_results(test_name)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_test_results_timestamp
                ON test_results(timestamp)
            """
            )

    def store_result(
        self,
        test_name: str,
        score: float,
        passed: bool,
        cost: Optional[float] = None,
        latency: Optional[float] = None,
        tool_accuracy: Optional[float] = None,
        output_quality: Optional[float] = None,
        sequence_correct: Optional[bool] = None,
        hallucination_detected: Optional[bool] = None,
        safety_passed: Optional[bool] = None,
        git_commit: Optional[str] = None,
        git_branch: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Store a test result.

        Args:
            test_name: Name of the test
            score: Overall score
            passed: Whether test passed
            cost: Cost in dollars
            latency: Latency in milliseconds
            tool_accuracy: Tool accuracy score
            output_quality: Output quality score
            sequence_correct: Whether sequence was correct
            hallucination_detected: Whether hallucination was detected
            safety_passed: Whether safety check passed
            git_commit: Git commit hash
            git_branch: Git branch name
            metadata: Additional metadata

        Returns:
            ID of inserted row
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO test_results (
                    test_name, timestamp, score, passed, cost, latency,
                    tool_accuracy, output_quality, sequence_correct,
                    hallucination_detected, safety_passed,
                    git_commit, git_branch, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    test_name,
                    datetime.now(),
                    score,
                    1 if passed else 0,
                    cost,
                    latency,
                    tool_accuracy,
                    output_quality,
                    1 if sequence_correct else 0 if sequence_correct is not None else None,
                    (
                        1
                        if hallucination_detected
                        else 0 if hallucination_detected is not None else None
                    ),
                    1 if safety_passed else 0 if safety_passed is not None else None,
                    git_commit,
                    git_branch,
                    json.dumps(metadata) if metadata else None,
                ),
            )

            return cursor.lastrowid

    def get_baseline(self, test_name: str) -> Optional[Dict[str, Any]]:
        """
        Get baseline for a test.

        Args:
            test_name: Name of the test

        Returns:
            Baseline data or None if not set
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM baselines WHERE test_name = ?", (test_name,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def set_baseline(
        self,
        test_name: str,
        score: float,
        cost: Optional[float] = None,
        latency: Optional[float] = None,
        tool_accuracy: Optional[float] = None,
        output_quality: Optional[float] = None,
        git_commit: Optional[str] = None,
        git_branch: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Set or update baseline for a test.

        Args:
            test_name: Name of the test
            score: Baseline score
            cost: Baseline cost
            latency: Baseline latency
            tool_accuracy: Baseline tool accuracy
            output_quality: Baseline output quality
            git_commit: Git commit hash
            git_branch: Git branch name
            metadata: Additional metadata
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO baselines (
                    test_name, score, cost, latency, tool_accuracy, output_quality,
                    created_at, git_commit, git_branch, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    test_name,
                    score,
                    cost,
                    latency,
                    tool_accuracy,
                    output_quality,
                    datetime.now(),
                    git_commit,
                    git_branch,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def get_test_history(self, test_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get historical results for a test.

        Args:
            test_name: Name of the test
            days: Number of days to look back

        Returns:
            List of historical results
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM test_results
                WHERE test_name = ?
                AND timestamp >= datetime('now', '-' || ? || ' days')
                ORDER BY timestamp DESC
                """,
                (test_name, days),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_recent_results(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get all recent test results.

        Args:
            days: Number of days to look back

        Returns:
            List of recent results
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM test_results
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                ORDER BY timestamp DESC
                """,
                (days,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def update_daily_trends(self):
        """Update daily trends from test results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO daily_trends (
                    date, avg_score, avg_cost, avg_latency,
                    total_tests, passed_tests, failed_tests
                )
                SELECT
                    DATE(timestamp) as date,
                    AVG(score) as avg_score,
                    AVG(cost) as avg_cost,
                    AVG(latency) as avg_latency,
                    COUNT(*) as total_tests,
                    SUM(passed) as passed_tests,
                    COUNT(*) - SUM(passed) as failed_tests
                FROM test_results
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY DATE(timestamp)
            """
            )

    def get_daily_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily performance trends.

        Args:
            days: Number of days to look back

        Returns:
            List of daily trend data
        """
        self.update_daily_trends()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM daily_trends
                WHERE date >= date('now', '-' || ? || ' days')
                ORDER BY date ASC
                """,
                (days,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def clear_baselines(self):
        """Clear all baselines."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM baselines")

    def clear_history(self, days: Optional[int] = None):
        """
        Clear test history.

        Args:
            days: If specified, only clear results older than this many days
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if days:
                cursor.execute(
                    """
                    DELETE FROM test_results
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                    """,
                    (days,),
                )
            else:
                cursor.execute("DELETE FROM test_results")
