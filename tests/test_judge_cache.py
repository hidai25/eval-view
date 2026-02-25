"""Tests for LLM judge response cache."""

import os
import tempfile
from unittest.mock import patch

from evalview.core.judge_cache import JudgeCache


class TestJudgeCacheKey:
    """Tests for cache key generation."""

    def test_same_inputs_produce_same_key(self):
        k1 = JudgeCache.make_key("test-1", "hello", "world", ["a"], ["b"])
        k2 = JudgeCache.make_key("test-1", "hello", "world", ["a"], ["b"])
        assert k1 == k2

    def test_different_test_name_produces_different_key(self):
        k1 = JudgeCache.make_key("test-1", "q", "out")
        k2 = JudgeCache.make_key("test-2", "q", "out")
        assert k1 != k2

    def test_different_query_produces_different_key(self):
        k1 = JudgeCache.make_key("t", "query-a", "out")
        k2 = JudgeCache.make_key("t", "query-b", "out")
        assert k1 != k2

    def test_different_output_produces_different_key(self):
        k1 = JudgeCache.make_key("t", "q", "output-a")
        k2 = JudgeCache.make_key("t", "q", "output-b")
        assert k1 != k2

    def test_different_contains_produces_different_key(self):
        k1 = JudgeCache.make_key("t", "q", "o", contains=["x"])
        k2 = JudgeCache.make_key("t", "q", "o", contains=["y"])
        assert k1 != k2

    def test_different_not_contains_produces_different_key(self):
        k1 = JudgeCache.make_key("t", "q", "o", not_contains=["x"])
        k2 = JudgeCache.make_key("t", "q", "o", not_contains=["y"])
        assert k1 != k2

    def test_contains_order_does_not_matter(self):
        """Contains lists are sorted before hashing, so order is irrelevant."""
        k1 = JudgeCache.make_key("t", "q", "o", contains=["b", "a"])
        k2 = JudgeCache.make_key("t", "q", "o", contains=["a", "b"])
        assert k1 == k2

    def test_none_criteria_handled(self):
        k = JudgeCache.make_key("t", "q", "o", contains=None, not_contains=None)
        assert isinstance(k, str) and len(k) == 64  # SHA-256 hex


class TestJudgeCacheMemory:
    """Tests for in-memory cache operations."""

    def test_put_and_get(self):
        cache = JudgeCache()
        value = {"score": 85, "rationale": "good"}
        cache.put("k1", value)
        assert cache.get("k1") == value

    def test_miss_returns_none(self):
        cache = JudgeCache()
        assert cache.get("nonexistent") is None

    def test_stats_tracking(self):
        cache = JudgeCache()
        cache.put("k1", {"score": 90, "rationale": "great"})

        cache.get("k1")  # hit
        cache.get("k1")  # hit
        cache.get("missing")  # miss

        s = cache.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["total"] == 3
        assert s["entries"] == 1

    def test_disabled_cache_always_misses(self):
        cache = JudgeCache(enabled=False)
        cache.put("k1", {"score": 50, "rationale": "meh"})
        assert cache.get("k1") is None

    @patch("evalview.core.judge_cache.time")
    def test_ttl_expiry(self, mock_time):
        """Entries older than TTL are evicted — uses time mocking, not sleep."""
        mock_time.time.return_value = 1000.0
        cache = JudgeCache(ttl=60)

        cache.put("k1", {"score": 80, "rationale": "ok"})

        # Still valid at t=1059
        mock_time.time.return_value = 1059.0
        assert cache.get("k1") is not None

        # Expired at t=1061
        mock_time.time.return_value = 1061.0
        assert cache.get("k1") is None

    @patch("evalview.core.judge_cache.time")
    def test_ttl_zero_never_expires(self, mock_time):
        mock_time.time.return_value = 1000.0
        cache = JudgeCache(ttl=0)
        cache.put("k1", {"score": 90, "rationale": "great"})

        mock_time.time.return_value = 999_999.0
        assert cache.get("k1") is not None


class TestJudgeCacheSQLite:
    """Tests for SQLite persistence."""

    def test_persist_and_recover(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Write with one cache instance
            c1 = JudgeCache(persist_path=db_path)
            c1.put("k1", {"score": 77, "rationale": "decent"})

            # Read from a fresh instance pointing at the same db
            c2 = JudgeCache(persist_path=db_path)
            result = c2.get("k1")
            assert result is not None
            assert result["score"] == 77
            assert result["rationale"] == "decent"
        finally:
            os.unlink(db_path)

    @patch("evalview.core.judge_cache.time")
    def test_expired_entries_deleted_from_db(self, mock_time):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            mock_time.time.return_value = 1000.0
            c1 = JudgeCache(persist_path=db_path, ttl=60)
            c1.put("k1", {"score": 50, "rationale": "stale"})

            mock_time.time.return_value = 1100.0
            c2 = JudgeCache(persist_path=db_path, ttl=60)
            assert c2.get("k1") is None
        finally:
            os.unlink(db_path)


class TestJudgeCacheIntegration:
    """Tests for cache integration with OutputEvaluator."""

    @patch("evalview.evaluators.output_evaluator.LLMClient")
    def test_output_evaluator_accepts_cache_param(self, _mock_llm):
        """OutputEvaluator constructor should accept the cache kwarg."""
        from evalview.evaluators.output_evaluator import OutputEvaluator

        cache = JudgeCache()
        evaluator = OutputEvaluator(cache=cache)
        assert evaluator.cache is cache

    def test_evaluator_accepts_judge_cache_param(self):
        """Evaluator constructor should accept the judge_cache kwarg."""
        from evalview.evaluators.evaluator import Evaluator

        cache = JudgeCache()
        evaluator = Evaluator(judge_cache=cache, skip_llm_judge=True)
        assert evaluator.judge_cache is cache
