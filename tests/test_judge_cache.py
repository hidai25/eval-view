"""Tests for the LLM judge response cache."""

import tempfile
import time
from pathlib import Path

from evalview.core.judge_cache import JudgeCache


class TestJudgeCacheInMemory:
    """Test in-memory cache behavior."""

    def test_put_and_get(self):
        cache = JudgeCache()
        result = {"score": 85, "rationale": "Good response"}
        cache.put("hello world", "accuracy", result, test_case_id="t1")

        cached = cache.get("hello world", "accuracy", test_case_id="t1")
        assert cached == result

    def test_cache_miss(self):
        cache = JudgeCache()
        assert cache.get("no such output", "criteria") is None

    def test_different_criteria_different_key(self):
        cache = JudgeCache()
        r1 = {"score": 80, "rationale": "ok"}
        r2 = {"score": 90, "rationale": "great"}
        cache.put("same output", "criteria_a", r1)
        cache.put("same output", "criteria_b", r2)

        assert cache.get("same output", "criteria_a") == r1
        assert cache.get("same output", "criteria_b") == r2

    def test_different_test_case_id_scopes_entries(self):
        cache = JudgeCache()
        r1 = {"score": 70, "rationale": "a"}
        r2 = {"score": 95, "rationale": "b"}
        cache.put("output", "crit", r1, test_case_id="test1")
        cache.put("output", "crit", r2, test_case_id="test2")

        assert cache.get("output", "crit", "test1") == r1
        assert cache.get("output", "crit", "test2") == r2

    def test_stats_tracking(self):
        cache = JudgeCache()
        cache.put("x", "c", {"score": 1, "rationale": ""})
        cache.get("x", "c")  # hit
        cache.get("y", "c")  # miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_ttl_expiry(self):
        cache = JudgeCache(ttl_seconds=1)
        cache.put("x", "c", {"score": 1, "rationale": ""})
        assert cache.get("x", "c") is not None

        time.sleep(1.1)
        assert cache.get("x", "c") is None

    def test_disabled_cache(self):
        cache = JudgeCache(enabled=False)
        cache.put("x", "c", {"score": 1, "rationale": ""})
        assert cache.get("x", "c") is None

    def test_clear(self):
        cache = JudgeCache()
        cache.put("x", "c", {"score": 1, "rationale": ""})
        cache.clear()
        assert cache.get("x", "c") is None


class TestJudgeCacheSQLite:
    """Test SQLite persistence."""

    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "cache.db")
            result = {"score": 88, "rationale": "solid"}

            # Write
            cache1 = JudgeCache(persist_path=db_path)
            cache1.put("output", "criteria", result)
            cache1.close()

            # Read from new instance
            cache2 = JudgeCache(persist_path=db_path)
            cached = cache2.get("output", "criteria")
            assert cached == result
            cache2.close()
