"""Tests for evalview/core/semantic_diff.py."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch


class TestSemanticDiffAvailability:
    """Tests for is_available() and error conditions."""

    def test_is_available_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from evalview.core.semantic_diff import SemanticDiff
        assert SemanticDiff.is_available() is False

    def test_is_available_with_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from evalview.core.semantic_diff import SemanticDiff
        assert SemanticDiff.is_available() is True

    def test_init_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from evalview.core.semantic_diff import SemanticDiff
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            SemanticDiff()

    def test_init_with_explicit_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from evalview.core.semantic_diff import SemanticDiff
        sd = SemanticDiff(api_key="sk-explicit")
        assert sd.api_key == "sk-explicit"

    def test_cost_notice_is_string(self, monkeypatch):
        from evalview.core.semantic_diff import SemanticDiff
        notice = SemanticDiff.cost_notice()
        assert isinstance(notice, str)
        assert "embedding" in notice.lower()


class TestSemanticDiffSimilarity:
    """Tests for the similarity() computation."""

    @pytest.fixture
    def semantic_diff(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from evalview.core.semantic_diff import SemanticDiff
        return SemanticDiff()

    @pytest.mark.asyncio
    async def test_identical_texts_score_one(self, semantic_diff):
        """Identical texts should have cosine similarity of 1.0."""
        # Mock embeddings where both vectors are the same
        fake_emb = [1.0, 0.0, 0.0]
        with patch.object(semantic_diff, "_embed", new=AsyncMock(return_value=[fake_emb, fake_emb])):
            score = await semantic_diff.similarity("hello", "hello")
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_orthogonal_texts_score_zero(self, semantic_diff):
        """Orthogonal embedding vectors should produce similarity of 0.0."""
        emb_a = [1.0, 0.0]
        emb_b = [0.0, 1.0]
        with patch.object(semantic_diff, "_embed", new=AsyncMock(return_value=[emb_a, emb_b])):
            score = await semantic_diff.similarity("cats", "quantum physics")
        assert score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_similar_texts_score_high(self, semantic_diff):
        """Semantically similar texts should score > 0.8."""
        # Vectors at 30° apart → cos(30°) ≈ 0.866
        import math
        emb_a = [1.0, 0.0]
        emb_b = [math.cos(math.radians(30)), math.sin(math.radians(30))]
        with patch.object(semantic_diff, "_embed", new=AsyncMock(return_value=[emb_a, emb_b])):
            score = await semantic_diff.similarity("The cat sat", "A cat was sitting")
        assert score > 0.8

    @pytest.mark.asyncio
    async def test_zero_vector_returns_zero(self, semantic_diff):
        """Zero-norm embedding should return 0.0 without division error."""
        zero = [0.0, 0.0, 0.0]
        non_zero = [1.0, 0.0, 0.0]
        with patch.object(semantic_diff, "_embed", new=AsyncMock(return_value=[zero, non_zero])):
            score = await semantic_diff.similarity("", "something")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_embed_called_with_truncated_texts(self, semantic_diff):
        """Texts longer than 8192 chars should be truncated before API call."""
        long_text = "x" * 10_000
        fake_emb = [1.0, 0.0]
        mock_embed = AsyncMock(return_value=[fake_emb, fake_emb])
        with patch.object(semantic_diff, "_embed", new=mock_embed):
            await semantic_diff.similarity(long_text, long_text)
        call_args = mock_embed.call_args[0][0]  # first positional arg (list of texts)
        assert len(call_args[0]) == 8192
        assert len(call_args[1]) == 8192


class TestSemanticDiffHTTPCall:
    """Tests for the _embed() HTTP call."""

    @pytest.mark.asyncio
    async def test_embed_calls_openai_endpoint(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from evalview.core.semantic_diff import SemanticDiff
        sd = SemanticDiff()

        fake_response_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = fake_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await sd._embed(["hello", "world"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "embeddings" in call_kwargs[0][0]  # endpoint URL
        assert call_kwargs[1]["json"]["model"] == "text-embedding-3-small"
