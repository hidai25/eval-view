"""Semantic similarity using OpenAI's embedding API.

Provides embedding-based output comparison as an opt-in enhancement
to the default lexical (SequenceMatcher) comparison in DiffEngine.

Requires:
    OPENAI_API_KEY environment variable (or explicit api_key parameter)

Cost:
    ~$0.00002 per call with text-embedding-3-small pricing.
    Each comparison makes a single batched call (2 texts), so ~$0.00004 per check.

Enable via .evalview/config.yaml:
    diff:
      semantic_diff_enabled: true

Or per-run:
    evalview check --semantic-diff
"""

import math
import os
from typing import List, Optional

import httpx


class SemanticDiff:
    """Embedding-based semantic similarity for output comparison.

    Uses OpenAI's text-embedding-3-small model to compute cosine similarity
    between two texts. This catches semantic drift (meaning changes) that
    lexical comparison misses — e.g., an updated model that switches from
    bullet-point summaries to prose while preserving the same information.

    Example:
        >>> diff = SemanticDiff()
        >>> score = asyncio.run(diff.similarity(golden_output, actual_output))
        >>> print(f"Semantic similarity: {score:.0%}")
    """

    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_ENDPOINT = "https://api.openai.com/v1/embeddings"
    # text-embedding-3-small: $0.02 per 1M tokens
    # ~1000 tokens per typical agent output → ~$0.00002 per call
    COST_PER_CALL_USD = 0.00002

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
    ):
        """Initialize SemanticDiff.

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            model: Embedding model to use (default: text-embedding-3-small).

        Raises:
            ValueError: If no API key is available.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable or explicit api_key required "
                "for semantic diff. Set OPENAI_API_KEY or disable "
                "semantic_diff_enabled in .evalview/config.yaml."
            )
        self.model = model

    async def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text_a: First text (e.g., golden output).
            text_b: Second text (e.g., actual output).

        Returns:
            Cosine similarity in [0.0, 1.0]. Higher means more similar.
            Returns 1.0 for identical inputs, ~0.0 for unrelated texts.
        """
        # Truncate to stay within embedding model token limits (8192 tokens)
        embeddings = await self._embed([text_a[:8192], text_b[:8192]])
        a, b = embeddings[0], embeddings[1]
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embeddings API (single batched call for both texts).

        Args:
            texts: List of texts to embed in one API call.

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            httpx.HTTPStatusError: If the API call fails.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.EMBEDDING_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"input": texts, "model": self.model},
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]

    @classmethod
    def is_available(cls) -> bool:
        """Check if semantic diff can be used without raising errors."""
        return bool(os.environ.get("OPENAI_API_KEY"))

    @classmethod
    def cost_notice(cls) -> str:
        """Return a human-readable cost notice for display in CLI output."""
        return (
            f"Semantic diff enabled — 1 batched embedding call per test "
            f"(~${cls.COST_PER_CALL_USD * 2:.5f}, {cls.EMBEDDING_MODEL})"
        )
