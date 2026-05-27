"""Tiny shared text helpers used by the Jaccard-style analyses.

Centralizing the stoplist here keeps ``freshness``, ``goal_drift``, and
``retrieval_lineage`` consistent. Kept intentionally small: Jaccard on a bag
of words is already coarse, and aggressive stopword filtering throws away
signal.
"""
from __future__ import annotations

from typing import FrozenSet


STOPWORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "ours",
    "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "about", "into", "than",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "not", "no", "so", "just", "also", "very", "really", "please",
})
