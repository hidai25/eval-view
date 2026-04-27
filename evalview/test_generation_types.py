"""Datatypes used by the agent test generator.

Extracted from test_generation.py so the generator class can import them
without dragging in the surrounding 1.5k-line module.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

from evalview.core.types import TestCase


@dataclass
class ProbeResult:
    """Captured behavior from a single probe."""

    query: str
    trace: Any
    tools: List[str]
    signature: str
    behavior_class: str
    rationale: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    safety_probe: bool = False
    prompt_source: str = "unknown"


@dataclass
class GenerationResult:
    """Suite-generation output."""

    tests: List[TestCase] = field(default_factory=list)
    probes_run: int = 0
    signatures_seen: Counter[str] = field(default_factory=Counter)
    tools_seen: Counter[str] = field(default_factory=Counter)
    failures: List[str] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptCandidate:
    """A prompt sourced from workspace docs / existing tests / synthesis."""

    text: str
    source: str


def _normalize_text_for_comparison(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for dedupe checks."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()
