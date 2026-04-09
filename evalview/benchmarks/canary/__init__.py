"""Bundled canary suites for `evalview model-check`.

This package holds stable prompt suites shipped with EvalView that are
designed to detect behavioral drift in closed-weight models over time.

See ``README.md`` in this directory for the design principles that all
canary suites must follow (structural scoring only, no dates, versioned
not mutated, etc.).
"""
from __future__ import annotations

from pathlib import Path

CANARY_DIR = Path(__file__).parent
PUBLIC_SUITE_PATH = CANARY_DIR / "suite.v1.public.yaml"
HELD_OUT_SUITE_PATH = CANARY_DIR / "suite.v1.held-out.yaml"

__all__ = ["CANARY_DIR", "PUBLIC_SUITE_PATH", "HELD_OUT_SUITE_PATH"]
