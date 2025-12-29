"""Golden trace storage and management.

Golden traces are "blessed" baseline traces that represent expected behavior.
When running tests with --diff, new traces are compared against golden traces
to detect regressions.

Storage format:
  .evalview/golden/
    <test-name>.golden.json    # The golden trace
    <test-name>.meta.json      # Metadata (when blessed, by whom, etc.)
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging

from evalview.core.types import EvaluationResult, ExecutionTrace, StepTrace

logger = logging.getLogger(__name__)


class GoldenMetadata(BaseModel):
    """Metadata about a golden trace."""

    test_name: str
    blessed_at: datetime
    blessed_by: str = "user"  # Could be "ci", "user", etc.
    source_result_file: Optional[str] = None
    score: float
    notes: Optional[str] = None
    version: int = 1  # For future format migrations


class GoldenTrace(BaseModel):
    """A golden trace with metadata."""

    metadata: GoldenMetadata
    trace: ExecutionTrace
    # Key fields extracted for easy comparison
    tool_sequence: List[str] = Field(default_factory=list)
    output_hash: str = ""  # Hash of final output for quick comparison


class GoldenStore:
    """Manages golden trace storage and retrieval."""

    def __init__(self, base_path: Path = None):
        """
        Initialize golden store.

        Args:
            base_path: Base directory for .evalview (default: current dir)
        """
        self.base_path = base_path or Path(".")
        self.golden_dir = self.base_path / ".evalview" / "golden"

    def _get_golden_path(self, test_name: str) -> Path:
        """Get path to golden trace file for a test."""
        # Sanitize test name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in test_name)
        return self.golden_dir / f"{safe_name}.golden.json"

    def _hash_output(self, output: str) -> str:
        """Create a hash of the output for quick comparison."""
        return hashlib.md5(output.encode()).hexdigest()[:8]

    def save_golden(
        self,
        result: EvaluationResult,
        notes: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> Path:
        """
        Save a test result as the golden trace.

        Args:
            result: The evaluation result to bless
            notes: Optional notes about why this is golden
            source_file: Original result file path

        Returns:
            Path to saved golden file
        """
        self.golden_dir.mkdir(parents=True, exist_ok=True)

        # Extract tool sequence
        tool_sequence = [step.tool_name for step in result.trace.steps]

        # Create golden trace
        golden = GoldenTrace(
            metadata=GoldenMetadata(
                test_name=result.test_case,
                blessed_at=datetime.now(),
                blessed_by="user",
                source_result_file=source_file,
                score=result.score,
                notes=notes,
            ),
            trace=result.trace,
            tool_sequence=tool_sequence,
            output_hash=self._hash_output(result.trace.final_output),
        )

        # Save
        golden_path = self._get_golden_path(result.test_case)
        with open(golden_path, "w") as f:
            f.write(golden.model_dump_json(indent=2))

        logger.info(f"Saved golden trace: {golden_path}")
        return golden_path

    def load_golden(self, test_name: str) -> Optional[GoldenTrace]:
        """
        Load golden trace for a test.

        Args:
            test_name: Name of the test

        Returns:
            GoldenTrace or None if not found
        """
        golden_path = self._get_golden_path(test_name)
        if not golden_path.exists():
            return None

        with open(golden_path) as f:
            data = json.load(f)

        return GoldenTrace.model_validate(data)

    def has_golden(self, test_name: str) -> bool:
        """Check if a golden trace exists for a test."""
        return self._get_golden_path(test_name).exists()

    def list_golden(self) -> List[GoldenMetadata]:
        """List all golden traces."""
        if not self.golden_dir.exists():
            return []

        results = []
        for path in self.golden_dir.glob("*.golden.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                results.append(GoldenMetadata.model_validate(data["metadata"]))
            except Exception as e:
                logger.warning(f"Failed to load golden {path}: {e}")

        return results

    def delete_golden(self, test_name: str) -> bool:
        """Delete a golden trace."""
        golden_path = self._get_golden_path(test_name)
        if golden_path.exists():
            golden_path.unlink()
            return True
        return False


# Convenience functions
_default_store: Optional[GoldenStore] = None


def get_store(base_path: Path = None) -> GoldenStore:
    """Get the golden store (creates if needed)."""
    global _default_store
    if _default_store is None or base_path is not None:
        _default_store = GoldenStore(base_path)
    return _default_store
