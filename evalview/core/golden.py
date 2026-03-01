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

from evalview.core.types import EvaluationResult, ExecutionTrace

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

    # Model fingerprint — captured from the API response at snapshot time.
    # Both fields are Optional so existing .golden.json files load without error.
    model_id: Optional[str] = None        # e.g. "claude-3-5-sonnet-20241022"
    model_provider: Optional[str] = None  # e.g. "anthropic"


class GoldenTrace(BaseModel):
    """A golden trace with metadata."""

    metadata: GoldenMetadata
    trace: ExecutionTrace
    # Key fields extracted for easy comparison
    tool_sequence: List[str] = Field(default_factory=list)
    output_hash: str = ""  # Hash of final output for quick comparison


class GoldenStore:
    """Manages golden trace storage and retrieval."""

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize golden store.

        Args:
            base_path: Base directory for .evalview (default: current dir)
        """
        self.base_path = base_path or Path(".")
        self.golden_dir = self.base_path / ".evalview" / "golden"

    def _get_golden_path(self, test_name: str, variant_name: Optional[str] = None) -> Path:
        """Get path to golden trace file for a test.

        Args:
            test_name: Name of the test
            variant_name: Optional variant name for multi-reference goldens

        Returns:
            Path to golden file
        """
        # Sanitize test name for filesystem (remove dots to prevent path traversal)
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in test_name)

        if variant_name:
            # Sanitize variant name too (remove dots to prevent path traversal)
            safe_variant = "".join(c if c.isalnum() or c in "_-" else "_" for c in variant_name)
            return self.golden_dir / f"{safe_name}.variant_{safe_variant}.golden.json"
        else:
            return self.golden_dir / f"{safe_name}.golden.json"

    def _hash_output(self, output: str) -> str:
        """Create a hash of the output for quick comparison."""
        return hashlib.md5(output.encode()).hexdigest()[:8]

    def save_golden(
        self,
        result: EvaluationResult,
        notes: Optional[str] = None,
        source_file: Optional[str] = None,
        variant_name: Optional[str] = None,
    ) -> Path:
        """
        Save a test result as the golden trace.

        Args:
            result: The evaluation result to bless
            notes: Optional notes about why this is golden
            source_file: Original result file path
            variant_name: Optional variant name for multi-reference goldens (max 5 variants)

        Returns:
            Path to saved golden file

        Raises:
            ValueError: If trying to save more than 5 variants
        """
        self.golden_dir.mkdir(parents=True, exist_ok=True)

        # Check variant limit (max 5 variants per test)
        if variant_name:
            existing_variants = self.count_variants(result.test_case)
            if existing_variants >= 5 and not self._get_golden_path(result.test_case, variant_name).exists():
                raise ValueError(
                    f"Maximum 5 variants allowed per test. Test '{result.test_case}' already has {existing_variants} variants. "
                    f"Delete an existing variant first."
                )

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
                model_id=getattr(result.trace, "model_id", None),
                model_provider=getattr(result.trace, "model_provider", None),
            ),
            trace=result.trace,
            tool_sequence=tool_sequence,
            output_hash=self._hash_output(result.trace.final_output),
        )

        # Save
        golden_path = self._get_golden_path(result.test_case, variant_name)
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

    def delete_golden(self, test_name: str, variant_name: Optional[str] = None) -> bool:
        """Delete a golden trace.

        Args:
            test_name: Name of the test
            variant_name: Optional variant to delete (deletes default if None)

        Returns:
            True if deleted, False if not found
        """
        golden_path = self._get_golden_path(test_name, variant_name)
        if golden_path.exists():
            golden_path.unlink()
            return True
        return False

    def load_all_golden_variants(self, test_name: str) -> List[GoldenTrace]:
        """Load all golden variants for a test (default + all named variants).

        Args:
            test_name: Name of the test

        Returns:
            List of GoldenTrace objects (empty if none found)
        """
        variants = []

        # Load default golden
        default = self.load_golden(test_name)
        if default:
            variants.append(default)

        # Load all variant goldens
        if not self.golden_dir.exists():
            return variants

        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in test_name)
        pattern = f"{safe_name}.variant_*.golden.json"

        for path in self.golden_dir.glob(pattern):
            try:
                with open(path) as f:
                    data = json.load(f)
                variants.append(GoldenTrace.model_validate(data))
            except Exception as e:
                logger.warning(f"Failed to load variant golden {path}: {e}")

        return variants

    def count_variants(self, test_name: str) -> int:
        """Count how many variants exist for a test.

        Args:
            test_name: Name of the test

        Returns:
            Number of variants (including default)
        """
        return len(self.load_all_golden_variants(test_name))

    def list_golden_with_variants(self) -> List[Dict[str, Any]]:
        """List all golden traces with variant counts.

        Returns:
            List of dicts with 'metadata' and 'variant_count' keys
        """
        if not self.golden_dir.exists():
            return []

        # Group by test name
        test_groups: Dict[str, Dict[str, Any]] = {}

        for path in self.golden_dir.glob("*.golden.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                metadata = GoldenMetadata.model_validate(data["metadata"])
                test_name = metadata.test_name

                if test_name not in test_groups:
                    test_groups[test_name] = {
                        "metadata": metadata,
                        "variant_count": 0
                    }

                test_groups[test_name]["variant_count"] += 1

            except Exception as e:
                logger.warning(f"Failed to load golden {path}: {e}")

        return list(test_groups.values())


# Convenience functions
_default_store: Optional[GoldenStore] = None


def get_store(base_path: Optional[Path] = None) -> GoldenStore:
    """Get the golden store (creates if needed)."""
    global _default_store
    if _default_store is None or base_path is not None:
        _default_store = GoldenStore(base_path)
    return _default_store
