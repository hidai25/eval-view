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
import shutil
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Iterator, List
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
    per_turn_tool_sequences: Optional[List[List[str]]] = Field(default=None)
    per_turn_outputs: Optional[List[Optional[str]]] = Field(default=None)


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
        self._reset_dir = self.golden_dir.parent / ".snapshot-reset"
        self._previous = self._reset_dir / "previous"
        self._prepared = self._reset_dir / "prepared"
        if self._previous.exists() or self._prepared.exists():
            # A reset is publishing, or was interrupted after moving the originals.
            with self._write_lock():
                pass

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        """Serialize golden mutations, including recovery after process termination."""
        self._reset_dir.mkdir(parents=True, exist_ok=True)
        with (self._reset_dir / "lock").open("a+b") as lock:
            if sys.platform == "win32":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            # Both platforms release the lock when the file closes, including on
            # process death. Never unlink the lock file while another writer waits.
            self._finish_reset()
            yield

    def _finish_reset(self) -> None:
        """Restore an interrupted handoff or discard the backup of a completed one."""
        if self._previous.is_dir():
            if not self.golden_dir.exists() or not any(self.golden_dir.iterdir()):
                if self.golden_dir.exists():
                    self.golden_dir.rmdir()
                self._previous.rename(self.golden_dir)
            elif self._prepared.exists():
                raise OSError(f"Cannot restore baselines over {self.golden_dir}. Original files are in {self._previous}.")
            else:
                # The prepared directory has been renamed into place.
                shutil.rmtree(self._previous)
        if self._prepared.is_dir():
            shutil.rmtree(self._prepared)

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
        # Extract tool sequence
        tool_sequence = [step.tool_name for step in result.trace.steps]

        # Build per-turn tool sequences for multi-turn tests
        per_turn: Optional[List[List[str]]] = None
        if any(step.turn_index is not None for step in result.trace.steps):
            turns_map: Dict[int, List[str]] = defaultdict(list)
            for step in result.trace.steps:
                idx = step.turn_index if step.turn_index is not None else 1
                turns_map[idx].append(step.tool_name)
            max_turn = max(turns_map.keys()) if turns_map else 0
            per_turn = [turns_map.get(t, []) for t in range(1, max_turn + 1)]

        # Build per-turn outputs for multi-turn tests
        per_turn_outputs: Optional[List[Optional[str]]] = None
        if result.trace.turns:
            per_turn_outputs = [t.output for t in result.trace.turns]

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
            per_turn_tool_sequences=per_turn,
            per_turn_outputs=per_turn_outputs,
        )

        # Save
        golden_path = self._get_golden_path(result.test_case, variant_name)
        with self._write_lock():
            self.golden_dir.mkdir(parents=True, exist_ok=True)
            if variant_name:
                existing_variants = self.count_variants(result.test_case)
                if existing_variants >= 5 and not golden_path.exists():
                    raise ValueError(
                        f"Maximum 5 variants allowed per test. Test '{result.test_case}' already has {existing_variants} variants. "
                        f"Delete an existing variant first."
                    )
            golden_path.write_text(golden.model_dump_json(indent=2), encoding="utf-8")

        logger.info(f"Saved golden trace: {golden_path}")
        return golden_path

    def replace_all(
        self, results: List[EvaluationResult], notes: Optional[str], variant_name: Optional[str]
    ) -> Dict[str, Path]:
        """Publish a complete snapshot reset after every replacement has been written."""
        if not results or any(not result.passed for result in results):
            raise ValueError("A reset requires a passing result for every selected test.")

        self._reset_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="stage-", dir=self._reset_dir) as workspace:
            staged = GoldenStore(Path(workspace))
            saved = {}
            for result in results:
                path = staged.save_golden(result, notes=notes, variant_name=variant_name)
                saved[result.test_case] = self.golden_dir / path.name
            if len(set(saved.values())) != len(results):
                raise ValueError("Multiple selected tests map to the same baseline file.")
            for name, destination in saved.items():
                path = staged.golden_dir / destination.name
                baseline = GoldenTrace.model_validate_json(path.read_bytes())
                if baseline.metadata.test_name != name:
                    raise ValueError(f"Staged baseline {path.name} does not belong to {name}.")

            with self._write_lock():
                if self.golden_dir.exists():
                    if self.golden_dir.is_symlink():
                        raise ValueError(f"Cannot reset a symlinked baseline directory: {self.golden_dir}.")
                    # Keep project files that are not baselines, e.g. a .gitkeep.
                    shutil.copytree(
                        self.golden_dir, staged.golden_dir, dirs_exist_ok=True, symlinks=True,
                        ignore=lambda directory, names: [
                            name for name in names
                            if Path(directory) == self.golden_dir and name.endswith(".golden.json")
                        ],
                    )
                try:
                    staged.golden_dir.rename(self._prepared)
                    if self.golden_dir.exists():
                        self.golden_dir.rename(self._previous)
                    self._prepared.rename(self.golden_dir)
                finally:
                    self._finish_reset()
        return saved

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
        with self._write_lock():
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

    def save_golden_from_dict(self, test_name: str, data: dict) -> None:
        """Restore a golden baseline from a dict (e.g., downloaded from cloud).

        Args:
            test_name: Name of the test
            data: Golden trace data as a dict
        """
        golden = GoldenTrace.model_validate(data)
        path = self._get_golden_path(test_name, None)
        with self._write_lock():
            self.golden_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(golden.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Restored golden from cloud: {path}")

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
