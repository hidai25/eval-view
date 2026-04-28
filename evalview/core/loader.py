"""Test case loader from YAML and TOML files."""

import sys
from pathlib import Path
from typing import Any, List, Union
import yaml
from evalview.core.types import TestCase

# tomllib is stdlib on Python 3.11+. Fall back to the third-party `tomli`
# package on 3.9/3.10 (same parser, same API).
if sys.version_info >= (3, 11):
    import tomllib as _tomllib
else:  # pragma: no cover - exercised on 3.9/3.10 CI runners
    try:
        import tomli as _tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        _tomllib = None  # type: ignore[assignment]

# Files to skip when loading test cases (not test cases themselves)
CONFIG_FILE_PATTERNS = {
    "config.yaml",
    "config.yml",
    "config.toml",
    ".evalview.yaml",
    ".evalview.yml",
    ".evalview.toml",
}

# Suffixes the directory loader recognises as test-case files.
TEST_CASE_SUFFIXES = (".yaml", ".yml", ".toml")


def _is_config_file(file_path: Path) -> bool:
    """Check if a file is a config file (not a test case)."""
    return file_path.name.lower() in CONFIG_FILE_PATTERNS


def _parse_test_case_file(file_path: Path) -> Any:
    """Parse a YAML or TOML test case file into a dict.

    Dispatches on suffix so callers don't need to know the file format.
    TOML support requires `tomllib` (Python 3.11+) or `tomli` on 3.9/3.10.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".toml":
        if _tomllib is None:
            raise ImportError(
                "TOML test cases require Python 3.11+ or the 'tomli' package "
                "on Python 3.9/3.10. Install with: pip install tomli"
            )
        with open(file_path, "rb") as f:
            return _tomllib.load(f)
    # Default to YAML for .yaml / .yml / unknown.
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


class TestCaseLoader:
    """Loads test cases from YAML and TOML files."""

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> TestCase:
        """
        Load a single test case from a YAML or TOML file.

        Format is dispatched on file extension: ``.toml`` is parsed with
        ``tomllib`` (Python 3.11+) or ``tomli`` (3.9/3.10); everything else
        is parsed as YAML.

        Args:
            file_path: Path to YAML or TOML file

        Returns:
            TestCase instance
        """
        path = Path(file_path)
        data = _parse_test_case_file(path)
        test_case = TestCase(**data)
        test_case.source_file = str(path)
        return test_case

    @staticmethod
    def load_from_directory(directory: Union[str, Path], pattern: str = "*.yaml") -> List[TestCase]:
        """
        Load all test cases from a directory.

        Automatically skips config files (config.yaml, config.yml,
        config.toml, .evalview.*). When called with the default
        ``pattern="*.yaml"``, ``.yml`` and ``.toml`` files in the same
        directory tree are also picked up so callers don't need to
        invoke the loader once per format.

        Args:
            directory: Directory containing test case files
            pattern: File pattern to match (default: *.yaml)

        Returns:
            List of TestCase instances
        """
        dir_path = Path(directory)
        test_cases: List[TestCase] = []
        seen: set = set()

        def _ingest(file_path: Path) -> None:
            if not file_path.is_file() or _is_config_file(file_path):
                return
            resolved = file_path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            test_cases.append(TestCaseLoader.load_from_file(file_path))

        for file_path in dir_path.rglob(pattern):
            _ingest(file_path)

        # When the caller didn't specify a custom pattern, also pick up the
        # other supported extensions. rglob each one so traversal stays
        # recursive (matches the previous .yml behaviour, extended to .toml).
        if pattern == "*.yaml":
            for suffix in TEST_CASE_SUFFIXES:
                if suffix == ".yaml":
                    continue
                for file_path in dir_path.rglob(f"*{suffix}"):
                    _ingest(file_path)

        return test_cases
