"""Test case loader from YAML files."""

from pathlib import Path
from typing import List, Union
import yaml
from evalview.core.types import TestCase

# Files to skip when loading test cases (not test cases themselves)
CONFIG_FILE_PATTERNS = {"config.yaml", "config.yml", ".evalview.yaml", ".evalview.yml"}


def _is_config_file(file_path: Path) -> bool:
    """Check if a file is a config file (not a test case)."""
    return file_path.name.lower() in CONFIG_FILE_PATTERNS


class TestCaseLoader:
    """Loads test cases from YAML files."""

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> TestCase:
        """
        Load a single test case from a YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            TestCase instance
        """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return TestCase(**data)

    @staticmethod
    def load_from_directory(directory: Union[str, Path], pattern: str = "*.yaml") -> List[TestCase]:
        """
        Load all test cases from a directory.

        Automatically skips config files (config.yaml, config.yml, etc.).

        Args:
            directory: Directory containing YAML files
            pattern: File pattern to match (default: *.yaml)

        Returns:
            List of TestCase instances
        """
        dir_path = Path(directory)
        test_cases = []

        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and not _is_config_file(file_path):
                test_cases.append(TestCaseLoader.load_from_file(file_path))

        # Also check for .yml extension
        if pattern == "*.yaml":
            for file_path in dir_path.glob("*.yml"):
                if file_path.is_file() and not _is_config_file(file_path):
                    test_cases.append(TestCaseLoader.load_from_file(file_path))

        return test_cases
