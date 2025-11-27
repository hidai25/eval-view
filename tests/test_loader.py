"""Tests for YAML test case loader."""

import pytest
from pydantic import ValidationError
import yaml

from evalview.core.loader import TestCaseLoader
from evalview.core.types import TestCase as TestCaseModel


class TestTestCaseLoaderFromFile:
    """Tests for loading a single test case from a file."""

    def test_load_valid_yaml_file(self, temp_yaml_file):
        """Test loading a valid YAML test case file."""
        test_case = TestCaseLoader.load_from_file(temp_yaml_file)

        assert isinstance(test_case, TestCaseModel)
        assert test_case.name == "test_search"
        assert test_case.description == "Test search functionality"
        assert test_case.input.query == "What is the capital of France?"
        assert test_case.input.context == {"language": "en"}
        assert test_case.expected.tools == ["search", "summarize"]
        assert test_case.expected.tool_sequence == ["search", "summarize"]
        assert test_case.expected.output.contains == ["Paris", "France"]
        assert test_case.expected.output.not_contains == ["London", "error"]
        assert test_case.thresholds.min_score == 70.0
        assert test_case.thresholds.max_cost == 0.50
        assert test_case.thresholds.max_latency == 5000.0

    def test_load_with_string_path(self, temp_yaml_file):
        """Test loading with string path instead of Path object."""
        test_case = TestCaseLoader.load_from_file(str(temp_yaml_file))
        assert isinstance(test_case, TestCaseModel)
        assert test_case.name == "test_search"

    def test_load_with_path_object(self, temp_yaml_file):
        """Test loading with Path object."""
        test_case = TestCaseLoader.load_from_file(temp_yaml_file)
        assert isinstance(test_case, TestCaseModel)
        assert test_case.name == "test_search"

    def test_load_invalid_yaml_syntax(self, temp_invalid_yaml_file):
        """Test loading a file with invalid YAML syntax."""
        with pytest.raises(yaml.YAMLError):
            TestCaseLoader.load_from_file(temp_invalid_yaml_file)

    def test_load_invalid_schema(self, temp_invalid_schema_file):
        """Test loading a YAML file with invalid schema (missing required fields)."""
        with pytest.raises(ValidationError) as exc_info:
            TestCaseLoader.load_from_file(temp_invalid_schema_file)

        # Check that the error mentions missing fields
        error_str = str(exc_info.value)
        assert "input" in error_str or "expected" in error_str or "thresholds" in error_str

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a file that doesn't exist."""
        nonexistent_file = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            TestCaseLoader.load_from_file(nonexistent_file)

    def test_load_minimal_valid_case(self, tmp_path):
        """Test loading a minimal valid test case (only required fields)."""
        minimal_yaml = """
name: minimal_test
input:
  query: test query
expected:
  tools: []
thresholds:
  min_score: 50.0
"""
        file_path = tmp_path / "minimal.yaml"
        file_path.write_text(minimal_yaml)

        test_case = TestCaseLoader.load_from_file(file_path)
        assert test_case.name == "minimal_test"
        assert test_case.input.query == "test query"
        assert test_case.thresholds.min_score == 50.0
        assert test_case.description is None  # Optional field

    def test_load_with_adapter_override(self, tmp_path):
        """Test loading a test case with adapter override fields."""
        yaml_content = """
name: test_with_adapter
input:
  query: test query
expected:
  tools: []
thresholds:
  min_score: 70.0
adapter: langgraph
endpoint: http://localhost:8000
adapter_config:
  streaming: true
  timeout: 60.0
"""
        file_path = tmp_path / "adapter_test.yaml"
        file_path.write_text(yaml_content)

        test_case = TestCaseLoader.load_from_file(file_path)
        assert test_case.adapter == "langgraph"
        assert test_case.endpoint == "http://localhost:8000"
        assert test_case.adapter_config == {"streaming": True, "timeout": 60.0}

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty YAML file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises((ValidationError, TypeError)):
            TestCaseLoader.load_from_file(empty_file)

    def test_load_yaml_with_null_values(self, tmp_path):
        """Test loading YAML with explicit null values."""
        yaml_content = """
name: test_null
description: null
input:
  query: test query
  context: null
expected:
  tools: null
  tool_sequence: null
  output: null
thresholds:
  min_score: 50.0
  max_cost: null
  max_latency: null
"""
        file_path = tmp_path / "null_values.yaml"
        file_path.write_text(yaml_content)

        test_case = TestCaseLoader.load_from_file(file_path)
        assert test_case.name == "test_null"
        assert test_case.description is None
        assert test_case.input.context is None
        assert test_case.expected.tools is None
        assert test_case.thresholds.max_cost is None


class TestTestCaseLoaderFromDirectory:
    """Tests for loading multiple test cases from a directory."""

    def test_load_from_directory(self, temp_yaml_directory):
        """Test loading all test cases from a directory."""
        test_cases = TestCaseLoader.load_from_directory(temp_yaml_directory)

        assert len(test_cases) == 2  # test1.yaml and test2.yml
        names = [tc.name for tc in test_cases]
        assert "test1" in names
        assert "test2" in names

    def test_load_from_directory_string_path(self, temp_yaml_directory):
        """Test loading from directory using string path."""
        test_cases = TestCaseLoader.load_from_directory(str(temp_yaml_directory))
        assert len(test_cases) == 2

    def test_load_from_directory_both_extensions(self, temp_yaml_directory):
        """Test that both .yaml and .yml files are loaded."""
        test_cases = TestCaseLoader.load_from_directory(temp_yaml_directory)

        # Verify we have one .yaml and one .yml file
        yaml_files = list(temp_yaml_directory.glob("*.yaml"))
        yml_files = list(temp_yaml_directory.glob("*.yml"))

        assert len(yaml_files) == 1
        assert len(yml_files) == 1
        assert len(test_cases) == 2  # Both should be loaded

    def test_load_from_directory_ignores_non_yaml(self, temp_yaml_directory):
        """Test that non-YAML files are ignored."""
        # temp_yaml_directory already has a readme.txt file
        test_cases = TestCaseLoader.load_from_directory(temp_yaml_directory)

        # Should only load YAML files, not readme.txt
        assert len(test_cases) == 2
        for tc in test_cases:
            assert tc.name in ["test1", "test2"]

    def test_load_from_empty_directory(self, tmp_path):
        """Test loading from an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        test_cases = TestCaseLoader.load_from_directory(empty_dir)
        assert test_cases == []

    def test_load_from_nonexistent_directory(self, tmp_path):
        """Test loading from a directory that doesn't exist."""
        nonexistent_dir = tmp_path / "does_not_exist"

        # Path.glob() on nonexistent directory returns empty iterator, so we get empty list
        # This is consistent with Python's Path behavior
        test_cases = TestCaseLoader.load_from_directory(nonexistent_dir)
        assert test_cases == []

    def test_load_from_directory_with_subdirectories(self, tmp_path):
        """Test that subdirectories are not searched (only top-level files)."""
        test_dir = tmp_path / "test_cases"
        test_dir.mkdir()
        sub_dir = test_dir / "subdir"
        sub_dir.mkdir()

        # Create YAML in top-level
        (test_dir / "top_level.yaml").write_text(
            """
name: top_level
input:
  query: test
expected:
  tools: []
thresholds:
  min_score: 50.0
"""
        )

        # Create YAML in subdirectory
        (sub_dir / "nested.yaml").write_text(
            """
name: nested
input:
  query: test
expected:
  tools: []
thresholds:
  min_score: 50.0
"""
        )

        test_cases = TestCaseLoader.load_from_directory(test_dir)

        # Should only load top-level file (glob pattern *.yaml doesn't recurse)
        assert len(test_cases) == 1
        assert test_cases[0].name == "top_level"

    def test_load_from_directory_custom_pattern(self, tmp_path):
        """Test loading with a custom file pattern."""
        test_dir = tmp_path / "test_cases"
        test_dir.mkdir()

        # Create files with different extensions
        (test_dir / "test1.yaml").write_text(
            """
name: test1
input:
  query: test
expected:
  tools: []
thresholds:
  min_score: 50.0
"""
        )

        (test_dir / "test2.yml").write_text(
            """
name: test2
input:
  query: test
expected:
  tools: []
thresholds:
  min_score: 50.0
"""
        )

        (test_dir / "test3.json").write_text('{"name": "test3"}')

        # Load only .yaml files (not .yml)
        # Note: When pattern is not "*.yaml", the loader won't auto-include .yml files
        test_cases = TestCaseLoader.load_from_directory(test_dir, pattern="*.yaml")

        # Should load test1.yaml and test2.yml (because of the special .yml handling)
        assert len(test_cases) == 2

    def test_load_from_directory_with_invalid_file(self, tmp_path):
        """Test that one invalid file doesn't prevent loading others."""
        test_dir = tmp_path / "test_cases"
        test_dir.mkdir()

        # Create a valid file
        (test_dir / "valid.yaml").write_text(
            """
name: valid
input:
  query: test
expected:
  tools: []
thresholds:
  min_score: 50.0
"""
        )

        # Create an invalid file (missing required fields)
        (test_dir / "invalid.yaml").write_text(
            """
name: invalid
description: This is missing required fields
"""
        )

        # Loading should raise an error when it hits the invalid file
        with pytest.raises(ValidationError):
            TestCaseLoader.load_from_directory(test_dir)

    def test_load_preserves_field_values(self, tmp_path):
        """Test that all field values are correctly preserved when loading."""
        test_dir = tmp_path / "test_cases"
        test_dir.mkdir()

        yaml_content = """
name: comprehensive_test
description: Test with all fields populated
input:
  query: What is 2+2?
  context:
    difficulty: easy
    category: math
expected:
  tools:
    - calculator
    - formatter
  tool_sequence:
    - calculator
    - formatter
  output:
    contains:
      - "4"
      - "four"
    not_contains:
      - "error"
      - "undefined"
    json_schema:
      type: object
      properties:
        answer:
          type: number
  metrics:
    accuracy:
      value: 0.95
      tolerance: 0.05
thresholds:
  min_score: 85.0
  max_cost: 0.25
  max_latency: 3000.0
adapter: custom_adapter
endpoint: http://example.com/api
adapter_config:
  timeout: 45
  retries: 3
"""
        file_path = test_dir / "comprehensive.yaml"
        file_path.write_text(yaml_content)

        test_cases = TestCaseLoader.load_from_directory(test_dir)
        assert len(test_cases) == 1

        tc = test_cases[0]
        assert tc.name == "comprehensive_test"
        assert tc.description == "Test with all fields populated"
        assert tc.input.query == "What is 2+2?"
        assert tc.input.context == {"difficulty": "easy", "category": "math"}
        assert tc.expected.tools == ["calculator", "formatter"]
        assert tc.expected.tool_sequence == ["calculator", "formatter"]
        assert tc.expected.output.contains == ["4", "four"]
        assert tc.expected.output.not_contains == ["error", "undefined"]
        assert tc.expected.output.json_schema["type"] == "object"
        assert "accuracy" in tc.expected.metrics
        assert tc.expected.metrics["accuracy"].value == 0.95
        assert tc.thresholds.min_score == 85.0
        assert tc.thresholds.max_cost == 0.25
        assert tc.thresholds.max_latency == 3000.0
        assert tc.adapter == "custom_adapter"
        assert tc.endpoint == "http://example.com/api"
        assert tc.adapter_config == {"timeout": 45, "retries": 3}
