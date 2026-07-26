"""Regression tests for directory loading alongside skill test suites.

`load_from_directory` recurses, so a skill suite sitting in a subdirectory
(`evalview skill` territory, identified by its `skill:` key) used to be parsed
as a plain TestCase and fail the entire directory load. `evalview check
demo-tests` was broken this way — and because the error named no file, you got
"3 validation errors for TestCase" with nothing to grep for.
"""
from __future__ import annotations

import pytest

from evalview.core.loader import TestCaseLoader, _is_skill_suite

VALID_TEST_CASE = """\
name: good-test
description: A minimal well-formed test case
input:
  query: "say hi"
expected:
  output:
    contains:
      - "hi"
thresholds:
  min_score: 50
"""

SKILL_SUITE = """\
name: test-some-skill
description: Suite for a skill, consumed by `evalview skill`
skill: ./SKILL.md
agent:
  type: custom
  script_path: ./runner.py
min_pass_rate: 0.7
tests:
  - name: does-a-thing
    category: explicit
    input: do the thing
    expected:
      output_contains:
        - thing
"""


class TestSkillSuiteDiscriminator:
    def test_detects_skill_suite(self):
        assert _is_skill_suite({"skill": "./SKILL.md", "tests": []}) is True

    def test_plain_test_case_is_not_a_suite(self):
        assert _is_skill_suite({"name": "x", "input": "y"}) is False

    def test_non_dict_is_not_a_suite(self):
        assert _is_skill_suite(["not", "a", "dict"]) is False
        assert _is_skill_suite(None) is False

    def test_test_case_model_has_no_skill_field(self):
        """Pins the discriminator: if TestCase ever grows a `skill` field,
        `_is_skill_suite` starts silently swallowing real test cases."""
        from evalview.core.types import TestCase

        assert "skill" not in TestCase.model_fields


class TestDirectoryLoading:
    def test_skips_skill_suite_in_subdirectory(self, tmp_path):
        """The demo-tests regression, reduced."""
        (tmp_path / "good.yaml").write_text(VALID_TEST_CASE)
        nested = tmp_path / "some-skill"
        nested.mkdir()
        (nested / "tests.yaml").write_text(SKILL_SUITE)

        cases = TestCaseLoader.load_from_directory(tmp_path)

        assert [c.name for c in cases] == ["good-test"]

    def test_still_skips_config_files(self, tmp_path):
        (tmp_path / "good.yaml").write_text(VALID_TEST_CASE)
        (tmp_path / "config.yaml").write_text("agent:\n  url: http://x\n")

        cases = TestCaseLoader.load_from_directory(tmp_path)

        assert [c.name for c in cases] == ["good-test"]

    def test_malformed_test_case_names_the_file(self, tmp_path):
        """A directory of 30 tests must not fail anonymously."""
        (tmp_path / "good.yaml").write_text(VALID_TEST_CASE)
        (tmp_path / "broken.yaml").write_text("name: broken\ndescription: no input\n")

        with pytest.raises(ValueError) as exc:
            TestCaseLoader.load_from_directory(tmp_path)

        assert "broken.yaml" in str(exc.value)

    def test_source_file_is_recorded(self, tmp_path):
        (tmp_path / "good.yaml").write_text(VALID_TEST_CASE)

        cases = TestCaseLoader.load_from_directory(tmp_path)

        assert cases[0].source_file is not None
        assert cases[0].source_file.endswith("good.yaml")


class TestShippedDemoTests:
    def test_demo_tests_directory_loads(self):
        """`evalview check demo-tests` is a documented entry point."""
        cases = TestCaseLoader.load_from_directory("demo-tests")

        assert len(cases) >= 3
        names = {c.name for c in cases}
        assert "calculator-addition" in names
