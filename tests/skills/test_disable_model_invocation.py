"""Regression tests for `disable-model-invocation` handling.

Covers:
- Parser: hyphenated frontmatter key is honored and exposed as
  ``metadata.disable_model_invocation`` (default False).
- Skill doctor: manual-only skills are excluded from the 15k character
  budget math but still counted for validation (files, duplicates, etc).
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from evalview.cli import main
from evalview.skills.parser import SkillParser
from evalview.skills.validator import SkillValidator


# ---------------------------------------------------------------------------
# Parser behavior
# ---------------------------------------------------------------------------


def _write_skill(
    directory: Path,
    *,
    name: str,
    description: str,
    disable_model_invocation: bool | None = None,
) -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
    ]
    if disable_model_invocation is not None:
        lines.append(f"disable-model-invocation: {str(disable_model_invocation).lower()}")
    lines.extend(["---", "", "# Instructions", "", "Do the thing.", ""])
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("\n".join(lines))
    return skill_path


def test_parser_defaults_disable_model_invocation_to_false(tmp_path: Path) -> None:
    skill_path = _write_skill(
        tmp_path,
        name="plain-skill",
        description="A plain skill without the manual-only flag set.",
    )

    skill = SkillParser.parse_file(str(skill_path))

    assert skill.metadata.disable_model_invocation is False


def test_parser_honors_disable_model_invocation_true(tmp_path: Path) -> None:
    skill_path = _write_skill(
        tmp_path,
        name="manual-skill",
        description="A manual-only skill that should not be auto-invoked.",
        disable_model_invocation=True,
    )

    skill = SkillParser.parse_file(str(skill_path))

    assert skill.metadata.disable_model_invocation is True


def test_parser_honors_disable_model_invocation_false(tmp_path: Path) -> None:
    skill_path = _write_skill(
        tmp_path,
        name="auto-skill",
        description="An auto-invokable skill with the flag explicitly false.",
        disable_model_invocation=False,
    )

    skill = SkillParser.parse_file(str(skill_path))

    assert skill.metadata.disable_model_invocation is False


def test_validator_accepts_manual_only_skill(tmp_path: Path) -> None:
    skill_path = _write_skill(
        tmp_path,
        name="manual-skill",
        description="Manual-only skill — validator must still treat it as valid.",
        disable_model_invocation=True,
    )

    result = SkillValidator.validate_file(str(skill_path))

    assert result.valid, result.errors
    assert result.skill is not None
    assert result.skill.metadata.disable_model_invocation is True


# ---------------------------------------------------------------------------
# Skill doctor budget exclusion
# ---------------------------------------------------------------------------


def _long_description(prefix: str, target_chars: int) -> str:
    # Keep it single-line (multiline descriptions are validation warnings),
    # avoid XML-like tags or colons (which would confuse YAML parsing as an
    # unquoted mapping value), and stay well under the 1024 per-skill max.
    base = f"{prefix} - " + "alpha beta gamma delta epsilon zeta eta theta iota kappa "
    padded = (base * ((target_chars // len(base)) + 1))[:target_chars]
    return padded.replace("\n", " ").strip()


def test_doctor_excludes_manual_only_skills_from_budget(tmp_path: Path) -> None:
    """Manual-only skills must not push the budget over.

    Two ~900-char auto-invoke skills (~1.8k chars) stay well under the 15k
    budget. If we additionally add many manual-only skills whose descriptions
    *would* exceed the budget when counted, the doctor must still report the
    budget as green and must not claim any skill is "INVISIBLE".
    """
    # Two normal skills — safely under budget on their own.
    _write_skill(
        tmp_path,
        name="auto-one",
        description=_long_description("auto-one", 900),
    )
    _write_skill(
        tmp_path,
        name="auto-two",
        description=_long_description("auto-two", 900),
    )
    # Many manual-only skills whose descriptions, if counted, would blow the
    # 15k budget several times over.
    for i in range(30):
        _write_skill(
            tmp_path,
            name=f"manual-{i:02d}",
            description=_long_description(f"manual-{i}", 900),
            disable_model_invocation=True,
        )

    runner = CliRunner()
    result = runner.invoke(main, ["skill", "doctor", str(tmp_path), "-r"])

    assert result.exit_code == 0, result.output
    # Under budget → green summary banner, no "ignoring" / "INVISIBLE" noise.
    assert "Character Budget:" in result.output
    assert "OVER" not in result.output, result.output
    assert "INVISIBLE" not in result.output, result.output
    assert "ignoring" not in result.output, result.output
    # Manual-only line is surfaced with the right count and context.
    assert "Manual-only:" in result.output
    assert "30" in result.output
    assert "disable-model-invocation" in result.output
    # All 32 skills are counted as valid (validation is not skipped).
    assert "Total Skills:      32" in result.output


def test_doctor_counts_auto_invoke_skills_in_budget(tmp_path: Path) -> None:
    """Sanity check: without the flag, the same skills DO exceed the budget.

    This proves the exclusion in ``test_doctor_excludes_manual_only_skills_from_budget``
    is load-bearing — the descriptions themselves are heavy enough to trip the
    budget when they are not marked manual-only.
    """
    for i in range(30):
        _write_skill(
            tmp_path,
            name=f"auto-{i:02d}",
            description=_long_description(f"auto-{i}", 900),
        )

    runner = CliRunner()
    result = runner.invoke(main, ["skill", "doctor", str(tmp_path), "-r"])

    assert result.exit_code == 0, result.output
    assert "OVER" in result.output, result.output
    assert "model-invokable skills" in result.output, result.output
