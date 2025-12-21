"""Skill validator for comprehensive validation checks."""

import re
from pathlib import Path
from typing import List, Optional

from evalview.skills.types import (
    Skill,
    SkillValidationResult,
    SkillValidationError,
    SkillSeverity,
)
from evalview.skills.parser import SkillParser, SkillParseError


class SkillValidator:
    """Comprehensive validator for Claude Code skills.

    Performs multiple validation checks:
    - Structure: Frontmatter format, required fields
    - Naming: Valid skill names, no conflicts
    - Content: Instructions quality, token size
    - Policy: No prohibited patterns, safe instructions
    - Best practices: Guidelines adherence
    """

    # Token limit recommendations
    METADATA_TOKEN_ESTIMATE = 100  # ~100 tokens for metadata scanning
    MAX_RECOMMENDED_TOKENS = 5000  # Skills should be under 5k tokens when loaded

    # Naming rules (per official Anthropic spec)
    NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$|^[a-z]$")
    NAME_MAX_LENGTH = 64  # Official spec
    DESCRIPTION_MAX_LENGTH = 1024  # Official spec
    RESERVED_WORDS = ["anthropic", "claude"]  # Cannot appear in name
    XML_TAG_PATTERN = re.compile(r"<[^>]+>")

    # Prohibited patterns in instructions (policy compliance)
    PROHIBITED_PATTERNS = [
        (r"\bignore\s+(all\s+)?previous\s+instructions?\b", "Prompt injection attempt"),
        (r"\bforget\s+(everything|all)\b", "Prompt injection attempt"),
        (r"\byou\s+are\s+now\b", "Role hijacking attempt"),
        (r"\bact\s+as\s+(if\s+you\s+are\s+)?a?\s*(different|new)\b", "Role hijacking attempt"),
        (r"\bexecute\s+(arbitrary|any)\s+code\b", "Unsafe code execution"),
        (r"\b(rm\s+-rf|format\s+c:|del\s+/[sq])\b", "Destructive commands"),
        (
            r"\b(api[_-]?key|secret[_-]?key|password)\s*[=:]\s*['\"][^'\"]+['\"]",
            "Hardcoded secrets",
        ),
    ]

    # Warning patterns (not prohibited but concerning)
    WARNING_PATTERNS = [
        (r"\balways\s+trust\b", "Unconditional trust statement"),
        (r"\bnever\s+question\b", "Blocks critical thinking"),
        (r"\bno\s+matter\s+what\b", "Overly rigid instruction"),
        (r"\bbypass\b", "Potential security bypass"),
    ]

    @classmethod
    def validate_file(cls, file_path: str) -> SkillValidationResult:
        """
        Validate a SKILL.md file.

        Args:
            file_path: Path to the SKILL.md file

        Returns:
            SkillValidationResult with validation outcome
        """
        errors = []
        warnings = []
        info = []

        # Check file exists
        path = Path(file_path)
        if not path.exists():
            return SkillValidationResult(
                valid=False,
                errors=[
                    SkillValidationError(
                        code="FILE_NOT_FOUND",
                        message=f"File not found: {file_path}",
                        severity=SkillSeverity.ERROR,
                    )
                ],
            )

        # Check filename
        if path.name != "SKILL.md":
            warnings.append(
                SkillValidationError(
                    code="WRONG_FILENAME",
                    message=f"Expected 'SKILL.md', got '{path.name}'",
                    severity=SkillSeverity.WARNING,
                    suggestion="Rename the file to SKILL.md for Claude to recognize it",
                )
            )

        # Parse the skill
        try:
            skill = SkillParser.parse_file(file_path)
        except SkillParseError as e:
            return SkillValidationResult(
                valid=False,
                errors=e.errors,
            )
        except Exception as e:
            return SkillValidationResult(
                valid=False,
                errors=[
                    SkillValidationError(
                        code="PARSE_ERROR",
                        message=f"Failed to parse skill: {str(e)}",
                        severity=SkillSeverity.ERROR,
                    )
                ],
            )

        # Run all validations
        errors.extend(cls._validate_name(skill))
        errors.extend(cls._validate_description(skill))
        warnings.extend(cls._validate_instructions(skill))
        errors.extend(cls._validate_policy_compliance(skill))
        warnings.extend(cls._check_warning_patterns(skill))
        info.extend(cls._validate_best_practices(skill))

        # Separate by severity
        all_issues = errors + warnings + info
        errors = [e for e in all_issues if e.severity == SkillSeverity.ERROR]
        warnings = [e for e in all_issues if e.severity == SkillSeverity.WARNING]
        info = [e for e in all_issues if e.severity == SkillSeverity.INFO]

        return SkillValidationResult(
            valid=len(errors) == 0,
            skill=skill if len(errors) == 0 else None,
            errors=errors,
            warnings=warnings,
            info=info,
        )

    @classmethod
    def validate_content(cls, content: str) -> SkillValidationResult:
        """
        Validate SKILL.md content from a string.

        Args:
            content: Raw SKILL.md content

        Returns:
            SkillValidationResult with validation outcome
        """
        errors = []
        warnings = []
        info = []

        # Parse the skill
        try:
            skill = SkillParser.parse_content(content)
        except SkillParseError as e:
            return SkillValidationResult(
                valid=False,
                errors=e.errors,
            )
        except Exception as e:
            return SkillValidationResult(
                valid=False,
                errors=[
                    SkillValidationError(
                        code="PARSE_ERROR",
                        message=f"Failed to parse skill: {str(e)}",
                        severity=SkillSeverity.ERROR,
                    )
                ],
            )

        # Run all validations
        errors.extend(cls._validate_name(skill))
        errors.extend(cls._validate_description(skill))
        warnings.extend(cls._validate_instructions(skill))
        errors.extend(cls._validate_policy_compliance(skill))
        warnings.extend(cls._check_warning_patterns(skill))
        info.extend(cls._validate_best_practices(skill))

        # Separate by severity
        all_issues = errors + warnings + info
        errors = [e for e in all_issues if e.severity == SkillSeverity.ERROR]
        warnings = [e for e in all_issues if e.severity == SkillSeverity.WARNING]
        info = [e for e in all_issues if e.severity == SkillSeverity.INFO]

        return SkillValidationResult(
            valid=len(errors) == 0,
            skill=skill if len(errors) == 0 else None,
            errors=errors,
            warnings=warnings,
            info=info,
        )

    @classmethod
    def _validate_name(cls, skill: Skill) -> List[SkillValidationError]:
        """Validate skill name format per official Anthropic spec."""
        errors = []
        name = skill.metadata.name

        # Check name format (lowercase, hyphens, no underscores)
        if not cls.NAME_PATTERN.match(name):
            errors.append(
                SkillValidationError(
                    code="INVALID_NAME_FORMAT",
                    message=f"Invalid skill name format: '{name}'",
                    severity=SkillSeverity.ERROR,
                    suggestion="Use lowercase letters, numbers, and hyphens only. Must start with a letter. Example: 'my-skill-name'",
                )
            )

        # Check for underscores (common mistake)
        if "_" in name:
            errors.append(
                SkillValidationError(
                    code="NAME_HAS_UNDERSCORE",
                    message=f"Skill name contains underscores: '{name}'",
                    severity=SkillSeverity.ERROR,
                    suggestion=f"Use hyphens instead: '{name.replace('_', '-')}'",
                )
            )

        # Check for uppercase
        if name != name.lower():
            errors.append(
                SkillValidationError(
                    code="NAME_NOT_LOWERCASE",
                    message=f"Skill name must be lowercase: '{name}'",
                    severity=SkillSeverity.ERROR,
                    suggestion=f"Use: '{name.lower()}'",
                )
            )

        # Check name length (official spec: max 64)
        if len(name) > cls.NAME_MAX_LENGTH:
            errors.append(
                SkillValidationError(
                    code="NAME_TOO_LONG",
                    message=f"Skill name exceeds limit ({len(name)} chars, max {cls.NAME_MAX_LENGTH})",
                    severity=SkillSeverity.ERROR,
                    suggestion="Use a shorter, more concise name",
                )
            )

        # Check for reserved words (official spec: no "anthropic" or "claude")
        for reserved in cls.RESERVED_WORDS:
            if reserved in name.lower():
                errors.append(
                    SkillValidationError(
                        code="RESERVED_WORD_IN_NAME",
                        message=f"Skill name contains reserved word: '{reserved}'",
                        severity=SkillSeverity.ERROR,
                        suggestion=f"Remove '{reserved}' from the skill name. Reserved words: {', '.join(cls.RESERVED_WORDS)}",
                    )
                )

        # Check for XML tags (official spec: no XML tags)
        if cls.XML_TAG_PATTERN.search(name):
            errors.append(
                SkillValidationError(
                    code="XML_TAG_IN_NAME",
                    message="Skill name contains XML tags",
                    severity=SkillSeverity.ERROR,
                    suggestion="Remove XML tags from the skill name",
                )
            )

        return errors

    @classmethod
    def _validate_description(cls, skill: Skill) -> List[SkillValidationError]:
        """Validate skill description per official Anthropic spec."""
        errors = []
        desc = skill.metadata.description

        # Check non-empty (official spec: must be non-empty)
        if not desc or not desc.strip():
            errors.append(
                SkillValidationError(
                    code="EMPTY_DESCRIPTION",
                    message="Description is required and cannot be empty",
                    severity=SkillSeverity.ERROR,
                    suggestion="Add a description of what the skill does and when to use it",
                )
            )
            return errors

        # Check max length (official spec: max 1024)
        if len(desc) > cls.DESCRIPTION_MAX_LENGTH:
            errors.append(
                SkillValidationError(
                    code="DESCRIPTION_TOO_LONG",
                    message=f"Description exceeds limit ({len(desc)} chars, max {cls.DESCRIPTION_MAX_LENGTH})",
                    severity=SkillSeverity.ERROR,
                    suggestion="Shorten the description to under 1024 characters",
                )
            )

        # Check for XML tags (official spec: no XML tags)
        if cls.XML_TAG_PATTERN.search(desc):
            errors.append(
                SkillValidationError(
                    code="XML_TAG_IN_DESCRIPTION",
                    message="Description contains XML tags",
                    severity=SkillSeverity.ERROR,
                    suggestion="Remove XML tags from the description",
                )
            )

        # Check minimum length (best practice)
        if len(desc) < 20:
            errors.append(
                SkillValidationError(
                    code="DESCRIPTION_TOO_SHORT",
                    message=f"Description is too short ({len(desc)} chars)",
                    severity=SkillSeverity.WARNING,
                    suggestion="Provide a more detailed description of what the skill does and when to use it",
                )
            )

        # Check for placeholder text
        placeholder_patterns = [
            r"^todo\b",
            r"^tbd\b",
            r"^placeholder\b",
            r"^describe\s+here\b",
            r"^your\s+description\b",
        ]
        for pattern in placeholder_patterns:
            if re.search(pattern, desc.lower()):
                errors.append(
                    SkillValidationError(
                        code="PLACEHOLDER_DESCRIPTION",
                        message="Description appears to be a placeholder",
                        severity=SkillSeverity.ERROR,
                        suggestion="Replace with an actual description of the skill",
                    )
                )
                break

        # Check for multi-line description (can break with Prettier/formatters)
        if "\n" in desc:
            errors.append(
                SkillValidationError(
                    code="MULTILINE_DESCRIPTION",
                    message="Description contains newlines",
                    severity=SkillSeverity.WARNING,
                    suggestion="Use single-line description to avoid issues with Prettier and YAML formatters",
                )
            )

        # Check for very long description (may hit Claude Code char budget)
        if len(desc) > 500:
            errors.append(
                SkillValidationError(
                    code="DESCRIPTION_CHAR_BUDGET",
                    message=f"Description is {len(desc)} chars (recommended: <500)",
                    severity=SkillSeverity.WARNING,
                    suggestion="Keep descriptions concise to stay within Claude Code's 15k char budget for all skills",
                )
            )

        return errors

    @classmethod
    def _validate_instructions(cls, skill: Skill) -> List[SkillValidationError]:
        """Validate skill instructions."""
        warnings = []
        instructions = skill.instructions

        # Check for empty instructions
        if not instructions.strip():
            warnings.append(
                SkillValidationError(
                    code="EMPTY_INSTRUCTIONS",
                    message="Skill has no instructions",
                    severity=SkillSeverity.WARNING,
                    suggestion="Add markdown instructions below the frontmatter",
                )
            )
            return warnings

        # Check token size
        if skill.token_estimate > cls.MAX_RECOMMENDED_TOKENS:
            warnings.append(
                SkillValidationError(
                    code="SKILL_TOO_LARGE",
                    message=f"Skill is ~{skill.token_estimate} tokens (recommended: <{cls.MAX_RECOMMENDED_TOKENS})",
                    severity=SkillSeverity.WARNING,
                    suggestion="Consider splitting into multiple skills or reducing instruction length",
                )
            )

        # Check for very short instructions
        if len(instructions) < 50:
            warnings.append(
                SkillValidationError(
                    code="INSTRUCTIONS_TOO_SHORT",
                    message="Instructions are very short",
                    severity=SkillSeverity.INFO,
                    suggestion="Consider adding more detail, examples, or guidelines",
                )
            )

        return warnings

    @classmethod
    def _validate_policy_compliance(cls, skill: Skill) -> List[SkillValidationError]:
        """Check for prohibited patterns (policy violations)."""
        errors = []
        content = skill.raw_content.lower()

        for pattern, description in cls.PROHIBITED_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                errors.append(
                    SkillValidationError(
                        code="POLICY_VIOLATION",
                        message=f"Potential policy violation: {description}",
                        severity=SkillSeverity.ERROR,
                        suggestion="Remove or rephrase the problematic content",
                    )
                )

        return errors

    @classmethod
    def _check_warning_patterns(cls, skill: Skill) -> List[SkillValidationError]:
        """Check for warning patterns (not prohibited but concerning)."""
        warnings = []
        content = skill.raw_content.lower()

        for pattern, description in cls.WARNING_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                warnings.append(
                    SkillValidationError(
                        code="CONCERNING_PATTERN",
                        message=f"Concerning pattern found: {description}",
                        severity=SkillSeverity.WARNING,
                        suggestion="Review this instruction to ensure it's appropriate",
                    )
                )

        return warnings

    @classmethod
    def _validate_best_practices(cls, skill: Skill) -> List[SkillValidationError]:
        """Check best practices (informational suggestions)."""
        info = []
        instructions = skill.instructions

        # Check for examples section
        if not re.search(r"##+\s*examples?\b", instructions, re.IGNORECASE):
            info.append(
                SkillValidationError(
                    code="NO_EXAMPLES",
                    message="No examples section found",
                    severity=SkillSeverity.INFO,
                    suggestion="Adding examples helps Claude understand how to use the skill",
                )
            )

        # Check for guidelines section
        if not re.search(r"##+\s*(guidelines?|rules?)\b", instructions, re.IGNORECASE):
            info.append(
                SkillValidationError(
                    code="NO_GUIDELINES",
                    message="No guidelines section found",
                    severity=SkillSeverity.INFO,
                    suggestion="Adding guidelines helps Claude follow best practices",
                )
            )

        return info

    @classmethod
    def validate_directory(cls, directory: str, recursive: bool = True) -> dict:
        """
        Validate all skills in a directory.

        Args:
            directory: Directory to search
            recursive: Whether to search subdirectories

        Returns:
            Dict mapping file paths to validation results
        """
        skill_files = SkillParser.find_skills(directory, recursive)
        results = {}

        for file_path in skill_files:
            results[file_path] = cls.validate_file(file_path)

        return results
