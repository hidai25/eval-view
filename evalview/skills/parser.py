"""SKILL.md file parser."""

import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import yaml  # type: ignore[import-untyped]

from evalview.skills.types import (
    Skill,
    SkillMetadata,
    SkillValidationError,
    SkillSeverity,
)


class SkillParseError(Exception):
    """Raised when a skill file cannot be parsed."""

    def __init__(self, message: str, errors: Optional[list] = None):
        super().__init__(message)
        self.errors = errors or []


class SkillParser:
    """Parser for SKILL.md files.

    SKILL.md files have the format:
    ```
    ---
    name: my-skill
    description: What this skill does
    ---

    # Instructions

    Markdown content here...
    ```
    """

    # Regex to match YAML frontmatter
    FRONTMATTER_PATTERN = re.compile(
        r"^---\s*\n(.*?)\n---\s*\n?",
        re.DOTALL | re.MULTILINE,
    )

    @classmethod
    def parse_file(cls, file_path: str) -> Skill:
        """
        Parse a SKILL.md file.

        Args:
            file_path: Path to the SKILL.md file

        Returns:
            Parsed Skill object

        Raises:
            SkillParseError: If the file cannot be parsed
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {file_path}")

        if not path.is_file():
            raise SkillParseError(f"Path is not a file: {file_path}")

        content = path.read_text(encoding="utf-8")
        return cls.parse_content(content, file_path=str(path.absolute()))

    @classmethod
    def parse_content(cls, content: str, file_path: Optional[str] = None) -> Skill:
        """
        Parse SKILL.md content from a string.

        Args:
            content: Raw SKILL.md content
            file_path: Optional path for error messages

        Returns:
            Parsed Skill object

        Raises:
            SkillParseError: If the content cannot be parsed
        """
        errors = []

        # Check for empty content
        if not content or not content.strip():
            raise SkillParseError(
                "Skill file is empty",
                errors=[
                    SkillValidationError(
                        code="EMPTY_FILE",
                        message="SKILL.md file is empty",
                        severity=SkillSeverity.ERROR,
                    )
                ],
            )

        # Extract frontmatter
        frontmatter_dict, instructions, parse_errors = cls._extract_frontmatter(content)
        errors.extend(parse_errors)

        if errors:
            raise SkillParseError(
                f"Failed to parse skill: {errors[0].message}",
                errors=errors,
            )

        # Parse metadata
        try:
            metadata = SkillMetadata(**frontmatter_dict)
        except Exception as e:
            raise SkillParseError(
                f"Invalid metadata: {str(e)}",
                errors=[
                    SkillValidationError(
                        code="INVALID_METADATA",
                        message=str(e),
                        severity=SkillSeverity.ERROR,
                    )
                ],
            )

        return Skill(
            metadata=metadata,
            instructions=instructions,
            raw_content=content,
            file_path=file_path,
        )

    @classmethod
    def _extract_frontmatter(cls, content: str) -> Tuple[Dict[str, Any], str, list]:
        """
        Extract YAML frontmatter and markdown body from content.

        Returns:
            Tuple of (frontmatter_dict, markdown_body, errors)
        """
        errors = []

        # Check for frontmatter delimiter
        if not content.startswith("---"):
            errors.append(
                SkillValidationError(
                    code="MISSING_FRONTMATTER",
                    message="SKILL.md must start with YAML frontmatter (---)",
                    severity=SkillSeverity.ERROR,
                    line=1,
                    suggestion="Add frontmatter at the start:\n---\nname: my-skill\ndescription: What this skill does\n---",
                )
            )
            return {}, content, errors

        # Extract frontmatter using regex
        match = cls.FRONTMATTER_PATTERN.match(content)
        if not match:
            errors.append(
                SkillValidationError(
                    code="INVALID_FRONTMATTER",
                    message="Could not parse YAML frontmatter. Make sure it's enclosed in --- delimiters.",
                    severity=SkillSeverity.ERROR,
                    line=1,
                    suggestion="Ensure frontmatter is properly formatted:\n---\nname: my-skill\ndescription: What this skill does\n---",
                )
            )
            return {}, content, errors

        frontmatter_yaml = match.group(1)
        markdown_body = content[match.end() :].strip()

        # Parse YAML
        try:
            frontmatter_dict = yaml.safe_load(frontmatter_yaml) or {}
        except yaml.YAMLError as e:
            errors.append(
                SkillValidationError(
                    code="YAML_ERROR",
                    message=f"Invalid YAML in frontmatter: {str(e)}",
                    severity=SkillSeverity.ERROR,
                    suggestion="Check your YAML syntax. Common issues: missing quotes around special characters, incorrect indentation.",
                )
            )
            return {}, markdown_body, errors

        if not isinstance(frontmatter_dict, dict):
            errors.append(
                SkillValidationError(
                    code="INVALID_FRONTMATTER_TYPE",
                    message="Frontmatter must be a YAML dictionary/object",
                    severity=SkillSeverity.ERROR,
                )
            )
            return {}, markdown_body, errors

        return frontmatter_dict, markdown_body, errors

    @classmethod
    def find_skills(cls, directory: str, recursive: bool = True) -> list:
        """
        Find all SKILL.md files in a directory.

        Args:
            directory: Directory to search
            recursive: Whether to search subdirectories

        Returns:
            List of paths to SKILL.md files
        """
        path = Path(directory)
        if not path.exists():
            return []

        pattern = "**/SKILL.md" if recursive else "SKILL.md"
        return [str(p) for p in path.glob(pattern)]
