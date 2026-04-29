"""GenerationHelpersMixin — workspace utilities for `AgentTestGenerator`.

Inherits-into AgentTestGenerator. Three families of helpers grouped here
because they're all small, stateless-ish, and were previously crowding
the main orchestration class:

  - seed-prompt harvesting: `_workspace_seed_prompts`,
    `_existing_test_queries`, `_project_doc_prompts`,
    `_extract_prompts_from_text`, `_looks_like_prompt`
  - report deltas: `_load_previous_report`, `_compute_report_delta`
  - dangerous-tool detection: `_is_dangerous_tool_name`,
    `_is_dangerous_tool_schema`
  - YAML emission and assertion extraction: `_write_test_yaml`,
    `_extract_example_queries`, `_extract_stable_phrases`

Reads `self.project_root` from the parent.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import yaml  # type: ignore[import-untyped]

from evalview.core.types import TestCase
from evalview.test_generation_constants import (
    _BACKTICK_PROMPT,
    _DANGEROUS_TOOL_KEYWORDS,
    _FRAGMENT_ENDINGS,
    _PROJECT_CONTEXT_FILES,
    _PROMPT_LIKE_LINE,
    _QUOTED_PROMPT,
    _SIDE_EFFECT_TOOL_KEYWORDS,
)
from evalview.test_generation_discovery import _normalize_name
from evalview.test_generation_types import (
    PromptCandidate,
    _normalize_text_for_comparison,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:

    class _ParentProtocol:
        """Forward declarations for mypy. Resolved at runtime via MRO."""

        project_root: Path

        def _prompt_is_allowed(self, prompt: str) -> bool: ...

    _MixinBase = _ParentProtocol
else:
    _MixinBase = object


class GenerationHelpersMixin(_MixinBase):
    """Workspace, report-delta, dangerous-tool, and YAML-emission helpers."""


    def _workspace_seed_prompts(self) -> List[PromptCandidate]:
        prompts: List[PromptCandidate] = []
        seen: Set[str] = set()

        for candidate in self._existing_test_queries():
            normalized = _normalize_text_for_comparison(candidate.text)
            if normalized and normalized not in seen and self._prompt_is_allowed(candidate.text):
                seen.add(normalized)
                prompts.append(candidate)

        for candidate in self._project_doc_prompts():
            normalized = _normalize_text_for_comparison(candidate.text)
            if normalized and normalized not in seen and self._prompt_is_allowed(candidate.text):
                seen.add(normalized)
                prompts.append(candidate)

        return prompts[:20]

    def _existing_test_queries(self) -> List[PromptCandidate]:
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            return []

        queries: List[PromptCandidate] = []
        for yaml_path in sorted(tests_dir.rglob("*.yaml")):
            if "generated" in yaml_path.parts:
                continue
            try:
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            query = (((data.get("input") or {}).get("query")) or "").strip()
            if query and self._looks_like_prompt(query):
                queries.append(PromptCandidate(query, "existing_tests"))
        return queries

    def _project_doc_prompts(self) -> List[PromptCandidate]:
        prompts: List[PromptCandidate] = []
        for relative_name in _PROJECT_CONTEXT_FILES:
            path = self.project_root / relative_name
            if not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            prompts.extend(self._extract_prompts_from_text(text, source=f"project_docs:{relative_name}"))
        return prompts

    def _extract_prompts_from_text(self, text: str, source: str) -> List[PromptCandidate]:
        prompts: List[PromptCandidate] = []

        for match in _BACKTICK_PROMPT.findall(text):
            if self._looks_like_prompt(match):
                prompts.append(PromptCandidate(match.strip(), source))

        for match in _QUOTED_PROMPT.findall(text):
            if self._looks_like_prompt(match):
                prompts.append(PromptCandidate(match.strip(), source))

        for line in text.splitlines():
            stripped = line.strip().strip("|").strip()
            if not stripped:
                continue
            if "|" in stripped:
                first_cell = stripped.split("|", 1)[0].strip().strip("`")
                if self._looks_like_prompt(first_cell):
                    prompts.append(PromptCandidate(first_cell, source))
            else:
                cleaned = stripped.lstrip("-*0123456789. ").strip().strip("`")
                if self._looks_like_prompt(cleaned):
                    prompts.append(PromptCandidate(cleaned, source))

        return prompts

    def _looks_like_prompt(self, value: str) -> bool:
        candidate = " ".join(value.split()).strip()
        if not candidate or len(candidate) < 8 or len(candidate) > 180:
            return False
        lowered = candidate.lower()
        if candidate.startswith(("http://", "https://")):
            return False
        if lowered.startswith(("evalview ", "python ", "pip ", "make ", "uvicorn ", "cd ")):
            return False
        if any(token in candidate for token in ("{", "}", "$(", "&&", "||", "::")):
            return False
        if "/" in candidate and " " not in candidate:
            return False
        return bool(
            "?" in candidate
            or any(word in lowered for word in ("show me", "what ", "which ", "how ", "find ", "search ", "top ", "recent "))
            or _PROMPT_LIKE_LINE.match(candidate)
        )

    def _load_previous_report(self, report_path: Path) -> Optional[Dict[str, Any]]:
        if not report_path.exists():
            return None
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _compute_report_delta(
        self,
        previous_report: Dict[str, Any],
        current_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        previous_signatures = set((previous_report.get("behavior_signatures") or {}).keys())
        current_signatures = set((current_report.get("behavior_signatures") or {}).keys())
        previous_tools = set((previous_report.get("tools_seen") or {}).keys())
        current_tools = set((current_report.get("tools_seen") or {}).keys())
        previous_gaps = set(previous_report.get("gaps") or [])
        current_gaps = set(current_report.get("gaps") or [])

        return {
            "new_signatures": sorted(current_signatures - previous_signatures),
            "resolved_signatures": sorted(previous_signatures - current_signatures),
            "new_tools": sorted(current_tools - previous_tools),
            "resolved_gaps": sorted(previous_gaps - current_gaps),
            "new_gaps": sorted(current_gaps - previous_gaps),
            "tests_generated_delta": current_report.get("tests_generated", 0)
            - previous_report.get("tests_generated", 0),
        }

    def _is_dangerous_tool_name(self, tool_name: str) -> bool:
        normalized = _normalize_name(tool_name)
        return any(keyword in normalized for keyword in _DANGEROUS_TOOL_KEYWORDS + _SIDE_EFFECT_TOOL_KEYWORDS)

    def _is_dangerous_tool_schema(self, tool: Dict[str, Any]) -> bool:
        text_parts = [
            tool.get("name", ""),
            tool.get("description", ""),
            " ".join((tool.get("inputSchema") or {}).get("properties", {}).keys()),
        ]
        combined = _normalize_name(" ".join(part for part in text_parts if part))
        return any(keyword in combined for keyword in _DANGEROUS_TOOL_KEYWORDS + _SIDE_EFFECT_TOOL_KEYWORDS)

    def _write_test_yaml(self, test: TestCase, path: Path) -> None:
        payload = test.model_dump(exclude_none=True)
        header = (
            "# Auto-generated by: evalview generate\n"
            "# Review assertions before running evalview snapshot.\n"
        )
        path.write_text(
            header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

    def _extract_example_queries(self, text: str) -> List[str]:
        quoted = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]{20,100})["\u201c\u201d]', text)
        bulleted = re.findall(r'[-•]\s+"?([A-Z][^"\n]{20,100})"?\s*$', text, re.MULTILINE)
        candidates = quoted + bulleted
        valid = []
        for query in candidates:
            query = query.strip().rstrip(",.")
            words = query.split()
            if len(words) < 3:
                continue
            if query.lower().endswith(_FRAGMENT_ENDINGS):
                continue
            if "?" in query or query[0].isupper():
                valid.append(query)
        return valid[:4]

    def _extract_stable_phrases(
        self,
        text: str,
        behavior_class: str,
        has_tools: bool,
        max_phrases: int = 3,
    ) -> List[str]:
        if not text:
            return []

        # For tool-using flows, don't extract wording assertions at all —
        # the tool trajectory IS the assertion.  Brittle phrases like
        # "monitoring product pain points" cause false failures on acceptable
        # rewrites.  Let the LLM-as-judge handle output quality instead.
        if has_tools or behavior_class in {"multi_turn", "tool_path"}:
            return []

        # For non-tool flows (direct_answer, clarification), extract minimal
        # keyword anchors — only proper nouns / entity names, not phrases.
        entity_like = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?\b", text)
        seen: Set[str] = set()
        unique: List[str] = []
        for phrase in entity_like:
            key = phrase.lower()
            if key in seen or len(phrase) < 3:
                continue
            if re.match(r"^[\d.,$%/:-]+$", phrase):
                continue
            seen.add(key)
            unique.append(phrase)
        return unique[:max_phrases]

