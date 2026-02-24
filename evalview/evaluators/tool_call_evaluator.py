"""Tool call accuracy evaluator."""

import re
from typing import List, Set, Tuple, Optional, Dict
from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    ToolEvaluation,
    ForbiddenToolEvaluation,
    CategoryResult,
    ReasonCode,
)
from evalview.core.tool_categories import ToolCategoryMatcher, get_default_matcher


def _normalize_tool_name(name: str) -> str:
    """Normalize tool name for comparison (lowercase, remove underscores/dashes)."""
    return re.sub(r"[-_]", "", name.lower())


def _is_case_mismatch(expected: str, actual: str) -> bool:
    """Check if two tool names differ only by case or underscore/camelCase convention."""
    return _normalize_tool_name(expected) == _normalize_tool_name(actual)


def _find_similar_tools(
    missing: List[str], unexpected: List[str]
) -> List[Tuple[str, str]]:
    """Find pairs of missing/unexpected tools that are likely the same (case mismatch)."""
    similar_pairs = []
    for m in missing:
        for u in unexpected:
            if _is_case_mismatch(m, u):
                similar_pairs.append((m, u))
    return similar_pairs


class ToolCallEvaluator:
    """Evaluates whether the agent called the expected tools."""

    def __init__(self, category_matcher: Optional[ToolCategoryMatcher] = None):
        """
        Initialize with optional custom category matcher.

        Args:
            category_matcher: Custom matcher, or None to use defaults
        """
        self.category_matcher = category_matcher or get_default_matcher()

    def evaluate(self, test_case: TestCase, trace: ExecutionTrace) -> ToolEvaluation:
        """
        Evaluate tool call accuracy.

        Supports two modes:
        1. Exact tool matching: expected.tools = ["bash", "text_editor"]
        2. Category matching: expected.tool_categories = ["file_read", "search"]

        Category matching is more flexible - it passes if ANY tool in the category was used.

        Args:
            test_case: Test case with expected tools or tool_categories
            trace: Execution trace with actual tool calls

        Returns:
            ToolEvaluation with accuracy metrics and helpful hints
        """
        expected_tools = set(test_case.expected.tools or [])
        expected_categories = test_case.expected.tool_categories or []
        actual_tools = [step.tool_name for step in trace.steps]

        correct: List[str] = []
        missing: List[str] = []
        unexpected: List[str] = []
        category_results: List[CategoryResult] = []

        # --- Exact tool matching ---
        for tool in expected_tools:
            if tool in actual_tools:
                correct.append(tool)
            else:
                missing.append(tool)

        for tool in actual_tools:
            if tool not in expected_tools:
                unexpected.append(tool)

        # --- Category matching ---
        categories_satisfied = 0
        for category in expected_categories:
            matched = self.category_matcher.get_matching_tools(category, actual_tools)
            satisfied = len(matched) > 0
            if satisfied:
                categories_satisfied += 1
            category_results.append(
                CategoryResult(
                    category=category,
                    satisfied=satisfied,
                    matched_tools=matched,
                )
            )

        # --- Calculate accuracy ---
        # Combine exact tools and categories for accuracy calculation
        total_expected = len(expected_tools) + len(expected_categories)
        total_correct = len(correct) + categories_satisfied

        if total_expected == 0:
            accuracy = 1.0  # No expectations = 100% accuracy
        else:
            accuracy = total_correct / total_expected

        # Generate helpful hints for mismatches
        hints = self._generate_hints(
            missing, unexpected, expected_tools, set(actual_tools),
            expected_categories, category_results
        )

        # Generate structured reason codes
        reason_codes = self._generate_reason_codes(
            missing, unexpected, expected_tools, set(actual_tools),
            expected_categories, category_results
        )

        return ToolEvaluation(
            accuracy=accuracy,
            correct=correct,
            missing=missing,
            unexpected=unexpected,
            hints=hints,
            category_results=category_results,
            categories_satisfied=categories_satisfied,
            categories_total=len(expected_categories),
            reason_codes=reason_codes,
        )

    def evaluate_forbidden(
        self, test_case: TestCase, trace: ExecutionTrace
    ) -> Optional[ForbiddenToolEvaluation]:
        """Check whether any forbidden tools were invoked.

        Returns None when the test case defines no forbidden_tools contract,
        so callers can distinguish "not configured" from "passed with no violations".

        Args:
            test_case: Test case that may declare forbidden_tools.
            trace: Execution trace with the actual tool calls.

        Returns:
            ForbiddenToolEvaluation if forbidden_tools is configured, else None.
        """
        forbidden = test_case.expected.forbidden_tools
        if not forbidden:
            return None

        forbidden_set: Set[str] = set(forbidden)
        # Precompute the normalised forbidden set once so the per-step check
        # is O(1) rather than O(|forbidden|). Normalisation strips casing and
        # word separators so "EditFile" catches "edit_file" and "edit-file".
        normalized_forbidden: Set[str] = {_normalize_tool_name(f) for f in forbidden_set}
        violations = [
            step.tool_name
            for step in trace.steps
            # Exact match first (fast path), then normalised comparison.
            if step.tool_name in forbidden_set
            or _normalize_tool_name(step.tool_name) in normalized_forbidden
        ]
        # Deduplicate while preserving first-seen order.
        seen: Set[str] = set()
        unique_violations: List[str] = []
        for v in violations:
            if v not in seen:
                seen.add(v)
                unique_violations.append(v)

        return ForbiddenToolEvaluation(violations=unique_violations, passed=len(unique_violations) == 0)

    def _generate_hints(
        self,
        missing: List[str],
        unexpected: List[str],
        expected: Set[str],
        actual: Set[str],
        expected_categories: List[str] = None,
        category_results: List[CategoryResult] = None,
    ) -> List[str]:
        """Generate helpful hints for debugging tool mismatches."""
        hints: List[str] = []
        expected_categories = expected_categories or []
        category_results = category_results or []

        # --- Category hints ---
        unsatisfied_categories = [
            cr for cr in category_results if not cr.satisfied
        ]
        if unsatisfied_categories:
            for cr in unsatisfied_categories:
                category_tools = self.category_matcher.get_tools_in_category(cr.category)
                hints.append(
                    f"Category '{cr.category}' not satisfied. "
                    f"Expected one of: {category_tools[:5]}{'...' if len(category_tools) > 5 else ''}"
                )
            hints.append(
                "Tip: Use tool_categories instead of exact tools for flexible matching. "
                "Categories: file_read, file_write, file_list, search, shell, git, web, python"
            )

        # --- Exact tool hints ---
        if not missing and not unexpected:
            return hints

        # Find case/naming convention mismatches
        similar_pairs = _find_similar_tools(missing, unexpected)
        for expected_name, actual_name in similar_pairs:
            hints.append(
                f"Possible naming mismatch: expected '{expected_name}' but agent called '{actual_name}'. "
                f"Update your test case to use '{actual_name}' if this is correct."
            )

        # Suggest categories for common mismatches
        if missing and unexpected and not similar_pairs:
            # Check if unexpected tools could satisfy categories for missing tools
            for m in missing:
                for u in unexpected:
                    m_cats = self.category_matcher.get_categories_for_tool(m)
                    u_cats = self.category_matcher.get_categories_for_tool(u)
                    common = m_cats & u_cats
                    if common:
                        hints.append(
                            f"'{m}' and '{u}' are both in category '{list(common)[0]}'. "
                            f"Consider using tool_categories: ['{list(common)[0]}'] instead of exact tools."
                        )
                        break

        # General hints
        if similar_pairs:
            hints.append(
                "Tip: Tool names are case-sensitive. Check for snake_case vs camelCase differences."
            )
        elif missing and unexpected:
            hints.append(
                "The agent called different tools than expected. "
                "Consider using tool_categories for flexible matching."
            )
        elif missing and not unexpected:
            hints.append(
                "The agent did not call the expected tools. "
                "Check that your agent has access to these tools and the query triggers their use."
            )
        elif unexpected and not missing:
            hints.append(
                "The agent called additional tools beyond what was expected. "
                "This may be intentional - consider adding them to expected_tools in your test case."
            )

        return hints

    def _generate_reason_codes(
        self,
        missing: List[str],
        unexpected: List[str],
        expected: Set[str],
        actual: Set[str],
        expected_categories: List[str] = None,
        category_results: List[CategoryResult] = None,
    ) -> List[ReasonCode]:
        """Generate structured reason codes for tool mismatches."""
        reason_codes: List[ReasonCode] = []
        expected_categories = expected_categories or []
        category_results = category_results or []

        # --- Category violations ---
        unsatisfied_categories = [
            cr for cr in category_results if not cr.satisfied
        ]
        for cr in unsatisfied_categories:
            category_tools = self.category_matcher.get_tools_in_category(cr.category)
            reason_codes.append(ReasonCode(
                code="CATEGORY_NOT_SATISFIED",
                severity="error",
                message=f"Category '{cr.category}' not satisfied",
                context={
                    "category": cr.category,
                    "expected_tools": category_tools[:10],
                    "actual_tools": list(actual)
                },
                remediation=f"Ensure your agent has access to tools in the '{cr.category}' category and the query triggers their use."
            ))

        # --- Missing tools ---
        similar_pairs = _find_similar_tools(missing, unexpected)
        similar_dict = dict(similar_pairs)

        for tool in missing:
            if tool in similar_dict:
                # Naming mismatch
                actual_name = similar_dict[tool]
                reason_codes.append(ReasonCode(
                    code="TOOL_NAME_MISMATCH",
                    severity="warning",
                    message=f"Tool name mismatch: expected '{tool}' but got '{actual_name}'",
                    context={
                        "expected": tool,
                        "actual": actual_name
                    },
                    remediation=f"Update your test case to use '{actual_name}' if this is the correct tool name."
                ))
            else:
                # Tool not called
                reason_codes.append(ReasonCode(
                    code="TOOL_MISSING",
                    severity="error",
                    message=f"Expected tool '{tool}' was not called",
                    context={
                        "expected_tool": tool,
                        "actual_tools": list(actual)
                    },
                    remediation=f"Ensure your agent has access to '{tool}' and the query triggers its use."
                ))

        # --- Unexpected tools ---
        for tool in unexpected:
            if tool not in similar_dict.values():  # Skip if it's a naming mismatch we already handled
                reason_codes.append(ReasonCode(
                    code="TOOL_UNEXPECTED",
                    severity="info",
                    message=f"Agent called unexpected tool '{tool}'",
                    context={
                        "unexpected_tool": tool,
                        "expected_tools": list(expected)
                    },
                    remediation=f"If '{tool}' is correct, add it to expected_tools in your test case."
                ))

        return reason_codes
