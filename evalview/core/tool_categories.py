"""Tool categories for flexible tool matching.

This module allows tests to specify expected behavior by category (e.g., "file_read")
instead of exact tool names. This prevents false failures when agents use different
but equivalent tools (e.g., `bash cat` vs `text_editor` vs `read_file`).

Categories can be customized per-project via config.yaml.
"""

from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


# Default tool categories - covers common AI agent tools
DEFAULT_CATEGORIES: Dict[str, List[str]] = {
    # File reading operations
    "file_read": [
        "bash",  # cat, head, tail, less
        "text_editor",
        "read_file",
        "read",
        "view",
        "cat",
        "view_file",
        "str_replace_editor",  # Anthropic's editor tool
    ],
    # File writing operations
    "file_write": [
        "bash",  # echo >, cat <<
        "text_editor",
        "write_file",
        "write",
        "edit_file",
        "create_file",
        "str_replace_editor",
    ],
    # File/directory listing
    "file_list": [
        "bash",  # ls, find
        "list_directory",
        "list_dir",
        "ls",
        "find",
        "directory_tree",
        "analyze",  # Goose analyze tool does exploration
    ],
    # Code/text search
    "search": [
        "bash",  # grep, rg, ag
        "grep",
        "ripgrep",
        "rg",
        "search",
        "search_files",
        "code_search",
        "find_in_files",
        "analyze",
    ],
    # Shell/command execution
    "shell": [
        "bash",
        "shell",
        "terminal",
        "execute",
        "run_command",
        "exec",
        "developer",  # Goose developer extension
    ],
    # Web/HTTP operations
    "web": [
        "web_search",
        "browse",
        "fetch_url",
        "http_request",
        "curl",
        "wget",
    ],
    # Git operations
    "git": [
        "bash",  # git commands
        "git",
        "git_commit",
        "git_push",
        "git_pull",
        "git_status",
        "github",
    ],
    # Python execution
    "python": [
        "bash",  # python -c
        "python",
        "python_repl",
        "execute_python",
        "ipython",
        "jupyter",
    ],
}


class ToolCategoryMatcher:
    """Matches tools against categories for flexible evaluation."""

    def __init__(self, custom_categories: Optional[Dict[str, List[str]]] = None):
        """
        Initialize with optional custom categories.

        Args:
            custom_categories: Project-specific categories that extend/override defaults
        """
        # Start with defaults
        self.categories = dict(DEFAULT_CATEGORIES)

        # Merge in custom categories
        if custom_categories:
            for category, tools in custom_categories.items():
                if category in self.categories:
                    # Extend existing category
                    existing = set(self.categories[category])
                    existing.update(tools)
                    self.categories[category] = list(existing)
                else:
                    # Add new category
                    self.categories[category] = tools

        # Build reverse lookup: tool -> categories it belongs to
        self._tool_to_categories: Dict[str, Set[str]] = {}
        for category, tools in self.categories.items():
            for tool in tools:
                if tool not in self._tool_to_categories:
                    self._tool_to_categories[tool] = set()
                self._tool_to_categories[tool].add(category)

    def get_tools_in_category(self, category: str) -> List[str]:
        """Get all tools that belong to a category."""
        return self.categories.get(category, [])

    def get_categories_for_tool(self, tool: str) -> Set[str]:
        """Get all categories a tool belongs to."""
        return self._tool_to_categories.get(tool, set())

    def tool_matches_category(self, tool: str, category: str) -> bool:
        """Check if a tool belongs to a category."""
        return tool in self.categories.get(category, [])

    def any_tool_matches_category(self, tools: List[str], category: str) -> bool:
        """Check if any tool in the list matches the category."""
        category_tools = set(self.categories.get(category, []))
        return bool(set(tools) & category_tools)

    def evaluate_categories(
        self, expected_categories: List[str], actual_tools: List[str]
    ) -> Dict[str, bool]:
        """
        Evaluate whether actual tools satisfy expected categories.

        Args:
            expected_categories: List of category names that should be satisfied
            actual_tools: List of tool names that were actually called

        Returns:
            Dict mapping category -> whether it was satisfied
        """
        results = {}
        actual_set = set(actual_tools)

        for category in expected_categories:
            category_tools = set(self.categories.get(category, []))
            # Category is satisfied if any actual tool is in the category
            results[category] = bool(actual_set & category_tools)

        return results

    def get_matching_tools(self, category: str, actual_tools: List[str]) -> List[str]:
        """Get which actual tools matched a category."""
        category_tools = set(self.categories.get(category, []))
        return [t for t in actual_tools if t in category_tools]


# Singleton instance with default categories
_default_matcher: Optional[ToolCategoryMatcher] = None


def get_default_matcher() -> ToolCategoryMatcher:
    """Get the default category matcher (singleton)."""
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = ToolCategoryMatcher()
    return _default_matcher


def load_categories_from_config(config: Dict) -> ToolCategoryMatcher:
    """Load categories from a config dict (e.g., from config.yaml)."""
    custom = config.get("tool_categories", {})
    return ToolCategoryMatcher(custom)
