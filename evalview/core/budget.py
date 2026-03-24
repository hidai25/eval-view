"""Budget tracking with mid-execution circuit breaker."""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple


class BudgetExhausted(Exception):
    """Raised when the budget limit has been reached."""

    def __init__(self, spent: float, limit: float, completed: int, total: int):
        self.spent = spent
        self.limit = limit
        self.completed = completed
        self.total = total
        super().__init__(
            f"Budget exhausted: ${spent:.4f} spent of ${limit:.2f} limit "
            f"after {completed}/{total} tests"
        )


class CostBreakdown:
    """Per-test and per-tool cost tracking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_test: Dict[str, float] = {}
        self.by_tool: Dict[str, float] = {}
        self.by_adapter: Dict[str, float] = {}
        self.test_details: Dict[str, Dict[str, float]] = {}  # test -> {tool: cost}

    def record(
        self,
        test_name: str,
        cost: float,
        adapter: str = "",
        tool_costs: Optional[Dict[str, float]] = None,
    ) -> None:
        with self._lock:
            self.by_test[test_name] = self.by_test.get(test_name, 0) + cost
            if adapter:
                self.by_adapter[adapter] = self.by_adapter.get(adapter, 0) + cost
            if tool_costs:
                if test_name not in self.test_details:
                    self.test_details[test_name] = {}
                for tool, tcost in tool_costs.items():
                    self.by_tool[tool] = self.by_tool.get(tool, 0) + tcost
                    self.test_details[test_name][tool] = (
                        self.test_details[test_name].get(tool, 0) + tcost
                    )

    @property
    def total(self) -> float:
        return sum(self.by_test.values())

    def top_costs(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top N most expensive tests."""
        return sorted(self.by_test.items(), key=lambda x: x[1], reverse=True)[:n]

    def top_tools(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top N most expensive tools."""
        return sorted(self.by_tool.items(), key=lambda x: x[1], reverse=True)[:n]


class BudgetTracker:
    """Thread-safe budget tracker with circuit breaker.

    Usage:
        tracker = BudgetTracker(limit=1.00)
        tracker.record_cost("test-1", 0.05, adapter="http")
        tracker.check_budget(completed=1, total=5)  # raises BudgetExhausted if over
    """

    def __init__(self, limit: Optional[float] = None):
        self._lock = threading.Lock()
        self.limit = limit
        self.spent: float = 0.0
        self.breakdown = CostBreakdown()
        self._halted = False

    @property
    def is_active(self) -> bool:
        return self.limit is not None

    @property
    def remaining(self) -> Optional[float]:
        if self.limit is None:
            return None
        return max(0, self.limit - self.spent)

    @property
    def halted(self) -> bool:
        return self._halted

    def record_cost(
        self,
        test_name: str,
        cost: float,
        adapter: str = "",
        tool_costs: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record cost for a test execution."""
        with self._lock:
            self.spent += cost
            self.breakdown.record(test_name, cost, adapter, tool_costs)

    def check_budget(self, completed: int, total: int) -> None:
        """Check if budget is exceeded. Raises BudgetExhausted if so.

        Call this after each test completes.
        """
        if self.limit is None:
            return

        with self._lock:
            if self.spent >= self.limit:
                self._halted = True
                raise BudgetExhausted(self.spent, self.limit, completed, total)

    def estimate_remaining_cost(self, completed: int, total: int) -> Optional[float]:
        """Estimate total cost if remaining tests cost the same as completed ones."""
        if completed <= 0 or total <= completed:
            return None
        avg_cost = self.spent / completed
        return avg_cost * (total - completed)

    def should_warn(self, completed: int, total: int) -> bool:
        """Return True if projected cost would exceed 80% of budget."""
        if self.limit is None:
            return False
        estimated = self.estimate_remaining_cost(completed, total)
        if estimated is None:
            return False
        projected_total = self.spent + estimated
        return projected_total > self.limit * 0.8
