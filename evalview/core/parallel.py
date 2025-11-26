"""Parallel test execution with concurrency limiting."""

import asyncio
import logging
from typing import List, Callable, Any, Optional, TypeVar
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class TestProgress:
    """Progress tracking for parallel test execution."""

    total: int = 0
    completed: int = 0
    passed: int = 0
    failed: int = 0
    running: List[str] = field(default_factory=list)

    @property
    def pending(self) -> int:
        return self.total - self.completed - len(self.running)


@dataclass
class ParallelResult:
    """Result from parallel test execution."""

    test_name: str
    success: bool
    result: Optional[Any] = None
    exception: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


class ParallelExecutor:
    """Execute tests in parallel with configurable concurrency."""

    def __init__(
        self,
        max_workers: int = 8,
        on_start: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str, bool, Any], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum concurrent test executions
            on_start: Callback when test starts (test_name)
            on_complete: Callback when test completes (test_name, passed, result)
            on_error: Callback when test errors (test_name, exception)
        """
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.on_start = on_start
        self.on_complete = on_complete
        self.on_error = on_error
        self.progress = TestProgress()

    async def execute_all(
        self,
        test_cases: List[Any],
        execute_fn: Callable[[Any], Any],
    ) -> List[ParallelResult]:
        """
        Execute all tests in parallel with concurrency limiting.

        Args:
            test_cases: List of test cases to execute
            execute_fn: Async function to execute a single test case
                       Should return (passed: bool, result: Any) or raise exception

        Returns:
            List of ParallelResult in same order as test_cases
        """
        self.progress = TestProgress(total=len(test_cases))

        async def run_with_limit(index: int, test_case: Any) -> ParallelResult:
            test_name = getattr(test_case, "name", f"test_{index}")

            async with self.semaphore:
                self.progress.running.append(test_name)

                if self.on_start:
                    self.on_start(test_name)

                start_time = datetime.now()

                try:
                    passed, result = await execute_fn(test_case)
                    end_time = datetime.now()

                    self.progress.completed += 1
                    if passed:
                        self.progress.passed += 1
                    else:
                        self.progress.failed += 1
                    self.progress.running.remove(test_name)

                    if self.on_complete:
                        self.on_complete(test_name, passed, result)

                    return ParallelResult(
                        test_name=test_name,
                        success=True,
                        result=result,
                        start_time=start_time,
                        end_time=end_time,
                    )

                except Exception as e:
                    end_time = datetime.now()

                    self.progress.completed += 1
                    self.progress.failed += 1
                    self.progress.running.remove(test_name)

                    if self.on_error:
                        self.on_error(test_name, e)

                    logger.error(f"Test {test_name} failed with error: {e}")

                    return ParallelResult(
                        test_name=test_name,
                        success=False,
                        exception=e,
                        start_time=start_time,
                        end_time=end_time,
                    )

        # Create all tasks
        tasks = [
            run_with_limit(i, test_case) for i, test_case in enumerate(test_cases)
        ]

        # Execute all in parallel (semaphore limits concurrency)
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results


async def execute_tests_parallel(
    test_cases: List[Any],
    execute_fn: Callable[[Any], Any],
    max_workers: int = 8,
    on_start: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[str, bool, Any], None]] = None,
    on_error: Optional[Callable[[str, Exception], None]] = None,
) -> List[ParallelResult]:
    """
    Convenience function to execute tests in parallel.

    Args:
        test_cases: List of test cases to execute
        execute_fn: Async function to execute a single test case
        max_workers: Maximum concurrent test executions (default 8)
        on_start: Callback when test starts
        on_complete: Callback when test completes
        on_error: Callback when test errors

    Returns:
        List of ParallelResult in same order as test_cases
    """
    executor = ParallelExecutor(
        max_workers=max_workers,
        on_start=on_start,
        on_complete=on_complete,
        on_error=on_error,
    )
    return await executor.execute_all(test_cases, execute_fn)
