"""File watcher for auto-running tests on changes."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Set, Any

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None  # type: ignore[misc, assignment]
    FileSystemEventHandler = object  # type: ignore[misc, assignment]


# Use a base class that works with or without watchdog
_BaseHandler = FileSystemEventHandler if WATCHDOG_AVAILABLE else object


class DebouncedTestHandler(_BaseHandler):  # type: ignore[valid-type, misc]
    """Handler that debounces file changes and triggers test runs."""

    def __init__(
        self,
        callback: Callable[[], None],
        debounce_seconds: float = 2.0,
        patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the debounced handler.

        Args:
            callback: Function to call when files change (after debounce)
            debounce_seconds: Time to wait after last change before triggering
            patterns: File patterns to watch (default: *.yaml, *.yml)
        """
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.patterns = patterns or [".yaml", ".yml"]
        self._last_change: float = 0
        self._pending_task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._changed_files: Set[str] = set()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for scheduling callbacks."""
        self._loop = loop

    def _should_handle(self, path: str) -> bool:
        """Check if this file should trigger a re-run."""
        return any(path.endswith(p) for p in self.patterns)

    def _schedule_run(self, path: str) -> None:
        """Schedule a debounced test run."""
        if not self._loop:
            return

        self._changed_files.add(path)
        self._last_change = time.time()

        # Cancel any pending task
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

        # Schedule new task
        async def debounced_callback() -> None:
            await asyncio.sleep(self.debounce_seconds)

            # Check if more changes came in during the wait
            if time.time() - self._last_change >= self.debounce_seconds - 0.1:
                files = self._changed_files.copy()
                self._changed_files.clear()
                logger.info(f"Changes detected in: {', '.join(files)}")

                try:
                    result = self.callback()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error running callback: {e}")

        self._pending_task = self._loop.create_task(debounced_callback())

    def on_modified(self, event: Any) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return
        if self._should_handle(event.src_path):
            self._schedule_run(event.src_path)

    def on_created(self, event: Any) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return
        if self._should_handle(event.src_path):
            self._schedule_run(event.src_path)


class TestWatcher:
    """Watch test files and re-run tests on changes."""

    def __init__(
        self,
        paths: List[str],
        run_callback: Callable[[], None],
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize the test watcher.

        Args:
            paths: Directories to watch for changes
            run_callback: Async function to call when files change
            debounce_seconds: Time to wait after last change before running
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError(
                "watchdog is required for watch mode. Install with: pip install watchdog"
            )

        self.paths = [Path(p) for p in paths]
        self.run_callback = run_callback
        self.debounce_seconds = debounce_seconds
        self._observer: Any = None  # Type is Observer when watchdog is available
        self._handler: Optional[DebouncedTestHandler] = None
        self._running = False

    async def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            return

        loop = asyncio.get_event_loop()

        self._handler = DebouncedTestHandler(
            callback=self.run_callback,
            debounce_seconds=self.debounce_seconds,
        )
        self._handler.set_event_loop(loop)

        self._observer = Observer()

        for path in self.paths:
            if path.exists():
                self._observer.schedule(self._handler, str(path), recursive=True)
                logger.info(f"Watching: {path}")

        self._observer.start()
        self._running = True

    def stop(self) -> None:
        """Stop watching for file changes."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        self._running = False


async def watch_and_run(
    paths: List[str],
    run_callback: Callable[[], None],
    debounce_seconds: float = 2.0,
    on_start: Optional[Callable[[], None]] = None,
) -> None:
    """
    Watch test files and re-run on changes.

    Args:
        paths: Directories to watch
        run_callback: Async function to run tests
        debounce_seconds: Debounce delay
        on_start: Optional callback when starting watch mode
    """
    if not WATCHDOG_AVAILABLE:
        raise ImportError(
            "watchdog is required for watch mode. Install with: pip install watchdog"
        )

    watcher = TestWatcher(paths, run_callback, debounce_seconds)

    try:
        await watcher.start()

        if on_start:
            on_start()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        pass
    finally:
        watcher.stop()
