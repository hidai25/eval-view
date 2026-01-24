"""Background telemetry client.

Sends events in a background thread to never block the CLI.
All failures are silently ignored - telemetry should never break functionality.
"""

import atexit
import os
import queue
import threading
from typing import Optional, Dict, Any

from evalview.telemetry.config import is_telemetry_enabled, get_install_id
from evalview.telemetry.events import BaseEvent

# PostHog configuration from environment
# Set POSTHOG_API_KEY in .env.local for local development
POSTHOG_API_KEY = os.environ.get("POSTHOG_API_KEY", "")
POSTHOG_HOST = os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com")

# Singleton client
_client: Optional["TelemetryClient"] = None


class TelemetryClient:
    """Background telemetry sender.

    Uses a queue and background thread to send events without blocking.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._posthog = None
        self._started = False

    def _lazy_init_posthog(self) -> bool:
        """Lazily initialize PostHog client."""
        if self._posthog is not None:
            return True

        # Skip if no API key configured
        if not POSTHOG_API_KEY:
            return False

        try:
            import posthog

            posthog.project_api_key = POSTHOG_API_KEY
            posthog.host = POSTHOG_HOST
            # Disable PostHog's own debug logging
            posthog.debug = False
            # Use sync mode for our background thread
            posthog.sync_mode = True
            self._posthog = posthog
            return True
        except ImportError:
            # PostHog not installed - that's fine
            return False

    def _worker(self):
        """Background worker that sends events."""
        while not self._shutdown.is_set():
            try:
                # Wait for events with timeout so we can check shutdown
                try:
                    event = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if event is None:  # Shutdown signal
                    break

                # Try to send the event
                self._send_event(event)
                self._queue.task_done()

            except Exception:
                # Never let any error escape the worker
                pass

        # Drain remaining events on shutdown
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                if event is not None:
                    self._send_event(event)
            except Exception:
                pass

    def _send_event(self, event: Dict[str, Any]):
        """Actually send an event to PostHog."""
        if not self._lazy_init_posthog():
            return

        try:
            install_id = get_install_id()
            self._posthog.capture(
                distinct_id=install_id,
                event=event.get("event_type", "unknown"),
                properties=event,
            )
        except Exception:
            # Silently ignore all errors
            pass

    def start(self):
        """Start the background worker thread."""
        if self._started:
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._started = True

        # Register shutdown handler
        atexit.register(self.shutdown)

    def track(self, event: BaseEvent):
        """Queue an event for sending.

        Non-blocking - returns immediately.
        """
        if not is_telemetry_enabled():
            return

        # Start worker if needed
        if not self._started:
            self.start()

        try:
            # Don't block if queue is full
            self._queue.put_nowait(event.to_dict())
        except queue.Full:
            # Drop event if queue is full - never block
            pass

    def shutdown(self):
        """Shutdown the background worker gracefully."""
        if not self._started:
            return

        self._shutdown.set()
        self._queue.put(None)  # Signal to exit

        if self._thread and self._thread.is_alive():
            # Wait briefly for remaining events
            self._thread.join(timeout=1.0)


def get_client() -> TelemetryClient:
    """Get the singleton telemetry client."""
    global _client
    if _client is None:
        _client = TelemetryClient()
    return _client


def track(event: BaseEvent):
    """Convenience function to track an event."""
    get_client().track(event)
