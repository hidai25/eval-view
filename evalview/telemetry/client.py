"""Telemetry client.

Sends events to PostHog for anonymous usage analytics.
All failures are silently ignored - telemetry should never break functionality.
"""

from typing import Optional

from evalview.telemetry.config import is_telemetry_enabled, get_install_id
from evalview.telemetry.events import BaseEvent

# PostHog configuration (project API key is write-only, safe to expose)
POSTHOG_API_KEY = "phc_jvMrFFBXisMCJHKZlLshtrrCj1XoWWs321Yuy2ARlYx"
POSTHOG_HOST = "https://us.i.posthog.com"

# Singleton client
_client: Optional["TelemetryClient"] = None


class TelemetryClient:
    """Telemetry sender."""

    def __init__(self):
        self._posthog = None

    def _lazy_init_posthog(self) -> bool:
        """Lazily initialize PostHog client."""
        if self._posthog is not None:
            return True

        try:
            from posthog import Posthog

            self._posthog = Posthog(project_api_key=POSTHOG_API_KEY, host=POSTHOG_HOST)
            return True
        except ImportError:
            # PostHog not installed - that's fine
            return False

    def track(self, event: BaseEvent):
        """Send an event to PostHog."""
        if not is_telemetry_enabled():
            return

        if not self._lazy_init_posthog():
            return

        try:
            install_id = get_install_id()
            # Determine event name:
            # - ErrorEvent always uses "error" to allow filtering all errors
            # - Other events use command_name if available, otherwise event_type
            if event.event_type == "error":
                event_name = "error"
            else:
                event_name = getattr(event, "command_name", None) or event.event_type

            self._posthog.capture(
                distinct_id=install_id,
                event=event_name,
                properties=event.to_dict(),
            )
            self._posthog.flush()
        except Exception:
            # Silently ignore all errors
            pass


def get_client() -> TelemetryClient:
    """Get the singleton telemetry client."""
    global _client
    if _client is None:
        _client = TelemetryClient()
    return _client


def track(event: BaseEvent):
    """Convenience function to track an event."""
    get_client().track(event)
