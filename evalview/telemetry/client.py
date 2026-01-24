"""Telemetry client.

Sends events to PostHog for anonymous usage analytics.
All failures are silently ignored - telemetry should never break functionality.
"""

import os
from typing import Optional, Dict, Any

from evalview.telemetry.config import is_telemetry_enabled, get_install_id
from evalview.telemetry.events import BaseEvent

# PostHog configuration - read lazily to allow .env.local to load first
POSTHOG_HOST_DEFAULT = "https://us.i.posthog.com"

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

        # Read env vars lazily (after .env.local is loaded by CLI)
        api_key = os.environ.get("POSTHOG_API_KEY", "")
        host = os.environ.get("POSTHOG_HOST", POSTHOG_HOST_DEFAULT)

        # Skip if no API key configured
        if not api_key:
            return False

        try:
            from posthog import Posthog

            self._posthog = Posthog(project_api_key=api_key, host=host)
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
            self._posthog.capture(
                distinct_id=install_id,
                event=event.event_type,
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
