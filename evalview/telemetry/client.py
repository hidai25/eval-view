"""Telemetry client.

Sends events to PostHog for anonymous usage analytics.
All failures are silently ignored - telemetry should never break functionality.

Person identity strategy:
    Each installation gets a stable UUID (install_id). On first use we call
    posthog.identify() to attach readable person properties — this is what
    makes users show up with names and OS info in PostHog instead of raw UUIDs.

Developer filtering:
    Set EVALVIEW_DEV=1 to mark events with is_developer=True.
    Filter these out in PostHog with: Properties → is_developer → is not set / false.
"""

import os
import platform
import sys
from typing import Optional

from evalview.telemetry.config import (
    is_telemetry_enabled,
    get_install_id,
    load_config,
    save_config,
)
from evalview.telemetry.events import BaseEvent

# PostHog configuration (project API key is write-only, safe to expose)
POSTHOG_API_KEY = "phc_jvMrFFBXisMCJHKZlLshtrrCj1XoWWs321Yuy2ARlYx"
POSTHOG_HOST = "https://us.i.posthog.com"

# Singleton client
_client: Optional["TelemetryClient"] = None


def _is_developer() -> bool:
    """Return True if running in developer/maintainer mode.

    Set EVALVIEW_DEV=1 in your environment to tag your own events.
    Filter these out in PostHog: Properties → is_developer = true.
    """
    return os.environ.get("EVALVIEW_DEV", "").lower() in ("1", "true", "yes")


def _readable_name(install_id: str) -> str:
    """Generate a short readable name from the install UUID.

    E.g. "3f8a2b" → "EvalView-3f8a2b"
    This shows up as the person's $name in PostHog.
    """
    return f"EvalView-{install_id[:6]}"


def _get_python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _get_os_info() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if system == "Windows":
        return f"Windows {platform.release()}"
    return f"Linux {platform.release()}"


class TelemetryClient:
    """Telemetry sender."""

    def __init__(self) -> None:
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
            return False

    def _identify_once(self, install_id: str) -> None:
        """Send posthog.identify() the first time this install is seen.

        This populates the person record in PostHog with readable properties
        so users show up as 'EvalView-3f8a2b (macOS 14.0)' instead of raw UUIDs.
        Only fires once per installation — tracked via `identified` flag in config.
        """
        try:
            config = load_config()
            if getattr(config, "identified", False):
                return  # Already identified

            self._posthog.identify(
                distinct_id=install_id,
                properties={
                    "$name": _readable_name(install_id),
                    "os_info": _get_os_info(),
                    "python_version": _get_python_version(),
                    "install_date": config.created_at,
                    "is_developer": _is_developer(),
                },
            )
            # Mark as identified so we don't repeat this on every run
            config.identified = True  # type: ignore[attr-defined]
            save_config(config)
        except Exception:
            pass

    def track(self, event: BaseEvent) -> None:
        """Send an event to PostHog."""
        if not is_telemetry_enabled():
            return

        if not self._lazy_init_posthog():
            return

        try:
            install_id = get_install_id()

            # Identify the person once so PostHog shows readable names
            self._identify_once(install_id)

            # Determine event name
            if event.event_type == "error":
                event_name = "error"
            else:
                event_name = getattr(event, "command_name", None) or event.event_type

            properties = event.to_dict()
            # Always attach developer flag so events can be filtered in PostHog
            properties["is_developer"] = _is_developer()
            # Attach $session_id so PostHog groups events into sessions and
            # can calculate session duration, events-per-session, and retention.
            try:
                from evalview.telemetry.decorators import _session_id
                properties["$session_id"] = _session_id
            except ImportError:
                pass

            self._posthog.capture(
                distinct_id=install_id,
                event=event_name,
                properties=properties,
            )
            self._posthog.flush()
        except Exception:
            pass


def get_client() -> TelemetryClient:
    """Get the singleton telemetry client."""
    global _client
    if _client is None:
        _client = TelemetryClient()
    return _client


def track(event: BaseEvent) -> None:
    """Convenience function to track an event."""
    get_client().track(event)
