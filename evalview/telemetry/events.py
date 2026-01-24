"""Telemetry event definitions.

All events are anonymous and contain no PII or sensitive data.
"""

import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any


def _get_os_info() -> str:
    """Get OS name and version."""
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    elif system == "Windows":
        return f"Windows {platform.release()}"
    elif system == "Linux":
        # Try to get distribution info
        try:
            import distro

            return f"Linux {distro.name()} {distro.version()}"
        except ImportError:
            return f"Linux {platform.release()}"
    return system


def _get_python_version() -> str:
    """Get Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


@dataclass
class BaseEvent:
    """Base event with common fields."""

    event_type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    os_info: str = field(default_factory=_get_os_info)
    python_version: str = field(default_factory=_get_python_version)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for sending."""
        return asdict(self)


@dataclass
class CommandEvent(BaseEvent):
    """Event for CLI command execution."""

    event_type: str = "command"
    command_name: str = ""
    duration_ms: Optional[float] = None
    success: bool = True
    # Additional command-specific properties
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunEvent(BaseEvent):
    """Event for evalview run command with additional metrics."""

    event_type: str = "run"
    command_name: str = "run"
    adapter_type: Optional[str] = None
    test_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    duration_ms: Optional[float] = None
    success: bool = True
    # Feature flags used
    diff_mode: bool = False
    watch_mode: bool = False
    parallel: bool = False


@dataclass
class ErrorEvent(BaseEvent):
    """Event for errors (only error class name, never message content)."""

    event_type: str = "error"
    command_name: str = ""
    error_class: str = ""  # e.g., "ValueError", "ConnectionError"
    # Never include error message content
