"""Telemetry event definitions.

All events are anonymous and contain no PII or sensitive data.
"""

import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any


def _detect_ci_environment() -> str:
    """Detect if running in CI and which provider.

    Returns 'local' if not in CI, otherwise the CI provider name.
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "github_actions"
    if os.environ.get("GITLAB_CI"):
        return "gitlab_ci"
    if os.environ.get("CIRCLECI"):
        return "circleci"
    if os.environ.get("JENKINS_URL"):
        return "jenkins"
    if os.environ.get("BUILDKITE"):
        return "buildkite"
    if os.environ.get("TRAVIS"):
        return "travis"
    if os.environ.get("BITBUCKET_BUILD_NUMBER"):
        return "bitbucket"
    if os.environ.get("AZURE_PIPELINES") or os.environ.get("TF_BUILD"):
        return "azure_devops"
    # Generic CI detection (many CI systems set CI=true)
    if os.environ.get("CI", "").lower() in ("true", "1", "yes"):
        return "unknown_ci"
    return "local"


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
    ci_environment: str = field(default_factory=_detect_ci_environment)

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
class ChatEvent(BaseEvent):
    """Event for chat session tracking."""

    event_type: str = "chat_session"
    provider: str = ""  # e.g., "ollama", "openai", "anthropic"
    model: str = ""  # e.g., "llama3.2", "gpt-4o" (model name only, no keys)
    message_count: int = 0
    slash_commands_used: Dict[str, int] = field(default_factory=dict)  # e.g., {"/run": 3, "/trace": 1}
    duration_ms: Optional[float] = None


@dataclass
class ErrorEvent(BaseEvent):
    """Event for errors (only error class name, never message content)."""

    event_type: str = "error"
    command_name: str = ""
    error_class: str = ""  # e.g., "ValueError", "ConnectionError"
    # Never include error message content
