"""Telemetry configuration management.

Stores configuration in ~/.evalview/telemetry.json including:
- enabled: Whether telemetry is on/off
- install_id: Anonymous UUID for this installation
- first_run_notice_shown: Whether we've shown the notice
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Environment variable to disable telemetry (always wins)
TELEMETRY_DISABLED_ENV = "EVALVIEW_TELEMETRY_DISABLED"

# Config file path
CONFIG_DIR = Path.home() / ".evalview"
CONFIG_FILE = CONFIG_DIR / "telemetry.json"

# Current schema version
SCHEMA_VERSION = 1


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""

    schema_version: int = SCHEMA_VERSION
    enabled: bool = True
    install_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    first_run_notice_shown: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TelemetryConfig":
        """Create from dictionary."""
        # Handle missing fields with defaults
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            enabled=data.get("enabled", True),
            install_id=data.get("install_id", str(uuid.uuid4())),
            first_run_notice_shown=data.get("first_run_notice_shown", False),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
        )


def load_config() -> TelemetryConfig:
    """Load telemetry config from disk.

    Creates a new config with defaults if none exists.
    """
    if not CONFIG_FILE.exists():
        # Create new config
        config = TelemetryConfig()
        save_config(config)
        return config

    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        return TelemetryConfig.from_dict(data)
    except (json.JSONDecodeError, IOError):
        # Corrupted file, create new config
        config = TelemetryConfig()
        save_config(config)
        return config


def save_config(config: TelemetryConfig) -> None:
    """Save telemetry config to disk."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
    except IOError:
        # Silently fail - telemetry errors should never break functionality
        pass


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled.

    Environment variable EVALVIEW_TELEMETRY_DISABLED=1 always wins.
    Otherwise, check the config file.
    """
    # Environment override always wins
    env_disabled = os.environ.get(TELEMETRY_DISABLED_ENV, "").lower()
    if env_disabled in ("1", "true", "yes"):
        return False

    try:
        config = load_config()
        return config.enabled
    except Exception:
        # If we can't load config, assume enabled
        return True


def get_install_id() -> str:
    """Get the anonymous install UUID."""
    config = load_config()
    return config.install_id


def should_show_first_run_notice() -> bool:
    """Check if we should show the first-run telemetry notice."""
    # Don't show if disabled by env
    env_disabled = os.environ.get(TELEMETRY_DISABLED_ENV, "").lower()
    if env_disabled in ("1", "true", "yes"):
        return False

    try:
        config = load_config()
        return not config.first_run_notice_shown
    except Exception:
        return False


def mark_first_run_notice_shown() -> None:
    """Mark that we've shown the first-run notice."""
    try:
        config = load_config()
        config.first_run_notice_shown = True
        save_config(config)
    except Exception:
        # Silently fail
        pass


def set_telemetry_enabled(enabled: bool) -> None:
    """Enable or disable telemetry."""
    config = load_config()
    config.enabled = enabled
    save_config(config)
