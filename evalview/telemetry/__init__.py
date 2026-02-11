"""Anonymous telemetry for EvalView CLI usage analytics."""

from evalview.telemetry.config import (
    TelemetryConfig,
    load_config,
    save_config,
    is_telemetry_enabled,
    get_install_id,
    should_show_first_run_notice,
    mark_first_run_notice_shown,
)
from evalview.telemetry.events import (
    CommandEvent,
    RunEvent,
    ChatEvent,
    ErrorEvent,
)
from evalview.telemetry.client import TelemetryClient, get_client
from evalview.telemetry.decorators import track_command

__all__ = [
    # Config
    "TelemetryConfig",
    "load_config",
    "save_config",
    "is_telemetry_enabled",
    "get_install_id",
    "should_show_first_run_notice",
    "mark_first_run_notice_shown",
    # Events
    "CommandEvent",
    "RunEvent",
    "ChatEvent",
    "ErrorEvent",
    # Client
    "TelemetryClient",
    "get_client",
    # Decorators
    "track_command",
]
