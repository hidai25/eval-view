"""Decorators for tracking CLI command usage."""

import atexit
import functools
import time
from typing import Callable, Any, Optional, Dict

from evalview.telemetry.config import is_telemetry_enabled
from evalview.telemetry.events import CommandEvent, ErrorEvent, SessionEvent
from evalview.telemetry.client import get_client

# ── Session tracking ──────────────────────────────────────────────────────────
# Tracks total time spent in this CLI invocation.

_session_start: float = time.perf_counter()
_session_command: str = ""
_session_commands_run: int = 0


def _send_session_event() -> None:
    """Send session duration event at process exit."""
    if not is_telemetry_enabled():
        return
    try:
        duration_ms = (time.perf_counter() - _session_start) * 1000
        # Only send if session was meaningful (> 500ms, a command actually ran)
        if duration_ms < 500 or _session_commands_run == 0:
            return
        event = SessionEvent(
            command_name=_session_command,
            session_duration_ms=round(duration_ms),
            commands_run=_session_commands_run,
        )
        get_client().track(event)
    except Exception:
        pass


atexit.register(_send_session_event)


def track_command(
    command_name: Optional[str] = None,
    properties_extractor: Optional[Callable[..., Dict[str, Any]]] = None,
):
    """Decorator to track CLI command execution.

    Args:
        command_name: Name of the command (defaults to function name)
        properties_extractor: Optional function to extract additional properties
                            from command arguments. Should return a dict.

    Example:
        @track_command("init")
        def init(dir: str, interactive: bool):
            ...

        @track_command("run", lambda **kw: {"adapter": kw.get("adapter_type")})
        def run(adapter_type: str, ...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Quick check - skip tracking overhead if disabled
            if not is_telemetry_enabled():
                return func(*args, **kwargs)

            global _session_command, _session_commands_run
            name = command_name or func.__name__

            # Record this as the primary command for session tracking
            if _session_commands_run == 0:
                _session_command = name
            _session_commands_run += 1

            start_time = time.perf_counter()
            success = True
            error_class = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_class = type(e).__name__
                raise
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000

                properties = {}
                if properties_extractor:
                    try:
                        properties = properties_extractor(*args, **kwargs) or {}
                    except Exception:
                        pass

                event = CommandEvent(
                    command_name=name,
                    duration_ms=duration_ms,
                    success=success,
                    properties=properties,
                )
                get_client().track(event)

                if error_class:
                    error_event = ErrorEvent(
                        command_name=name,
                        error_class=error_class,
                    )
                    get_client().track(error_event)

        return wrapper

    return decorator


def track_run_command(
    adapter_type: Optional[str] = None,
    test_count: int = 0,
    pass_count: int = 0,
    fail_count: int = 0,
    duration_ms: float = 0,
    diff_mode: bool = False,
    watch_mode: bool = False,
    parallel: bool = False,
):
    """Track a run command execution with full metrics.

    This is called manually after the run completes to capture all metrics.
    """
    from evalview.telemetry.events import RunEvent

    if not is_telemetry_enabled():
        return

    event = RunEvent(
        adapter_type=adapter_type,
        test_count=test_count,
        pass_count=pass_count,
        fail_count=fail_count,
        duration_ms=duration_ms,
        diff_mode=diff_mode,
        watch_mode=watch_mode,
        parallel=parallel,
        success=fail_count == 0,
    )
    get_client().track(event)
