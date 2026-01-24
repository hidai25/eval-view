"""Decorators for tracking CLI command usage."""

import functools
import time
from typing import Callable, Any, Optional, Dict

from evalview.telemetry.config import is_telemetry_enabled
from evalview.telemetry.events import CommandEvent, ErrorEvent
from evalview.telemetry.client import get_client


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

            name = command_name or func.__name__
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
                # Calculate duration
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Extract additional properties if extractor provided
                properties = {}
                if properties_extractor:
                    try:
                        properties = properties_extractor(*args, **kwargs) or {}
                    except Exception:
                        pass

                # Send command event
                event = CommandEvent(
                    command_name=name,
                    duration_ms=duration_ms,
                    success=success,
                    properties=properties,
                )
                get_client().track(event)

                # Send error event if failed
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
