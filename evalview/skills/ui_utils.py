"""UI utilities for skills testing commands.

Provides reusable components for console output, progress display, and formatting.
"""

import time
from typing import Callable, Optional
from rich.console import Console
from rich.live import Live

from evalview.skills.constants import (
    SPINNER_FRAMES,
    SPINNER_REFRESH_RATE,
    SPINNER_SLEEP_INTERVAL,
)


def print_evalview_banner(console: Console, subtitle: Optional[str] = None) -> None:
    """Print the EvalView ASCII art banner.

    Args:
        console: Rich console instance
        subtitle: Optional subtitle text to display below the banner
    """
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗██╗   ██╗ █████╗ ██╗    ██╗   ██╗██╗███████╗██╗    ██╗[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔════╝██║   ██║██╔══██╗██║    ██║   ██║██║██╔════╝██║    ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]█████╗  ██║   ██║███████║██║    ██║   ██║██║█████╗  ██║ █╗ ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔══╝  ╚██╗ ██╔╝██╔══██║██║    ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗ ╚████╔╝ ██║  ██║███████╗╚████╔╝ ██║███████╗╚███╔███╔╝[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ [/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")

    if subtitle:
        # Center the subtitle within the banner width (66 chars inside)
        subtitle_padded = subtitle.center(66)
        console.print(f"[bold cyan]║[/bold cyan]{subtitle_padded}[bold cyan]║[/bold cyan]")

    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print()


def format_elapsed_time(start_time: float) -> str:
    """Format elapsed time as MM:SS.mmm.

    Args:
        start_time: Start time from time.time()

    Returns:
        Formatted time string like "02:34.567"
    """
    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    secs_int = int(secs)
    ms = int((secs - secs_int) * 1000)
    return f"{int(mins):02d}:{secs_int:02d}.{ms:03d}"


class ProgressSpinner:
    """Animated spinner for displaying progress during long operations.

    Example:
        spinner = ProgressSpinner(console, "Running tests...")
        with spinner:
            # Do work
            pass
    """

    def __init__(
        self,
        console: Console,
        message: str,
        show_elapsed: bool = True,
    ):
        """Initialize progress spinner.

        Args:
            console: Rich console instance
            message: Message to display with spinner
            show_elapsed: Whether to show elapsed time
        """
        self.console = console
        self.message = message
        self.show_elapsed = show_elapsed
        self.start_time = time.time()
        self.spinner_idx = 0
        self.live: Optional[Live] = None

    def __enter__(self) -> "ProgressSpinner":
        """Start the spinner display."""
        self.start_time = time.time()
        self.spinner_idx = 0
        self.live = Live(
            self._get_display(),
            console=self.console,
            refresh_per_second=SPINNER_REFRESH_RATE,
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the spinner display."""
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)

    def update(self) -> None:
        """Update the spinner display (call in a loop)."""
        if self.live:
            self.live.update(self._get_display())
            time.sleep(SPINNER_SLEEP_INTERVAL)

    def _get_display(self) -> str:
        """Get the current spinner display string."""
        spinner = SPINNER_FRAMES[self.spinner_idx % len(SPINNER_FRAMES)]
        self.spinner_idx += 1

        if self.show_elapsed:
            elapsed = format_elapsed_time(self.start_time)
            return f"{spinner} {self.message} [yellow]{elapsed}[/yellow]"
        else:
            return f"{spinner} {self.message}"


def run_with_spinner(
    console: Console,
    message: str,
    operation: Callable[[], any],
    show_elapsed: bool = True,
) -> any:
    """Run a synchronous operation with a progress spinner.

    Args:
        console: Rich console instance
        message: Message to display with spinner
        operation: Callable to execute
        show_elapsed: Whether to show elapsed time

    Returns:
        Result of the operation

    Example:
        result = run_with_spinner(
            console,
            "Loading data...",
            lambda: expensive_operation()
        )
    """
    spinner = ProgressSpinner(console, message, show_elapsed)
    with spinner:
        while True:
            # For sync operations, we just run and return
            # This is a simplified version - for real async,
            # use run_async_with_spinner instead
            return operation()
