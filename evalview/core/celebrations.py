"""Celebration and delight moments for EvalView.

This module handles all the "soul" of EvalView - making regression detection
feel memorable, fun, and habit-forming.

Philosophy: "Serious about regressions, playful about everything else"
"""

from rich.console import Console
from rich.panel import Panel
from evalview.core.project_state import ProjectState

console = Console()

# Milestone thresholds for streak celebrations
STREAK_MILESTONE_START = 1
STREAK_MILESTONE_SMALL = 3
STREAK_MILESTONE_MEDIUM = 5
STREAK_MILESTONE_LARGE = 10
STREAK_MILESTONE_LEGENDARY = 25
STREAK_MILESTONE_INCREDIBLE = 50

# Streak empathy threshold
STREAK_BREAK_EMPATHY_THRESHOLD = 5

# Banner widths
BANNER_WIDTH = 60


class Celebrations:
    """Handles all delight moments and celebrations."""

    @staticmethod
    def first_snapshot(test_count: int) -> None:
        """Celebrate the first snapshot ever.

        Args:
            test_count: Number of tests snapshotted
        """
        console.print()
        console.print("[bold cyan]" + "=" * BANNER_WIDTH + "[/bold cyan]")
        console.print()
        console.print("       📸  [bold]BASELINE CAPTURED[/bold]  📸")
        console.print()
        console.print(f"  You've snapshotted {test_count} test(s).")
        console.print("  Regression detection is now [bold green]ACTIVE[/bold green]. ✅")
        console.print()
        console.print("  Next time your agent changes behavior,")
        console.print("  [cyan]evalview check[/cyan] will catch it.")
        console.print()
        console.print("[bold cyan]" + "=" * BANNER_WIDTH + "[/bold cyan]")
        console.print()

    @staticmethod
    def clean_check_streak(state: ProjectState) -> None:
        """Celebrate based on streak length.

        Args:
            state: Current project state
        """
        streak = state.current_streak

        if streak == STREAK_MILESTONE_START:
            # Don't celebrate single check, just acknowledge
            console.print("[dim]Streak started at 1. Keep it going! 🔄[/dim]\n")

        elif streak == STREAK_MILESTONE_SMALL:
            console.print(f"[green]🎯 {STREAK_MILESTONE_SMALL} clean checks in a row! You're on a roll.[/green]\n")

        elif streak == STREAK_MILESTONE_MEDIUM:
            console.print()
            console.print(Panel(
                f"[bold green]🔥 {STREAK_MILESTONE_MEDIUM}-CHECK STREAK![/bold green]\n\n"
                "Your agent is stable. That's what we like to see.",
                border_style="green"
            ))
            console.print()

        elif streak == STREAK_MILESTONE_LARGE:
            console.print()
            console.print(Panel(
                f"[bold green]🌟 {STREAK_MILESTONE_LARGE}-CHECK STREAK! 🌟[/bold green]\n\n"
                "[dim]Achievement unlocked: Reliability Champion[/dim]\n"
                f"Your agent hasn't regressed in {STREAK_MILESTONE_LARGE} checks. Beautiful.",
                border_style="green"
            ))
            console.print("""
        ⭐
       ⭐⭐⭐
      ⭐⭐⭐⭐⭐
            """)
            console.print()

        elif streak == STREAK_MILESTONE_LEGENDARY:
            console.print()
            console.print(Panel(
                f"[bold cyan]💎 LEGENDARY: {STREAK_MILESTONE_LEGENDARY}-Check Streak[/bold cyan]\n\n"
                "This is production-grade stability.\n"
                "Consider sharing this achievement! 🏆",
                border_style="cyan"
            ))
            console.print()

            # Offer shareable badge
            Celebrations.shareable_badge(streak)

        elif streak == STREAK_MILESTONE_INCREDIBLE:
            console.print()
            console.print(Panel(
                f"[bold magenta]🚀 INCREDIBLE: {STREAK_MILESTONE_INCREDIBLE}-Check Streak! 🚀[/bold magenta]\n\n"
                "Your agent is rock solid.\n"
                "This deserves recognition! 🎖️",
                border_style="magenta"
            ))
            console.print()
            Celebrations.shareable_badge(streak)

        elif streak % 10 == 0 and streak > STREAK_MILESTONE_INCREDIBLE:
            console.print(f"[cyan]🚀 {streak}-check streak! Legendary stability.[/cyan]\n")

    @staticmethod
    def streak_broken(state: ProjectState, diff_status: str) -> None:
        """Empathetic message when streak breaks.

        Args:
            state: Current project state (with old streak)
            diff_status: Status that broke the streak
        """
        old_streak = state.current_streak

        if old_streak >= STREAK_BREAK_EMPATHY_THRESHOLD:
            console.print(f"[yellow]Streak ended at {old_streak} 😔[/yellow]")
            console.print(f"[dim]Status: {diff_status}[/dim]")
            console.print("[dim]It happens! Fix the regression and start a new streak.[/dim]\n")

        if old_streak > state.longest_streak:
            console.print(f"[cyan]✨ New personal record: {old_streak} checks![/cyan]\n")

    @staticmethod
    def regression_guidance(diff_summary: str) -> None:
        """Helpful next steps panel for regressions.

        Args:
            diff_summary: Summary of what changed
        """
        console.print()
        console.print(Panel(
            "[yellow]⚠️  REGRESSION DETECTED[/yellow]\n\n"
            "Your agent's behavior changed. This might be intentional!\n\n"
            "[bold]What changed?[/bold]\n"
            f"  {diff_summary}\n\n"
            "[bold]What to do:[/bold]\n"
            "  • If this change is good: [cyan]evalview snapshot[/cyan] to update baseline\n"
            "  • If this is a bug: fix it and [cyan]evalview check[/cyan] again\n"
            "  • See details: [cyan]evalview view[/cyan] (coming soon)",
            title="Regression Detected",
            border_style="yellow"
        ))
        console.print()

    @staticmethod
    def no_tests_found() -> None:
        """Helpful message when no tests found."""
        console.print()
        console.print(Panel(
            "[yellow]🤷 No test cases found[/yellow]\n\n"
            "I looked in tests/test-cases/ but didn't find any YAML files.\n\n"
            "[bold]Let's fix that:[/bold]\n"
            "  • Create a new project: [cyan]evalview init[/cyan]\n"
            "  • Try the demo: [cyan]evalview demo[/cyan]\n"
            "  • Or see docs: [cyan]evalview --help[/cyan]",
            title="No Tests Found",
            border_style="yellow"
        ))
        console.print()

    @staticmethod
    def no_snapshot_found() -> None:
        """Helpful message when no snapshot exists."""
        console.print()
        console.print(Panel(
            "[yellow]🤔 No baseline found yet[/yellow]\n\n"
            "Before you can check for regressions, you need a baseline.\n\n"
            "[bold]Create one now:[/bold]\n"
            "  [cyan]evalview snapshot[/cyan]\n\n"
            "[dim]This captures your agent's current behavior as the 'golden' reference.[/dim]",
            title="No Baseline",
            border_style="yellow"
        ))
        console.print()

    @staticmethod
    def welcome_back(days_inactive: int) -> None:
        """Welcome message after long break.

        Args:
            days_inactive: Number of days since last check
        """
        console.print(f"[cyan]Welcome back! It's been {days_inactive} days. 👋[/cyan]")
        console.print("[dim]Let's see if your agent stayed stable...[/dim]\n")

    @staticmethod
    def fixed_regression(state: ProjectState) -> None:
        """Celebrate fixing a regression.

        Args:
            state: Current project state
        """
        console.print("[green]🎉 Fixed! Back to baseline.[/green]")
        console.print(f"[dim]Streak restarted at {state.current_streak}.[/dim]\n")

    @staticmethod
    def first_check() -> None:
        """Encourage on first check ever."""
        console.print("[cyan]This is your first check! 🎯[/cyan]")
        console.print("[dim]From now on, I'll catch when your agent drifts.[/dim]\n")

    @staticmethod
    def health_summary(state: ProjectState) -> None:
        """Show project health at a glance.

        Args:
            state: Current project state
        """
        if state.total_checks < 5:
            return  # Not enough data

        total = state.total_checks
        clean = total - state.regression_count
        pass_rate = (clean / total) * 100

        # Visual health indicator
        health_emoji = "🟢" if pass_rate >= 90 else "🟡" if pass_rate >= 75 else "🔴"

        console.print()
        console.print(Panel(
            f"{health_emoji} [bold]Project Health: {pass_rate:.0f}%[/bold]\n\n"
            f"  Total checks: {total}\n"
            f"  Clean: {clean}\n"
            f"  Regressions: {state.regression_count}\n"
            f"  Current streak: {state.current_streak}\n"
            f"  Best streak: {state.longest_streak}",
            title="Health Summary",
            border_style="blue"
        ))
        console.print()

    @staticmethod
    def shareable_badge(streak: int) -> None:
        """Display shareable ASCII badge for social proof.

        Args:
            streak: Streak length to celebrate
        """
        console.print()
        console.print("[dim]Want to celebrate? Copy this badge:[/dim]")
        console.print()
        console.print("[cyan]" + "=" * 50 + "[/cyan]")
        console.print(f"  🏆 EvalView: {streak}-Check Streak")
        console.print("  My agent hasn't regressed in", streak, "checks!")
        console.print("  https://github.com/hidai25/EvalView")
        console.print("[cyan]" + "=" * 50 + "[/cyan]")
        console.print()

    @staticmethod
    def conversion_suggestion(passed_count: int) -> None:
        """Suggest snapshot workflow after successful run.

        Args:
            passed_count: Number of tests that passed
        """
        console.print()
        console.print(Panel(
            "[bold]💡 Tip: Enable regression detection[/bold]\n\n"
            f"Your {passed_count} test(s) passed! Save this as a baseline:\n"
            "  [cyan]evalview snapshot[/cyan]\n\n"
            "Then catch regressions automatically:\n"
            "  [cyan]evalview check[/cyan]\n\n"
            "[dim]This creates a habit loop: snapshot → check → fix → snapshot...[/dim]",
            title="Snapshot Workflow",
            border_style="blue"
        ))
        console.print()

    @staticmethod
    def reactivation_nudge(days_inactive: int) -> None:
        """Gentle nudge if inactive for a while.

        Args:
            days_inactive: Number of days since last check
        """
        if days_inactive < 7:
            return  # Too soon for nudge

        console.print()
        console.print(Panel(
            f"[yellow]It's been {days_inactive} days since your last check.[/yellow]\n\n"
            "Agent behavior can drift over time. Run a quick check:\n"
            "  [cyan]evalview check[/cyan]\n\n"
            "[dim]Takes just a few seconds to verify stability.[/dim]",
            title="⏰ Reminder",
            border_style="yellow"
        ))
        console.print()
