"""Streak celebrations and onboarding prompts."""

import os

from rich.console import Console
from rich.panel import Panel
from evalview.core.project_state import ProjectState

# When running inside evalview demo, suppress personal onboarding messages
# that assume this is the user's real project (it's a throwaway temp dir).
_IS_DEMO = os.environ.get("EVALVIEW_DEMO") == "1"

console = Console()

# Milestone thresholds for streak celebrations
STREAK_1 = 1
STREAK_3 = 3
STREAK_5 = 5
STREAK_10 = 10
STREAK_25 = 25
STREAK_50 = 50

# Streak empathy threshold
STREAK_BREAK_THRESHOLD = 5

# Banner widths
BANNER_WIDTH = 60


class Celebrations:
    """Streak milestones, onboarding panels, and regression guidance."""

    @staticmethod
    def first_snapshot(test_count: int) -> None:
        """Celebrate the first snapshot ever.

        Args:
            test_count: Number of tests snapshotted
        """
        if _IS_DEMO:
            return  # individual "✓ Snapshotted: X" lines from snapshot command are sufficient
        console.print()
        console.print("[bold cyan]" + "=" * BANNER_WIDTH + "[/bold cyan]")
        console.print()
        console.print("       📸  [bold]BASELINE CAPTURED[/bold]  📸")
        console.print()
        console.print(f"  You've snapshotted {test_count} test(s).")
        console.print("  Regression detection is now [bold green]ACTIVE[/bold green]. ✅")
        console.print()
        console.print("[bold cyan]" + "=" * BANNER_WIDTH + "[/bold cyan]")
        console.print()
        console.print(Panel(
            "[bold]Make it stick — 3 steps:[/bold]\n\n"
            "[bold]1.[/bold] Verify it works:\n"
            "     [cyan]evalview check[/cyan]\n\n"
            "[bold]2.[/bold] Commit your goldens so your team shares the baseline:\n"
            "     [cyan]git add .evalview/golden/[/cyan]\n"
            "     [cyan]git commit -m 'Add agent test baselines'[/cyan]\n\n"
            "[bold]3.[/bold] Block regressions from merging (GitHub Actions):\n"
            "     [dim]# .github/workflows/eval.yml[/dim]\n"
            "     [cyan]- run: evalview check --fail-on REGRESSION[/cyan]\n"
            "     [cyan]  env:[/cyan]\n"
            "     [cyan]    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}[/cyan]",
            title="What's next",
            border_style="cyan",
            padding=(1, 2),
        ))
        console.print()

    @staticmethod
    def clean_check_streak(state: ProjectState) -> None:
        """Celebrate based on streak length.

        Args:
            state: Current project state
        """
        streak = state.current_streak

        if streak == STREAK_1:
            # Don't celebrate single check, just acknowledge
            console.print("[dim]Streak started at 1. Keep it going! 🔄[/dim]\n")

        elif streak == 2:
            console.print("[green]Two clean checks in a row. 🎯[/green]\n")

        elif streak == STREAK_3:
            console.print(f"[green]🎯 {STREAK_3} clean checks in a row! You're on a roll.[/green]\n")

        elif streak == STREAK_5:
            console.print()
            console.print(Panel(
                f"[bold green]🔥 {STREAK_5}-CHECK STREAK![/bold green]\n\n"
                "Your agent is stable. That's what we like to see.",
                border_style="green"
            ))
            console.print()

        elif streak == STREAK_10:
            console.print()
            console.print(Panel(
                f"[bold green]🌟 {STREAK_10}-CHECK STREAK! 🌟[/bold green]\n\n"
                "[dim]Achievement unlocked: Reliability Champion[/dim]\n"
                f"Your agent hasn't regressed in {STREAK_10} checks. Beautiful.",
                border_style="green"
            ))
            console.print("""
        ⭐
       ⭐⭐⭐
      ⭐⭐⭐⭐⭐
            """)
            console.print()

        elif streak == STREAK_25:
            console.print()
            console.print(Panel(
                f"[bold cyan]💎 LEGENDARY: {STREAK_25}-Check Streak[/bold cyan]\n\n"
                "This is production-grade stability.\n"
                "Consider sharing this achievement! 🏆",
                border_style="cyan"
            ))
            console.print()

            # Offer shareable badge
            Celebrations.shareable_badge(streak)

        elif streak == STREAK_50:
            console.print()
            console.print(Panel(
                f"[bold magenta]🚀 INCREDIBLE: {STREAK_50}-Check Streak! 🚀[/bold magenta]\n\n"
                "Your agent is rock solid.\n"
                "This deserves recognition! 🎖️",
                border_style="magenta"
            ))
            console.print()
            Celebrations.shareable_badge(streak)

        elif streak % 10 == 0 and streak > STREAK_50:
            console.print(f"[cyan]🚀 {streak}-check streak! Legendary stability.[/cyan]\n")

    @staticmethod
    def streak_broken(state: ProjectState, diff_status: str) -> None:
        """Empathetic message when streak breaks.

        Args:
            state: Current project state (with old streak)
            diff_status: Status that broke the streak
        """
        old_streak = state.current_streak

        if old_streak >= STREAK_BREAK_THRESHOLD:
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
        if _IS_DEMO:
            return
        console.print(Panel(
            "[yellow]⚠️  REGRESSION DETECTED[/yellow]\n\n"
            "Your agent's behavior changed.\n\n"
            "[bold]What changed?[/bold]\n"
            f"  {diff_summary}\n\n"
            "[bold]What to do:[/bold]\n"
            "  • If this change is intentional: [cyan]evalview snapshot[/cyan] to update baseline\n"
            "  • If this is a bug: fix it and [cyan]evalview check[/cyan] again",
            title="Regression Detected",
            border_style="yellow"
        ))
        console.print()

    @staticmethod
    def no_tests_found() -> None:
        """Helpful message when no tests found."""
        from evalview.cloud.auth import CloudAuth
        logged_in = CloudAuth().is_logged_in()

        if logged_in:
            body = (
                "No test cases found in [bold]tests/[/bold]\n\n"
                "Run [cyan]evalview init[/cyan] to create your first test —\n"
                "it will sync to your cloud account automatically."
            )
        else:
            body = (
                "No test cases found in [bold]tests/[/bold]\n\n"
                "[bold]Get started:[/bold]\n"
                "  [cyan]evalview init[/cyan]   scaffold a test for your agent\n"
                "  [cyan]evalview demo[/cyan]   see a live 30-second example"
            )

        console.print()
        console.print(Panel(body, title="No Tests Found", border_style="yellow"))
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
        if _IS_DEMO:
            return
        console.print(Panel(
            "You've completed the loop: [bold]snapshot → check[/bold] ✅\n\n"
            "Now make it automatic so you never have to remember:\n\n"
            "[bold]Pre-push hook[/bold] (catches it before you push):\n"
            "  [cyan]echo 'evalview check' >> .git/hooks/pre-push[/cyan]\n"
            "  [cyan]chmod +x .git/hooks/pre-push[/cyan]\n\n"
            "[bold]GitHub Actions[/bold] (blocks the PR if agent regresses):\n"
            "  [cyan]- run: evalview check --fail-on REGRESSION[/cyan]\n\n"
            "[dim]That's it. You'll never ship a broken agent by accident.[/dim]",
            title="First check done 🎯",
            border_style="cyan",
            padding=(1, 2),
        ))
        console.print()

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
            f"{health_emoji} [bold]Historical Check Health: {pass_rate:.0f}%[/bold]\n\n"
            "[dim]Based on your full local check history, not just this run.[/dim]\n\n"
            f"  Total checks: {total}\n"
            f"  Clean: {clean}\n"
            f"  Regressions: {state.regression_count}\n"
            f"  Current streak: {state.current_streak}\n"
            f"  Best streak: {state.longest_streak}",
            title="History Summary",
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
