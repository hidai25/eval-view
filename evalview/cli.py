"""CLI entry point for EvalView — thin orchestrator that wires command modules."""
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    _EVALVIEW_VERSION = _pkg_version("evalview")
except PackageNotFoundError:
    _EVALVIEW_VERSION = "dev"

import click

from evalview.commands.shared import console
from evalview.telemetry.config import (
    should_show_first_run_notice,
    mark_first_run_notice_shown,
)

# ── Command modules ──────────────────────────────────────────────────────────
from evalview.commands.run import run
from evalview.commands.listing_cmd import list_cmd, adapters, report, view, connect, validate_adapter, record
from evalview.commands.skill_cmd import skill
from evalview.commands.capture_cmd import capture
from evalview.commands.init_cmd import init, quickstart
from evalview.commands.add_cmd import add
from evalview.commands.demo_cmd import demo
from evalview.commands.judge_cmd import judge
from evalview.commands.expand_cmd import expand
from evalview.commands.trends_cmd import trends
from evalview.commands.golden_cmd import golden
from evalview.commands.telemetry_cmd import telemetry
from evalview.commands.ci_cmd import ci
from evalview.commands.gym_cmd import gym
from evalview.commands.cloud_cmd import login, logout, whoami
from evalview.commands.hooks_cmd import install_hooks, uninstall_hooks
from evalview.commands.import_cmd import import_logs
from evalview.commands.snapshot_cmd import snapshot
from evalview.commands.check_cmd import check, replay
from evalview.commands.benchmark_cmd import benchmark_cmd
from evalview.commands.mcp_cmd import mcp
from evalview.commands.visual_cmd import inspect_cmd, visualize_cmd, compare_cmd
from evalview.commands.chat_cmd import chat, trace_cmd
from evalview.commands.traces_cmd import traces
from evalview.commands.baseline_cmd import baseline


@click.group(context_settings={"allow_interspersed_args": False})
@click.version_option(version=_EVALVIEW_VERSION)
@click.pass_context
def main(ctx: click.Context) -> None:
    """EvalView — Proof that your agent still works.

    \b
    New here? Start with:
      capture --agent <url>   Record real traffic as tests  ← best first step
      snapshot                Save as your regression baseline
      check                   Catch regressions after every change

    \b
    Check Your Agent:
      run                     Run all tests with full evaluation
      run --diff              Compare against golden baseline
      demo                    See regression detection in action (~30 seconds)
      quickstart              Set up a working example in 2 minutes

    \b
    Golden Traces:
      golden save <file>      Save a known-good baseline
      golden list             List saved baselines
      golden show <name>      View baseline details

    \b
    Explore & Learn:
      chat                    Interactive AI assistant for eval guidance
      gym                     Practice agent eval patterns

    \b
    Reports:
      report <file>           Generate report from results
      view                    View specific test results
      trends                  Performance trends over time

    \b
    CI/CD:
      ci comment              Post results to a GitHub PR
      init --ci               Generate GitHub Actions workflow

    \b
    Advanced:
      skill                   Test Claude Code skills
      trace                   Trace LLM calls in scripts
      traces                  Query stored trace data
      expand                  Generate test variations with LLM
    """
    # Show first-run telemetry notice (once only)
    if should_show_first_run_notice():
        if ctx.invoked_subcommand not in ("telemetry",):
            console.print()
            console.print("[dim]╭──────────────────────────────────────────────────────────────╮[/dim]")
            console.print("[dim]│[/dim] EvalView collects anonymous usage data to improve the tool. [dim]│[/dim]")
            console.print("[dim]│[/dim] No personal info or test content is collected.              [dim]│[/dim]")
            console.print("[dim]│[/dim] Disable with: [cyan]evalview telemetry off[/cyan]                      [dim]│[/dim]")
            console.print("[dim]╰──────────────────────────────────────────────────────────────╯[/dim]")
            console.print()
            mark_first_run_notice_shown()


# ── Register commands ────────────────────────────────────────────────────────
main.add_command(run)
main.add_command(list_cmd, name="list")
main.add_command(adapters)
main.add_command(report)
main.add_command(view)
main.add_command(connect)
main.add_command(validate_adapter, name="validate-adapter")
main.add_command(record)
main.add_command(skill)
main.add_command(capture)
main.add_command(init)
main.add_command(quickstart)
main.add_command(add)
main.add_command(demo)
main.add_command(judge)
main.add_command(expand)
main.add_command(trends)
main.add_command(golden)
main.add_command(telemetry)
main.add_command(ci)
main.add_command(gym)
main.add_command(login)
main.add_command(logout)
main.add_command(whoami)
main.add_command(install_hooks, name="install-hooks")
main.add_command(uninstall_hooks, name="uninstall-hooks")
main.add_command(import_logs, name="import")
main.add_command(snapshot)
main.add_command(check)
main.add_command(replay)
main.add_command(benchmark_cmd, name="benchmark")
main.add_command(mcp)
main.add_command(inspect_cmd, name="inspect")
main.add_command(visualize_cmd, name="visualize")
main.add_command(compare_cmd, name="compare")
main.add_command(chat)
main.add_command(trace_cmd, name="trace")
main.add_command(traces)
main.add_command(baseline)


if __name__ == "__main__":
    main()
