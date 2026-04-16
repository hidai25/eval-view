"""Skill commands — validate, list, doctor, test, generate-tests."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import click
from rich.panel import Panel
from rich.table import Table

from evalview.commands.skill_agent_runner import run_agent_skill_test
from evalview.commands.shared import console
from evalview.skills.constants import (
    AVG_CHARS_PER_SKILL,
    CHAR_BUDGET_CRITICAL_PCT,
    CHAR_BUDGET_WARNING_PCT,
    CLAUDE_CODE_CHAR_BUDGET,
    MAX_DESCRIPTION_LENGTH,
    SCORE_THRESHOLD_HIGH,
    SCORE_THRESHOLD_MEDIUM,
    TRUNCATE_OUTPUT_LONG,
    TRUNCATE_OUTPUT_MEDIUM,
    TRUNCATE_OUTPUT_SHORT,
)
from evalview.skills.ui_utils import print_evalview_banner
from evalview.telemetry.decorators import track_command


@click.group("skill")
def skill():
    """Commands for testing Claude Code skills."""
    pass


# ---------------------------------------------------------------------------
# skill validate
# ---------------------------------------------------------------------------

@skill.command("validate")
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Search subdirectories for SKILL.md files")
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
@click.option("--verbose", "-v", is_flag=True, help="Show INFO suggestions")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@track_command("skill_validate", lambda **kw: {"strict": kw.get("strict"), "recursive": kw.get("recursive")})
def skill_validate(path: str, recursive: bool, strict: bool, verbose: bool, output_json: bool) -> None:
    """Validate Claude Code skill(s).

    Validates SKILL.md files for:
    - Correct structure and frontmatter
    - Valid naming conventions
    - Policy compliance
    - Best practices

    Examples:
        evalview skill validate ./my-skill/SKILL.md
        evalview skill validate ./skills/ --recursive
        evalview skill validate ./SKILL.md --strict
        evalview skill validate ./skills/ -rv  # verbose with suggestions
    """
    from evalview.skills import SkillValidator, SkillParser

    path_obj = Path(path)

    if path_obj.is_file():
        files = [str(path_obj)]
    else:
        files = SkillParser.find_skills(str(path_obj), recursive=recursive)
        if not files:
            if output_json:
                console.print(json.dumps({"error": "No SKILL.md files found", "files": []}))
            else:
                console.print(f"[yellow]No SKILL.md files found in {path}[/yellow]")
                if not recursive:
                    console.print("[dim]Tip: Use --recursive to search subdirectories[/dim]")
            return

    start_time = time.time()

    results = {}
    total_errors = 0
    total_warnings = 0
    total_valid = 0

    for file_path in files:
        result = SkillValidator.validate_file(file_path)
        results[file_path] = result

        total_errors += len(result.errors)
        total_warnings += len(result.warnings)
        if result.valid:
            total_valid += 1

    elapsed_ms = (time.time() - start_time) * 1000

    if output_json:
        json_output = {
            "summary": {
                "total_files": len(files),
                "valid": total_valid,
                "invalid": len(files) - total_valid,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
            },
            "results": {
                fp: {
                    "valid": r.valid,
                    "errors": [e.model_dump() for e in r.errors],
                    "warnings": [w.model_dump() for w in r.warnings],
                    "info": [i.model_dump() for i in r.info],
                }
                for fp, r in results.items()
            },
        }
        console.print(json.dumps(json_output, indent=2))
        return

    print_evalview_banner(console, subtitle="[dim]Catch agent regressions before you ship[/dim]")
    console.print("[dim]Validating against official Anthropic spec...[/dim]")
    console.print()

    for file_path, result in results.items():
        status_icon = "[green]✓[/green]" if result.valid else "[red]✗[/red]"
        console.print(f"{status_icon} [bold]{file_path}[/bold]")

        if result.skill:
            console.print(f"   [dim]Name: {result.skill.metadata.name}[/dim]")
            console.print(f"   [dim]Tokens: ~{result.skill.token_estimate}[/dim]")

        for error in result.errors:
            console.print(f"   [red]ERROR[/red] [{error.code}] {error.message}")
            if error.suggestion:
                console.print(f"         [dim]→ {error.suggestion}[/dim]")

        for warning in result.warnings:
            console.print(f"   [yellow]WARN[/yellow]  [{warning.code}] {warning.message}")
            if warning.suggestion:
                console.print(f"         [dim]→ {warning.suggestion}[/dim]")

        if verbose:
            for info in result.info:
                console.print(f"   [blue]INFO[/blue]  [{info.code}] {info.message}")
                if info.suggestion:
                    console.print(f"         [dim]→ {info.suggestion}[/dim]")

        console.print()

    console.print("[bold]Summary:[/bold]")
    console.print(f"  Files:    {len(files)}")
    console.print(f"  Valid:    [green]{total_valid}[/green]")
    console.print(f"  Invalid:  [red]{len(files) - total_valid}[/red]")
    console.print(f"  Errors:   [red]{total_errors}[/red]")
    console.print(f"  Warnings: [yellow]{total_warnings}[/yellow]")
    console.print(f"  Time:     [dim]{elapsed_ms:.0f}ms[/dim]")
    console.print()

    if total_errors > 0 or (strict and total_warnings > 0):
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# skill list
# ---------------------------------------------------------------------------

@skill.command("list")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--recursive", "-r", is_flag=True, default=True, help="Search subdirectories")
@track_command("skill_list")
def skill_list(path: str, recursive: bool) -> None:
    """List all skills in a directory.

    Examples:
        evalview skill list
        evalview skill list ./my-skills/
        evalview skill list ~/.claude/skills/
    """
    from evalview.skills import SkillParser, SkillValidator

    files = SkillParser.find_skills(path, recursive=recursive)

    if not files:
        console.print(f"[yellow]No SKILL.md files found in {path}[/yellow]")
        return

    console.print(f"\n[bold cyan]━━━ Skills in {path} ━━━[/bold cyan]\n")

    for file_path in files:
        result = SkillValidator.validate_file(file_path)
        status = "[green]✓[/green]" if result.valid else "[red]✗[/red]"

        if result.skill:
            console.print(f"  {status} [bold]{result.skill.metadata.name}[/bold]")
            desc = result.skill.metadata.description
            if len(desc) > MAX_DESCRIPTION_LENGTH:
                console.print(f"      [dim]{desc[:MAX_DESCRIPTION_LENGTH]}...[/dim]")
            else:
                console.print(f"      [dim]{desc}[/dim]")
            console.print(f"      [dim]{file_path}[/dim]")
        else:
            console.print(f"  {status} [red]{file_path}[/red]")
            if result.errors:
                console.print(f"      [red]{result.errors[0].message}[/red]")

        console.print()

    console.print(f"[dim]Total: {len(files)} skill(s)[/dim]\n")


# ---------------------------------------------------------------------------
# skill doctor
# ---------------------------------------------------------------------------

@skill.command("doctor")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--recursive", "-r", is_flag=True, default=True, help="Search subdirectories")
@click.option(
    "--security-scan", "-s", is_flag=True, default=False,
    help="LLM-powered scan for harmful/malicious instructions",
)
@track_command("skill_doctor")
def skill_doctor(path: str, recursive: bool, security_scan: bool) -> None:
    """Diagnose skill issues that cause Claude Code problems.

    Checks for common issues:
    - Total description chars exceeding Claude Code's 15k budget
    - Duplicate skill names
    - Invalid skills
    - Multi-line descriptions that break formatters

    Examples:
        evalview skill doctor ~/.claude/skills/
        evalview skill doctor .claude/skills/
        evalview skill doctor ./my-skills/ -r
    """
    from evalview.skills import SkillParser, SkillValidator

    start_time = time.time()

    files = SkillParser.find_skills(path, recursive=recursive)

    if not files:
        console.print(f"[yellow]No SKILL.md files found in {path}[/yellow]\n")
        console.print("[bold white]Here's what skill doctor catches:[/bold white]\n")
        console.print(
            Panel(
                "[bold red]⚠️  Character Budget: 127% OVER[/bold red]\n"
                "[red]Claude is ignoring ~4 of your 24 skills[/red]\n\n"
                "[red]✗[/red] my-claude-helper [dim]- reserved word \"claude\" in name[/dim]\n"
                "[red]✗[/red] api-tools [dim]- multiline description (breaks with Prettier)[/dim]\n"
                "[red]✗[/red] code-review [dim]- description too long (1847 chars)[/dim]\n"
                "[green]✓[/green] git-commit [dim]- OK[/dim]\n"
                "[green]✓[/green] test-runner [dim]- OK[/dim]",
                title="[bold]Example Output[/bold]",
                border_style="dim",
            )
        )
        console.print("\n[dim]Create skills in .claude/skills/ or ~/.claude/skills/[/dim]")
        return

    skills_data = []
    total_desc_chars = 0
    names_seen: dict = {}
    invalid_count = 0
    multiline_count = 0
    manual_only_count = 0

    for file_path in files:
        result = SkillValidator.validate_file(file_path)
        if result.valid and result.skill:
            name = result.skill.metadata.name
            desc = result.skill.metadata.description
            desc_len = len(desc)
            # Manual-only skills (disable-model-invocation: true) are not loaded
            # into Claude's skill-description context, so they don't consume the
            # 15k char budget. Still count them for duplicates/validation.
            manual_only = bool(result.skill.metadata.disable_model_invocation)
            if manual_only:
                manual_only_count += 1
            else:
                total_desc_chars += desc_len

            if name in names_seen:
                names_seen[name].append(file_path)
            else:
                names_seen[name] = [file_path]

            if "\n" in desc:
                multiline_count += 1

            skills_data.append({
                "name": name,
                "path": file_path,
                "desc_chars": desc_len,
                "valid": True,
                "manual_only": manual_only,
            })
        else:
            invalid_count += 1
            skills_data.append({
                "name": "INVALID",
                "path": file_path,
                "desc_chars": 0,
                "valid": False,
                "manual_only": False,
                "error": result.errors[0].message if result.errors else "Unknown error",
            })

    elapsed_ms = (time.time() - start_time) * 1000

    duplicates = {name: paths for name, paths in names_seen.items() if len(paths) > 1}

    print_evalview_banner(console, subtitle="[dim]Skill Doctor - Diagnose Claude Code Issues[/dim]")

    budget_pct = (total_desc_chars / CLAUDE_CODE_CHAR_BUDGET) * 100
    skills_over = max(0, int((total_desc_chars - CLAUDE_CODE_CHAR_BUDGET) / AVG_CHARS_PER_SKILL))
    # Only model-invokable skills can be "ignored" by a budget overflow.
    model_invokable_count = max(0, (len(files) - invalid_count) - manual_only_count)

    if budget_pct > CHAR_BUDGET_CRITICAL_PCT:
        console.print(
            f"[bold red]⚠️  Character Budget: {budget_pct:.0f}% OVER - "
            f"Claude is ignoring ~{skills_over} of your {model_invokable_count} "
            f"model-invokable skills[/bold red]"
        )
    elif budget_pct > CHAR_BUDGET_WARNING_PCT:
        console.print(f"[bold yellow]⚠️  Character Budget: {budget_pct:.0f}% - approaching limit[/bold yellow]")
    else:
        console.print(
            f"[bold green]✓ Character Budget: {budget_pct:.0f}% "
            f"({total_desc_chars:,} / {CLAUDE_CODE_CHAR_BUDGET:,} chars)[/bold green]"
        )
    console.print(f"[bold]Total Skills:[/bold]      {len(files)}")
    console.print(f"[bold]Valid:[/bold]             [green]{len(files) - invalid_count}[/green]")
    console.print(f"[bold]Invalid:[/bold]           [red]{invalid_count}[/red]")
    if manual_only_count:
        console.print(
            f"[bold]Manual-only:[/bold]      [cyan]{manual_only_count}[/cyan] "
            "[dim](disable-model-invocation — excluded from budget)[/dim]"
        )
    dup_color = "red" if duplicates else "green"
    console.print(f"[bold]Duplicates:[/bold]        [{dup_color}]{len(duplicates)}[/{dup_color}]")
    ml_color = "yellow" if multiline_count else "green"
    console.print(f"[bold]Multi-line Desc:[/bold]   [{ml_color}]{multiline_count}[/{ml_color}]")
    console.print()

    has_issues = False

    if budget_pct > CHAR_BUDGET_CRITICAL_PCT:
        has_issues = True
        console.print("[bold red]ISSUE: Character budget exceeded[/bold red]")
        console.print("  Claude Code won't see all your skills.")
        console.print(
            f"  [dim]Fix: Set SLASH_COMMAND_TOOL_CHAR_BUDGET={CLAUDE_CODE_CHAR_BUDGET * 2} "
            "or reduce descriptions[/dim]"
        )
        console.print()

    if duplicates:
        has_issues = True
        console.print("[bold red]ISSUE: Duplicate skill names[/bold red]")
        for name, paths in duplicates.items():
            console.print(f"  [yellow]{name}[/yellow] defined in:")
            for p in paths:
                console.print(f"    - {p}")
        console.print()

    if invalid_count > 0:
        has_issues = True
        console.print("[bold red]ISSUE: Invalid skills[/bold red]")
        for s in skills_data:
            if not s["valid"]:
                console.print(f"  [red]✗[/red] {s['path']}")
                console.print(f"    [dim]{s.get('error', 'Unknown error')}[/dim]")
        console.print()

    if multiline_count > 0:
        console.print("[bold yellow]WARNING: Multi-line descriptions[/bold yellow]")
        console.print("  These may break with Prettier or YAML formatters.")
        console.print("  [dim]Fix: Use single-line descriptions[/dim]")
        console.print()

    if not has_issues and multiline_count == 0:
        console.print("[bold green]✓ All skills visible to Claude[/bold green]")
    elif not has_issues:
        console.print("[bold yellow]⚠ Minor warnings - skills should work[/bold yellow]")
    else:
        invisible_count = skills_over + invalid_count + len(duplicates)
        if invisible_count > 0:
            console.print(
                f"[bold red]✗ {invisible_count} skill(s) are INVISIBLE to Claude - fix now[/bold red]"
            )
        else:
            console.print("[bold red]✗ Issues found - fix before deploying[/bold red]")

    if security_scan:
        console.print()
        console.print("[bold]Security Scan[/bold]  [dim](LLM-powered — semantic analysis)[/dim]")
        console.print()

        try:
            from evalview.skills.security_scanner import SkillSecurityScanner
            from evalview.skills import SkillParser as _SkillParser

            scanner = SkillSecurityScanner()
            security_issues = 0

            for s in skills_data:
                if not s["valid"]:
                    continue
                skill_obj = _SkillParser.parse_file(s["path"])
                scan_result = scanner.scan(skill_obj)
                color = scan_result.verdict_color
                icon = scan_result.verdict_icon

                console.print(
                    f"  [{color}]{icon}[/{color}] [bold]{s['name']}[/bold]  "
                    f"[{color}]{scan_result.verdict}[/{color}]  "
                    f"[dim]{scan_result.confidence}% confidence — {scan_result.summary}[/dim]"
                )

                for finding in scan_result.findings:
                    sev_color = (
                        "red" if finding.severity == "high"
                        else "yellow" if finding.severity == "medium"
                        else "dim"
                    )
                    console.print(
                        f"      [{sev_color}]↳ [{finding.severity.upper()}] "
                        f"{finding.category}: {finding.description}[/{sev_color}]"
                    )

                if scan_result.error:
                    console.print(f"      [dim red]scan error: {scan_result.error}[/dim red]")

                if scan_result.verdict != "SAFE":
                    security_issues += 1

            console.print()
            if security_issues == 0:
                console.print("[bold green]✓ No security issues found[/bold green]")
            else:
                console.print(
                    f"[bold red]✗ {security_issues} skill(s) flagged — review before trusting[/bold red]"
                )

        except Exception as e:
            console.print(f"[yellow]Security scan unavailable: {e}[/yellow]")
            console.print("[dim]Set OPENAI_API_KEY or another provider key to enable[/dim]")

    console.print(f"\n[dim]Time: {elapsed_ms:.0f}ms[/dim]\n")


# ---------------------------------------------------------------------------
# skill test
# ---------------------------------------------------------------------------

@skill.command("test")
@click.argument("test_file", type=click.Path(exists=True))
@click.option(
    "--model", "-m", default=None,
    help="Model to use (auto-selected per provider from central config)",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "openai-compatible"]),
    default=None,
    help="Legacy mode provider override (env alternative: SKILL_TEST_PROVIDER)",
)
@click.option(
    "--base-url", default=None,
    help="Legacy mode OpenAI-compatible base URL override (env alternative: SKILL_TEST_BASE_URL)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--agent", "-a",
    type=click.Choice([
        "system-prompt", "claude-code", "codex", "openclaw",
        "langgraph", "crewai", "openai-assistants", "custom",
    ]),
    default=None,
    help="Agent type (overrides YAML). Default: system-prompt (legacy mode)",
)
@click.option(
    "--trace", "-t", type=click.Path(), default=None,
    help="Directory to save JSONL traces for debugging",
)
@click.option("--no-rubric", is_flag=True, help="Skip Phase 2 rubric evaluation (deterministic only)")
@click.option("--cwd", type=click.Path(exists=True), default=None, help="Working directory for agent execution")
@click.option("--max-turns", type=int, default=None, help="Maximum conversation turns (default: 10)")
@track_command("skill_test", lambda **kw: {"agent": kw.get("agent"), "no_rubric": kw.get("no_rubric")})
def skill_test(
    test_file: str,
    model: str,
    provider: str,
    base_url: str,
    verbose: bool,
    output_json: bool,
    agent: str,
    trace: str,
    no_rubric: bool,
    cwd: str,
    max_turns: int,
) -> None:
    """Run behavior tests against a skill.

    TEST_FILE is a YAML file defining test cases for a skill.

    Legacy mode (system prompt + output matching):

      evalview skill test tests/code-reviewer.yaml

    Agent mode with Claude Code:

      evalview skill test tests/my-skill.yaml --agent claude-code

    Save traces for debugging:

      evalview skill test tests/my-skill.yaml -a claude-code -t ./traces/

    Deterministic checks only (no LLM judge cost):

      evalview skill test tests/my-skill.yaml -a claude-code --no-rubric
    """
    import yaml as yaml_module

    with open(test_file) as f:
        yaml_data = yaml_module.safe_load(f)

    yaml_agent_type = None
    if "agent" in yaml_data and yaml_data["agent"]:
        yaml_agent_type = yaml_data["agent"].get("type")

    use_agent_mode = (
        (agent is not None and agent != "system-prompt")
        or (yaml_agent_type is not None and yaml_agent_type != "system-prompt")
    )

    if use_agent_mode:
        run_agent_skill_test(
            test_file=test_file,
            agent=agent,
            trace_dir=trace,
            no_rubric=no_rubric,
            cwd=cwd,
            max_turns=max_turns,
            verbose=verbose,
            output_json=output_json,
            model=model,
        )
        return

    # Legacy mode
    from evalview.skills import SkillRunner

    try:
        runner = SkillRunner(model=model, provider=provider, base_url=base_url)
        suite = runner.load_test_suite(test_file)
    except Exception as e:
        if "API key" in str(e) or "base URL" in str(e) or "provider" in str(e).lower():
            console.print(f"[red]Provider configuration error: {e}[/red]")
            console.print(
                "[dim]Use --provider/--base-url or set env vars such as "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, SKILL_TEST_BASE_URL[/dim]"
            )
        else:
            console.print(f"[red]Error loading test suite: {e}[/red]")
        raise SystemExit(1)

    print_evalview_banner(console, subtitle="[dim]Catch agent regressions before you ship[/dim]")
    console.print(f"  [bold]Suite:[/bold]  {suite.name}")
    console.print(f"  [bold]Skill:[/bold]  [cyan]{suite.skill}[/cyan]")
    console.print(f"  [bold]Model:[/bold]  {runner.model}")
    console.print(f"  [bold]Provider:[/bold]  {getattr(runner, 'provider', 'unknown')}")
    if getattr(runner, "provider", None) == "openai":
        console.print(f"  [bold]Base URL:[/bold]  {getattr(runner, 'base_url', None) or '[dim]default[/dim]'}")
    console.print(f"  [bold]Tests:[/bold]  {len(suite.tests)}")
    console.print()

    start_time = time.time()
    total_tests = len(suite.tests)
    completed_count = [0]

    console.print(f"[cyan]Running {total_tests} tests in parallel...[/cyan]\n")

    def on_test_complete(test_result: Any) -> None:
        completed_count[0] += 1
        icon = "[green]✓[/green]" if test_result.passed else "[red]✗[/red]"
        score_str = f"[dim]{test_result.score:.0f}%[/dim]"
        latency_str = f"[dim]{test_result.latency_ms / 1000:.1f}s[/dim]"
        console.print(
            f"  {icon} [{completed_count[0]}/{total_tests}] "
            f"[bold]{test_result.test_name}[/bold]  {score_str}  {latency_str}"
        )

    run_error = None
    result = None
    try:
        result = runner.run_suite(suite, on_test_complete=on_test_complete)
    except Exception as exc:
        run_error = exc

    console.print()

    if run_error:
        console.print(f"[red]Error running tests: {run_error}[/red]")
        raise SystemExit(1)

    assert result is not None  # guarded by the SystemExit above
    elapsed_ms = (time.time() - start_time) * 1000

    if output_json:
        json_output = {
            "suite_name": result.suite_name,
            "skill_name": result.skill_name,
            "passed": result.passed,
            "total_tests": result.total_tests,
            "passed_tests": result.passed_tests,
            "failed_tests": result.failed_tests,
            "pass_rate": result.pass_rate,
            "total_latency_ms": result.total_latency_ms,
            "avg_latency_ms": result.avg_latency_ms,
            "total_tokens": result.total_tokens,
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "score": r.score,
                    "input": r.input_query,
                    "output": (
                        r.output[:TRUNCATE_OUTPUT_LONG] + "..."
                        if len(r.output) > TRUNCATE_OUTPUT_LONG
                        else r.output
                    ),
                    "contains_failed": r.contains_failed,
                    "not_contains_failed": r.not_contains_failed,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in result.results
            ],
        }
        console.print(json.dumps(json_output, indent=2))
        return

    table = Table(title="Test Results", show_header=True, header_style="bold cyan")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Test", style="cyan")
    table.add_column("Score", justify="right", width=8)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Tokens", justify="right", width=8)

    for r in result.results:
        status = "[green]✅ PASS[/green]" if r.passed else "[red]❌ FAIL[/red]"
        score_color = (
            "green" if r.score >= SCORE_THRESHOLD_HIGH
            else "yellow" if r.score >= SCORE_THRESHOLD_MEDIUM
            else "red"
        )
        table.add_row(
            status,
            r.test_name,
            f"[{score_color}]{r.score:.0f}%[/{score_color}]",
            f"{r.latency_ms:.0f}ms",
            f"{r.input_tokens + r.output_tokens:,}",
        )

    console.print(table)
    console.print()

    failed_results = [r for r in result.results if not r.passed]
    show_results = result.results if verbose else failed_results

    if show_results:
        for r in show_results:
            status_icon = "✅" if r.passed else "❌"
            status_color = "green" if r.passed else "red"

            console.print(f"[bold {status_color}]{status_icon} {r.test_name}[/bold {status_color}]")

            console.print("\n[bold]Input:[/bold]")
            input_q = (
                r.input_query[:TRUNCATE_OUTPUT_SHORT] + "..."
                if len(r.input_query) > TRUNCATE_OUTPUT_SHORT
                else r.input_query
            )
            for line in input_q.split("\n"):
                console.print(f"  [dim]{line}[/dim]")

            if verbose or not r.passed:
                console.print("\n[bold]Response:[/bold]")
                output = (
                    r.output[:TRUNCATE_OUTPUT_MEDIUM] + "..."
                    if len(r.output) > TRUNCATE_OUTPUT_MEDIUM
                    else r.output
                )
                for line in output.split("\n")[:8]:
                    console.print(f"  {line}")
                if len(r.output.split("\n")) > 8:
                    console.print("  [dim]...[/dim]")

            console.print("\n[bold]Evaluation Checks:[/bold]")

            if r.contains_passed:
                for phrase in r.contains_passed:
                    console.print(f'  [green]✓[/green] Contains: "{phrase}"')
            if r.contains_failed:
                for phrase in r.contains_failed:
                    console.print(f'  [red]✗[/red] Missing:  "{phrase}"')

            if r.not_contains_passed:
                for phrase in r.not_contains_passed:
                    console.print(f'  [green]✓[/green] Excludes: "{phrase}"')
            if r.not_contains_failed:
                for phrase in r.not_contains_failed:
                    console.print(f'  [red]✗[/red] Found:    "{phrase}" (should not appear)')

            if r.error:
                console.print(f"\n[bold red]Error:[/bold red] {r.error}")

            if not r.passed:
                console.print("\n[bold yellow]How to Fix:[/bold yellow]")
                if r.contains_failed:
                    console.print("  [yellow]• Your skill's instructions should guide Claude to mention:[/yellow]")
                    for phrase in r.contains_failed:
                        console.print(f'    [yellow]  - "{phrase}"[/yellow]')
                    console.print("  [yellow]• Consider adding explicit guidance in your SKILL.md[/yellow]")
                if r.not_contains_failed:
                    console.print("  [yellow]• Your skill is producing unwanted phrases:[/yellow]")
                    for phrase in r.not_contains_failed:
                        console.print(f'    [yellow]  - "{phrase}"[/yellow]')
                    console.print("  [yellow]• Add constraints or negative examples to your SKILL.md[/yellow]")
                if r.error:
                    console.print("  [yellow]• Check your API key and model availability[/yellow]")

            console.print()

    pass_rate_color = (
        "green" if result.pass_rate >= 0.8
        else "yellow" if result.pass_rate >= 0.5
        else "red"
    )
    status_text = (
        "[green]● All Tests Passed[/green]"
        if result.passed
        else "[bold red]● Some Tests Failed[/bold red]"
    )
    border_color = "green" if result.passed else "red"

    summary_content = (
        f"  {status_text}\n"
        f"\n"
        f"  [bold]✅ Passed:[/bold]       [green]{result.passed_tests}[/green]\n"
        f"  [bold]❌ Failed:[/bold]       [red]{result.failed_tests}[/red]\n"
        f"  [bold]📈 Pass Rate:[/bold]    [{pass_rate_color}]{result.pass_rate:.0%}[/{pass_rate_color}]"
        f" (required: {suite.min_pass_rate:.0%})\n"
        f"\n"
        f"  [bold]⏱️  Avg Latency:[/bold] {result.avg_latency_ms:.0f}ms\n"
        f"  [bold]🔤 Total Tokens:[/bold] {result.total_tokens:,}\n"
        f"  [bold]⏲️  Total Time:[/bold]  {elapsed_ms:.0f}ms"
    )

    console.print(Panel(summary_content, title="[bold]Overall Statistics[/bold]", border_style=border_color))

    if not result.passed:
        console.print()
        console.print("[bold yellow]⚠️  Skill Test Failed[/bold yellow]")
        console.print()
        console.print("[bold]Next Steps to Fix Your Skill:[/bold]")
        console.print("  1. Review the [bold]How to Fix[/bold] guidance above for each failed test")
        console.print("  2. Update your [cyan]SKILL.md[/cyan] instructions to address the issues")
        console.print(f"  3. Re-run: [dim]evalview skill test {test_file}[/dim]")
        console.print()
        console.print("[dim]Tip: Use --verbose to see full responses for passing tests too[/dim]")
        console.print()
        raise SystemExit(1)
    else:
        console.print()
        console.print("[bold green]✓ Skill ready for deployment[/bold green]")
        console.print()


# ---------------------------------------------------------------------------
# skill generate-tests
# ---------------------------------------------------------------------------

@skill.command("generate-tests")
@click.argument("skill_file", type=click.Path(exists=True))
@click.option("--count", "-c", default=10, type=int, help="Number of tests to generate")
@click.option("--output", "-o", type=click.Path(), help="Output path for tests.yaml (default: ./tests.yaml)")
@click.option(
    "--categories", type=str,
    help="Comma-separated test categories (explicit,implicit,contextual,negative)",
)
@click.option("--model", "-m", type=str, help="LLM model to use for generation (skips interactive selection)")
@click.option("--auto", is_flag=True, help="Auto-select cheapest model (skip interactive selection)")
@click.option("--dry-run", is_flag=True, help="Preview generated tests without saving")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed generation process")
@track_command("skill_generate_tests")
def skill_generate_tests(
    skill_file: str,
    count: int,
    output: Optional[str],
    categories: Optional[str],
    model: Optional[str],
    auto: bool,
    dry_run: bool,
    verbose: bool,
):
    """Auto-generate comprehensive test suites from SKILL.md.

    Uses LLM-powered few-shot learning to generate test cases across
    all categories: explicit, implicit, contextual, and negative.

    By default, shows an interactive model selector with cost estimates.
    Use --auto to skip selection and use the cheapest available model.

    Examples:
        evalview skill generate-tests ./SKILL.md
        evalview skill generate-tests ./SKILL.md --auto --count 15
        evalview skill generate-tests ./SKILL.md --model gpt-4o --dry-run
        evalview skill generate-tests ./SKILL.md -o my-tests.yaml
    """
    asyncio.run(
        _skill_generate_tests_async(skill_file, count, output, categories, model, auto, dry_run, verbose)
    )


async def _skill_generate_tests_async(
    skill_file: str,
    count: int,
    output: Optional[str],
    categories: Optional[str],
    model: Optional[str],
    auto: bool,
    dry_run: bool,
    verbose: bool,
):
    import sys
    import yaml as yaml_module

    from evalview.skills.test_generator import SkillTestGenerator
    from evalview.skills.parser import SkillParser
    from evalview.skills.agent_types import TestCategory
    from evalview.telemetry.client import track
    from evalview.telemetry.events import (
        SkillTestGenerationStartEvent,
        SkillTestGenerationCompleteEvent,
        SkillTestGenerationFailedEvent,
        UserFeedbackEvent,
    )

    start_time = time.time()
    print_evalview_banner(console, subtitle="[dim]Auto-generate comprehensive test suites[/dim]")

    category_list = None
    if categories:
        cat_names = [c.strip().lower() for c in categories.split(",")]
        category_list = []
        for cat_name in cat_names:
            try:
                category_list.append(TestCategory(cat_name))
            except ValueError:
                console.print(f"[red]❌ Invalid category: {cat_name}[/red]")
                console.print("[dim]Valid categories: explicit, implicit, contextual, negative[/dim]")
                raise SystemExit(1)

    console.print("[bold]Parsing skill file...[/bold]")
    try:
        skill = SkillParser.parse_file(skill_file)
    except Exception as e:
        console.print(f"[red]❌ Failed to parse skill: {e}[/red]")
        raise SystemExit(1)

    console.print(f"[green]✓[/green] Loaded skill: [bold]{skill.metadata.name}[/bold]")
    console.print()

    try:
        is_interactive = sys.stdin.isatty() and sys.stdout.isatty()

        if model:
            generator = SkillTestGenerator(model=model)
            if verbose:
                console.print(f"[dim]Using model: {model}[/dim]")
                console.print()
        elif auto or not is_interactive:
            if not is_interactive and not auto:
                console.print("[yellow]⚠️  Non-interactive environment detected (CI/CD, Docker, etc.)[/yellow]")
                console.print("[dim]Auto-selecting cheapest model...[/dim]")
                console.print()

            generator = SkillTestGenerator()
            console.print(
                f"[dim]Auto-selected: {generator.client.config.display_name} / "
                f"{generator.client.config.default_model}[/dim]"
            )
            console.print()
        else:
            provider, api_key, selected_model = SkillTestGenerator.select_model_interactive(console)
            generator = SkillTestGenerator(model=selected_model)
            console.print()

    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        console.print()
        console.print("[bold]To fix this:[/bold]")
        console.print("  1. Set an API key:")
        console.print("     export OPENAI_API_KEY=sk-...")
        console.print("     export ANTHROPIC_API_KEY=sk-ant-...")
        console.print("     export GEMINI_API_KEY=...")
        console.print("     export DEEPSEEK_API_KEY=...")
        console.print()
        raise SystemExit(1)

    track(
        SkillTestGenerationStartEvent(
            skill_name=skill.metadata.name,
            test_count=count,
            categories=[c.value for c in (category_list or [])] if category_list else [],
            model=generator.client.config.default_model,
            has_example_suite=False,
        )
    )

    console.print(
        f"[bold]Generating {count} tests for [cyan]{skill.metadata.name}[/cyan]...[/bold]"
    )
    console.print()

    try:
        with console.status("[bold green]Generating tests..."):
            suite = await generator.generate_test_suite(skill=skill, count=count, categories=category_list)

        generation_time_ms = int((time.time() - start_time) * 1000)

        validation_errors = generator.validate_test_suite(suite)
        if validation_errors:
            console.print("[yellow]⚠️  Validation warnings:[/yellow]")
            for error in validation_errors:
                console.print(f"  - {error}")
            console.print()

        track(
            SkillTestGenerationCompleteEvent(
                skill_name=skill.metadata.name,
                tests_generated=len(suite.tests),
                generation_latency_ms=generation_time_ms,
                estimated_cost_usd=generator.get_generation_cost(),
                model=generator.client.config.default_model,
                validation_errors=len(validation_errors),
                categories_distribution=generator.get_category_distribution(suite),
            )
        )

    except Exception as e:
        track(
            SkillTestGenerationFailedEvent(
                skill_name=skill.metadata.name,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                model=generator.client.config.default_model,
                attempt_number=1,
            )
        )
        console.print(f"[red]❌ Generation failed: {e}[/red]")
        raise SystemExit(1)

    console.print("[bold green]✓ Generated test suite[/bold green]")
    console.print()

    console.print("[bold]Test Suite Summary[/bold]")
    console.print(f"  Name: [cyan]{suite.name}[/cyan]")
    console.print(f"  Tests: {len(suite.tests)}")
    console.print(f"  Estimated Cost: [green]~${generator.get_generation_cost():.4f}[/green]")
    console.print()

    dist = generator.get_category_distribution(suite)
    console.print("[bold]Category Distribution[/bold]")
    for cat, cnt in dist.items():
        console.print(f"  {cat}: {cnt}")
    console.print()

    console.print("[bold]Generated Tests[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Should Trigger", style="green")
    table.add_column("Assertions", justify="right")

    for test in suite.tests:
        assertion_count = 0
        if test.expected:
            assertion_count += bool(test.expected.tool_calls_contain)
            assertion_count += bool(test.expected.files_created)
            assertion_count += bool(test.expected.commands_ran)
            assertion_count += bool(test.expected.output_contains)
            assertion_count += bool(test.expected.output_not_contains)
        if test.rubric:
            assertion_count += 1

        trigger_emoji = "✓" if test.should_trigger else "✗"
        table.add_row(test.name, test.category.value, trigger_emoji, str(assertion_count))

    console.print(table)
    console.print()

    if dry_run:
        console.print("[yellow]Dry-run mode: Not saving to disk[/yellow]")
        console.print()
        if verbose:
            console.print("[bold]Full YAML preview:[/bold]")
            if suite.tests:
                preview = {
                    "name": suite.name,
                    "description": suite.description,
                    "skill": suite.skill,
                    "agent": {"type": suite.agent.type.value},
                    "tests": [generator._serialize_test(suite.tests[0])],
                }
                console.print(Panel(yaml_module.dump(preview, default_flow_style=False)))
                console.print(f"[dim]... and {len(suite.tests) - 1} more tests[/dim]")
        return

    output_path = Path(output) if output else Path.cwd() / "tests.yaml"

    non_interactive = auto or not sys.stdin.isatty()

    if output_path.exists() and not non_interactive:
        console.print(f"[yellow]⚠️  File already exists: {output_path}[/yellow]")
        if not click.confirm("Overwrite?", default=False):
            console.print("[yellow]Cancelled[/yellow]")
            return

    if not non_interactive:
        console.print(f"Save to: [cyan]{output_path}[/cyan]")
        if not click.confirm("Save generated tests?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        generator.save_as_yaml(suite, output_path)
        console.print()
        console.print(f"[bold green]✓ Saved to {output_path}[/bold green]")
        console.print()

        console.print("[bold]Next Steps:[/bold]")
        console.print(f"  1. Review the generated tests: [dim]cat {output_path}[/dim]")
        console.print(f"  2. Run the tests: [dim]evalview skill test {output_path}[/dim]")
        console.print(f"  3. Iterate on failing tests by editing {output_path}")
        console.print()

        if not non_interactive:
            try:
                rating = click.prompt(
                    "Rate this generation (1-5)",
                    type=click.IntRange(1, 5),
                    default=4,
                    show_default=True,
                )
                would_use_again = click.confirm("Would you use auto-generation again?", default=True)

                track(
                    UserFeedbackEvent(
                        skill_name=skill.metadata.name,
                        rating=rating,
                        would_use_again=would_use_again,
                        feedback_text=None,
                    )
                )
            except (KeyboardInterrupt, click.Abort):
                pass

    except Exception as e:
        console.print(f"[red]❌ Failed to save: {e}[/red]")
        raise SystemExit(1)
