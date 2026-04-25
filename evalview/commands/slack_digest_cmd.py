"""`evalview slack-digest` — post a team digest to Slack.

The social layer of the habit loop. This is an **optional** ritual —
solo devs never need it, teams live and die by it. When a team runs
`evalview slack-digest --webhook https://hooks.slack.com/…` at 3pm on
cron or from a scheduled job, everyone on the team sees the same
sentence about the day's agent health in the team channel.

Design rules (same as `since` brief):
  - One hero number (pass rate), one concern, one action
  - Under 2 seconds; no network calls except the Slack POST itself
  - Never crashes on a failed webhook — logs and exits cleanly so a
    broken Slack config never breaks CI that uses this command

Usage:
    evalview slack-digest --webhook <url>                # send now
    evalview slack-digest --webhook <url> --since 7d     # custom window
    evalview slack-digest --webhook <url> --dry-run      # preview, no post

Webhook URL can come from:
  - --webhook flag
  - EVALVIEW_SLACK_WEBHOOK env var
  - .evalview/config.yaml under monitor.slack_webhook (future)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from evalview.commands.shared import console
from evalview.commands.since_cmd import (
    _detect_drifting_tests,
    _entries_since,
    _load_history,
    _parse_since,
    _summarize,
)
from evalview.core.noise_tracker import NoiseStats, load_noise_stats
from evalview.telemetry.decorators import track_command


_HISTORY_PATH = Path(".evalview") / "history.jsonl"
_DEFAULT_TIMEOUT = 10.0  # seconds


# ───────────────────────── message builder ─────────────────────────


def _build_message(
    label: str,
    window: Dict[str, Any],
    drift_rows: List[Any],
    stale_quarantine: List[Dict[str, Any]],
    noise_stats: Optional[NoiseStats] = None,
) -> Dict[str, Any]:
    """Build a Slack `blocks` message matching the Slack block-kit schema.

    Pure function — no I/O, easy to unit-test. Returns the JSON payload
    ready to POST. We use blocks rather than plain markdown so the
    message renders cleanly on desktop, mobile, and Slack search.

    Args:
        noise_stats: Optional aggregated noise counters over the window.
            When present, a "Noise" section is rendered that publicly
            reports alerts fired vs. suppressed — the point is to hold
            ourselves accountable for false-positive rate rather than
            hide it. If the stats are empty (no alert activity in the
            window), the section is skipped so the digest stays tight.
    """
    pass_rate = window.get("pass_rate")
    total = window.get("total", 0)
    regression = window.get("regression", 0)
    tools_changed = window.get("tools_changed", 0)
    output_changed = window.get("output_changed", 0)

    if total == 0:
        headline = f"📊 *EvalView digest — {label}*"
        body = (
            "No runs in this window. "
            "Run `evalview check` to seed the next digest."
        )
        return {
            "text": f"{headline}\n{body}",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": headline}},
                {"type": "section", "text": {"type": "mrkdwn", "text": body}},
            ],
        }

    pct = int(round((pass_rate or 0) * 100))
    emoji = "🟢" if pct >= 95 else "🟡" if pct >= 80 else "🔴"
    headline = f"📊 *EvalView digest — {label}*"

    summary_parts: List[str] = [
        f"{emoji} *{pct}%* pass rate across *{total}* runs",
    ]
    if regression:
        summary_parts.append(f"❌ {regression} regression(s)")
    if tools_changed + output_changed:
        summary_parts.append(
            f"⚠️ {tools_changed + output_changed} soft change(s)"
        )

    summary_text = "\n".join(summary_parts)

    blocks: List[Dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": headline}},
        {"type": "section", "text": {"type": "mrkdwn", "text": summary_text}},
    ]

    # Drift block (top 3 concerning tests)
    if drift_rows:
        drift_lines = []
        for name, spark in drift_rows[:3]:
            drift_lines.append(f"`{spark}` {name}")
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Drift*\n" + "\n".join(drift_lines),
                },
            }
        )

    # Noise block — public false-positive rate. We render this before
    # stale quarantine because it's the meta-metric users should glance
    # at first; a noisy product is a distrusted product, and the digest
    # is the right place to hold ourselves visible to it.
    #
    # Critically, we ALSO render the list of tests that were silently
    # suppressed, so "suppression" never becomes "hidden signal". The
    # user can still see which tests self-resolved and decide whether
    # the gate was right or whether something deeper is wrong.
    if noise_stats is not None and (
        noise_stats.alerts_fired + noise_stats.suppressed > 0
    ):
        # false_positive_rate can only be None when both alerts_fired
        # and suppressed are zero — the outer guard already excluded
        # that case, so we can safely treat fpr as a float below.
        fpr = noise_stats.false_positive_rate
        assert fpr is not None, "outer guard ensures denom > 0"
        pct = int(round(fpr * 100))
        if pct <= 5:
            emoji_noise = "🟢"
        elif pct <= 20:
            emoji_noise = "🟡"
        else:
            emoji_noise = "🔴"
        noise_headline = (
            f"{emoji_noise} "
            f"{noise_stats.alerts_fired} fired · "
            f"{noise_stats.real_alerts} real · "
            f"{noise_stats.suppressed} suppressed "
            f"({pct}% noise)"
        )

        noise_text = f"*Noise*\n{noise_headline}"

        if noise_stats.suppressed_by_test:
            # Show the top-5 most-suppressed tests so the user can spot
            # patterns — a test that self-resolved 4 times this week is
            # no longer a flake, it's a signal. Keep the list bounded
            # so the digest doesn't become its own firehose.
            #
            # Escape backticks in test names before wrapping them in
            # inline code spans — an unescaped backtick would break the
            # span and leak the count annotation into prose. Same
            # markdown-safety rule the PR comment uses.
            top = noise_stats.suppressed_by_test[:5]
            lines = [
                "_Suppressed (self-resolved before confirming):_",
            ]
            for entry in top:
                safe_name = entry.test_name.replace("`", "&#96;")
                count_str = f"× {entry.count}" if entry.count > 1 else ""
                lines.append(f"• `{safe_name}` {count_str}".rstrip())
            if len(noise_stats.suppressed_by_test) > 5:
                extra = len(noise_stats.suppressed_by_test) - 5
                lines.append(f"…and {extra} more")
            noise_text += "\n" + "\n".join(lines)

        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": noise_text,
                },
            }
        )

    # Stale quarantine block
    if stale_quarantine:
        n = len(stale_quarantine)
        preview_lines = []
        for stale in stale_quarantine[:3]:
            owner = stale.get("owner") or "<unknown>"
            age = stale.get("age_days")
            age_str = f"{age}d" if age is not None else "?"
            preview_lines.append(f"• {stale.get('test_name')} — {owner} — {age_str}")
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*⏰ Stale quarantine*\n{n} overdue\n"
                        + "\n".join(preview_lines)
                    ),
                },
            }
        )

    # Observability signals — aggregated from history entries that carry
    # has_anomalies / trust_score / has_coherence_issues. These fields are
    # recorded by drift_tracker.record_check() since the observability
    # integration; older entries simply lack them.
    _anomaly_entries = [
        e for e in (window.get("_entries") or [])
        if e.get("has_anomalies")
    ]
    from evalview.core.observability import LOW_TRUST_THRESHOLD
    _low_trust_entries = [
        e for e in (window.get("_entries") or [])
        if e.get("trust_score") is not None and e.get("trust_score", 1.0) < LOW_TRUST_THRESHOLD
    ]
    _coherence_entries = [
        e for e in (window.get("_entries") or [])
        if e.get("has_coherence_issues")
    ]
    if _anomaly_entries or _low_trust_entries or _coherence_entries:
        obs_lines = []
        if _anomaly_entries:
            tests = sorted({e.get("test", "?") for e in _anomaly_entries})
            obs_lines.append(
                f"⚠️ *{len(_anomaly_entries)} check(s)* with behavioral anomalies "
                f"({', '.join(f'`{t}`' for t in tests[:3])})"
            )
        if _low_trust_entries:
            tests = sorted({e.get("test", "?") for e in _low_trust_entries})
            obs_lines.append(
                f"⚠️ *{len(_low_trust_entries)} check(s)* with low trust score "
                f"({', '.join(f'`{t}`' for t in tests[:3])})"
            )
        if _coherence_entries:
            tests = sorted({e.get("test", "?") for e in _coherence_entries})
            obs_lines.append(
                f"⚠️ *{len(_coherence_entries)} check(s)* with coherence issues "
                f"({', '.join(f'`{t}`' for t in tests[:3])})"
            )
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Agent behavior*\n" + "\n".join(obs_lines),
                },
            }
        )

    # Footer — one actionable next step
    action = _next_action(window, drift_rows, stale_quarantine)
    blocks.append(
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"🎯 *Next:* {action}"}],
        }
    )

    # Plain-text fallback for notifications + accessibility
    plain_text = f"{emoji} EvalView: {pct}% pass rate, {total} runs"
    if regression:
        plain_text += f", {regression} regressions"

    return {"text": plain_text, "blocks": blocks}


def _next_action(
    window: Dict[str, Any],
    drift_rows: List[Any],
    stale_quarantine: List[Any],
) -> str:
    """Pick one concrete command to recommend in the digest footer.

    Priority order matches the since-brief: regressions > drift > stale
    quarantine > clean state. The rule is "one action, never more than
    one" — a list of three things nobody does is worse than a single
    command nobody does.
    """
    if window.get("regression", 0) > 0:
        return "`evalview check --fail-on REGRESSION`"
    if drift_rows:
        name = drift_rows[0][0] if drift_rows else "<test>"
        return f"`evalview drift {name}`"
    if stale_quarantine:
        return "`evalview quarantine list --stale-only`"
    return "Keep shipping. Your agent is stable."


# ───────────────────────── transport ─────────────────────────


def _post_to_slack(webhook: str, payload: Dict[str, Any]) -> bool:
    """POST the message to Slack. Returns True on success.

    Uses stdlib `urllib.request` so we don't add a requests/httpx
    dependency for a single POST. Timeout enforced so a hanging Slack
    doesn't hang CI.
    """
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        console.print(
            f"[red]Slack webhook rejected: {exc.code} {exc.reason}[/red]"
        )
        return False
    except urllib.error.URLError as exc:
        console.print(f"[red]Slack webhook unreachable: {exc.reason}[/red]")
        return False
    except Exception as exc:  # pragma: no cover — defensive
        console.print(f"[red]Slack POST failed: {exc}[/red]")
        return False


# ───────────────────────── command ─────────────────────────


@click.command("slack-digest")
@click.option(
    "--webhook",
    "webhook",
    default=None,
    envvar="EVALVIEW_SLACK_WEBHOOK",
    help="Slack incoming webhook URL. Falls back to $EVALVIEW_SLACK_WEBHOOK.",
)
@click.option(
    "--since",
    "since",
    default=None,
    help='Time window: "yesterday" | "Nd" | ISO date (default: last run).',
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    help="Print the payload instead of posting to Slack.",
)
@track_command("slack_digest")
def slack_digest_cmd(
    webhook: Optional[str],
    since: Optional[str],
    dry_run: bool,
) -> None:
    """Post a daily EvalView digest to Slack.

    The social layer of the habit loop — optional for solo devs, the
    primary touchpoint for teams. Designed to land in a team channel
    at 3pm every day and spark "did you see the drift thing?"
    conversations that turn EvalView from a tool into a ritual.
    """
    if not webhook and not dry_run:
        console.print(
            "[red]No Slack webhook provided.[/red] "
            "Pass --webhook or set $EVALVIEW_SLACK_WEBHOOK."
        )
        raise click.Abort()

    entries = _load_history(_HISTORY_PATH)
    cutoff_dt, cutoff_sha, label = _parse_since(since, entries)
    window_entries = _entries_since(entries, cutoff_dt, cutoff_sha)
    window = _summarize(window_entries)
    # Stash raw entries for the observability signals section
    window["_entries"] = window_entries
    drift_rows = _detect_drifting_tests(window_entries, entries)

    stale_quarantine: List[Dict[str, Any]] = []
    try:
        from evalview.core.quarantine import QuarantineStore
        store = QuarantineStore()
        for entry in store.list_stale():
            stale_quarantine.append(entry.to_dict())
    except Exception:
        pass

    # Load noise stats over the same window as `_summarize` so the
    # "X% noise" line is consistent with the other digest numbers.
    noise_stats = load_noise_stats(since=cutoff_dt)

    payload = _build_message(
        label, window, drift_rows, stale_quarantine, noise_stats=noise_stats
    )

    if dry_run:
        console.print("[bold]Slack payload preview:[/bold]")
        console.print(json.dumps(payload, indent=2))
        return

    # The early `not webhook and not dry_run` guard above guarantees
    # a non-None webhook by the time we reach this branch.
    assert webhook is not None
    ok = _post_to_slack(webhook, payload)
    if ok:
        console.print("[green]✓ Digest posted to Slack.[/green]")
    else:
        # Return cleanly rather than raising — a broken Slack config
        # should not break CI pipelines that run this command nightly.
        raise click.exceptions.Exit(code=2)
