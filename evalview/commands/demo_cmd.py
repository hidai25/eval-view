"""Demo command — live regression demo with embedded agent."""
from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

import click
from rich.panel import Panel
from rich.rule import Rule

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("demo")
@track_command("demo", lambda **kw: {"is_demo": True})
def demo():
    """Live regression demo — spins up a real agent and catches a real regression."""
    import subprocess as _subprocess
    from evalview.skills.ui_utils import print_evalview_banner

    print_evalview_banner(console, subtitle="[dim]Live Regression Demo[/dim]")

    # ── Intro ─────────────────────────────────────────────────────────────────
    console.print(Rule(" EvalView — Live Regression Demo ", style="bold cyan"))
    console.print()
    console.print("  [bold]Scenario:[/bold] Your customer support AI handles 50,000 tickets a day.")
    console.print("  Engineering just shipped a model update. Let's check it before customers do.")
    console.print()
    console.print("  [dim]Everything below is live — a real HTTP server, real evaluation.[/dim]")
    console.print()

    # ── Start embedded demo agent ─────────────────────────────────────────────
    _state: Dict[str, bool] = {"broken": False}

    class _DemoHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                query = body.get("query", "").lower()
                resp = self._broken(query) if _state["broken"] else self._good(query)
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(500)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # type: ignore[override]
            pass

        def _good(self, query: str) -> Dict[str, Any]:
            if "refund" in query or "return" in query or "jacket" in query:
                return {
                    "response": (
                        "I've found your order #4821 for $84.99 placed 12 days ago. "
                        "Our 30-day return policy covers this — I've initiated your full refund. "
                        "You'll see $84.99 back in 3–5 business days. "
                        "You'll get a confirmation email shortly."
                    ),
                    "steps": [
                        {"tool": "lookup_order", "parameters": {"query": query}, "output": "Order #4821, $84.99, 12 days ago"},
                        {"tool": "check_policy", "parameters": {"type": "return"}, "output": "30-day return window, full refund eligible"},
                        {"tool": "process_refund", "parameters": {"order_id": "4821", "amount": 84.99}, "output": "Refund initiated"},
                    ],
                }
            if "charge" in query or "billing" in query or "129" in query:
                return {
                    "response": (
                        "That $129 charge is your annual plan renewal from March 3rd. "
                        "You signed up for annual billing last year with auto-renewal enabled. "
                        "I can email you the full invoice or switch you to monthly billing — which would you prefer?"
                    ),
                    "steps": [
                        {"tool": "lookup_account", "parameters": {"query": query}, "output": "Account #8821, annual plan"},
                        {"tool": "check_billing_history", "parameters": {"account_id": "8821"}, "output": "$129 annual renewal, March 3rd, auto-renewal on"},
                    ],
                }
            return {"response": "How can I help you today?", "steps": []}

        def _broken(self, query: str) -> Dict[str, Any]:
            if "refund" in query or "return" in query or "jacket" in query:
                # TOOLS_CHANGED: model now escalates every refund to a human agent
                return {
                    "response": (
                        "I've found your order #4821 for $84.99 placed 12 days ago. "
                        "Our 30-day return policy covers this — I've initiated your full refund. "
                        "You'll see $84.99 back in 3–5 business days. "
                        "You'll get a confirmation email shortly."
                    ),
                    "steps": [
                        {"tool": "lookup_order", "parameters": {"query": query}, "output": "Order #4821, $84.99, 12 days ago"},
                        {"tool": "check_policy", "parameters": {"type": "return"}, "output": "30-day return window, full refund eligible"},
                        {"tool": "process_refund", "parameters": {"order_id": "4821", "amount": 84.99}, "output": "Refund initiated"},
                        {"tool": "escalate_to_human", "parameters": {"reason": "refund_processed"}, "output": "Ticket #9921 opened"},
                    ],
                }
            if "charge" in query or "billing" in query or "129" in query:
                # REGRESSION: model skips billing lookup, gives vague non-answer
                return {
                    "response": (
                        "I understand your concern about this charge. "
                        "I'll look into this billing issue and have someone follow up with you within 24–48 hours."
                    ),
                    "steps": [
                        {"tool": "lookup_account", "parameters": {"query": query}, "output": "Account #8821, annual plan"},
                    ],
                }
            return {"response": "How can I help you today?", "steps": []}

    # Pick a random free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.bind(("", 0))
        _port = _s.getsockname()[1]

    _server = HTTPServer(("127.0.0.1", _port), _DemoHandler)
    _server_thread = threading.Thread(target=_server.serve_forever)
    _server_thread.daemon = True
    _server_thread.start()

    # ── Create isolated temp workspace ────────────────────────────────────────
    _tmpdir = tempfile.mkdtemp(prefix="evalview-demo-")
    _tmp = Path(_tmpdir)

    try:
        (_tmp / "tests").mkdir()
        (_tmp / ".evalview").mkdir()

        (_tmp / "tests" / "refund-request.yaml").write_text(
            "name: refund-request\n"
            "description: Customer requests refund for a recent purchase\n"
            "input:\n"
            "  query: I bought a jacket 12 days ago and it doesn't fit. Can I get a refund?\n"
            "expected:\n"
            "  tools:\n"
            "    - lookup_order\n"
            "    - check_policy\n"
            "    - process_refund\n"
            "  output:\n"
            "    contains:\n"
            "      - '84.99'\n"
            "      - refund\n"
            "thresholds:\n"
            "  min_score: 70\n"
        )
        (_tmp / "tests" / "billing-dispute.yaml").write_text(
            "name: billing-dispute\n"
            "description: Customer disputes an unrecognized charge\n"
            "input:\n"
            "  query: There's a $129 charge on my account from last Tuesday I don't recognize.\n"
            "expected:\n"
            "  tools:\n"
            "    - lookup_account\n"
            "    - check_billing_history\n"
            "  output:\n"
            "    contains:\n"
            "      - annual\n"
            "      - '129'\n"
            "thresholds:\n"
            "  min_score: 70\n"
        )
        (_tmp / ".evalview" / "config.yaml").write_text(
            f"adapter: http\n"
            f"endpoint: http://127.0.0.1:{_port}/execute\n"
            f"timeout: 15.0\n"
            f"allow_private_urls: true\n"
        )

        # ── Phase 1: Snapshot good behavior ──────────────────────────────────
        console.print(Rule(" Phase 1 — Baseline: the agent before the update ", style="cyan"))
        console.print()
        console.print("  [dim]Running the test suite against today's production agent...[/dim]")
        console.print()

        _demo_env_base = {
            **os.environ,
            "EVALVIEW_DEMO": "1",
            "EVALVIEW_TELEMETRY_DISABLED": "1",  # prevent telemetry blocking subprocess exit
        }

        _subprocess.run(
            ["evalview", "snapshot", "tests/"],
            cwd=_tmpdir,
            env={**_demo_env_base, "EVALVIEW_DEMO_PHASE": "snapshot"},
            stderr=_subprocess.DEVNULL,
        )

        console.print()

        # ── Phase 2: Break the agent, run check ──────────────────────────────
        console.print(Rule(" Phase 2 — Model update deployed to staging ", style="yellow"))
        console.print()
        console.print("  [bold]The new model is live. Running regression check before it hits production...[/bold]")
        console.print()

        _state["broken"] = True  # switch agent to broken mode

        _subprocess.run(
            ["evalview", "check"],
            cwd=_tmpdir,
            env={**_demo_env_base, "EVALVIEW_DEMO_PHASE": "check"},
            stderr=_subprocess.DEVNULL,
        )

        console.print()
        console.print("  [bold red]At 50K tickets/day, this update would have cost:[/bold red]")
        console.print("  [red]  • escalate_to_human on every refund  →  $125K/day in unnecessary ops[/red]")
        console.print("  [red]  • billing-dispute non-answer  →  chargebacks, churn, manual escalations[/red]")
        console.print()
        console.print("  [green]EvalView caught both before a single customer was affected.[/green]")
        console.print()

        # ── CTA ──────────────────────────────────────────────────────────────
        console.print(Panel(
            "[bold green]Now run this on your own agent:[/bold green]\n"
            "\n"
            "  [cyan]$ evalview snapshot[/cyan]   [dim]# save today's behavior as baseline[/dim]\n"
            "  [cyan]$ evalview check[/cyan]      [dim]# run before every deploy[/dim]",
            border_style="green",
            padding=(1, 3),
        ))
        console.print()
        console.print(
            "  [yellow]⭐[/yellow] [dim]Star the repo:[/dim]"
            " [link=https://github.com/hidai25/eval-view]github.com/hidai25/eval-view[/link]"
        )
        console.print()

    finally:
        shutil.rmtree(_tmpdir, ignore_errors=True)
        os._exit(0)  # force-exit — PostHog consumer thread would otherwise block indefinitely
