"""Cloud authentication commands (login, logout, whoami).

`evalview login` runs the OAuth-loopback flow used by `gh`, `vercel`,
and `supabase` CLIs:

  1. CLI generates a random state token and binds a free port on
     127.0.0.1 for a one-shot HTTP listener.
  2. CLI opens ``$EVALVIEW_CLOUD_WEB_URL/cli-auth`` with the loopback
     URL and state in the query string.
  3. User authenticates with EvalView Cloud (Google / GitHub / magic
     link — same dashboard login) and clicks "Authorize CLI".
  4. Cloud mints an ``ev_…`` API token, redirects the browser back
     to ``http://127.0.0.1:<port>/callback?token=…&state=…``.
  5. The local listener captures the token, verifies the state, and
     saves it to ``~/.evalview/auth.json`` (chmod 600). All
     subsequent ``evalview check`` / ``run`` / ``monitor`` calls
     read it from there.

The token never leaves the user's machine after step 4 — it only
travels browser → loopback listener.
"""
from __future__ import annotations

import json
import os
import secrets
import socket
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

import click
from rich.panel import Panel

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


CALLBACK_HTML = b"""<!DOCTYPE html><html><head><title>EvalView CLI authorized</title>
<style>body{font-family:system-ui;text-align:center;padding:3rem;color:#222}</style></head>
<body><h2>EvalView CLI authorized</h2>
<p>You can close this tab and return to your terminal.</p></body></html>"""

ERROR_HTML = b"""<!DOCTYPE html><html><head><title>EvalView CLI error</title>
<style>body{font-family:system-ui;text-align:center;padding:3rem;color:#b00}</style></head>
<body><h2>Login failed</h2>
<p>Return to your terminal for details.</p></body></html>"""


def _default_web_url() -> str:
    """Where the cloud dashboard lives (the /cli-auth page).

    Distinct from the API base URL (``EVALVIEW_CLOUD_URL``) because the
    API may sit behind an /api/v1 prefix on the same host.
    """
    return os.environ.get("EVALVIEW_CLOUD_WEB_URL", "https://evalview.com")


def _default_api_url() -> str:
    return os.environ.get("EVALVIEW_CLOUD_URL", "https://evalview.com/api/v1")


def _pick_free_port() -> int:
    """Return a free port in 8000-8100. Falls back to OS-assigned."""
    for candidate in range(8000, 8101):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", candidate)) != 0:
                return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_loopback_capture(port: int, expected_state: str, timeout_s: int = 300) -> Dict[str, Optional[str]]:
    """Run a one-shot HTTP server that captures /callback?token=…&state=…."""
    captured: Dict[str, Optional[str]] = {
        "token": None,
        "state": None,
        "email": None,
        "project_slug": None,
        "org_slug": None,
        "error": None,
    }

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            try:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != "/callback":
                    self.send_response(404)
                    self.end_headers()
                    return
                params = urllib.parse.parse_qs(parsed.query)
                token = (params.get("token") or [""])[0]
                state = (params.get("state") or [""])[0]
                if not token or state != expected_state:
                    captured["error"] = "state mismatch or missing token"
                    self.send_response(400)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(ERROR_HTML)))
                    self.end_headers()
                    self.wfile.write(ERROR_HTML)
                    return
                captured["token"] = token
                captured["state"] = state
                captured["email"] = (params.get("email") or [""])[0] or None
                captured["project_slug"] = (params.get("project_slug") or [""])[0] or None
                captured["org_slug"] = (params.get("org_slug") or [""])[0] or None
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(CALLBACK_HTML)))
                self.end_headers()
                self.wfile.write(CALLBACK_HTML)
            except Exception as exc:  # pragma: no cover — defensive
                captured["error"] = str(exc)
                self.send_response(500)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # Suppress access logs.

    httpd = HTTPServer(("127.0.0.1", port), _Handler)
    httpd.timeout = timeout_s
    try:
        httpd.handle_request()
    finally:
        httpd.server_close()
    return captured


@click.command("login")
@track_command("login")
def login() -> None:
    """Connect this CLI to your EvalView Cloud account."""
    import webbrowser

    from evalview.cloud.auth import CloudAuth

    auth = CloudAuth()

    if auth.is_logged_in():
        email = auth.get_email() or "(unknown)"
        prefix = auth.get_token_prefix() or ""
        console.print(f"[green]Already logged in as {email}[/green]")
        if prefix:
            console.print(f"[dim]Token: {prefix}…[/dim]")
        console.print("[dim]Run [bold]evalview logout[/bold] first to switch accounts.[/dim]")
        return

    web_url = _default_web_url().rstrip("/")
    api_url = _default_api_url()

    port = _pick_free_port()
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    state = secrets.token_urlsafe(24)

    auth_url = (
        f"{web_url}/cli-auth"
        f"?redirect_uri={urllib.parse.quote(redirect_uri, safe='')}"
        f"&state={urllib.parse.quote(state, safe='')}"
    )

    console.print("\n[cyan]Opening EvalView Cloud in your browser…[/cyan]")
    console.print("[dim]If the browser doesn't open, visit:[/dim]")
    console.print(f"[dim]{auth_url}[/dim]\n")
    webbrowser.open(auth_url)
    console.print("[dim]Waiting for authorization (timeout: 5 min)…[/dim]")

    result = _run_loopback_capture(port, state)

    if result.get("error") or not result.get("token"):
        console.print(
            f"[red]Login failed: {result.get('error') or 'no token received'}.[/red]"
        )
        console.print("[dim]Try running [bold]evalview login[/bold] again.[/dim]")
        return

    token = result["token"] or ""
    email = result.get("email") or "(unknown)"
    project_slug = result.get("project_slug") or "default"
    org_slug = result.get("org_slug") or ""

    auth.save_api_token(
        api_token=token,
        cloud_url=api_url,
        email=email,
        project_slug=project_slug,
        org_slug=org_slug,
        token_prefix=token[:11],
    )

    console.print(
        Panel(
            f"[green]✓ Logged in as {email}[/green]\n"
            f"[dim]Project: {org_slug}/{project_slug}[/dim]\n"
            f"[dim]Token:   {token[:11]}…[/dim]\n\n"
            "Run [cyan]evalview check[/cyan] — results will appear in your cloud dashboard.\n"
            "Revoke this token any time at [dim]evalview.com/dashboard[/dim].",
            title="EvalView Cloud",
            border_style="green",
        )
    )


@click.command("logout")
@track_command("logout")
def logout() -> None:
    """Disconnect this CLI from EvalView Cloud."""
    from evalview.cloud.auth import CloudAuth

    auth = CloudAuth()
    data = auth.load()
    if not data:
        console.print("[dim]Not logged in.[/dim]")
        return

    email = data.get("email", "(unknown)")
    auth.clear()
    console.print(f"[green]✓ Logged out[/green] (was {email})")
    console.print(
        "[dim]The API token remains valid in the cloud — "
        "revoke it from your dashboard if you want to invalidate it.[/dim]"
    )


@click.command("whoami")
def whoami() -> None:
    """Show current cloud login status."""
    from evalview.cloud.auth import CloudAuth

    auth = CloudAuth()
    data = auth.load()
    if not data:
        console.print("[dim]Not logged in. Run [bold]evalview login[/bold] to connect.[/dim]")
        return

    if "api_token" in data:
        console.print(f"[green]Logged in[/green] as {data.get('email', '(unknown)')}")
        org = data.get("org_slug") or ""
        proj = data.get("project_slug") or ""
        if org and proj:
            console.print(f"[dim]Project: {org}/{proj}[/dim]")
        prefix = data.get("token_prefix") or ""
        if prefix:
            console.print(f"[dim]Token:   {prefix}…[/dim]")
        return

    # Legacy v1 session (Supabase-A access_token). Unusable for cloud
    # push — point the user at a re-login.
    email = data.get("email", "(unknown)")
    console.print(f"[yellow]Legacy session for {email}[/yellow]")
    console.print(
        "[dim]Run [bold]evalview logout && evalview login[/bold] to upgrade to "
        "an API-token session (required for cloud push).[/dim]"
    )
