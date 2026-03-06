"""Cloud authentication commands (login, logout, whoami)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.panel import Panel

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("login")
@track_command("login")
def login() -> None:
    """Connect to EvalView Cloud. Baselines sync automatically after login."""
    import webbrowser
    import socket
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from evalview.cloud.auth import CloudAuth
    from evalview.cloud.client import CloudClient

    auth = CloudAuth()

    if auth.is_logged_in():
        email = auth.get_email()
        console.print(f"[green]Already logged in as {email}[/green]")
        console.print("[dim]Run evalview logout first to switch accounts.[/dim]")
        return

    # Find a free port in 8000-8100
    port = 8000
    for candidate in range(8000, 8101):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", candidate)) != 0:
                port = candidate
                break

    redirect_uri = f"http://127.0.0.1:{port}/callback"
    auth_url = CloudClient.build_oauth_url(redirect_uri)

    # Supabase uses the implicit flow: tokens arrive in the URL *fragment* (#),
    # which browsers never send to the server. We serve an HTML page that reads
    # the fragment via JavaScript and POSTs the tokens to /token.
    token_result: Dict[str, Optional[str]] = {"access_token": None, "refresh_token": None}

    CALLBACK_HTML = """\
<!DOCTYPE html><html><head><title>EvalView Login</title></head><body>
<h2>Completing login\u2026</h2>
<script>
var params = new URLSearchParams(window.location.hash.substring(1));
var at = params.get('access_token');
var rt = params.get('refresh_token');
if (at) {
  fetch('/token', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({access_token: at, refresh_token: rt})
  }).then(function() {
    document.body.innerHTML = '<h2>Login successful! You can close this tab.</h2>';
  });
} else {
  document.body.innerHTML = '<h2>Login failed \u2014 no token found. Please try again.</h2>';
}
</script></body></html>"""

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            # Serve the JS page that reads the fragment
            body = CALLBACK_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            # Receive tokens POSTed by the JS page
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode())
            token_result["access_token"] = data.get("access_token")
            token_result["refresh_token"] = data.get("refresh_token")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format: str, *args: Any) -> None:
            pass  # Suppress server access logs

    httpd = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    httpd.timeout = 300  # 5 minutes

    console.print(f"\n[cyan]Opening GitHub in your browser...[/cyan]")
    console.print(f"[dim]If the browser doesn't open, visit:[/dim]")
    console.print(f"[dim]{auth_url}[/dim]\n")

    webbrowser.open(auth_url)
    console.print("[dim]Waiting for GitHub authorization (timeout: 5 min)...[/dim]")

    httpd.handle_request()  # GET /callback — serves the HTML page
    httpd.handle_request()  # POST /token   — receives the tokens
    httpd.server_close()

    access_token = token_result.get("access_token")
    refresh_token = token_result.get("refresh_token") or ""

    if not access_token:
        console.print("[red]Login failed: no token received.[/red]")
        console.print("[dim]Try running evalview login again.[/dim]")
        return

    # Fetch user info (email, id) from Supabase
    user = asyncio.run(CloudClient.get_user_info(access_token))
    if not user:
        console.print("[red]Login failed: could not fetch user info.[/red]")
        return

    user_id = user.get("id", "")
    email = user.get("email", "")

    if not user_id:
        console.print("[red]Login failed: incomplete user data.[/red]")
        return

    auth.save(access_token, refresh_token, user_id, email)

    has_tests = (Path("tests") / "test-cases").exists() and any(
        (Path("tests") / "test-cases").glob("*.yaml")
    )
    has_golden = Path(".evalview") / "golden"

    if has_tests and has_golden.exists() and any(has_golden.glob("*.golden.json")):
        next_step = "  [cyan]evalview snapshot[/cyan]   push your existing baselines to cloud"
    elif has_tests:
        next_step = "  [cyan]evalview snapshot[/cyan]   capture a baseline and sync it to cloud"
    else:
        next_step = "  [cyan]evalview init[/cyan]       create your first test case"

    console.print(Panel(
        f"[green]✓ Logged in as {email}[/green]\n\n"
        "Your golden baselines will now sync to cloud automatically.\n\n"
        "[bold]Next step:[/bold]\n"
        f"{next_step}",
        title="EvalView Cloud",
        border_style="green",
    ))


@click.command("logout")
@track_command("logout")
def logout() -> None:
    """Disconnect from EvalView Cloud."""
    from evalview.cloud.auth import CloudAuth

    auth = CloudAuth()
    if not auth.is_logged_in():
        console.print("[dim]Not logged in.[/dim]")
        return

    email = auth.get_email()
    auth.clear()
    console.print(f"[green]✓ Logged out[/green] (was {email})")
    console.print("[dim]Your local baselines are untouched.[/dim]")


@click.command("whoami")
def whoami() -> None:
    """Show current cloud login status."""
    from evalview.cloud.auth import CloudAuth

    auth = CloudAuth()
    if auth.is_logged_in():
        console.print(f"[green]Logged in[/green] as {auth.get_email()}")
        console.print(f"[dim]User ID: {auth.get_user_id()}[/dim]")
    else:
        console.print("[dim]Not logged in. Run [bold]evalview login[/bold] to connect.[/dim]")
