"""Capture command — transparent proxy that saves real traffic as test YAMLs."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import httpx

from evalview.commands.shared import console, _detect_agent_endpoint
from evalview.telemetry.decorators import track_command


def _escape_yaml_str(s: str) -> str:
    """Escape *s* so it is safe inside a YAML double-quoted scalar."""
    s = (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\r", "\\r")
         .replace("\t", "\\t")
    )
    return re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        lambda m: f"\\u{ord(m.group()):04x}",
        s,
    )


def _extract_keywords_from_output(output: str, query: str) -> List[str]:
    """Extract up to 3 keywords from *output* to use as ``contains:`` assertions."""
    if not output:
        return []

    keywords: List[str] = []

    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", output)
    for n in numbers[:2]:
        if n not in keywords:
            keywords.append(n)

    skip_words = {"the", "this", "that", "they", "their", "then", "there"}
    proper = re.findall(r"\b[A-Z][a-z]{3,}\b", output)
    for w in proper:
        if len(keywords) >= 3:
            break
        if w.lower() not in skip_words and w not in keywords:
            keywords.append(w)

    if not keywords:
        words = re.findall(r"\b[a-zA-Z]{5,}\b", output)
        seen: set = set()
        for w in words[:15]:
            if len(keywords) >= 2:
                break
            if w.lower() not in seen:
                keywords.append(w)
                seen.add(w.lower())

    return keywords[:3]


def _save_captures_as_tests(captures: List[Dict[str, Any]], output_dir: Path) -> Tuple[int, List[str]]:
    """Save captured interactions as test YAML files. Returns (count, list of file paths)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    saved_paths: List[str] = []

    for cap in captures:
        query: str = cap.get("query", "")
        output: str = cap.get("output", "")
        tools: List[str] = cap.get("tools", [])
        idx: int = cap.get("idx", saved + 1)

        if not query.strip():
            continue

        slug = re.sub(r"[^a-z0-9-]", "-", query[:40].lower()).strip("-")
        slug = re.sub(r"-+", "-", slug) or f"capture-{idx}"

        path = output_dir / f"capture-{idx:02d}-{slug}.yaml"

        contains = _extract_keywords_from_output(output, query)

        lines = [
            f'name: "capture-{idx:02d}"',
            'description: "Real interaction captured by evalview capture"',
            "",
            "input:",
            f'  query: "{_escape_yaml_str(query)}"',
            "",
            "expected:",
        ]

        if tools:
            lines.append("  tools:")
            for t in tools:
                lines.append(f"    - {t}")

        lines += [
            "  output:",
            "    contains:",
        ]
        if contains:
            for kw in contains:
                lines.append(f'      - "{_escape_yaml_str(kw)}"')
        else:
            lines.append('      []  # Add phrases your agent always includes')

        lines += [
            "    not_contains:",
            '      - "error"',
            "",
            "thresholds:",
            "  min_score: 70",
            "  max_latency: 15000",
        ]

        path.write_text("\n".join(lines) + "\n")
        saved_paths.append(str(path))
        saved += 1

    return saved, saved_paths


def _save_multi_turn_test(captures: List[Dict[str, Any]], output_dir: Path) -> Tuple[int, List[str]]:
    """Save all captures as a single multi-turn test YAML. Returns (1, [path]) if saved, (0, []) if empty."""
    if len(captures) < 2:
        return _save_captures_as_tests(captures, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    first_query = captures[0].get("query", "capture")
    slug = re.sub(r"[^a-z0-9-]", "-", first_query[:40].lower()).strip("-")
    slug = re.sub(r"-+", "-", slug) or "multi-turn"
    path = output_dir / f"multi-turn-{slug}.yaml"

    lines = [
        f'name: "multi-turn-{slug}"',
        f'description: "Multi-turn conversation captured by evalview capture ({len(captures)} turns)"',
        "",
        "turns:",
    ]

    for cap in captures:
        query = cap.get("query", "")
        output = cap.get("output", "")
        tools: List[str] = cap.get("tools", [])

        if not query.strip():
            continue

        lines.append(f'  - query: "{_escape_yaml_str(query)}"')
        lines.append("    expected:")

        if tools:
            lines.append("      tools:")
            for t in tools:
                lines.append(f"        - {t}")

        contains = _extract_keywords_from_output(output, query)
        if contains:
            lines.append("      output:")
            lines.append("        contains:")
            for kw in contains:
                lines.append(f'          - "{_escape_yaml_str(kw)}"')

    lines += [
        "",
        "thresholds:",
        "  min_score: 70",
        "  max_latency: 30000",
    ]

    path.write_text("\n".join(lines) + "\n")
    return 1, [str(path)]


@click.command("capture")
@click.option("--agent", default=None, help="Real agent URL to proxy to (auto-detected if not set)")
@click.option("--port", default=8091, type=int, show_default=True, help="Local port for the proxy to listen on")
@click.option(
    "--output-dir", "output_dir",
    default="tests/test-cases", show_default=True,
    help="Directory where captured test YAMLs are saved",
)
@click.option("--multi-turn", "multi_turn", is_flag=True, default=False, help="Save all captures as one multi-turn conversation test")
@track_command("capture")
def capture(agent: Optional[str], port: int, output_dir: str, multi_turn: bool) -> None:
    """🎯 Capture real traffic as tests — tests from real usage, not guesses.

    \b
    Starts a transparent proxy that sits between your client and your agent.
    Every request/response pair is saved as a test YAML automatically.

    \b
    Usage:
      evalview capture --agent http://localhost:8000/invoke
      # Now point your app/client to http://localhost:8091 instead
      # Use your agent normally — each interaction becomes a test
      # Press Ctrl+C when done — tests are written automatically

    \b
    Then:
      evalview snapshot   ← save as your regression baseline
      evalview check      ← catch regressions before they ship
    """
    import http.server
    import socketserver
    import signal
    import threading as _threading
    import json as _json
    from urllib.parse import urlparse

    if not agent:
        detected = _detect_agent_endpoint()
        if detected:
            console.print(f"[green]✓ Auto-detected agent at {detected}[/green]")
            agent = detected
        else:
            agent = click.prompt("Agent URL", default="http://localhost:8000/invoke")

    _parsed = urlparse(agent)
    if not _parsed.scheme or not _parsed.netloc:
        console.print(
            f"[red]Error: Invalid agent URL — {agent!r}[/red]\n"
            "[dim]Must include scheme and host, e.g. http://localhost:8000/invoke[/dim]"
        )
        raise SystemExit(1)

    agent_url: str = agent

    captures: List[Dict[str, Any]] = []
    lock = _threading.Lock()

    _HOP_BY_HOP = frozenset({
        "connection", "keep-alive", "proxy-authenticate",
        "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade",
    })

    class _ProxyHandler(http.server.BaseHTTPRequestHandler):
        """Transparent forwarding proxy that captures every POST to agent_url."""

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def _read_body(self) -> Optional[bytes]:
            raw = self.headers.get("Content-Length", "0")
            try:
                length = int(raw)
            except ValueError:
                self.send_error(400, f"Invalid Content-Length: {raw!r}")
                return None
            if length < 0:
                self.send_error(400, "Content-Length must not be negative")
                return None
            return self.rfile.read(length) if length else b""

        def _extract_query(self, body: bytes) -> str:
            if not body:
                return ""
            try:
                data = _json.loads(body)
            except _json.JSONDecodeError:
                return ""
            if not isinstance(data, dict):
                return ""
            if "query" in data:
                return str(data["query"])
            if "messages" in data:
                msgs = data["messages"]
                if isinstance(msgs, list):
                    user_msgs = [
                        m for m in msgs
                        if isinstance(m, dict) and m.get("role") == "user"
                    ]
                    if user_msgs:
                        return str(user_msgs[-1].get("content", ""))
            return ""

        def _extract_agent_response(self, body: bytes) -> "Tuple[str, List[str]]":
            output = ""
            tools: List[str] = []
            if not body:
                return output, tools
            try:
                data = _json.loads(body)
                output = str(data.get("output", ""))
                raw_tools = data.get("tool_calls", [])
                if isinstance(raw_tools, list):
                    for t in raw_tools:
                        if isinstance(t, dict):
                            tool_name = t.get("tool") or t.get("name") or ""
                            if tool_name:
                                tools.append(str(tool_name))
            except Exception:
                pass
            return output, tools

        def _forward_headers(self) -> Dict[str, str]:
            forwarded: Dict[str, str] = {}
            for key, value in self.headers.items():
                if key.lower() not in _HOP_BY_HOP and key.lower() != "host":
                    forwarded[key] = value
            forwarded.setdefault("Content-Type", "application/json")
            return forwarded

        def do_POST(self) -> None:
            body = self._read_body()
            if body is None:
                return

            query = self._extract_query(body)

            try:
                import time as _time
                _req_start = _time.monotonic()
                resp = httpx.post(
                    agent_url,
                    content=body,
                    headers=self._forward_headers(),
                    timeout=60.0,
                )
                resp_body = resp.content
                _req_latency = (_time.monotonic() - _req_start) * 1000
                status_code = resp.status_code
            except Exception as exc:
                err = _json.dumps({"error": str(exc)}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(err)
                return

            output, tools = self._extract_agent_response(resp_body)

            if query.strip():
                with lock:
                    captures.append({
                        "query": query,
                        "output": output,
                        "tools": tools,
                        "idx": len(captures) + 1,
                        "latency_ms": _req_latency,
                    })

            self.send_response(status_code)
            for key, value in resp.headers.items():
                if key.lower() not in _HOP_BY_HOP | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def do_GET(self) -> None:
            body = b'{"status":"proxy-active"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class _ReuseAddrServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    try:
        server = _ReuseAddrServer(("", port), _ProxyHandler)
    except OSError as exc:
        console.print(
            f"[red]Error: Could not bind to port {port}[/red]\n"
            f"[dim]{exc}[/dim]\n"
            f"[dim]Try a different port: evalview capture --port 8092[/dim]"
        )
        raise SystemExit(1)

    server_thread = _threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    def _handle_shutdown(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_shutdown)

    from rich.panel import Panel
    from rich.table import Table

    mode_label = "[yellow]multi-turn[/yellow] — all requests become one conversation test" if multi_turn else "each request becomes a separate test"
    console.print()
    console.print(Panel(
        f"[bold green]Proxy live on http://localhost:{port}[/bold green]\n\n"
        f"[dim]Forwarding to:[/dim] [cyan]{agent_url}[/cyan]\n"
        f"[dim]Mode:[/dim] {mode_label}\n\n"
        f"Point your client to [bold cyan]http://localhost:{port}[/bold cyan]\n"
        f"instead of your agent — every interaction is captured.\n\n"
        f"[dim]Press [bold]Ctrl+C[/bold] when done.[/dim]",
        title="[bold]EvalView Capture[/bold]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()

    from rich.live import Live

    def _build_table() -> Table:
        with lock:
            snap = list(captures)

        t = Table(
            title=(
                f"Captured Interactions — {len(snap)} so far  "
                "[dim](Ctrl+C to save & exit)[/dim]"
            ),
            box=None,
            show_header=True,
            header_style="bold",
        )
        t.add_column("#", style="dim", width=3)
        t.add_column("Query", style="cyan", max_width=48)
        t.add_column("Tools", style="yellow", max_width=22)
        t.add_column("Output preview", style="white", max_width=38)

        for cap in snap:
            q = cap["query"]
            o = cap["output"]
            tool_str = ", ".join(cap["tools"]) if cap["tools"] else "[dim]—[/dim]"
            t.add_row(
                str(cap["idx"]),
                (q[:48] + "…") if len(q) > 48 else q,
                tool_str,
                (o[:38] + "…") if len(o) > 38 else o,
            )
        return t

    try:
        with Live(_build_table(), refresh_per_second=2, console=console) as live:
            while True:
                live.update(_build_table())
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()

    console.print()

    # Also record latency if available in captures
    saved_paths: List[str] = []

    if multi_turn and len(captures) >= 2:
        n_saved, saved_paths = _save_multi_turn_test(captures, Path(output_dir))
        test_type = f"1 multi-turn test ({len(captures)} turns)"
    else:
        n_saved, saved_paths = _save_captures_as_tests(captures, Path(output_dir))
        test_type = f"{n_saved} test(s)"

    if n_saved > 0:
        console.print(f"[green]Saved {test_type} to {output_dir}/[/green]")
        console.print()

        # Run the assertion wizard
        if len(captures) >= 2:
            from evalview.commands.assertion_wizard import enhance_captured_tests
            enhance_captured_tests(captures, output_dir, saved_paths)

        console.print(Panel(
            f"[bold]You now have {n_saved} test(s) from real traffic.[/bold]\n\n"
            "[bold]Next:[/bold]\n"
            "  [cyan]evalview snapshot[/cyan]  ← save as your regression baseline\n"
            "  [cyan]evalview check[/cyan]     ← check for regressions anytime",
            title="Capture complete",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[yellow]No interactions were captured.[/yellow]\n\n"
            f"Make sure your client is pointing to [cyan]http://localhost:{port}[/cyan]\n"
            f"and your agent is running at [cyan]{agent_url}[/cyan].\n\n"
            "Run [cyan]evalview capture[/cyan] again after sending some queries.",
            title="Nothing captured",
            border_style="yellow",
        ))
