"""EvalView MCP Server — exposes evalview check/snapshot as MCP tools for Claude Code."""

import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

TOOLS = [
    {
        "name": "run_check",
        "description": (
            "Check for regressions against the golden baseline. "
            "Returns diff output showing what changed vs the last snapshot. "
            "A regression means the agent's behavior changed unexpectedly. "
            "Use this after refactoring agent code to confirm nothing broke."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "test": {
                    "type": "string",
                    "description": "Check only this specific test by name (optional, checks all by default)",
                },
                "test_path": {
                    "type": "string",
                    "description": "Path to the test directory (default: tests)",
                },
            },
        },
    },
    {
        "name": "run_snapshot",
        "description": (
            "Run tests and save passing results as the new golden baseline. "
            "Use this to establish or update the expected behavior after an intentional change. "
            "Future `run_check` calls will compare against this snapshot."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "test": {
                    "type": "string",
                    "description": "Snapshot only this specific test by name (optional, snapshots all by default)",
                },
                "notes": {
                    "type": "string",
                    "description": "Human-readable note about why this snapshot was taken",
                },
                "test_path": {
                    "type": "string",
                    "description": "Path to the test directory (default: tests)",
                },
            },
        },
    },
    {
        "name": "list_tests",
        "description": (
            "List all available golden baselines in this EvalView project. "
            "Shows test names, variant counts, and when each baseline was last updated."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


class MCPServer:
    """Synchronous stdio JSON-RPC MCP server for EvalView."""

    def __init__(self, test_path: str = "tests") -> None:
        self.test_path = test_path

    def serve(self) -> None:
        """Run the synchronous stdin/stdout JSON-RPC loop."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = self._handle(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

    def _handle(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = req.get("method", "")
        req_id = req.get("id")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "evalview", "version": "0.2.5"},
                },
            }

        if method == "notifications/initialized":
            return None  # notifications don't get a response

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            output = self._call_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": output}],
                    "isError": False,
                },
            }

        # Unknown method — return error only if it has an id (i.e. it's a request not a notification)
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

        return None

    def _call_tool(self, name: str, args: Dict[str, Any]) -> str:
        if not shutil.which("evalview"):
            return "Error: evalview not found in PATH. Run: pip install -e ."

        if name == "run_check":
            cmd = ["evalview", "check", args.get("test_path", self.test_path), "--json"]
            if args.get("test"):
                cmd += ["--test", args["test"]]

        elif name == "run_snapshot":
            cmd = ["evalview", "snapshot", args.get("test_path", self.test_path)]
            if args.get("test"):
                cmd += ["--test", args["test"]]
            if args.get("notes"):
                cmd += ["--notes", args["notes"]]

        elif name == "list_tests":
            cmd = ["evalview", "golden", "list"]

        else:
            return f"Unknown tool: {name}"

        env = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        output = result.stdout
        if result.stderr:
            output += result.stderr
        return output.strip() or f"Command exited with code {result.returncode}"
