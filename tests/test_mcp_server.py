"""Tests for MCP server tool definitions, protocol handling, and CLI flag wiring.

Covers:
- Protocol handshake (initialize, tools/list)
- Tool schema contract (names, required params, param types)
- Flag-to-CLI mapping for run_check subprocess path
- Flag-to-CLI mapping for run_snapshot
- Flag-to-CLI mapping for compare_agents and replay
- Response contract stability (JSON output from both direct and subprocess paths)
- Timeout tier assignments
- Error handling for unknown tools/methods
"""

import os
import subprocess
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest

from evalview.core.loader import TestCaseLoader
from evalview.mcp_server import MCPServer, TOOLS, ToolError, _EVALVIEW_VERSION


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def server():
    return MCPServer(test_path="tests")


@pytest.fixture
def tmp_test_dir():
    d = tempfile.mkdtemp(prefix="evalview-mcp-test-")
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


# ============================================================================
# Protocol Tests
# ============================================================================

class TestProtocol:
    def test_initialize(self, server):
        resp = server._handle({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        })
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "evalview"
        assert resp["result"]["serverInfo"]["version"] == _EVALVIEW_VERSION

    def test_initialized_notification_returns_none(self, server):
        resp = server._handle({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert resp is None

    def test_tools_list(self, server):
        resp = server._handle({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        tools = resp["result"]["tools"]
        names = [t["name"] for t in tools]
        assert len(tools) == 10
        assert "create_test" in names
        assert "run_check" in names
        assert "run_snapshot" in names
        assert "list_tests" in names
        assert "validate_skill" in names
        assert "generate_skill_tests" in names
        assert "run_skill_test" in names
        assert "generate_visual_report" in names
        assert "compare_agents" in names
        assert "replay" in names

    def test_unknown_method_with_id_returns_error(self, server):
        resp = server._handle({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "nonexistent/method",
        })
        assert resp["error"]["code"] == -32601
        assert "nonexistent/method" in resp["error"]["message"]

    def test_unknown_method_without_id_returns_none(self, server):
        resp = server._handle({
            "jsonrpc": "2.0",
            "method": "nonexistent/notification",
        })
        assert resp is None

    def test_unknown_tool_returns_error(self, server):
        resp = server._handle({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "does_not_exist", "arguments": {}},
        })
        text = resp["result"]["content"][0]["text"]
        assert "Unknown tool" in text
        assert resp["result"]["isError"] is True

    def test_tool_validation_error_is_flagged(self, server):
        resp = server._handle({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "create_test", "arguments": {"name": "x"}},
        })
        assert resp["result"]["isError"] is True
        assert "'name' and 'query' are required" in resp["result"]["content"][0]["text"]


class TestBadArguments:
    """A bad tool argument must return an error, not kill the stdio server."""

    def _serve(self, server, requests):
        import io
        import json

        # A str is sent as a raw line; anything else is JSON-encoded.
        stdin = io.StringIO(
            "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in requests)
        )
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            server.serve()
        return [json.loads(line) for line in stdout.getvalue().splitlines()]

    @pytest.mark.parametrize(
        "arguments",
        [{"name": 5, "query": "x"}, None, ["not", "an", "object"], []],
    )
    def test_bad_create_test_arguments_return_error_and_keep_serving(
        self, server, tmp_test_dir, arguments
    ):
        server.test_path = tmp_test_dir
        responses = self._serve(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "create_test", "arguments": arguments},
                },
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ],
        )
        assert [r["id"] for r in responses] == [1, 2]
        assert responses[0]["result"]["content"][0]["text"].startswith("Error")
        assert responses[0]["result"]["isError"] is True
        assert "tools" in responses[1]["result"]

    def test_tool_exception_returns_is_error(self, server):
        with patch.object(server, "_call_tool", side_effect=RuntimeError("boom")):
            resp = server._handle(
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {"name": "create_test", "arguments": {}},
                }
            )
        assert resp["result"]["isError"] is True
        assert "boom" in resp["result"]["content"][0]["text"]

    def test_handler_exception_returns_jsonrpc_error_and_keeps_serving(self, server):
        real_handle = server._handle

        def handle_failing_on_first(req):
            if req.get("id") == 1:
                raise RuntimeError("boom")
            return real_handle(req)

        with patch.object(server, "_handle", side_effect=handle_failing_on_first):
            responses = self._serve(
                server,
                [
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                ],
            )
        assert [r["id"] for r in responses] == [1, 2]
        assert responses[0]["error"]["code"] == -32603
        assert "tools" in responses[1]["result"]

    def test_deeply_nested_json_is_skipped_and_serving_continues(self, server, capsys):
        # Deep enough to exceed the parser's recursion limit.
        responses = self._serve(
            server,
            ["[" * 100_000 + "]" * 100_000, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}],
        )
        assert [r["id"] for r in responses] == [2]
        assert "could not be parsed as JSON" in capsys.readouterr().err

    def test_invalid_utf8_does_not_stop_the_server(self, server):
        import io
        import json

        raw = (
            b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "x": "\xff\xfe"}\n'
            b'{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}\n'
        )
        # A real stdin is a TextIOWrapper; serve() switches it to UTF-8 with replacement.
        stdin = io.TextIOWrapper(io.BytesIO(raw), encoding="ascii")
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            server.serve()
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        assert [r["id"] for r in responses] == [1, 2]

    def test_tools_call_without_id_is_not_run(self, server, tmp_test_dir):
        server.test_path = tmp_test_dir
        responses = self._serve(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "create_test", "arguments": {"name": "n", "query": "q"}},
                },
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ],
        )
        assert [r["id"] for r in responses] == [2]
        assert os.listdir(tmp_test_dir) == []

    def test_null_params_is_treated_as_empty(self, server):
        resp = server._handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": None})
        assert resp["result"]["isError"] is True
        assert "Unknown tool" in resp["result"]["content"][0]["text"]

    def test_non_object_params_return_invalid_params(self, server):
        resp = server._handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1]})
        assert resp["error"]["code"] == -32602

    def test_request_without_method_returns_invalid_request(self, server):
        resp = server._handle({"jsonrpc": "2.0", "id": 1})
        assert resp["error"]["code"] == -32600
        assert server._handle({"jsonrpc": "2.0", "id": 1, "method": 5})["error"]["code"] == -32600

    def test_response_shaped_message_gets_no_reply(self, server):
        assert server._handle({"jsonrpc": "2.0", "id": 1, "result": {}}) is None
        assert server._handle({"jsonrpc": "2.0", "id": 1, "error": {"code": 1}}) is None

    def test_lines_without_an_id_are_logged_and_skipped(self, server, capsys):
        responses = self._serve(
            server,
            [
                "not json",
                "5",
                '[{"jsonrpc": "2.0", "id": 9, "method": "tools/list"}]',
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ],
        )
        assert [r["id"] for r in responses] == [2]
        err = capsys.readouterr().err
        assert "could not be parsed as JSON" in err
        assert err.count("not a JSON object") == 2


# ============================================================================
# Tool Schema Contract Tests
# ============================================================================

class TestToolSchemas:
    """Lock in tool schemas so CLI flag drift is caught."""

    def _get_tool(self, name: str) -> Dict[str, Any]:
        for t in TOOLS:
            if t["name"] == name:
                return t
        raise KeyError(f"Tool {name} not found")

    def _get_props(self, name: str) -> Dict[str, Any]:
        return self._get_tool(name)["inputSchema"].get("properties", {})

    def _get_required(self, name: str) -> List[str]:
        return self._get_tool(name)["inputSchema"].get("required", [])

    def test_create_test_schema(self):
        props = self._get_props("create_test")
        required = self._get_required("create_test")
        assert "name" in required
        assert "query" in required
        assert props["name"]["type"] == "string"
        assert props["query"]["type"] == "string"
        assert props["expected_tools"]["type"] == "array"
        assert props["forbidden_tools"]["type"] == "array"
        assert props["min_score"]["type"] == "number"

    def test_run_check_schema(self):
        props = self._get_props("run_check")
        assert props["heal"]["type"] == "boolean"
        assert props["strict"]["type"] == "boolean"
        assert props["ai_root_cause"]["type"] == "boolean"
        assert props["statistical"]["type"] == "integer"
        assert props["auto_variant"]["type"] == "boolean"
        assert props["budget"]["type"] == "number"
        assert props["dry_run"]["type"] == "boolean"
        assert props["tag"]["type"] == "array"
        assert props["fail_on"]["type"] == "string"
        assert props["timeout"]["type"] == "number"
        assert props["report"]["type"] == "string"
        assert props["judge"]["type"] == "string"
        # No required params — bare run_check is valid
        assert self._get_required("run_check") == []

    def test_run_snapshot_schema(self):
        props = self._get_props("run_snapshot")
        assert props["variant"]["type"] == "string"
        assert props["preview"]["type"] == "boolean"
        assert props["reset"]["type"] == "boolean"
        assert props["judge"]["type"] == "string"
        assert props["timeout"]["type"] == "number"

    def test_compare_agents_schema(self):
        required = self._get_required("compare_agents")
        assert "v1" in required
        assert "v2" in required
        props = self._get_props("compare_agents")
        assert props["v1"]["type"] == "string"
        assert props["v2"]["type"] == "string"
        assert props["no_judge"]["type"] == "boolean"

    def test_replay_schema(self):
        props = self._get_props("replay")
        assert props["test_name"]["type"] == "string"
        assert props["test_path"]["type"] == "string"
        assert props["no_browser"]["type"] == "boolean"


# ============================================================================
# CLI Flag Wiring Tests (subprocess mocking)
# ============================================================================

class TestCheckFlagWiring:
    """Verify run_check flags map to correct CLI args."""

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_heal_flag(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout='{"summary":{}}', stderr="", returncode=0)
        server = MCPServer()
        server._run_check_subprocess({"heal": True})
        cmd = mock_run.call_args[0][0]
        assert "--heal" in cmd
        assert "--json" in cmd

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_strict_flag(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout='{}', stderr="", returncode=0)
        server = MCPServer()
        server._run_check_subprocess({"strict": True})
        cmd = mock_run.call_args[0][0]
        assert "--strict" in cmd
        assert "--json" in cmd

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_single_string_tag_is_one_tag(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout='{}', stderr="", returncode=0)
        server = MCPServer()
        server._run_check_subprocess({"tag": "tool_use"})
        cmd = mock_run.call_args[0][0]
        assert cmd.count("--tag") == 1
        assert cmd[cmd.index("--tag") + 1] == "tool_use"

    @patch("evalview.api.gate", side_effect=RuntimeError("api down"))
    @patch(
        "evalview.mcp_server.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="evalview check", timeout=60),
    )
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_direct_check_fallback_failure_reports_both_causes(
        self, mock_which, mock_run, mock_gate
    ):
        server = MCPServer()
        with pytest.raises(ToolError) as excinfo:
            server._run_check_direct({})
        assert "api down" in str(excinfo.value)
        assert "timed out" in str(excinfo.value)

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_all_flags(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout='{}', stderr="", returncode=0)
        server = MCPServer()
        server._run_check_subprocess({
            "test": "my-test",
            "heal": True,
            "strict": True,
            "ai_root_cause": True,
            "statistical": 5,
            "auto_variant": True,
            "budget": 0.50,
            "dry_run": True,
            "tag": ["tool_use", "retrieval"],
            "fail_on": "REGRESSION,TOOLS_CHANGED",
            "timeout": 60,
            "report": "/tmp/report.html",
            "judge": "gpt-5",
        })
        cmd = mock_run.call_args[0][0]
        assert "--json" in cmd
        assert "--heal" in cmd
        assert "--strict" in cmd
        assert "--ai-root-cause" in cmd
        assert "--statistical" in cmd and "5" in cmd
        assert "--auto-variant" in cmd
        assert "--budget" in cmd and "0.5" in cmd
        assert "--dry-run" in cmd
        assert cmd.count("--tag") == 2
        assert "tool_use" in cmd
        assert "retrieval" in cmd
        assert "--fail-on" in cmd
        assert "REGRESSION,TOOLS_CHANGED" in cmd
        assert "--timeout" in cmd and "60" in cmd
        assert "--report" in cmd
        assert "--judge" in cmd and "gpt-5" in cmd

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_ci_env_set(self, mock_which, mock_run):
        """CI=1 must be set to prevent browser auto-open from --report."""
        mock_run.return_value = MagicMock(stdout='{}', stderr="", returncode=0)
        server = MCPServer()
        server._run_check_subprocess({"report": "/tmp/r.html"})
        env = mock_run.call_args[1].get("env") or mock_run.call_args[0][3]
        assert env.get("CI") == "1"

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_stdin_devnull(self, mock_which, mock_run):
        """subprocess must not hang waiting for stdin."""
        mock_run.return_value = MagicMock(stdout='{}', stderr="", returncode=0)
        server = MCPServer()
        server._run_check_subprocess({"heal": True})
        assert mock_run.call_args[1].get("stdin") == subprocess.DEVNULL


class TestCheckRouting:
    """Verify run_check routes to direct vs subprocess correctly."""

    @patch.object(MCPServer, "_run_check_direct", return_value='{"routed":"direct"}')
    @patch.object(MCPServer, "_run_check_subprocess", return_value='{"routed":"subprocess"}')
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_bare_check_uses_direct(self, mock_which, mock_sub, mock_direct):
        server = MCPServer()
        server._call_tool("run_check", {})
        mock_direct.assert_called_once()
        mock_sub.assert_not_called()

    @patch.object(MCPServer, "_run_check_direct", return_value='{"routed":"direct"}')
    @patch.object(MCPServer, "_run_check_subprocess", return_value='{"routed":"subprocess"}')
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_heal_routes_to_subprocess(self, mock_which, mock_sub, mock_direct):
        server = MCPServer()
        server._call_tool("run_check", {"heal": True})
        mock_sub.assert_called_once()
        mock_direct.assert_not_called()

    @patch.object(MCPServer, "_run_check_direct", return_value='{"routed":"direct"}')
    @patch.object(MCPServer, "_run_check_subprocess", return_value='{"routed":"subprocess"}')
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_tag_routes_to_subprocess(self, mock_which, mock_sub, mock_direct):
        server = MCPServer()
        server._call_tool("run_check", {"tag": ["tool_use"]})
        mock_sub.assert_called_once()


class TestSnapshotFlagWiring:
    """Verify run_snapshot flags map to correct CLI args."""

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_snapshot_all_flags(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout="Snapshot saved", stderr="", returncode=0)
        server = MCPServer()
        server._call_tool("run_snapshot", {
            "test": "my-test",
            "notes": "after refactor",
            "variant": "v2",
            "preview": True,
            "reset": True,
            "judge": "sonnet",
            "timeout": 45,
        })
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "evalview"
        assert cmd[1] == "snapshot"
        assert "--path" in cmd
        assert "--test" in cmd and "my-test" in cmd
        assert "--notes" in cmd and "after refactor" in cmd
        assert "--variant" in cmd and "v2" in cmd
        assert "--preview" in cmd
        assert "--reset" in cmd
        assert "--judge" in cmd and "sonnet" in cmd
        assert "--timeout" in cmd and "45" in cmd


class TestCompareFlagWiring:
    """Verify compare_agents flags map to correct CLI args."""

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_compare_all_flags(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout="Report generated", stderr="", returncode=0)
        server = MCPServer()
        server._call_tool("compare_agents", {
            "v1": "http://localhost:8000",
            "v2": "http://localhost:8001",
            "tests": "tests/",
            "adapter": "http",
            "label_v1": "old",
            "label_v2": "new",
            "no_judge": True,
        })
        cmd = mock_run.call_args[0][0]
        assert "--v1" in cmd and "http://localhost:8000" in cmd
        assert "--v2" in cmd and "http://localhost:8001" in cmd
        assert "--no-open" in cmd  # must not open browser in MCP
        assert "--adapter" in cmd and "http" in cmd
        assert "--label-v1" in cmd and "old" in cmd
        assert "--label-v2" in cmd and "new" in cmd
        assert "--no-judge" in cmd

    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_compare_missing_endpoints(self, mock_which):
        server = MCPServer()
        with pytest.raises(ToolError, match="'v1' and 'v2'"):
            server._call_tool("compare_agents", {"v1": "http://localhost:8000"})

    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_compare_missing_both(self, mock_which):
        server = MCPServer()
        with pytest.raises(ToolError, match="'v1' and 'v2'"):
            server._call_tool("compare_agents", {})


class TestReplayFlagWiring:
    """Verify replay flags map to correct CLI args."""

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_replay_with_test_name(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout="Report at /tmp/r.html", stderr="", returncode=0)
        server = MCPServer()
        server._call_tool("replay", {"test_name": "billing-test", "no_browser": True})
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "evalview"
        assert cmd[1] == "replay"
        assert "billing-test" in cmd
        assert "--no-browser" in cmd


# ============================================================================
# Timeout Tier Tests
# ============================================================================

class TestTimeoutTiers:
    """Verify commands get appropriate timeout values."""

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_heal_gets_300s(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout='{}', stderr="", returncode=0)
        server = MCPServer()
        server._run_check_subprocess({"heal": True})
        assert mock_run.call_args[1]["timeout"] == 300

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_statistical_gets_600s(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout='{}', stderr="", returncode=0)
        server = MCPServer()
        server._run_check_subprocess({"statistical": 10})
        assert mock_run.call_args[1]["timeout"] == 600

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_statistical_overrides_heal(self, mock_which, mock_run):
        """statistical (600s) should override heal (300s) when both are set."""
        mock_run.return_value = MagicMock(stdout='{}', stderr="", returncode=0)
        server = MCPServer()
        server._run_check_subprocess({"heal": True, "statistical": 5})
        assert mock_run.call_args[1]["timeout"] == 600

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_replay_gets_120s(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout="done", stderr="", returncode=0)
        server = MCPServer()
        server._call_tool("replay", {"test_name": "test1"})
        assert mock_run.call_args[1]["timeout"] == 120

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_golden_list_gets_30s(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout="No baselines", stderr="", returncode=0)
        server = MCPServer()
        server._call_tool("list_tests", {})
        assert mock_run.call_args[1]["timeout"] == 30


# ============================================================================
# create_test Tests (in-process, no subprocess)
# ============================================================================

class TestCreateTest:
    def test_creates_yaml_file(self, tmp_test_dir):
        server = MCPServer(test_path=tmp_test_dir)
        result = server._call_tool("create_test", {
            "name": "hello-world",
            "query": "Say hello",
            "expected_tools": ["greet"],
            "forbidden_tools": ["delete_all"],
            "expected_output_contains": ["hello"],
            "min_score": 80,
        })
        assert "Created" in result
        path = os.path.join(tmp_test_dir, "hello-world.yaml")
        test_case = TestCaseLoader.load_from_file(path)
        assert test_case.name == "hello-world"
        assert test_case.input.query == "Say hello"
        assert test_case.expected.tools == ["greet"]
        assert test_case.expected.forbidden_tools == ["delete_all"]
        assert test_case.expected.output.contains == ["hello"]
        assert test_case.thresholds.min_score == 80

    def test_minimal_test_loads(self, tmp_test_dir):
        # Only the required fields: the file must still be a valid test case.
        server = MCPServer(test_path=tmp_test_dir)
        server._call_tool("create_test", {"name": "minimal", "query": "Say hello"})
        test_case = TestCaseLoader.load_from_file(os.path.join(tmp_test_dir, "minimal.yaml"))
        assert test_case.input.query == "Say hello"
        assert test_case.expected.tools is None
        assert test_case.thresholds.min_score == 70

    @pytest.mark.parametrize(
        "query",
        [
            'Say "hi" to me',
            "line one\nline two",
            "key: value # not a comment",
            'x"\nthresholds:\n  min_score: 0\nz: "',
        ],
    )
    def test_special_characters_round_trip(self, tmp_test_dir, query):
        server = MCPServer(test_path=tmp_test_dir)
        server._call_tool("create_test", {
            "name": "special",
            "query": query,
            "description": 'has "quotes": and colons',
            "expected_output_contains": ['he said "ok"', "a: b"],
        })
        test_case = TestCaseLoader.load_from_file(os.path.join(tmp_test_dir, "special.yaml"))
        assert test_case.input.query == query
        assert test_case.description == 'has "quotes": and colons'
        assert test_case.expected.output.contains == ['he said "ok"', "a: b"]
        assert test_case.thresholds.min_score == 70

    @pytest.mark.parametrize("name", ["../escaped", "a/b", "café"])
    def test_name_the_loader_rejects_writes_nothing(self, tmp_test_dir, name):
        # "../escaped" would otherwise write outside test_path.
        server = MCPServer(test_path=os.path.join(tmp_test_dir, "tests"))
        with pytest.raises(ToolError, match="name must contain only"):
            server._call_tool("create_test", {"name": name, "query": "x"})
        assert os.listdir(tmp_test_dir) == []

    @pytest.mark.parametrize("name", ["_", "."])
    def test_name_without_letters_or_digits_is_rejected(self, tmp_test_dir, name):
        server = MCPServer(test_path=tmp_test_dir)
        with pytest.raises(ToolError, match="letter or digit"):
            server._call_tool("create_test", {"name": name, "query": "x"})

    @pytest.mark.parametrize("name", ["config", "Config"])
    def test_name_the_loader_skips_as_config_is_rejected(self, tmp_test_dir, name):
        server = MCPServer(test_path=tmp_test_dir)
        with pytest.raises(ToolError, match="skips as a config file"):
            server._call_tool("create_test", {"name": name, "query": "x"})
        assert os.listdir(tmp_test_dir) == []

    def test_non_ascii_text_is_written_as_ascii_and_round_trips(self, tmp_test_dir):
        # The loader opens files in the locale encoding (cp1252 on Windows).
        server = MCPServer(test_path=tmp_test_dir)
        query = "Où est le café? Ёлка"
        server._call_tool("create_test", {"name": "accents", "query": query})
        path = os.path.join(tmp_test_dir, "accents.yaml")
        with open(path, "rb") as f:
            f.read().decode("ascii")
        assert TestCaseLoader.load_from_file(path).input.query == query

    @pytest.mark.parametrize("min_score, expected", [(99.5, 99.5), (80.0, 80), ("75", 75), (None, 70)])
    def test_min_score_values(self, tmp_test_dir, min_score, expected):
        server = MCPServer(test_path=tmp_test_dir)
        server._call_tool("create_test", {"name": "scored", "query": "x", "min_score": min_score})
        test_case = TestCaseLoader.load_from_file(os.path.join(tmp_test_dir, "scored.yaml"))
        assert test_case.thresholds.min_score == expected

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need extra privileges on Windows")
    def test_existing_symlink_is_not_followed(self, tmp_test_dir):
        test_path = os.path.join(tmp_test_dir, "tests")
        os.makedirs(test_path)
        target = os.path.join(tmp_test_dir, "outside.txt")
        os.symlink(target, os.path.join(test_path, "linked.yaml"))  # dangling
        server = MCPServer(test_path=test_path)
        with pytest.raises(ToolError, match="already exists"):
            server._call_tool("create_test", {"name": "linked", "query": "x"})
        assert not os.path.exists(target)

    def test_single_string_tool_list_is_one_tool(self, tmp_test_dir):
        server = MCPServer(test_path=tmp_test_dir)
        server._call_tool("create_test", {"name": "one", "query": "x", "expected_tools": "calculator"})
        test_case = TestCaseLoader.load_from_file(os.path.join(tmp_test_dir, "one.yaml"))
        assert test_case.expected.tools == ["calculator"]

    @pytest.mark.parametrize(
        "bad_args, message",
        [
            ({"name": 5}, "'name' must be a string"),
            ({"description": ["x"]}, "'description' must be a string"),
            ({"expected_tools": [1]}, "'expected_tools' must be a list"),
            ({"forbidden_tools": [""]}, "'forbidden_tools' must be a list"),
            ({"expected_output_contains": {"a": 1}}, "'expected_output_contains' must be a list"),
            ({"min_score": "high"}, "'min_score' must be a number"),
            ({"min_score": 150}, "'min_score' must be a number"),
            ({"min_score": True}, "'min_score' must be a number"),
            ({"test_path": 3}, "'test_path' must be a string"),
        ],
    )
    def test_rejects_wrong_types(self, tmp_test_dir, bad_args, message):
        server = MCPServer(test_path=tmp_test_dir)
        with pytest.raises(ToolError, match=message):
            server._call_tool("create_test", {"name": "typed", "query": "x", **bad_args})
        assert os.listdir(tmp_test_dir) == []

    def test_rejects_duplicate(self, tmp_test_dir):
        server = MCPServer(test_path=tmp_test_dir)
        server._call_tool("create_test", {"name": "dup", "query": "test"})
        with pytest.raises(ToolError, match="already exists"):
            server._call_tool("create_test", {"name": "dup", "query": "test"})

    def test_rejects_missing_fields(self):
        server = MCPServer()
        for args in ({"name": "x"}, {"query": "x"}, {}):
            with pytest.raises(ToolError, match="'name' and 'query' are required"):
                server._call_tool("create_test", args)


# ============================================================================
# Evalview not in PATH
# ============================================================================

class TestMissingBinary:
    @patch("evalview.mcp_server.shutil.which", return_value=None)
    def test_returns_install_error(self, mock_which):
        server = MCPServer()
        with pytest.raises(ToolError, match="evalview not found in PATH. Run: pip install"):
            server._call_tool("list_tests", {})


class TestSkillToolArguments:
    @pytest.mark.parametrize(
        "tool, key",
        [
            ("validate_skill", "skill_path"),
            ("generate_skill_tests", "skill_path"),
            ("run_skill_test", "test_file"),
        ],
    )
    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_missing_path_is_rejected(self, mock_which, mock_run, tool, key):
        # normpath("") is ".", which used to slip past the required check.
        server = MCPServer()
        with pytest.raises(ToolError, match=f"'{key}' is required"):
            server._call_tool(tool, {})
        mock_run.assert_not_called()

    @patch("evalview.mcp_server.subprocess.run")
    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_path_is_normalized(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout="ok", stderr="", returncode=0)
        server = MCPServer()
        server._call_tool("validate_skill", {"skill_path": "skills/../skills/demo"})
        assert mock_run.call_args[0][0][-1] == os.path.normpath("skills/demo")
