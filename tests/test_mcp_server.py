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

from evalview.mcp_server import MCPServer, TOOLS, _EVALVIEW_VERSION


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
        result = server._call_tool("compare_agents", {"v1": "http://localhost:8000"})
        assert "Error" in result

    @patch("evalview.mcp_server.shutil.which", return_value="/usr/bin/evalview")
    def test_compare_missing_both(self, mock_which):
        server = MCPServer()
        result = server._call_tool("compare_agents", {})
        assert "Error" in result


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
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert 'name: "hello-world"' in content
        assert 'query: "Say hello"' in content
        assert "greet" in content
        assert "delete_all" in content
        assert "min_score: 80" in content

    def test_rejects_duplicate(self, tmp_test_dir):
        server = MCPServer(test_path=tmp_test_dir)
        server._call_tool("create_test", {"name": "dup", "query": "test"})
        result = server._call_tool("create_test", {"name": "dup", "query": "test"})
        assert "Error" in result
        assert "already exists" in result

    def test_rejects_missing_fields(self):
        server = MCPServer()
        assert "Error" in server._call_tool("create_test", {"name": "x"})
        assert "Error" in server._call_tool("create_test", {"query": "x"})
        assert "Error" in server._call_tool("create_test", {})


# ============================================================================
# Evalview not in PATH
# ============================================================================

class TestMissingBinary:
    @patch("evalview.mcp_server.shutil.which", return_value=None)
    def test_returns_install_error(self, mock_which):
        server = MCPServer()
        result = server._call_tool("list_tests", {})
        assert "evalview not found" in result
        assert "pip install" in result
