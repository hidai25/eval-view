"""EvalView MCP Server — exposes the full evalview CLI as MCP tools for Claude Code."""

import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from typing import Any, Dict, List, Optional

import yaml

# Matches CSI sequences (\x1b[...m), OSC sequences (\x1b]...\x07), and single-char escapes
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-9;]*[A-Za-z]|\][^\x07]*\x07|[@-_][0-`]?)")

try:
    _EVALVIEW_VERSION = _pkg_version("evalview")
except PackageNotFoundError:
    _EVALVIEW_VERSION = "dev"

TOOLS = [
    {
        "name": "create_test",
        "description": (
            "Create a new EvalView test case YAML file for an agent. "
            "Call this when the user asks to add a test, or when you want to capture "
            "expected agent behavior. After creating a test, call run_snapshot to establish "
            "the baseline. No YAML knowledge required — just describe the test. "
            "IMPORTANT: Automatically detect test_path by looking for a 'tests/evalview/' "
            "directory in the current project. If found, use it. Otherwise use 'tests'."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["name", "query"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Test name (e.g. 'calculator-division', 'weather-lookup')",
                },
                "query": {
                    "type": "string",
                    "description": "The input query to send to the agent",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of what this test covers",
                },
                "expected_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names the agent should call (e.g. ['calculator', 'search'])",
                },
                "forbidden_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tool names the agent must NEVER call. "
                        "Any violation is an immediate hard-fail (score=0, passed=false) "
                        "regardless of output quality. "
                        "Use this for safety contracts — e.g. a read-only agent that must "
                        "never call edit_file, bash, or write_file. "
                        "Matching is case-insensitive: 'EditFile' catches 'edit_file'."
                    ),
                },
                "expected_output_contains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Strings that must appear in the agent's output",
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum passing score 0-100 (default: 70)",
                },
                "test_path": {
                    "type": "string",
                    "description": (
                        "Directory to save the test file. "
                        "Auto-detect: use 'tests/evalview/' if it exists in the project, otherwise 'tests'."
                    ),
                },
            },
        },
    },
    {
        "name": "run_check",
        "description": (
            "Check for regressions against the golden baseline. "
            "Returns a diff summary for each test: PASSED, OUTPUT_CHANGED, TOOLS_CHANGED, or REGRESSION. "
            "REGRESSION means the score dropped significantly — treat this as a blocking failure. "
            "TOOLS_CHANGED / OUTPUT_CHANGED are warnings: the agent's behavior shifted but may be intentional. "
            "Also returns observability signals: behavioral anomalies (tool loops, stalls), "
            "trust scores (benchmark gaming detection), and coherence issues (multi-turn context loss). "
            "Use this after any code change (prompt, model, tools) to confirm nothing broke. "
            "If you see a regression, show the diff to the user and offer to fix it before moving on. "
            "Use heal=true to auto-retry flaky failures and distinguish non-determinism from real drift. "
            "IMPORTANT: Automatically detect test_path by looking for a 'tests/evalview/' "
            "directory in the current project. If it exists, pass it as test_path. "
            "If the project has a custom test location, use that instead."
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
                    "description": (
                        "Path to the test directory. "
                        "Auto-detect: use 'tests/evalview/' if it exists, otherwise 'tests'."
                    ),
                },
                "heal": {
                    "type": "boolean",
                    "description": "Auto-retry flaky failures, propose candidate variants, distinguish non-determinism from real regressions. Default: false.",
                },
                "strict": {
                    "type": "boolean",
                    "description": "Fail on any change (REGRESSION, TOOLS_CHANGED, OUTPUT_CHANGED). Default: false.",
                },
                "ai_root_cause": {
                    "type": "boolean",
                    "description": "Use AI to explain low-confidence regressions with root-cause analysis. Default: false.",
                },
                "statistical": {
                    "type": "integer",
                    "description": "Run each test N times for variance analysis (e.g. 5). Omit for single run.",
                },
                "auto_variant": {
                    "type": "boolean",
                    "description": "Auto-discover distinct execution paths as golden variants. Requires statistical. Default: false.",
                },
                "budget": {
                    "type": "number",
                    "description": "Maximum total budget in dollars (e.g. 0.50). Remaining tests skipped when limit hit.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview test plan and estimate cost without executing. Default: false.",
                },
                "tag": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Check only tests tagged with these behaviors (OR match). E.g. ['tool_use', 'retrieval'].",
                },
                "fail_on": {
                    "type": "string",
                    "description": "Comma-separated statuses to fail on (default: REGRESSION). E.g. 'REGRESSION,TOOLS_CHANGED'.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout per test in seconds (default: 120).",
                },
                "report": {
                    "type": "string",
                    "description": "Generate HTML report at this path (auto-opens in browser).",
                },
                "judge": {
                    "type": "string",
                    "description": "Judge model for scoring (e.g. 'gpt-5', 'sonnet').",
                },
            },
        },
    },
    {
        "name": "run_snapshot",
        "description": (
            "Run tests and save passing results as the new golden baseline. "
            "Use this to establish or update the expected behavior after an intentional change. "
            "Future `run_check` calls will compare against this snapshot. "
            "Call this: (1) after creating a new test with create_test, "
            "(2) after confirming a behavioral change is intentional, "
            "(3) before making large refactors so you have a clean rollback point. "
            "Only passing tests are saved — failing tests are skipped with a warning. "
            "IMPORTANT: Automatically detect test_path by looking for a 'tests/evalview/' "
            "directory in the current project. If it exists, pass it as test_path."
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
                    "description": (
                        "Path to the test directory. "
                        "Auto-detect: use 'tests/evalview/' if it exists, otherwise 'tests'."
                    ),
                },
                "variant": {
                    "type": "string",
                    "description": "Save as named variant for non-deterministic agents (max 5 per test). E.g. 'v2', 'async-path'.",
                },
                "preview": {
                    "type": "boolean",
                    "description": "Show what would change without saving (dry-run mode). Default: false.",
                },
                "reset": {
                    "type": "boolean",
                    "description": "Delete all existing baselines before capturing new ones. Default: false.",
                },
                "judge": {
                    "type": "string",
                    "description": "Judge model for scoring (e.g. 'gpt-5', 'sonnet').",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout per test in seconds (default: 30).",
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
    {
        "name": "validate_skill",
        "description": (
            "Validate a SKILL.md file for correct structure, naming conventions, and completeness. "
            "Call this after writing or editing a SKILL.md before running tests. "
            "Returns a list of issues found and whether the skill is valid."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["skill_path"],
            "properties": {
                "skill_path": {
                    "type": "string",
                    "description": "Path to the SKILL.md file or directory containing skills (e.g. '.claude/skills/my-skill/SKILL.md')",
                },
            },
        },
    },
    {
        "name": "generate_skill_tests",
        "description": (
            "Auto-generate test cases from a SKILL.md file. "
            "Call this when the user asks to create tests for a skill — it reads the skill "
            "definition and generates a ready-to-run YAML test suite covering explicit, "
            "implicit, contextual, and negative test categories. "
            "After generating, call run_skill_test to execute them."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["skill_path"],
            "properties": {
                "skill_path": {
                    "type": "string",
                    "description": "Path to the SKILL.md file to generate tests from",
                },
                "output_path": {
                    "type": "string",
                    "description": "Where to save the generated test YAML (default: same directory as SKILL.md)",
                },
                "count": {
                    "type": "number",
                    "description": "Number of test cases to generate (default: 10)",
                },
            },
        },
    },
    {
        "name": "run_skill_test",
        "description": (
            "Run a skill test suite against a SKILL.md. "
            "Executes two evaluation phases: "
            "Phase 1 (deterministic) checks tool calls, file operations, commands run, output content, and token budgets. "
            "Phase 2 (rubric) uses LLM-as-judge to score output quality against a defined rubric. "
            "Call this after writing skill tests or after any change to the skill or agent. "
            "Use --no-rubric for fast Phase 1-only checks with no LLM cost."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["test_file"],
            "properties": {
                "test_file": {
                    "type": "string",
                    "description": "Path to the skill test YAML file (e.g. 'tests/my-skill-tests.yaml')",
                },
                "agent": {
                    "type": "string",
                    "description": "Agent type to test against: 'claude-code', 'system-prompt', 'codex', 'langgraph', 'crewai', 'openai-assistants', 'custom'. Defaults to value in YAML.",
                },
                "no_rubric": {
                    "type": "boolean",
                    "description": "Skip Phase 2 rubric evaluation — run deterministic checks only (faster, no LLM cost). Default: false.",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use for evaluation (default: claude-sonnet-4-20250514)",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Show detailed output for all tests, not just failures. Default: false.",
                },
            },
        },
    },
    {
        "name": "generate_visual_report",
        "description": (
            "Generate a beautiful self-contained HTML visual report from the latest "
            "evalview check or run results. Opens automatically in the browser. "
            "Call this after run_check or run_snapshot to give the user a visual breakdown "
            "of traces, diffs, scores, and timelines. "
            "Returns the absolute path to the generated HTML file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "results_file": {
                    "type": "string",
                    "description": "Path to a specific results JSON file. If omitted, uses the latest file in .evalview/results/.",
                },
                "title": {
                    "type": "string",
                    "description": "Report title shown in the header (default: 'EvalView Report')",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional note shown in the report header (e.g. 'after refactor PR #42')",
                },
                "no_auto_open": {
                    "type": "boolean",
                    "description": "Set to true to suppress auto-opening the browser (useful in CI). Default: false.",
                },
            },
        },
    },
    {
        "name": "compare_agents",
        "description": (
            "Compare two agent endpoints side-by-side on the same test suite. "
            "Useful for A/B testing a new model, prompt change, or architecture swap. "
            "Returns per-test score deltas, tool diffs, and an optional HTML report."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["v1", "v2"],
            "properties": {
                "v1": {
                    "type": "string",
                    "description": "Baseline agent endpoint URL (e.g. 'http://localhost:8000/invoke')",
                },
                "v2": {
                    "type": "string",
                    "description": "Candidate agent endpoint URL (e.g. 'http://localhost:8001/invoke')",
                },
                "tests": {
                    "type": "string",
                    "description": "Test directory (default: 'tests/')",
                },
                "adapter": {
                    "type": "string",
                    "description": "Adapter type (http, langgraph, crewai, etc.). Default: from config or http.",
                },
                "label_v1": {
                    "type": "string",
                    "description": "Label for v1 in report (default: 'baseline')",
                },
                "label_v2": {
                    "type": "string",
                    "description": "Label for v2 in report (default: 'candidate')",
                },
                "no_judge": {
                    "type": "boolean",
                    "description": "Skip LLM-as-judge, deterministic scoring only. Default: false.",
                },
            },
        },
    },
    {
        "name": "replay",
        "description": (
            "Open a trajectory diff viewer for a specific test. "
            "Shows a side-by-side HTML comparison of baseline vs current behavior — "
            "tool calls, parameters, outputs, and score changes. "
            "Opens automatically in the browser."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "test_name": {
                    "type": "string",
                    "description": "Name of the test to replay (optional — shows latest if omitted)",
                },
                "test_path": {
                    "type": "string",
                    "description": "Path to test directory (default: 'tests/')",
                },
                "no_browser": {
                    "type": "boolean",
                    "description": "Don't auto-open HTML report in browser. Default: false.",
                },
            },
        },
    },
]


_TOOL_NAMES = {t["name"] for t in TOOLS}


class ToolError(Exception):
    """A tool call failed in a way the caller can fix; reported with ``isError: true``."""


def _jsonrpc_error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _log(message: str) -> None:
    # stdout carries the protocol, so diagnostics go to stderr (MCP hosts log it).
    sys.stderr.write(f"evalview mcp: {message}\n")
    sys.stderr.flush()


def _str_arg(args: Dict[str, Any], key: str) -> str:
    value = args.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ToolError(f"Error: '{key}' must be a string.")
    return value


def _str_list_arg(args: Dict[str, Any], key: str) -> List[str]:
    value = args.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]  # a single name sent without a list
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ToolError(f"Error: '{key}' must be a list of non-empty strings.")
    return value


def _min_score_arg(value: Any) -> int:
    message = "Error: 'min_score' must be a number from 0 to 100."
    if isinstance(value, bool):
        raise ToolError(message)
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ToolError(message) from None
    if not 0 <= score <= 100:
        raise ToolError(message)
    return int(score)


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
            # MCP requires a reply to carry the request's id. A line that is not
            # a JSON object has no id to answer, so it is logged and skipped.
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                _log(f"skipped a line that is not valid JSON: {line[:200]}")
                continue
            if not isinstance(request, dict):
                _log("skipped a message that is not a JSON object (batches are not supported)")
                continue
            try:
                response = self._handle(request)
            except Exception as exc:
                # One bad request must not take the whole server down.
                traceback.print_exc(file=sys.stderr)
                req_id = request.get("id")
                if req_id is None:
                    continue
                response = _jsonrpc_error(req_id, -32603, f"Internal error: {exc}")
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

    def _handle(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params")
        if params is None:
            params = {}

        if not isinstance(method, str):
            # A message without a method is either a response (this server sends
            # no requests, so there is nothing to match it to) or malformed.
            if req_id is None or "result" in req or "error" in req:
                return None
            return _jsonrpc_error(req_id, -32600, "Invalid Request: 'method' must be a string.")

        if not isinstance(params, dict):
            if req_id is None:
                return None
            return _jsonrpc_error(req_id, -32602, "Invalid params: 'params' must be an object.")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "evalview", "version": _EVALVIEW_VERSION},
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
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            is_error = False
            if not isinstance(arguments, dict):
                output = "Error: 'arguments' must be an object."
                is_error = True
            else:
                try:
                    output = self._call_tool(tool_name, arguments)
                except ToolError as exc:
                    output = str(exc)
                    is_error = True
                except Exception as exc:
                    output = f"Error: {tool_name} failed: {exc}"
                    is_error = True
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": output}],
                    "isError": is_error,
                },
            }

        # Unknown method — return error only if it has an id (i.e. it's a request not a notification)
        if req_id is not None:
            return _jsonrpc_error(req_id, -32601, f"Unknown method: {method}")

        return None

    def _create_test(self, args: Dict[str, Any]) -> str:
        test_name = _str_arg(args, "name").strip()
        query = _str_arg(args, "query").strip()
        if not test_name or not query:
            raise ToolError("Error: 'name' and 'query' are required.")
        description = _str_arg(args, "description").strip()
        expected_tools = _str_list_arg(args, "expected_tools")
        forbidden_tools = _str_list_arg(args, "forbidden_tools")
        expected_output = _str_list_arg(args, "expected_output_contains")
        min_score = _min_score_arg(args.get("min_score", 70))
        test_path = _str_arg(args, "test_path") or self.test_path

        expected: Dict[str, Any] = {}
        if expected_tools:
            expected["tools"] = expected_tools
        if forbidden_tools:
            # Any call to one of these is a hard fail.
            expected["forbidden_tools"] = forbidden_tools
        if expected_output:
            expected["output"] = {"contains": expected_output}

        test_case: Dict[str, Any] = {"name": test_name}
        if description:
            test_case["description"] = description
        test_case["input"] = {"query": query}
        test_case["expected"] = expected
        test_case["thresholds"] = {"min_score": min_score}

        # Validate with the model the loader uses, so the file always loads. Its
        # name rule also keeps path separators out of the filename below.
        from pydantic import ValidationError

        from evalview.core.types import TestCase

        try:
            TestCase(**test_case)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            raise ToolError(f"Error: invalid test case: {problems}") from None

        slug = test_name.lower().replace(" ", "-").replace("_", "-")
        filename = os.path.join(test_path, f"{slug}.yaml")
        if os.path.exists(filename):
            raise ToolError(
                f"Error: test already exists at {filename}. Delete it first or choose a different name."
            )

        os.makedirs(test_path, exist_ok=True)
        # safe_dump quotes and escapes values, so quotes, colons or newlines in
        # them can't break the file or add keys to it.
        with open(filename, "w", encoding="utf-8") as f:
            yaml.safe_dump(test_case, f, sort_keys=False, allow_unicode=True)

        summary_parts = [f"query: {query}"]
        if expected_tools:
            summary_parts.append(f"tools: {', '.join(expected_tools)}")
        if forbidden_tools:
            summary_parts.append(f"forbidden_tools (hard-fail): {', '.join(forbidden_tools)}")
        if expected_output:
            summary_parts.append(f"output contains: {', '.join(expected_output)}")

        return (
            f"Created {filename}\n"
            + "\n".join(f"  {p}" for p in summary_parts)
            + "\n\nRun run_snapshot to capture the baseline for this test."
        )

    def _call_tool(self, name: str, args: Dict[str, Any]) -> str:
        if not isinstance(name, str) or name not in _TOOL_NAMES:
            raise ToolError(f"Unknown tool: {name}")

        if name == "create_test":
            return self._create_test(args)

        if not shutil.which("evalview"):
            raise ToolError("Error: evalview not found in PATH. Run: pip install -e .")

        if name == "run_check":
            # Use subprocess when advanced flags are present; direct API otherwise
            advanced_flags = ("heal", "strict", "ai_root_cause", "statistical",
                              "auto_variant", "budget", "dry_run", "tag",
                              "fail_on", "report", "judge")
            if any(args.get(f) for f in advanced_flags):
                return self._run_check_subprocess(args)
            return self._run_check_direct(args)

        elif name == "run_snapshot":
            test_path = os.path.normpath(args.get("test_path", self.test_path))
            cmd = ["evalview", "snapshot", "--path", test_path]
            if args.get("test"):
                cmd += ["--test", args["test"]]
            if args.get("notes"):
                cmd += ["--notes", args["notes"]]
            if args.get("variant"):
                cmd += ["--variant", args["variant"]]
            if args.get("preview") is True:
                cmd += ["--preview"]
            if args.get("reset") is True:
                cmd += ["--reset"]
            if args.get("judge"):
                cmd += ["--judge", args["judge"]]
            if args.get("timeout"):
                cmd += ["--timeout", str(args["timeout"])]

        elif name == "list_tests":
            cmd = ["evalview", "golden", "list"]

        elif name == "validate_skill":
            skill_path = os.path.normpath(args.get("skill_path", ""))
            if not skill_path:
                raise ToolError("Error: 'skill_path' is required.")
            cmd = ["evalview", "skill", "validate", skill_path]

        elif name == "generate_skill_tests":
            skill_path = os.path.normpath(args.get("skill_path", ""))
            if not skill_path:
                raise ToolError("Error: 'skill_path' is required.")
            cmd = ["evalview", "skill", "generate-tests", skill_path, "--auto"]
            if args.get("output_path"):
                cmd += ["-o", os.path.normpath(args["output_path"])]
            if args.get("count"):
                cmd += ["-c", str(int(args["count"]))]

        elif name == "run_skill_test":
            test_file = os.path.normpath(args.get("test_file", ""))
            if not test_file:
                raise ToolError("Error: 'test_file' is required.")
            cmd = ["evalview", "skill", "test", test_file, "--json"]
            if args.get("agent"):
                cmd += ["--agent", args["agent"]]
            if args.get("no_rubric") is True:
                cmd += ["--no-rubric"]
            if args.get("model"):
                cmd += ["--model", args["model"]]
            if args.get("verbose") is True:
                cmd += ["--verbose"]

        elif name == "generate_visual_report":
            return self._generate_visual_report(args)

        elif name == "compare_agents":
            v1 = args.get("v1", "")
            v2 = args.get("v2", "")
            if not v1 or not v2:
                raise ToolError("Error: 'v1' and 'v2' endpoint URLs are required.")
            cmd = ["evalview", "compare", "--v1", v1, "--v2", v2, "--no-open"]
            if args.get("tests"):
                cmd += ["--tests", os.path.normpath(args["tests"])]
            if args.get("adapter"):
                cmd += ["--adapter", args["adapter"]]
            if args.get("label_v1"):
                cmd += ["--label-v1", args["label_v1"]]
            if args.get("label_v2"):
                cmd += ["--label-v2", args["label_v2"]]
            if args.get("no_judge") is True:
                cmd += ["--no-judge"]

        elif name == "replay":
            cmd = ["evalview", "replay"]
            if args.get("test_name"):
                cmd += [args["test_name"]]
            if args.get("test_path"):
                cmd += ["--test-path", os.path.normpath(args["test_path"])]
            if not args.get("no_browser"):
                # Default to opening browser for replay (visual tool)
                pass
            else:
                cmd += ["--no-browser"]

        else:
            raise ToolError(f"Unknown tool: {name}")

        env = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"}
        # Timeout tiers based on the subcommand, not substring matching on args:
        # - skill test: 600s (N tests × ~15s each can easily exceed 2 min)
        # - generate-tests / snapshot / run: 120s (single LLM call)
        # - everything else (check, golden list, validate): 30s
        subcommands = cmd[1:3]  # e.g. ["skill", "test"] or ["snapshot"]
        if subcommands == ["skill", "test"]:
            timeout = 600
        elif any(s in subcommands for s in ("generate-tests", "snapshot", "run", "compare", "replay")):
            timeout = 120
        else:
            timeout = 30
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env,
                timeout=timeout, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"Error: Command timed out after {timeout}s.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Tip: Check that your API key is set and the model is reachable."
            ) from None
        output = result.stdout
        if result.stderr:
            output += result.stderr
        output = _ANSI_ESCAPE.sub("", output).strip()
        if not output and result.returncode != 0:
            raise ToolError(f"Command exited with code {result.returncode}")
        return output or f"Command exited with code {result.returncode}"

    def _run_check_subprocess(self, args: Dict[str, Any]) -> str:
        """Run regression check via subprocess with full flag support.

        Always forces --json so the output contract matches the direct API path.
        HTML reports are generated as a side effect but never auto-opened (CI=1).
        """
        test_path = os.path.normpath(args.get("test_path", self.test_path))
        # check takes test_path as a positional argument; always --json for stable contract
        cmd = ["evalview", "check", test_path, "--json"]
        if args.get("test"):
            cmd += ["--test", args["test"]]
        if args.get("heal") is True:
            cmd += ["--heal"]
        if args.get("strict") is True:
            cmd += ["--strict"]
        if args.get("ai_root_cause") is True:
            cmd += ["--ai-root-cause"]
        if args.get("statistical"):
            cmd += ["--statistical", str(int(args["statistical"]))]
        if args.get("auto_variant") is True:
            cmd += ["--auto-variant"]
        if args.get("budget"):
            cmd += ["--budget", str(args["budget"])]
        if args.get("dry_run") is True:
            cmd += ["--dry-run"]
        if args.get("tag"):
            for t in args["tag"]:
                cmd += ["--tag", t]
        if args.get("fail_on"):
            cmd += ["--fail-on", args["fail_on"]]
        if args.get("timeout"):
            cmd += ["--timeout", str(args["timeout"])]
        if args.get("report"):
            cmd += ["--report", os.path.normpath(args["report"])]
        if args.get("judge"):
            cmd += ["--judge", args["judge"]]
        # Timeout tiers: statistical > heal > default
        proc_timeout = 120
        if args.get("heal"):
            proc_timeout = 300
        if args.get("statistical"):
            proc_timeout = 600
        # CI=1 suppresses browser auto-open from --report
        env = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0", "CI": "1"}
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env,
                timeout=proc_timeout, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"Error: Command timed out after {proc_timeout}s.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Tip: Heal and statistical modes take longer — try fewer tests or a higher timeout."
            ) from None
        output = result.stdout
        if result.stderr:
            output += result.stderr
        output = _ANSI_ESCAPE.sub("", output).strip()
        if not output and result.returncode != 0:
            raise ToolError(f"Command exited with code {result.returncode}")
        return output or f"Command exited with code {result.returncode}"

    def _run_check_direct(self, args: Dict[str, Any]) -> str:
        """Run regression check via the Python API instead of subprocess.

        Calls evalview.api.gate() directly — no subprocess overhead, no output
        parsing, no ANSI stripping.  Returns the same JSON format that
        ``evalview check --json`` would produce.
        """
        try:
            from evalview.api import gate, DiffStatus

            test_path = os.path.normpath(args.get("test_path", self.test_path))
            test_name = args.get("test") or None

            user_timeout = float(args.get("timeout", 120.0))
            result = gate(
                test_dir=test_path,
                test_name=test_name,
                fail_on={DiffStatus.REGRESSION},
                timeout=user_timeout,
            )

            # Build the same JSON structure as `evalview check --json`
            output = {
                "summary": {
                    "total_tests": result.summary.total,
                    "unchanged": result.summary.unchanged,
                    "regressions": result.summary.regressions,
                    "tools_changed": result.summary.tools_changed,
                    "output_changed": result.summary.output_changed,
                    "all_passed": result.summary.regressions == 0 and result.summary.tools_changed == 0 and result.summary.output_changed == 0,
                    "has_regressions": result.summary.regressions > 0,
                    "has_tools_changed": result.summary.tools_changed > 0,
                    "has_output_changed": result.summary.output_changed > 0,
                },
                "diffs": [
                    {
                        "test_name": d.test_name,
                        "status": d.status.value,
                        "score_delta": d.score_delta,
                        "has_tool_diffs": d.tool_changes > 0,
                        "output_similarity": d.output_similarity,
                        "model_changed": d.model_changed,
                    }
                    for d in result.diffs
                ],
            }

            # Include observability signals when present
            obs = result.observability
            payload = obs.to_payload()
            if payload:
                output["observability"] = payload

            return json.dumps(output, indent=2)

        except Exception as e:
            # Fall back to subprocess if the API call fails for any reason
            test_path = os.path.normpath(args.get("test_path", self.test_path))
            cmd = ["evalview", "check", test_path, "--json"]
            if args.get("test"):
                cmd += ["--test", args["test"]]
            env = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"}
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60, stdin=subprocess.DEVNULL)
            except Exception:
                raise ToolError(f"Error running check: {e}") from None
            fallback_output = _ANSI_ESCAPE.sub("", proc.stdout + (proc.stderr or "")).strip()
            if not fallback_output:
                raise ToolError(f"Error: {e}")
            return fallback_output

    def _generate_visual_report(self, args: Dict[str, Any]) -> str:
        """Generate a beautiful HTML visual report from results JSON."""
        import glob
        import json as _json

        # Resolve results file
        results_file = args.get("results_file", "")
        if not results_file:
            # Find latest in .evalview/results/
            pattern = ".evalview/results/*.json"
            files = sorted(glob.glob(pattern))
            if not files:
                raise ToolError(
                    "No results found in .evalview/results/. "
                    "Run `evalview run` or `evalview snapshot` first."
                )
            results_file = files[-1]

        results_file = os.path.normpath(results_file)
        if not os.path.exists(results_file):
            raise ToolError(f"Results file not found: {results_file}")

        try:
            with open(results_file, encoding="utf-8") as f:
                _json.load(f)
        except Exception as exc:
            raise ToolError(f"Failed to load results: {exc}") from exc

        # Convert raw dicts to EvaluationResult objects
        try:
            from evalview.reporters.json_reporter import JSONReporter
            results = JSONReporter.load_as_results(results_file)
        except Exception as exc:
            raise ToolError(
                f"Failed to parse results — file may be in an unsupported format. ({exc})"
            ) from exc

        try:
            from evalview.visualization import generate_visual_report
            path = generate_visual_report(
                results=results,
                diffs=None,
                title=args.get("title", "EvalView Report"),
                notes=args.get("notes", ""),
                auto_open=not args.get("no_auto_open", False),
            )
            total = len(results)
            passed = sum(1 for r in results if r.passed)
            return (
                f"Report generated: {path}\n"
                f"{passed}/{total} tests passing\n\n"
                f"Open in browser or share the HTML file."
            )
        except Exception as exc:
            raise ToolError(f"Failed to generate report: {exc}") from exc
