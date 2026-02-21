"""EvalView MCP Server — exposes evalview check/snapshot as MCP tools for Claude Code."""

import json
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from typing import Any, Dict, Optional

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
            "Returns diff output showing what changed vs the last snapshot. "
            "A regression means the agent's behavior changed unexpectedly. "
            "Use this after refactoring agent code to confirm nothing broke. "
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
            },
        },
    },
    {
        "name": "run_snapshot",
        "description": (
            "Run tests and save passing results as the new golden baseline. "
            "Use this to establish or update the expected behavior after an intentional change. "
            "Future `run_check` calls will compare against this snapshot. "
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

    def _create_test(self, args: Dict[str, Any]) -> str:
        test_name = args.get("name", "").strip()
        query = args.get("query", "").strip()
        if not test_name or not query:
            return "Error: 'name' and 'query' are required."

        test_path = args.get("test_path", self.test_path)
        slug = test_name.lower().replace(" ", "-").replace("_", "-")
        filename = os.path.join(test_path, f"{slug}.yaml")

        if os.path.exists(filename):
            return f"Error: test already exists at {filename}. Delete it first or choose a different name."

        os.makedirs(test_path, exist_ok=True)

        lines = [f'name: "{test_name}"']

        description = args.get("description", "")
        if description:
            lines.append(f'description: "{description}"')

        lines += ["", "input:", f'  query: "{query}"', "", "expected:"]

        expected_tools = args.get("expected_tools", [])
        if expected_tools:
            lines.append("  tools:")
            for t in expected_tools:
                lines.append(f"    - {t}")

        expected_output = args.get("expected_output_contains", [])
        if expected_output:
            lines.append("  output:")
            lines.append("    contains:")
            for s in expected_output:
                lines.append(f'      - "{s}"')

        min_score = args.get("min_score", 70)
        lines += ["", "thresholds:", f"  min_score: {int(min_score)}"]

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        summary_parts = [f"query: {query}"]
        if expected_tools:
            summary_parts.append(f"tools: {', '.join(expected_tools)}")
        if expected_output:
            summary_parts.append(f"output contains: {', '.join(expected_output)}")

        return (
            f"Created {filename}\n"
            + "\n".join(f"  {p}" for p in summary_parts)
            + "\n\nRun run_snapshot to capture the baseline for this test."
        )

    def _call_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "create_test":
            return self._create_test(args)

        if not shutil.which("evalview"):
            return "Error: evalview not found in PATH. Run: pip install -e ."

        if name == "run_check":
            test_path = os.path.normpath(args.get("test_path", self.test_path))
            cmd = ["evalview", "check", test_path, "--json"]
            if args.get("test"):
                cmd += ["--test", args["test"]]

        elif name == "run_snapshot":
            test_path = os.path.normpath(args.get("test_path", self.test_path))
            cmd = ["evalview", "snapshot", test_path]
            if args.get("test"):
                cmd += ["--test", args["test"]]
            if args.get("notes"):
                cmd += ["--notes", args["notes"]]

        elif name == "list_tests":
            cmd = ["evalview", "golden", "list"]

        elif name == "validate_skill":
            skill_path = os.path.normpath(args.get("skill_path", ""))
            if not skill_path:
                return "Error: 'skill_path' is required."
            cmd = ["evalview", "skill", "validate", skill_path]

        elif name == "generate_skill_tests":
            skill_path = os.path.normpath(args.get("skill_path", ""))
            if not skill_path:
                return "Error: 'skill_path' is required."
            cmd = ["evalview", "skill", "generate-tests", skill_path, "--auto"]
            if args.get("output_path"):
                cmd += ["-o", os.path.normpath(args["output_path"])]
            if args.get("count"):
                cmd += ["-c", str(int(args["count"]))]

        elif name == "run_skill_test":
            test_file = os.path.normpath(args.get("test_file", ""))
            if not test_file:
                return "Error: 'test_file' is required."
            cmd = ["evalview", "skill", "test", test_file, "--json"]
            if args.get("agent"):
                cmd += ["--agent", args["agent"]]
            if args.get("no_rubric"):
                cmd += ["--no-rubric"]
            if args.get("model"):
                cmd += ["--model", args["model"]]
            if args.get("verbose"):
                cmd += ["--verbose"]

        elif name == "generate_visual_report":
            return self._generate_visual_report(args)

        else:
            return f"Unknown tool: {name}"

        env = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        output = result.stdout
        if result.stderr:
            output += result.stderr
        output = _ANSI_ESCAPE.sub("", output).strip()
        return output or f"Command exited with code {result.returncode}"

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
                return (
                    "No results found in .evalview/results/. "
                    "Run `evalview run` or `evalview snapshot` first."
                )
            results_file = files[-1]

        results_file = os.path.normpath(results_file)
        if not os.path.exists(results_file):
            return f"Results file not found: {results_file}"

        try:
            with open(results_file, encoding="utf-8") as f:
                raw = _json.load(f)
        except Exception as exc:
            return f"Failed to load results: {exc}"

        # Convert raw dicts to EvaluationResult objects
        try:
            from evalview.reporters.json_reporter import JSONReporter
            results = JSONReporter.load_as_results(results_file)
        except Exception:
            return "Failed to parse results — file may be in an unsupported format."

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
            return f"Failed to generate report: {exc}"
