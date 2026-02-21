"""CLI entry point for EvalView."""

import asyncio
import json
import os
import re
import sys
import threading
import time
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

try:
    _EVALVIEW_VERSION = _pkg_version("evalview")
except PackageNotFoundError:
    _EVALVIEW_VERSION = "dev"
import click
import httpx
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from evalview.core.loader import TestCaseLoader
from evalview.core.pricing import get_model_pricing_info
from evalview.core.llm_provider import (
    detect_available_providers,
    get_or_select_provider,
    save_provider_preference,
    PROVIDER_CONFIGS,
    judge_cost_tracker,
)
from evalview.adapters.http_adapter import HTTPAdapter
from evalview.adapters.tapescope_adapter import TapeScopeAdapter
from evalview.adapters.langgraph_adapter import LangGraphAdapter
from evalview.adapters.crewai_adapter import CrewAIAdapter
from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter
from evalview.evaluators.evaluator import Evaluator
from evalview.reporters.json_reporter import JSONReporter
from evalview.reporters.console_reporter import ConsoleReporter

# Telemetry (lazy imports for optional dependency)
from evalview.telemetry.config import (
    is_telemetry_enabled,
    should_show_first_run_notice,
    mark_first_run_notice_shown,
    set_telemetry_enabled,
    load_config as load_telemetry_config,
    TELEMETRY_DISABLED_ENV,
)
from evalview.telemetry.decorators import track_command, track_run_command

# Skills module imports
from evalview.skills.constants import (
    CLAUDE_CODE_CHAR_BUDGET,
    AVG_CHARS_PER_SKILL,
    SCORE_THRESHOLD_HIGH,
    SCORE_THRESHOLD_MEDIUM,
    TRUNCATE_OUTPUT_SHORT,
    TRUNCATE_OUTPUT_MEDIUM,
    TRUNCATE_OUTPUT_LONG,
    CHAR_BUDGET_WARNING_PCT,
    CHAR_BUDGET_CRITICAL_PCT,
    MAX_DESCRIPTION_LENGTH,
    MAX_PREVIEW_LINES,
)
from evalview.skills.ui_utils import print_evalview_banner, format_elapsed_time, run_async_with_spinner
from evalview.skills.test_helpers import (
    validate_and_parse_agent_type,
    load_test_suite,
    print_suite_info,
    format_results_as_json,
    build_results_table,
    print_detailed_test_results,
    build_summary_panel,
    handle_test_completion,
)

# Load environment variables (.env is the OSS standard, .env.local for overrides)
load_dotenv()  # Loads .env by default
load_dotenv(dotenv_path=".env.local", override=True)  # Override with .env.local if exists

console = Console()


# Helper Functions
def _create_adapter(adapter_type: str, endpoint: str, timeout: float = 30.0, allow_private_urls: bool = True) -> "AgentAdapter":
    """Factory function for creating adapters based on type.

    Args:
        adapter_type: Type of adapter ("http", "langgraph", "tapescope", "crewai", "openai")
        endpoint: API endpoint URL
        timeout: Request timeout in seconds
        allow_private_urls: If True, allow requests to private/internal networks (default: True for local dev)

    Returns:
        Adapter instance

    Raises:
        ValueError: If adapter_type is unknown
    """
    adapter_map = {
        "http": HTTPAdapter,
        "langgraph": LangGraphAdapter,
        "tapescope": TapeScopeAdapter,
        "crewai": CrewAIAdapter,
        "openai": OpenAIAssistantsAdapter,
    }

    adapter_class = adapter_map.get(adapter_type)
    if not adapter_class:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    if adapter_type == "http":
        return adapter_class(endpoint=endpoint, timeout=timeout, allow_private_urls=allow_private_urls)
    return adapter_class(endpoint=endpoint, timeout=timeout)


@click.group(context_settings={"allow_interspersed_args": False})
@click.version_option(version=_EVALVIEW_VERSION)
@click.pass_context
def main(ctx):
    """EvalView — Proof that your agent still works.

    \b
    Check Your Agent:
      run                     Check agent health
      run --diff              Compare against golden baseline
      run --save-golden       Save passing results as baseline
      demo                    See regression detection in action
      quickstart              Set up a working example in 2 minutes

    \b
    Golden Traces:
      golden save <file>      Save a known-good baseline
      golden list             List saved baselines
      golden show <name>      View baseline details

    \b
    Explore & Learn:
      chat                    Interactive AI assistant for eval guidance
      gym                     Practice agent eval patterns

    \b
    Reports:
      report <file>           Generate report from results
      view                    View specific test results
      trends                  Performance trends over time

    \b
    CI/CD:
      ci comment              Post results to a GitHub PR
      init --ci               Generate GitHub Actions workflow

    \b
    Advanced:
      skill                   Test Claude Code skills
      trace                   Trace LLM calls in scripts
      traces                  Query stored trace data
      expand                  Generate test variations with LLM
    """
    # Show first-run telemetry notice (once only)
    if should_show_first_run_notice():
        # Don't show for telemetry subcommands themselves
        if ctx.invoked_subcommand not in ("telemetry",):
            console.print()
            console.print("[dim]╭──────────────────────────────────────────────────────────────╮[/dim]")
            console.print("[dim]│[/dim] EvalView collects anonymous usage data to improve the tool. [dim]│[/dim]")
            console.print("[dim]│[/dim] No personal info or test content is collected.              [dim]│[/dim]")
            console.print("[dim]│[/dim] Disable with: [cyan]evalview telemetry off[/cyan]                      [dim]│[/dim]")
            console.print("[dim]╰──────────────────────────────────────────────────────────────╯[/dim]")
            console.print()
            mark_first_run_notice_shown()


@main.command()
@click.option(
    "--dir",
    default=".",
    help="Directory to initialize (default: current directory)",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Interactive setup (default: True)",
)
@click.option(
    "--wizard",
    is_flag=True,
    help="[EXPERIMENTAL] Run auto-detection wizard to find and configure agents",
)
@click.option(
    "--ci",
    is_flag=True,
    help="Generate a GitHub Actions workflow for running EvalView in CI.",
)
@track_command("init", lambda **kw: {"ci": kw.get("ci", False)})
def init(dir: str, interactive: bool, wizard: bool, ci: bool):
    """Initialize EvalView in the current directory."""
    if ci:
        _init_ci_workflow(dir)
        return

    if wizard:
        asyncio.run(_init_wizard_async(dir))
        return

    _init_standard(dir, interactive)


def _init_ci_workflow(dir: str):
    """Generate a GitHub Actions workflow for EvalView."""
    base_path = Path(dir)
    workflow_dir = base_path / ".github" / "workflows"
    workflow_file = workflow_dir / "evalview.yml"

    if workflow_file.exists():
        console.print(f"[yellow]Workflow already exists: {workflow_file}[/yellow]")
        console.print("[dim]Delete it first if you want to regenerate.[/dim]\n")
        return

    workflow_dir.mkdir(parents=True, exist_ok=True)

    workflow_content = """name: Agent Health Check

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evalview:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install EvalView
        run: pip install evalview

      - name: Check agent health
        run: evalview run --diff --save-golden
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
"""

    workflow_file.write_text(workflow_content)

    console.print("[green]✓ GitHub Actions workflow created[/green]")
    console.print(f"  {workflow_file}\n")
    console.print("[dim]Next steps:[/dim]")
    console.print("[dim]  1. Add OPENAI_API_KEY to your repo secrets (optional — works without it)[/dim]")
    console.print("[dim]  2. Commit and push to trigger the workflow[/dim]")
    console.print("[dim]  3. EvalView will check agent health on every PR[/dim]\n")


def _init_standard(dir: str, interactive: bool):
    """Standard init flow (non-wizard)."""
    console.print("[blue]━━━ EvalView Setup ━━━[/blue]\n")

    base_path = Path(dir)

    # Create directories
    (base_path / ".evalview").mkdir(exist_ok=True)
    (base_path / "tests" / "test-cases").mkdir(parents=True, exist_ok=True)

    # Interactive configuration
    adapter_type = "http"
    endpoint = "http://localhost:3000/api/agent"
    timeout = 30.0
    model_name = "gpt-5-mini"
    custom_pricing = None

    if interactive:
        console.print("[bold]Step 1: API Configuration[/bold]")

        # Ask adapter type
        console.print("\nWhat type of API does your agent use?")
        console.print("  1. Standard REST API (returns complete JSON)")
        console.print("  2. Streaming API (JSONL/Server-Sent Events)")
        adapter_choice = click.prompt("Choice", type=int, default=1)
        adapter_type = "streaming" if adapter_choice == 2 else "http"

        # Ask endpoint
        endpoint = click.prompt("\nAPI endpoint URL", default=endpoint)
        timeout = click.prompt("Timeout (seconds)", type=float, default=timeout)

        console.print("\n[bold]Step 2: Model & Pricing Configuration[/bold]")
        console.print("\nWhich model does your agent use?")
        console.print("  1. gpt-5-mini (recommended for testing)")
        console.print("  2. gpt-5")
        console.print("  3. gpt-5-nano")
        console.print("  4. gpt-4o or gpt-4o-mini")
        console.print("  5. Custom model")

        model_choice = click.prompt("Choice", type=int, default=1)

        model_map = {
            1: "gpt-5-mini",
            2: "gpt-5",
            3: "gpt-5-nano",
            4: "gpt-4o-mini",
        }

        if model_choice == 5:
            model_name = click.prompt("Model name")
        else:
            model_name = model_map.get(model_choice, "gpt-5-mini")

        # Show pricing
        pricing = get_model_pricing_info(model_name)
        console.print(f"\n[cyan]Pricing for {model_name}:[/cyan]")
        console.print(f"  • Input tokens:  ${pricing['input_price_per_1m']:.2f} per 1M tokens")
        console.print(f"  • Output tokens: ${pricing['output_price_per_1m']:.2f} per 1M tokens")
        console.print(f"  • Cached tokens: ${pricing['cached_price_per_1m']:.3f} per 1M tokens")

        # Ask if pricing is correct
        if click.confirm("\nIs this pricing correct for your use case?", default=True):
            console.print("[green]✅ Using standard pricing[/green]")
        else:
            console.print("\n[yellow]Let's set custom pricing:[/yellow]")
            input_price = click.prompt(
                "Input tokens ($ per 1M)", type=float, default=pricing["input_price_per_1m"]
            )
            output_price = click.prompt(
                "Output tokens ($ per 1M)", type=float, default=pricing["output_price_per_1m"]
            )
            cached_price = click.prompt(
                "Cached tokens ($ per 1M)", type=float, default=pricing["cached_price_per_1m"]
            )

            custom_pricing = {
                "input": input_price,
                "output": output_price,
                "cached": cached_price,
            }
            console.print("[green]✅ Custom pricing saved[/green]")

    # Create config file
    config_path = base_path / ".evalview" / "config.yaml"
    if not config_path.exists():
        config_content = f"""# EvalView Configuration
adapter: {adapter_type}
endpoint: {endpoint}
timeout: {timeout}
headers: {{}}

# Model configuration
model:
  name: {model_name}
"""
        if custom_pricing:
            config_content += f"""  pricing:
    input_per_1m: {custom_pricing['input']}
    output_per_1m: {custom_pricing['output']}
    cached_per_1m: {custom_pricing['cached']}
"""
        else:
            config_content += """  # Uses standard OpenAI pricing
  # Override with custom pricing if needed:
  # pricing:
  #   input_per_1m: 0.25
  #   output_per_1m: 2.0
  #   cached_per_1m: 0.025
"""

        config_path.write_text(config_content)
        console.print("\n[green]✅ Created .evalview/config.yaml[/green]")
    else:
        console.print("\n[yellow]⚠️  .evalview/config.yaml already exists[/yellow]")

    # Create example test case (simple calculator that works with the demo agent)
    example_path = base_path / "tests" / "test-cases" / "example.yaml"
    if not example_path.exists():
        example_content = """name: "Hello World - Calculator"
description: "Simple test to verify EvalView is working"

input:
  query: "What is 2 plus 3?"

expected:
  tools:
    - calculator
  output:
    contains:
      - "5"
    not_contains:
      - "error"

thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 5000
"""
        example_path.write_text(example_content)
        console.print("[green]✅ Created tests/test-cases/example.yaml[/green]")
    else:
        console.print("[yellow]⚠️  tests/test-cases/example.yaml already exists[/yellow]")

    # Create demo agent directory and files
    demo_agent_dir = base_path / "demo-agent"
    if not demo_agent_dir.exists():
        demo_agent_dir.mkdir(exist_ok=True)

        # Create the demo agent
        demo_agent_content = '''"""
EvalView Demo Agent - A simple FastAPI agent for testing.

Run with: python demo-agent/agent.py
Then test with: evalview run
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import time
import re

app = FastAPI(title="EvalView Demo Agent")


class Message(BaseModel):
    role: str
    content: str


class ExecuteRequest(BaseModel):
    # Support both formats:
    # 1. EvalView HTTPAdapter format: {"query": "...", "context": {...}}
    # 2. OpenAI-style format: {"messages": [...]}
    query: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    messages: Optional[List[Message]] = None
    enable_tracing: bool = True


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result: Any


class ExecuteResponse(BaseModel):
    output: str
    tool_calls: List[ToolCall]
    cost: float
    latency: float


def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations."""
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b != 0 else 0}
    return ops.get(operation, 0)


def get_weather(city: str) -> Dict[str, Any]:
    """Get weather for a city."""
    weather = {
        "new york": {"temp": 72, "condition": "sunny"},
        "london": {"temp": 55, "condition": "rainy"},
        "tokyo": {"temp": 68, "condition": "cloudy"},
    }
    return weather.get(city.lower(), {"error": f"City '{city}' not found"})


def simple_agent(query: str) -> tuple:
    """Simple rule-based agent logic."""
    query_lower = query.lower()
    tool_calls = []
    cost = 0.001

    # Calculator queries
    if any(op in query_lower for op in ["plus", "add", "+", "sum"]):
        numbers = re.findall(r"\\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("add", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "add", "a": a, "b": b}, result=result))
            return f"The result of {a} + {b} = {result}", tool_calls, cost

    elif any(op in query_lower for op in ["minus", "subtract", "-"]):
        numbers = re.findall(r"\\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("subtract", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "subtract", "a": a, "b": b}, result=result))
            return f"The result of {a} - {b} = {result}", tool_calls, cost

    elif any(op in query_lower for op in ["times", "multiply", "*"]):
        numbers = re.findall(r"\\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("multiply", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "multiply", "a": a, "b": b}, result=result))
            return f"The result of {a} * {b} = {result}", tool_calls, cost

    elif any(op in query_lower for op in ["divided", "divide", "/"]):
        numbers = re.findall(r"\\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("divide", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "divide", "a": a, "b": b}, result=result))
            return f"The result of {a} / {b} = {result}", tool_calls, cost

    # Weather queries
    elif "weather" in query_lower:
        for city in ["new york", "london", "tokyo"]:
            if city in query_lower:
                result = get_weather(city)
                tool_calls.append(ToolCall(name="get_weather", arguments={"city": city}, result=result))
                return f"Weather in {city.title()}: {result['temp']}°F, {result['condition']}", tool_calls, cost

    return f"I received your query: {query}", tool_calls, cost


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    """Execute agent with given messages."""
    start = time.time()

    # Support both request formats
    if request.query:
        query = request.query
    elif request.messages:
        user_msgs = [m for m in request.messages if m.role == "user"]
        if not user_msgs:
            raise HTTPException(status_code=400, detail="No user message")
        query = user_msgs[-1].content
    else:
        raise HTTPException(status_code=400, detail="Either query or messages must be provided")

    output, tools, cost = simple_agent(query)
    return ExecuteResponse(output=output, tool_calls=tools, cost=cost, latency=(time.time() - start) * 1000)


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    print("🚀 Demo Agent running on http://localhost:8000")
    print("📖 API docs: http://localhost:8000/docs")
    print("\\n💡 Test with: evalview run")
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        (demo_agent_dir / "agent.py").write_text(demo_agent_content)

        # Create requirements.txt for the demo agent
        demo_requirements = """fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
"""
        (demo_agent_dir / "requirements.txt").write_text(demo_requirements)

        console.print("[green]✅ Created demo-agent/ with working example agent[/green]")
    else:
        console.print("[yellow]⚠️  demo-agent/ already exists[/yellow]")

    console.print("\n[blue]━━━ Quick Start (2 minutes) ━━━[/blue]")
    console.print("\n[bold]1. Start the demo agent:[/bold]")
    console.print("   [cyan]pip install fastapi uvicorn[/cyan]")
    console.print("   [cyan]python demo-agent/agent.py[/cyan]")
    console.print("\n[bold]2. In another terminal, set an API key (any one):[/bold]")
    console.print("   [cyan]export ANTHROPIC_API_KEY='your-key'[/cyan]  [dim]# or OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY[/dim]")
    console.print("\n[bold]3. Run tests:[/bold]")
    console.print("   [cyan]evalview run[/cyan]")
    console.print("\n[dim]The demo agent runs on http://localhost:8000[/dim]")
    console.print("[dim]Edit tests/test-cases/example.yaml to add more tests[/dim]\n")


@main.command()
@track_command("quickstart")
def quickstart():
    """🚀 Quick start: Set up and run a demo in under 2 minutes."""
    import subprocess
    import atexit

    console.print("[blue]━━━ EvalView Quickstart ━━━[/blue]\n")
    console.print("This will set up a working demo in under 2 minutes.\n")

    base_path = Path(".")

    # Step 1: Create demo agent if it doesn't exist
    demo_agent_dir = base_path / "demo-agent"
    if not demo_agent_dir.exists():
        console.print("[bold]Step 1/4:[/bold] Creating demo agent...")
        _create_demo_agent(base_path)
        console.print("[green]✅ Demo agent created[/green]\n")
    else:
        console.print("[bold]Step 1/4:[/bold] Demo agent already exists\n")

    # Step 2: Create test cases if they don't exist
    test_dir = base_path / "tests" / "test-cases"
    test_dir.mkdir(parents=True, exist_ok=True)

    test_files = [
        ("01-calculator.yaml", """name: "Calculator Test"
description: "Division test - tests basic tool calling"

input:
  query: "What is 144 divided by 12?"

expected:
  tools:
    - calculator
  output:
    contains:
      - "12"

thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 5000
"""),
        ("02-weather.yaml", """name: "Weather Test"
description: "Weather query test - tests single tool with structured output"

input:
  query: "What's the weather in Tokyo?"

expected:
  tools:
    - get_weather
  output:
    contains:
      - "Tokyo"
      - "22"

thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 5000
"""),
        ("03-multi-tool.yaml", """name: "Multi-Tool Test"
description: "Multi-tool sequence test - tests weather lookup + temperature conversion"

input:
  query: "What's the weather in London in Fahrenheit?"

expected:
  tools:
    - get_weather
    - calculator
  tool_sequence:
    - get_weather
    - calculator
  output:
    contains:
      - "London"
      - "F"

thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 5000
"""),
        ("04-multiplication.yaml", """name: "Multiplication Test"
description: "Tests multiplication operation"

input:
  query: "What is 25 times 4?"

expected:
  tools:
    - calculator
  output:
    contains:
      - "100"

thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 5000
"""),
    ]

    created_tests = False
    for filename, content in test_files:
        test_file = test_dir / filename
        if not test_file.exists():
            if not created_tests:
                console.print("[bold]Step 2/4:[/bold] Creating test cases...")
                created_tests = True
            test_file.write_text(content)

    if created_tests:
        console.print(f"[green]✅ {len(test_files)} test cases created[/green]\n")
    else:
        console.print("[bold]Step 2/4:[/bold] Test cases already exist\n")

    # Step 3: Create config for demo agent
    config_dir = base_path / ".evalview"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        console.print("[bold]Step 3/4:[/bold] Creating config...")
        config_content = """# EvalView Quickstart Config
adapter: http
endpoint: http://localhost:8000/execute
timeout: 30.0
headers: {}
allow_private_urls: true  # Allow localhost for demo agent

model:
  name: gpt-4o-mini
"""
        config_file.write_text(config_content)
        console.print("[green]✅ Config created[/green]\n")
    else:
        console.print("[bold]Step 3/4:[/bold] Config already exists\n")

    # Check for any LLM provider API key (not just Ollama)
    # We need an actual cloud API key for reliable LLM-as-judge evaluation
    has_api_key = any([
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
        os.getenv("XAI_API_KEY"),
    ])
    use_deterministic_scoring = not has_api_key

    try:
        from evalview.telemetry.client import get_client as _tc
        from evalview.telemetry.events import CommandEvent as _CE
        _tc().track(_CE(
            command_name="quickstart_setup_complete",
            properties={"has_api_key": has_api_key},
        ))
    except Exception:
        pass

    if use_deterministic_scoring:
        console.print("[yellow]⚠️  No LLM provider API key found[/yellow]")
        console.print("[dim]   Using deterministic scoring (string matching + tool assertions)[/dim]")
        console.print("[dim]   For full LLM-as-judge evaluation, set: export ANTHROPIC_API_KEY='...'[/dim]\n")

    # Step 4: Start demo agent and run test
    console.print("[bold]Step 4/4:[/bold] Starting demo agent and running test...\n")

    # Check if dependencies are installed
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        console.print("[yellow]Installing demo agent dependencies...[/yellow]")
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"],
                      capture_output=True, check=True)
        console.print("[green]✅ Dependencies installed[/green]\n")

    # Start the demo agent in background
    console.print("[dim]Starting demo agent on http://localhost:8000...[/dim]")
    agent_process = subprocess.Popen(
        [sys.executable, str(demo_agent_dir / "agent.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Register cleanup
    def cleanup():
        agent_process.terminate()
        try:
            agent_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            agent_process.kill()

    atexit.register(cleanup)

    # Wait for agent to start
    import time
    console.print("[dim]Waiting for agent to be ready...[/dim]")
    for _ in range(10):
        time.sleep(0.5)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000/health", timeout=1)
            break
        except Exception:
            continue
    else:
        console.print("[red]❌ Demo agent failed to start[/red]")
        cleanup()
        return

    console.print("[green]✅ Demo agent running[/green]\n")

    try:
        from evalview.telemetry.client import get_client as _tc
        from evalview.telemetry.events import CommandEvent as _CE
        _tc().track(_CE(
            command_name="quickstart_agent_ready",
            properties={"has_api_key": has_api_key},
        ))
    except Exception:
        pass

    # Welcome banner
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗██╗   ██╗ █████╗ ██╗    ██╗   ██╗██╗███████╗██╗    ██╗[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔════╝██║   ██║██╔══██╗██║    ██║   ██║██║██╔════╝██║    ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]█████╗  ██║   ██║███████║██║    ██║   ██║██║█████╗  ██║ █╗ ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔══╝  ╚██╗ ██╔╝██╔══██║██║    ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗ ╚████╔╝ ██║  ██║███████╗╚████╔╝ ██║███████╗╚███╔███╔╝[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ [/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]        [dim]Catch agent regressions before you ship[/dim]               [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print()

    # Run all tests
    console.print("[bold]Running tests...[/bold]\n")
    try:
        # Import and run the tests programmatically
        import time as time_module
        from rich.live import Live
        from rich.panel import Panel
        from evalview.core.loader import TestCaseLoader
        from evalview.adapters.http_adapter import HTTPAdapter
        from evalview.evaluators.evaluator import Evaluator

        # Load all test cases
        test_cases = TestCaseLoader.load_from_directory(test_dir)
        adapter = HTTPAdapter(
            endpoint="http://localhost:8000/execute",
            headers={},
            timeout=30.0,
            allow_private_urls=True,  # Allow localhost for demo
        )
        evaluator = Evaluator(skip_llm_judge=use_deterministic_scoring)

        # Timer and tracking
        start_time = time_module.time()
        passed = 0
        failed = 0
        tests_completed = 0
        current_test = ""
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0

        def format_elapsed():
            elapsed = time_module.time() - start_time
            mins, secs = divmod(elapsed, 60)
            secs_int = int(secs)
            ms = int((secs - secs_int) * 1000)
            return f"{int(mins):02d}:{secs_int:02d}.{ms:03d}"

        def get_status_display():
            nonlocal spinner_idx
            spinner = spinner_frames[spinner_idx % len(spinner_frames)]
            spinner_idx += 1
            test_display = f"  [yellow]{spinner}[/yellow] [dim]{current_test}...[/dim]" if current_test else f"  [yellow]{spinner}[/yellow] [dim]Starting...[/dim]"

            # Status indicator
            if failed > 0:
                status = "[bold red]● Running[/bold red]"
            else:
                status = "[green]● Running[/green]"

            content = (
                f"  {status}\n"
                f"\n"
                f"  [bold]⏱️  Elapsed:[/bold]    [yellow]{format_elapsed()}[/yellow]\n"
                f"  [bold]📋 Progress:[/bold]   {tests_completed}/{len(test_cases)} tests\n"
                f"\n"
                f"{test_display}\n"
                f"\n"
                f"  [green]✓ Passed:[/green] {passed}    [red]✗ Failed:[/red] {failed}"
            )

            border = "red" if failed > 0 else "cyan"
            return Panel(content, title="[bold]Test Execution[/bold]", border_style=border, padding=(0, 1))

        async def run_all_tests():
            nonlocal passed, failed, tests_completed, current_test
            results = []
            score_suffix = "*" if use_deterministic_scoring else ""
            for test_case in sorted(test_cases, key=lambda t: t.name):
                current_test = test_case.name[:30]
                trace = await adapter.execute(test_case.input.query, test_case.input.context)
                result = await evaluator.evaluate(test_case, trace)
                result.adapter_name = adapter.name
                results.append(result)
                if result.passed:
                    passed += 1
                    console.print(f"[green]✅ {test_case.name} - PASSED (score: {result.score}{score_suffix})[/green]")
                else:
                    failed += 1
                    console.print(f"[red]❌ {test_case.name} - FAILED (score: {result.score}{score_suffix})[/red]")
                tests_completed += 1
            current_test = ""
            return results

        # Run with live display
        if sys.stdin.isatty():
            with Live(get_status_display(), console=console, refresh_per_second=10) as live:
                async def run_with_display():
                    task = asyncio.create_task(run_all_tests())
                    while not task.done():
                        live.update(get_status_display())
                        await asyncio.sleep(0.1)
                    return await task

                results = asyncio.run(run_with_display())

            # Final box
            final_elapsed = format_elapsed()
            console.print()
            console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            if failed == 0:
                console.print(f"[bold cyan]║[/bold cyan]  [bold green]✓ AGENT HEALTHY[/bold green]                                               [bold cyan]║[/bold cyan]")
            else:
                console.print(f"[bold cyan]║[/bold cyan]  [bold red]✗ REGRESSION DETECTED[/bold red]                                        [bold cyan]║[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            console.print(f"[bold cyan]║[/bold cyan]  [green]✓ Passed:[/green] {passed:<4}  [red]✗ Failed:[/red] {failed:<4}  [dim]Time:[/dim] {final_elapsed}               [bold cyan]║[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
            if use_deterministic_scoring:
                console.print()
                console.print("[dim]* Deterministic mode: scores capped at 75, no LLM judge.[/dim]")
                console.print("[dim]  For production scoring, set ANTHROPIC_API_KEY or OPENAI_API_KEY.[/dim]")
            console.print()
        else:
            results = asyncio.run(run_all_tests())

        # Use ConsoleReporter for proper table display
        from evalview.reporters.console_reporter import ConsoleReporter
        reporter = ConsoleReporter()
        reporter.print_summary(results)

        passed = sum(1 for r in results if r.passed)

        try:
            from evalview.telemetry.client import get_client as _tc
            from evalview.telemetry.events import CommandEvent as _CE
            _tc().track(_CE(
                command_name="quickstart_run_complete",
                properties={
                    "passed": passed,
                    "failed": len(results) - passed,
                    "total": len(results),
                    "all_passed": passed == len(results),
                    "deterministic_scoring": use_deterministic_scoring,
                    "has_api_key": has_api_key,
                },
            ))
        except Exception:
            pass

        if passed == len(results):
            console.print("\n[green bold]🎉 All tests passed! Quickstart complete![/green bold]")
        else:
            console.print("\n[yellow]Some tests failed. Check the output above for details.[/yellow]")

        console.print("\n[dim]Note: Cost/tokens shown are mock data from the demo agent.[/dim]")
        console.print("[dim]Your real agent will report actual LLM usage.[/dim]")

        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Connect your real agent:")
        console.print("     [cyan]evalview connect[/cyan]  ← Auto-detect running agents")
        console.print("     [dim]or edit .evalview/config.yaml manually[/dim]")
        console.print("  2. Write or generate test cases:")
        console.print("     [cyan]evalview expand your-test.yaml --count 20[/cyan]  ← generate variations from a seed")
        console.print("     [cyan]evalview record --interactive[/cyan]              ← record live sessions as tests")
        console.print("     [dim]or write YAML manually in tests/[/dim]")
        console.print("  3. Run [cyan]evalview run[/cyan] for full results")
        console.print("  4. See regression detection in action:")
        console.print("     [cyan]evalview demo[/cyan]  ← snapshot + break + catch, live in 30 seconds")

        console.print()
        console.print("[dim]⭐ EvalView helped? Star us: [link=https://github.com/hidai25/eval-view]github.com/hidai25/eval-view[/link][/dim]\n")

    except Exception as e:
        console.print(f"[red]❌ Tests failed: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


def _create_demo_agent(base_path: Path):
    """Create the demo agent files."""
    demo_agent_dir = base_path / "demo-agent"
    demo_agent_dir.mkdir(exist_ok=True)

    demo_agent_content = '''"""
EvalView Demo Agent - A simple FastAPI agent for testing.
Supports calculator and weather tools with multi-tool sequences.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import time
import re

app = FastAPI(title="EvalView Demo Agent")


class Message(BaseModel):
    role: str
    content: str


class ExecuteRequest(BaseModel):
    query: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    messages: Optional[List[Message]] = None
    enable_tracing: bool = True


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result: Any
    latency: float = 0.0
    cost: float = 0.0


class ExecuteResponse(BaseModel):
    output: str
    tool_calls: List[ToolCall]
    cost: float
    latency: float
    tokens: Optional[Dict[str, int]] = None


def calculator(operation: str, a: float, b: float) -> float:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b != 0 else 0}
    return ops.get(operation, 0)


def get_weather(city: str) -> Dict[str, Any]:
    weather_db = {
        "tokyo": {"temp": 22, "condition": "cloudy", "humidity": 70},
        "london": {"temp": 12, "condition": "rainy", "humidity": 85},
        "new york": {"temp": 18, "condition": "sunny", "humidity": 60},
        "paris": {"temp": 15, "condition": "partly cloudy", "humidity": 72},
        "sydney": {"temp": 25, "condition": "sunny", "humidity": 55},
    }
    return weather_db.get(city.lower(), {"temp": 20, "condition": "partly cloudy", "humidity": 65})


def simple_agent(query: str) -> tuple:
    query_lower = query.lower()
    tool_calls = []
    total_cost = 0.0
    time.sleep(0.015)  # Simulate LLM processing

    # Math operations
    if any(op in query_lower for op in ["plus", "add", "+", "sum"]):
        numbers = re.findall(r"\\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("add", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "add", "a": a, "b": b}, result=result, cost=0.001))
            return f"The result of {a} + {b} = {result}", tool_calls, 0.001

    elif any(op in query_lower for op in ["minus", "subtract", "-"]):
        numbers = re.findall(r"\\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("subtract", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "subtract", "a": a, "b": b}, result=result, cost=0.001))
            return f"The result of {a} - {b} = {result}", tool_calls, 0.001

    elif any(op in query_lower for op in ["times", "multiply", "*"]):
        numbers = re.findall(r"\\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("multiply", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "multiply", "a": a, "b": b}, result=result, cost=0.001))
            return f"The result of {a} * {b} = {result}", tool_calls, 0.001

    elif any(op in query_lower for op in ["divided", "divide", "/"]):
        numbers = re.findall(r"\\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("divide", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "divide", "a": a, "b": b}, result=result, cost=0.001))
            return f"The result of {a} / {b} = {result}", tool_calls, 0.001

    # Weather + Fahrenheit conversion (multi-tool)
    elif "weather" in query_lower and "fahrenheit" in query_lower:
        city = "tokyo"
        for c in ["tokyo", "london", "new york", "paris", "sydney"]:
            if c in query_lower:
                city = c
                break
        weather = get_weather(city)
        temp_c = weather["temp"]
        tool_calls.append(ToolCall(name="get_weather", arguments={"city": city}, result=weather, cost=0.001))
        temp_f = calculator("multiply", temp_c, 1.8)
        tool_calls.append(ToolCall(name="calculator", arguments={"operation": "multiply", "a": temp_c, "b": 1.8}, result=temp_f, cost=0.001))
        temp_f = calculator("add", temp_f, 32)
        tool_calls.append(ToolCall(name="calculator", arguments={"operation": "add", "a": temp_f - 32, "b": 32}, result=temp_f, cost=0.001))
        return f"The weather in {city.title()} is {temp_c}C ({temp_f:.1f}F), {weather['condition']}", tool_calls, 0.003

    # Simple weather query
    elif "weather" in query_lower:
        city = "tokyo"
        for c in ["tokyo", "london", "new york", "paris", "sydney"]:
            if c in query_lower:
                city = c
                break
        weather = get_weather(city)
        tool_calls.append(ToolCall(name="get_weather", arguments={"city": city}, result=weather, cost=0.001))
        return f"The weather in {city.title()} is {weather['temp']}C, {weather['condition']} with {weather['humidity']}% humidity", tool_calls, 0.001

    return f"I received your query: {query}", tool_calls, 0.0


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    start = time.time()
    if request.query:
        query = request.query
    elif request.messages:
        user_msgs = [m for m in request.messages if m.role == "user"]
        if not user_msgs:
            raise HTTPException(status_code=400, detail="No user message")
        query = user_msgs[-1].content
    else:
        raise HTTPException(status_code=400, detail="Either query or messages must be provided")

    output, tools, cost = simple_agent(query)
    total_latency = (time.time() - start) * 1000
    if tools:
        per_step = total_latency / len(tools)
        tools = [ToolCall(name=t.name, arguments=t.arguments, result=t.result, latency=per_step, cost=t.cost) for t in tools]
    tokens = {"input": 50 + len(query), "output": 80 + len(output), "cached": 0}
    return ExecuteResponse(output=output, tool_calls=tools, cost=cost, latency=total_latency, tokens=tokens)


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    print("Demo Agent running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
'''
    (demo_agent_dir / "agent.py").write_text(demo_agent_content)

    requirements = "fastapi>=0.100.0\nuvicorn>=0.23.0\npydantic>=2.0.0\n"
    (demo_agent_dir / "requirements.txt").write_text(requirements)


async def _init_wizard_async(dir: str):
    """Interactive wizard to auto-detect and configure agents."""

    console.print("[blue]━━━ EvalView Setup Wizard ━━━[/blue]\n")
    console.print("[cyan]🔍 Auto-detecting agent servers...[/cyan]\n")

    base_path = Path(dir)

    # Create directories
    (base_path / ".evalview").mkdir(exist_ok=True)
    (base_path / "tests" / "test-cases").mkdir(parents=True, exist_ok=True)

    # Common ports and endpoints to scan
    common_ports = [8000, 2024, 3000, 8080, 5000, 8888, 7860]
    common_patterns = [
        ("langgraph", "LangGraph Cloud", "/ok", "GET"),
        ("langgraph", "LangGraph Cloud", "/info", "GET"),
        ("langgraph", "LangGraph", "/invoke", "POST"),
        ("langgraph", "LangGraph", "/api/chat", "POST"),
        ("http", "LangServe", "/agent", "POST"),
        ("streaming", "LangServe Streaming", "/agent/stream", "POST"),
        ("streaming", "TapeScope", "/api/unifiedchat", "POST"),
        ("crewai", "CrewAI", "/crew", "POST"),
        ("http", "FastAPI", "/api/agent", "POST"),
        ("http", "FastAPI", "/chat", "POST"),
    ]

    detected_agents = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning ports...", total=None)

        async with httpx.AsyncClient(timeout=2.0) as client:
            for port in common_ports:
                progress.update(task, description=f"Scanning port {port}...")

                for adapter_type, framework_name, path, method in common_patterns:
                    url = f"http://127.0.0.1:{port}{path}"

                    try:
                        if method == "GET":
                            response = await client.get(url)
                        else:
                            response = await client.post(
                                url,
                                json={
                                    "query": "test",
                                    "message": "test",
                                    "messages": [{"role": "user", "content": "test"}],
                                },
                                headers={"Content-Type": "application/json"},
                            )

                        if response.status_code in [200, 201, 422]:
                            content_type = response.headers.get("content-type", "")
                            if not content_type.startswith("application/json"):
                                continue

                            # Try to detect actual adapter from response
                            detected_adapter = adapter_type
                            response_info = {}
                            try:
                                data = response.json()
                                response_info = {"keys": list(data.keys())[:5]}

                                # Refine detection based on response
                                if "messages" in data or "thread_id" in data:
                                    detected_adapter = "langgraph"
                                elif "tasks" in data or "crew_id" in data or "crew" in data:
                                    detected_adapter = "crewai"
                            except Exception:
                                pass

                            # For LangGraph Cloud health endpoints, use base URL
                            endpoint_url = url
                            if detected_adapter == "langgraph" and (
                                path == "/ok" or path == "/info"
                            ):
                                endpoint_url = f"http://127.0.0.1:{port}"

                            detected_agents.append({
                                "port": port,
                                "path": path,
                                "url": endpoint_url,
                                "adapter": detected_adapter,
                                "framework": framework_name,
                                "response_info": response_info,
                            })

                    except (httpx.ConnectError, httpx.TimeoutException, Exception):
                        continue

    # Show results
    if not detected_agents:
        console.print("[yellow]⚠️  No agent servers detected.[/yellow]\n")
        console.print("Make sure your agent server is running on one of these ports:")
        console.print(f"  {', '.join(str(p) for p in common_ports)}\n")
        console.print("[blue]To start a LangGraph agent:[/blue]")
        console.print("  langgraph dev  # Runs on port 2024")
        console.print()
        console.print("[blue]Or run standard init:[/blue]")
        console.print("  evalview init")
        return

    # Deduplicate by port (prefer more specific detections)
    unique_agents = {}
    for agent in detected_agents:
        port = agent["port"]
        # Prefer non-health-check endpoints
        if port not in unique_agents or agent["path"] not in ["/ok", "/info"]:
            unique_agents[port] = agent

    detected_agents = list(unique_agents.values())

    console.print(f"[green]✅ Found {len(detected_agents)} agent server(s)![/green]\n")

    # Show detected agents
    for i, agent in enumerate(detected_agents, 1):
        console.print(f"  [{i}] [bold]{agent['framework']}[/bold] on port {agent['port']}")
        console.print(f"      Endpoint: {agent['url']}")
        console.print(f"      Adapter: {agent['adapter']}")
        if agent.get("response_info", {}).get("keys"):
            console.print(f"      Response keys: {agent['response_info']['keys']}")
        console.print()

    # Let user choose if multiple detected
    selected_agent = detected_agents[0]
    if len(detected_agents) > 1:
        console.print("[bold]Which agent should EvalView connect to?[/bold]")
        choice = click.prompt(
            "Enter number",
            type=int,
            default=1,
        )
        if 1 <= choice <= len(detected_agents):
            selected_agent = detected_agents[choice - 1]

    console.print()
    console.print(f"[cyan]Configuring for {selected_agent['framework']}...[/cyan]\n")

    # Create config file
    config_path = base_path / ".evalview" / "config.yaml"
    config_content = f"""# EvalView Configuration
# Auto-generated by wizard

adapter: {selected_agent['adapter']}
endpoint: {selected_agent['url']}
timeout: 30.0
headers: {{}}

# Enable for local development (SSRF protection disabled)
allow_private_urls: true

# Model configuration
model:
  name: gpt-4o-mini
  # Uses standard OpenAI pricing
  # Override with custom pricing if needed:
  # pricing:
  #   input_per_1m: 0.15
  #   output_per_1m: 0.60
  #   cached_per_1m: 0.075
"""

    config_path.write_text(config_content)
    console.print("[green]✅ Created .evalview/config.yaml[/green]")

    # Create a sample test case tailored to the detected framework
    example_path = base_path / "tests" / "test-cases" / "example.yaml"
    if not example_path.exists():
        if selected_agent["adapter"] == "langgraph":
            example_content = """name: "LangGraph Basic Test"
description: "Test basic agent functionality"

input:
  query: "What is 2+2?"
  context: {}

expected:
  tools: []  # Add expected tools if your agent uses them
  output:
    contains:
      - "4"
    not_contains:
      - "error"

thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 10000
"""
        elif selected_agent["adapter"] == "crewai":
            example_content = """name: "CrewAI Basic Test"
description: "Test CrewAI agent execution"

input:
  query: "Research the weather in New York"
  context: {}

expected:
  tools: []  # CrewAI auto-detects tools from tasks
  output:
    contains:
      - "weather"
    not_contains:
      - "error"

thresholds:
  min_score: 70
  max_cost: 0.50
  max_latency: 60000  # CrewAI crews may take longer
"""
        else:
            example_content = """name: "Agent Basic Test"
description: "Test basic agent functionality"

input:
  query: "Hello, how are you?"
  context: {}

expected:
  tools: []
  output:
    contains: []
    not_contains:
      - "error"

thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 10000
"""
        example_path.write_text(example_content)
        console.print("[green]✅ Created tests/test-cases/example.yaml[/green]")
    else:
        console.print("[yellow]⚠️  tests/test-cases/example.yaml already exists[/yellow]")

    # Test connection
    console.print()
    if click.confirm("Test the connection now?", default=True):
        console.print("\n[cyan]Testing connection...[/cyan]")

        try:
            # Import adapter registry
            from evalview.adapters.registry import AdapterRegistry

            test_adapter = AdapterRegistry.create(
                name=selected_agent["adapter"],
                endpoint=selected_agent["url"],
                timeout=10.0,
                allow_private_urls=True,
            )

            trace = await test_adapter.execute("What is 2+2?")

            console.print("[green]✅ Connection successful![/green]\n")
            console.print(f"  Response: {trace.final_output[:100]}{'...' if len(trace.final_output) > 100 else ''}")
            console.print(f"  Steps: {len(trace.steps)}")
            console.print(f"  Latency: {trace.metrics.total_latency:.0f}ms")

        except Exception as e:
            console.print(f"[yellow]⚠️  Connection test failed: {e}[/yellow]")
            console.print("[dim]The config has been saved - you can fix the issue and try again.[/dim]")

    console.print()
    console.print("[blue]━━━ Setup Complete! ━━━[/blue]\n")
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Create tests:")
    console.print("     • [cyan]evalview record[/cyan]     ← Record agent interactions as tests")
    console.print("     • [cyan]evalview expand[/cyan]     ← Generate variations from a seed test")
    console.print("     • Or edit tests/test-cases/example.yaml")
    console.print("  2. Run: [cyan]evalview run[/cyan]")
    console.print()
    console.print("[dim]Tip: Use 'evalview validate-adapter --endpoint URL' to debug adapter issues[/dim]\n")


@main.command()
@click.argument("path", required=False, default=None)
@click.option(
    "--pattern",
    default="*.yaml",
    help="Test case file pattern (default: *.yaml)",
)
@click.option(
    "--test",
    "-t",
    multiple=True,
    help="Specific test name(s) to run (can specify multiple: -t test1 -t test2)",
)
@click.option(
    "--filter",
    "-f",
    help="Filter tests by name pattern (e.g., 'LangGraph*', '*simple*')",
)
@click.option(
    "--output",
    default=".evalview/results",
    help="Output directory for results",
)
@click.option(
    "--verbose/--no-verbose",
    default=True,
    help="Verbose output with full test details (default: enabled)",
)
@click.option(
    "--track",
    is_flag=True,
    help="Track results for regression analysis",
)
@click.option(
    "--compare-baseline",
    is_flag=True,
    help="Compare results against baseline and show regressions",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Show detailed debug info: raw API response, parsed trace, type conversions",
)
@click.option(
    "--sequential",
    is_flag=True,
    help="Run tests sequentially instead of in parallel (default: parallel)",
)
@click.option(
    "--max-workers",
    default=8,
    type=int,
    help="Maximum parallel test executions (default: 8)",
)
@click.option(
    "--max-retries",
    default=0,
    type=int,
    help="Maximum retries for flaky tests (default: 0 = no retries)",
)
@click.option(
    "--retry-delay",
    default=1.0,
    type=float,
    help="Base delay between retries in seconds (default: 1.0)",
)
@click.option(
    "--watch",
    is_flag=True,
    help="Watch test files and re-run on changes",
)
@click.option(
    "--html-report",
    type=click.Path(),
    help="Generate HTML report to specified path",
)
@click.option(
    "--summary",
    is_flag=True,
    help="Compact output with deltas vs last run and regression detection. Great for CI/CD and sharing.",
)
@click.option(
    "--coverage",
    is_flag=True,
    help="Show behavior coverage report: tasks tested, tools exercised, paths covered, eval dimensions.",
)
@click.option(
    "--judge-model",
    type=str,
    help="Model for LLM-as-judge (e.g., gpt-5, sonnet, llama-70b, gpt-4o). Aliases auto-resolve to full names.",
)
@click.option(
    "--judge-provider",
    type=click.Choice(["openai", "anthropic", "huggingface", "gemini", "grok", "ollama"]),
    help="Provider for LLM-as-judge evaluation (ollama = free local)",
)
@click.option(
    "--adapter",
    type=click.Choice(["http", "langgraph", "crewai", "anthropic", "openai-assistants", "tapescope", "huggingface", "goose", "ollama", "mcp"]),
    help="Override adapter type (e.g., goose, langgraph, mcp). Overrides config file.",
)
@click.option(
    "--diff",
    is_flag=True,
    help="Compare against golden baselines. Shows REGRESSION/TOOLS_CHANGED/OUTPUT_CHANGED/PASSED status.",
)
@click.option(
    "--diff-report",
    type=click.Path(),
    help="Generate HTML diff report to specified path (requires --diff)",
)
@click.option(
    "--fail-on",
    type=str,
    default=None,
    help="Comma-separated statuses that cause exit code 1: REGRESSION, TOOLS_CHANGED, OUTPUT_CHANGED, CONTRACT_DRIFT (default: REGRESSION)",
)
@click.option(
    "--warn-on",
    type=str,
    default=None,
    help="Comma-separated diff statuses that print warning but exit 0 (default: TOOLS_CHANGED,OUTPUT_CHANGED, or from ci.warn_on in config.yaml)",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Strict mode: fail on any non-PASSED status (equivalent to --fail-on REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED)",
)
@click.option(
    "--trace",
    is_flag=True,
    help="Show live trace output: LLM calls, tool executions, costs, and latency.",
)
@click.option(
    "--trace-out",
    type=click.Path(),
    help="Export trace to JSONL file for debugging or sharing.",
)
@click.option(
    "--runs",
    type=int,
    default=None,
    help="Run each test N times for statistical evaluation (enables pass@k metrics). Overrides per-test variance config.",
)
@click.option(
    "--pass-rate",
    type=float,
    default=0.8,
    help="Required pass rate for statistical mode (0.0-1.0, default: 0.8). Only used with --runs.",
)
@click.option(
    "--difficulty",
    type=click.Choice(["trivial", "easy", "medium", "hard", "expert"]),
    default=None,
    help="Filter tests by difficulty level.",
)
@click.option(
    "--contracts",
    is_flag=True,
    help="Check MCP contracts for interface drift before running tests. Fails fast if external servers changed.",
)
@click.option(
    "--save-golden",
    is_flag=True,
    default=False,
    help="Save results as golden baseline if all tests pass.",
)
@track_command("run", lambda **kw: {"adapter": kw.get("adapter") or "auto", "has_path": bool(kw.get("path"))})
def run(
    path: Optional[str],
    pattern: str,
    test: tuple,
    filter: str,
    output: str,
    verbose: bool,
    track: bool,
    compare_baseline: bool,
    debug: bool,
    sequential: bool,
    max_workers: int,
    max_retries: int,
    retry_delay: float,
    watch: bool,
    html_report: str,
    summary: bool,
    coverage: bool,
    judge_model: Optional[str],
    judge_provider: Optional[str],
    adapter: Optional[str],
    diff: bool,
    diff_report: Optional[str],
    fail_on: Optional[str],
    warn_on: Optional[str],
    strict: bool,
    trace: bool,
    trace_out: Optional[str],
    runs: Optional[int],
    pass_rate: float,
    difficulty: Optional[str],
    contracts: bool,
    save_golden: bool,
):
    """Run test cases against the agent.

    PATH can be a directory containing test cases (e.g., examples/anthropic)
    or a specific test file (e.g., examples/anthropic/test-case.yaml).
    """
    # Set judge model/provider via env vars if specified (CLI overrides env)
    if judge_provider:
        os.environ["EVAL_PROVIDER"] = judge_provider
    if judge_model:
        # Resolve model aliases (e.g., 'gpt-5' -> 'gpt-5-2025-08-07')
        from evalview.core.llm_provider import resolve_model_alias
        os.environ["EVAL_MODEL"] = resolve_model_alias(judge_model)

    # Handle --strict flag (overrides config and CLI)
    if strict:
        fail_on = "REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED,CONTRACT_DRIFT"
        warn_on = ""

    asyncio.run(_run_async(
        path, pattern, test, filter, output, verbose, track, compare_baseline, debug,
        sequential, max_workers, max_retries, retry_delay, watch, html_report, summary, coverage,
        adapter_override=adapter, diff=diff, diff_report=diff_report,
        fail_on=fail_on, warn_on=warn_on, trace=trace, trace_out=trace_out,
        runs=runs, pass_rate=pass_rate, difficulty_filter=difficulty,
        contracts=contracts, save_golden=save_golden,
    ))


def _display_no_agent_guide(endpoint: Optional[str] = None) -> None:
    """Minimal prompt shown when no agent is reachable."""
    console.print()
    if endpoint:
        console.print(f"  [yellow]No agent at[/yellow] [bold]{endpoint}[/bold] [yellow]— is it running?[/yellow]")
        console.print()
        console.print("  Start your agent server, then re-run [cyan]evalview run[/cyan].")
    else:
        console.print("  [yellow]No agent configured.[/yellow]")
        console.print()
        console.print("  Point EvalView at your agent in [cyan].evalview/config.yaml[/cyan]:")
        console.print()
        console.print("    [dim]adapter: http[/dim]")
        console.print("    [dim]endpoint: http://localhost:8080/execute[/dim]")
    console.print()
    console.print("  [dim]Need a running agent?[/dim]")
    console.print("  [dim]HTTP:      https://github.com/hidai25/eval-view/blob/main/demo-agent/agent.py[/dim]")
    console.print("  [dim]LangGraph: https://github.com/hidai25/eval-view/tree/main/examples/langgraph[/dim]")
    console.print()
    console.print("  Or see EvalView catch a real regression right now:")
    console.print("  [bold cyan]→ evalview demo[/bold cyan]   [dim](no setup, 30 seconds)[/dim]")
    console.print()


async def _run_async(
    path: Optional[str],
    pattern: str,
    test: tuple,
    filter: str,
    output: str,
    verbose: bool,
    track: bool,
    compare_baseline: bool,
    debug: bool = False,
    sequential: bool = False,
    max_workers: int = 8,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    watch: bool = False,
    html_report: str = None,
    summary: bool = False,
    coverage: bool = False,
    adapter_override: Optional[str] = None,
    diff: bool = False,
    diff_report: Optional[str] = None,
    fail_on: Optional[str] = None,
    warn_on: Optional[str] = None,
    trace: bool = False,
    trace_out: Optional[str] = None,
    runs: Optional[int] = None,
    pass_rate: float = 0.8,
    difficulty_filter: Optional[str] = None,
    contracts: bool = False,
    save_golden: bool = False,
):
    """Async implementation of run command."""
    import fnmatch
    import json as json_module
    from evalview.tracking import RegressionTracker
    from evalview.core.parallel import execute_tests_parallel
    from evalview.core.retry import RetryConfig, with_retry
    from evalview.core.config import ScoringWeights
    from evalview.evaluators.statistical_evaluator import (
        StatisticalEvaluator,
        is_statistical_mode,
    )
    from evalview.reporters.console_reporter import ConsoleReporter
    from evalview.reporters.trace_live_reporter import create_trace_reporter

    # Load environment variables from path directory if provided
    if path:
        target_dir = Path(path) if Path(path).is_dir() else Path(path).parent
        path_env = target_dir / ".env.local"
        if path_env.exists():
            load_dotenv(dotenv_path=str(path_env), override=True)

    # Load config EARLY to get judge settings before provider selection
    config_path = Path(".evalview/config.yaml")
    if path:
        target_dir = Path(path) if Path(path).is_dir() else Path(path).parent
        path_config = target_dir / ".evalview" / "config.yaml"
        if path_config.exists():
            config_path = path_config

    early_config = {}
    if config_path.exists():
        with open(config_path) as f:
            early_config = yaml.safe_load(f) or {}

    # ── Connectivity check — before ANY output ────────────────────────────────
    # Do this first so users with no agent never see the banner or verbose flags.
    _ec_adapter = (adapter_override or early_config.get("adapter", "http")).lower()
    _ec_no_http_check = {"openai-assistants", "anthropic", "ollama", "goose"}
    # Skip endpoint check for API-key-based adapters; always check URL-based ones
    _ec_endpoint = early_config.get("endpoint") if _ec_adapter not in _ec_no_http_check else None

    if _ec_endpoint and _ec_adapter not in _ec_no_http_check:
        import socket as _socket
        from urllib.parse import urlparse as _urlparse
        try:
            _p = _urlparse(_ec_endpoint)
            _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            _sock.settimeout(1.5)
            _ok = _sock.connect_ex((_p.hostname or "localhost", _p.port or 80)) == 0
            _sock.close()
            if not _ok:
                _display_no_agent_guide(_ec_endpoint)
                return
        except Exception:
            _display_no_agent_guide(_ec_endpoint)
            return
    elif not _ec_endpoint and not early_config and _ec_adapter not in _ec_no_http_check:
        _display_no_agent_guide(None)
        return

    # Apply judge config from config file BEFORE provider selection
    # Config.yaml judge settings OVERRIDE .env.local (explicit config takes priority)
    judge_config = early_config.get("judge", {})
    if judge_config:
        if judge_config.get("provider"):
            os.environ["EVAL_PROVIDER"] = judge_config["provider"]
        if judge_config.get("model"):
            from evalview.core.llm_provider import resolve_model_alias
            os.environ["EVAL_MODEL"] = resolve_model_alias(judge_config["model"])

    # Interactive provider selection for LLM-as-judge
    result = get_or_select_provider(console)
    if result is None:
        try:
            from evalview.telemetry.client import get_client as _tc
            from evalview.telemetry.events import CommandEvent as _CE
            _tc().track(_CE(
                command_name="run_failed_early",
                success=False,
                properties={"failure_reason": "no_provider_configured", "has_config": bool(early_config)},
            ))
        except Exception:
            pass
        return

    selected_provider, selected_api_key = result

    # Save preference for future runs
    save_provider_preference(selected_provider)

    # Set environment variable for the evaluators to use (only if not already set from config)
    config_for_provider = PROVIDER_CONFIGS[selected_provider]
    if not os.environ.get("EVAL_PROVIDER"):
        os.environ["EVAL_PROVIDER"] = selected_provider.value
    # Don't set OLLAMA_HOST to "ollama" placeholder - Ollama doesn't need it
    from evalview.core.llm_provider import LLMProvider
    if selected_provider != LLMProvider.OLLAMA:
        os.environ[config_for_provider.env_var] = selected_api_key

    # Welcome banner
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗██╗   ██╗ █████╗ ██╗    ██╗   ██╗██╗███████╗██╗    ██╗[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔════╝██║   ██║██╔══██╗██║    ██║   ██║██║██╔════╝██║    ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]█████╗  ██║   ██║███████║██║    ██║   ██║██║█████╗  ██║ █╗ ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔══╝  ╚██╗ ██╔╝██╔══██║██║    ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗ ╚████╔╝ ██║  ██║███████╗╚████╔╝ ██║███████╗╚███╔███╔╝[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ [/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]        [dim]Catch agent regressions before you ship[/dim]               [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print()

    if debug:
        console.print("[dim]🐛 Debug mode enabled - will show raw responses[/dim]\n")
        verbose = True  # Debug implies verbose

    if verbose:
        console.print("[dim]🔍 Verbose mode enabled[/dim]\n")

    if track or compare_baseline:
        console.print("[dim]📊 Regression tracking enabled[/dim]\n")

    # Display execution mode
    if sequential:
        console.print("[dim]⏳ Running tests sequentially[/dim]\n")
    else:
        console.print(f"[dim]⚡ Running tests in parallel (max {max_workers} workers)[/dim]\n")

    if max_retries > 0:
        console.print(f"[dim]🔄 Retry enabled: up to {max_retries} retries with {retry_delay}s base delay[/dim]\n")

    # Initialize trace reporter if trace mode enabled
    trace_reporter = None
    if trace or trace_out:
        trace_reporter = create_trace_reporter(
            console=console,
            trace_out_path=trace_out,
        )
        if trace:
            console.print("[dim]📡 Trace mode enabled - showing live execution details[/dim]\n")
        if trace_out:
            console.print(f"[dim]📄 Trace output: {trace_out}[/dim]\n")

    # Handle watch mode - wrap test execution in a loop
    if watch:
        try:
            from evalview.core.watcher import WATCHDOG_AVAILABLE
            if not WATCHDOG_AVAILABLE:
                console.print("[yellow]⚠️  Watch mode requires watchdog. Install with: pip install watchdog[/yellow]")
                console.print("[dim]Falling back to single run mode...[/dim]\n")
                watch = False
            else:
                console.print("[dim]👀 Watch mode enabled - press Ctrl+C to stop[/dim]\n")
        except ImportError:
            console.print("[yellow]⚠️  Watch mode requires watchdog. Install with: pip install watchdog[/yellow]")
            watch = False

    # Load config - check path directory first, then current directory
    config_path = None
    if path:
        # Check for config in the provided path directory
        target_dir = Path(path) if Path(path).is_dir() else Path(path).parent
        path_config = target_dir / ".evalview" / "config.yaml"
        if path_config.exists():
            config_path = path_config
            if verbose:
                console.print(f"[dim]📂 Using config from: {path_config}[/dim]")

    # Fall back to current directory config
    if config_path is None:
        config_path = Path(".evalview/config.yaml")

    config_exists = config_path.exists()
    if config_exists:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        # No config file - use empty config, will try to use test case adapter/endpoint
        config = {}
        if verbose:
            console.print("[dim]No config file found - will use test case adapter/endpoint if available[/dim]")

    _run_endpoint = config.get("endpoint", "")
    _run_adapter = config.get("adapter", "http")
    if _run_endpoint:
        console.print(f"[blue]Running test cases...[/blue]  [dim]→ {_run_adapter}  {_run_endpoint}[/dim]\n")
    else:
        console.print("[blue]Running test cases...[/blue]\n")

    # Apply CI config from config.yaml (if CLI flags not provided)
    # Priority: CLI flags > config.yaml > hardcoded defaults
    ci_config = config.get("ci", {})
    if fail_on is None:
        config_fail_on = ci_config.get("fail_on", ["REGRESSION"])
        # Note: can't use isinstance(x, list) because 'list' is shadowed by the list command
        if type(config_fail_on).__name__ == "list":
            fail_on = ",".join(config_fail_on)
        else:
            fail_on = str(config_fail_on)
    if warn_on is None:
        config_warn_on = ci_config.get("warn_on", ["TOOLS_CHANGED", "OUTPUT_CHANGED"])
        if type(config_warn_on).__name__ == "list":
            warn_on = ",".join(config_warn_on)
        else:
            warn_on = str(config_warn_on)

    # --- MCP Contract Check (runs before tests, fail fast) ---
    # Placed after fail_on/warn_on resolution so config.yaml defaults are available.
    contract_drifts = []
    if contracts:
        from evalview.core.mcp_contract import ContractStore
        from evalview.core.contract_diff import diff_contract, ContractDriftStatus
        from evalview.adapters.mcp_adapter import MCPAdapter as MCPContractAdapter

        contract_store = ContractStore()
        all_contracts = contract_store.list_contracts()

        if all_contracts:
            console.print("[cyan]━━━ MCP Contract Check ━━━[/cyan]\n")

            for meta in all_contracts:
                contract = contract_store.load_contract(meta.server_name)
                if not contract:
                    continue

                mcp_adapter = MCPContractAdapter(endpoint=contract.metadata.endpoint, timeout=30.0)
                try:
                    current_tools = await mcp_adapter.discover_tools()
                except Exception as e:
                    console.print(f"  [yellow]WARN: {meta.server_name}[/yellow] - could not connect: {e}")
                    continue

                contract_result = diff_contract(contract, current_tools)

                if contract_result.status == ContractDriftStatus.CONTRACT_DRIFT:
                    contract_drifts.append(contract_result)
                    console.print(f"  [red]CONTRACT_DRIFT: {meta.server_name}[/red] - {contract_result.summary()}")
                    for change in contract_result.breaking_changes:
                        console.print(f"    [red]{change.kind.value}: {change.tool_name}[/red] - {change.detail}")
                else:
                    console.print(f"  [green]PASSED: {meta.server_name}[/green]")

            console.print()

            # Fail fast if contract drift detected and fail_on includes it
            if contract_drifts and "CONTRACT_DRIFT" in fail_on.upper():
                console.print("[bold red]Aborting: MCP contract drift detected. Fix contracts before running tests.[/bold red]")
                console.print("[dim]Accept changes: evalview mcp snapshot <endpoint> --name <name>[/dim]\n")
                raise SystemExit(1)
        else:
            console.print("[dim]--contracts: No contracts found. Create one: evalview mcp snapshot <endpoint> --name <name>[/dim]\n")

    # Extract model config (can be string or dict)
    model_config = config.get("model", {})
    if verbose and model_config:
        if isinstance(model_config, str):
            console.print(f"[dim]💰 Model: {model_config}[/dim]")
        elif isinstance(model_config, dict):
            console.print(f"[dim]💰 Model: {model_config.get('name', 'gpt-5-mini')}[/dim]")
            if "pricing" in model_config:
                console.print(
                    f"[dim]💵 Custom pricing: ${model_config['pricing']['input_per_1m']:.2f} in, ${model_config['pricing']['output_per_1m']:.2f} out[/dim]"
                )

    # SSRF protection config - defaults to True for local development
    # Set to False in production when using untrusted test cases
    allow_private_urls = config.get("allow_private_urls", True)
    if verbose:
        if allow_private_urls:
            console.print("[dim]🔓 SSRF protection: allowing private URLs (local dev mode)[/dim]")
        else:
            console.print("[dim]🔒 SSRF protection: blocking private URLs[/dim]")

    # Load judge config from config file (config.yaml overrides .env.local)
    judge_config = config.get("judge", {})
    if judge_config:
        if judge_config.get("provider"):
            os.environ["EVAL_PROVIDER"] = judge_config["provider"]
        if judge_config.get("model"):
            from evalview.core.llm_provider import resolve_model_alias
            os.environ["EVAL_MODEL"] = resolve_model_alias(judge_config["model"])
        if verbose:
            console.print(f"[dim]⚖️  Judge: {judge_config.get('provider', 'default')} / {judge_config.get('model', 'default')}[/dim]")

    # Initialize adapter based on type (if config has endpoint or is a special adapter type)
    # CLI --adapter flag overrides config file
    adapter_type = adapter_override if adapter_override else config.get("adapter", "http")
    adapter = None  # Will be None if no config - test cases must provide their own adapter/endpoint

    if adapter_override and verbose:
        console.print(f"[dim]🔌 Adapter override: {adapter_override}[/dim]")

    # Only initialize global adapter if config has necessary info
    has_endpoint = "endpoint" in config
    is_api_adapter = adapter_type in ["openai-assistants", "anthropic", "ollama"]
    is_cli_adapter = adapter_type in ["goose"]  # CLI-based adapters don't need endpoint

    if has_endpoint or is_api_adapter or is_cli_adapter:
        if adapter_type == "langgraph":
            adapter = LangGraphAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30.0),
                streaming=config.get("streaming", False),
                verbose=verbose,
                model_config=model_config,
                assistant_id=config.get("assistant_id", "agent"),  # Cloud API support
                allow_private_urls=allow_private_urls,
            )
        elif adapter_type == "crewai":
            adapter = CrewAIAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 120.0),
                verbose=verbose,
                model_config=model_config,
                allow_private_urls=allow_private_urls,
            )
        elif adapter_type == "openai-assistants":
            adapter = OpenAIAssistantsAdapter(
                assistant_id=config.get("assistant_id"),
                timeout=config.get("timeout", 120.0),
                verbose=verbose,
                model_config=model_config,
            )
        elif adapter_type in ["streaming", "tapescope", "jsonl"]:
            # Streaming adapter supports JSONL streaming APIs
            # (tapescope/jsonl are aliases for backward compatibility)
            adapter = TapeScopeAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 60.0),
                verbose=verbose,
                model_config=model_config,
                allow_private_urls=allow_private_urls,
            )
        elif adapter_type == "anthropic":
            # Anthropic Claude adapter for direct API testing
            # Check for API key first
            if not os.getenv("ANTHROPIC_API_KEY"):
                console.print("[red]❌ ANTHROPIC_API_KEY not found in environment.[/red]")
                console.print("[dim]Set it in your .env.local file or export it:[/dim]")
                console.print("[dim]  export ANTHROPIC_API_KEY=sk-ant-...[/dim]")
                return

            from evalview.adapters.anthropic_adapter import AnthropicAdapter

            # Handle model config - can be string or dict with 'name' key
            anthropic_model = config.get("model", "claude-sonnet-4-5-20250929")
            if isinstance(anthropic_model, dict):
                anthropic_model = anthropic_model.get("name", "claude-sonnet-4-5-20250929")

            adapter = AnthropicAdapter(
                model=anthropic_model,
                tools=config.get("tools", []),
                system_prompt=config.get("system_prompt"),
                max_tokens=config.get("max_tokens", 4096),
                timeout=config.get("timeout", 120.0),
                verbose=verbose,
            )
        elif adapter_type in ["huggingface", "hf", "gradio"]:
            # HuggingFace Spaces adapter for Gradio-based agents
            from evalview.adapters.huggingface_adapter import HuggingFaceAdapter

            adapter = HuggingFaceAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 120.0),
                hf_token=os.getenv("HF_TOKEN"),
                function_name=config.get("function_name"),
                verbose=verbose,
                model_config=model_config,
                allow_private_urls=allow_private_urls,
            )
        elif adapter_type == "ollama":
            # Ollama adapter for local LLMs
            from evalview.adapters.ollama_adapter import OllamaAdapter

            ollama_model = config.get("model", "llama3.2")
            if isinstance(ollama_model, dict):
                ollama_model = ollama_model.get("name", "llama3.2")

            adapter = OllamaAdapter(
                model=ollama_model,
                endpoint=config.get("endpoint", "http://localhost:11434"),
                timeout=config.get("timeout", 60.0),
                verbose=verbose,
                model_config=model_config,
            )
        elif adapter_type == "goose":
            # Goose CLI adapter for Block's open-source AI agent
            from evalview.adapters.goose_adapter import GooseAdapter

            adapter = GooseAdapter(
                timeout=config.get("timeout", 300.0),
                cwd=config.get("cwd"),
                extensions=config.get("extensions", ["developer"]),
                provider=config.get("provider"),
                model=config.get("goose_model"),  # Separate from judge model
            )
            if verbose:
                console.print("[dim]🪿 Using Goose CLI adapter[/dim]")
        else:
            # HTTP adapter for standard REST APIs
            adapter = HTTPAdapter(
                endpoint=config["endpoint"],
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30.0),
                model_config=model_config,
                allow_private_urls=allow_private_urls,
            )

    # Initialize evaluator with configurable weights
    # (LLM provider is auto-detected by the OutputEvaluator)
    scoring_weights = None
    if "scoring" in config and "weights" in config["scoring"]:
        try:
            scoring_weights = ScoringWeights(**config["scoring"]["weights"])
            if verbose:
                console.print(f"[dim]⚖️  Custom weights: tool={scoring_weights.tool_accuracy}, output={scoring_weights.output_quality}, sequence={scoring_weights.sequence_correctness}[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Invalid scoring weights in config: {e}. Using defaults.[/yellow]")

    evaluator = Evaluator(
        default_weights=scoring_weights,
    )

    # Setup retry config
    retry_config = RetryConfig(
        max_retries=max_retries,
        base_delay=retry_delay,
        exponential=True,
        jitter=True,
    )

    # Initialize tracker if tracking enabled
    tracker = None
    regression_reports = {}
    if track or compare_baseline:
        tracker = RegressionTracker()

    # Load test cases
    # Priority: 1. path argument, 2. pattern option, 3. default tests/test-cases/

    # Check if path argument is provided (e.g., evalview run examples/anthropic)
    if path:
        target_path = Path(path)
        if target_path.exists() and target_path.is_file():
            # Load single file directly
            try:
                test_cases = [TestCaseLoader.load_from_file(target_path)]
                if verbose:
                    console.print(f"[dim]📄 Loading test case from: {path}[/dim]\n")
            except Exception as e:
                console.print(f"[red]❌ Failed to load test case: {e}[/red]")
                return
        elif target_path.exists() and target_path.is_dir():
            # Load all YAML files from specified directory
            test_cases = TestCaseLoader.load_from_directory(target_path, "*.yaml")
            if verbose:
                console.print(f"[dim]📁 Loading test cases from: {path}[/dim]\n")
        else:
            console.print(f"[red]❌ Path not found: {path}[/red]")
            return
    # Check if pattern is a direct file path
    elif (pattern_path := Path(pattern)).exists() and pattern_path.is_file():
        # Load single file directly
        try:
            test_cases = [TestCaseLoader.load_from_file(pattern_path)]
            if verbose:
                console.print(f"[dim]📄 Loading test case from: {pattern}[/dim]\n")
        except Exception as e:
            console.print(f"[red]❌ Failed to load test case: {e}[/red]")
            return
    elif pattern_path.exists() and pattern_path.is_dir():
        # Load from specified directory
        test_cases = TestCaseLoader.load_from_directory(pattern_path, "*.yaml")
        if verbose:
            console.print(f"[dim]📁 Loading test cases from: {pattern}[/dim]\n")
    else:
        # Default: look in tests/test-cases/
        test_cases_dir = Path("tests/test-cases")
        if not test_cases_dir.exists():
            console.print("[red]❌ Test cases directory not found: tests/test-cases[/red]")
            console.print("[dim]Tip: You can specify a path or file directly:[/dim]")
            console.print("[dim]  evalview run examples/anthropic[/dim]")
            console.print("[dim]  evalview run path/to/test-case.yaml[/dim]")
            return
        test_cases = TestCaseLoader.load_from_directory(test_cases_dir, pattern)

    if not test_cases:
        console.print(f"[yellow]⚠️  No test cases found matching pattern: {pattern}[/yellow]\n")
        console.print("[bold]💡 Create tests by:[/bold]")
        console.print("   • [cyan]evalview record --interactive[/cyan]   (record agent interactions)")
        console.print("   • [cyan]evalview expand <test.yaml>[/cyan]     (generate variations from seed)")
        console.print("   • Or create YAML files manually in tests/test-cases/")
        console.print()
        console.print("[dim]Example: evalview record → evalview expand recorded-001.yaml --count 50[/dim]")
        return

    # Filter by difficulty if specified
    if difficulty_filter:
        original_count = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.difficulty == difficulty_filter]
        if not test_cases:
            console.print(f"[yellow]⚠️  No test cases with difficulty '{difficulty_filter}' found[/yellow]")
            console.print(f"[dim]Original count: {original_count} tests[/dim]")
            return
        if verbose:
            console.print(f"[dim]🎯 Filtered to {len(test_cases)}/{original_count} tests with difficulty: {difficulty_filter}[/dim]\n")

    # Inject variance config for --runs flag (enables statistical/pass@k mode)
    if runs is not None:
        if runs < 2:
            console.print("[red]❌ --runs must be at least 2 for statistical mode[/red]")
            return
        if runs > 100:
            console.print("[red]❌ --runs cannot exceed 100[/red]")
            return

        from evalview.core.types import VarianceConfig
        cli_variance_config = VarianceConfig(
            runs=runs,
            pass_rate=pass_rate,
        )
        # Inject variance config into each test case (overrides per-test config)
        for tc in test_cases:
            tc.thresholds.variance = cli_variance_config

        console.print(f"[cyan]📊 Statistical mode: Running each test {runs} times (pass rate: {pass_rate:.0%})[/cyan]\n")

    # Interactive test selection menu - show when no explicit filter provided
    # and pattern is the default "*.yaml"
    if pattern == "*.yaml" and not test and not filter and sys.stdin.isatty():
        # Group tests by adapter type
        tests_by_adapter = {}
        for tc in test_cases:
            adapter_name = tc.adapter or config.get("adapter", "http")
            if adapter_name not in tests_by_adapter:
                tests_by_adapter[adapter_name] = []
            tests_by_adapter[adapter_name].append(tc)

        # Get unique endpoints for each adapter
        adapter_endpoints = {}
        for adapter_name, adapter_tests in tests_by_adapter.items():
            # Find the endpoint for this adapter
            for tc in adapter_tests:
                if tc.endpoint:
                    adapter_endpoints[adapter_name] = tc.endpoint
                    break
            if adapter_name not in adapter_endpoints:
                adapter_endpoints[adapter_name] = config.get("endpoint", "")

        # Check server health for each adapter using TCP socket (fast & reliable)
        def check_health_sync(endpoint: str) -> bool:
            """Quick health check - test if port is open."""
            if not endpoint:
                return False
            try:
                # Parse host and port from endpoint URL
                from urllib.parse import urlparse
                import socket
                parsed = urlparse(endpoint)
                host = parsed.hostname or "localhost"
                port = parsed.port or 80

                # TCP socket connection check - very fast
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            except Exception:
                return False

        adapter_health = {}
        for adapter_name, endpoint in adapter_endpoints.items():
            adapter_health[adapter_name] = check_health_sync(endpoint)

        # Show interactive menu if multiple adapters
        if len(tests_by_adapter) > 1:
            console.print("[bold]📋 Test cases found:[/bold]\n")

            menu_options = []
            for i, (adapter_name, adapter_tests) in enumerate(tests_by_adapter.items(), 1):
                health_status = "[green]✅[/green]" if adapter_health.get(adapter_name) else "[red]❌[/red]"
                endpoint = adapter_endpoints.get(adapter_name, "N/A")
                console.print(f"  [{i}] [bold]{adapter_name.upper()}[/bold] ({len(adapter_tests)} tests) {health_status}")
                console.print(f"      Endpoint: {endpoint}")
                for tc in adapter_tests[:3]:  # Show first 3 test names
                    console.print(f"        • {tc.name}")
                if len(adapter_tests) > 3:
                    console.print(f"        • ... and {len(adapter_tests) - 3} more")
                console.print()
                menu_options.append((adapter_name, adapter_tests))

            # Add "All tests" option
            console.print(f"  [{len(menu_options) + 1}] [bold]All tests[/bold] ({len(test_cases)} tests)")
            console.print()

            # Get user choice
            choice = click.prompt(
                "Which tests to run?",
                type=int,
                default=len(menu_options) + 1,  # Default to all
            )

            if 1 <= choice <= len(menu_options):
                selected_adapter, test_cases = menu_options[choice - 1]
                console.print(f"\n[cyan]Running {selected_adapter.upper()} tests...[/cyan]")
            elif choice == len(menu_options) + 1:
                console.print("\n[cyan]Running all tests...[/cyan]")
            else:
                console.print("[yellow]Invalid choice. Running all tests.[/yellow]")

            # Ask about run mode (parallel vs sequential)
            console.print("\n[bold]Run mode:[/bold]")
            console.print("  [1] Parallel (faster, default)")
            console.print("  [2] Sequential (easier to follow)")
            run_mode = click.prompt("Select run mode", type=int, default=1)
            if run_mode == 2:
                _ = False  # Sequential mode (not yet implemented)
                console.print("[dim]Running tests sequentially...[/dim]\n")
            else:
                console.print("[dim]Running tests in parallel...[/dim]\n")

            # Show cost calculation info
            cost_model = config.get("model", "gpt-4o-mini")
            console.print(f"[dim]💰 Cost calculated using: {cost_model} pricing[/dim]")
            console.print("[dim]   (Configure in .evalview/config.yaml or test case)[/dim]\n")

            # Ask about HTML report
            if not html_report:
                generate_html = click.confirm("Generate HTML report?", default=True)
                if generate_html:
                    html_report = f".evalview/results/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    console.print(f"[dim]📊 HTML report will be saved to: {html_report}[/dim]\n")

    # Filter test cases by name if --test or --filter specified
    if test or filter:
        original_count = len(test_cases)
        filtered_cases = []

        for test_case in test_cases:
            # Check if test name is in the --test list (case-insensitive)
            if test:
                test_name_lower = test_case.name.lower()
                if any(t.lower() == test_name_lower for t in test):
                    filtered_cases.append(test_case)
                    continue

            # Check if test name matches --filter pattern (case-insensitive, fuzzy)
            if filter:
                filter_lower = filter.lower()
                test_name_lower = test_case.name.lower()

                # If filter has wildcards, use pattern matching
                if "*" in filter or "?" in filter:
                    if fnmatch.fnmatch(test_name_lower, filter_lower):
                        filtered_cases.append(test_case)
                        continue
                # Otherwise, do substring match (more user-friendly)
                elif filter_lower in test_name_lower:
                    filtered_cases.append(test_case)
                    continue

        test_cases = filtered_cases

        if not test_cases:
            console.print("[yellow]⚠️  No test cases matched the filter criteria[/yellow]")
            return

        if verbose:
            console.print(f"[dim]Filtered {original_count} → {len(test_cases)} test(s)[/dim]\n")

    console.print(f"Found {len(test_cases)} test case(s)\n")

    # Helper function to get adapter for a test case
    def get_adapter_for_test(test_case):
        """Get adapter for test case - use test-specific if specified, otherwise global."""
        # If test specifies its own adapter, create it
        # Note: openai-assistants and goose don't need an endpoint (use SDK/CLI directly)
        if test_case.adapter and (test_case.endpoint or test_case.adapter in ["openai-assistants", "goose"]):
            test_adapter_type = test_case.adapter
            test_endpoint = test_case.endpoint
            test_config = test_case.adapter_config or {}

            if verbose:
                console.print(
                    f"[dim]  Using test-specific adapter: {test_adapter_type} @ {test_endpoint}[/dim]"
                )

            # Create adapter based on type
            if test_adapter_type == "langgraph":
                return LangGraphAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 30.0),
                    streaming=test_config.get("streaming", False),
                    verbose=verbose,
                    model_config=model_config,
                    assistant_id=test_config.get("assistant_id", "agent"),
                    allow_private_urls=allow_private_urls,
                )
            elif test_adapter_type == "crewai":
                # Merge global model_config with test-specific config
                merged_model_config = {**model_config, **test_config}
                return CrewAIAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 120.0),
                    verbose=verbose,
                    model_config=merged_model_config,
                    allow_private_urls=allow_private_urls,
                )
            elif test_adapter_type == "openai-assistants":
                return OpenAIAssistantsAdapter(
                    assistant_id=test_config.get("assistant_id"),
                    timeout=test_config.get("timeout", 120.0),
                    verbose=verbose,
                    model_config=model_config,
                )
            elif test_adapter_type == "tapescope":
                return TapeScopeAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 120.0),
                    verbose=verbose,
                    model_config=model_config,
                    allow_private_urls=allow_private_urls,
                )
            elif test_adapter_type == "mcp":
                from evalview.adapters.mcp_adapter import MCPAdapter
                return MCPAdapter(
                    endpoint=test_endpoint,
                    timeout=test_config.get("timeout", 30.0),
                )
            elif test_adapter_type == "goose":
                from evalview.adapters.goose_adapter import GooseAdapter
                return GooseAdapter(
                    timeout=test_config.get("timeout", 300.0),
                    cwd=test_case.input.context.get("cwd") if test_case.input.context else None,
                    extensions=test_case.input.context.get("extensions") if test_case.input.context else None,
                    provider=test_config.get("provider"),
                    model=test_config.get("model"),
                )
            else:  # Default to HTTP adapter
                return HTTPAdapter(
                    endpoint=test_endpoint,
                    headers=test_config.get("headers", {}),
                    timeout=test_config.get("timeout", 30.0),
                    model_config=model_config,
                    allow_private_urls=allow_private_urls,
                )

        # Use global adapter
        if adapter is None:
            console.print(f"[red]❌ No adapter configured for test: {test_case.name}[/red]")
            console.print("[dim]Either add adapter/endpoint to the test case YAML, or create .evalview/config.yaml[/dim]")
            console.print("[dim]Example in test case:[/dim]")
            console.print("[dim]  adapter: http[/dim]")
            console.print("[dim]  endpoint: http://localhost:8000[/dim]")
            raise ValueError(f"No adapter for test: {test_case.name}")
        return adapter

    # Initialize statistical evaluator and console reporter for variance mode
    statistical_evaluator = StatisticalEvaluator()
    stats_reporter = ConsoleReporter()

    # Helper function to execute a single test with retry support
    async def execute_single_test(test_case):
        """Execute a single test case with optional retry logic."""
        test_adapter = get_adapter_for_test(test_case)

        # Merge test case tools into context for adapters that support them
        context = dict(test_case.input.context) if test_case.input.context else {}
        if hasattr(test_case, 'tools') and test_case.tools:
            context['tools'] = test_case.tools

        async def _execute():
            return await test_adapter.execute(test_case.input.query, context)

        # Check if this test uses statistical mode
        if is_statistical_mode(test_case):
            variance_config = test_case.thresholds.variance
            num_runs = variance_config.runs
            console.print(f"\n[cyan]📊 Statistical mode: Running {test_case.name} {num_runs} times...[/cyan]")

            # Collect results from multiple runs
            individual_results = []
            for run_idx in range(num_runs):
                try:
                    # Execute with retry if configured
                    if retry_config.max_retries > 0:
                        retry_result = await with_retry(
                            _execute,
                            retry_config,
                            on_retry=lambda attempt, delay, exc: None,
                        )
                        if not retry_result.success:
                            console.print(f"  [red]Run {run_idx + 1}/{num_runs}: ERROR[/red]")
                            continue
                        trace = retry_result.result
                    else:
                        trace = await _execute()

                    # Evaluate this run
                    adapter_name = getattr(test_adapter, 'name', None)
                    result = await evaluator.evaluate(test_case, trace, adapter_name=adapter_name)
                    individual_results.append(result)

                    status = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
                    console.print(f"  Run {run_idx + 1}/{num_runs}: {status} score={result.score:.1f}")

                except Exception as e:
                    console.print(f"  [red]Run {run_idx + 1}/{num_runs}: ERROR - {str(e)[:50]}[/red]")

            if not individual_results:
                raise ValueError(f"All {num_runs} runs failed for {test_case.name}")

            # Compute statistical result
            stat_result = statistical_evaluator.evaluate_from_results(
                test_case, individual_results, variance_config
            )

            # Print statistical summary
            stats_reporter.print_statistical_summary(stat_result, show_individual_runs=verbose)

            # Return the statistical pass/fail and use the mean score for display
            # Create a synthetic result for compatibility with the rest of the CLI
            best_result = individual_results[0]
            best_result.passed = stat_result.passed
            best_result.score = stat_result.score_stats.mean

            return (stat_result.passed, best_result)

        # Standard single-run execution
        # Execute with retry if configured
        if retry_config.max_retries > 0:
            retry_result = await with_retry(
                _execute,
                retry_config,
                on_retry=lambda attempt, delay, exc: console.print(
                    f"[yellow]  ↻ Retry {attempt}/{retry_config.max_retries} for {test_case.name} after {delay:.1f}s ({type(exc).__name__})[/yellow]"
                ) if verbose else None,
            )
            if not retry_result.success:
                raise retry_result.exception
            trace = retry_result.result
            # Show trace output if enabled (retry path)
            if trace_reporter:
                trace_reporter.report_from_execution_trace(trace, test_case.name)
        else:
            trace = await _execute()
            # Show trace output if enabled (standard path)
            if trace_reporter:
                trace_reporter.report_from_execution_trace(trace, test_case.name)

        # Show debug information if enabled
        if debug:
            console.print(f"\n[cyan]{'─' * 60}[/cyan]")
            console.print(f"[cyan]DEBUG: {test_case.name}[/cyan]")
            console.print(f"[cyan]{'─' * 60}[/cyan]\n")

            if hasattr(test_adapter, '_last_raw_response') and test_adapter._last_raw_response:
                console.print("[bold]Raw API Response:[/bold]")
                try:
                    raw_json = json_module.dumps(test_adapter._last_raw_response, indent=2, default=str)[:2000]
                    console.print(f"[dim]{raw_json}[/dim]")
                    if len(json_module.dumps(test_adapter._last_raw_response, default=str)) > 2000:
                        console.print("[dim]... (truncated)[/dim]")
                except Exception:
                    console.print(f"[dim]{str(test_adapter._last_raw_response)[:500]}[/dim]")
                console.print()

            console.print("[bold]Parsed ExecutionTrace:[/bold]")
            console.print(f"  Session ID: {trace.session_id}")
            console.print(f"  Duration: {trace.start_time} → {trace.end_time}")
            console.print(f"  Steps: {len(trace.steps)}")
            for i, step in enumerate(trace.steps):
                console.print(f"    [{i+1}] {step.tool_name}")
                console.print(f"        params: {str(step.parameters)[:100]}")
                console.print(f"        metrics: latency={step.metrics.latency:.1f}ms, cost=${step.metrics.cost:.4f}")
                if step.metrics.tokens:
                    console.print(f"        tokens: in={step.metrics.tokens.input_tokens}, out={step.metrics.tokens.output_tokens}")
            console.print(f"  Final Output: {trace.final_output[:200]}{'...' if len(trace.final_output) > 200 else ''}")
            console.print()
            console.print("[bold]Aggregated Metrics:[/bold]")
            console.print(f"  Total Cost: ${trace.metrics.total_cost:.4f}")
            console.print(f"  Total Latency: {trace.metrics.total_latency:.0f}ms")
            if trace.metrics.total_tokens:
                console.print(f"  Total Tokens: in={trace.metrics.total_tokens.input_tokens}, out={trace.metrics.total_tokens.output_tokens}, cached={trace.metrics.total_tokens.cached_tokens}")
            console.print()

        # Evaluate
        adapter_name = getattr(test_adapter, 'name', None)
        result = await evaluator.evaluate(test_case, trace, adapter_name=adapter_name)

        # Track result if enabled
        if tracker:
            if track:
                tracker.store_result(result)
            if compare_baseline:
                regression_report = tracker.compare_to_baseline(result)
                regression_reports[test_case.name] = regression_report

        return (result.passed, result)

    # Run evaluations
    results = []
    passed = 0
    failed = 0
    execution_errors = 0  # Separate from failed tests - execution issues (network, timeout, etc.)

    if sequential:
        # Sequential execution (original behavior)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for test_case in test_cases:
                task = progress.add_task(f"Running {test_case.name}...", total=None)

                try:
                    test_passed, result = await execute_single_test(test_case)
                    results.append(result)

                    if test_passed:
                        passed += 1
                        progress.update(task, description=f"[green]✅ {test_case.name} - PASSED (score: {result.score})[/green]")
                    else:
                        failed += 1
                        progress.update(task, description=f"[red]❌ {test_case.name} - FAILED (score: {result.score})[/red]")

                except Exception as e:
                    import httpx
                    execution_errors += 1
                    error_msg = str(e)
                    if isinstance(e, httpx.ConnectError):
                        error_msg = f"Cannot connect to {config['endpoint']}"
                    elif isinstance(e, httpx.TimeoutException):
                        error_msg = "Request timeout"
                    progress.update(task, description=f"[red]⚠ {test_case.name} - EXECUTION ERROR: {error_msg}[/red]")

                progress.remove_task(task)
    else:
        # Parallel execution (new default)
        def on_start(test_name):
            if verbose:
                console.print(f"[dim]  ▶ Starting: {test_name}[/dim]")

        def on_complete(test_name, test_passed, result):
            nonlocal passed, failed
            if test_passed:
                passed += 1
                console.print(f"[green]✅ {test_name} - PASSED (score: {result.score})[/green]")
            else:
                failed += 1
                console.print(f"[red]❌ {test_name} - FAILED (score: {result.score})[/red]")

        def on_error(test_name, exc):
            nonlocal execution_errors
            import httpx
            execution_errors += 1
            error_msg = str(exc)
            if isinstance(exc, httpx.ConnectError):
                error_msg = f"Cannot connect to {config['endpoint']}"
            elif isinstance(exc, httpx.TimeoutException):
                error_msg = "Request timeout"
            console.print(f"[red]⚠ {test_name} - EXECUTION ERROR: {error_msg}[/red]")

        console.print(f"[dim]Executing {len(test_cases)} tests with up to {max_workers} parallel workers...[/dim]\n")

        # Track elapsed time during execution
        import time as time_module
        from rich.live import Live
        from rich.panel import Panel

        # Reset judge cost tracker for this run
        judge_cost_tracker.reset()

        start_time = time_module.time()
        tests_running = set()
        tests_completed = 0

        def format_elapsed():
            elapsed = time_module.time() - start_time
            mins, secs = divmod(elapsed, 60)
            secs_int = int(secs)
            ms = int((secs - secs_int) * 1000)
            return f"{int(mins):02d}:{secs_int:02d}.{ms:03d}"

        # Spinner frames for animation
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0

        def get_status_display():
            nonlocal spinner_idx
            elapsed_str = format_elapsed()
            spinner = spinner_frames[spinner_idx % len(spinner_frames)]
            spinner_idx += 1

            # Build content for panel
            running_tests = [*tests_running][:3]  # Use unpacking instead of list() to avoid shadowing
            if running_tests:
                running_lines = "\n".join([f"  [yellow]{spinner}[/yellow] [dim]{t}...[/dim]" for t in running_tests])
            else:
                running_lines = f"  [yellow]{spinner}[/yellow] [dim]Starting tests...[/dim]"

            # Status indicator
            if failed > 0:
                status = "[bold red]● Running[/bold red]"
            else:
                status = "[green]● Running[/green]"

            # Get judge cost summary
            judge_cost = judge_cost_tracker.get_summary()

            content = (
                f"  {status}\n"
                f"\n"
                f"  [bold]⏱️  Elapsed:[/bold]    [yellow]{elapsed_str}[/yellow]\n"
                f"  [bold]📋 Progress:[/bold]   {tests_completed}/{len(test_cases)} tests\n"
                f"  [bold]💰 Judge:[/bold]      [dim]{judge_cost}[/dim]\n"
                f"\n"
                f"{running_lines}\n"
                f"\n"
                f"  [green]✓ Passed:[/green] {passed}    [red]✗ Failed:[/red] {failed}"
            )

            border = "red" if failed > 0 else "cyan"
            return Panel(
                content,
                title="[bold]Test Execution[/bold]",
                border_style=border,
                padding=(0, 1),
            )

        def on_start_with_tracking(test_name):
            nonlocal tests_running
            tests_running.add(test_name[:30])
            on_start(test_name)

        def on_complete_with_tracking(test_name, test_passed, result):
            nonlocal tests_running, tests_completed
            tests_running.discard(test_name[:30])
            tests_completed += 1
            on_complete(test_name, test_passed, result)

        def on_error_with_tracking(test_name, exc):
            nonlocal tests_running, tests_completed
            tests_running.discard(test_name[:30])
            tests_completed += 1
            on_error(test_name, exc)

        # Use Live display for timer (only in interactive mode)
        if sys.stdin.isatty():
            with Live(get_status_display(), console=console, refresh_per_second=10) as live:
                async def update_display():
                    while tests_completed < len(test_cases):
                        live.update(get_status_display())
                        await asyncio.sleep(0.1)  # Faster updates for smooth spinner
                    # Final update to show completion
                    live.update(get_status_display())

                # Run both tasks concurrently
                parallel_task = execute_tests_parallel(
                    test_cases,
                    execute_single_test,
                    max_workers=max_workers,
                    on_start=on_start_with_tracking,
                    on_complete=on_complete_with_tracking,
                    on_error=on_error_with_tracking,
                )
                display_task = update_display()

                parallel_results, _ = await asyncio.gather(parallel_task, display_task, return_exceptions=True)

            # Final completion box
            final_elapsed = format_elapsed()
            final_judge_cost = judge_cost_tracker.get_summary()
            console.print()
            console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            if execution_errors > 0:
                console.print(f"[bold cyan]║[/bold cyan]  [bold red]⚠ EXECUTION ERRORS OCCURRED[/bold red]                                  [bold cyan]║[/bold cyan]")
            elif failed == 0:
                console.print(f"[bold cyan]║[/bold cyan]  [bold green]✓ AGENT HEALTHY[/bold green]                                               [bold cyan]║[/bold cyan]")
            else:
                console.print(f"[bold cyan]║[/bold cyan]  [bold red]✗ REGRESSION DETECTED[/bold red]                                        [bold cyan]║[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            if execution_errors > 0:
                console.print(f"[bold cyan]║[/bold cyan]  [green]✓ Passed:[/green] {passed:<4}  [red]✗ Failed:[/red] {failed:<4}  [red]⚠ Errors:[/red] {execution_errors:<4}         [bold cyan]║[/bold cyan]")
            else:
                console.print(f"[bold cyan]║[/bold cyan]  [green]✓ Passed:[/green] {passed:<4}  [red]✗ Failed:[/red] {failed:<4}  [dim]Time:[/dim] {final_elapsed}               [bold cyan]║[/bold cyan]")
            console.print(f"[bold cyan]║[/bold cyan]  [dim]💰 Judge cost:[/dim] {final_judge_cost:<45}[bold cyan]║[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
            console.print()
        else:
            parallel_results = await execute_tests_parallel(
                test_cases,
                execute_single_test,
                max_workers=max_workers,
                on_start=on_start,
                on_complete=on_complete,
                on_error=on_error,
            )

        # Collect results (maintaining order)
        # Debug: check if parallel_results is an exception from gather
        if isinstance(parallel_results, Exception):
            logger.error(f"parallel_results is an exception: {parallel_results}")
            console.print(f"[red]Error in parallel execution: {parallel_results}[/red]")
        elif parallel_results:
            for pr in parallel_results:
                if pr.success and pr.result:
                    results.append(pr.result)

    # Print summary
    console.print()
    reporter = ConsoleReporter()
    if summary:
        # Compact, screenshot-friendly output
        # Get suite name from path
        suite_name = None
        if path:
            suite_name = Path(path).name if Path(path).is_dir() else Path(path).stem

        # Load previous results for delta comparison
        previous_results = None
        output_dir = Path(output)
        if output_dir.exists():
            previous_results = JSONReporter.get_latest_results(output_dir)

        reporter.print_compact_summary(results, suite_name=suite_name, previous_results=previous_results)
    else:
        reporter.print_summary(results)

    # Print behavior coverage report if enabled
    if coverage:
        suite_name = None
        if path:
            suite_name = Path(path).name if Path(path).is_dir() else Path(path).stem
        reporter.print_coverage_report(test_cases, results, suite_name=suite_name)

    # Print regression analysis if enabled
    if compare_baseline and regression_reports:
        console.print()
        console.print("[bold cyan]📊 Regression Analysis[/bold cyan]")
        console.print("━" * 60)
        console.print()

        any_regressions = False
        for test_name, report in regression_reports.items():
            if report.baseline_score is None:
                continue  # Skip tests without baselines

            # Color code based on severity
            if report.is_regression:
                any_regressions = True
                if report.severity == "critical":
                    status = "[red]🔴 CRITICAL REGRESSION[/red]"
                elif report.severity == "moderate":
                    status = "[yellow]🟡 MODERATE REGRESSION[/yellow]"
                else:
                    status = "[yellow]🟠 MINOR REGRESSION[/yellow]"
            else:
                status = "[green]✅ No regression[/green]"

            console.print(f"[bold]{test_name}[/bold]: {status}")

            # Show score comparison
            if report.score_delta is not None:
                delta_str = f"{report.score_delta:+.1f}"
                percent_str = f"({report.score_delta_percent:+.1f}%)"
                if report.score_delta < 0:
                    console.print(
                        f"  Score: {report.current_score:.1f} [red]↓ {delta_str}[/red] {percent_str} vs baseline {report.baseline_score:.1f}"
                    )
                else:
                    console.print(
                        f"  Score: {report.current_score:.1f} [green]↑ {delta_str}[/green] {percent_str} vs baseline {report.baseline_score:.1f}"
                    )

            # Show cost comparison
            if report.cost_delta is not None and report.cost_delta_percent is not None:
                delta_str = f"${report.cost_delta:+.4f}"
                percent_str = f"({report.cost_delta_percent:+.1f}%)"
                if report.cost_delta_percent > 20:
                    console.print(
                        f"  Cost: ${report.current_cost:.4f} [red]↑ {delta_str}[/red] {percent_str}"
                    )
                else:
                    console.print(f"  Cost: ${report.current_cost:.4f} {delta_str} {percent_str}")

            # Show latency comparison
            if report.latency_delta is not None and report.latency_delta_percent is not None:
                delta_str = f"{report.latency_delta:+.0f}ms"
                percent_str = f"({report.latency_delta_percent:+.1f}%)"
                if report.latency_delta_percent > 30:
                    console.print(
                        f"  Latency: {report.current_latency:.0f}ms [red]↑ {delta_str}[/red] {percent_str}"
                    )
                else:
                    console.print(
                        f"  Latency: {report.current_latency:.0f}ms {delta_str} {percent_str}"
                    )

            # Show specific issues
            if report.is_regression and report.issues:
                console.print(f"  Issues: {', '.join(report.issues)}")

            console.print()

        if any_regressions:
            console.print("[red]⚠️  Regressions detected! Review changes before deploying.[/red]\n")

    # Save results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    JSONReporter.save(results, results_file)

    console.print(f"\n[dim]Results saved to: {results_file}[/dim]\n")

    # Auto-save golden baseline if --save-golden and all tests passed cleanly
    if save_golden and failed == 0 and execution_errors == 0 and results:
        try:
            from evalview.core.golden import GoldenStore
            store = GoldenStore()
            saved_count = 0
            for result in results:
                if result and result.score > 0:
                    store.save_golden(result, notes="Auto-saved via --save-golden", source_file=str(results_file))
                    saved_count += 1
            if saved_count > 0:
                console.print(f"[green]Golden baseline saved for {saved_count} test{'s' if saved_count != 1 else ''}.[/green]")
                console.print("[dim]Future runs with --diff will compare against this baseline.[/dim]\n")
        except Exception as e:
            console.print(f"[yellow]Could not save golden baseline: {e}[/yellow]\n")

    # Initialize for diff tracking (used by both diff display and diff report)
    diffs_found = []

    # --- Golden Diff Display ---
    if diff and results:
        from evalview.core.golden import GoldenStore
        from evalview.core.diff import compare_to_golden, DiffStatus
        from rich.panel import Panel

        store = GoldenStore()

        for result in results:
            golden = store.load_golden(result.test_case)
            if golden:
                trace_diff = compare_to_golden(golden, result.trace, result.score)
                if trace_diff.has_differences:
                    diffs_found.append((result.test_case, trace_diff))

        if diffs_found:
            console.print("\n[bold cyan]━━━ Golden Diff Report ━━━[/bold cyan]\n")

            for test_name, trace_diff in diffs_found:
                # Status-based display with developer-friendly terminology
                status = trace_diff.overall_severity
                if status == DiffStatus.REGRESSION:
                    icon = "[red]✗ REGRESSION[/red]"
                elif status == DiffStatus.TOOLS_CHANGED:
                    icon = "[yellow]⚠ TOOLS_CHANGED[/yellow]"
                elif status == DiffStatus.OUTPUT_CHANGED:
                    icon = "[dim]~ OUTPUT_CHANGED[/dim]"
                else:
                    icon = "[green]✓ PASSED[/green]"

                console.print(f"{icon} [bold]{test_name}[/bold]")
                console.print(f"    Summary: {trace_diff.summary()}")

                # Tool diffs
                if trace_diff.tool_diffs:
                    console.print("    [bold]Tool Changes:[/bold]")
                    for td in trace_diff.tool_diffs[:5]:  # Limit display
                        if td.type == "added":
                            console.print(f"      [green]+ {td.actual_tool}[/green] (new step)")
                        elif td.type == "removed":
                            console.print(f"      [red]- {td.golden_tool}[/red] (missing)")
                        elif td.type == "changed":
                            if td.golden_tool == td.actual_tool and td.parameter_diffs:
                                # Same tool, different parameters
                                console.print(f"      [yellow]~ {td.golden_tool}[/yellow] (parameters changed)")
                            else:
                                # Different tool
                                console.print(f"      [yellow]~ {td.golden_tool} -> {td.actual_tool}[/yellow]")

                        # Show parameter diffs if present
                        if td.parameter_diffs:
                            console.print("        [dim]Parameter differences:[/dim]")
                            for pd in td.parameter_diffs[:10]:  # Limit to 10 params
                                if pd.diff_type == "missing":
                                    console.print(f"          [red]- {pd.param_name}[/red]: {pd.golden_value}")
                                elif pd.diff_type == "added":
                                    console.print(f"          [green]+ {pd.param_name}[/green]: {pd.actual_value}")
                                elif pd.diff_type == "type_changed":
                                    console.print(f"          [yellow]~ {pd.param_name}[/yellow]: type changed")
                                    console.print(f"            golden: {type(pd.golden_value).__name__} = {pd.golden_value}")
                                    console.print(f"            actual: {type(pd.actual_value).__name__} = {pd.actual_value}")
                                elif pd.diff_type == "value_changed":
                                    sim_str = ""
                                    if pd.similarity is not None:
                                        sim_pct = int(pd.similarity * 100)
                                        sim_str = f" (similarity: {sim_pct}%)"
                                    console.print(f"          [yellow]~ {pd.param_name}[/yellow]:{sim_str}")
                                    console.print(f"            [dim]golden:[/dim] {pd.golden_value}")
                                    console.print(f"            [dim]actual:[/dim] {pd.actual_value}")

                # Score diff
                if abs(trace_diff.score_diff) > 1:
                    direction = "[green]↑[/green]" if trace_diff.score_diff > 0 else "[red]↓[/red]"
                    console.print(f"    Score: {direction} {trace_diff.score_diff:+.1f}")

                console.print()

            # Summary with developer-friendly terminology
            regressions = sum(1 for _, d in diffs_found if d.overall_severity == DiffStatus.REGRESSION)
            tools_changed = sum(1 for _, d in diffs_found if d.overall_severity == DiffStatus.TOOLS_CHANGED)
            output_changed = sum(1 for _, d in diffs_found if d.overall_severity == DiffStatus.OUTPUT_CHANGED)

            if regressions > 0:
                console.print(f"[red]✗ {regressions} REGRESSION(s) - score dropped, fix before deploy[/red]")
                console.print()
                console.print("[dim]⭐ EvalView caught this before prod! Star → github.com/hidai25/eval-view[/dim]\n")
            elif tools_changed > 0:
                console.print(f"[yellow]⚠ {tools_changed} TOOLS_CHANGED - agent behavior shifted, review before deploy[/yellow]")
                console.print()
                console.print("[dim]⭐ EvalView caught this! Star → github.com/hidai25/eval-view[/dim]\n")
            elif output_changed > 0:
                console.print(f"[dim]~ {output_changed} OUTPUT_CHANGED - response changed, review before deploy[/dim]\n")
        else:
            # Check if any golden traces exist
            goldens = store.list_golden()
            matched = sum(1 for g in goldens if any(r.test_case == g.test_name for r in results))
            if matched > 0:
                console.print(f"[green]✓ PASSED - No differences from golden baseline ({matched} tests compared)[/green]\n")
            elif goldens:
                console.print("[yellow]No golden traces match these tests[/yellow]")
                console.print("[dim]Save one with: evalview golden save " + str(results_file) + "[/dim]\n")
            else:
                console.print("[yellow]No golden traces found[/yellow]")
                console.print("[dim]Create baseline: evalview golden save " + str(results_file) + "[/dim]\n")

    # Generate HTML report if requested
    if html_report and results:
        try:
            from evalview.reporters.html_reporter import HTMLReporter
            html_reporter = HTMLReporter()
            html_path = html_reporter.generate(results, html_report)
            console.print("\n[bold green]📊 HTML Report Generated![/bold green]")
            console.print(f"   [link=file://{Path(html_path).absolute()}]{html_path}[/link]")
            console.print(f"   [dim]Open in browser: open {html_path}[/dim]\n")
        except ImportError as e:
            console.print(f"[yellow]⚠️  Could not generate HTML report: {e}[/yellow]")
            console.print("[dim]Install with: pip install jinja2 plotly[/dim]\n")

    # Generate HTML diff report if requested
    if diff_report and results:
        if not diff:
            console.print("[yellow]⚠️  --diff-report requires --diff flag[/yellow]")
            console.print("[dim]Usage: evalview run --diff --diff-report diff.html[/dim]\n")
        elif diffs_found:
            try:
                from evalview.reporters.html_reporter import DiffReporter
                diff_reporter = DiffReporter()
                diff_path = diff_reporter.generate(
                    diffs=[d for _, d in diffs_found],  # Extract TraceDiff objects
                    results=results,
                    output_path=diff_report,
                )
                console.print("\n[bold cyan]📊 Diff Report Generated![/bold cyan]")
                console.print(f"   [link=file://{Path(diff_path).absolute()}]{diff_path}[/link]")
                console.print(f"   [dim]Open in browser: open {diff_path}[/dim]\n")
            except ImportError as e:
                console.print(f"[yellow]⚠️  Could not generate diff report: {e}[/yellow]")
                console.print("[dim]Install with: pip install jinja2[/dim]\n")
        else:
            console.print("[dim]No differences to report - all tests match golden baseline[/dim]\n")

    if track:
        console.print("[dim]📊 Results tracked for regression analysis[/dim]")
        console.print("[dim]   View trends: evalview trends[/dim]")
        console.print("[dim]   Set baseline: evalview baseline set[/dim]\n")

    # Tip about HTML report (only if not already generated)
    if not watch and not html_report:
        console.print("[dim]💡 Tip: Generate an interactive HTML report:[/dim]")
        console.print("[dim]   evalview run --html-report report.html[/dim]\n")

    # Quick tips (compact, one-line each)
    if not watch and results:
        if not summary and not coverage:
            console.print("[dim]Quick views:  evalview run --summary | evalview run --coverage[/dim]")
        if diff:
            console.print("[dim]Compare runs: evalview view --run-id <id>[/dim]")
        console.print()

    # Guided conversion to snapshot workflow (if appropriate)
    if not watch and not diff and results:
        from evalview.core.golden import GoldenStore
        from evalview.core.project_state import ProjectStateStore
        from evalview.core.celebrations import Celebrations

        store = GoldenStore()
        state_store = ProjectStateStore()

        # Only show if no goldens exist, all tests passed, and haven't shown before
        goldens = store.list_golden()
        all_passed = all(r.passed for r in results)

        if not goldens and all_passed and not state_store.load().conversion_suggestion_shown:
            Celebrations.conversion_suggestion(len(results))
            state_store.mark_conversion_shown()

    # --- Exit Code Logic (for CI) ---
    # Exit 2 for execution errors (network, timeout, etc.)
    # Exit 1 for test failures
    # Exit 0 for success
    if execution_errors > 0:
        exit_code = 2
    elif failed > 0:
        exit_code = 1
    else:
        exit_code = 0

    # Additional exit code logic for --diff mode
    if diff and diffs_found:
        from evalview.core.diff import DiffStatus

        # Parse fail_on and warn_on into sets
        fail_statuses = set()
        warn_statuses = set()
        valid_statuses = {"REGRESSION", "TOOLS_CHANGED", "OUTPUT_CHANGED", "PASSED", "CONTRACT_DRIFT"}

        for s in fail_on.upper().split(","):
            s = s.strip()
            if not s:
                continue
            if s in valid_statuses:
                fail_statuses.add(DiffStatus[s])
            else:
                console.print(f"[yellow]Warning: Unknown status '{s}' in --fail-on (valid: {', '.join(valid_statuses)})[/yellow]")

        for s in warn_on.upper().split(","):
            s = s.strip()
            if not s:
                continue
            if s in valid_statuses:
                warn_statuses.add(DiffStatus[s])
            else:
                console.print(f"[yellow]Warning: Unknown status '{s}' in --warn-on (valid: {', '.join(valid_statuses)})[/yellow]")

        # Count statuses found
        fail_count = 0
        warn_count = 0
        status_counts = {}

        for _, trace_diff in diffs_found:
            status = trace_diff.overall_severity
            status_counts[status] = status_counts.get(status, 0) + 1
            if status in fail_statuses:
                fail_count += 1
            elif status in warn_statuses:
                warn_count += 1

        # Print CI summary if there are issues
        if fail_count > 0 or warn_count > 0:
            console.print("[bold]━━━ CI Summary ━━━[/bold]")
            for status, count in sorted(status_counts.items(), key=lambda x: x[0].value):
                if status in fail_statuses:
                    console.print(f"  [red]✗ {count} {status.value.upper()}[/red] [dim][FAIL][/dim]")
                elif status in warn_statuses:
                    console.print(f"  [yellow]⚠ {count} {status.value.upper()}[/yellow] [dim][WARN][/dim]")
                else:
                    console.print(f"  [green]✓ {count} {status.value.upper()}[/green]")

            if fail_count > 0:
                exit_code = max(exit_code, 1)  # Don't override exit code 2 (execution errors)
                console.print(f"\n[bold red]Exit: {exit_code}[/bold red] ({fail_count} failure(s) in fail_on set)\n")
            else:
                console.print(f"\n[bold green]Exit: {exit_code}[/bold green] ({warn_count} warning(s) only)\n")

    # Trust-framing summary
    if not watch:
        console.print("[dim]━" * 50 + "[/dim]")
        if execution_errors > 0:
            console.print(f"[bold yellow]{execution_errors} test{'s' if execution_errors != 1 else ''} could not run.[/bold yellow] Check network, timeouts, or agent availability.\n")
        elif failed == 0 and passed > 0:
            console.print(f"[bold green]Agent healthy.[/bold green] {passed}/{passed} checks passed. No regressions detected.\n")
            # Check if golden baselines exist for these tests
            try:
                from evalview.core.golden import GoldenStore
                store = GoldenStore()
                has_any_golden = any(store.has_golden(r.test_case) for r in results if r)
                if not has_any_golden:
                    console.print("[dim]Tip: Save this as your baseline so future runs detect regressions:[/dim]")
                    console.print(f"[dim]   evalview golden save {results_file}[/dim]\n")
            except Exception:
                pass
        elif failed > 0:
            console.print(f"[bold red]Regression detected in {failed} test{'s' if failed != 1 else ''}.[/bold red] Review changes before shipping.\n")
        if passed > 0 or failed > 0:
            console.print("[dim]⭐ EvalView helped? Star us: [link=https://github.com/hidai25/eval-view]github.com/hidai25/eval-view[/link][/dim]\n")

    # Track run command telemetry (non-blocking, in background)
    try:
        import time as time_module
        duration_ms = (time_module.time() - start_time) * 1000 if "start_time" in dir() else None
        track_run_command(
            adapter_type=adapter_type,
            test_count=len(test_cases),
            pass_count=passed,
            fail_count=failed,
            duration_ms=duration_ms,
            diff_mode=diff,
            watch_mode=watch,
            parallel=not sequential,
        )
    except Exception:
        pass  # Telemetry errors should never break functionality

    # Watch mode: re-run tests on file changes
    if watch:
        from evalview.core.watcher import TestWatcher

        console.print("[cyan]━" * 60 + "[/cyan]")
        console.print("[cyan]👀 Watching for changes... (Ctrl+C to stop)[/cyan]")
        console.print("[cyan]━" * 60 + "[/cyan]\n")

        run_count = 1

        async def run_tests_again():
            nonlocal run_count
            run_count += 1
            console.print(f"\n[blue]━━━ Run #{run_count} ━━━[/blue]\n")

            # Re-run the full test suite (simplified re-execution)
            await _run_async(
                path=path,
                pattern=pattern,
                test=test,
                filter=filter,
                output=output,
                verbose=verbose,
                track=track,
                compare_baseline=compare_baseline,
                debug=debug,
                sequential=sequential,
                max_workers=max_workers,
                max_retries=max_retries,
                retry_delay=retry_delay,
                watch=False,  # Prevent infinite nesting
                html_report=html_report,
            )

        watcher = TestWatcher(
            paths=["tests/test-cases", ".evalview"],
            run_callback=run_tests_again,
            debounce_seconds=2.0,
        )

        try:
            await watcher.start()
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Watch mode stopped.[/yellow]")
        finally:
            watcher.stop()
            if trace_reporter:
                trace_reporter.close()
    else:
        # Cleanup trace reporter
        if trace_reporter:
            trace_reporter.close()
        # Exit with appropriate code (only when not in watch mode)
        if exit_code != 0:
            sys.exit(exit_code)


@main.command()
@click.option(
    "--pattern",
    default="*.yaml",
    help="Test case file pattern (default: *.yaml)",
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed information for each test",
)
@track_command("list")
def list(pattern: str, detailed: bool):
    """List all available test cases."""
    asyncio.run(_list_async(pattern, detailed))


@main.command()
@track_command("adapters")
def adapters():
    """List all available adapters."""
    from rich.table import Table
    from evalview.adapters.registry import AdapterRegistry

    console.print("[blue]Available Adapters[/blue]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Adapter", style="white")
    table.add_column("Description", style="dim")
    table.add_column("Needs Endpoint", style="yellow", justify="center")

    adapter_info = {
        "http": ("Generic REST API adapter", "Yes"),
        "langgraph": ("LangGraph / LangGraph Cloud", "Yes"),
        "crewai": ("CrewAI multi-agent", "Yes"),
        "openai-assistants": ("OpenAI Assistants API", "No (uses SDK)"),
        "anthropic": ("Anthropic Claude API", "Yes"),
        "claude": ("Alias for anthropic", "Yes"),
        "huggingface": ("HuggingFace Inference", "Yes"),
        "hf": ("Alias for huggingface", "Yes"),
        "gradio": ("Alias for huggingface", "Yes"),
        "goose": ("Block's Goose CLI agent", "No (uses CLI)"),
        "tapescope": ("JSONL streaming API", "Yes"),
        "streaming": ("Alias for tapescope", "Yes"),
        "jsonl": ("Alias for tapescope", "Yes"),
        "mcp": ("Model Context Protocol", "Yes"),
    }

    for name in sorted(AdapterRegistry.list_names()):
        desc, needs_endpoint = adapter_info.get(name, ("Custom adapter", "Yes"))
        table.add_row(name, desc, needs_endpoint)

    console.print(table)
    console.print(f"\n[dim]Total: {len(AdapterRegistry.list_names())} adapters[/dim]")


async def _list_async(pattern: str, detailed: bool):
    """Async implementation of list command."""
    from rich.table import Table

    console.print("[blue]Loading test cases...[/blue]\n")

    # Load test cases
    test_dir = Path("tests/test-cases")
    if not test_dir.exists():
        console.print(f"[yellow]Test directory not found: {test_dir}[/yellow]")
        return

    loader = TestCaseLoader()
    test_cases = loader.load_from_directory(test_dir, pattern=pattern)

    if not test_cases:
        console.print(f"[yellow]No test cases found matching pattern: {pattern}[/yellow]")
        return

    console.print(f"[green]Found {len(test_cases)} test case(s)[/green]\n")

    # Create table
    table = Table(title="Available Test Cases", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="white", no_wrap=False)
    table.add_column("Adapter", style="yellow", justify="center")
    table.add_column("Endpoint", style="dim", no_wrap=False)

    if detailed:
        table.add_column("Description", style="dim", no_wrap=False)

    # Add rows
    for test_case in test_cases:
        adapter_name = test_case.adapter or "[dim](from config)[/dim]"
        endpoint = test_case.endpoint or "[dim](from config)[/dim]"

        if detailed:
            description = test_case.description or "[dim]No description[/dim]"
            table.add_row(test_case.name, adapter_name, endpoint, description)
        else:
            table.add_row(test_case.name, adapter_name, endpoint)

    console.print(table)
    console.print()


@main.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed results for each test case",
)
@click.option(
    "--html",
    type=click.Path(),
    help="Generate HTML report to specified path",
)
@track_command("report", lambda **kw: {"html": bool(kw.get("html"))})
def report(results_file: str, detailed: bool, html: str):
    """Generate report from results file."""
    console.print(f"[blue]Loading results from {results_file}...[/blue]\n")

    results_data = JSONReporter.load(results_file)

    if not results_data:
        console.print("[yellow]No results found in file[/yellow]")
        return

    # Convert back to EvaluationResult objects
    from evalview.core.types import EvaluationResult

    results = [EvaluationResult(**data) for data in results_data]

    # Generate HTML report if requested
    if html:
        try:
            from evalview.reporters.html_reporter import HTMLReporter
            html_reporter = HTMLReporter()
            html_path = html_reporter.generate(results, html)
            console.print(f"[green]✅ HTML report saved to: {html_path}[/green]\n")
        except ImportError as e:
            console.print(f"[yellow]⚠️  Could not generate HTML report: {e}[/yellow]")
            console.print("[dim]Install with: pip install jinja2 plotly[/dim]\n")
        return

    reporter = ConsoleReporter()

    if detailed:
        for result in results:
            reporter.print_detailed(result)
    else:
        reporter.print_summary(results)


@main.command()
@click.argument("run_id", required=False)
@click.option(
    "-t", "--test",
    help="Filter by test name (substring match)",
)
@click.option(
    "--llm-only",
    is_flag=True,
    help="Only show LLM call spans",
)
@click.option(
    "--tools-only",
    is_flag=True,
    help="Only show tool call spans",
)
@click.option(
    "--prompts",
    is_flag=True,
    help="Show LLM prompts (truncated)",
)
@click.option(
    "--completions",
    is_flag=True,
    help="Show LLM completions (truncated)",
)
@click.option(
    "--table",
    is_flag=True,
    help="Show span table instead of tree",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.option(
    "--llm-summary",
    is_flag=True,
    help="Show LLM call summary with token/cost breakdown",
)
@track_command("view")
def view(
    run_id: Optional[str],
    test: Optional[str],
    llm_only: bool,
    tools_only: bool,
    prompts: bool,
    completions: bool,
    table: bool,
    output_json: bool,
    llm_summary: bool,
):
    """View execution trace for debugging.

    RUN_ID can be:
      - "latest" (default): View the most recent run
      - A timestamp or partial match of a result file
      - A full path to a results JSON file

    Examples:
        evalview view                    # View latest run
        evalview view latest             # Same as above
        evalview view latest -t "stock"  # Filter by test name
        evalview view abc123 --llm-only  # Show only LLM calls
        evalview view --json             # Output as JSON
        evalview view --llm-summary      # Show LLM token/cost breakdown
    """
    from evalview.reporters.trace_reporter import TraceReporter
    from evalview.core.types import EvaluationResult

    # Default to "latest" if no run_id provided
    if not run_id:
        run_id = "latest"

    # Find the results file
    results_path = _find_results_file(run_id)
    if not results_path:
        console.print(f"[red]Could not find results for: {run_id}[/red]")
        console.print("[dim]Run 'evalview run' first to generate results[/dim]")
        return

    console.print(f"[blue]Loading results from {results_path}...[/blue]\n")

    # Load results
    results_data = JSONReporter.load(str(results_path))
    if not results_data:
        console.print("[yellow]No results found in file[/yellow]")
        return

    # Convert to EvaluationResult objects
    results = [EvaluationResult(**data) for data in results_data]

    # Filter by test name if specified
    if test:
        results = [r for r in results if test.lower() in r.test_case.lower()]
        if not results:
            console.print(f"[yellow]No tests matching '{test}'[/yellow]")
            return

    reporter = TraceReporter()

    for result in results:
        console.print(f"[bold cyan]Test: {result.test_case}[/bold cyan]")
        console.print()

        if output_json:
            # Output trace context as JSON
            from evalview.core.tracing import steps_to_trace_context

            if result.trace.trace_context:
                trace_context = result.trace.trace_context
            else:
                trace_context = steps_to_trace_context(
                    steps=result.trace.steps,
                    session_id=result.trace.session_id,
                    start_time=result.trace.start_time,
                    end_time=result.trace.end_time,
                )
            console.print(reporter.export_json(trace_context))
        elif table:
            # Show span table
            from evalview.core.tracing import steps_to_trace_context

            if result.trace.trace_context:
                trace_context = result.trace.trace_context
            else:
                trace_context = steps_to_trace_context(
                    steps=result.trace.steps,
                    session_id=result.trace.session_id,
                    start_time=result.trace.start_time,
                    end_time=result.trace.end_time,
                )
            reporter.print_trace_table(trace_context)
        elif llm_summary:
            # Show LLM summary
            from evalview.core.tracing import steps_to_trace_context

            if result.trace.trace_context:
                trace_context = result.trace.trace_context
            else:
                trace_context = steps_to_trace_context(
                    steps=result.trace.steps,
                    session_id=result.trace.session_id,
                    start_time=result.trace.start_time,
                    end_time=result.trace.end_time,
                )
            reporter.print_llm_summary(trace_context)
        else:
            # Default: show trace tree
            reporter.print_trace_from_result(
                result,
                show_prompts=prompts,
                show_completions=completions,
                llm_only=llm_only,
                tools_only=tools_only,
            )

        console.print()


def _find_results_file(run_id: str) -> Optional[Path]:
    """Find a results file by run ID or path.

    Args:
        run_id: "latest", a timestamp substring, or a file path

    Returns:
        Path to the results file, or None if not found.
    """
    # Check if it's a direct path
    if Path(run_id).exists():
        return Path(run_id)

    # Look in the default results directory
    results_dir = Path(".evalview/results")
    if not results_dir.exists():
        return None

    # Get all JSON files
    result_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not result_files:
        return None

    # Handle "latest"
    if run_id.lower() == "latest":
        return result_files[0]

    # Search for matching file
    for f in result_files:
        if run_id in f.stem:
            return f

    return None


@main.command()
@click.option(
    "--endpoint",
    help="Agent endpoint URL to test (optional - will auto-detect common ones)",
)
@track_command("connect")
def connect(endpoint: str):
    """Test connection to your agent API and auto-configure endpoint."""
    asyncio.run(_connect_async(endpoint))


async def _connect_async(endpoint: Optional[str]):
    """Async implementation of connect command."""

    console.print("[blue]🔍 Testing agent connection...[/blue]\n")

    # Common ports to check
    common_ports = [8000, 2024, 3000, 8080, 5000, 8888, 7860]

    # Common endpoints to try (framework_type, name, path, adapter_type, method)
    # Will be combined with common_ports
    common_patterns = [
        ("langgraph", "LangGraph Cloud", "/ok", "langgraph", "GET"),  # LangGraph Cloud health
        ("langgraph", "LangGraph Cloud", "/info", "langgraph", "GET"),  # LangGraph Cloud info
        ("langgraph", "LangGraph", "/api/chat", "langgraph", "POST"),
        ("langgraph", "LangGraph", "/invoke", "langgraph", "POST"),
        ("http", "LangServe", "/agent", "http", "POST"),
        ("streaming", "LangServe", "/agent/stream", "streaming", "POST"),
        ("streaming", "TapeScope", "/api/unifiedchat", "streaming", "POST"),
        ("crewai", "CrewAI", "/crew", "crewai", "POST"),
        ("http", "FastAPI", "/api/agent", "http", "POST"),
        ("http", "FastAPI", "/chat", "http", "POST"),
    ]

    # Generate all port+path combinations
    common_endpoints = []
    for port in common_ports:
        for framework, name, path, adapter, method in common_patterns:
            url = f"http://127.0.0.1:{port}{path}"
            common_endpoints.append((framework, f"{name} (:{port})", url, adapter, method))

    endpoints_to_test = []
    if endpoint:
        # User provided specific endpoint - try to detect adapter type
        endpoints_to_test = [("http", "Custom", endpoint, "http", "POST")]
    else:
        # Try common ones
        endpoints_to_test = common_endpoints

    successful = None
    tested_count = 0

    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning for agent servers...", total=None)

        async with httpx.AsyncClient(timeout=3.0) as client:
            for adapter_type, name, url, default_adapter, method in endpoints_to_test:
                tested_count += 1
                progress.update(task, description=f"Scanning... ({tested_count} endpoints checked)")

                try:
                    # Use appropriate HTTP method
                    if method == "GET":
                        response = await client.get(url)
                    else:
                        # Try a simple POST request
                        response = await client.post(
                            url,
                            json={
                                "query": "test",
                                "message": "test",
                                "messages": [{"role": "user", "content": "test"}],
                            },
                            headers={"Content-Type": "application/json"},
                        )

                    if response.status_code in [
                        200,
                        201,
                        422,
                    ]:  # 422 might be validation error but server is running
                        # Skip non-JSON responses (e.g., macOS AirPlay on port 5000)
                        content_type = response.headers.get("content-type", "")
                        if not content_type.startswith("application/json"):
                            continue

                        # Try to detect framework from response
                        detected_adapter = default_adapter
                        try:
                            data = response.json()
                            # LangGraph detection
                            if "messages" in data or "thread_id" in data:
                                detected_adapter = "langgraph"
                            # CrewAI detection - look for crew-specific fields
                            # Note: "detail" alone is just a FastAPI validation error, not CrewAI-specific
                            elif "tasks" in data or "crew_id" in data or "crew" in data:
                                detected_adapter = "crewai"
                        except Exception:
                            continue  # Skip if can't parse JSON

                        successful = (name, url, response, detected_adapter)
                        break

                except (httpx.ConnectError, httpx.TimeoutException, Exception):
                    continue

    console.print()

    if successful:
        name, url, response, detected_adapter = successful
        console.print(f"[green]✅ Successfully connected to {name}![/green]\n")

        # Show response info
        console.print("[cyan]Response details:[/cyan]")
        console.print(f"  • Status: {response.status_code}")
        console.print(f"  • Content-Type: {response.headers.get('content-type', 'N/A')}")
        console.print(f"  • Detected adapter: {detected_adapter}")

        # Try to show response preview
        try:
            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
                if data and isinstance(data, dict):
                    keys_str = ", ".join(str(k) for k in data.keys())
                    if keys_str:
                        console.print(f"  • Response keys: [{keys_str}]")
        except Exception:
            pass

        # Ask if user wants to update config
        console.print()
        if click.confirm("Update .evalview/config.yaml to use this endpoint?", default=True):
            config_path = Path(".evalview/config.yaml")

            if not config_path.exists():
                console.print(
                    "[yellow]⚠️  Config file not found. Run 'evalview init' first.[/yellow]"
                )
                return

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Update config with detected adapter
            config["adapter"] = detected_adapter
            # For LangGraph Cloud, use base URL (strip /ok or /info)
            endpoint_url = url
            if detected_adapter == "langgraph" and (url.endswith("/ok") or url.endswith("/info")):
                endpoint_url = url.rsplit("/", 1)[0]
            config["endpoint"] = endpoint_url

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            console.print("[green]✅ Updated config:[/green]")
            console.print(f"  • adapter: {detected_adapter}")
            console.print(f"  • endpoint: {endpoint_url}")
            console.print()
            console.print("[blue]Next steps:[/blue]")
            console.print("  1. Create test cases in tests/test-cases/")
            console.print("  2. Run: evalview run")
        return  # Exit after successful connection
    else:
        console.print("[red]❌ Could not connect to any agent endpoint.[/red]\n")

        # Try to find open ports
        console.print("[cyan]🔍 Scanning for open ports...[/cyan]")
        open_ports = []
        test_ports = [8000, 2024, 3000, 8080, 5000, 8888, 7860, 8501, 7000]

        async with httpx.AsyncClient(timeout=2.0) as client:
            for port in test_ports:
                try:
                    response = await client.get(f"http://127.0.0.1:{port}")
                    open_ports.append(port)
                    console.print(
                        f"  • Port {port}: [green]Open[/green] (HTTP {response.status_code})"
                    )
                except Exception:
                    pass

        if open_ports:
            console.print()
            console.print(f"[green]Found {len(open_ports)} open port(s)![/green]")
            console.print()

            if click.confirm("Configure connection manually?", default=True):
                custom_port = click.prompt(
                    "Port number", type=int, default=open_ports[0] if open_ports else 8000
                )

                # Suggest common paths based on framework
                console.print("\n[cyan]Common endpoint paths:[/cyan]")
                console.print("  1. /crew         (CrewAI)")
                console.print("  2. /invoke       (LangGraph/LangServe)")
                console.print("  3. /api/chat     (Generic)")
                console.print("  4. Custom path")

                path_choice = click.prompt("Choose (1-4)", type=int, default=1)
                path_map = {1: "/crew", 2: "/invoke", 3: "/api/chat"}

                if path_choice == 4:
                    custom_path = click.prompt("Enter custom path", default="/api/chat")
                else:
                    custom_path = path_map.get(path_choice, "/api/chat")
                custom_url = f"http://127.0.0.1:{custom_port}{custom_path}"

                console.print(f"\n[cyan]Testing {custom_url}...[/cyan]")

                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.post(
                            custom_url,
                            json={
                                "query": "test",
                                "message": "test",
                                "messages": [{"role": "user", "content": "test"}],
                            },
                            headers={"Content-Type": "application/json"},
                        )

                        if response.status_code in [200, 201, 422]:
                            console.print("[green]✅ Connected![/green]\n")

                            # Auto-detect adapter
                            detected_adapter = "http"
                            try:
                                data = response.json()
                                if "messages" in data or "thread_id" in data:
                                    detected_adapter = "langgraph"
                                elif "tasks" in data or "crew_id" in data or "crew" in data:
                                    detected_adapter = "crewai"
                            except Exception:
                                pass

                            # Update config
                            config_path = Path(".evalview/config.yaml")
                            if config_path.exists():
                                with open(config_path) as f:
                                    config = yaml.safe_load(f)

                                config["adapter"] = detected_adapter
                                config["endpoint"] = custom_url

                                with open(config_path, "w") as f:
                                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                                console.print("[green]✅ Config updated:[/green]")
                                console.print(f"  • adapter: {detected_adapter}")
                                console.print(f"  • endpoint: {custom_url}")
                                return
                        else:
                            console.print(f"[red]❌ HTTP {response.status_code}[/red]")
                except Exception as e:
                    console.print(f"[red]❌ Failed: {e}[/red]")

        console.print()
        console.print("[yellow]Common issues:[/yellow]")
        console.print("  1. Agent server not running")
        console.print("  2. Non-standard port (check your server logs)")
        console.print("  3. Different endpoint path")
        console.print()
        console.print("[blue]To start an agent:[/blue]")
        console.print("  # LangGraph example:")
        console.print("  cd examples/langgraph/agent && langgraph dev  # port 2024")
        console.print()
        console.print("  # Or the demo agent:")
        console.print("  python demo_agent.py  # port 8000")
        console.print()
        console.print("[blue]Then run:[/blue]")
        console.print("  evalview connect")
        console.print("  # or specify endpoint:")
        console.print("  evalview connect --endpoint http://127.0.0.1:YOUR_PORT/api/chat")


@main.command("validate-adapter")
@click.option(
    "--endpoint",
    required=True,
    help="Endpoint URL to validate",
)
@click.option(
    "--adapter",
    default="http",
    type=click.Choice(["http", "langgraph", "crewai", "streaming", "tapescope"]),
    help="Adapter type to use (default: http)",
)
@click.option(
    "--query",
    default="What is 2+2?",
    help="Test query to send (default: 'What is 2+2?')",
)
@click.option(
    "--timeout",
    default=30.0,
    type=float,
    help="Request timeout in seconds (default: 30)",
)
@track_command("validate_adapter", lambda **kw: {"adapter": kw.get("adapter")})
def validate_adapter(endpoint: str, adapter: str, query: str, timeout: float):
    """Validate an adapter endpoint and show detailed response analysis."""
    asyncio.run(_validate_adapter_async(endpoint, adapter, query, timeout))


async def _validate_adapter_async(endpoint: str, adapter_type: str, query: str, timeout: float):
    """Async implementation of validate-adapter command."""
    import json as json_module

    console.print("[blue]🔍 Validating adapter endpoint...[/blue]\n")
    console.print(f"  Endpoint: {endpoint}")
    console.print(f"  Adapter:  {adapter_type}")
    console.print(f"  Timeout:  {timeout}s")
    console.print(f"  Query:    {query}")
    console.print()

    # Create adapter based on type
    try:
        if adapter_type == "langgraph":
            test_adapter = LangGraphAdapter(
                endpoint=endpoint,
                timeout=timeout,
                verbose=True,
                allow_private_urls=True,
            )
        elif adapter_type == "crewai":
            test_adapter = CrewAIAdapter(
                endpoint=endpoint,
                timeout=timeout,
                verbose=True,
                allow_private_urls=True,
            )
        elif adapter_type in ["streaming", "tapescope"]:
            test_adapter = TapeScopeAdapter(
                endpoint=endpoint,
                timeout=timeout,
                verbose=True,
                allow_private_urls=True,
            )
        else:
            test_adapter = HTTPAdapter(
                endpoint=endpoint,
                timeout=timeout,
                allow_private_urls=True,
            )

        console.print("[cyan]Executing test query...[/cyan]")

        # Execute
        trace = await test_adapter.execute(query)

        console.print("[green]✅ Adapter validation successful![/green]\n")

        # Show results
        console.print("[bold]Execution Summary:[/bold]")
        console.print(f"  Session ID: {trace.session_id}")
        console.print(f"  Steps captured: {len(trace.steps)}")

        if trace.steps:
            console.print("\n[bold]Tools Used:[/bold]")
            for i, step in enumerate(trace.steps):
                console.print(f"  [{i+1}] {step.tool_name}")
                if step.parameters:
                    params_str = str(step.parameters)[:80]
                    console.print(f"      params: {params_str}{'...' if len(str(step.parameters)) > 80 else ''}")

        console.print("\n[bold]Metrics:[/bold]")
        console.print(f"  Total Cost: ${trace.metrics.total_cost:.4f}")
        console.print(f"  Total Latency: {trace.metrics.total_latency:.0f}ms")
        if trace.metrics.total_tokens:
            console.print(f"  Total Tokens: {trace.metrics.total_tokens.total_tokens}")
            console.print(f"    - Input: {trace.metrics.total_tokens.input_tokens}")
            console.print(f"    - Output: {trace.metrics.total_tokens.output_tokens}")

        console.print("\n[bold]Final Output:[/bold]")
        output_preview = trace.final_output[:500]
        console.print(f"  {output_preview}{'...' if len(trace.final_output) > 500 else ''}")

        # Show raw response if available
        if hasattr(test_adapter, '_last_raw_response') and test_adapter._last_raw_response:
            console.print("\n[bold]Raw API Response (first 1000 chars):[/bold]")
            try:
                raw_json = json_module.dumps(test_adapter._last_raw_response, indent=2, default=str)[:1000]
                console.print(f"[dim]{raw_json}[/dim]")
            except Exception:
                console.print(f"[dim]{str(test_adapter._last_raw_response)[:500]}[/dim]")

        # Warnings
        warnings = []
        if not trace.steps:
            warnings.append("No tool calls detected - ensure your agent uses tools")
        if trace.metrics.total_cost == 0:
            warnings.append("Cost is 0 - token tracking may not be configured")
        if not trace.metrics.total_tokens:
            warnings.append("No token usage data - check adapter response format")

        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for w in warnings:
                console.print(f"  ⚠️  {w}")

        console.print()

    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]\n")
        console.print("[yellow]Troubleshooting tips:[/yellow]")
        console.print("  1. Check if the agent server is running")
        console.print("  2. Verify the endpoint URL is correct")
        console.print("  3. Try a different adapter type")
        console.print("  4. Increase timeout with --timeout")
        console.print()
        console.print("[dim]For detailed error info, check the server logs.[/dim]")


@main.command()
@click.option(
    "--query",
    help="Query to record (non-interactive mode)",
)
@click.option(
    "--output",
    help="Output file path (default: auto-generate in tests/test-cases/)",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Interactive mode - record multiple interactions (default: True)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed execution information",
)
@track_command("record")
def record(query: str, output: str, interactive: bool, verbose: bool):
    """Record agent interactions and generate test cases."""
    asyncio.run(_record_async(query, output, interactive, verbose))


async def _record_async(
    query: Optional[str], output: Optional[str], interactive: bool, verbose: bool
):
    """Async implementation of record command."""
    from evalview.recorder import TestCaseRecorder

    console.print("[blue]🎬 Recording mode started[/blue]")
    console.print("━" * 60)
    console.print()

    # Load config
    config_path = Path(".evalview/config.yaml")
    if not config_path.exists():
        console.print("[red]❌ Config file not found. Run 'evalview init' first.[/red]")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract model config
    model_config = config.get("model", {})

    # SSRF protection config - defaults to True for local development
    allow_private_urls = config.get("allow_private_urls", True)

    # Initialize adapter
    adapter_type = config.get("adapter", "http")

    if adapter_type == "langgraph":
        adapter = LangGraphAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            streaming=config.get("streaming", False),
            verbose=verbose,
            model_config=model_config,
            assistant_id=config.get("assistant_id", "agent"),
            allow_private_urls=allow_private_urls,
        )
    elif adapter_type == "crewai":
        adapter = CrewAIAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            verbose=verbose,
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )
    elif adapter_type in ["streaming", "tapescope", "jsonl"]:
        adapter = TapeScopeAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 60.0),
            verbose=verbose,
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )
    else:
        # HTTP adapter for standard REST APIs
        adapter = HTTPAdapter(
            endpoint=config["endpoint"],
            headers=config.get("headers", {}),
            timeout=config.get("timeout", 30.0),
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )

    # Initialize recorder
    recorder = TestCaseRecorder(adapter)

    # Determine output directory
    if output:
        output_path = Path(output)
    else:
        test_dir = Path("tests/test-cases")
        test_dir.mkdir(parents=True, exist_ok=True)
        output_path = None  # Will auto-generate

    recorded_cases = []

    # Non-interactive mode with single query
    if query and not interactive:
        try:
            console.print(f"[dim]📝 Query: {query}[/dim]\n")
            console.print("[dim]🤖 Calling agent...[/dim]", end=" ")

            interaction = await recorder.record_interaction(query)

            console.print("[green]✓[/green]\n")

            # Show detected info
            console.print("[cyan]📊 Detected:[/cyan]")
            if interaction.trace.tool_calls:
                tools = [tc.name for tc in interaction.trace.tool_calls]
                console.print(f"  • Tools: {', '.join(tools)}")
            if interaction.trace.cost:
                console.print(f"  • Cost: ${interaction.trace.cost:.4f}")
            if interaction.trace.latency:
                console.print(f"  • Latency: {interaction.trace.latency:.0f}ms")

            if verbose:
                console.print(f"\n[dim]Output: {interaction.trace.final_output}[/dim]")

            console.print()

            # Generate test case
            test_case = recorder.generate_test_case(interaction)
            recorded_cases.append((interaction, test_case))

        except Exception as e:
            console.print(f"[red]✗ Failed: {e}[/red]")
            return

    # Interactive mode
    elif interactive:
        console.print(
            "[yellow]💡 Tip: Type 'done' when finished, 'skip' to cancel current recording[/yellow]\n"
        )

        query_num = 1
        while True:
            # Get query from user
            if not query:
                console.print(
                    f"[bold]📝 Enter query #{query_num} (or 'done' to finish):[/bold] ", end=""
                )
                user_input = input().strip()

                if user_input.lower() == "done":
                    break
                elif user_input.lower() == "skip":
                    continue
                elif not user_input:
                    console.print("[yellow]⚠️  Empty query, skipping[/yellow]\n")
                    continue

                query = user_input

            try:
                console.print()
                console.print("[dim]🤖 Calling agent...[/dim]", end=" ")

                interaction = await recorder.record_interaction(query)

                console.print("[green]✓ Agent response received[/green]\n")

                # Show detected info
                console.print("[cyan]📊 Detected:[/cyan]")
                if interaction.trace.tool_calls:
                    tools = [tc.name for tc in interaction.trace.tool_calls]
                    console.print(f"  • Tools: {', '.join(tools)}")
                else:
                    console.print("  • Tools: None")

                if interaction.trace.cost:
                    console.print(f"  • Cost: ${interaction.trace.cost:.4f}")
                if interaction.trace.latency:
                    console.print(f"  • Latency: {interaction.trace.latency:.0f}ms")

                if verbose:
                    console.print(f"\n[dim]Output: {interaction.trace.final_output}[/dim]")

                console.print()

                # Generate test case
                test_case = recorder.generate_test_case(interaction)

                # Ask for custom name
                console.print(
                    f"[bold]✍️  Test case name [[dim]{test_case.name}[/dim]]:[/bold] ", end=""
                )
                custom_name = input().strip()
                if custom_name:
                    test_case.name = custom_name

                recorded_cases.append((interaction, test_case))

                console.print("[green]✅ Test case saved![/green]\n")

                query_num += 1
                query = None  # Reset for next iteration

            except Exception as e:
                console.print(f"[red]✗ Failed: {e}[/red]\n")
                if verbose:
                    import traceback

                    console.print(f"[dim]{traceback.format_exc()}[/dim]\n")

                query = None  # Reset
                continue
    else:
        console.print("[red]❌ Must provide --query or use --interactive mode[/red]")
        return

    # Save recorded test cases
    if not recorded_cases:
        console.print("[yellow]⚠️  No test cases recorded[/yellow]")
        return

    console.print()
    console.print("━" * 60)

    saved_files = []
    for interaction, test_case in recorded_cases:
        if output_path and len(recorded_cases) == 1:
            # Single file output
            file_path = output_path
        else:
            # Auto-generate filenames
            test_dir = Path("tests/test-cases")
            test_dir.mkdir(parents=True, exist_ok=True)
            file_path = recorder.generate_filename(test_dir)

        recorder.save_to_yaml(test_case, file_path)
        saved_files.append(file_path)

    # Print summary
    console.print(f"[green]✅ Recorded {len(recorded_cases)} test case(s)[/green]\n")

    for file_path in saved_files:
        console.print(f"  • {file_path}")

    console.print()
    console.print("[blue]Run with:[/blue] evalview run\n")


@main.group()
def baseline():
    """Manage test baselines for regression detection."""
    pass


@baseline.command("set")
@click.option(
    "--test",
    help="Specific test name to set baseline for (default: all recent tests)",
)
@click.option(
    "--from-latest",
    is_flag=True,
    help="Set baseline from most recent test run",
)
@track_command("baseline_set")
def baseline_set(test: str, from_latest: bool):
    """Set baseline from recent test results."""
    from evalview.tracking import RegressionTracker

    tracker = RegressionTracker()

    if test:
        # Set baseline for specific test
        if from_latest:
            try:
                tracker.set_baseline_from_latest(test)
                console.print(f"[green]✅ Baseline set for test: {test}[/green]")
            except ValueError as e:
                console.print(f"[red]❌ Error: {e}[/red]")
        else:
            console.print("[yellow]⚠️  Must specify --from-latest or run tests first[/yellow]")
    else:
        # Set baselines for all recent tests
        results = tracker.db.get_recent_results(days=1)
        unique_tests = set(r["test_name"] for r in results)

        if not unique_tests:
            console.print("[yellow]⚠️  No recent test results found. Run tests first.[/yellow]")
            return

        for test_name in unique_tests:
            tracker.set_baseline_from_latest(test_name)

        console.print(f"[green]✅ Baselines set for {len(unique_tests)} test(s)[/green]")


@baseline.command("show")
@click.option(
    "--test",
    help="Specific test name to show baseline for",
)
@track_command("baseline_show")
def baseline_show(test: str):
    """Show current baselines."""
    from evalview.tracking import RegressionTracker
    from rich.table import Table

    tracker = RegressionTracker()

    if test:
        # Show specific baseline
        baseline = tracker.db.get_baseline(test)
        if not baseline:
            console.print(f"[yellow]⚠️  No baseline set for test: {test}[/yellow]")
            return

        console.print(f"\n[bold]Baseline for: {test}[/bold]\n")
        console.print(f"  Score: {baseline['score']:.2f}")
        if baseline.get("cost"):
            console.print(f"  Cost: ${baseline['cost']:.4f}")
        if baseline.get("latency"):
            console.print(f"  Latency: {baseline['latency']:.0f}ms")
        console.print(f"  Created: {baseline['created_at']}")
        if baseline.get("git_commit"):
            console.print(
                f"  Git: {baseline['git_commit']} ({baseline.get('git_branch', 'unknown')})"
            )
        console.print()
    else:
        # Show all baselines
        # Get all unique test names from results
        results = tracker.db.get_recent_results(days=30)
        unique_tests = set(r["test_name"] for r in results)

        table = Table(title="Test Baselines", show_header=True, header_style="bold cyan")
        table.add_column("Test Name", style="white")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Cost", justify="right", style="yellow")
        table.add_column("Latency", justify="right", style="blue")
        table.add_column("Created", style="dim")

        has_baselines = False
        for test_name in sorted(unique_tests):
            baseline = tracker.db.get_baseline(test_name)
            if baseline:
                has_baselines = True
                table.add_row(
                    test_name,
                    f"{baseline['score']:.1f}",
                    f"${baseline.get('cost', 0):.4f}" if baseline.get("cost") else "N/A",
                    f"{baseline.get('latency', 0):.0f}ms" if baseline.get("latency") else "N/A",
                    baseline["created_at"][:10],
                )

        if not has_baselines:
            console.print(
                "[yellow]⚠️  No baselines set. Run 'evalview baseline set' first.[/yellow]"
            )
        else:
            console.print()
            console.print(table)
            console.print()


@baseline.command("clear")
@click.option(
    "--test",
    help="Specific test name to clear baseline for",
)
@click.confirmation_option(prompt="Are you sure you want to clear baselines?")
@track_command("baseline_clear")
def baseline_clear(test: str):
    """Clear baselines."""
    from evalview.tracking import RegressionTracker

    tracker = RegressionTracker()

    if test:
        # Clear specific baseline (would need to add this to DB class)
        console.print("[yellow]⚠️  Clear specific baseline not yet implemented[/yellow]")
    else:
        tracker.db.clear_baselines()
        console.print("[green]✅ All baselines cleared[/green]")


@main.command()
@click.option(
    "--days",
    default=30,
    help="Number of days to analyze (default: 30)",
)
@click.option(
    "--test",
    help="Specific test name to show trends for",
)
@track_command("trends")
def trends(days: int, test: str):
    """Show performance trends over time."""
    from evalview.tracking import RegressionTracker
    from rich.table import Table

    tracker = RegressionTracker()

    if test:
        # Show trends for specific test
        stats = tracker.get_statistics(test, days)

        if stats["total_runs"] == 0:
            console.print(f"[yellow]⚠️  No data found for test: {test}[/yellow]")
            return

        console.print(f"\n[bold]Performance Trends: {test}[/bold]")
        console.print(f"Period: Last {days} days\n")

        console.print("[cyan]Test Runs:[/cyan]")
        console.print(f"  Total: {stats['total_runs']}")
        console.print(f"  Passed: {stats['passed_runs']} ({stats['pass_rate']:.1f}%)")
        console.print(f"  Failed: {stats['failed_runs']}")

        if stats["score"]["current"]:
            console.print("\n[cyan]Score:[/cyan]")
            console.print(f"  Current: {stats['score']['current']:.1f}")
            console.print(f"  Average: {stats['score']['avg']:.1f}")
            console.print(f"  Range: {stats['score']['min']:.1f} - {stats['score']['max']:.1f}")

        if stats["cost"]["current"]:
            console.print("\n[cyan]Cost:[/cyan]")
            console.print(f"  Current: ${stats['cost']['current']:.4f}")
            console.print(f"  Average: ${stats['cost']['avg']:.4f}")
            console.print(f"  Range: ${stats['cost']['min']:.4f} - ${stats['cost']['max']:.4f}")

        if stats["latency"]["current"]:
            console.print("\n[cyan]Latency:[/cyan]")
            console.print(f"  Current: {stats['latency']['current']:.0f}ms")
            console.print(f"  Average: {stats['latency']['avg']:.0f}ms")
            console.print(
                f"  Range: {stats['latency']['min']:.0f}ms - {stats['latency']['max']:.0f}ms"
            )

        console.print()

    else:
        # Show overall trends
        daily_trends = tracker.db.get_daily_trends(days)

        if not daily_trends:
            console.print(f"[yellow]⚠️  No trend data available for last {days} days[/yellow]")
            return

        console.print("\n[bold]Overall Performance Trends[/bold]")
        console.print(f"Period: Last {days} days\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Date", style="white")
        table.add_column("Avg Score", justify="right", style="green")
        table.add_column("Avg Cost", justify="right", style="yellow")
        table.add_column("Avg Latency", justify="right", style="blue")
        table.add_column("Tests", justify="center", style="dim")
        table.add_column("Pass Rate", justify="right", style="green")

        for trend in daily_trends[-14:]:  # Show last 14 days
            pass_rate = (
                trend["passed_tests"] / trend["total_tests"] * 100
                if trend["total_tests"] > 0
                else 0
            )

            table.add_row(
                trend["date"],
                f"{trend['avg_score']:.1f}" if trend["avg_score"] else "N/A",
                f"${trend['avg_cost']:.4f}" if trend.get("avg_cost") else "N/A",
                f"{trend['avg_latency']:.0f}ms" if trend.get("avg_latency") else "N/A",
                str(trend["total_tests"]),
                f"{pass_rate:.0f}%",
            )

        console.print(table)
        console.print()


@main.command()
@click.argument("test_file", type=click.Path(exists=True))
@click.option(
    "--count",
    "-n",
    default=10,
    type=int,
    help="Number of variations to generate (default: 10)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for generated tests (default: same as input)",
)
@click.option(
    "--edge-cases/--no-edge-cases",
    default=True,
    help="Include edge case variations (default: True)",
)
@click.option(
    "--focus",
    "-f",
    help="Focus variations on specific aspect (e.g., 'different stock tickers')",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview generated tests without saving",
)
@track_command("expand", lambda **kw: {"count": kw.get("count"), "edge_cases": kw.get("edge_cases")})
def expand(test_file: str, count: int, output_dir: str, edge_cases: bool, focus: str, dry_run: bool):
    """Expand a test case into variations using LLM.

    Takes a base test case and generates variations with different inputs,
    edge cases, and scenarios. Great for building comprehensive test suites
    from a few seed tests.

    Example:
        evalview expand tests/test-cases/stock-basic.yaml --count 20
    """
    asyncio.run(_expand_async(test_file, count, output_dir, edge_cases, focus, dry_run))


async def _expand_async(
    test_file: str,
    count: int,
    output_dir: str,
    edge_cases: bool,
    focus: str,
    dry_run: bool,
):
    """Async implementation of expand command."""
    from evalview.expander import TestExpander
    from evalview.core.loader import TestCaseLoader
    from rich.table import Table

    console.print("[blue]🔄 Expanding test case...[/blue]\n")

    # Load base test
    test_path = Path(test_file)
    console.print(f"[dim]Loading: {test_path}[/dim]")

    try:
        base_test = TestCaseLoader.load_from_file(test_path)
        if not base_test:
            console.print(f"[red]❌ No test cases found in {test_file}[/red]")
            return
    except Exception as e:
        console.print(f"[red]❌ Failed to load test: {e}[/red]")
        return

    console.print(f"[green]✓[/green] Base test: [bold]{base_test.name}[/bold]")
    console.print(f"  Query: \"{base_test.input.query}\"")
    console.print()

    # Initialize expander
    try:
        expander = TestExpander()
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        return

    # Show provider info
    if expander.message:
        console.print(f"[yellow]ℹ️  {expander.message}[/yellow]")
    console.print(f"[dim]Using {expander.provider.capitalize()} for test generation[/dim]")
    console.print()

    # Generate variations
    console.print(f"[cyan]🤖 Generating {count} variations...[/cyan]")
    if focus:
        console.print(f"[dim]   Focus: {focus}[/dim]")
    if edge_cases:
        console.print("[dim]   Including edge cases[/dim]")
    console.print()

    try:
        variations = await expander.expand(
            base_test,
            count=count,
            include_edge_cases=edge_cases,
            variation_focus=focus,
        )
    except Exception as e:
        console.print(f"[red]❌ Failed to generate variations: {e}[/red]")
        console.print("[dim]Make sure OPENAI_API_KEY or ANTHROPIC_API_KEY is set[/dim]")
        return

    if not variations:
        console.print("[yellow]⚠️  No variations generated[/yellow]")
        return

    console.print(f"[green]✓[/green] Generated {len(variations)} variations\n")

    # Convert to TestCase objects
    test_cases = [
        expander.convert_to_test_case(v, base_test, i)
        for i, v in enumerate(variations, 1)
    ]

    # Show preview table
    table = Table(title="Generated Test Variations", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="white", no_wrap=False)
    table.add_column("Query", style="dim", no_wrap=False)
    table.add_column("Edge?", style="yellow", justify="center", width=5)

    for i, (variation, tc) in enumerate(zip(variations, test_cases), 1):
        is_edge = "⚠️" if variation.get("is_edge_case") else ""
        query_preview = tc.input.query[:50] + "..." if len(tc.input.query) > 50 else tc.input.query
        table.add_row(str(i), tc.name, query_preview, is_edge)

    console.print(table)
    console.print()

    if dry_run:
        console.print("[yellow]Dry run - no files saved[/yellow]")
        return

    # Ask for confirmation
    if not click.confirm("Save these test variations?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Determine output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = test_path.parent

    # Generate prefix from base test name
    prefix = re.sub(r'[^a-z0-9]+', '-', base_test.name.lower()).strip('-')[:20]
    prefix = f"{prefix}-var"

    # Save variations
    console.print(f"\n[cyan]💾 Saving to {out_path}/...[/cyan]")
    saved_paths = expander.save_variations(test_cases, out_path, prefix=prefix)

    console.print(f"\n[green]✅ Saved {len(saved_paths)} test variations:[/green]")
    for path in saved_paths[:5]:  # Show first 5
        console.print(f"   • {path.name}")
    if len(saved_paths) > 5:
        console.print(f"   • ... and {len(saved_paths) - 5} more")

    # Suggest run command with correct path (use --pattern for file matching)
    console.print(f"\n[blue]Run with:[/blue] evalview run {out_path} --pattern '{prefix}*.yaml'")


@main.command()
@track_command("demo", lambda **kw: {"is_demo": True})
def demo():
    """🎬 Live regression demo — spins up a real agent and catches a real regression."""
    import socket
    import shutil
    import tempfile
    import threading
    import subprocess as _subprocess
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from rich.rule import Rule

    console.print()

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
                # $2.50/escalation × 50K daily tickets = $125K/day in unnecessary ops cost
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
                # Customers can't understand their bill → chargebacks, escalations, churn
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



@main.command()
@click.argument("pattern", required=False)
@click.option("--tool", help="Tool name to use in the test")
@click.option("--query", help="Query to use in the test")
@click.option("--list", "list_patterns", is_flag=True, help="List available patterns")
@click.option("--output", "-o", help="Output file path (default: tests/<pattern>.yaml)")
@track_command("add")
def add(pattern: Optional[str], tool: Optional[str], query: Optional[str], list_patterns: bool, output: Optional[str]):
    """Add a test pattern to your project.

    Examples:
        evalview add                           # List available patterns
        evalview add tool-not-called           # Copy pattern to tests/
        evalview add cost-budget --output my-test.yaml
        evalview add tool-not-called --tool get_weather --query "What's the weather?"
    """
    import shutil

    # Find templates directory
    templates_dir = Path(__file__).parent / "templates" / "patterns"

    if not templates_dir.exists():
        console.print("[red]Error: Templates directory not found[/red]")
        return

    # List available patterns
    available_patterns = [f.stem for f in templates_dir.glob("*.yaml")]

    if list_patterns or not pattern:
        console.print("\n[bold cyan]Available Test Patterns[/bold cyan]\n")

        for p in sorted(available_patterns):
            # Read description from file
            pattern_file = templates_dir / f"{p}.yaml"
            with open(pattern_file) as f:
                content = f.read()
                # Extract first comment line as description
                lines = content.split("\n")
                desc = ""
                for line in lines:
                    if line.startswith("# Pattern:"):
                        desc = line.replace("# Pattern:", "").strip()
                        break
                    elif line.startswith("#") and not line.startswith("# "):
                        continue
                    elif line.startswith("# ") and "Common failure" not in line and "Customize" not in line:
                        desc = line.replace("# ", "").strip()
                        if desc:
                            break

            console.print(f"  [green]{p}[/green]")
            if desc:
                console.print(f"    [dim]{desc}[/dim]")

        console.print("\n[dim]Usage: evalview add <pattern-name>[/dim]")
        console.print("[dim]       evalview add <pattern-name> --tool my_tool --query \"My query\"[/dim]\n")
        return

    # Check if pattern exists
    if pattern not in available_patterns:
        console.print(f"[red]Error: Pattern '{pattern}' not found[/red]")
        console.print(f"[dim]Available: {', '.join(available_patterns)}[/dim]")
        return

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        # Create tests directory if needed
        tests_dir = Path("tests")
        tests_dir.mkdir(exist_ok=True)
        output_path = tests_dir / f"{pattern}.yaml"

    # Check if file exists
    if output_path.exists():
        if not click.confirm(f"File {output_path} already exists. Overwrite?"):
            console.print("[yellow]Aborted[/yellow]")
            return

    # Read template
    template_path = templates_dir / f"{pattern}.yaml"
    with open(template_path) as f:
        content = f.read()

    # Apply substitutions if provided
    if tool:
        # Replace tool names in common patterns
        content = content.replace("calculator", tool)
        content = content.replace("retriever", tool)

    if query:
        # Replace query strings
        import re
        content = re.sub(r'query: "[^"]*"', f'query: "{query}"', content)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    console.print(f"\n[green]✓[/green] Created [cyan]{output_path}[/cyan]")

    # Show what was created
    console.print(f"\n[dim]━━━ {output_path} ━━━[/dim]")
    # Show first 20 lines
    lines = content.split("\n")[:20]
    for line in lines:
        if line.startswith("#"):
            console.print(f"[dim]{line}[/dim]")
        else:
            console.print(line)
    if len(content.split("\n")) > 20:
        console.print("[dim]...[/dim]")

    console.print(f"\n[bold]Next steps:[/bold]")
    console.print(f"  1. Edit [cyan]{output_path}[/cyan] to match your agent")
    console.print(f"  2. Run: [green]evalview run {output_path}[/green]\n")


# ============================================================================
# Judge Configuration Command
# ============================================================================


@main.command()
@click.argument("provider", required=False, type=click.Choice(["openai", "anthropic", "gemini", "grok", "ollama"]))
@click.argument("model", required=False)
@track_command("judge", lambda **kw: {"provider": kw.get("provider")})
def judge(provider: Optional[str], model: Optional[str]):
    """Set the LLM-as-judge provider and model.

    Examples:
        evalview judge                     # Show current judge config
        evalview judge openai              # Switch to OpenAI (default model)
        evalview judge openai gpt-4o       # Switch to OpenAI with specific model
        evalview judge anthropic           # Switch to Anthropic
        evalview judge ollama llama3.2     # Use local Ollama
    """
    config_path = Path(".evalview/config.yaml")

    # Load existing config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # If no provider specified, show current config
    if not provider:
        current = config.get("judge", {})
        if current:
            console.print(f"\n[bold]Current LLM-as-judge:[/bold]")
            console.print(f"  Provider: [cyan]{current.get('provider', 'not set')}[/cyan]")
            console.print(f"  Model: [cyan]{current.get('model', 'default')}[/cyan]\n")
        else:
            console.print("\n[dim]No judge configured. Using interactive selection.[/dim]")
            console.print("\n[bold]Set a judge:[/bold]")
            console.print("  evalview judge openai gpt-4o")
            console.print("  evalview judge anthropic claude-sonnet-4-5-20250929")
            console.print("  evalview judge ollama llama3.2\n")
        return

    # Default models per provider
    default_models = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-5-20250929",
        "gemini": "gemini-1.5-pro",
        "grok": "grok-beta",
        "ollama": "llama3.2",
    }

    # Set the judge config
    config["judge"] = {
        "provider": provider,
        "model": model or default_models.get(provider, "default"),
    }

    # Ensure directory exists
    config_path.parent.mkdir(exist_ok=True)

    # Write config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]✓[/green] Judge set to [bold]{provider}[/bold] / [cyan]{config['judge']['model']}[/cyan]")
    console.print(f"[dim]  Saved to {config_path}[/dim]\n")


# ============================================================================
# Skills Commands
# ============================================================================


@main.group()
def skill():
    """Commands for testing Claude Code skills."""
    pass


@skill.command("validate")
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Search subdirectories for SKILL.md files")
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
@click.option("--verbose", "-v", is_flag=True, help="Show INFO suggestions")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@track_command("skill_validate", lambda **kw: {"strict": kw.get("strict"), "recursive": kw.get("recursive")})
def skill_validate(path: str, recursive: bool, strict: bool, verbose: bool, output_json: bool) -> None:
    """Validate Claude Code skill(s).

    Validates SKILL.md files for:
    - Correct structure and frontmatter
    - Valid naming conventions
    - Policy compliance
    - Best practices

    Args:
        path: Path to SKILL.md file or directory
        recursive: Search subdirectories for SKILL.md files
        strict: Treat warnings as errors
        verbose: Show INFO-level suggestions
        output_json: Output results as JSON

    Examples:
        evalview skill validate ./my-skill/SKILL.md
        evalview skill validate ./skills/ --recursive
        evalview skill validate ./SKILL.md --strict
        evalview skill validate ./skills/ -rv  # verbose with suggestions
    """
    from pathlib import Path as PathLib
    from evalview.skills import SkillValidator, SkillParser

    path_obj = PathLib(path)

    # Collect files to validate
    if path_obj.is_file():
        files = [str(path_obj)]
    else:
        files = SkillParser.find_skills(str(path_obj), recursive=recursive)
        if not files:
            if output_json:
                console.print(json.dumps({"error": "No SKILL.md files found", "files": []}))
            else:
                console.print(f"[yellow]No SKILL.md files found in {path}[/yellow]")
                if not recursive:
                    console.print("[dim]Tip: Use --recursive to search subdirectories[/dim]")
            return

    # Validate each file with timing
    start_time = time.time()

    results = {}
    total_errors = 0
    total_warnings = 0
    total_valid = 0

    for file_path in files:
        result = SkillValidator.validate_file(file_path)
        results[file_path] = result

        total_errors += len(result.errors)
        total_warnings += len(result.warnings)
        if result.valid:
            total_valid += 1

    elapsed_ms = (time.time() - start_time) * 1000

    # Output results
    if output_json:
        json_output = {
            "summary": {
                "total_files": len(files),
                "valid": total_valid,
                "invalid": len(files) - total_valid,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
            },
            "results": {
                path: {
                    "valid": r.valid,
                    "errors": [e.model_dump() for e in r.errors],
                    "warnings": [w.model_dump() for w in r.warnings],
                    "info": [i.model_dump() for i in r.info],
                }
                for path, r in results.items()
            },
        }
        console.print(json.dumps(json_output, indent=2))
        return

    # Rich console output with EvalView banner
    print_evalview_banner(console, subtitle="[dim]Catch agent regressions before you ship[/dim]")
    console.print("[dim]Validating against official Anthropic spec...[/dim]")
    console.print()

    for file_path, result in results.items():
        # File header
        status_icon = "[green]✓[/green]" if result.valid else "[red]✗[/red]"
        console.print(f"{status_icon} [bold]{file_path}[/bold]")

        # Skill info if valid
        if result.skill:
            console.print(f"   [dim]Name: {result.skill.metadata.name}[/dim]")
            console.print(f"   [dim]Tokens: ~{result.skill.token_estimate}[/dim]")

        # Errors
        for error in result.errors:
            console.print(f"   [red]ERROR[/red] [{error.code}] {error.message}")
            if error.suggestion:
                console.print(f"         [dim]→ {error.suggestion}[/dim]")

        # Warnings
        for warning in result.warnings:
            console.print(f"   [yellow]WARN[/yellow]  [{warning.code}] {warning.message}")
            if warning.suggestion:
                console.print(f"         [dim]→ {warning.suggestion}[/dim]")

        # Info (only show if verbose flag is set)
        if verbose:
            for info in result.info:
                console.print(f"   [blue]INFO[/blue]  [{info.code}] {info.message}")
                if info.suggestion:
                    console.print(f"         [dim]→ {info.suggestion}[/dim]")

        console.print()

    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Files:    {len(files)}")
    console.print(f"  Valid:    [green]{total_valid}[/green]")
    console.print(f"  Invalid:  [red]{len(files) - total_valid}[/red]")
    console.print(f"  Errors:   [red]{total_errors}[/red]")
    console.print(f"  Warnings: [yellow]{total_warnings}[/yellow]")
    console.print(f"  Time:     [dim]{elapsed_ms:.0f}ms[/dim]")
    console.print()

    # Exit with error code if validation failed
    if total_errors > 0 or (strict and total_warnings > 0):
        raise SystemExit(1)


@skill.command("list")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--recursive", "-r", is_flag=True, default=True, help="Search subdirectories")
@track_command("skill_list")
def skill_list(path: str, recursive: bool) -> None:
    """List all skills in a directory.

    Args:
        path: Directory path to search for skills
        recursive: Search subdirectories

    Examples:
        evalview skill list
        evalview skill list ./my-skills/
        evalview skill list ~/.claude/skills/
    """
    from pathlib import Path as PathLib
    from evalview.skills import SkillParser, SkillValidator

    files = SkillParser.find_skills(path, recursive=recursive)

    if not files:
        console.print(f"[yellow]No SKILL.md files found in {path}[/yellow]")
        return

    console.print(f"\n[bold cyan]━━━ Skills in {path} ━━━[/bold cyan]\n")

    for file_path in files:
        result = SkillValidator.validate_file(file_path)
        status = "[green]✓[/green]" if result.valid else "[red]✗[/red]"

        if result.skill:
            console.print(f"  {status} [bold]{result.skill.metadata.name}[/bold]")
            # Use constant for max description length
            desc = result.skill.metadata.description
            if len(desc) > MAX_DESCRIPTION_LENGTH:
                console.print(f"      [dim]{desc[:MAX_DESCRIPTION_LENGTH]}...[/dim]")
            else:
                console.print(f"      [dim]{desc}[/dim]")
            console.print(f"      [dim]{file_path}[/dim]")
        else:
            console.print(f"  {status} [red]{file_path}[/red]")
            if result.errors:
                console.print(f"      [red]{result.errors[0].message}[/red]")

        console.print()

    console.print(f"[dim]Total: {len(files)} skill(s)[/dim]\n")


@skill.command("doctor")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--recursive", "-r", is_flag=True, default=True, help="Search subdirectories")
@track_command("skill_doctor")
def skill_doctor(path: str, recursive: bool) -> None:
    """Diagnose skill issues that cause Claude Code problems.

    Checks for common issues:
    - Total description chars exceeding Claude Code's 15k budget
    - Duplicate skill names
    - Invalid skills
    - Multi-line descriptions that break formatters

    Args:
        path: Directory path to search for skills
        recursive: Search subdirectories

    Examples:
        evalview skill doctor ~/.claude/skills/
        evalview skill doctor .claude/skills/
        evalview skill doctor ./my-skills/ -r
    """
    from pathlib import Path as PathLib
    from evalview.skills import SkillParser, SkillValidator

    start_time = time.time()

    files = SkillParser.find_skills(path, recursive=recursive)

    if not files:
        console.print(f"[yellow]No SKILL.md files found in {path}[/yellow]\n")
        console.print("[bold white]Here's what skill doctor catches:[/bold white]\n")
        console.print(
            Panel(
                "[bold red]⚠️  Character Budget: 127% OVER[/bold red]\n"
                "[red]Claude is ignoring ~4 of your 24 skills[/red]\n\n"
                "[red]✗[/red] my-claude-helper [dim]- reserved word \"claude\" in name[/dim]\n"
                "[red]✗[/red] api-tools [dim]- multiline description (breaks with Prettier)[/dim]\n"
                "[red]✗[/red] code-review [dim]- description too long (1847 chars)[/dim]\n"
                "[green]✓[/green] git-commit [dim]- OK[/dim]\n"
                "[green]✓[/green] test-runner [dim]- OK[/dim]",
                title="[bold]Example Output[/bold]",
                border_style="dim",
            )
        )
        console.print("\n[dim]Create skills in .claude/skills/ or ~/.claude/skills/[/dim]")
        return

    # Analyze all skills
    skills_data = []
    total_desc_chars = 0
    names_seen = {}
    invalid_count = 0
    multiline_count = 0

    for file_path in files:
        result = SkillValidator.validate_file(file_path)
        if result.valid and result.skill:
            name = result.skill.metadata.name
            desc = result.skill.metadata.description
            desc_len = len(desc)
            total_desc_chars += desc_len

            # Track duplicates
            if name in names_seen:
                names_seen[name].append(file_path)
            else:
                names_seen[name] = [file_path]

            # Track multi-line
            if "\n" in desc:
                multiline_count += 1

            skills_data.append({
                "name": name,
                "path": file_path,
                "desc_chars": desc_len,
                "valid": True,
            })
        else:
            invalid_count += 1
            skills_data.append({
                "name": "INVALID",
                "path": file_path,
                "desc_chars": 0,
                "valid": False,
                "error": result.errors[0].message if result.errors else "Unknown error",
            })

    elapsed_ms = (time.time() - start_time) * 1000

    # Find duplicates
    duplicates = {name: paths for name, paths in names_seen.items() if len(paths) > 1}

    # Output report
    print_evalview_banner(console, subtitle="[dim]Skill Doctor - Diagnose Claude Code Issues[/dim]")

    # Character budget check
    budget_pct = (total_desc_chars / CLAUDE_CODE_CHAR_BUDGET) * 100
    skills_over = max(0, int((total_desc_chars - CLAUDE_CODE_CHAR_BUDGET) / AVG_CHARS_PER_SKILL))

    if budget_pct > CHAR_BUDGET_CRITICAL_PCT:
        console.print(f"[bold red]⚠️  Character Budget: {budget_pct:.0f}% OVER - Claude is ignoring ~{skills_over} of your {len(files)} skills[/bold red]")
    elif budget_pct > CHAR_BUDGET_WARNING_PCT:
        console.print(f"[bold yellow]⚠️  Character Budget: {budget_pct:.0f}% - approaching limit[/bold yellow]")
    else:
        console.print(f"[bold green]✓ Character Budget: {budget_pct:.0f}% ({total_desc_chars:,} / {CLAUDE_CODE_CHAR_BUDGET:,} chars)[/bold green]")
    console.print(f"[bold]Total Skills:[/bold]      {len(files)}")
    console.print(f"[bold]Valid:[/bold]             [green]{len(files) - invalid_count}[/green]")
    console.print(f"[bold]Invalid:[/bold]           [red]{invalid_count}[/red]")
    console.print(f"[bold]Duplicates:[/bold]        [{'red' if duplicates else 'green'}]{len(duplicates)}[/{'red' if duplicates else 'green'}]")
    console.print(f"[bold]Multi-line Desc:[/bold]   [{'yellow' if multiline_count else 'green'}]{multiline_count}[/{'yellow' if multiline_count else 'green'}]")
    console.print()

    # Show issues
    has_issues = False

    if budget_pct > CHAR_BUDGET_CRITICAL_PCT:
        has_issues = True
        console.print("[bold red]ISSUE: Character budget exceeded[/bold red]")
        console.print("  Claude Code won't see all your skills.")
        console.print(f"  [dim]Fix: Set SLASH_COMMAND_TOOL_CHAR_BUDGET={CLAUDE_CODE_CHAR_BUDGET * 2} or reduce descriptions[/dim]")
        console.print()

    if duplicates:
        has_issues = True
        console.print("[bold red]ISSUE: Duplicate skill names[/bold red]")
        for name, paths in duplicates.items():
            console.print(f"  [yellow]{name}[/yellow] defined in:")
            for p in paths:
                console.print(f"    - {p}")
        console.print()

    if invalid_count > 0:
        has_issues = True
        console.print("[bold red]ISSUE: Invalid skills[/bold red]")
        for s in skills_data:
            if not s["valid"]:
                console.print(f"  [red]✗[/red] {s['path']}")
                console.print(f"    [dim]{s.get('error', 'Unknown error')}[/dim]")
        console.print()

    if multiline_count > 0:
        console.print("[bold yellow]WARNING: Multi-line descriptions[/bold yellow]")
        console.print("  These may break with Prettier or YAML formatters.")
        console.print("  [dim]Fix: Use single-line descriptions[/dim]")
        console.print()

    # Summary
    if not has_issues and multiline_count == 0:
        console.print("[bold green]✓ All skills visible to Claude[/bold green]")
    elif not has_issues:
        console.print("[bold yellow]⚠ Minor warnings - skills should work[/bold yellow]")
    else:
        invisible_count = skills_over + invalid_count + len(duplicates)
        if invisible_count > 0:
            console.print(f"[bold red]✗ {invisible_count} skill(s) are INVISIBLE to Claude - fix now[/bold red]")
        else:
            console.print("[bold red]✗ Issues found - fix before deploying[/bold red]")

    console.print(f"\n[dim]Time: {elapsed_ms:.0f}ms[/dim]\n")


def _run_agent_skill_test(
    test_file: str,
    agent: str,
    trace_dir: str,
    no_rubric: bool,
    cwd: str,
    max_turns: int,
    verbose: bool,
    output_json: bool,
    model: str,
) -> None:
    """Run agent-based skill tests (internal helper).

    This is called when --agent is specified or when the YAML
    contains an agent config with type != system-prompt.

    Args:
        test_file: Path to YAML test file
        agent: Agent type to override (if any)
        trace_dir: Directory to save traces
        no_rubric: Skip rubric evaluation
        cwd: Working directory override
        max_turns: Max conversation turns
        verbose: Show detailed output
        output_json: Output as JSON
        model: Model to use for rubric evaluation
    """
    # 1. Validate agent type
    agent_type_enum = validate_and_parse_agent_type(agent, console)

    # 2. Load test suite
    suite, runner = load_test_suite(
        test_file, agent_type_enum, trace_dir,
        no_rubric, cwd, max_turns, verbose, model,
        console
    )

    # 3. Print banner and suite info
    print_evalview_banner(console, subtitle="[dim]Agent-Based Skill Testing[/dim]")
    print_suite_info(suite, trace_dir, console)

    # 4. Run tests asynchronously with spinner
    start_time = time.time()

    async def run_tests_async():
        return await runner.run_suite(suite)

    result, run_error = run_async_with_spinner(
        console,
        "Running agent tests...",
        run_tests_async
    )

    if run_error:
        console.print(f"[red]Error running tests: {run_error}[/red]")
        raise SystemExit(1)

    elapsed_ms = (time.time() - start_time) * 1000

    # 5. Output JSON if requested
    if output_json:
        json_output = format_results_as_json(result)
        console.print(json.dumps(json_output, indent=2))
        return

    # 6. Display results table
    table = build_results_table(result)
    console.print(table)
    console.print()

    # 7. Print detailed results
    print_detailed_test_results(result, verbose, console)

    # 8. Display summary panel
    summary_panel = build_summary_panel(result, elapsed_ms)
    console.print(summary_panel)

    # 9. Handle completion (success/failure)
    handle_test_completion(result, test_file, suite, console)


@skill.command("test")
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="Model to use")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--agent", "-a",
    type=click.Choice(["system-prompt", "claude-code", "codex", "openclaw",
                       "langgraph", "crewai", "openai-assistants", "custom"]),
    default=None,
    help="Agent type (overrides YAML). Default: system-prompt (legacy mode)"
)
@click.option(
    "--trace", "-t",
    type=click.Path(),
    default=None,
    help="Directory to save JSONL traces for debugging"
)
@click.option(
    "--no-rubric",
    is_flag=True,
    help="Skip Phase 2 rubric evaluation (deterministic only)"
)
@click.option(
    "--cwd",
    type=click.Path(exists=True),
    default=None,
    help="Working directory for agent execution"
)
@click.option(
    "--max-turns",
    type=int,
    default=None,
    help="Maximum conversation turns (default: 10)"
)
@track_command("skill_test", lambda **kw: {"agent": kw.get("agent"), "no_rubric": kw.get("no_rubric")})
def skill_test(
    test_file: str,
    model: str,
    verbose: bool,
    output_json: bool,
    agent: str,
    trace: str,
    no_rubric: bool,
    cwd: str,
    max_turns: int,
) -> None:
    """Run behavior tests against a skill.

    TEST_FILE is a YAML file defining test cases for a skill.

    Args:
        test_file: Path to YAML test file
        model: Model to use for evaluation
        verbose: Show detailed output
        output_json: Output as JSON
        agent: Agent type override
        trace: Directory to save traces
        no_rubric: Skip rubric evaluation
        cwd: Working directory override
        max_turns: Max conversation turns

    Example test file (legacy mode):
        name: test-code-reviewer
        skill: ./skills/code-reviewer/SKILL.md
        tests:
          - name: detects-sql-injection
            input: "Review: query = f'SELECT * FROM users WHERE id = {id}'"
            expected:
              output_contains: ["SQL injection", "parameterized"]

    Example test file (agent mode):
        name: test-code-reviewer
        skill: ./skills/code-reviewer/SKILL.md
        agent:
          type: claude-code
          max_turns: 10
        tests:
          - name: writes-review-file
            input: "Review the code and save to review.md"
            expected:
              files_created: ["review.md"]
              tool_calls_contain: ["Write"]

    Examples:
        # Legacy mode (system prompt + string matching)
        evalview skill test tests/code-reviewer.yaml

        # Agent mode with Claude Code
        evalview skill test tests/my-skill.yaml --agent claude-code

        # Save traces for debugging
        evalview skill test tests/my-skill.yaml -a claude-code -t ./traces/

        # Deterministic checks only (skip rubric)
        evalview skill test tests/my-skill.yaml -a claude-code --no-rubric
    """
    import json as json_module
    import os
    import asyncio

    # Determine whether to use agent mode
    # Load YAML to check for agent config
    import yaml as yaml_module
    with open(test_file) as f:
        yaml_data = yaml_module.safe_load(f)

    yaml_agent_type = None
    if "agent" in yaml_data and yaml_data["agent"]:
        yaml_agent_type = yaml_data["agent"].get("type")

    use_agent_mode = (
        (agent is not None and agent != "system-prompt")
        or (yaml_agent_type is not None and yaml_agent_type != "system-prompt")
    )

    if use_agent_mode:
        # Use new agent-based runner
        _run_agent_skill_test(
            test_file=test_file,
            agent=agent,
            trace_dir=trace,
            no_rubric=no_rubric,
            cwd=cwd,
            max_turns=max_turns,
            verbose=verbose,
            output_json=output_json,
            model=model,
        )
        return

    # Legacy mode - use original SkillRunner
    from evalview.skills import SkillRunner

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]Error: ANTHROPIC_API_KEY environment variable required[/red]")
        console.print("[dim]Set it with: export ANTHROPIC_API_KEY=your-key[/dim]")
        raise SystemExit(1)

    try:
        runner = SkillRunner(model=model)
        suite = runner.load_test_suite(test_file)
    except Exception as e:
        console.print(f"[red]Error loading test suite: {e}[/red]")
        raise SystemExit(1)

    # EvalView banner
    print_evalview_banner(console, subtitle="[dim]Catch agent regressions before you ship[/dim]")
    console.print(f"  [bold]Suite:[/bold]  {suite.name}")
    console.print(f"  [bold]Skill:[/bold]  [cyan]{suite.skill}[/cyan]")
    console.print(f"  [bold]Model:[/bold]  {model}")
    console.print(f"  [bold]Tests:[/bold]  {len(suite.tests)}")
    console.print()

    # Run the suite with live spinner
    from evalview.skills.ui_utils import run_async_with_spinner

    start_time = time.time()

    async def run_tests_async():
        # Wrap synchronous runner.run_suite in async
        return runner.run_suite(suite)

    result, run_error = run_async_with_spinner(
        console,
        "Running tests...",
        run_tests_async
    )

    if run_error:
        console.print(f"[red]Error running tests: {run_error}[/red]")
        raise SystemExit(1)

    elapsed_ms = (time.time() - start_time) * 1000

    # Output results
    if output_json:
        json_output = {
            "suite_name": result.suite_name,
            "skill_name": result.skill_name,
            "passed": result.passed,
            "total_tests": result.total_tests,
            "passed_tests": result.passed_tests,
            "failed_tests": result.failed_tests,
            "pass_rate": result.pass_rate,
            "total_latency_ms": result.total_latency_ms,
            "avg_latency_ms": result.avg_latency_ms,
            "total_tokens": result.total_tokens,
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "score": r.score,
                    "input": r.input_query,
                    "output": r.output[:TRUNCATE_OUTPUT_LONG] + "..." if len(r.output) > TRUNCATE_OUTPUT_LONG else r.output,
                    "contains_failed": r.contains_failed,
                    "not_contains_failed": r.not_contains_failed,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in result.results
            ],
        }
        console.print(json.dumps(json_output, indent=2))
        return

    # Results table
    table = Table(title="Test Results", show_header=True, header_style="bold cyan")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Test", style="cyan")
    table.add_column("Score", justify="right", width=8)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Tokens", justify="right", width=8)

    for r in result.results:
        status = "[green]✅ PASS[/green]" if r.passed else "[red]❌ FAIL[/red]"
        score_color = "green" if r.score >= SCORE_THRESHOLD_HIGH else "yellow" if r.score >= SCORE_THRESHOLD_MEDIUM else "red"
        table.add_row(
            status,
            r.test_name,
            f"[{score_color}]{r.score:.0f}%[/{score_color}]",
            f"{r.latency_ms:.0f}ms",
            f"{r.input_tokens + r.output_tokens:,}",
        )

    console.print(table)
    console.print()

    # Detailed results for failed tests (or all if verbose)
    failed_results = [r for r in result.results if not r.passed]
    show_results = result.results if verbose else failed_results

    if show_results:
        for r in show_results:
            status_icon = "✅" if r.passed else "❌"
            status_color = "green" if r.passed else "red"

            console.print(f"[bold {status_color}]{status_icon} {r.test_name}[/bold {status_color}]")

            # Show query
            console.print("\n[bold]Input:[/bold]")
            query = r.input_query[:TRUNCATE_OUTPUT_SHORT] + "..." if len(r.input_query) > TRUNCATE_OUTPUT_SHORT else r.input_query
            for line in query.split('\n'):
                console.print(f"  [dim]{line}[/dim]")

            # Show response preview
            if verbose or not r.passed:
                console.print("\n[bold]Response:[/bold]")
                output = r.output[:TRUNCATE_OUTPUT_MEDIUM] + "..." if len(r.output) > TRUNCATE_OUTPUT_MEDIUM else r.output
                for line in output.split('\n')[:8]:
                    console.print(f"  {line}")
                if len(r.output.split('\n')) > 8:
                    console.print("  [dim]...[/dim]")

            # Show evaluation checks
            console.print("\n[bold]Evaluation Checks:[/bold]")

            # Contains checks
            if r.contains_passed:
                for phrase in r.contains_passed:
                    console.print(f"  [green]✓[/green] Contains: \"{phrase}\"")
            if r.contains_failed:
                for phrase in r.contains_failed:
                    console.print(f"  [red]✗[/red] Missing:  \"{phrase}\"")

            # Not contains checks
            if r.not_contains_passed:
                for phrase in r.not_contains_passed:
                    console.print(f"  [green]✓[/green] Excludes: \"{phrase}\"")
            if r.not_contains_failed:
                for phrase in r.not_contains_failed:
                    console.print(f"  [red]✗[/red] Found:    \"{phrase}\" (should not appear)")

            # Error if any
            if r.error:
                console.print(f"\n[bold red]Error:[/bold red] {r.error}")

            # Guidance for failed tests
            if not r.passed:
                console.print("\n[bold yellow]How to Fix:[/bold yellow]")
                if r.contains_failed:
                    console.print("  [yellow]• Your skill's instructions should guide Claude to mention:[/yellow]")
                    for phrase in r.contains_failed:
                        console.print(f"    [yellow]  - \"{phrase}\"[/yellow]")
                    console.print("  [yellow]• Consider adding explicit guidance in your SKILL.md[/yellow]")
                if r.not_contains_failed:
                    console.print("  [yellow]• Your skill is producing unwanted phrases:[/yellow]")
                    for phrase in r.not_contains_failed:
                        console.print(f"    [yellow]  - \"{phrase}\"[/yellow]")
                    console.print("  [yellow]• Add constraints or negative examples to your SKILL.md[/yellow]")
                if r.error:
                    console.print("  [yellow]• Check your API key and model availability[/yellow]")

            console.print()

    # Summary panel
    pass_rate_color = "green" if result.pass_rate >= 0.8 else "yellow" if result.pass_rate >= 0.5 else "red"
    status_text = "[green]● All Tests Passed[/green]" if result.passed else "[bold red]● Some Tests Failed[/bold red]"
    border_color = "green" if result.passed else "red"

    summary_content = (
        f"  {status_text}\n"
        f"\n"
        f"  [bold]✅ Passed:[/bold]       [green]{result.passed_tests}[/green]\n"
        f"  [bold]❌ Failed:[/bold]       [red]{result.failed_tests}[/red]\n"
        f"  [bold]📈 Pass Rate:[/bold]    [{pass_rate_color}]{result.pass_rate:.0%}[/{pass_rate_color}] (required: {suite.min_pass_rate:.0%})\n"
        f"\n"
        f"  [bold]⏱️  Avg Latency:[/bold] {result.avg_latency_ms:.0f}ms\n"
        f"  [bold]🔤 Total Tokens:[/bold] {result.total_tokens:,}\n"
        f"  [bold]⏲️  Total Time:[/bold]  {elapsed_ms:.0f}ms"
    )

    console.print(Panel(summary_content, title="[bold]Overall Statistics[/bold]", border_style=border_color))

    # Actionable next steps for failures
    if not result.passed:
        console.print()
        console.print("[bold yellow]⚠️  Skill Test Failed[/bold yellow]")
        console.print()
        console.print("[bold]Next Steps to Fix Your Skill:[/bold]")
        console.print("  1. Review the [bold]How to Fix[/bold] guidance above for each failed test")
        console.print("  2. Update your [cyan]SKILL.md[/cyan] instructions to address the issues")
        console.print("  3. Re-run: [dim]evalview skill test " + test_file + "[/dim]")
        console.print()
        console.print("[dim]Tip: Use --verbose to see full responses for passing tests too[/dim]")
        console.print()
        raise SystemExit(1)
    else:
        console.print()
        console.print("[bold green]✓ Skill ready for deployment[/bold green]")
        console.print()


@skill.command("generate-tests")
@click.argument("skill_file", type=click.Path(exists=True))
@click.option("--count", "-c", default=10, type=int, help="Number of tests to generate")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for tests.yaml (default: ./tests.yaml)",
)
@click.option(
    "--categories",
    type=str,
    help="Comma-separated test categories (explicit,implicit,contextual,negative)",
)
@click.option("--model", "-m", type=str, help="LLM model to use for generation (skips interactive selection)")
@click.option(
    "--auto",
    is_flag=True,
    help="Auto-select cheapest model (skip interactive selection)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview generated tests without saving",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed generation process")
@track_command("skill_generate_tests")
def skill_generate_tests(
    skill_file: str,
    count: int,
    output: Optional[str],
    categories: Optional[str],
    model: Optional[str],
    auto: bool,
    dry_run: bool,
    verbose: bool,
):
    """Auto-generate comprehensive test suites from SKILL.md.

    Uses LLM-powered few-shot learning to generate test cases across
    all categories: explicit, implicit, contextual, and negative.

    By default, shows an interactive model selector with cost estimates.
    Use --auto to skip selection and use the cheapest available model.

    Examples:
        evalview skill generate-tests ./SKILL.md
        evalview skill generate-tests ./SKILL.md --auto --count 15
        evalview skill generate-tests ./SKILL.md --model gpt-4o --dry-run
        evalview skill generate-tests ./SKILL.md -o my-tests.yaml
    """
    asyncio.run(
        _skill_generate_tests_async(
            skill_file,
            count,
            output,
            categories,
            model,
            auto,
            dry_run,
            verbose,
        )
    )


async def _skill_generate_tests_async(
    skill_file: str,
    count: int,
    output: Optional[str],
    categories: Optional[str],
    model: Optional[str],
    auto: bool,
    dry_run: bool,
    verbose: bool,
):
    """Async handler for skill generate-tests command."""
    from evalview.skills.test_generator import SkillTestGenerator
    from evalview.skills.parser import SkillParser
    from evalview.skills.agent_types import TestCategory
    from evalview.telemetry.client import track
    from evalview.telemetry.events import (
        SkillTestGenerationStartEvent,
        SkillTestGenerationCompleteEvent,
        SkillTestGenerationFailedEvent,
        UserFeedbackEvent,
    )

    start_time = time.time()
    print_evalview_banner(console, subtitle="[dim]Auto-generate comprehensive test suites[/dim]")

    # Parse categories
    category_list = None
    if categories:
        cat_names = [c.strip().lower() for c in categories.split(",")]
        category_list = []
        for cat_name in cat_names:
            try:
                category_list.append(TestCategory(cat_name))
            except ValueError:
                console.print(
                    f"[red]❌ Invalid category: {cat_name}[/red]",
                )
                console.print(
                    f"[dim]Valid categories: explicit, implicit, contextual, negative[/dim]"
                )
                raise SystemExit(1)

    # Parse skill file
    console.print("[bold]Parsing skill file...[/bold]")
    try:
        skill = SkillParser.parse_file(skill_file)
    except Exception as e:
        console.print(f"[red]❌ Failed to parse skill: {e}[/red]")
        raise SystemExit(1)

    console.print(f"[green]✓[/green] Loaded skill: [bold]{skill.metadata.name}[/bold]")
    console.print()

    # Initialize generator (with model selection if needed)
    try:
        # Check if running in non-interactive environment (CI/CD, Docker, etc.)
        import sys
        is_interactive = sys.stdin.isatty() and sys.stdout.isatty()

        if model:
            # User specified model explicitly
            generator = SkillTestGenerator(model=model)
            if verbose:
                console.print(f"[dim]Using model: {model}[/dim]")
                console.print()
        elif auto or not is_interactive:
            # Auto-select cheapest (explicit --auto or non-TTY environment)
            if not is_interactive and not auto:
                console.print(
                    "[yellow]⚠️  Non-interactive environment detected (CI/CD, Docker, etc.)[/yellow]"
                )
                console.print("[dim]Auto-selecting cheapest model...[/dim]")
                console.print()

            generator = SkillTestGenerator()
            console.print(
                f"[dim]Auto-selected: {generator.client.config.display_name} / "
                f"{generator.client.config.default_model}[/dim]"
            )
            console.print()
        else:
            # Interactive selection
            provider, api_key, selected_model = SkillTestGenerator.select_model_interactive(console)
            generator = SkillTestGenerator(model=selected_model)
            console.print()

    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        console.print()
        console.print("[bold]To fix this:[/bold]")
        console.print("  1. Set an API key:")
        console.print("     export OPENAI_API_KEY=sk-...")
        console.print("     export ANTHROPIC_API_KEY=sk-ant-...")
        console.print("     export GEMINI_API_KEY=...")
        console.print()
        raise SystemExit(1)

    # Track generation start
    track(
        SkillTestGenerationStartEvent(
            skill_name=skill.metadata.name,
            test_count=count,
            categories=[c.value for c in (category_list or [])] if category_list else [],
            model=generator.client.config.default_model,
            has_example_suite=False,  # Golden example always bundled
        )
    )

    # Generate tests with spinner
    console.print(
        f"[bold]Generating {count} tests for [cyan]{skill.metadata.name}[/cyan]...[/bold]"
    )
    console.print()

    try:
        with console.status("[bold green]Generating tests..."):
            suite = await generator.generate_test_suite(
                skill=skill,
                count=count,
                categories=category_list,
            )

        generation_time_ms = int((time.time() - start_time) * 1000)

        # Validate generated suite
        validation_errors = generator.validate_test_suite(suite)
        if validation_errors:
            console.print("[yellow]⚠️  Validation warnings:[/yellow]")
            for error in validation_errors:
                console.print(f"  - {error}")
            console.print()

        # Track success
        track(
            SkillTestGenerationCompleteEvent(
                skill_name=skill.metadata.name,
                tests_generated=len(suite.tests),
                generation_latency_ms=generation_time_ms,
                estimated_cost_usd=generator.get_generation_cost(),
                model=generator.client.config.default_model,
                validation_errors=len(validation_errors),
                categories_distribution=generator.get_category_distribution(suite),
            )
        )

    except Exception as e:
        # Track failure
        track(
            SkillTestGenerationFailedEvent(
                skill_name=skill.metadata.name,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                model=generator.client.config.default_model,
                attempt_number=1,
            )
        )
        console.print(f"[red]❌ Generation failed: {e}[/red]")
        raise SystemExit(1)

    # Display preview
    console.print("[bold green]✓ Generated test suite[/bold green]")
    console.print()

    # Show summary
    console.print("[bold]Test Suite Summary[/bold]")
    console.print(f"  Name: [cyan]{suite.name}[/cyan]")
    console.print(f"  Tests: {len(suite.tests)}")
    console.print(f"  Estimated Cost: [green]~${generator.get_generation_cost():.4f}[/green]")
    console.print()

    # Category breakdown
    dist = generator.get_category_distribution(suite)
    console.print("[bold]Category Distribution[/bold]")
    for cat, cnt in dist.items():
        console.print(f"  {cat}: {cnt}")
    console.print()

    # Show tests
    console.print("[bold]Generated Tests[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Should Trigger", style="green")
    table.add_column("Assertions", justify="right")

    for test in suite.tests:
        assertion_count = 0
        if test.expected:
            # Count non-None assertion fields
            assertion_count += bool(test.expected.tool_calls_contain)
            assertion_count += bool(test.expected.files_created)
            assertion_count += bool(test.expected.commands_ran)
            assertion_count += bool(test.expected.output_contains)
            assertion_count += bool(test.expected.output_not_contains)
        if test.rubric:
            assertion_count += 1

        trigger_emoji = "✓" if test.should_trigger else "✗"
        table.add_row(
            test.name,
            test.category.value,
            trigger_emoji,
            str(assertion_count),
        )

    console.print(table)
    console.print()

    # Dry-run mode
    if dry_run:
        console.print("[yellow]Dry-run mode: Not saving to disk[/yellow]")
        console.print()
        if verbose:
            console.print("[bold]Full YAML preview:[/bold]")
            # Show first test in detail
            if suite.tests:
                import yaml

                preview = {
                    "name": suite.name,
                    "description": suite.description,
                    "skill": suite.skill,
                    "agent": {"type": suite.agent.type.value},
                    "tests": [generator._serialize_test(suite.tests[0])],
                }
                console.print(Panel(yaml.dump(preview, default_flow_style=False)))
                console.print(
                    f"[dim]... and {len(suite.tests) - 1} more tests[/dim]"
                )
        return

    # Determine output path
    output_path = Path(output) if output else Path.cwd() / "tests.yaml"

    # Check if file exists
    if output_path.exists():
        console.print(f"[yellow]⚠️  File already exists: {output_path}[/yellow]")
        if not click.confirm("Overwrite?", default=False):
            console.print("[yellow]Cancelled[/yellow]")
            return

    # Confirm save
    console.print(f"Save to: [cyan]{output_path}[/cyan]")
    if not click.confirm("Save generated tests?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Save
    try:
        generator.save_as_yaml(suite, output_path)
        console.print()
        console.print(f"[bold green]✓ Saved to {output_path}[/bold green]")
        console.print()

        # Suggest next steps
        console.print("[bold]Next Steps:[/bold]")
        console.print(f"  1. Review the generated tests: [dim]cat {output_path}[/dim]")
        console.print(
            f"  2. Run the tests: [dim]evalview skill test {output_path}[/dim]"
        )
        console.print(
            f"  3. Iterate on failing tests by editing {output_path}"
        )
        console.print()

        # Prompt for feedback
        try:
            rating = click.prompt(
                "Rate this generation (1-5)",
                type=click.IntRange(1, 5),
                default=4,
                show_default=True,
            )
            would_use_again = click.confirm(
                "Would you use auto-generation again?", default=True
            )

            track(
                UserFeedbackEvent(
                    skill_name=skill.metadata.name,
                    rating=rating,
                    would_use_again=would_use_again,
                    feedback_text=None,
                )
            )
        except (KeyboardInterrupt, click.Abort):
            # User skipped feedback
            pass

    except Exception as e:
        console.print(f"[red]❌ Failed to save: {e}[/red]")
        raise SystemExit(1)


# ============================================================================
# Golden Trace Commands
# ============================================================================


@main.group()
def golden():
    """Manage golden traces (blessed baselines for regression detection).

    Golden traces are "blessed" test results that represent expected behavior.
    Use them with `evalview run --diff` to detect regressions.

    Examples:
        evalview golden save .evalview/results/2024-01-15T10:30:00.json
        evalview golden list
        evalview golden delete "My Test Case"
    """
    pass


@golden.command("save")
@click.argument("result_file", type=click.Path(exists=True))
@click.option("--notes", "-n", help="Notes about why this is the golden baseline")
@click.option("--test", "-t", help="Save only specific test (by name)")
@click.option("--variant", "-v", help="Save as named variant (for multi-reference goldens)")
@track_command("golden_save")
def golden_save(result_file: str, notes: str, test: str, variant: str):
    """Save a test result as the golden baseline.

    RESULT_FILE is a JSON file from `evalview run` (e.g., .evalview/results/xxx.json)

    Examples:
        evalview golden save .evalview/results/latest.json
        evalview golden save results.json --notes "v1.0 release baseline"
        evalview golden save results.json --test "List Directory Contents"
        evalview golden save results.json --variant "fast-path" --notes "Optimized path"
    """
    import json
    from evalview.core.golden import GoldenStore
    from evalview.core.types import EvaluationResult

    console.print("\n[cyan]━━━ Saving Golden Trace ━━━[/cyan]\n")

    # Load result file
    with open(result_file) as f:
        data = json.load(f)

    # Handle both single result and batch results
    results = []
    if type(data).__name__ == "list":
        results = data
    elif isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = [data]

    # Filter by test name if specified
    if test:
        results = [r for r in results if r.get("test_case") == test]
        if not results:
            console.print(f"[red]❌ No test found with name: {test}[/red]")
            return

    store = GoldenStore()

    for result_data in results:
        try:
            result = EvaluationResult.model_validate(result_data)

            # Check if golden already exists (for default or this specific variant)
            variant_exists = store._get_golden_path(result.test_case, variant).exists()
            if variant_exists:
                variant_label = f"variant '{variant}'" if variant else "default golden"
                if not click.confirm(
                    f"{variant_label.capitalize()} already exists for '{result.test_case}'. Overwrite?",
                    default=False,
                ):
                    console.print(f"[yellow]Skipped: {result.test_case}[/yellow]")
                    continue

            # Save golden (may raise ValueError if too many variants)
            try:
                path = store.save_golden(result, notes=notes, source_file=result_file, variant_name=variant)
            except ValueError as e:
                console.print(f"[red]❌ {e}[/red]")
                continue

            variant_label = f" (variant: {variant})" if variant else ""
            console.print(f"[green]✓ Saved golden:[/green] {result.test_case}{variant_label}")
            console.print(f"  [dim]Score: {result.score:.1f}[/dim]")
            console.print(f"  [dim]Tools: {len(result.trace.steps)} steps[/dim]")
            console.print(f"  [dim]File: {path}[/dim]")
            console.print()

        except Exception as e:
            console.print(f"[red]❌ Failed to save: {e}[/red]")

    console.print("[green]Done![/green]")
    console.print()
    console.print("[dim]⭐ EvalView saved your baseline! Star if it helped → github.com/hidai25/eval-view[/dim]\n")


@golden.command("list")
@track_command("golden_list")
def golden_list():
    """List all golden traces.

    Shows all saved golden baselines with metadata and variant counts.
    """
    from evalview.core.golden import GoldenStore

    store = GoldenStore()
    goldens_with_variants = store.list_golden_with_variants()

    if not goldens_with_variants:
        console.print("\n[yellow]No golden traces found.[/yellow]")
        console.print("[dim]Save one with: evalview golden save <result.json>[/dim]\n")
        return

    console.print("\n[cyan]━━━ Golden Traces ━━━[/cyan]\n")

    for item in sorted(goldens_with_variants, key=lambda x: x["metadata"].test_name):
        g = item["metadata"]
        variant_count = item["variant_count"]

        variant_label = f" ({variant_count} variants)" if variant_count > 1 else ""
        console.print(f"  [bold]{g.test_name}[/bold]{variant_label}")
        console.print(f"    [dim]Score: {g.score:.1f}[/dim]")
        console.print(f"    [dim]Blessed: {g.blessed_at.strftime('%Y-%m-%d %H:%M')}[/dim]")
        if g.notes:
            console.print(f"    [dim]Notes: {g.notes}[/dim]")
        console.print()

    console.print(f"[dim]Total: {len(goldens_with_variants)} test(s) with golden trace(s)[/dim]\n")


@golden.command("delete")
@click.argument("test_name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@track_command("golden_delete")
def golden_delete(test_name: str, force: bool):
    """Delete a golden trace.

    TEST_NAME is the name of the test case to delete.
    """
    from evalview.core.golden import GoldenStore

    store = GoldenStore()

    if not store.has_golden(test_name):
        console.print(f"\n[yellow]No golden trace found for: {test_name}[/yellow]\n")
        return

    if not force:
        if not click.confirm(f"Delete golden trace for '{test_name}'?", default=False):
            console.print("[dim]Cancelled[/dim]")
            return

    store.delete_golden(test_name)
    console.print(f"\n[green]✓ Deleted golden trace: {test_name}[/green]\n")


@golden.command("show")
@click.argument("test_name")
@track_command("golden_show")
def golden_show(test_name: str):
    """Show details of a golden trace.

    TEST_NAME is the name of the test case.
    """
    from evalview.core.golden import GoldenStore
    from rich.panel import Panel

    store = GoldenStore()
    golden = store.load_golden(test_name)

    if not golden:
        console.print(f"\n[yellow]No golden trace found for: {test_name}[/yellow]\n")
        return

    console.print(f"\n[cyan]━━━ Golden Trace: {test_name} ━━━[/cyan]\n")

    # Metadata
    console.print("[bold]Metadata:[/bold]")
    console.print(f"  Score: {golden.metadata.score:.1f}")
    console.print(f"  Blessed: {golden.metadata.blessed_at.strftime('%Y-%m-%d %H:%M')}")
    console.print(f"  Source: {golden.metadata.source_result_file or 'N/A'}")
    if golden.metadata.notes:
        console.print(f"  Notes: {golden.metadata.notes}")
    console.print()

    # Tool sequence
    console.print("[bold]Tool Sequence:[/bold]")
    for i, tool in enumerate(golden.tool_sequence, 1):
        console.print(f"  {i}. {tool}")
    console.print()

    # Output preview
    console.print("[bold]Output Preview:[/bold]")
    preview = golden.trace.final_output[:500]
    if len(golden.trace.final_output) > 500:
        preview += "..."
    console.print(Panel(preview, border_style="dim"))
    console.print()


# ============================================================================
# Snapshot/Check Commands (Simplified Workflow)
# ============================================================================


def _execute_snapshot_tests(
    test_cases: List["TestCase"],
    config: Optional["EvalViewConfig"]
) -> List["EvaluationResult"]:
    """Execute tests and evaluate results for snapshot command.

    Returns:
        List of EvaluationResult objects
    """
    from evalview.evaluators.evaluator import Evaluator

    results = []

    for tc in test_cases:
        try:
            # Get adapter config
            adapter_type = tc.adapter or (config.adapter if config else None)
            endpoint = tc.endpoint or (config.endpoint if config else None)

            if not adapter_type or not endpoint:
                console.print(f"[yellow]⚠ Skipping {tc.name}: No adapter/endpoint configured[/yellow]")
                continue

            # Create adapter
            allow_private = getattr(config, "allow_private_urls", True) if config else True
            try:
                adapter = _create_adapter(adapter_type, endpoint, allow_private_urls=allow_private)
            except ValueError as e:
                console.print(f"[yellow]⚠ Skipping {tc.name}: {e}[/yellow]")
                continue

            # Run test
            try:
                trace = asyncio.run(adapter.execute(tc.input.query, tc.input.context))
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                console.print(f"[red]✗ {tc.name}: Async execution failed - {e}[/red]")
                continue
            except Exception:
                # Re-raise to be caught by outer exception handler
                raise

            # Evaluate
            evaluator = Evaluator()
            result = asyncio.run(evaluator.evaluate(tc, trace))

            results.append(result)

            # Show result
            if result.passed:
                console.print(f"[green]✓ {tc.name}:[/green] {result.score:.1f}/100")
            else:
                console.print(f"[red]✗ {tc.name}:[/red] {result.score:.1f}/100")

        except Exception as e:
            console.print(f"[red]✗ {tc.name}: Failed - {e}[/red]")
            continue

    return results


def _save_snapshot_results(
    results: List["EvaluationResult"],
    notes: Optional[str]
) -> int:
    """Save passing test results as golden baselines.

    Returns:
        Number of tests successfully saved
    """
    from evalview.core.golden import GoldenStore

    store = GoldenStore()

    # Filter to passing results
    passing = [r for r in results if r.passed]

    if not passing:
        console.print("\n[yellow]No passing tests to snapshot.[/yellow]")
        console.print("[dim]Fix failing tests first, then run evalview snapshot again.[/dim]\n")
        return 0

    # Save passing results as golden
    console.print()
    saved_count = 0
    for result in passing:
        try:
            store.save_golden(result, notes=notes)
            console.print(f"[green]✓ Snapshotted:[/green] {result.test_case}")
            saved_count += 1
        except Exception as e:
            console.print(f"[red]❌ Failed to save {result.test_case}: {e}[/red]")

    return saved_count


@main.command("snapshot")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--notes", "-n", help="Notes about this snapshot")
@click.option("--test", "-t", help="Snapshot only this specific test (by name)")
@track_command("snapshot")
def snapshot(test_path: str, notes: str, test: str):
    """Run tests and snapshot passing results as baseline.

    This is the simple workflow: snapshot → check → fix → snapshot.

    TEST_PATH is the directory containing test cases (default: tests/).

    Examples:
        evalview snapshot                    # Snapshot all passing tests
        evalview snapshot --test "my-test"   # Snapshot one test only
        evalview snapshot --notes "v2.0"     # Add notes to snapshot
    """
    from evalview.core.project_state import ProjectStateStore
    from evalview.core.celebrations import Celebrations
    from evalview.core.messages import get_random_checking_message

    # Initialize stores
    state_store = ProjectStateStore()

    # Check if this is the first snapshot ever
    is_first = state_store.is_first_snapshot()

    console.print(f"\n[cyan]▶ {get_random_checking_message()}[/cyan]\n")

    # Load test cases
    loader = TestCaseLoader()
    try:
        test_cases = loader.load_from_directory(Path(test_path))
    except Exception as e:
        console.print(f"[red]❌ Failed to load test cases: {e}[/red]\n")
        Celebrations.no_tests_found()
        return

    if not test_cases:
        Celebrations.no_tests_found()
        return

    # Filter to specific test if requested
    if test:
        test_cases = [tc for tc in test_cases if tc.name == test]
        if not test_cases:
            console.print(f"[red]❌ No test found with name: {test}[/red]\n")
            return

    # Run tests
    console.print(f"[cyan]Running {len(test_cases)} test(s)...[/cyan]\n")

    # Load config
    config = _load_config_if_exists()

    # Execute tests
    results = _execute_snapshot_tests(test_cases, config)

    # Save passing results as golden
    saved_count = _save_snapshot_results(results, notes)

    if saved_count == 0:
        return

    # Update project state
    state_store.update_snapshot(test_count=saved_count)

    # Celebrate!
    if is_first:
        Celebrations.first_snapshot(saved_count)
    else:
        console.print(f"\n[green]Baseline updated: {saved_count} test(s)[/green]")
        console.print("[dim]Run: evalview check[/dim]\n")


def _load_config_if_exists() -> Optional["EvalViewConfig"]:
    """Load config from .evalview/config.yaml if it exists."""
    from evalview.core.config import EvalViewConfig

    config_path = Path(".evalview/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            return EvalViewConfig.model_validate(config_data)
    return None


def _execute_check_tests(
    test_cases: List["TestCase"],
    config: Optional["EvalViewConfig"],
    json_output: bool
) -> Tuple[List[Tuple[str, "TraceDiff"]], List["EvaluationResult"]]:
    """Execute tests and compare against golden variants.

    Returns:
        Tuple of (diffs, results) where diffs is [(test_name, TraceDiff)]
    """
    from evalview.core.golden import GoldenStore
    from evalview.core.diff import DiffEngine
    from evalview.evaluators.evaluator import Evaluator

    store = GoldenStore()
    diff_engine = DiffEngine()
    evaluator = Evaluator()

    results = []
    diffs = []

    for tc in test_cases:
        try:
            # Get adapter config
            adapter_type = tc.adapter or (config.adapter if config else None)
            endpoint = tc.endpoint or (config.endpoint if config else None)

            if not adapter_type or not endpoint:
                continue

            # Create adapter
            allow_private = getattr(config, "allow_private_urls", True) if config else True
            try:
                adapter = _create_adapter(adapter_type, endpoint, allow_private_urls=allow_private)
            except ValueError as e:
                import sys
                print(f"warning: skipping {tc.name}: {e}", file=sys.stderr)
                if not json_output:
                    console.print(f"[yellow]⚠ Skipping {tc.name}: {e}[/yellow]")
                continue

            # Run test (wrap asyncio.run to catch async exceptions)
            try:
                trace = asyncio.run(adapter.execute(tc.input.query, tc.input.context))
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                if not json_output:
                    console.print(f"[red]✗ {tc.name}: Async execution failed - {e}[/red]")
                continue
            except Exception:
                # Re-raise to be caught by outer exception handler
                raise

            # Evaluate
            result = asyncio.run(evaluator.evaluate(tc, trace))
            results.append(result)

            # Compare against golden (use multi-reference)
            golden_variants = store.load_all_golden_variants(tc.name)
            if golden_variants:
                # Use multi-reference comparison for best match
                diff = diff_engine.compare_multi_reference(golden_variants, trace, result.score)
                diffs.append((tc.name, diff))

        except Exception as e:
            if not json_output:
                console.print(f"[red]✗ {tc.name}: Failed - {e}[/red]")
            continue

    return diffs, results


def _analyze_check_diffs(diffs: List[Tuple[str, "TraceDiff"]]) -> Dict[str, Any]:
    """Analyze diffs and return summary statistics.

    Returns:
        Dict with keys: has_regressions, has_tools_changed, has_output_changed, all_passed
    """
    from evalview.core.diff import DiffStatus

    has_regressions = any(d.overall_severity == DiffStatus.REGRESSION for _, d in diffs)
    has_tools_changed = any(d.overall_severity == DiffStatus.TOOLS_CHANGED for _, d in diffs)
    has_output_changed = any(d.overall_severity == DiffStatus.OUTPUT_CHANGED for _, d in diffs)
    all_passed = not has_regressions and not has_tools_changed and not has_output_changed

    return {
        "has_regressions": has_regressions,
        "has_tools_changed": has_tools_changed,
        "has_output_changed": has_output_changed,
        "all_passed": all_passed,
    }


def _display_check_results(
    diffs: List[Tuple[str, "TraceDiff"]],
    analysis: Dict[str, Any],
    state: "ProjectState",
    is_first_check: bool,
    json_output: bool
) -> None:
    """Display check results in JSON or console format."""
    from evalview.core.diff import DiffStatus
    from evalview.core.celebrations import Celebrations
    from evalview.core.messages import get_random_clean_check_message

    if json_output:
        # JSON output for CI
        output = {
            "summary": {
                "total_tests": len(diffs),
                "unchanged": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED),
                "regressions": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION),
                "tools_changed": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED),
                "output_changed": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED),
            },
            "diffs": [
                {
                    "test_name": name,
                    "status": diff.overall_severity.value,
                    "score_delta": diff.score_diff,
                    "has_tool_diffs": len(diff.tool_diffs) > 0,
                    "output_similarity": diff.output_diff.similarity if diff.output_diff else 1.0,
                }
                for name, diff in diffs
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        # Console output with personality
        if is_first_check:
            Celebrations.first_check()

        if analysis["all_passed"]:
            # Clean check!
            console.print(f"[green]{get_random_clean_check_message()}[/green]\n")

            # Show streak celebration
            if state.current_streak >= 3:
                Celebrations.clean_check_streak(state)

            # Show health summary periodically
            if state.total_checks >= 5 and state.total_checks % 5 == 0:
                Celebrations.health_summary(state)
        else:
            # Show diffs
            console.print("\n[bold]Diff Summary[/bold]")
            unchanged = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED)
            console.print(f"  {unchanged}/{len(diffs)} unchanged")
            if analysis["has_regressions"]:
                count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION)
                console.print(f"  {count} {'regression' if count == 1 else 'regressions'}")
            if analysis["has_tools_changed"]:
                count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED)
                console.print(f"  {count} tool {'change' if count == 1 else 'changes'}")
            if analysis["has_output_changed"]:
                count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED)
                console.print(f"  {count} output {'change' if count == 1 else 'changes'}")

            console.print()

            # Show details of changed tests
            for name, diff in diffs:
                if diff.overall_severity != DiffStatus.PASSED:
                    severity_icon = {
                        DiffStatus.REGRESSION: "[red]✗ REGRESSION[/red]",
                        DiffStatus.TOOLS_CHANGED: "[yellow]⚠ TOOLS_CHANGED[/yellow]",
                        DiffStatus.OUTPUT_CHANGED: "[dim]~ OUTPUT_CHANGED[/dim]",
                    }.get(diff.overall_severity, "?")

                    console.print(f"{severity_icon}: {name}")
                    console.print(f"    {diff.summary()}")
                    console.print()

            # Show guidance
            if analysis["has_regressions"]:
                Celebrations.regression_guidance("See details above")


def _compute_check_exit_code(
    diffs: List[Tuple[str, "TraceDiff"]],
    fail_on: Optional[str],
    strict: bool
) -> int:
    """Compute exit code based on diff results and fail conditions.

    Returns:
        0 if no failures match fail conditions, 1 otherwise
    """
    if strict:
        fail_on = "REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED"

    if not fail_on:
        fail_on = "REGRESSION"  # Default

    fail_statuses = set(s.strip().upper() for s in fail_on.split(","))

    for _, diff in diffs:
        if diff.overall_severity.value.upper() in fail_statuses:
            return 1

    return 0


@main.command("check")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--test", "-t", help="Check only this specific test")
@click.option("--json", "json_output", is_flag=True, help="Output JSON for CI")
@click.option("--fail-on", help="Comma-separated statuses to fail on (default: REGRESSION)")
@click.option("--strict", is_flag=True, help="Fail on any change (REGRESSION, TOOLS_CHANGED, OUTPUT_CHANGED)")
@track_command("check")
def check(test_path: str, test: str, json_output: bool, fail_on: str, strict: bool):
    """Check current behavior against snapshot baseline.

    This command runs tests and compares them against your saved baselines,
    showing only what changed. Perfect for CI/CD and daily development.

    TEST_PATH is the directory containing test cases (default: tests/).

    Examples:
        evalview check                                   # Check all tests
        evalview check --test "my-test"                  # Check one test
        evalview check --json                            # JSON output for CI
        evalview check --fail-on REGRESSION,TOOLS_CHANGED
        evalview check --strict                          # Fail on any change
    """
    from evalview.core.golden import GoldenStore
    from evalview.core.project_state import ProjectStateStore
    from evalview.core.celebrations import Celebrations
    from evalview.core.messages import get_random_checking_message

    # Initialize stores
    store = GoldenStore()
    state_store = ProjectStateStore()

    # Check if this is the first check
    is_first_check = state_store.is_first_check()

    # Show recap
    if not is_first_check and not json_output:
        days_since = state_store.days_since_last_check()
        if days_since and days_since >= 7:
            Celebrations.welcome_back(days_since)

    # Verify snapshots exist
    goldens = store.list_golden()
    if not goldens:
        if not json_output:
            Celebrations.no_snapshot_found()
        sys.exit(1)

    # Show status message
    if not json_output:
        console.print(f"[cyan]▶ {get_random_checking_message()}[/cyan]\n")

    # Load test cases
    loader = TestCaseLoader()
    try:
        test_cases = loader.load_from_directory(Path(test_path))
    except Exception as e:
        console.print(f"[red]❌ Failed to load test cases: {e}[/red]\n")
        sys.exit(1)

    # Filter to specific test if requested
    if test:
        test_cases = [tc for tc in test_cases if tc.name == test]
        if not test_cases:
            console.print(f"[red]❌ No test found with name: {test}[/red]\n")
            sys.exit(1)

    # Load config
    config = _load_config_if_exists()

    # Execute tests and compare against golden
    diffs, results = _execute_check_tests(test_cases, config, json_output)

    # Analyze diffs
    analysis = _analyze_check_diffs(diffs)

    # Update project state
    state = state_store.update_check(
        has_regressions=(not analysis["all_passed"]),
        status="passed" if analysis["all_passed"] else "regression"
    )

    # Display results
    _display_check_results(diffs, analysis, state, is_first_check, json_output)

    # Compute and exit with code
    exit_code = _compute_check_exit_code(diffs, fail_on, strict)
    sys.exit(exit_code)


# ============================================================================
# MCP Contract Commands
# ============================================================================


@main.group()
def mcp():
    """Manage MCP contracts (detect external server interface drift).

    MCP contracts are snapshots of an external MCP server's tool definitions.
    Use them with `evalview run --contracts` to detect when servers change
    their interface before your tests break.

    Examples:
        evalview mcp snapshot "npx:@modelcontextprotocol/server-github" --name server-github
        evalview mcp check server-github
        evalview mcp list
    """
    pass


@mcp.command("snapshot")
@click.argument("endpoint")
@click.option("--name", "-n", required=True, help="Server name (used as contract identifier)")
@click.option("--notes", help="Notes about this snapshot")
@click.option("--timeout", default=30.0, type=float, help="Connection timeout in seconds")
def mcp_snapshot(endpoint: str, name: str, notes: str, timeout: float):
    """Snapshot an MCP server's tool definitions as a contract.

    ENDPOINT is the MCP server endpoint (e.g., "npx:@modelcontextprotocol/server-github").

    Examples:
        evalview mcp snapshot "npx:@modelcontextprotocol/server-filesystem /tmp" --name fs-server
        evalview mcp snapshot "http://localhost:8080" --name my-server --notes "v2.1 release"
    """
    import asyncio
    from evalview.adapters.mcp_adapter import MCPAdapter
    from evalview.core.mcp_contract import ContractStore

    console.print(f"\n[cyan]━━━ MCP Contract Snapshot ━━━[/cyan]\n")
    console.print(f"  Server: [bold]{name}[/bold]")
    console.print(f"  Endpoint: {endpoint}")
    console.print()

    adapter = MCPAdapter(endpoint=endpoint, timeout=timeout)

    try:
        tools = asyncio.run(adapter.discover_tools())
    except Exception as e:
        console.print(f"[red]Failed to connect to MCP server: {e}[/red]")
        console.print("[dim]Check that the server is running and the endpoint is correct.[/dim]\n")
        raise SystemExit(1)

    if not tools:
        console.print("[yellow]Server returned no tools.[/yellow]\n")
        raise SystemExit(1)

    store = ContractStore()

    if store.has_contract(name):
        if not click.confirm(
            f"Contract '{name}' already exists. Overwrite?",
            default=False,
        ):
            console.print("[dim]Cancelled[/dim]\n")
            return

    path = store.save_contract(
        server_name=name,
        endpoint=endpoint,
        tools=tools,
        notes=notes,
    )

    console.print(f"[green]Snapshot saved: {path}[/green]")
    console.print(f"  Tools discovered: [bold]{len(tools)}[/bold]")
    for tool in tools:
        desc = tool.get("description", "")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        console.print(f"    [dim]- {tool['name']}[/dim]  {desc}")
    console.print()
    console.print("[dim]Check for drift: evalview mcp check " + name + "[/dim]")
    console.print("[dim]Use in CI: evalview run --contracts --fail-on CONTRACT_DRIFT[/dim]\n")


@mcp.command("check")
@click.argument("name")
@click.option("--endpoint", help="Override endpoint (default: use endpoint from snapshot)")
@click.option("--timeout", default=30.0, type=float, help="Connection timeout in seconds")
def mcp_check(name: str, endpoint: str, timeout: float):
    """Check an MCP server for contract drift.

    NAME is the contract name (from `evalview mcp snapshot --name`).

    Examples:
        evalview mcp check server-github
        evalview mcp check my-server --endpoint "http://new-host:8080"
    """
    import asyncio
    from evalview.adapters.mcp_adapter import MCPAdapter
    from evalview.core.mcp_contract import ContractStore
    from evalview.core.contract_diff import diff_contract, ContractDriftStatus

    store = ContractStore()
    contract = store.load_contract(name)

    if not contract:
        console.print(f"\n[red]No contract found: {name}[/red]")
        console.print("[dim]Create one with: evalview mcp snapshot <endpoint> --name " + name + "[/dim]\n")
        raise SystemExit(1)

    target_endpoint = endpoint or contract.metadata.endpoint
    adapter = MCPAdapter(endpoint=target_endpoint, timeout=timeout)

    console.print(f"\n[cyan]━━━ MCP Contract Check ━━━[/cyan]\n")
    console.print(f"  Contract: [bold]{name}[/bold]")
    console.print(f"  Endpoint: {target_endpoint}")

    # Show snapshot age
    age = datetime.now() - contract.metadata.snapshot_at
    age_days = age.days
    if age_days > 30:
        console.print(f"  Snapshot age: [yellow]{age_days} days (consider refreshing)[/yellow]")
    else:
        console.print(f"  Snapshot age: [dim]{age_days} day(s)[/dim]")
    console.print()

    try:
        current_tools = asyncio.run(adapter.discover_tools())
    except Exception as e:
        console.print(f"[red]Failed to connect to MCP server: {e}[/red]")
        console.print("[dim]The server may be down. Use --endpoint to try a different host.[/dim]\n")
        raise SystemExit(2)

    result = diff_contract(contract, current_tools)

    if result.status == ContractDriftStatus.PASSED:
        if result.informational_changes:
            console.print(f"[green]PASSED[/green] - No breaking changes ({result.summary()})")
            console.print()
            for change in result.informational_changes:
                console.print(f"  [dim]+ {change.tool_name}: {change.detail}[/dim]")
        else:
            console.print("[green]PASSED[/green] - Interface matches snapshot exactly")
        console.print()
    else:
        console.print(f"[red]CONTRACT_DRIFT[/red] - {result.summary()}\n")

        for change in result.breaking_changes:
            if change.kind.value == "removed":
                console.print(f"  [red]REMOVED: {change.tool_name}[/red] - {change.detail}")
            else:
                console.print(f"  [red]CHANGED: {change.tool_name}[/red] - {change.detail}")

        if result.informational_changes:
            console.print()
            for change in result.informational_changes:
                console.print(f"  [dim]INFO: {change.tool_name} - {change.detail}[/dim]")

        console.print()
        console.print("[dim]To accept the new interface: evalview mcp snapshot " + target_endpoint + " --name " + name + "[/dim]\n")
        raise SystemExit(1)


@mcp.command("list")
def mcp_list():
    """List all MCP contract snapshots.

    Shows all saved contracts with metadata.
    """
    from evalview.core.mcp_contract import ContractStore

    store = ContractStore()
    contracts = store.list_contracts()

    if not contracts:
        console.print("\n[yellow]No MCP contracts found.[/yellow]")
        console.print("[dim]Create one: evalview mcp snapshot <endpoint> --name <name>[/dim]\n")
        return

    console.print("\n[cyan]━━━ MCP Contracts ━━━[/cyan]\n")

    for c in sorted(contracts, key=lambda x: x.server_name):
        age = datetime.now() - c.snapshot_at
        age_str = f"{age.days}d ago" if age.days > 0 else "today"

        console.print(f"  [bold]{c.server_name}[/bold]")
        console.print(f"    [dim]Endpoint: {c.endpoint}[/dim]")
        console.print(f"    [dim]Tools: {c.tool_count} | Snapshot: {age_str}[/dim]")
        if c.notes:
            console.print(f"    [dim]Notes: {c.notes}[/dim]")
        console.print()

    console.print(f"[dim]Total: {len(contracts)} contract(s)[/dim]")
    console.print("[dim]Check for drift: evalview mcp check <name>[/dim]\n")


@mcp.command("delete")
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def mcp_delete(name: str, force: bool):
    """Delete an MCP contract snapshot.

    NAME is the contract name to delete.
    """
    from evalview.core.mcp_contract import ContractStore

    store = ContractStore()

    if not store.has_contract(name):
        console.print(f"\n[yellow]No contract found: {name}[/yellow]\n")
        return

    if not force:
        if not click.confirm(f"Delete contract '{name}'?", default=False):
            console.print("[dim]Cancelled[/dim]")
            return

    store.delete_contract(name)
    console.print(f"\n[green]Deleted contract: {name}[/green]\n")


@mcp.command("show")
@click.argument("name")
def mcp_show(name: str):
    """Show details of an MCP contract snapshot.

    NAME is the contract name.
    """
    from evalview.core.mcp_contract import ContractStore
    from rich.panel import Panel

    store = ContractStore()
    contract = store.load_contract(name)

    if not contract:
        console.print(f"\n[yellow]No contract found: {name}[/yellow]")
        console.print("[dim]Create one: evalview mcp snapshot <endpoint> --name " + name + "[/dim]\n")
        return

    meta = contract.metadata
    age = datetime.now() - meta.snapshot_at

    console.print(f"\n[cyan]━━━ MCP Contract: {meta.server_name} ━━━[/cyan]\n")
    console.print(f"  Endpoint: {meta.endpoint}")
    console.print(f"  Snapshot: {meta.snapshot_at.strftime('%Y-%m-%d %H:%M')} ({age.days}d ago)")
    console.print(f"  Protocol: {meta.protocol_version}")
    console.print(f"  Schema hash: {meta.schema_hash}")
    if meta.notes:
        console.print(f"  Notes: {meta.notes}")
    console.print()

    console.print(f"[bold]Tools ({meta.tool_count}):[/bold]\n")

    for tool in contract.tools:
        console.print(f"  [bold]{tool.name}[/bold]")
        if tool.description:
            console.print(f"    {tool.description}")
        if tool.inputSchema.get("properties"):
            props = tool.inputSchema["properties"]
            required = set(tool.inputSchema.get("required", []))
            for pname, pdef in props.items():
                ptype = pdef.get("type", "any")
                req_marker = " [red]*[/red]" if pname in required else ""
                console.print(f"    [dim]- {pname}: {ptype}{req_marker}[/dim]")
        console.print()


@mcp.command("serve")
@click.option("--test-path", default="tests", help="Path to test directory")
def mcp_serve(test_path: str) -> None:
    """Start EvalView as an MCP server for Claude Code.

    Exposes run_check, run_snapshot, and list_tests as MCP tools so you can
    run regression checks inline without switching to a terminal.

    \b
    One-time setup:
        claude mcp add --transport stdio evalview -- evalview mcp serve

    \b
    Verify:
        claude mcp list

    \b
    Then ask Claude: "Did my refactor break the golden baseline?"
    """
    from evalview.mcp_server import MCPServer

    MCPServer(test_path=test_path).serve()


@main.command("inspect")
@click.argument("target", default="latest", required=False)
@click.option("--title", default="EvalView Report", help="Report title")
@click.option("--notes", default="", help="Optional note shown in the report header")
@click.option("--no-open", is_flag=True, help="Do not auto-open in browser")
@click.option("--output", "-o", default=None, help="Output HTML path")
def inspect_cmd(target: str, title: str, notes: str, no_open: bool, output: Optional[str]) -> None:
    """Generate a beautiful visual HTML report and open it in the browser.

    TARGET can be 'latest' (default), a path to a results JSON file, or a
    timestamp string matching a file in .evalview/results/.

    \b
    Examples:
        evalview inspect
        evalview inspect latest
        evalview inspect .evalview/results/20260221_103000.json
        evalview inspect latest --notes "after refactor PR #42"
    """
    import glob as _glob

    from evalview.reporters.json_reporter import JSONReporter
    from evalview.visualization import generate_visual_report

    # Resolve target to a results file
    results_file: Optional[str] = None
    if target == "latest" or not target:
        files = sorted(_glob.glob(".evalview/results/*.json"))
        if not files:
            console.print("\n[red]No results found in .evalview/results/[/red]")
            console.print("[dim]Run [bold]evalview run[/bold] or [bold]evalview snapshot[/bold] first.[/dim]\n")
            raise SystemExit(1)
        results_file = files[-1]
    elif os.path.exists(target):
        results_file = target
    else:
        # Try as partial timestamp match
        matches = sorted(_glob.glob(f".evalview/results/*{target}*.json"))
        if not matches:
            console.print(f"\n[red]No results file found matching: {target}[/red]\n")
            raise SystemExit(1)
        results_file = matches[-1]

    console.print(f"\n[cyan]◈ Generating visual report from {results_file}...[/cyan]")

    try:
        results = JSONReporter.load_as_results(results_file)
    except Exception as exc:
        console.print(f"[red]Failed to load results: {exc}[/red]\n")
        raise SystemExit(1)

    path = generate_visual_report(
        results=results,
        title=title,
        notes=notes,
        output_path=output,
        auto_open=not no_open,
    )

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    rate = round(passed / total * 100) if total else 0

    console.print(f"[green]✓ Report generated:[/green] {path}")
    console.print(f"  {passed}/{total} tests passing ({rate}%)\n")


@main.command("visualize")
@click.argument("target", default="latest", required=False)
@click.option("--compare", "-c", multiple=True, help="Additional result files to compare (use multiple times)")
@click.option("--title", default=None, help="Report title (auto-generated if omitted)")
@click.option("--notes", default="", help="Optional note shown in the report header")
@click.option("--no-open", is_flag=True, help="Do not auto-open in browser (useful in CI)")
@click.option("--output", "-o", default=None, help="Output HTML path")
def visualize_cmd(target: str, compare: tuple, title: Optional[str], notes: str, no_open: bool, output: Optional[str]) -> None:
    """Generate a visual HTML report, optionally comparing multiple runs.

    TARGET can be 'latest' (default), a path to a results JSON file, or a
    partial timestamp string matching a file in .evalview/results/.

    Use --compare to add more runs for side-by-side comparison.

    \b
    Examples:
        evalview visualize
        evalview visualize latest
        evalview visualize latest --compare .evalview/results/20260220_110044.json
        evalview visualize --compare run1.json --compare run2.json --compare run3.json
        evalview visualize latest --notes "after PR #42" --no-open
    """
    import glob as _glob

    from evalview.reporters.json_reporter import JSONReporter
    from evalview.visualization import generate_visual_report

    def _resolve(t: str) -> Optional[str]:
        if t == "latest":
            files = sorted(_glob.glob(".evalview/results/*.json"))
            return files[-1] if files else None
        if os.path.exists(t):
            return t
        matches = sorted(_glob.glob(f".evalview/results/*{t}*.json"))
        return matches[-1] if matches else None

    # Resolve primary target
    primary = _resolve(target)
    if not primary:
        console.print("\n[red]No results found.[/red] Run [bold]evalview run[/bold] first.\n")
        raise SystemExit(1)

    # Resolve comparison targets
    compare_files = []
    for c in compare:
        r = _resolve(c)
        if r:
            compare_files.append(r)
        else:
            console.print(f"[yellow]⚠ Could not find comparison file: {c}[/yellow]")

    all_files = [primary] + compare_files
    is_multi = len(all_files) > 1

    console.print(f"\n[cyan]◈ Generating visual report{'s' if is_multi else ''} from {len(all_files)} run{'s' if is_multi else ''}...[/cyan]")

    try:
        all_results = [JSONReporter.load_as_results(f) for f in all_files]
    except Exception as exc:
        console.print(f"[red]Failed to load results: {exc}[/red]\n")
        raise SystemExit(1)

    # Primary results are first; flatten for single-run view
    results = all_results[0]

    auto_title = title or (
        f"Comparison: {len(all_files)} runs" if is_multi else "EvalView Report"
    )
    auto_notes = notes or (
        " · ".join(os.path.basename(f).replace(".json", "") for f in all_files)
        if is_multi else notes
    )

    path = generate_visual_report(
        results=results,
        compare_results=all_results[1:] if is_multi else None,
        compare_labels=[os.path.basename(f).replace(".json", "") for f in all_files],
        title=auto_title,
        notes=auto_notes,
        output_path=output,
        auto_open=not no_open,
    )

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    rate = round(passed / total * 100) if total else 0

    console.print(f"[green]✓ Report generated:[/green] {path}")
    if is_multi:
        console.print(f"  Comparing {len(all_files)} runs — primary: {passed}/{total} passing ({rate}%)\n")
    else:
        console.print(f"  {passed}/{total} tests passing ({rate}%)\n")


@main.command()
@click.option(
    "--provider",
    type=click.Choice(["ollama", "openai", "anthropic"]),
    default=None,
    help="LLM provider to use (default: auto-detect, prefers Ollama)",
)
@click.option(
    "--model",
    default=None,
    help="Model to use (default: provider's default)",
)
@click.option(
    "--judge-model",
    type=str,
    help="Model for LLM-as-judge (e.g., gpt-5, sonnet, llama-70b, gpt-4o). Aliases auto-resolve to full names.",
)
@click.option(
    "--judge-provider",
    type=click.Choice(["openai", "anthropic", "huggingface", "gemini", "grok", "ollama"]),
    help="Provider for LLM-as-judge evaluation (ollama = free local)",
)
@click.option("--demo_1", is_flag=True, help="Run '3am panic' demo")
@click.option("--demo_2", is_flag=True, help="Run 'instant action' demo")
@click.option("--demo_3", is_flag=True, help="Run 'cost explosion' demo")
@click.option("--demo_chat", is_flag=True, help="Run 'interactive chat' demo")
@track_command("chat", lambda **kw: {"provider": kw.get("provider"), "is_demo": any([kw.get("demo_1"), kw.get("demo_2"), kw.get("demo_3"), kw.get("demo_chat")])})
def chat(provider: str, model: str, judge_model: str, judge_provider: str, demo_1: bool, demo_2: bool, demo_3: bool, demo_chat: bool):
    """Interactive chat interface for EvalView.

    Ask questions about testing your AI agents in natural language.
    The assistant can help you:

    \b
    - Run test cases
    - Generate new test cases
    - Explain test failures
    - Suggest testing strategies

    Examples:

    \b
      evalview chat                    # Auto-detect provider (prefers Ollama)
      evalview chat --provider ollama  # Use Ollama (free, local)
      evalview chat --provider openai  # Use OpenAI
      evalview chat --demo_1           # "3am panic" demo
      evalview chat --demo_2           # "Instant action" demo
      evalview chat --demo_3           # "Cost explosion" demo
      evalview chat --demo_chat        # "Interactive chat" demo

    Type 'exit' or 'quit' to leave the chat.
    """
    from evalview.chat import run_chat, run_demo

    if demo_1:
        asyncio.run(run_demo(provider=provider, model=model, style=1))
    elif demo_2:
        asyncio.run(run_demo(provider=provider, model=model, style=2))
    elif demo_3:
        asyncio.run(run_demo(provider=provider, model=model, style=3))
    elif demo_chat:
        asyncio.run(run_demo(provider=provider, model=model, style=4))
    else:
        asyncio.run(run_chat(provider=provider, model=model, judge_model=judge_model, judge_provider=judge_provider))


@main.command("trace")
@click.option("--output", "-o", type=click.Path(), help="Save trace to file (JSONL format)")
@click.argument("script", type=click.Path(exists=True))
@click.argument("script_args", nargs=-1)
@track_command("trace")
def trace_cmd(output: Optional[str], script: str, script_args: tuple):
    """Trace LLM calls in any Python script.

    Automatically instruments OpenAI, Anthropic, and Ollama SDK calls
    to capture execution traces without code changes.

    \b
    Examples:
        evalview trace my_agent.py
        evalview trace -o trace.jsonl my_agent.py arg1 arg2
        evalview trace scripts/test.py --verbose

    The trace shows:
        - LLM API calls with token counts and costs
        - Call duration and latency
        - Model and provider information
        - Error details if calls fail
    """
    from evalview.trace_cmd import run_traced_command

    # Build command: python <script> [args...]
    cmd = ["python", script]
    cmd.extend(script_args)

    exit_code, trace_file = run_traced_command(
        command=cmd,
        output_path=output,
        console=console,
    )

    sys.exit(exit_code)


# ============================================================================
# traces - Local trace storage commands
# ============================================================================


@main.group()
def traces():
    """Query and manage local trace storage.

    \b
    Examples:
        evalview traces list              # List recent traces
        evalview traces list --last-24h   # Last 24 hours
        evalview traces show abc123       # Show specific trace
        evalview traces export abc123     # Export trace to HTML
        evalview traces cost-report       # Cost report for last 7 days
    """
    pass


@traces.command("list")
@click.option("--last-24h", "last_24h", is_flag=True, help="Show traces from last 24 hours")
@click.option("--last-7d", "last_7d", is_flag=True, help="Show traces from last 7 days")
@click.option("--source", type=click.Choice(["eval", "trace_cmd"]), help="Filter by source")
@click.option("--limit", "-n", default=20, help="Max traces to show (default: 20)")
@track_command("traces_list")
def traces_list(last_24h: bool, last_7d: bool, source: Optional[str], limit: int):
    """List recent traces."""
    from evalview.storage import TraceDB

    try:
        with TraceDB() as db:
            last_hours = 24 if last_24h else None
            last_days = 7 if last_7d else None

            traces_data = db.list_traces(
                last_hours=last_hours,
                last_days=last_days,
                source=source,
                limit=limit,
            )

            if not traces_data:
                console.print("[dim]No traces found.[/dim]")
                console.print("[dim]Run 'evalview trace <script.py>' to capture traces.[/dim]")
                return

            console.print("[bold cyan]━━━ Recent Traces ━━━[/bold cyan]")
            console.print()

            for trace in traces_data:
                # Parse timestamp
                created = trace["created_at"][:16].replace("T", " ")

                # Format cost
                cost = trace.get("total_cost", 0)
                if cost == 0:
                    cost_str = "$0.00"
                elif cost < 0.01:
                    cost_str = f"${cost:.4f}"
                else:
                    cost_str = f"${cost:.2f}"

                # Format source
                src = trace.get("source", "unknown")
                src_icon = "📊" if src == "eval" else "🔍"

                # Script name
                script = trace.get("script_name") or "-"

                console.print(
                    f"[bold]{trace['run_id']}[/bold]  {src_icon} {created}  "
                    f"{trace.get('total_calls', 0)} calls  {cost_str}  [dim]{script}[/dim]"
                )

            console.print()
            console.print(f"[dim]Showing {len(traces_data)} traces. Use --limit to see more.[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@traces.command("show")
@click.argument("trace_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@track_command("traces_show")
def traces_show(trace_id: str, as_json: bool):
    """Show details of a specific trace."""
    import json as json_module
    from evalview.storage import TraceDB

    try:
        with TraceDB() as db:
            trace = db.get_trace(trace_id)

            if not trace:
                console.print(f"[red]Trace not found: {trace_id}[/red]")
                sys.exit(1)

            spans = db.get_trace_spans(trace_id)

            if as_json:
                output = {
                    "trace": trace,
                    "spans": spans,
                }
                console.print(json_module.dumps(output, indent=2, default=str))
                return

            # Pretty print
            console.print("[bold cyan]━━━ Trace Details ━━━[/bold cyan]")
            console.print()

            console.print(f"[bold]Trace ID:[/bold]     {trace['run_id']}")
            console.print(f"[bold]Created:[/bold]      {trace['created_at'][:19].replace('T', ' ')}")
            console.print(f"[bold]Source:[/bold]       {trace.get('source', 'unknown')}")
            if trace.get("script_name"):
                console.print(f"[bold]Script:[/bold]       {trace['script_name']}")
            console.print()

            # Stats
            console.print("[bold]Summary:[/bold]")
            console.print(f"  Total calls:    {trace.get('total_calls', 0)}")
            tokens = trace.get("total_tokens", 0)
            in_tokens = trace.get("total_input_tokens", 0)
            out_tokens = trace.get("total_output_tokens", 0)
            console.print(f"  Total tokens:   {tokens:,} (in: {in_tokens:,} / out: {out_tokens:,})")

            cost = trace.get("total_cost", 0)
            cost_str = f"${cost:.4f}" if cost < 0.01 and cost > 0 else f"${cost:.2f}"
            console.print(f"  Total cost:     {cost_str}")

            latency = trace.get("total_latency_ms", 0)
            if latency < 1000:
                latency_str = f"{latency:.0f}ms"
            else:
                latency_str = f"{latency/1000:.1f}s"
            console.print(f"  Total time:     {latency_str}")
            console.print()

            # Spans
            if spans:
                console.print("[bold]LLM Calls:[/bold]")
                for i, span in enumerate(spans, 1):
                    if span.get("span_type") == "llm":
                        model = span.get("model", "unknown")
                        duration = span.get("duration_ms", 0)
                        dur_str = f"{duration:.0f}ms" if duration < 1000 else f"{duration/1000:.1f}s"
                        span_cost = span.get("cost_usd", 0)
                        span_cost_str = f"${span_cost:.4f}" if span_cost < 0.01 and span_cost > 0 else f"${span_cost:.2f}"
                        status = span.get("status", "success")
                        status_icon = "✓" if status == "success" else "✗"

                        console.print(
                            f"  {i}. {status_icon} {model:<25} {dur_str:>8}  {span_cost_str}"
                        )

            console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@traces.command("cost-report")
@click.option("--last-7d", "last_7d", is_flag=True, default=True, help="Report for last 7 days (default)")
@click.option("--last-30d", "last_30d", is_flag=True, help="Report for last 30 days")
@click.option("--by-model", "by_model", is_flag=True, help="Show breakdown by model")
@track_command("traces_cost_report")
def traces_cost_report(last_7d: bool, last_30d: bool, by_model: bool):
    """Show cost report for recent traces."""
    from evalview.storage import TraceDB

    try:
        with TraceDB() as db:
            days = 30 if last_30d else 7
            report = db.get_cost_report(last_days=days)

            totals = report["totals"]
            total_cost = totals.get("total_cost") or 0
            total_calls = totals.get("total_calls") or 0

            console.print(f"[bold cyan]━━━ Cost Report (Last {days} Days) ━━━[/bold cyan]")
            console.print()

            # Format total cost
            if total_cost == 0:
                cost_str = "$0.00"
            elif total_cost < 0.01:
                cost_str = f"${total_cost:.4f}"
            else:
                cost_str = f"${total_cost:.2f}"

            console.print(f"[bold]Total:[/bold]     {cost_str} across {total_calls:,} LLM calls")
            console.print()

            # By model breakdown
            models = report.get("by_model", [])
            if models:
                console.print("[bold]By Model:[/bold]")
                max_cost = max((m.get("total_cost") or 0) for m in models) if models else 1

                for m in models[:10]:
                    model_name = m.get("model") or "unknown"
                    model_cost = m.get("total_cost") or 0

                    # Calculate percentage
                    pct = (model_cost / total_cost * 100) if total_cost > 0 else 0

                    # Format cost
                    if model_cost == 0:
                        mc_str = "$0.00"
                    elif model_cost < 0.01:
                        mc_str = f"${model_cost:.4f}"
                    else:
                        mc_str = f"${model_cost:.2f}"

                    # Progress bar
                    bar_width = 16
                    filled = int((model_cost / max_cost) * bar_width) if max_cost > 0 else 0
                    bar = "█" * filled + "░" * (bar_width - filled)

                    console.print(f"  {model_name:<22} {mc_str:>8}  ({pct:>4.0f}%)  {bar}")

                console.print()

            # By day breakdown
            days_data = report.get("by_day", [])
            if days_data:
                console.print("[bold]By Day:[/bold]")
                max_day_cost = max((d.get("total_cost") or 0) for d in days_data) if days_data else 1

                for d in days_data[-7:]:  # Show last 7 days max
                    day = d.get("day", "")
                    day_cost = d.get("total_cost") or 0

                    # Format cost
                    if day_cost == 0:
                        dc_str = "$0.00"
                    elif day_cost < 0.01:
                        dc_str = f"${day_cost:.4f}"
                    else:
                        dc_str = f"${day_cost:.2f}"

                    # Progress bar
                    bar_width = 8
                    filled = int((day_cost / max_day_cost) * bar_width) if max_day_cost > 0 else 0
                    bar = "█" * filled + "░" * (bar_width - filled)

                    console.print(f"  {day}  {dc_str:>8}  {bar}")

                console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@traces.command("export")
@click.argument("trace_id")
@click.option("--json", "as_json", is_flag=True, help="Export as JSON instead of HTML")
@click.option("-o", "--output", "output_path", help="Output file path")
@track_command("traces_export")
def traces_export(trace_id: str, as_json: bool, output_path: Optional[str]):
    """Export a trace to HTML or JSON.

    \b
    Examples:
        evalview traces export abc123            # Export to HTML
        evalview traces export abc123 --json    # Export to JSON
        evalview traces export abc123 -o report.html
    """
    import json as json_module
    from evalview.storage import TraceDB

    try:
        with TraceDB() as db:
            trace = db.get_trace(trace_id)

            if not trace:
                console.print(f"[red]Trace not found: {trace_id}[/red]")
                sys.exit(1)

            spans = db.get_trace_spans(trace_id)

            if as_json:
                output = output_path or f"trace_{trace_id}.json"
                data = {"trace": trace, "spans": spans}
                Path(output).write_text(
                    json_module.dumps(data, indent=2, default=str),
                    encoding="utf-8",
                )
                console.print(f"[green]Exported to: {output}[/green]")
            else:
                # HTML export (default)
                try:
                    from evalview.exporters import TraceHTMLExporter
                except ImportError:
                    console.print("[red]HTML export requires jinja2. Install with:[/red]")
                    console.print("  pip install evalview[reports]")
                    sys.exit(1)

                output = output_path or f"trace_{trace_id}.html"
                exporter = TraceHTMLExporter()
                exporter.export(trace, spans, output)
                console.print(f"[green]Exported to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# ============================================================================
# Telemetry Commands
# ============================================================================


@main.group()
def telemetry():
    """Manage anonymous usage telemetry.

    EvalView collects anonymous usage data to improve the tool.
    No personal info, API keys, or test content is collected.

    \b
    Examples:
        evalview telemetry status   # Check current status
        evalview telemetry off      # Disable telemetry
        evalview telemetry on       # Enable telemetry
    """
    pass


@telemetry.command("status")
def telemetry_status():
    """Show current telemetry status."""
    import os

    env_disabled = os.environ.get(TELEMETRY_DISABLED_ENV, "").lower() in ("1", "true", "yes")
    config = load_telemetry_config()

    console.print("\n[cyan]━━━ Telemetry Status ━━━[/cyan]\n")

    if env_disabled:
        console.print("[yellow]Status:[/yellow] [red]Disabled[/red] (via environment variable)")
        console.print(f"[dim]${TELEMETRY_DISABLED_ENV} is set[/dim]")
    elif config.enabled:
        console.print("[yellow]Status:[/yellow] [green]Enabled[/green]")
    else:
        console.print("[yellow]Status:[/yellow] [red]Disabled[/red]")

    console.print(f"[yellow]Install ID:[/yellow] [dim]{config.install_id}[/dim]")
    console.print()
    console.print("[dim]What we collect:[/dim]")
    console.print("  • Command name (run, init, chat, skill_test, etc.)")
    console.print("  • Adapter type (langgraph, crewai, etc.)")
    console.print("  • Test count, pass/fail count")
    console.print("  • OS + Python version")
    console.print("  • CI environment (github_actions, gitlab_ci, or local)")
    console.print("  • Chat session: provider, model name, message count, slash commands used")
    console.print("  • Skill commands: agent type, validation mode")
    console.print("  • Golden trace operations (save, list, delete)")
    console.print()
    console.print("[dim]What we DON'T collect:[/dim]")
    console.print("  • API keys or credentials")
    console.print("  • Test content, queries, or outputs")
    console.print("  • File paths or IP addresses")
    console.print("  • Error messages (only error class name)")
    console.print("  • Chat conversation content")
    console.print()


@telemetry.command("on")
def telemetry_on():
    """Enable anonymous telemetry."""
    import os

    env_disabled = os.environ.get(TELEMETRY_DISABLED_ENV, "").lower() in ("1", "true", "yes")

    if env_disabled:
        console.print(
            f"[yellow]Warning:[/yellow] ${TELEMETRY_DISABLED_ENV} is set. "
            "Unset it to enable telemetry."
        )
        console.print()
        return

    set_telemetry_enabled(True)
    console.print("[green]✓ Telemetry enabled[/green]")
    console.print("[dim]Thank you for helping improve EvalView![/dim]")
    console.print()


@telemetry.command("off")
def telemetry_off():
    """Disable anonymous telemetry."""
    set_telemetry_enabled(False)
    console.print("[green]✓ Telemetry disabled[/green]")
    console.print("[dim]You can re-enable anytime with: evalview telemetry on[/dim]")
    console.print()


# ============================================================================
# CI Commands
# ============================================================================


@main.group()
def ci():
    """CI/CD integration commands.

    Commands for integrating EvalView with CI/CD pipelines.

    \b
    Examples:
        evalview ci comment              # Post results as PR comment
        evalview ci comment --dry-run    # Preview comment without posting
    """
    pass


@ci.command("comment")
@click.option(
    "--results",
    "-r",
    type=click.Path(exists=True),
    help="Path to results JSON file (default: latest in .evalview/results/)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print comment to stdout instead of posting to PR",
)
@click.option(
    "--update/--no-update",
    default=True,
    help="Update existing comment instead of creating new one (default: True)",
)
@track_command("ci_comment", lambda **kw: {"dry_run": kw.get("dry_run")})
def ci_comment(results: Optional[str], dry_run: bool, update: bool):
    """Post test results as a PR comment.

    Automatically detects PR context from GitHub Actions environment.
    Uses the `gh` CLI to post comments (pre-installed in GitHub Actions).

    \b
    Add to your workflow:
        - name: Post PR comment
          if: github.event_name == 'pull_request'
          run: evalview ci comment
          env:
            GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    """
    import json as json_module
    from evalview.ci.comment import (
        load_latest_results,
        generate_pr_comment,
        post_pr_comment,
        update_or_create_comment,
    )

    # Load results
    if results:
        with open(results) as f:
            data = json_module.load(f)
    else:
        data = load_latest_results()

    if not data:
        console.print("[red]No results found.[/red]")
        console.print("[dim]Run 'evalview run' first, or specify --results path.[/dim]")
        sys.exit(1)

    # Handle both list and dict formats
    if type(data).__name__ == "list":
        results_list = data
    elif type(data).__name__ == "dict" and "results" in data:
        results_list = data["results"]
    else:
        results_list = [data]

    # Check for diff results
    diff_results = None
    if type(data).__name__ == "dict" and "diff_results" in data:
        diff_results = data["diff_results"]

    # Get run URL from environment
    run_url = None
    github_server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    github_repo = os.environ.get("GITHUB_REPOSITORY", "")
    github_run_id = os.environ.get("GITHUB_RUN_ID", "")
    if github_repo and github_run_id:
        run_url = f"{github_server}/{github_repo}/actions/runs/{github_run_id}"

    # Generate comment
    comment = generate_pr_comment(results_list, diff_results, run_url)

    if dry_run:
        console.print("[cyan]━━━ PR Comment Preview ━━━[/cyan]\n")
        console.print(comment)
        console.print()
        return

    # Post comment
    if update:
        success = update_or_create_comment(comment)
    else:
        success = post_pr_comment(comment)

    if success:
        console.print("[green]✓ Posted PR comment[/green]")
    else:
        # Not in PR context or gh CLI not available - just print
        console.print("[yellow]Not in PR context or gh CLI not available.[/yellow]")
        console.print("[dim]Comment preview:[/dim]\n")
        console.print(comment)


# ============================================================================
# gym - Practice agent eval patterns
# ============================================================================


@main.command()
@click.option(
    "--suite",
    type=click.Choice(["all", "failure-modes", "security"]),
    default="all",
    help="Which test suite to run (default: all)",
)
@click.option(
    "--endpoint",
    default="http://localhost:2024",
    help="Agent endpoint URL (default: http://localhost:2024)",
)
@click.option(
    "--list-only",
    is_flag=True,
    help="List scenarios without running them",
)
@track_command("gym")
def gym(suite: str, endpoint: str, list_only: bool):
    """Run the EvalView Gym - practice agent eval patterns.

    The Gym provides curated test scenarios for learning how to write
    production-grade agent evals. It includes:

    \b
    • failure-modes: 10 scenarios testing resilience (timeouts, errors, loops)
    • security: 5 scenarios testing injection/jailbreak resistance

    \b
    Quick start:
        1. Start the gym agent:
           cd gym/agents/support-bot && langgraph dev

        2. Run all scenarios:
           evalview gym

    \b
    Examples:
        evalview gym                        # Run all scenarios
        evalview gym --suite failure-modes  # Resilience tests only
        evalview gym --suite security       # Security tests only
        evalview gym --list-only            # List without running
    """
    import glob as glob_module

    console.print("[blue]━━━ EvalView Gym ━━━[/blue]\n")
    console.print("[dim]Practice environment for learning agent eval patterns[/dim]\n")

    # Find gym directory
    gym_paths = [
        Path("gym"),  # From repo root
        Path(__file__).parent.parent / "gym",  # Relative to evalview package
    ]

    gym_dir = None
    for path in gym_paths:
        if path.exists():
            gym_dir = path
            break

    if not gym_dir:
        console.print("[red]Error: gym/ directory not found.[/red]")
        console.print("[dim]Make sure you're in the EvalView repo root or gym is installed.[/dim]")
        sys.exit(1)

    # Collect scenarios based on suite
    scenarios = []

    if suite in ("all", "failure-modes"):
        fm_dir = gym_dir / "failure-modes"
        if fm_dir.exists():
            scenarios.extend(sorted(fm_dir.glob("*.yaml")))

    if suite in ("all", "security"):
        sec_dir = gym_dir / "security"
        if sec_dir.exists():
            scenarios.extend(sorted(sec_dir.glob("*.yaml")))

    if not scenarios:
        console.print(f"[yellow]No scenarios found for suite: {suite}[/yellow]")
        sys.exit(1)

    # List only mode
    if list_only:
        console.print(f"[cyan]Scenarios in suite '{suite}':[/cyan]\n")

        for scenario_path in scenarios:
            try:
                with open(scenario_path) as f:
                    data = yaml.safe_load(f)
                name = data.get("name", scenario_path.stem)
                desc = data.get("description", "").split("\n")[0][:60]
                suite_name = scenario_path.parent.name
                console.print(f"  [{suite_name}] [bold]{name}[/bold]")
                if desc:
                    console.print(f"           [dim]{desc}[/dim]")
            except Exception:
                console.print(f"  [red]{scenario_path.name}[/red] (failed to parse)")

        console.print(f"\n[dim]Total: {len(scenarios)} scenarios[/dim]")
        return

    # Run scenarios
    console.print(f"Running {len(scenarios)} scenarios against {endpoint}\n")

    # Check if endpoint is reachable
    try:
        response = httpx.get(f"{endpoint.rstrip('/')}/health", timeout=5.0)
        console.print("[green]✓ Agent endpoint reachable[/green]\n")
    except Exception:
        console.print("[yellow]⚠ Could not reach agent endpoint[/yellow]")
        console.print(f"[dim]  Make sure your agent is running at {endpoint}[/dim]")
        console.print("[dim]  Start with: cd gym/agents/support-bot && langgraph dev[/dim]\n")

        if not click.confirm("Continue anyway?", default=False):
            sys.exit(1)

    # Run each scenario
    passed = 0
    failed = 0
    errors = 0

    for scenario_path in scenarios:
        try:
            with open(scenario_path) as f:
                data = yaml.safe_load(f)

            name = data.get("name", scenario_path.stem)
            suite_name = scenario_path.parent.name

            # Override endpoint
            data["endpoint"] = endpoint

            console.print(f"[dim][{suite_name}][/dim] {name}... ", end="")

            # Run the test using existing infrastructure
            from evalview.core.runner import TestRunner

            runner = TestRunner()
            result = asyncio.run(runner.run_single_test(data, scenario_path))

            if result.get("passed"):
                console.print("[green]PASS[/green]")
                passed += 1
            else:
                console.print("[red]FAIL[/red]")
                failed += 1
                # Show failure reason
                if result.get("error"):
                    console.print(f"    [dim]{result['error']}[/dim]")

        except Exception as e:
            console.print(f"[red]ERROR[/red]")
            console.print(f"    [dim]{str(e)[:80]}[/dim]")
            errors += 1

    # Summary
    console.print()
    console.print("[cyan]━━━ Summary ━━━[/cyan]")
    total = passed + failed + errors
    console.print(f"  Passed:  [green]{passed}[/green]/{total}")
    console.print(f"  Failed:  [red]{failed}[/red]/{total}")
    if errors:
        console.print(f"  Errors:  [yellow]{errors}[/yellow]/{total}")

    if failed > 0 or errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
