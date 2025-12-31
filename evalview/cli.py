"""CLI entry point for EvalView."""

import asyncio
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import click
import httpx
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
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

# Load environment variables (.env is the OSS standard, .env.local for overrides)
load_dotenv()  # Loads .env by default
load_dotenv(dotenv_path=".env.local", override=True)  # Override with .env.local if exists

console = Console()


@click.group()
@click.version_option(version="0.1.7")
def main():
    """EvalView - Catch agent regressions before you ship.

    Detects tool changes, output changes, cost spikes, and latency spikes
    by comparing against golden baselines.

    Quick start:
      evalview quickstart              # Try it in 2 minutes
      evalview run --diff              # Compare against golden baseline
      evalview golden save result.json # Save a working run as baseline
    """
    pass


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
def init(dir: str, interactive: bool, wizard: bool):
    """Initialize EvalView in the current directory."""
    if wizard:
        asyncio.run(_init_wizard_async(dir))
        return

    _init_standard(dir, interactive)


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
                console.print(f"[bold cyan]║[/bold cyan]  [bold green]✓ ALL TESTS PASSED[/bold green]                                            [bold cyan]║[/bold cyan]")
            else:
                console.print(f"[bold cyan]║[/bold cyan]  [bold yellow]⚠ TESTS COMPLETED WITH FAILURES[/bold yellow]                              [bold cyan]║[/bold cyan]")
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
        if passed == len(results):
            console.print("\n[green bold]🎉 All tests passed! Quickstart complete![/green bold]")
        else:
            console.print("\n[yellow]Some tests failed. Check the output above for details.[/yellow]")

        console.print("\n[dim]Note: Cost/tokens shown are mock data from the demo agent.[/dim]")
        console.print("[dim]Your real agent will report actual LLM usage.[/dim]")

        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Connect your agent:")
        console.print("     [cyan]evalview connect[/cyan]  ← Auto-detect running agents")
        console.print("     [dim]or edit .evalview/config.yaml manually[/dim]")
        console.print("  2. Write test cases for your agent's capabilities")
        console.print("  3. Run [cyan]evalview run[/cyan] for detailed results")

        console.print("\n[bold cyan]💡 Pro tip: Scale your tests automatically[/bold cyan]")
        console.print("  [cyan]evalview expand your-test.yaml --count 100[/cyan]  # Generate variations")
        console.print("  [cyan]evalview record --interactive[/cyan]              # Record live sessions")

        # GitHub star CTA
        console.print()
        if passed == len(results):
            console.print("[green]✨ All tests passed![/green] If EvalView saved you time, a star helps others find it:")
        else:
            console.print("[dim]⭐ Like EvalView? Star us on GitHub:[/dim]")
        console.print("   [link=https://github.com/hidai25/eval-view]github.com/hidai25/eval-view[/link]\n")

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
    help="Comma-separated diff statuses that cause exit code 1 (default: REGRESSION, or from ci.fail_on in config.yaml)",
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
        fail_on = "REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED"
        warn_on = ""

    asyncio.run(_run_async(
        path, pattern, test, filter, output, verbose, track, compare_baseline, debug,
        sequential, max_workers, max_retries, retry_delay, watch, html_report, summary, coverage,
        adapter_override=adapter, diff=diff, diff_report=diff_report,
        fail_on=fail_on, warn_on=warn_on
    ))


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

    console.print("[blue]Running test cases...[/blue]\n")

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
        # Note: openai-assistants doesn't need an endpoint (uses SDK directly)
        if test_case.adapter and (test_case.endpoint or test_case.adapter == "openai-assistants"):
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
        else:
            trace = await _execute()

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
                console.print(f"[bold cyan]║[/bold cyan]  [bold green]✓ ALL TESTS PASSED[/bold green]                                            [bold cyan]║[/bold cyan]")
            else:
                console.print(f"[bold cyan]║[/bold cyan]  [bold yellow]⚠ TESTS COMPLETED WITH FAILURES[/bold yellow]                              [bold cyan]║[/bold cyan]")
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
                            console.print(f"      [yellow]~ {td.golden_tool} -> {td.actual_tool}[/yellow]")

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
                console.print(f"[red]✗ {regressions} REGRESSION(s) - score dropped, fix before deploy[/red]\n")
            elif tools_changed > 0:
                console.print(f"[yellow]⚠ {tools_changed} TOOLS_CHANGED - agent behavior shifted, review before deploy[/yellow]\n")
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

    # Tip about quick view modes (only if not in summary/coverage mode)
    if not watch and not summary and not coverage:
        console.print("[dim]💡 Quick views:[/dim]")
        console.print("[dim]   evalview run --summary   (deltas + regressions)[/dim]")
        console.print("[dim]   evalview run --coverage  (behavior coverage)[/dim]\n")

    # Tip about creating test cases
    if not watch and results:
        console.print("[dim]📝 Create your own test case:[/dim]")
        console.print("[dim]   1. Create a YAML file (e.g., my-test.yaml):[/dim]")
        console.print("[dim]   ┌────────────────────────────────────────┐[/dim]")
        console.print("[dim]   │ name: My Test                         │[/dim]")
        console.print("[dim]   │ input:                                │[/dim]")
        console.print("[dim]   │   query: \"Your question here\"        │[/dim]")
        console.print("[dim]   │ expected:                             │[/dim]")
        console.print("[dim]   │   output:                             │[/dim]")
        console.print("[dim]   │     contains: [\"expected\", \"words\"]   │[/dim]")
        console.print("[dim]   └────────────────────────────────────────┘[/dim]")
        console.print("[dim]   2. Run it: evalview run my-test.yaml[/dim]")
        console.print("[dim]   Docs: [link=https://github.com/hidai25/eval-view/blob/main/docs/YAML_SCHEMA.md]docs/YAML_SCHEMA.md[/link][/dim]\n")

    # Tip about test expansion (show after any test run)
    if not watch and results and test_cases:
        # Find a test file to use as example
        if path:
            test_file = f"{path}/test-case.yaml" if not path.endswith('.yaml') else path
        else:
            test_file = "tests/test-cases/your-test.yaml"
        console.print("[dim]🚀 Generate more tests automatically:[/dim]")
        console.print(f"[dim]   evalview expand {test_file} --count 20[/dim]\n")

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
        valid_statuses = {"REGRESSION", "TOOLS_CHANGED", "OUTPUT_CHANGED", "PASSED"}

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

    # GitHub star CTA (only show when not in watch mode)
    if not watch:
        console.print("[dim]━" * 50 + "[/dim]")
        if failed == 0 and passed > 0:
            # All tests passed - stronger CTA
            console.print("[green]✨ All tests passed![/green] If EvalView saved you time, a star helps others find it:")
            console.print("   [link=https://github.com/hidai25/eval-view]github.com/hidai25/eval-view[/link]\n")
        else:
            console.print("[dim]⭐ Enjoying EvalView? Star us on GitHub:[/dim]")
            console.print("[dim]   [link=https://github.com/hidai25/eval-view]https://github.com/hidai25/eval-view[/link][/dim]\n")

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
    else:
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
def list(pattern: str, detailed: bool):
    """List all available test cases."""
    asyncio.run(_list_async(pattern, detailed))


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
@click.option(
    "--endpoint",
    help="Agent endpoint URL to test (optional - will auto-detect common ones)",
)
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
def demo():
    """🎬 See EvalView catch agent regressions (no API keys needed)."""
    import time as time_module

    console.print()
    # EvalView banner
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗██╗   ██╗ █████╗ ██╗    ██╗   ██╗██╗███████╗██╗    ██╗[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔════╝██║   ██║██╔══██╗██║    ██║   ██║██║██╔════╝██║    ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]█████╗  ██║   ██║███████║██║    ██║   ██║██║█████╗  ██║ █╗ ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔══╝  ╚██╗ ██╔╝██╔══██║██║    ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗ ╚████╔╝ ██║  ██║███████╗╚████╔╝ ██║███████╗╚███╔███╔╝[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ [/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]           [bold yellow]🔍 Regression Detection Demo[/bold yellow]                         [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print()
    console.print("[dim]This demo shows how EvalView catches agent regressions.[/dim]")
    console.print("[dim]Using pre-baked traces - no API keys or LLM needed.[/dim]")
    console.print()
    time_module.sleep(1.5)

    # Pre-baked test results simulating golden vs current comparison
    test_results = [
        {
            "name": "auth-flow",
            "status": "PASSED",
            "golden_tools": ["get_user", "validate_token", "return_session"],
            "actual_tools": ["get_user", "validate_token", "return_session"],
            "golden_score": 95,
            "actual_score": 96,
            "detail": None,
        },
        {
            "name": "search-query",
            "status": "TOOLS_CHANGED",
            "golden_tools": ["parse_query", "db_search"],
            "actual_tools": ["parse_query", "web_search", "db_search"],
            "golden_score": 90,
            "actual_score": 88,
            "detail": "+web_search (new)",
        },
        {
            "name": "summarizer",
            "status": "OUTPUT_CHANGED",
            "golden_tools": ["fetch_doc", "summarize"],
            "actual_tools": ["fetch_doc", "summarize"],
            "golden_score": 92,
            "actual_score": 89,
            "detail": "similarity: 72%",
        },
        {
            "name": "data-analyzer",
            "status": "REGRESSION",
            "golden_tools": ["load_data", "analyze", "format_output"],
            "actual_tools": ["load_data", "analyze", "format_output"],
            "golden_score": 85,
            "actual_score": 71,
            "detail": "score: 85 → 71 (-14)",
        },
    ]

    # Cost and latency data
    golden_cost = 0.12
    actual_cost = 0.34
    golden_latency = 1.2
    actual_latency = 3.8

    console.print("[yellow]▶ Comparing current run against golden baseline...[/yellow]")
    console.print()
    time_module.sleep(1)

    # Animate running through tests
    for test in test_results:
        console.print(f"  [dim]Analyzing[/dim] {test['name']}...", end="")
        time_module.sleep(0.4)
        console.print(" [green]done[/green]")

    console.print()
    time_module.sleep(0.5)

    # Display the regression report
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print("[bold cyan]                       Regression Report                           [/bold cyan]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print()

    # Results table with aligned columns
    status_width = 14  # Width for status column
    for test in test_results:
        status = test["status"]
        name = test["name"]
        detail = test["detail"] or ""

        if status == "PASSED":
            icon = "[bold green]✓[/bold green]"
            status_text = "[green]PASSED[/green]"
        elif status == "TOOLS_CHANGED":
            icon = "[bold yellow]⚠[/bold yellow]"
            status_text = "[yellow]TOOLS_CHANGED[/yellow]"
        elif status == "OUTPUT_CHANGED":
            icon = "[bold blue]~[/bold blue]"
            status_text = "[blue]OUTPUT_CHANGED[/blue]"
        else:  # REGRESSION
            icon = "[bold red]✗[/bold red]"
            status_text = "[red]REGRESSION[/red]"

        # Calculate padding (account for rich markup not taking visual space)
        padding = " " * (status_width - len(status))
        detail_text = f"  [dim]{detail}[/dim]" if detail else ""
        console.print(f"  {icon} {status_text}{padding} {name:<18}{detail_text}")
        time_module.sleep(0.3)

    console.print()

    # Cost and latency deltas
    cost_delta = ((actual_cost - golden_cost) / golden_cost) * 100
    latency_delta = ((actual_latency - golden_latency) / golden_latency) * 100

    cost_warning = "[yellow]⚠[/yellow]" if cost_delta > 50 else ""
    latency_warning = "[yellow]⚠[/yellow]" if latency_delta > 50 else ""

    console.print(f"  [bold]Cost:[/bold]    ${golden_cost:.2f} → ${actual_cost:.2f}  [yellow](+{cost_delta:.0f}%)[/yellow]  {cost_warning}")
    console.print(f"  [bold]Latency:[/bold] {golden_latency:.1f}s → {actual_latency:.1f}s   [yellow](+{latency_delta:.0f}%)[/yellow]  {latency_warning}")
    console.print()

    time_module.sleep(0.5)

    # CI verdict
    console.print("[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")
    console.print("[bold red]  ❌ This would fail CI[/bold red]")
    console.print("[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")
    console.print()

    time_module.sleep(0.5)

    # Show the tool diff detail for the TOOLS_CHANGED case
    console.print("[bold cyan]Tool Diff: search-query[/bold cyan]")
    console.print()
    console.print("  [dim]Golden:[/dim]  parse_query → db_search")
    console.print("  [dim]Actual:[/dim]  parse_query → [yellow]web_search[/yellow] → db_search")
    console.print()
    console.print("  [yellow]+ web_search[/yellow]  [dim]<< new tool inserted[/dim]")
    console.print()

    time_module.sleep(0.5)

    # Show the output diff detail
    console.print("[bold cyan]Output Diff: summarizer[/bold cyan]")
    console.print()
    console.print("  [dim]Golden output:[/dim]")
    console.print('    [dim]"The quarterly report shows revenue increased by 15%..."[/dim]')
    console.print()
    console.print("  [dim]Actual output:[/dim]")
    console.print('    [dim]"Revenue was up in Q3. The report indicates growth..."[/dim]')
    console.print()
    console.print("  [blue]Similarity: 72%[/blue]  [dim](threshold: 90%)[/dim]")
    console.print()

    time_module.sleep(1)

    # Final CTA
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold white]Catch regressions before your users do.[/bold white]                      [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [dim]What EvalView caught:[/dim]                                          [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]    • 1 regression (score dropped 14 points)                     [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]    • 1 tool change (new tool added to flow)                     [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]    • 1 output change (response differs from baseline)           [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]    • Cost spike: +183%                                          [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]    • Latency spike: +217%                                       [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [green]Get started:[/green]                                                 [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]    $ evalview quickstart                                        [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print()


@main.command()
@click.argument("pattern", required=False)
@click.option("--tool", help="Tool name to use in the test")
@click.option("--query", help="Query to use in the test")
@click.option("--list", "list_patterns", is_flag=True, help="List available patterns")
@click.option("--output", "-o", help="Output file path (default: tests/<pattern>.yaml)")
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
def skill_validate(path: str, recursive: bool, strict: bool, verbose: bool, output_json: bool):
    """Validate Claude Code skill(s).

    Validates SKILL.md files for:
    - Correct structure and frontmatter
    - Valid naming conventions
    - Policy compliance
    - Best practices

    Examples:
        evalview skill validate ./my-skill/SKILL.md
        evalview skill validate ./skills/ --recursive
        evalview skill validate ./SKILL.md --strict
        evalview skill validate ./skills/ -rv  # verbose with suggestions
    """
    import json
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
    import time
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
def skill_list(path: str, recursive: bool):
    """List all skills in a directory.

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
            console.print(f"      [dim]{result.skill.metadata.description[:60]}...[/dim]" if len(result.skill.metadata.description) > 60 else f"      [dim]{result.skill.metadata.description}[/dim]")
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
def skill_doctor(path: str, recursive: bool):
    """Diagnose skill issues that cause Claude Code problems.

    Checks for common issues:
    - Total description chars exceeding Claude Code's 15k budget
    - Duplicate skill names
    - Invalid skills
    - Multi-line descriptions that break formatters

    Examples:
        evalview skill doctor ~/.claude/skills/
        evalview skill doctor .claude/skills/
        evalview skill doctor ./my-skills/ -r
    """
    import time
    from pathlib import Path as PathLib
    from rich.panel import Panel
    from evalview.skills import SkillParser, SkillValidator

    start_time = time.time()
    CHAR_BUDGET = 15000  # Claude Code's default limit

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
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗██╗   ██╗ █████╗ ██╗    ██╗   ██╗██╗███████╗██╗    ██╗[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔════╝██║   ██║██╔══██╗██║    ██║   ██║██║██╔════╝██║    ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]█████╗  ██║   ██║███████║██║    ██║   ██║██║█████╗  ██║ █╗ ██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]██╔══╝  ╚██╗ ██╔╝██╔══██║██║    ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]███████╗ ╚████╔╝ ██║  ██║███████╗╚████╔╝ ██║███████╗╚███╔███╔╝[/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold green]╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ [/bold green]  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]           [dim]Skill Doctor - Diagnose Claude Code Issues[/dim]           [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print()

    # Character budget check
    budget_pct = (total_desc_chars / CHAR_BUDGET) * 100
    skills_over = max(0, int((total_desc_chars - CHAR_BUDGET) / 500))  # Estimate skills ignored

    if budget_pct > 100:
        console.print(f"[bold red]⚠️  Character Budget: {budget_pct:.0f}% OVER - Claude is ignoring ~{skills_over} of your {len(files)} skills[/bold red]")
    elif budget_pct > 75:
        console.print(f"[bold yellow]⚠️  Character Budget: {budget_pct:.0f}% - approaching limit[/bold yellow]")
    else:
        console.print(f"[bold green]✓ Character Budget: {budget_pct:.0f}% ({total_desc_chars:,} / {CHAR_BUDGET:,} chars)[/bold green]")
    console.print(f"[bold]Total Skills:[/bold]      {len(files)}")
    console.print(f"[bold]Valid:[/bold]             [green]{len(files) - invalid_count}[/green]")
    console.print(f"[bold]Invalid:[/bold]           [red]{invalid_count}[/red]")
    console.print(f"[bold]Duplicates:[/bold]        [{'red' if duplicates else 'green'}]{len(duplicates)}[/{'red' if duplicates else 'green'}]")
    console.print(f"[bold]Multi-line Desc:[/bold]   [{'yellow' if multiline_count else 'green'}]{multiline_count}[/{'yellow' if multiline_count else 'green'}]")
    console.print()

    # Show issues
    has_issues = False

    if budget_pct > 100:
        has_issues = True
        console.print("[bold red]ISSUE: Character budget exceeded[/bold red]")
        console.print("  Claude Code won't see all your skills.")
        console.print("  [dim]Fix: Set SLASH_COMMAND_TOOL_CHAR_BUDGET=30000 or reduce descriptions[/dim]")
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


@skill.command("test")
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="Model to use")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def skill_test(test_file: str, model: str, verbose: bool, output_json: bool):
    """Run behavior tests against a skill.

    TEST_FILE is a YAML file defining test cases for a skill.

    Example test file:
        name: test-code-reviewer
        skill: ./skills/code-reviewer/SKILL.md
        tests:
          - name: detects-sql-injection
            input: "Review: query = f'SELECT * FROM users WHERE id = {id}'"
            expected:
              output_contains: ["SQL injection", "parameterized"]

    Examples:
        evalview skill test tests/code-reviewer.yaml
        evalview skill test tests/my-skill.yaml --model claude-sonnet-4-20250514
        evalview skill test tests/my-skill.yaml --json
    """
    import json
    import os
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

    from rich.table import Table
    from rich.panel import Panel

    # EvalView banner
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
    console.print(f"  [bold]Suite:[/bold]  {suite.name}")
    console.print(f"  [bold]Skill:[/bold]  [cyan]{suite.skill}[/cyan]")
    console.print(f"  [bold]Model:[/bold]  {model}")
    console.print(f"  [bold]Tests:[/bold]  {len(suite.tests)}")
    console.print()

    # Run the suite with live elapsed timer
    import time
    import threading
    from rich.live import Live

    start_time = time.time()
    result = None
    run_error = None

    def format_elapsed():
        elapsed = time.time() - start_time
        mins, secs = divmod(elapsed, 60)
        secs_int = int(secs)
        ms = int((secs - secs_int) * 1000)
        return f"{int(mins):02d}:{secs_int:02d}.{ms:03d}"

    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_idx = [0]

    def get_display():
        spinner = spinner_frames[spinner_idx[0] % len(spinner_frames)]
        spinner_idx[0] += 1
        return f"{spinner} Running tests... [yellow]{format_elapsed()}[/yellow]"

    def run_tests():
        nonlocal result, run_error
        try:
            result = runner.run_suite(suite)
        except Exception as e:
            run_error = e

    # Start test runner in background thread
    test_thread = threading.Thread(target=run_tests)
    test_thread.start()

    # Show live timer while tests run
    with Live(get_display(), console=console, refresh_per_second=10) as live:
        while test_thread.is_alive():
            live.update(get_display())
            time.sleep(0.1)

    test_thread.join()

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
                    "output": r.output[:500] + "..." if len(r.output) > 500 else r.output,
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
        score_color = "green" if r.score >= 80 else "yellow" if r.score >= 60 else "red"
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
            query = r.input_query[:200] + "..." if len(r.input_query) > 200 else r.input_query
            for line in query.split('\n'):
                console.print(f"  [dim]{line}[/dim]")

            # Show response preview
            if verbose or not r.passed:
                console.print("\n[bold]Response:[/bold]")
                output = r.output[:400] + "..." if len(r.output) > 400 else r.output
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
def golden_save(result_file: str, notes: str, test: str):
    """Save a test result as the golden baseline.

    RESULT_FILE is a JSON file from `evalview run` (e.g., .evalview/results/xxx.json)

    Examples:
        evalview golden save .evalview/results/latest.json
        evalview golden save results.json --notes "v1.0 release baseline"
        evalview golden save results.json --test "List Directory Contents"
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

            # Check if golden already exists
            if store.has_golden(result.test_case):
                if not click.confirm(
                    f"Golden trace already exists for '{result.test_case}'. Overwrite?",
                    default=False,
                ):
                    console.print(f"[yellow]Skipped: {result.test_case}[/yellow]")
                    continue

            path = store.save_golden(result, notes=notes, source_file=result_file)
            console.print(f"[green]✓ Saved golden:[/green] {result.test_case}")
            console.print(f"  [dim]Score: {result.score:.1f}[/dim]")
            console.print(f"  [dim]Tools: {len(result.trace.steps)} steps[/dim]")
            console.print(f"  [dim]File: {path}[/dim]")
            console.print()

        except Exception as e:
            console.print(f"[red]❌ Failed to save: {e}[/red]")

    console.print("[green]Done![/green]\n")


@golden.command("list")
def golden_list():
    """List all golden traces.

    Shows all saved golden baselines with metadata.
    """
    from evalview.core.golden import GoldenStore

    store = GoldenStore()
    goldens = store.list_golden()

    if not goldens:
        console.print("\n[yellow]No golden traces found.[/yellow]")
        console.print("[dim]Save one with: evalview golden save <result.json>[/dim]\n")
        return

    console.print("\n[cyan]━━━ Golden Traces ━━━[/cyan]\n")

    for g in sorted(goldens, key=lambda x: x.test_name):
        console.print(f"  [bold]{g.test_name}[/bold]")
        console.print(f"    [dim]Score: {g.score:.1f}[/dim]")
        console.print(f"    [dim]Blessed: {g.blessed_at.strftime('%Y-%m-%d %H:%M')}[/dim]")
        if g.notes:
            console.print(f"    [dim]Notes: {g.notes}[/dim]")
        console.print()

    console.print(f"[dim]Total: {len(goldens)} golden trace(s)[/dim]\n")


@golden.command("delete")
@click.argument("test_name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
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


if __name__ == "__main__":
    main()
