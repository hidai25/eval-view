"""Init and quickstart commands."""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import httpx

from evalview.commands.shared import console, _detect_agent_endpoint
from evalview.telemetry.decorators import track_command


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_ci_workflow(dir: str) -> None:
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


def _autogen_tests(endpoint: str, tests_dir: Path) -> int:
    """Probe the agent and generate test YAML files from real responses."""
    _FRAGMENT_ENDINGS = (
        " for", " the", " a", " an", " of", " in", " on", " to",
        " with", " and", " or", " e.g.", "(e.g.",
    )

    def _extract_example_queries(text: str) -> List[str]:
        quoted = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]{20,80})["\u201c\u201d]', text)
        bulleted = re.findall(r'[-•]\s+"?([A-Z][^"\n]{20,80})"?\s*$', text, re.MULTILINE)
        candidates = quoted + bulleted
        valid = []
        for q in candidates:
            q = q.strip().rstrip(",.")
            words = q.split()
            if len(words) < 3:
                continue
            if q.lower().endswith(_FRAGMENT_ENDINGS):
                continue
            if "?" in q or q[0].isupper():
                valid.append(q)
        return valid[:3]

    def _stable_phrases(text: str) -> List[str]:
        lines = [ln.strip().lstrip("#*• ") for ln in text.splitlines() if ln.strip()]
        phrases = []
        for line in lines[:5]:
            clean = re.sub(r"\*+|`", "", line).strip()
            if "http" in clean or len(clean) < 4:
                continue
            fragment = clean[:40].strip()
            if fragment:
                phrases.append(fragment)
                break
        return phrases

    def _probe(query: str) -> Optional[Dict[str, Any]]:
        try:
            r = httpx.post(endpoint, json={"query": query}, timeout=30.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _write_test(name: str, query: str, data: Dict[str, Any]) -> bool:
        query = query.strip()
        if len(query) < 10 or len(query.split()) < 3 or query.lower().endswith(_FRAGMENT_ENDINGS):
            return False

        output = data.get("output", "")
        tool_calls = data.get("tool_calls", [])
        tools = [tc["name"] for tc in tool_calls if isinstance(tc, dict)]
        phrases = _stable_phrases(output)

        tools_yaml = ""
        if tools:
            tools_list = "\n".join(f"    - {t}" for t in tools)
            tools_yaml = f"  tools:\n{tools_list}\n"

        contains_yaml = ""
        if phrases:
            contains_list = "\n".join(f'      - "{p}"' for p in phrases)
            contains_yaml = f"    contains:\n{contains_list}\n"

        content = f"""name: "{name}"
description: "Auto-generated from real agent response"
generated: true

endpoint: {endpoint}
adapter: http

input:
  query: "{query}"

expected:
{tools_yaml}  output:
{contains_yaml}    not_contains:
      - "error"
      - "Error"

thresholds:
  min_score: 60
  max_latency: 30000
"""
        safe_name = re.sub(r"[^a-z0-9-]", "-", name.lower())[:40]
        path = tests_dir / f"{safe_name}.yaml"
        path.write_text(content)
        return True

    tests_dir.mkdir(parents=True, exist_ok=True)

    console.print("[dim]  Sending capability probe...[/dim]")
    cap_data = _probe("Hello, what can you help me with?")
    if not cap_data:
        return 0

    generated = 0

    if _write_test("what-can-you-do", "Hello, what can you help me with?", cap_data):
        generated += 1

    examples = _extract_example_queries(cap_data.get("output", ""))
    for i, query in enumerate(examples[:2]):
        console.print(f"[dim]  Probing: {query[:50]}...[/dim]")
        data = _probe(query)
        if data:
            name = f"test-{i + 2}"
            if _write_test(name, query, data):
                generated += 1

    return generated


def _write_blank_template(tests_dir: Path, endpoint: str) -> None:
    """Write a minimal blank test template when auto-gen is not possible."""
    tests_dir.mkdir(parents=True, exist_ok=True)
    path = tests_dir / "my-first-test.yaml"
    if not path.exists():
        path.write_text(f"""name: "my-first-test"
description: "Test that my agent responds correctly"

endpoint: {endpoint}
adapter: http

input:
  query: "Hello, what can you help me with?"

expected:
  output:
    # contains:
    #   - "phrase your agent always says"  # uncomment and fill in
    not_contains:
      - "error"

thresholds:
  min_score: 70
  max_latency: 10000
""")
        console.print("[green]✅ Created tests/test-cases/my-first-test.yaml[/green]")
        console.print("[dim]   Edit the query to match what your agent actually does[/dim]")


def _detect_model() -> Optional[str]:
    """Infer model from environment variables."""
    env = os.environ

    if env.get("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-6"
    if env.get("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        return "gemini-2.0-flash"

    return None


def _create_demo_agent(base_path: Path) -> None:
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
    time.sleep(0.015)

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
        return f"The weather in {city.title()} is {temp_c}C ({temp_f:.1f}F), {weather[\'condition\']}", tool_calls, 0.003

    elif "weather" in query_lower:
        city = "tokyo"
        for c in ["tokyo", "london", "new york", "paris", "sydney"]:
            if c in query_lower:
                city = c
                break
        weather = get_weather(city)
        tool_calls.append(ToolCall(name="get_weather", arguments={"city": city}, result=weather, cost=0.001))
        return f"The weather in {city.title()} is {weather[\'temp\']}C, {weather[\'condition\']} with {weather[\'humidity\']}% humidity", tool_calls, 0.001

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


def _build_wizard_yaml(description: str, tools: List[str]) -> str:
    """Generate a personalized first test case YAML from wizard answers."""
    desc = description.strip()
    desc_lower = desc.lower()

    if any(kw in desc_lower for kw in ["support", "customer", "ticket", "order", "help desk"]):
        query = "I placed an order last week and haven't received a shipping update. Can you help?"
        contains = ["order", "help"]
    elif any(kw in desc_lower for kw in ["code", "review", "pr", "pull request", "github", "refactor"]):
        query = "Please review this function for bugs and suggest improvements: def add(a, b): return a + b"
        contains = ["function", "code"]
    elif any(kw in desc_lower for kw in ["data", "analys", "sql", "report", "dashboard", "metric"]):
        query = "What were the top 5 products by revenue last month?"
        contains = ["result", "data"]
    elif any(kw in desc_lower for kw in ["search", "research", "find", "look up", "lookup", "web"]):
        query = "Find recent information about the impact of AI on software development productivity."
        contains = ["found", "information"]
    elif any(kw in desc_lower for kw in ["schedule", "calendar", "book", "meeting", "appointment"]):
        query = "Can you schedule a 1-hour meeting with the engineering team for next Tuesday at 2pm?"
        contains = ["meeting", "scheduled"]
    elif any(kw in desc_lower for kw in ["email", "draft", "write", "compose", "message"]):
        query = "Draft a professional follow-up email to a client who missed our last meeting."
        contains = ["email", "follow"]
    elif any(kw in desc_lower for kw in ["summariz", "summary", "document", "read", "extract"]):
        query = "Summarize the key points from the quarterly business review document."
        contains = ["summary", "key"]
    else:
        stopwords = {"a", "an", "the", "for", "that", "with", "and", "or", "is", "are",
                     "of", "in", "to", "my", "your", "our", "agent", "bot", "assistant"}
        meaningful = [w.lower().strip(".,;:") for w in desc.split()
                      if w.lower().strip(".,;:") not in stopwords and len(w) > 3]
        subject = meaningful[0] if meaningful else "task"
        query = f"Help me with a typical {subject} request."
        contains = meaningful[:2] if meaningful else ["response"]

    name = desc[:60].strip()
    if name and not name[0].isupper():
        name = name[0].upper() + name[1:]

    lines = [
        f'name: "{name}"',
        f'description: "Verify the agent handles a typical {desc.lower()} request correctly"',
        "",
        "input:",
        f'  query: "{query}"',
        "",
        "expected:",
    ]

    if tools:
        lines.append("  # `tools:` checks that each tool was called (any order).")
        lines.append("  # Change to `tool_sequence:` if call order matters for your agent.")
        lines.append("  tools:")
        for t in tools:
            lines.append(f"    - {t}")

    lines += [
        "  output:",
        "    contains:",
    ]
    for kw in contains:
        lines.append(f'      - "{kw}"')
    lines += [
        "    not_contains:",
        '      - "error"',
        "",
        "thresholds:",
        "  min_score: 75",
        "  max_cost: 0.10",
        "  max_latency: 15000",
        "",
        "checks:",
        "  hallucination: true",
        "  safety: true",
    ]

    return "\n".join(lines) + "\n"


def _init_wizard(dir: str) -> None:
    """3-question wizard that generates one personalized, immediately-runnable test case."""
    console.print("[blue]━━━ EvalView Setup Wizard ━━━[/blue]\n")
    console.print("3 questions. One working test case. Let's go.\n")

    base_path = Path(dir)
    (base_path / ".evalview").mkdir(exist_ok=True)
    (base_path / "tests" / "test-cases").mkdir(parents=True, exist_ok=True)

    console.print("[bold]Step 1/3 — Framework[/bold]")
    console.print("What adapter does your agent use?\n")

    adapter_options = [
        ("http",        "HTTP / REST API    (most common)"),
        ("anthropic",   "Anthropic API      (direct Claude calls)"),
        ("openai",      "OpenAI API         (direct GPT calls)"),
        ("mistral",     "Mistral API        (direct Mistral calls)"),
        ("langgraph",   "LangGraph"),
        ("crewai",      "CrewAI"),
        ("ollama",      "Ollama             (local models)"),
        ("huggingface", "HuggingFace"),
    ]
    for i, (_, label) in enumerate(adapter_options, 1):
        console.print(f"  {i}. {label}")
    console.print(f"  {len(adapter_options) + 1}. Other (enter name)")

    choice = click.prompt("\nChoice", type=int, default=1)
    if 1 <= choice <= len(adapter_options):
        adapter = adapter_options[choice - 1][0]
    else:
        adapter = click.prompt("Adapter name")

    console.print("\n[bold]Step 2/3 — What does your agent do?[/bold]")
    console.print('[dim]Example: "customer support bot that handles order inquiries"[/dim]')
    description = click.prompt("Describe your agent", default="general-purpose assistant")

    console.print("\n[bold]Step 3/3 — Tools[/bold]")
    console.print("[dim]List the tools your agent exposes, comma-separated. Leave blank if none.[/dim]")
    console.print('[dim]Example: "lookup_order, create_ticket, send_email"[/dim]')
    tools_raw = click.prompt("Tools", default="")
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()]

    console.print()
    default_endpoint = "http://localhost:8000/api/agent"
    if adapter == "langgraph":
        default_endpoint = "http://localhost:2024"
    elif adapter == "crewai":
        default_endpoint = "http://localhost:8000/crew"
    endpoint = click.prompt("Agent endpoint URL", default=default_endpoint)
    model_name = click.prompt("Model name", default="gpt-4o")

    config_path = base_path / ".evalview" / "config.yaml"
    if not config_path.exists():
        config_content = f"""# EvalView Configuration
adapter: {adapter}
endpoint: {endpoint}
timeout: 30.0
allow_private_urls: true

model:
  name: {model_name}
"""
        config_path.write_text(config_content)
        console.print("\n[green]✓ Created .evalview/config.yaml[/green]")
    else:
        console.print("\n[yellow]⚠  .evalview/config.yaml already exists, skipping[/yellow]")

    test_path = base_path / "tests" / "test-cases" / "first-test.yaml"
    if not test_path.exists():
        test_path.write_text(_build_wizard_yaml(description, tools))
        console.print("[green]✓ Created tests/test-cases/first-test.yaml[/green]")
    else:
        console.print("[yellow]⚠  tests/test-cases/first-test.yaml already exists, skipping[/yellow]")

    console.print("\n[blue]━━━ Ready ━━━[/blue]")
    console.print("\n[bold]Run your first test:[/bold]")
    console.print("  [cyan]evalview run[/cyan]")
    console.print("\n[dim]Edit tests/test-cases/first-test.yaml to refine expected behaviour.[/dim]")
    console.print(f"[dim]Adapter: {adapter}  →  {endpoint}[/dim]\n")


def _init_standard(dir: str, interactive: bool) -> None:
    """Standard init flow — auto-detects agent and model, asks only when needed."""
    from rich.panel import Panel

    console.print("[blue]━━━ EvalView Setup ━━━[/blue]\n")

    base_path = Path(dir)

    (base_path / ".evalview").mkdir(exist_ok=True)
    (base_path / "tests" / "test-cases").mkdir(parents=True, exist_ok=True)

    console.print("[dim]Scanning for running agents...[/dim]")
    detected_endpoint = _detect_agent_endpoint()
    detected_model = _detect_model()

    adapter_type = "http"
    timeout = 30.0

    if detected_endpoint:
        console.print(f"[green]✓ Found agent at {detected_endpoint}[/green]")
        endpoint = detected_endpoint
    else:
        console.print("[yellow]  No agent found running locally.[/yellow]")
        endpoint = click.prompt("  Agent URL", default="http://localhost:8000")

    if detected_model:
        console.print(f"[green]✓ Detected model: {detected_model}[/green]")
        model_name = detected_model
    else:
        console.print("[yellow]  Could not detect model from environment.[/yellow]")
        model_name = click.prompt(
            "  Model name (e.g. claude-sonnet-4-6, gpt-4o-mini)",
            default="gpt-4o-mini",
        )

    console.print("[dim]  Change these anytime in .evalview/config.yaml[/dim]\n")

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
  # Uses standard OpenAI pricing
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

    tests_dir = base_path / "tests" / "test-cases"

    if detected_endpoint:
        console.print("\n[bold]How would you like to create your first tests?[/bold]\n")
        console.print(
            "  [bold green]1. Capture real interactions[/bold green] [dim](recommended)[/dim]\n"
            "     Use your agent normally — every query becomes a test automatically.\n"
            f"     [cyan]evalview capture --agent {endpoint}[/cyan]\n"
        )
        console.print(
            "  [bold]2. Auto-probe agent[/bold]\n"
            "     EvalView sends example queries now and saves the responses.\n"
        )
        console.print(
            "  [bold]3. Blank template[/bold]\n"
            "     Start from a hand-written YAML — full control, zero magic.\n"
        )
        path_choice = click.prompt("Choice", type=click.IntRange(1, 3), default=1)
    else:
        path_choice = 3

    if path_choice == 1:
        _write_blank_template(tests_dir, endpoint)
        console.print(
            f"\n[green]✅ Ready![/green] "
            f"Start capturing real traffic with:\n"
            f"\n  [cyan]evalview capture --agent {endpoint}[/cyan]\n"
            f"\n[dim]The proxy starts on localhost:8091. Point your client there instead\n"
            f"of {endpoint} and use your agent normally.\n"
            f"Tests are saved to tests/test-cases/ automatically.[/dim]"
        )
    elif path_choice == 2:
        console.print("\n[cyan]Generating test cases from your agent...[/cyan]")
        n = _autogen_tests(endpoint, tests_dir)
        if n > 0:
            console.print(f"[green]✅ Generated {n} test case(s) in tests/test-cases/[/green]")
            console.print("[dim]   Review and edit them, then run evalview snapshot[/dim]")
        else:
            console.print("[yellow]⚠️  Could not reach agent to auto-generate tests.[/yellow]")
            console.print("[dim]   Creating blank template instead.[/dim]")
            _write_blank_template(tests_dir, endpoint)
    else:
        _write_blank_template(tests_dir, endpoint)

    demo_agent_dir = base_path / "demo-agent"
    if not demo_agent_dir.exists():
        demo_agent_dir.mkdir(exist_ok=True)
        _create_demo_agent(base_path)
        console.print("[green]✅ Created demo-agent/ with working example agent[/green]")
    else:
        console.print("[yellow]⚠️  demo-agent/ already exists[/yellow]")

    from evalview.cloud.auth import CloudAuth
    logged_in = CloudAuth().is_logged_in()

    if detected_endpoint:
        step1 = f"[bold]✓[/bold] Agent detected at [cyan]{detected_endpoint}[/cyan]"
    else:
        step1 = "[bold]1.[/bold] Start your agent, then run [cyan]evalview init[/cyan] again"

    if detected_model:
        step2 = f"[bold]✓[/bold] Model detected: [cyan]{detected_model}[/cyan]"
    else:
        step2 = "[bold]2.[/bold] Set an API key\n   [cyan]export ANTHROPIC_API_KEY='sk-...'[/cyan]"

    snapshot_suffix = "   [dim]← syncs to cloud[/dim]" if logged_in else ""

    if detected_endpoint and path_choice == 1:
        step3 = (
            f"[bold]→[/bold] Generate tests from real traffic\n"
            f"   [cyan]evalview capture --agent {endpoint}[/cyan]\n"
            f"   [dim]Point your client to localhost:8091 and use your agent normally[/dim]"
        )
        step4 = (
            f"[bold]→[/bold] Save as your regression baseline\n"
            f"   [cyan]evalview snapshot[/cyan]{snapshot_suffix}"
        )
        step5 = "[bold]→[/bold] Check for regressions anytime\n   [cyan]evalview check[/cyan]"
        body = f"{step1}\n{step2}\n\n{step3}\n\n{step4}\n\n{step5}"
    else:
        step3 = f"[bold]→[/bold] Capture a baseline\n   [cyan]evalview snapshot[/cyan]{snapshot_suffix}"
        step4 = "[bold]→[/bold] Check for regressions anytime\n   [cyan]evalview check[/cyan]"
        body = (
            f"{step1}\n{step2}\n\n{step3}\n\n{step4}\n\n"
            f"[dim]Edit tests/test-cases/my-first-test.yaml to match your agent's queries[/dim]"
        )

    console.print(Panel(body, title="You're set up", border_style="green"))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@click.command("init")
@click.option("--dir", default=".", help="Directory to initialize (default: current directory)")
@click.option("--interactive/--no-interactive", default=True, help="Interactive setup (default: True)")
@click.option("--wizard", is_flag=True, help="Run 3-question wizard to generate a personalized first test case")
@click.option("--ci", is_flag=True, help="Generate a GitHub Actions workflow for running EvalView in CI.")
@track_command("init", lambda **kw: {"ci": kw.get("ci", False)})
def init(dir: str, interactive: bool, wizard: bool, ci: bool):
    """Initialize EvalView in the current directory."""
    if ci:
        _init_ci_workflow(dir)
        return

    if wizard:
        _init_wizard(dir)
        return

    _init_standard(dir, interactive)


@click.command("quickstart")
@track_command("quickstart")
def quickstart():
    """🚀 Quick start: Set up and run a demo in under 2 minutes."""
    import subprocess
    import atexit
    import time as time_module
    import urllib.request

    from rich.live import Live
    from rich.panel import Panel

    console.print("[blue]━━━ EvalView Quickstart ━━━[/blue]\n")
    console.print("This will set up a working demo in under 2 minutes.\n")

    base_path = Path(".")

    demo_agent_dir = base_path / "demo-agent"
    if not demo_agent_dir.exists():
        console.print("[bold]Step 1/4:[/bold] Creating demo agent...")
        _create_demo_agent(base_path)
        console.print("[green]✅ Demo agent created[/green]\n")
    else:
        console.print("[bold]Step 1/4:[/bold] Demo agent already exists\n")

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

    console.print("[bold]Step 4/4:[/bold] Starting demo agent and running test...\n")

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        console.print("[yellow]Installing demo agent dependencies...[/yellow]")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"],
            capture_output=True, check=True
        )
        console.print("[green]✅ Dependencies installed[/green]\n")

    console.print("[dim]Starting demo agent on http://localhost:8000...[/dim]")
    agent_process = subprocess.Popen(
        [sys.executable, str(demo_agent_dir / "agent.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def cleanup():
        agent_process.terminate()
        try:
            agent_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            agent_process.kill()

    atexit.register(cleanup)

    console.print("[dim]Waiting for agent to be ready...[/dim]")
    for _ in range(10):
        time_module.sleep(0.5)
        try:
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

    console.print("[bold]Running tests...[/bold]\n")
    try:
        from evalview.core.loader import TestCaseLoader
        from evalview.adapters.http_adapter import HTTPAdapter
        from evalview.evaluators.evaluator import Evaluator
        from evalview.reporters.console_reporter import ConsoleReporter

        test_cases = TestCaseLoader.load_from_directory(test_dir)
        adapter = HTTPAdapter(
            endpoint="http://localhost:8000/execute",
            headers={},
            timeout=30.0,
            allow_private_urls=True,
        )
        evaluator = Evaluator(skip_llm_judge=use_deterministic_scoring)

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
            test_display = (
                f"  [yellow]{spinner}[/yellow] [dim]{current_test}...[/dim]"
                if current_test
                else f"  [yellow]{spinner}[/yellow] [dim]Starting...[/dim]"
            )

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

        score_suffix = "*" if use_deterministic_scoring else ""

        async def run_all_tests():
            nonlocal passed, failed, tests_completed, current_test
            results = []
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

        if sys.stdin.isatty():
            with Live(get_status_display(), console=console, refresh_per_second=10) as live:
                async def run_with_display():
                    task = asyncio.create_task(run_all_tests())
                    while not task.done():
                        live.update(get_status_display())
                        await asyncio.sleep(0.1)
                    return await task

                results = asyncio.run(run_with_display())

            final_elapsed = format_elapsed()
            console.print()
            console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
            console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
            if failed == 0:
                console.print("[bold cyan]║[/bold cyan]  [bold green]✓ AGENT HEALTHY[/bold green]                                               [bold cyan]║[/bold cyan]")
            else:
                console.print("[bold cyan]║[/bold cyan]  [bold red]✗ REGRESSION DETECTED[/bold red]                                        [bold cyan]║[/bold cyan]")
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
