"""3-question setup wizard for `evalview init --wizard`.

Builds a personalized first test case from the user's adapter, agent
description, and tool list. Extracted from init_cmd.py so the main
init flow stays focused on auto-detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import click

from evalview.commands.shared import console
from evalview.core.eval_profiles import (
    detect_agent_type,
    display_profile_recommendation,
    generate_config_yaml,
    get_profile,
)


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


def _init_wizard(dir: str, profile_override: Optional[str] = None) -> None:
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

    # Detect agent profile from wizard answers
    if profile_override:
        profile_key = profile_override
    else:
        profile_key = detect_agent_type(tools=tools, description=description)

    # Display profile recommendation
    console.print()
    display_profile_recommendation(profile_key, tools)

    config_path = base_path / ".evalview" / "config.yaml"
    if not config_path.exists():
        config_content = generate_config_yaml(
            profile_key=profile_key,
            endpoint=endpoint,
            adapter=adapter,
            detected_tools=tools,
        )
        # Append model and timeout settings (not part of the profile template)
        config_content += f"timeout: 30.0\nallow_private_urls: true\n\nmodel:\n  name: {model_name}\n"
        config_path.write_text(config_content)
        console.print("[green]✓ Created .evalview/config.yaml[/green]")
    else:
        console.print("[yellow]⚠  .evalview/config.yaml already exists, skipping[/yellow]")

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
    console.print(f"[dim]Adapter: {adapter}  →  {endpoint}[/dim]")
    profile = get_profile(profile_key)
    console.print(f"[dim]Profile: {profile['icon']} {profile['name']}[/dim]\n")

