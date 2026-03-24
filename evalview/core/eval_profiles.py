"""Recommended evaluation profiles based on agent type detection."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Profile definitions — each maps an agent archetype to recommended config
EVAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "chat": {
        "name": "Chat Agent",
        "description": "Conversational agent without tool use",
        "icon": "💬",
        "recommended_checks": {
            "output_quality": True,
            "hallucination": True,
            "safety": True,
            "pii_detection": True,
            "tool_accuracy": False,
            "sequence_check": False,
        },
        "thresholds": {
            "min_score": 65,
            "max_latency": 10000,
        },
        "tips": [
            "Focus on output quality and hallucination detection",
            "Add `contains:` with key phrases your agent should always include",
            "Use `not_contains:` to catch off-topic or harmful responses",
        ],
    },
    "tool-use": {
        "name": "Tool-Use Agent",
        "description": "Agent that calls 1-3 tools per query",
        "icon": "🔧",
        "recommended_checks": {
            "output_quality": True,
            "hallucination": True,
            "safety": False,
            "pii_detection": False,
            "tool_accuracy": True,
            "sequence_check": False,
        },
        "thresholds": {
            "min_score": 70,
            "max_latency": 15000,
        },
        "yaml_additions": {
            "expected": {
                "tools": "auto",  # Will be filled from detected tools
            },
        },
        "tips": [
            "Lock down expected tools — tool regressions are the #1 issue",
            "Use `evalview capture` to record real tool sequences",
            "Set `max_latency` based on your SLA requirements",
        ],
    },
    "multi-step": {
        "name": "Multi-Step Agent",
        "description": "Agent with complex tool chains (3+ tools per query)",
        "icon": "🔗",
        "recommended_checks": {
            "output_quality": True,
            "hallucination": True,
            "safety": False,
            "pii_detection": False,
            "tool_accuracy": True,
            "sequence_check": True,
        },
        "thresholds": {
            "min_score": 70,
            "max_latency": 30000,
            "max_cost": 0.50,
        },
        "yaml_additions": {
            "expected": {
                "tools": "auto",
                "tool_sequence": "auto",
            },
        },
        "diff": {
            "tool_similarity_threshold": 0.75,
            "output_similarity_threshold": 0.85,
        },
        "tips": [
            "Use `tool_sequence` to catch ordering regressions",
            "Consider `evalview snapshot --variant` for non-deterministic paths",
            "Set a cost budget — multi-step agents can get expensive",
            "Use `--statistical 5` periodically to measure variance",
        ],
    },
    "rag": {
        "name": "RAG Agent",
        "description": "Retrieval-Augmented Generation agent",
        "icon": "📚",
        "recommended_checks": {
            "output_quality": True,
            "hallucination": True,
            "safety": False,
            "pii_detection": True,
            "tool_accuracy": True,
            "sequence_check": False,
        },
        "thresholds": {
            "min_score": 75,
            "max_latency": 20000,
        },
        "yaml_additions": {
            "expected": {
                "tools": "auto",
            },
            "hallucination": {
                "check": True,
                "confidence_threshold": 0.7,
            },
        },
        "tips": [
            "Hallucination detection is critical for RAG — always enable it",
            "Add `contains:` with facts from your knowledge base",
            "Test with queries that should return 'I don't know'",
            "Monitor output quality after knowledge base updates",
        ],
    },
    "coding": {
        "name": "Coding Agent",
        "description": "Agent that generates or modifies code",
        "icon": "💻",
        "recommended_checks": {
            "output_quality": True,
            "hallucination": False,
            "safety": True,
            "pii_detection": False,
            "tool_accuracy": True,
            "sequence_check": True,
        },
        "thresholds": {
            "min_score": 70,
            "max_latency": 60000,
            "max_cost": 1.00,
        },
        "tips": [
            "Long latency thresholds are normal for coding agents",
            "Test both code generation and code modification scenarios",
            "Use `not_contains:` to catch deprecated APIs or patterns",
            "Monitor cost — coding agents tend to use many tokens",
        ],
    },
}


def detect_agent_type(
    tools: List[str],
    output_sample: str = "",
    description: str = "",
) -> str:
    """Detect agent type from its behavior.

    Args:
        tools: List of tool names the agent used during probing.
        output_sample: Sample output text from the agent.
        description: User-provided description of the agent.

    Returns:
        Profile key: 'chat', 'tool-use', 'multi-step', 'rag', or 'coding'.
    """
    desc_lower = description.lower()

    # Check description for strong signals
    if any(kw in desc_lower for kw in ("rag", "retriev", "knowledge base", "document")):
        return "rag"
    if any(kw in desc_lower for kw in ("code", "coding", "program", "develop", "debug")):
        return "coding"

    # Check tools for strong RAG/coding signals (require specific tool names, not generic ones)
    tool_names = {t.lower() for t in tools}
    rag_signals = {"retrieve", "fetch_doc", "query_db", "embed", "vector_search", "rag_search"}
    coding_signals = {"write_file", "read_file", "run_code", "bash", "execute_code", "edit_file"}
    if tool_names & rag_signals:
        return "rag"
    if tool_names & coding_signals:
        return "coding"

    # Classify by tool count
    n_tools = len(tools)
    unique_tools = len(set(tools))

    if n_tools == 0:
        return "chat"
    elif unique_tools >= 3 or n_tools >= 4:
        return "multi-step"
    else:
        return "tool-use"


def get_profile(profile_key: str) -> Dict[str, Any]:
    """Get a profile by key. Returns 'tool-use' as fallback."""
    return EVAL_PROFILES.get(profile_key, EVAL_PROFILES["tool-use"])


def generate_config_yaml(
    profile_key: str,
    endpoint: str,
    adapter: str,
    detected_tools: Optional[List[str]] = None,
) -> str:
    """Generate a .evalview/config.yaml string from a profile.

    Args:
        profile_key: The detected profile type.
        endpoint: Agent endpoint URL.
        adapter: Adapter type (http, openai, etc.)
        detected_tools: Tools found during probing.

    Returns:
        YAML config string.
    """
    profile = get_profile(profile_key)
    thresholds = profile.get("thresholds", {})

    lines = [
        f"# EvalView config — {profile['icon']} {profile['name']} profile",
        f"# {profile['description']}",
        f"# Generated by: evalview init",
        "",
        f"endpoint: {endpoint}",
        f"adapter: {adapter}",
        "",
        "thresholds:",
    ]

    for key, value in thresholds.items():
        lines.append(f"  {key}: {value}")

    # Diff config for multi-step agents
    diff_cfg = profile.get("diff")
    if diff_cfg:
        lines.append("")
        lines.append("diff:")
        for key, value in diff_cfg.items():
            lines.append(f"  {key}: {value}")

    # Recommended checks
    checks = profile.get("recommended_checks", {})
    enabled = [k for k, v in checks.items() if v]
    disabled = [k for k, v in checks.items() if not v]

    if enabled or disabled:
        lines.append("")
        lines.append("# Recommended evaluators for this agent type:")
        for check in enabled:
            lines.append(f"#   ✓ {check}")
        for check in disabled:
            lines.append(f"#   ✗ {check} (not needed for {profile['name'].lower()})")

    # Tips
    tips = profile.get("tips", [])
    if tips:
        lines.append("")
        lines.append("# Tips:")
        for tip in tips:
            lines.append(f"#   • {tip}")

    lines.append("")
    return "\n".join(lines)


def generate_test_yaml(
    profile_key: str,
    test_name: str,
    query: str,
    detected_tools: Optional[List[str]] = None,
    output_keywords: Optional[List[str]] = None,
) -> str:
    """Generate a test YAML with profile-appropriate assertions.

    Args:
        profile_key: The detected profile type.
        test_name: Name for the test.
        query: The test query.
        detected_tools: Tools detected during probing.
        output_keywords: Keywords found in agent output.

    Returns:
        YAML test file content string.
    """
    profile = get_profile(profile_key)
    thresholds = profile.get("thresholds", {})
    checks = profile.get("recommended_checks", {})

    lines = [
        f'name: "{test_name}"',
        f'description: "Auto-generated test — {profile["icon"]} {profile["name"]} profile"',
        "",
        "input:",
        f'  query: "{query}"',
        "",
        "expected:",
    ]

    # Add tools if detected and profile recommends it
    if detected_tools and checks.get("tool_accuracy"):
        lines.append("  tools:")
        for tool in detected_tools:
            lines.append(f"    - {tool}")

    # Add tool sequence for multi-step
    if detected_tools and checks.get("sequence_check") and len(detected_tools) >= 2:
        lines.append("  tool_sequence:")
        for tool in detected_tools:
            lines.append(f"    - {tool}")

    # Output assertions
    lines.append("  output:")
    if output_keywords:
        lines.append("    contains:")
        for kw in output_keywords[:4]:
            lines.append(f'      - "{kw}"')

    lines.append("    not_contains:")
    lines.append('      - "error"')
    lines.append('      - "Error"')

    # Hallucination check for RAG
    if checks.get("hallucination"):
        lines.append("")
        lines.append("hallucination:")
        lines.append("  check: true")
        lines.append("  confidence_threshold: 0.7")

    # Safety check if recommended
    if checks.get("safety"):
        lines.append("")
        lines.append("safety:")
        lines.append("  check: true")

    # Thresholds
    lines.append("")
    lines.append("thresholds:")
    for key, value in thresholds.items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    return "\n".join(lines)


def display_profile_recommendation(profile_key: str, detected_tools: List[str]) -> None:
    """Display the recommended profile to the user with Rich formatting."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    profile = get_profile(profile_key)

    # Build recommendations table
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column("Check", style="dim")
    table.add_column("Status")

    for check, enabled in profile.get("recommended_checks", {}).items():
        status = "[green]✓ enabled[/green]" if enabled else "[dim]✗ skip[/dim]"
        table.add_row(check.replace("_", " ").title(), status)

    thresholds = profile.get("thresholds", {})
    threshold_str = "  ".join(f"{k}: {v}" for k, v in thresholds.items())

    tool_str = ""
    if detected_tools:
        tool_str = f"\n[dim]Detected tools:[/dim] [cyan]{', '.join(detected_tools)}[/cyan]"

    console.print(Panel(
        f"[bold]{profile['icon']} {profile['name']}[/bold] — {profile['description']}\n"
        f"{tool_str}\n\n"
        f"[bold]Recommended checks:[/bold]",
        title="[bold]Agent Profile[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print(table)
    console.print()

    tips = profile.get("tips", [])
    if tips:
        console.print("[bold]Tips for this agent type:[/bold]")
        for tip in tips:
            console.print(f"  [dim]•[/dim] {tip}")
        console.print()
