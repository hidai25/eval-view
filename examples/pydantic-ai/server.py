"""Minimal Pydantic AI agent wrapped in FastAPI for EvalView testing.

Run with:
    pip install pydantic-ai fastapi uvicorn
    uvicorn server:app --port 8000

Then test with EvalView:
    evalview init       # Select http adapter, http://localhost:8000/agent
    evalview snapshot   # Capture baseline
    evalview check      # Catch regressions
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic_ai import Agent, RunContext

# ── Define tools ──────────────────────────────────────────────────────────────

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt=(
        "You are a helpful customer support agent. "
        "Use your tools to look up information before answering. "
        "Never guess — always verify with a tool call first."
    ),
)


@agent.tool
async def lookup_order(ctx: RunContext[None], order_id: str) -> str:
    """Look up an order by ID."""
    # In production, this would query your database
    return f"Order {order_id}: shipped on 2026-03-20, arriving 2026-03-25, status: in transit"


@agent.tool
async def check_policy(ctx: RunContext[None], policy_type: str) -> str:
    """Check company policy for a given topic."""
    policies = {
        "refund": "Refunds allowed within 30 days of purchase with receipt.",
        "cancellation": "Orders can be cancelled before shipping.",
        "returns": "Returns accepted within 14 days, item must be unused.",
    }
    return policies.get(policy_type, f"No policy found for: {policy_type}")


# ── FastAPI wrapper ───────────────────────────────────────────────────────────

app = FastAPI(title="Pydantic AI Support Agent")


@app.post("/agent")
async def invoke(request: Dict[str, Any]) -> Dict[str, Any]:
    """EvalView-compatible endpoint.

    Expects: {"query": "..."}
    Returns: {"output": "...", "steps": [...], "usage": {...}}
    """
    query = request.get("query", "")
    result = await agent.run(query)

    # Extract tool calls from the run messages
    steps: List[Dict[str, Any]] = []
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if hasattr(part, "tool_name"):
                    steps.append({
                        "tool": part.tool_name,
                        "inputs": getattr(part, "args", {}),
                        "output": str(getattr(part, "content", "")),
                    })

    usage = result.usage()
    return {
        "output": result.data,
        "steps": steps,
        "usage": {
            "total_tokens": usage.total_tokens if usage else 0,
        },
    }
