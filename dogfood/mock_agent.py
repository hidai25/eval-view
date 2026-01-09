"""Deterministic mock agent for dogfooding EvalView.

This agent returns predictable responses based on the query, allowing us to
verify that EvalView's evaluation logic produces correct scores.

Scenarios:
- "calculate X" -> uses calculator tool, returns correct answer
- "search for X" -> uses search tool, returns relevant results
- "calculate X wrong" -> uses calculator but returns wrong answer (tests scoring)
- "hallucinate" -> makes up specific facts without tool calls (tests hallucination detection)
- "no tools" -> answers without using any tools
- "wrong tool" -> uses wrong tool for the task
"""

import re
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="EvalView Mock Agent")


class ExecuteRequest(BaseModel):
    query: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, str]]] = None


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result: Any
    latency: float = 10.0
    cost: float = 0.0


class ExecuteResponse(BaseModel):
    output: str
    tool_calls: List[ToolCall]
    cost: float
    latency: float
    tokens: Optional[Dict[str, int]] = None


# =============================================================================
# Deterministic Response Logic
# =============================================================================


def handle_calculate(query: str) -> tuple[str, List[ToolCall]]:
    """Handle calculation queries - returns correct answer with calculator tool."""
    # Extract numbers from "calculate X + Y" or "calculate X * Y" etc.
    match = re.search(r"calculate\s+(\d+)\s*([+\-*/])\s*(\d+)", query.lower())
    if match:
        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        elif op == "/":
            result = a // b if b != 0 else 0
        else:
            result = 0

        tool_calls = [
            ToolCall(
                name="calculator",
                arguments={"expression": f"{a} {op} {b}"},
                result=str(result),
            )
        ]
        return f"The result of {a} {op} {b} is {result}.", tool_calls

    return "I need a valid calculation expression.", []


def handle_calculate_wrong(query: str) -> tuple[str, List[ToolCall]]:
    """Handle calculation but return WRONG answer - tests that EvalView catches errors."""
    match = re.search(r"calculate\s+(\d+)\s*([+\-*/])\s*(\d+)", query.lower())
    if match:
        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
        # Calculate correct answer then return wrong one
        if op == "+":
            result = a + b
        elif op == "*":
            result = a * b
        else:
            result = a + b
        wrong_result = result + 999  # Intentionally wrong

        tool_calls = [
            ToolCall(
                name="calculator",
                arguments={"expression": f"{a} {op} {b}"},
                result=str(result),  # Tool returns correct
            )
        ]
        # But agent says wrong answer
        return f"The result of {a} {op} {b} is {wrong_result}.", tool_calls

    return "I need a valid calculation expression.", []


def handle_search(query: str) -> tuple[str, List[ToolCall]]:
    """Handle search queries - uses search tool and returns results."""
    search_term = query.lower().replace("search for", "").replace("search", "").strip()

    tool_calls = [
        ToolCall(
            name="search",
            arguments={"query": search_term},
            result=f"Found 3 results for '{search_term}': Result 1, Result 2, Result 3",
        )
    ]
    return (
        f"I found information about {search_term}. Here are the top results: Result 1, Result 2, Result 3.",
        tool_calls,
    )


def handle_hallucinate(_query: str) -> tuple[str, List[ToolCall]]:
    """Return fabricated specific facts without any tool calls."""
    # No tools called, but claims specific data
    return (
        "The current temperature in Paris is exactly 23.7°C with 47% humidity. "
        "The wind is blowing from the northwest at 14.3 km/h. "
        "Tomorrow's forecast shows rain starting at precisely 3:42 PM."
    ), []


def handle_wrong_tool(query: str) -> tuple[str, List[ToolCall]]:
    """Use wrong tool for the task - e.g., use weather for calculation."""
    match = re.search(r"calculate\s+(\d+)\s*([+\-*/])\s*(\d+)", query.lower())
    if match:
        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))

        # Use weather tool instead of calculator (wrong!)
        tool_calls = [
            ToolCall(
                name="weather",
                arguments={"location": "Paris"},
                result="Sunny, 22°C",
            )
        ]
        return f"The weather is nice today! By the way, {a} {op} {b} equals 42.", tool_calls

    return "I checked the weather instead.", []


def handle_multi_step(query: str) -> tuple[str, List[ToolCall]]:
    """Handle multi-step query requiring search then summarize."""
    tool_calls = [
        ToolCall(
            name="search",
            arguments={"query": "EvalView features"},
            result="EvalView is an AI agent testing framework with evaluators for tool calls, sequences, and output quality.",
        ),
        ToolCall(
            name="summarize",
            arguments={"text": "EvalView is an AI agent testing framework..."},
            result="EvalView: AI agent testing with multiple evaluators.",
        ),
    ]
    return (
        "EvalView is an AI agent testing framework that provides evaluators for tool calls, sequences, and output quality.",
        tool_calls,
    )


def handle_no_tools(_query: str) -> tuple[str, List[ToolCall]]:
    """Answer without using any tools."""
    return (
        "I can answer this from my knowledge: The sky is blue because of Rayleigh scattering.",
        [],
    )


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    start = time.time()

    # Get query from request
    if request.query:
        query = request.query
    elif request.messages:
        user_msgs = [m for m in request.messages if m.get("role") == "user"]
        query = user_msgs[-1].get("content", "") if user_msgs else ""
    else:
        query = ""

    query_lower = query.lower()

    # Route to appropriate handler based on query
    # More specific patterns must come first!
    if "wrong tool" in query_lower and "calculate" in query_lower:
        output, tool_calls = handle_wrong_tool(query)
    elif "wrong" in query_lower and "calculate" in query_lower:
        output, tool_calls = handle_calculate_wrong(query)
    elif "calculate" in query_lower:
        output, tool_calls = handle_calculate(query)
    elif "search" in query_lower and "summarize" in query_lower:
        output, tool_calls = handle_multi_step(query)
    elif "search" in query_lower:
        output, tool_calls = handle_search(query)
    elif "hallucinate" in query_lower:
        output, tool_calls = handle_hallucinate(query)
    elif "no tools" in query_lower:
        output, tool_calls = handle_no_tools(query)
    else:
        output = f"I received your query: {query}"
        tool_calls = []

    latency = (time.time() - start) * 1000

    return ExecuteResponse(
        output=output,
        tool_calls=tool_calls,
        cost=0.001 * len(tool_calls),
        latency=latency,
        tokens={"input": len(query) // 4, "output": len(output) // 4, "cached": 0},
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "mock"}


if __name__ == "__main__":
    import uvicorn

    print("Mock Agent running on http://localhost:8002")
    print("This agent returns deterministic responses for dogfood testing")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
