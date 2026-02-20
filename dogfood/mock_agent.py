"""Deterministic mock agent for dogfooding EvalView.

This agent returns predictable responses based on the query, allowing us to
verify that EvalView's evaluation logic produces correct scores.

Customer support scenarios (default test suite):
- "refund" / "return" -> lookup_order + check_policy + process_refund
- "charge" / "billing" / "129" -> lookup_account + check_billing_history
- "order" / "shipping" / "placed" -> lookup_order + check_shipping
- "upgrade" / "premium" / "plan" -> lookup_account + check_plans

Additional test scenarios:
- "calculate X" -> uses calculator tool, returns correct answer
- "search for X" -> uses search tool, returns relevant results
- "calculate X wrong" -> uses calculator but returns wrong answer (tests scoring)
- "hallucinate" -> makes up specific facts without tool calls (tests hallucination detection)
- "no tools" -> answers without using any tools
- "wrong tool" -> uses wrong tool for the task
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

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


def handle_calculate(query: str) -> Tuple[str, List[ToolCall]]:
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


def handle_calculate_wrong(query: str) -> Tuple[str, List[ToolCall]]:
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


def handle_search(query: str) -> Tuple[str, List[ToolCall]]:
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


def handle_hallucinate(_query: str) -> Tuple[str, List[ToolCall]]:
    """Return fabricated specific facts without any tool calls."""
    # No tools called, but claims specific data
    return (
        "The current temperature in Paris is exactly 23.7°C with 47% humidity. "
        "The wind is blowing from the northwest at 14.3 km/h. "
        "Tomorrow's forecast shows rain starting at precisely 3:42 PM."
    ), []


def handle_wrong_tool(query: str) -> Tuple[str, List[ToolCall]]:
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


def handle_weather(query: str) -> Tuple[str, List[ToolCall]]:
    """Handle weather queries - uses get_weather tool."""
    # Extract city name from common patterns
    city = "unknown"
    for pattern in [r"weather in (\w+)", r"weather for (\w+)"]:
        m = re.search(pattern, query.lower())
        if m:
            city = m.group(1).capitalize()
            break

    city_data = {
        "Tokyo": ("22°C", "sunny"),
        "London": ("15°C", "cloudy"),
        "Paris": ("18°C", "partly cloudy"),
    }
    temp, cond = city_data.get(city, ("20°C", "clear"))

    tool_calls = [
        ToolCall(
            name="get_weather",
            arguments={"city": city},
            result=f"{temp}, {cond}",
        )
    ]
    return f"The current weather in {city} is {temp} and {cond}.", tool_calls


def handle_weather_convert(query: str) -> Tuple[str, List[ToolCall]]:
    """Handle weather + unit conversion (multi-tool)."""
    city = "London"
    m = re.search(r"weather in (\w+)", query.lower())
    if m:
        city = m.group(1).capitalize()

    city_data = {
        "Tokyo": 22,
        "London": 15,
        "Paris": 18,
    }
    celsius = city_data.get(city, 20)
    fahrenheit = round(celsius * 9 / 5 + 32)

    tool_calls = [
        ToolCall(
            name="get_weather",
            arguments={"city": city},
            result=f"{celsius}°C, cloudy",
        ),
        ToolCall(
            name="calculator",
            arguments={"expression": f"{celsius} * 9/5 + 32"},
            result=str(fahrenheit),
        ),
    ]
    return (
        f"The weather in {city} is {celsius}°C ({fahrenheit}°F) and cloudy.",
        tool_calls,
    )


def handle_natural_math(query: str) -> Tuple[str, List[ToolCall]]:
    """Handle natural-language math: 'What is X times/divided by/plus/minus Y?'"""
    q = query.lower()

    # "X times Y" / "X multiplied by Y"
    m = re.search(r"(\d+)\s+(?:times|multiplied by|x)\s+(\d+)", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        result = a * b
        tool_calls = [ToolCall(name="calculator", arguments={"expression": f"{a} * {b}"}, result=str(result))]
        return f"The result of {a} times {b} is {result}.", tool_calls

    # "X divided by Y"
    m = re.search(r"(\d+)\s+divided by\s+(\d+)", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        result = a // b if b != 0 else 0
        tool_calls = [ToolCall(name="calculator", arguments={"expression": f"{a} / {b}"}, result=str(result))]
        return f"The result of {a} divided by {b} is {result}.", tool_calls

    # "X plus Y"
    m = re.search(r"(\d+)\s+plus\s+(\d+)", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        result = a + b
        tool_calls = [ToolCall(name="calculator", arguments={"expression": f"{a} + {b}"}, result=str(result))]
        return f"The result of {a} plus {b} is {result}.", tool_calls

    # "X minus Y"
    m = re.search(r"(\d+)\s+minus\s+(\d+)", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        result = a - b
        tool_calls = [ToolCall(name="calculator", arguments={"expression": f"{a} - {b}"}, result=str(result))]
        return f"The result of {a} minus {b} is {result}.", tool_calls

    return "I need a valid math expression.", []


def handle_refund(query: str) -> Tuple[str, List[ToolCall]]:
    """Handle refund requests."""
    tool_calls = [
        ToolCall(name="lookup_order", arguments={"query": query}, result="Order #4821, $84.99, 12 days ago"),
        ToolCall(name="check_policy", arguments={"type": "return"}, result="30-day return window, full refund eligible"),
        ToolCall(name="process_refund", arguments={"order_id": "4821", "amount": 84.99}, result="Refund initiated"),
    ]
    return (
        "I've found your order #4821 for $84.99 placed 12 days ago. "
        "Our 30-day return policy covers this — I've initiated your full refund. "
        "You'll see $84.99 back in 3–5 business days.",
        tool_calls,
    )


def handle_billing_dispute(query: str) -> Tuple[str, List[ToolCall]]:
    """Handle billing disputes."""
    tool_calls = [
        ToolCall(name="lookup_account", arguments={"query": query}, result="Account #8821, annual plan"),
        ToolCall(name="check_billing_history", arguments={"account_id": "8821"}, result="$129 annual renewal, March 3rd, auto-renewal on"),
    ]
    return (
        "That $129 charge is your annual plan renewal from March 3rd. "
        "You signed up for annual billing last year with auto-renewal enabled. "
        "I can email you the full invoice or switch you to monthly billing — which would you prefer?",
        tool_calls,
    )


def handle_order_status(query: str) -> Tuple[str, List[ToolCall]]:
    """Handle order status inquiries."""
    tool_calls = [
        ToolCall(name="lookup_order", arguments={"query": query}, result="Order #5503, placed 3 days ago, processing"),
        ToolCall(name="check_shipping", arguments={"order_id": "5503"}, result="Label created, pickup scheduled for tomorrow"),
    ]
    return (
        "I found your order #5503. It's currently being prepared — the shipping label has been created "
        "and carrier pickup is scheduled for tomorrow. You'll receive a tracking number via email once it ships.",
        tool_calls,
    )


def handle_account_upgrade(query: str) -> Tuple[str, List[ToolCall]]:
    """Handle account upgrade requests."""
    tool_calls = [
        ToolCall(name="lookup_account", arguments={"query": query}, result="Account #8821, currently on basic plan"),
        ToolCall(name="check_plans", arguments={"current": "basic"}, result="Premium: $29/mo — unlimited seats, priority support, analytics"),
    ]
    return (
        "You're currently on the Basic plan. The Premium plan is $29/month and includes: "
        "unlimited team seats, priority support (< 2hr response), and advanced analytics. "
        "Would you like me to upgrade your account now?",
        tool_calls,
    )


def handle_multi_step(query: str) -> Tuple[str, List[ToolCall]]:
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


def handle_no_tools(_query: str) -> Tuple[str, List[ToolCall]]:
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
    # --- Customer support scenarios (default test suite) ---
    if any(w in query_lower for w in ("refund", "return", "jacket", "doesn't fit")):
        output, tool_calls = handle_refund(query)
    elif any(w in query_lower for w in ("charge", "billing", "invoice")) or "129" in query_lower:
        output, tool_calls = handle_billing_dispute(query)
    elif any(w in query_lower for w in ("shipping confirmation", "where is", "placed an order", "placed it", "haven't received")):
        output, tool_calls = handle_order_status(query)
    elif any(w in query_lower for w in ("upgrade", "premium plan", "premium")):
        output, tool_calls = handle_account_upgrade(query)
    # --- Additional test scenarios ---
    elif "wrong tool" in query_lower and "calculate" in query_lower:
        output, tool_calls = handle_wrong_tool(query)
    elif "wrong" in query_lower and "calculate" in query_lower:
        output, tool_calls = handle_calculate_wrong(query)
    elif "calculate" in query_lower:
        output, tool_calls = handle_calculate(query)
    elif "weather" in query_lower and ("fahrenheit" in query_lower or "celsius" in query_lower):
        output, tool_calls = handle_weather_convert(query)
    elif "weather" in query_lower:
        output, tool_calls = handle_weather(query)
    elif "search" in query_lower and "summarize" in query_lower:
        output, tool_calls = handle_multi_step(query)
    elif "search" in query_lower:
        output, tool_calls = handle_search(query)
    elif "hallucinate" in query_lower:
        output, tool_calls = handle_hallucinate(query)
    elif "no tools" in query_lower:
        output, tool_calls = handle_no_tools(query)
    elif re.search(r"\d+\s+(times|multiplied by|divided by|plus|minus)\s+\d+", query_lower):
        output, tool_calls = handle_natural_math(query)
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
