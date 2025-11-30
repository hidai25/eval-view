"""
EvalView Demo Agent - A simple FastAPI agent for testing.
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
    # Per-step metrics for EvalView
    latency: float = 0.0
    cost: float = 0.0


class ExecuteResponse(BaseModel):
    output: str
    tool_calls: List[ToolCall]
    cost: float
    latency: float
    tokens: Optional[Dict[str, int]] = None  # Token usage for EvalView


def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic math operations."""
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b != 0 else 0}
    return ops.get(operation, 0)


def get_weather(city: str) -> Dict[str, Any]:
    """Get weather for a city (mock data)."""
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

    # Simulate realistic LLM processing time (10-20ms per tool)
    time.sleep(0.015)  # 15ms base delay

    if any(op in query_lower for op in ["plus", "add", "+", "sum"]):
        numbers = re.findall(r"\d+", query)
        if len(numbers) >= 2:
            start = time.time()
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("add", a, b)
            latency = (time.time() - start) * 1000
            step_cost = 0.001
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "add", "a": a, "b": b}, result=result, latency=latency, cost=step_cost))
            total_cost += step_cost
            return f"The result of {a} + {b} = {result}", tool_calls, total_cost

    elif any(op in query_lower for op in ["minus", "subtract", "-"]):
        numbers = re.findall(r"\d+", query)
        if len(numbers) >= 2:
            start = time.time()
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("subtract", a, b)
            latency = (time.time() - start) * 1000
            step_cost = 0.001
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "subtract", "a": a, "b": b}, result=result, latency=latency, cost=step_cost))
            total_cost += step_cost
            return f"The result of {a} - {b} = {result}", tool_calls, total_cost

    elif any(op in query_lower for op in ["times", "multiply", "*"]):
        numbers = re.findall(r"\d+", query)
        if len(numbers) >= 2:
            start = time.time()
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("multiply", a, b)
            latency = (time.time() - start) * 1000
            step_cost = 0.001
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "multiply", "a": a, "b": b}, result=result, latency=latency, cost=step_cost))
            total_cost += step_cost
            return f"The result of {a} * {b} = {result}", tool_calls, total_cost

    # Weather + Fahrenheit conversion (multi-tool)
    elif "weather" in query_lower and "fahrenheit" in query_lower:
        # Extract city name
        city = "tokyo"  # default
        for c in ["tokyo", "london", "new york", "paris", "sydney"]:
            if c in query_lower:
                city = c
                break

        # Step 1: Get weather
        start = time.time()
        weather = get_weather(city)
        temp_c = weather["temp"]
        latency = (time.time() - start) * 1000
        step_cost = 0.001
        tool_calls.append(ToolCall(
            name="get_weather",
            arguments={"city": city},
            result=weather,
            latency=latency,
            cost=step_cost
        ))
        total_cost += step_cost

        # Step 2: Convert C to F using calculator (F = C * 1.8 + 32)
        start = time.time()
        temp_f = calculator("multiply", temp_c, 1.8)
        latency = (time.time() - start) * 1000
        step_cost = 0.001
        tool_calls.append(ToolCall(
            name="calculator",
            arguments={"operation": "multiply", "a": temp_c, "b": 1.8},
            result=temp_f,
            latency=latency,
            cost=step_cost
        ))
        total_cost += step_cost

        start = time.time()
        temp_f = calculator("add", temp_f, 32)
        latency = (time.time() - start) * 1000
        step_cost = 0.001
        tool_calls.append(ToolCall(
            name="calculator",
            arguments={"operation": "add", "a": temp_f - 32, "b": 32},
            result=temp_f,
            latency=latency,
            cost=step_cost
        ))
        total_cost += step_cost

        return f"The weather in {city.title()} is {temp_c}°C ({temp_f:.1f}°F), {weather['condition']}", tool_calls, total_cost

    # Simple weather query
    elif "weather" in query_lower:
        # Extract city name
        city = "tokyo"  # default
        for c in ["tokyo", "london", "new york", "paris", "sydney"]:
            if c in query_lower:
                city = c
                break

        start = time.time()
        weather = get_weather(city)
        latency = (time.time() - start) * 1000
        step_cost = 0.001
        tool_calls.append(ToolCall(
            name="get_weather",
            arguments={"city": city},
            result=weather,
            latency=latency,
            cost=step_cost
        ))
        total_cost += step_cost
        return f"The weather in {city.title()} is {weather['temp']}°C, {weather['condition']} with {weather['humidity']}% humidity", tool_calls, total_cost

    return f"I received your query: {query}", tool_calls, total_cost


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
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
    total_latency = (time.time() - start) * 1000

    # Distribute total latency across steps for more realistic reporting
    if tools:
        per_step_latency = total_latency / len(tools)
        tools = [
            ToolCall(
                name=t.name,
                arguments=t.arguments,
                result=t.result,
                latency=per_step_latency,
                cost=t.cost,
            )
            for t in tools
        ]

    # Mock token usage (realistic for a simple agent)
    tokens = {"input": 50 + len(query), "output": 80 + len(output), "cached": 0}

    return ExecuteResponse(output=output, tool_calls=tools, cost=cost, latency=total_latency, tokens=tokens)


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    print("Demo Agent running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
