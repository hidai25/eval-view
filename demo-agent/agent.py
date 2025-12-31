"""
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
        numbers = re.findall(r"\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("add", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "add", "a": a, "b": b}, result=result, cost=0.001))
            return f"The result of {a} + {b} = {result}", tool_calls, 0.001

    elif any(op in query_lower for op in ["minus", "subtract", "-"]):
        numbers = re.findall(r"\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("subtract", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "subtract", "a": a, "b": b}, result=result, cost=0.001))
            return f"The result of {a} - {b} = {result}", tool_calls, 0.001

    elif any(op in query_lower for op in ["times", "multiply", "*"]):
        numbers = re.findall(r"\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("multiply", a, b)
            tool_calls.append(ToolCall(name="calculator", arguments={"operation": "multiply", "a": a, "b": b}, result=result, cost=0.001))
            return f"The result of {a} * {b} = {result}", tool_calls, 0.001

    elif any(op in query_lower for op in ["divided", "divide", "/"]):
        numbers = re.findall(r"\d+", query)
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
