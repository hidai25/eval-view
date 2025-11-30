"""
Reference Agent for EvalView Testing

A simple FastAPI agent with multiple tools to test EvalView compatibility.
This serves as a template for testing other agent frameworks.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import time
import json

app = FastAPI(title="Reference Test Agent")


class Message(BaseModel):
    role: str
    content: str


class ExecuteRequest(BaseModel):
    messages: List[Message]


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result: Any


class ExecuteResponse(BaseModel):
    output: str
    tool_calls: List[ToolCall]
    cost: float
    latency: float


# Simulated Tools
def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations."""
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf"),
    }
    return operations.get(operation, 0)


def get_weather(city: str) -> Dict[str, Any]:
    """Get weather information for a city."""
    # Simulated weather data
    weather_db = {
        "new york": {"temp": 72, "condition": "sunny", "humidity": 65},
        "london": {"temp": 55, "condition": "rainy", "humidity": 80},
        "tokyo": {"temp": 68, "condition": "cloudy", "humidity": 70},
        "paris": {"temp": 60, "condition": "partly cloudy", "humidity": 72},
        "invalid": None,
    }
    city_lower = city.lower()
    if city_lower not in weather_db or weather_db[city_lower] is None:
        return {"error": f"City '{city}' not found"}
    return weather_db[city_lower]


def search_web(query: str) -> List[Dict[str, str]]:
    """Search the web for information."""
    # Simulated search results
    return [
        {
            "title": f"Result for {query}",
            "url": f"https://example.com/{query.replace(' ', '-')}",
            "snippet": f"Information about {query}...",
        }
    ]


def convert_temperature(temp: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between Celsius and Fahrenheit."""
    if from_unit.lower() == "f" and to_unit.lower() == "c":
        return (temp - 32) * 5 / 9
    elif from_unit.lower() == "c" and to_unit.lower() == "f":
        return (temp * 9 / 5) + 32
    return temp


def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get stock price information."""
    # Simulated stock data
    stocks = {
        "AAPL": {"price": 178.50, "change": +2.30, "volume": 50000000},
        "GOOGL": {"price": 142.80, "change": -1.20, "volume": 25000000},
        "MSFT": {"price": 378.90, "change": +5.60, "volume": 30000000},
    }
    symbol_upper = symbol.upper()
    if symbol_upper not in stocks:
        return {"error": f"Stock symbol '{symbol}' not found"}
    return stocks[symbol_upper]


# Available tools registry
TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_web": search_web,
    "convert_temperature": convert_temperature,
    "get_stock_price": get_stock_price,
}


def simple_agent_logic(query: str) -> tuple[str, List[ToolCall], float]:
    """
    Simple rule-based agent that determines which tools to call.
    In production, this would be an LLM deciding which tools to use.
    """
    query_lower = query.lower()
    tool_calls = []
    cost = 0.0

    # Weather queries
    if "weather" in query_lower:
        cities = ["new york", "london", "tokyo", "paris"]
        for city in cities:
            if city in query_lower:
                result = get_weather(city)
                tool_calls.append(
                    ToolCall(name="get_weather", arguments={"city": city}, result=result)
                )
                cost += 0.001

                # If temperature conversion requested
                if "celsius" in query_lower or "fahrenheit" in query_lower:
                    if "temp" in result and "error" not in result:
                        from_unit = "F"
                        to_unit = "C" if "celsius" in query_lower else "F"
                        converted = convert_temperature(result["temp"], from_unit, to_unit)
                        tool_calls.append(
                            ToolCall(
                                name="convert_temperature",
                                arguments={
                                    "temp": result["temp"],
                                    "from_unit": from_unit,
                                    "to_unit": to_unit,
                                },
                                result=converted,
                            )
                        )
                        cost += 0.001
                break

    # Calculator queries
    elif any(op in query_lower for op in ["add", "plus", "+", "sum"]):
        # Simple number extraction (very basic)
        import re

        numbers = re.findall(r"\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("add", a, b)
            tool_calls.append(
                ToolCall(
                    name="calculator", arguments={"operation": "add", "a": a, "b": b}, result=result
                )
            )
            cost += 0.0005

    elif any(op in query_lower for op in ["multiply", "times", "*"]):
        import re

        numbers = re.findall(r"\d+", query)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            result = calculator("multiply", a, b)
            tool_calls.append(
                ToolCall(
                    name="calculator",
                    arguments={"operation": "multiply", "a": a, "b": b},
                    result=result,
                )
            )
            cost += 0.0005

    # Stock queries
    elif "stock" in query_lower or any(sym in query_lower.upper() for sym in ["AAPL", "GOOGL", "MSFT"]):
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            if symbol.lower() in query_lower:
                result = get_stock_price(symbol)
                tool_calls.append(
                    ToolCall(
                        name="get_stock_price", arguments={"symbol": symbol}, result=result
                    )
                )
                cost += 0.002
                break

    # Search queries (fallback)
    elif "search" in query_lower or len(tool_calls) == 0:
        result = search_web(query)
        tool_calls.append(ToolCall(name="search_web", arguments={"query": query}, result=result))
        cost += 0.003

    # Generate output based on tool results
    if not tool_calls:
        output = f"I couldn't find relevant tools to answer: {query}"
    else:
        output_parts = [f"Based on my analysis of '{query}':"]
        for tc in tool_calls:
            if isinstance(tc.result, dict):
                if "error" in tc.result:
                    output_parts.append(f"- {tc.name}: {tc.result['error']}")
                else:
                    output_parts.append(f"- {tc.name}: {json.dumps(tc.result)}")
            else:
                output_parts.append(f"- {tc.name}: {tc.result}")
        output = "\n".join(output_parts)

    return output, tool_calls, cost


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    """Execute agent with given messages."""
    start_time = time.time()

    try:
        # Get the user query from messages
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        query = user_messages[-1].content

        # Run agent logic
        output, tool_calls, cost = simple_agent_logic(query)

        # Calculate latency
        latency = (time.time() - start_time) * 1000  # Convert to ms

        return ExecuteResponse(output=output, tool_calls=tool_calls, cost=cost, latency=latency)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "tools": list(TOOLS.keys())}


@app.get("/tools")
async def list_tools():
    """List available tools."""
    return {
        "tools": [
            {
                "name": "calculator",
                "description": "Perform basic arithmetic operations",
                "parameters": ["operation", "a", "b"],
            },
            {
                "name": "get_weather",
                "description": "Get weather information for a city",
                "parameters": ["city"],
            },
            {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": ["query"],
            },
            {
                "name": "convert_temperature",
                "description": "Convert temperature between Celsius and Fahrenheit",
                "parameters": ["temp", "from_unit", "to_unit"],
            },
            {
                "name": "get_stock_price",
                "description": "Get stock price information",
                "parameters": ["symbol"],
            },
        ]
    }


if __name__ == "__main__":
    print("🚀 Starting Reference Test Agent on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    print("🔧 Available tools:", list(TOOLS.keys()))
    uvicorn.run(app, host="0.0.0.0", port=8000)
