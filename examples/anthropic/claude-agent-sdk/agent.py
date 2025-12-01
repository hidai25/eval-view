"""
Example Claude Agent SDK agent for EvalView testing.

This demonstrates how to build a simple agent with custom tools
using the Claude Agent SDK.

Install: pip install claude-agent-sdk
"""

import anyio
from claude_agent_sdk import ClaudeAgentOptions, query, custom_tool


# Define custom tools using the @custom_tool decorator
@custom_tool
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: The city name to get weather for
    """
    # Simulated weather data
    weather_data = {
        "tokyo": "72°F, Sunny",
        "london": "58°F, Cloudy",
        "new york": "65°F, Partly Cloudy",
        "paris": "61°F, Rainy",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather in {city}: 70°F, Clear skies"


@custom_tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit.

    Args:
        value: The temperature value to convert
        from_unit: Source unit ('celsius' or 'fahrenheit')
        to_unit: Target unit ('celsius' or 'fahrenheit')
    """
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        result = value * 9/5 + 32
        return f"{value}°C = {result:.1f}°F"
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        result = (value - 32) * 5/9
        return f"{value}°F = {result:.1f}°C"
    return f"Cannot convert from {from_unit} to {to_unit}"


@custom_tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like '2 + 2' or '10 * 5'
    """
    try:
        # Safe evaluation of math expressions
        allowed_chars = set("0123456789+-*/.(). ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return "Invalid expression"
    except Exception as e:
        return f"Error: {e}"


async def run_agent(prompt: str) -> str:
    """Run the agent with a prompt and return the response."""

    options = ClaudeAgentOptions(
        # Register our custom tools
        custom_tools=[get_weather, convert_temperature, calculate],
        # Model to use
        model="claude-sonnet-4-5-20250929",
        # Max tokens for response
        max_tokens=4096,
    )

    messages = []
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            messages.append(str(message.content))

    return "\n".join(messages) if messages else "No response"


# CLI interface for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "What's the weather in Tokyo and convert 25 celsius to fahrenheit?"

    print(f"Prompt: {prompt}\n")
    result = anyio.run(run_agent, prompt)
    print(f"Response:\n{result}")
