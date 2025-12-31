"""Simple LangGraph agent with search tool for EvalView testing.

Run with: langgraph dev
Server starts at: http://localhost:2024
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# Agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# Tools
@tool
def tavily_search_results_json(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query

    Returns:
        Search results as JSON string
    """
    # Try real Tavily if available, otherwise return mock data
    tavily_key = os.getenv("TAVILY_API_KEY")

    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=3)
            results = response.get("results", [])
            return str([{"title": r["title"], "content": r["content"][:200]} for r in results])
        except Exception as e:
            pass  # Fall through to mock

    # Mock response for demo/testing without API key
    query_lower = query.lower()

    if "weather" in query_lower:
        if "san francisco" in query_lower:
            return '[{"title": "San Francisco Weather", "content": "Current weather in San Francisco: 65°F, partly cloudy with light winds. Expect temperatures between 58-68°F today."}]'
        elif "tokyo" in query_lower:
            return '[{"title": "Tokyo Weather", "content": "Current weather in Tokyo: 72°F, sunny with humidity at 65%. Pleasant conditions expected throughout the day."}]'
        else:
            return '[{"title": "Weather Update", "content": "Current conditions: 70°F, clear skies. Check local forecasts for detailed information."}]'

    elif "stock" in query_lower or "price" in query_lower:
        return '[{"title": "Stock Market Update", "content": "Markets are showing mixed signals today. Tech stocks up 1.2%, while energy sector down 0.5%."}]'

    elif "news" in query_lower:
        return '[{"title": "Latest News", "content": "Breaking: Major developments in AI technology continue to reshape industries worldwide."}]'

    else:
        return f'[{{"title": "Search Results", "content": "Found relevant information about: {query}"}}]'


@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic math operations.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        Result of the operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return float("inf")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


# All tools
tools = [tavily_search_results_json, calculator]


def create_agent():
    """Create the LangGraph agent."""

    # Use GPT-4o-mini by default (cheap and fast)
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    ).bind_tools(tools)

    tool_node = ToolNode(tools)

    def should_continue(state: AgentState) -> str:
        """Decide whether to use tools or finish."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: AgentState) -> dict:
        """Call the LLM."""
        messages = state["messages"]

        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            system = SystemMessage(content=(
                "You are a helpful assistant with access to search and calculator tools. "
                "Use the search tool to find current information when needed. "
                "Always provide clear, concise answers based on the information you find."
            ))
            messages = [system] + messages

        response = model.invoke(messages)
        return {"messages": [response]}

    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Export the graph for langgraph dev
graph = create_agent()
