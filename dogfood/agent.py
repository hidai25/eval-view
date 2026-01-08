"""
EvalView Dogfood Agent - Wraps EvalView chat mode as an HTTP agent.
This allows testing EvalView with EvalView itself.
"""

import asyncio
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

# Load environment variables from .env.local
from dotenv import load_dotenv
load_dotenv(".env.local")

from fastapi import FastAPI
from pydantic import BaseModel

# Import EvalView chat internals
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalview.chat import ChatSession, extract_commands, SYSTEM_PROMPT
from evalview.core.llm_provider import LLMProvider, detect_available_providers

app = FastAPI(title="EvalView Dogfood Agent")


class ExecuteRequest(BaseModel):
    query: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, str]]] = None


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


def get_provider():
    """Get an LLM provider - prefer OpenAI/Anthropic over Ollama."""
    from evalview.core.llm_provider import PROVIDER_CONFIGS

    available = detect_available_providers()
    if not available:
        raise RuntimeError("No LLM provider available")

    # Prefer cloud providers (OpenAI, Anthropic) over Ollama
    for provider, _api_key in available:
        if provider in (LLMProvider.OPENAI, LLMProvider.ANTHROPIC):
            # Get default model from config
            model = PROVIDER_CONFIGS[provider].default_model
            return provider, model

    # Fall back to first available
    provider, _api_key = available[0]
    model = PROVIDER_CONFIGS[provider].default_model
    return provider, model


def execute_command(cmd: str) -> str:
    """Execute an evalview command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output = result.stdout
        if result.stderr:
            output += f"\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except Exception as e:
        return f"Error: {e}"


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    start = time.time()

    # Get the query
    if request.query:
        query = request.query
    elif request.messages:
        user_msgs = [m for m in request.messages if m.get("role") == "user"]
        if not user_msgs:
            return ExecuteResponse(
                output="No user message provided",
                tool_calls=[],
                cost=0.0,
                latency=0.0
            )
        query = user_msgs[-1].get("content", "")
    else:
        return ExecuteResponse(
            output="Either query or messages must be provided",
            tool_calls=[],
            cost=0.0,
            latency=0.0
        )

    tool_calls = []

    # Get LLM provider
    try:
        provider, model = get_provider()
    except RuntimeError as e:
        return ExecuteResponse(
            output=str(e),
            tool_calls=[],
            cost=0.0,
            latency=(time.time() - start) * 1000
        )

    # Create chat session and get response
    chat = ChatSession(provider=provider, model=model)

    # Collect full response from stream
    response_text = ""
    async for chunk in chat.stream_response(query):
        response_text += chunk

    llm_latency = (time.time() - start) * 1000

    # Extract and execute any evalview commands
    commands = extract_commands(response_text)

    for cmd in commands:
        cmd_start = time.time()
        result = execute_command(cmd)
        cmd_latency = (time.time() - cmd_start) * 1000

        tool_calls.append(ToolCall(
            name="evalview_cli",
            arguments={"command": cmd},
            result=result,
            latency=cmd_latency,
            cost=0.0
        ))

    total_latency = (time.time() - start) * 1000

    # Estimate cost (rough)
    input_tokens = len(query) // 4
    output_tokens = len(response_text) // 4

    # Pricing varies by provider
    if provider == LLMProvider.OPENAI:
        cost = (input_tokens * 0.00001) + (output_tokens * 0.00003)
    elif provider == LLMProvider.ANTHROPIC:
        cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
    else:
        cost = 0.0  # Ollama is free

    return ExecuteResponse(
        output=response_text,
        tool_calls=tool_calls,
        cost=cost,
        latency=total_latency,
        tokens={"input": input_tokens, "output": output_tokens, "cached": 0}
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    print("EvalView Dogfood Agent running on http://localhost:8001")
    print("This wraps EvalView chat mode as an HTTP agent for self-testing")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
