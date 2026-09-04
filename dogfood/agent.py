"""
EvalView Dogfood Agent - Wraps EvalView chat mode as an HTTP agent.
This allows testing EvalView with EvalView itself.
"""

import asyncio
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from evalview.chat import ChatSession, extract_commands  # noqa: E402
from evalview.core.llm_provider import LLMProvider, detect_available_providers  # noqa: E402

app = FastAPI(title="EvalView Dogfood Agent")

# Only these bounded, local operations are exercised by this harness. Each test
# must additionally opt in to its exact command through input.context.
ALLOWED_COMMANDS = {
    "evalview adapters",
    "evalview demo",
    "evalview snapshot --help",
}


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
            model = os.getenv("EVAL_MODEL") or PROVIDER_CONFIGS[provider].default_model
            return provider, model

    # Fall back to first available
    provider, _api_key = available[0]
    model = PROVIDER_CONFIGS[provider].default_model
    return provider, model


def execute_command(cmd: str, cwd: str) -> str:
    """Run an approved CLI operation without a shell or access to repo state."""
    if cmd not in ALLOWED_COMMANDS:
        raise ValueError(f"Command is not allowed by the dogfood harness: {cmd}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "evalview", *shlex.split(cmd)[1:]],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n{result.stderr}"
        if result.returncode:
            raise RuntimeError(f"Command exited with {result.returncode}: {output.strip()}")
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Command timed out after 30 seconds") from exc


async def get_chat_response(chat: ChatSession, prompt: str) -> str:
    async def collect() -> str:
        return "".join([chunk async for chunk in chat.stream_response(prompt, raise_errors=True)])

    try:
        return await asyncio.wait_for(collect(), timeout=40)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Chat provider request failed: {type(exc).__name__}"
        ) from exc


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
                output="No user message provided", tool_calls=[], cost=0.0, latency=0.0
            )
        query = user_msgs[-1].get("content", "")
    else:
        return ExecuteResponse(
            output="Either query or messages must be provided", tool_calls=[], cost=0.0, latency=0.0
        )

    tool_calls = []
    approved = (request.context or {}).get("allowed_commands", [])
    if not isinstance(approved, list) or any(
        not isinstance(cmd, str) or cmd not in ALLOWED_COMMANDS for cmd in approved
    ):
        raise HTTPException(status_code=422, detail="Invalid allowed_commands context")

    # Get LLM provider
    try:
        provider, model = get_provider()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="No LLM provider available") from exc

    # Create chat session and get response
    chat = ChatSession(provider=provider, model=model)

    # Collect full response from stream
    response_text = await get_chat_response(chat, query)

    # Extract and execute any evalview commands
    # Explanation-only tests never execute commands found in examples.
    commands = extract_commands(response_text) if approved else []
    if len(commands) > len(approved) or any(cmd not in approved for cmd in commands):
        raise HTTPException(status_code=422, detail="Chat proposed an unapproved command")

    with tempfile.TemporaryDirectory(prefix="evalview-dogfood-") as command_dir:
        for cmd in commands:
            cmd_start = time.time()
            try:
                result = await asyncio.to_thread(execute_command, cmd, command_dir)
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            cmd_latency = (time.time() - cmd_start) * 1000

            tool_calls.append(
                ToolCall(
                    name="evalview_cli",
                    arguments={"command": cmd},
                    result=result,
                    latency=cmd_latency,
                    cost=0.0,
                )
            )

    # Final claims must follow tool results, not precede them. Preserve the
    # observed output in the trace and let the assistant explain that evidence.
    generation_text = response_text
    grounding_prompt = ""
    if tool_calls:
        grounding_prompt = (
            "The approved commands have completed. These are their actual results:\n"
            + json.dumps([call.model_dump() for call in tool_calls])
            + "\nAnswer the original question using these results. Do not request more commands "
            "or claim any operations beyond those recorded here."
        )
        response_text = await get_chat_response(chat, grounding_prompt)

    total_latency = (time.time() - start) * 1000

    # Estimate cost (rough)
    input_tokens = (len(query) + len(grounding_prompt)) // 4
    output_tokens = (len(generation_text) + (len(response_text) if tool_calls else 0)) // 4

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
        tokens={"input": input_tokens, "output": output_tokens, "cached": 0},
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    from dotenv import load_dotenv
    import uvicorn

    load_dotenv(".env.local")
    print("EvalView Dogfood Agent running on http://localhost:8001")
    print("This wraps EvalView chat mode as an HTTP agent for self-testing")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
