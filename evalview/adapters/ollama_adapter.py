"""Ollama adapter for EvalView.

Test local LLMs running via Ollama with zero cloud costs.

Example:
    adapter = OllamaAdapter(model="llama3.2")
    trace = await adapter.execute("What is 2+2?")
"""

import httpx
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
    SpanKind,
)
from evalview.core.tracing import Tracer


class OllamaAdapter(AgentAdapter):
    """Adapter for Ollama local LLMs.

    Ollama runs models locally and exposes an OpenAI-compatible API.
    This adapter handles the request/response format automatically.

    Example:
        >>> adapter = OllamaAdapter(model="llama3.2")
        >>> trace = await adapter.execute("What is the capital of France?")
    """

    def __init__(
        self,
        model: str = "llama3.2",
        endpoint: str = "http://localhost:11434",
        timeout: float = 60.0,
        verbose: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Ollama adapter.

        Args:
            model: Ollama model name (e.g., llama3.2, mistral, codellama)
            endpoint: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
            model_config: Optional model configuration
        """
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.model_config = model_config or {}

    @property
    def name(self) -> str:
        return f"Ollama ({self.model})"

    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionTrace:
        """Execute a query against the Ollama model."""
        start_time = datetime.now()

        # Initialize tracer
        tracer = Tracer()

        # Build request
        url = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }

        # Start agent span
        async with tracer.start_span_async("Ollama Chat", SpanKind.AGENT):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                api_start = datetime.now()
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                api_end = datetime.now()
                api_latency = (api_end - api_start).total_seconds() * 1000

            # Extract response
            output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            finish_reason = data.get("choices", [{}])[0].get("finish_reason", "stop")

            # Build token usage
            token_usage = TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

            # Record LLM call span
            tracer.record_llm_call(
                model=self.model,
                provider="ollama",
                prompt=query,
                completion=output,
                prompt_tokens=token_usage.input_tokens,
                completion_tokens=token_usage.output_tokens,
                finish_reason=finish_reason,
                cost=0.0,  # Local = free
                duration_ms=api_latency,
            )

        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000

        # Create step trace (no tools for basic chat)
        step = StepTrace(
            step_id="1",
            step_name="chat",
            tool_name="chat",
            parameters={"query": query},
            output=output,
            success=True,
            metrics=StepMetrics(
                latency=latency_ms,
                cost=0.0,  # Local = free
                tokens=token_usage,
            ),
        )

        # Build trace context
        trace_context = tracer.build_trace_context()

        return ExecutionTrace(
            session_id=str(uuid.uuid4()),
            start_time=start_time,
            end_time=end_time,
            steps=[step],
            final_output=output,
            metrics=ExecutionMetrics(
                total_latency=latency_ms,
                total_cost=0.0,
                total_tokens=token_usage,
            ),
            trace_context=trace_context,
        )
