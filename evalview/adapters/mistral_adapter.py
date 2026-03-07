import os
import logging
from datetime import datetime
from typing import Any, Optional, Dict

from mistralai import Mistral

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionTrace,
    ExecutionMetrics,
    TokenUsage,
    SpanKind,
)
from evalview.core.pricing import calculate_cost
from evalview.core.tracing import Tracer

logger = logging.getLogger(__name__)

class MistralAdapter(AgentAdapter):
    """Adapter for Mistral agents."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set.")

        # Initialize the toolkit: pass the API key to the Mistral client
        self.client = Mistral(api_key=self.api_key)

        self.model = model or "mistral-large-latest"  # Default to a specific Mistral model

    @property
    def name(self) -> str:
        return "mistral"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute conversation and capture trace."""
        start_time = datetime.now()
        
        # Initialize tracer (monitor)
        tracer = Tracer()

        async with tracer.start_span_async("Mistral Agent", SpanKind.AGENT):
            api_start = datetime.now()
            
            # Send request to Mistral
            response = await self.client.chat.complete_async(
                model=self.model,
                messages=[{"role": "user", "content": query}]
            )
            
            api_end = datetime.now()
            api_latency = (api_end - api_start).total_seconds() * 1000

            # Extract AI's response text and token usage
            final_answer = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Calculate Cost
            total_cost = calculate_cost(
                model_name=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=0
            )

            # Record the specific LLM call to tracer
            tracer.record_llm_call(
                model=self.model,
                provider="mistral",
                prompt=query,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cost=total_cost,
                duration_ms=api_latency,
            )

        end_time = datetime.now()

        # Build Final Trace
        trace = ExecutionTrace(
            session_id=f"mistral-session-{int(start_time.timestamp())}",
            start_time=start_time,
            end_time=end_time,
            final_output=final_answer,
            steps=[],
            metrics=ExecutionMetrics(
                total_cost=total_cost,
                total_latency=(end_time - start_time).total_seconds() * 1000,
                total_tokens=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=0
                )
            ),
            model_id=self.model,
            model_provider="mistral"
        )

        trace.trace_context = tracer.build_trace_context()
        return trace

    async def health_check(self) -> bool:
        """Check if the Mistral endpoint and API key are valid."""
        try:
            # Try to fetch the model list from the Mistral server
            await self.client.models.list_async()
            return True
        except Exception as e:
            # If an error occurs, print the error message and return False
            logger.error(f"Mistral health check failed: {e}")
            return False