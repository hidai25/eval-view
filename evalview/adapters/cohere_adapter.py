import os
import logging
from datetime import datetime
from typing import Any, Optional, Dict

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


class CohereAdapter(AgentAdapter):
    """Adapter for Cohere API models."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            import cohere as _cohere
        except ImportError:
            raise ImportError(
                "The 'cohere' package is required for CohereAdapter. "
                "Install it with: pip install cohere"
            )
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key is required. Set COHERE_API_KEY in your environment or .env.local.")

        # Use AsyncClientV2 for non-blocking calls
        self.client = _cohere.AsyncClientV2(api_key=self.api_key)
        self.model = model or "command-r-plus-08-2024"

    @property
    def name(self) -> str:
        return "cohere"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute conversation and capture trace."""
        start_time = datetime.now()
        
        # 1. Initialize tracer
        tracer = Tracer()

        async with tracer.start_span_async("Cohere Agent", SpanKind.AGENT):
            api_start = datetime.now()
            
            # Send request to Cohere
            response = await self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": query}]
            )
            
            api_end = datetime.now()
            api_latency = (api_end - api_start).total_seconds() * 1000

            # Extract AI's response text
            final_answer = response.message.content[0].text 

            # 2. Extract Token Usage (Changes for compatibility with V1 and V2 SDKs)
            input_tokens = 0
            output_tokens = 0
            try:
                # Try the V2 version's usage (usage)
                if hasattr(response, 'usage') and response.usage:
                    tokens_obj = getattr(response.usage, 'tokens', None)
                    if tokens_obj:
                        input_tokens = getattr(tokens_obj, 'input_tokens', 0)
                        output_tokens = getattr(tokens_obj, 'output_tokens', 0)
                # Try the older version's usage (meta)
                elif hasattr(response, 'meta') and response.meta:
                    tokens_obj = getattr(response.meta, 'tokens', None)
                    if tokens_obj:
                        input_tokens = getattr(tokens_obj, 'input_tokens', 0)
                        output_tokens = getattr(tokens_obj, 'output_tokens', 0)
            except Exception as e:
                # If neither is found, don't crash the program, just log it
                logger.debug(f"Could not extract tokens: {e}")

            # 3. Calculate Cost
            total_cost = calculate_cost(
                model_name=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=0
            )

            # 4. Record the specific LLM call to tracer
            tracer.record_llm_call(
                model=self.model,
                provider="cohere",
                prompt=query,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cost=total_cost,
                duration_ms=api_latency,
            )

        end_time = datetime.now()

        # 5. Build Final Trace 
        trace = ExecutionTrace(
            session_id=f"cohere-session-{int(start_time.timestamp())}",
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
                ) if (input_tokens > 0 or output_tokens > 0) else None,
            ),
            model_id=self.model,
            model_provider="cohere"
        )

        trace.trace_context = tracer.build_trace_context()
        return trace

    async def health_check(self) -> bool:
        """Check if the Cohere endpoint and API key are valid."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
