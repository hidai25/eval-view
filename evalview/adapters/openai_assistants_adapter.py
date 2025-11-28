"""OpenAI Assistants API adapter for EvalView.

Supports testing OpenAI Assistants with proper step tracking.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class OpenAIAssistantsAdapter(AgentAdapter):
    """Adapter for OpenAI Assistants API.

    Requires:
    - OPENAI_API_KEY environment variable
    - assistant_id in context or configured
    """

    def __init__(
        self,
        assistant_id: Optional[str] = None,
        timeout: float = 120.0,
        verbose: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.assistant_id = assistant_id
        self.timeout = timeout
        self.verbose = verbose
        self.model_config = model_config or {}

    @property
    def name(self) -> str:
        return "openai-assistants"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute OpenAI Assistant and capture trace."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")

        context = context or {}
        # Check context, then adapter config, then environment variable
        assistant_id = context.get("assistant_id") or self.assistant_id or os.getenv("OPENAI_ASSISTANT_ID")

        if not assistant_id:
            raise ValueError(
                "assistant_id required. Set OPENAI_ASSISTANT_ID env var, "
                "add to config, or include in test case context"
            )

        start_time = datetime.now()

        if self.verbose:
            logger.info(f"🚀 Executing OpenAI Assistant: {query}...")
            logger.debug(f"Assistant ID: {assistant_id}")

        client = AsyncOpenAI()

        # Create thread
        thread = await client.beta.threads.create()

        # Add message
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query,
        )

        # Run assistant
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

        # Poll for completion
        max_wait = self.timeout
        waited = 0
        poll_interval = 0.5

        while run.status in ["queued", "in_progress", "requires_action"]:
            if waited >= max_wait:
                raise TimeoutError(f"Assistant run exceeded timeout of {max_wait}s")

            await asyncio.sleep(poll_interval)
            waited += poll_interval

            run = await client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )

            if self.verbose and run.status == "in_progress":
                logger.debug(f"⏳ Run status: {run.status}")

        if run.status != "completed":
            error_msg = f"Run failed with status: {run.status}"
            if run.last_error:
                error_msg += f" - {run.last_error.message}"
            raise RuntimeError(error_msg)

        # Extract steps
        steps = await self._extract_steps(client, thread.id, run.id)

        # Get final message
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        final_output = ""
        if messages.data:
            for content in messages.data[0].content:
                if content.type == "text":
                    final_output += content.text.value

        end_time = datetime.now()

        # Calculate metrics from run
        metrics = self._calculate_metrics(run, steps, start_time, end_time)

        if self.verbose:
            logger.info(f"✅ Assistant completed in {metrics.total_latency:.0f}ms")

        return ExecutionTrace(
            session_id=thread.id,
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=metrics,
        )

    async def _extract_steps(self, client, thread_id: str, run_id: str) -> List[StepTrace]:
        """Extract steps from run."""
        steps = []

        # Get run steps
        run_steps = await client.beta.threads.runs.steps.list(
            thread_id=thread_id,
            run_id=run_id,
        )

        for i, step in enumerate(run_steps.data):
            if step.type == "tool_calls":
                # Extract tool calls
                for tool_call in step.step_details.tool_calls:
                    if tool_call.type == "function":
                        step_trace = StepTrace(
                            step_id=tool_call.id,
                            step_name=tool_call.function.name,
                            tool_name=tool_call.function.name,
                            parameters=(
                                json.loads(tool_call.function.arguments)
                                if tool_call.function.arguments
                                else {}
                            ),
                            output=(
                                tool_call.function.output
                                if hasattr(tool_call.function, "output")
                                else None
                            ),
                            success=True,
                            metrics=StepMetrics(latency=0.0, cost=0.0),
                        )
                        steps.append(step_trace)

                    elif tool_call.type == "code_interpreter":
                        step_trace = StepTrace(
                            step_id=tool_call.id,
                            step_name="Code Interpreter",
                            tool_name="code_interpreter",
                            parameters={"input": tool_call.code_interpreter.input},
                            output="\n".join(
                                [log.get("text", "") for log in tool_call.code_interpreter.outputs]
                            ),
                            success=True,
                            metrics=StepMetrics(latency=0.0, cost=0.0),
                        )
                        steps.append(step_trace)

                    elif tool_call.type == "retrieval":
                        step_trace = StepTrace(
                            step_id=tool_call.id,
                            step_name="File Search",
                            tool_name="retrieval",
                            parameters={},
                            output=None,
                            success=True,
                            metrics=StepMetrics(latency=0.0, cost=0.0),
                        )
                        steps.append(step_trace)

            elif step.type == "message_creation":
                # Message creation step
                step_trace = StepTrace(
                    step_id=step.id,
                    step_name="Message Creation",
                    tool_name="message_creation",
                    parameters={},
                    output=None,
                    success=True,
                    metrics=StepMetrics(latency=0.0, cost=0.0),
                )
                steps.append(step_trace)

        return steps

    def _calculate_metrics(
        self, run, steps: List[StepTrace], start_time: datetime, end_time: datetime
    ) -> ExecutionMetrics:
        """Calculate execution metrics from run."""
        total_latency = (end_time - start_time).total_seconds() * 1000

        # OpenAI provides usage - convert to TokenUsage object
        token_usage = None
        if hasattr(run, "usage") and run.usage:
            token_usage = TokenUsage(
                input_tokens=getattr(run.usage, "prompt_tokens", 0),
                output_tokens=getattr(run.usage, "completion_tokens", 0),
                cached_tokens=0,
            )

        # Estimate cost based on model and tokens
        # This is approximate - adjust pricing as needed
        total_cost = 0.0
        if token_usage and hasattr(run, "model"):
            model = run.model
            total_token_count = token_usage.total_tokens
            # Rough estimates - update with actual pricing
            if "gpt-4" in model:
                total_cost = (total_token_count / 1000) * 0.03  # $0.03 per 1K tokens
            elif "gpt-3.5" in model:
                total_cost = (total_token_count / 1000) * 0.002  # $0.002 per 1K tokens

        return ExecutionMetrics(
            total_cost=total_cost,
            total_latency=total_latency,
            total_tokens=token_usage,
        )

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            # Try to list assistants
            await client.beta.assistants.list(limit=1)
            return True
        except Exception:
            return False
