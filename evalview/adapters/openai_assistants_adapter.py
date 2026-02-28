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
    SpanKind,
)
from evalview.core.tracing import Tracer

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
            # Try to auto-create with user confirmation
            assistant_id = await self._auto_create_assistant(client=None)
            if not assistant_id:
                raise ValueError(
                    "assistant_id required. Set OPENAI_ASSISTANT_ID env var, "
                    "add to config, or include in test case context"
                )

        start_time = datetime.now()

        # Initialize tracer for detailed span capture
        tracer = Tracer()

        if self.verbose:
            logger.info(f"🚀 Executing OpenAI Assistant: {query}...")
            logger.debug(f"Assistant ID: {assistant_id}")

        client = AsyncOpenAI()

        # Start agent-level span
        async with tracer.start_span_async("Assistant Execution", SpanKind.AGENT):
            # Create thread
            thread = await client.beta.threads.create()

            # Add message
            await client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query,
            )

            # Run assistant
            run_start = datetime.now()
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

            run_end = datetime.now()
            run_duration = (run_end - run_start).total_seconds() * 1000

            if run.status != "completed":
                error_msg = f"Run failed with status: {run.status}"
                if run.last_error:
                    error_msg += f" - {run.last_error.message}"
                raise RuntimeError(error_msg)

            # Record LLM call span
            model_name = run.model if hasattr(run, "model") else "gpt-4o"
            input_tokens = run.usage.prompt_tokens if hasattr(run, "usage") and run.usage else 0
            output_tokens = run.usage.completion_tokens if hasattr(run, "usage") and run.usage else 0

            # Calculate cost
            token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
            llm_cost = self._calculate_llm_cost(model_name, token_usage)

            tracer.record_llm_call(
                model=model_name,
                provider="openai",
                prompt=query,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                finish_reason="completed",
                cost=llm_cost,
                duration_ms=run_duration,
            )

            # Extract steps and record tool spans
            steps = await self._extract_steps_with_tracing(client, thread.id, run.id, tracer)

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

        # Build trace context
        trace_context = tracer.build_trace_context()

        if self.verbose:
            logger.info(f"✅ Assistant completed in {metrics.total_latency:.0f}ms")

        return ExecutionTrace(
            session_id=thread.id,
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=metrics,
            trace_context=trace_context,
        )

    async def _extract_steps(self, client, thread_id: str, run_id: str) -> List[StepTrace]:
        """Extract steps from run with actual timing from OpenAI."""
        return await self._extract_steps_with_tracing(client, thread_id, run_id, None)

    async def _extract_steps_with_tracing(
        self, client, thread_id: str, run_id: str, tracer: Optional[Tracer]
    ) -> List[StepTrace]:
        """Extract steps from run with actual timing and optional tracing."""
        steps = []

        # Get run steps
        run_steps = await client.beta.threads.runs.steps.list(
            thread_id=thread_id,
            run_id=run_id,
        )

        for i, step in enumerate(run_steps.data):
            # Calculate actual step latency from timestamps
            step_latency = 0.0
            if hasattr(step, 'created_at') and hasattr(step, 'completed_at'):
                if step.created_at and step.completed_at:
                    step_latency = (step.completed_at - step.created_at) * 1000  # ms

            if step.type == "tool_calls":
                # Extract tool calls
                for tool_call in step.step_details.tool_calls:
                    if tool_call.type == "function":
                        tool_name = tool_call.function.name
                        parameters = (
                            json.loads(tool_call.function.arguments)
                            if tool_call.function.arguments
                            else {}
                        )
                        output = (
                            tool_call.function.output
                            if hasattr(tool_call.function, "output")
                            else None
                        )

                        step_trace = StepTrace(
                            step_id=tool_call.id,
                            step_name=tool_name,
                            tool_name=tool_name,
                            parameters=parameters,
                            output=output,
                            success=True,
                            metrics=StepMetrics(latency=step_latency, cost=0.0),
                        )
                        steps.append(step_trace)

                        # Record tool span
                        if tracer:
                            tracer.record_tool_call(
                                tool_name=tool_name,
                                parameters=parameters,
                                result=output,
                                duration_ms=step_latency,
                            )

                    elif tool_call.type == "code_interpreter":
                        tool_name = "code_interpreter"
                        parameters = {"input": tool_call.code_interpreter.input}
                        output = "\n".join(
                            [log.get("text", "") for log in tool_call.code_interpreter.outputs]
                        )

                        step_trace = StepTrace(
                            step_id=tool_call.id,
                            step_name="Code Interpreter",
                            tool_name=tool_name,
                            parameters=parameters,
                            output=output,
                            success=True,
                            metrics=StepMetrics(latency=step_latency, cost=0.0),
                        )
                        steps.append(step_trace)

                        # Record tool span
                        if tracer:
                            tracer.record_tool_call(
                                tool_name=tool_name,
                                parameters=parameters,
                                result=output,
                                duration_ms=step_latency,
                            )

                    elif tool_call.type == "retrieval":
                        tool_name = "retrieval"
                        step_trace = StepTrace(
                            step_id=tool_call.id,
                            step_name="File Search",
                            tool_name=tool_name,
                            parameters={},
                            output=None,
                            success=True,
                            metrics=StepMetrics(latency=step_latency, cost=0.0),
                        )
                        steps.append(step_trace)

                        # Record tool span
                        if tracer:
                            tracer.record_tool_call(
                                tool_name=tool_name,
                                parameters={},
                                result=None,
                                duration_ms=step_latency,
                            )

            # Skip message_creation - it's an internal step, not a user-facing tool
            # Users shouldn't need to expect this in their test cases

        return steps

    def _calculate_llm_cost(self, model: str, token_usage: TokenUsage) -> float:
        """Calculate cost for OpenAI LLM call."""
        input_tokens = token_usage.input_tokens
        output_tokens = token_usage.output_tokens

        # GPT-4o pricing (per 1M tokens)
        if "gpt-4o" in model:
            return (input_tokens / 1_000_000) * 2.50 + (output_tokens / 1_000_000) * 10.00
        # GPT-4 Turbo pricing
        elif "gpt-4-turbo" in model or "gpt-4-1106" in model:
            return (input_tokens / 1_000_000) * 10.00 + (output_tokens / 1_000_000) * 30.00
        # GPT-4 pricing
        elif "gpt-4" in model:
            return (input_tokens / 1_000_000) * 30.00 + (output_tokens / 1_000_000) * 60.00
        # GPT-3.5 Turbo pricing
        elif "gpt-3.5" in model:
            return (input_tokens / 1_000_000) * 0.50 + (output_tokens / 1_000_000) * 1.50
        return 0.0

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

        # Calculate cost based on model and tokens (2024-2025 pricing)
        total_cost = 0.0
        if token_usage and hasattr(run, "model"):
            model = run.model
            input_tokens = token_usage.input_tokens
            output_tokens = token_usage.output_tokens

            # GPT-4o pricing (per 1M tokens)
            if "gpt-4o" in model:
                total_cost = (input_tokens / 1_000_000) * 2.50 + (output_tokens / 1_000_000) * 10.00
            # GPT-4 Turbo pricing
            elif "gpt-4-turbo" in model or "gpt-4-1106" in model:
                total_cost = (input_tokens / 1_000_000) * 10.00 + (output_tokens / 1_000_000) * 30.00
            # GPT-4 pricing
            elif "gpt-4" in model:
                total_cost = (input_tokens / 1_000_000) * 30.00 + (output_tokens / 1_000_000) * 60.00
            # GPT-3.5 Turbo pricing
            elif "gpt-3.5" in model:
                total_cost = (input_tokens / 1_000_000) * 0.50 + (output_tokens / 1_000_000) * 1.50

        # Distribute cost proportionally based on each step's latency
        if steps and total_cost > 0:
            total_step_latency = sum(s.metrics.latency for s in steps)
            if total_step_latency > 0:
                for step in steps:
                    # Cost proportional to time spent
                    step.metrics.cost = total_cost * (step.metrics.latency / total_step_latency)
            else:
                # Fallback: distribute evenly if no timing data
                per_step_cost = total_cost / len(steps)
                for step in steps:
                    step.metrics.cost = per_step_cost

        return ExecutionMetrics(
            total_cost=total_cost,
            total_latency=total_latency,
            total_tokens=token_usage,
        )

    async def _auto_create_assistant(self, client=None) -> Optional[str]:
        """Auto-create an assistant with user confirmation.

        Returns:
            The created assistant_id or None if user declined
        """
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console()

        console.print("\n[yellow]No OpenAI Assistant ID found.[/yellow]")
        console.print("\nWould you like to create one automatically?")
        console.print("[dim]This will create an assistant with code_interpreter tool for testing.[/dim]\n")

        if not Confirm.ask("[bold]Create assistant?[/bold]", default=True):
            console.print("[dim]Skipped. Set OPENAI_ASSISTANT_ID manually to continue.[/dim]")
            return None

        try:
            from openai import AsyncOpenAI

            if client is None:
                client = AsyncOpenAI()

            console.print("\n[dim]Creating assistant...[/dim]")

            # Create assistant with useful default tools
            assistant = await client.beta.assistants.create(
                name="EvalView Test Assistant",
                instructions="You are a helpful assistant for testing. Use tools when appropriate to answer questions accurately.",
                model=self.model_config.get("name", "gpt-4o"),
                tools=[
                    {"type": "code_interpreter"},  # For calculations, data analysis
                ],
            )

            assistant_id = assistant.id

            # Save to .env.local
            self._save_assistant_id(assistant_id)

            console.print(f"[green]✓ Created assistant: {assistant_id}[/green]")
            console.print("[dim]Saved to .env.local[/dim]\n")

            # Update environment for current session
            os.environ["OPENAI_ASSISTANT_ID"] = assistant_id
            self.assistant_id = assistant_id

            return assistant_id

        except Exception as e:
            console.print(f"[red]Failed to create assistant: {e}[/red]")
            return None

    def _save_assistant_id(self, assistant_id: str) -> None:
        """Save assistant ID to .env.local file."""
        env_file = ".env.local"
        line = f"OPENAI_ASSISTANT_ID={assistant_id}\n"

        # Read existing content
        existing_lines = []
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                existing_lines = f.readlines()

        # Remove existing OPENAI_ASSISTANT_ID line if present
        new_lines = [l for l in existing_lines if not l.startswith("OPENAI_ASSISTANT_ID=")]

        # Ensure last line ends with newline
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        # Add new assistant ID
        new_lines.append(line)

        # Write back
        with open(env_file, "w") as f:
            f.writelines(new_lines)

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
