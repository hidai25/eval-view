"""ChatSession — provider-agnostic streaming chat for the assistant.

Holds conversation history, dispatches to the right LLM SDK based on the
selected provider, and tracks rough token counts for the chat UI.
Extracted from chat.py to keep that module focused on the interactive loop.
"""

import os
from typing import AsyncGenerator, Optional

from rich.console import Console

from evalview.chat_prompt import SYSTEM_PROMPT
from evalview.core.llm_provider import LLMProvider, PROVIDER_CONFIGS


class ChatSession:
    """Interactive chat session with EvalView assistant."""

    def __init__(
        self,
        provider: LLMProvider,
        model: Optional[str] = None,
        console: Optional[Console] = None,
    ):
        self.provider = provider
        self.model = model or PROVIDER_CONFIGS[provider].default_model
        self.console = console or Console()
        self.history: list[dict] = []
        self.total_tokens = 0
        self.last_tokens = 0

    async def stream_response(self, user_message: str) -> AsyncGenerator[str, None]:
        """Get a response from the LLM via streaming."""
        self.history.append({"role": "user", "content": user_message})

        collected_text = ""

        try:
            if self.provider == LLMProvider.OLLAMA:
                stream_gen = self._stream_ollama()
            elif self.provider == LLMProvider.OPENAI:
                stream_gen = self._stream_openai()
            elif self.provider == LLMProvider.ANTHROPIC:
                stream_gen = self._stream_anthropic()
            else:
                yield f"Provider {self.provider.value} not yet supported for chat."
                return

            async for chunk in stream_gen:
                if chunk:
                    collected_text += chunk
                    yield chunk

            # Update tokens estimate (very rough approximation for now as streams differ)
            tokens = len(collected_text) // 4
            self.last_tokens = tokens
            self.total_tokens += tokens

            self.history.append({"role": "assistant", "content": collected_text})

        except Exception as e:
            error_msg = f"\n\n[Error: {str(e)}]"
            yield error_msg
            self.history.append({"role": "assistant", "content": error_msg})

    async def _stream_ollama(self) -> AsyncGenerator[str, None]:
        """Stream chat using Ollama."""
        from openai import AsyncOpenAI

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{ollama_host}/v1",
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_openai(self) -> AsyncGenerator[str, None]:
        """Stream chat using OpenAI."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        # Newer OpenAI models require max_completion_tokens instead of max_tokens
        is_gpt5 = self.model.startswith("gpt-5")
        is_gpt4o = self.model.startswith("gpt-4o")
        is_o_series = self.model.startswith("o1") or self.model.startswith("o3") or self.model.startswith("o4")
        uses_max_completion_tokens = is_gpt5 or is_gpt4o or is_o_series

        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if is_gpt5:
            params["temperature"] = 1
        elif not is_o_series:
            params["temperature"] = 0.7

        if uses_max_completion_tokens:
            params["max_completion_tokens"] = 2000
        else:
            params["max_tokens"] = 2000

        stream = await client.chat.completions.create(**params)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_anthropic(self) -> AsyncGenerator[str, None]:
        """Stream chat using Anthropic."""
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        async with client.messages.stream(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=self.history,  # type: ignore[arg-type]
            temperature=0.7,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    # Keep old methods as simple aliases for backward compatibility if needed,
    # but they are not used in the new loop
    async def get_response(self, user_message: str) -> str:
        text = ""
        async for chunk in self.stream_response(user_message):
            text += chunk
        return text
