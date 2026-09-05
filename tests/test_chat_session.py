"""Provider errors must remain failures when chat is used as an agent backend."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evalview.chat_session import ChatSession
from evalview.core.llm_provider import LLMProvider


@pytest.mark.asyncio
async def test_strict_stream_rolls_back_failed_turn_and_preserves_provider_error(monkeypatch):
    session = ChatSession(LLMProvider.OPENAI)
    previous = [{"role": "user", "content": "previous"}, {"role": "assistant", "content": "ok"}]
    session.history = previous.copy()
    failure = RuntimeError("provider quota exhausted")

    async def broken_stream():
        yield "partial answer"
        raise failure

    monkeypatch.setattr(session, "_stream_openai", broken_stream)
    chunks = []
    with pytest.raises(RuntimeError) as caught:
        async for chunk in session.stream_response("new question", raise_errors=True):
            chunks.append(chunk)

    assert caught.value is failure
    assert chunks == ["partial answer"]
    assert session.history == previous
    assert session.last_tokens == session.total_tokens == 0


@pytest.mark.asyncio
async def test_interactive_stream_keeps_displayable_error(monkeypatch):
    session = ChatSession(LLMProvider.OPENAI)

    async def broken_stream():
        raise RuntimeError("provider unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr(session, "_stream_openai", broken_stream)
    reply = await session.get_response("hello")
    assert "[Error: provider unavailable]" in reply
    assert session.history[-1] == {"role": "assistant", "content": reply}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [LLMProvider.OPENAI, LLMProvider.OLLAMA, LLMProvider.ANTHROPIC])
@pytest.mark.parametrize("fails", [False, True])
async def test_clients_and_streams_close_before_event_loop_finishes(monkeypatch, provider, fails):
    async def chunks():
        if fails:
            raise RuntimeError("stream failed")
        if provider == LLMProvider.ANTHROPIC:
            yield "hello"
        else:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))])

    stream = AsyncMock()
    stream.__aenter__.return_value = stream
    stream.__aiter__.side_effect = chunks
    stream.text_stream = chunks() if provider == LLMProvider.ANTHROPIC else None
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.chat.completions.create.return_value = stream
    if provider == LLMProvider.ANTHROPIC:
        client.messages = SimpleNamespace(stream=lambda **kwargs: stream)
        monkeypatch.setattr("anthropic.AsyncAnthropic", lambda **kwargs: client)
    else:
        monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: client)

    session = ChatSession(provider)
    if fails:
        with pytest.raises(RuntimeError, match="stream failed"):
            _ = [chunk async for chunk in session.stream_response("hello", raise_errors=True)]
        assert session.history == []
    else:
        assert "".join([chunk async for chunk in session.stream_response("hello")]) == "hello"
        assert session.history[-1]["content"] == "hello"

    stream.__aexit__.assert_awaited_once()
    client.__aexit__.assert_awaited_once()
