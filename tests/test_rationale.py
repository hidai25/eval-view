"""Tests for the RationaleCollector and its integration with adapters."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from typing import get_args

from evalview.core.rationale import (
    DECISION_TYPE_DESCRIPTIONS,
    RationaleCollector,
    compute_input_hash,
)
from evalview.core.types import (
    RATIONALE_MAX_EVENTS_PER_RUN,
    RATIONALE_MAX_TEXT_BYTES,
    DecisionType,
)


# ============================================================================
# DECISION_TYPE_DESCRIPTIONS — canonical plain-language mapping shared by
# local HTML replay, CI comments, and cloud UI tooltips. The keys must stay
# in lockstep with the DecisionType literal; otherwise some surfaces will
# silently render an empty tooltip when a new decision type ships.
# ============================================================================


class TestDecisionTypeDescriptions:
    def test_keys_match_decision_type_literal(self):
        assert set(DECISION_TYPE_DESCRIPTIONS.keys()) == set(get_args(DecisionType))

    def test_descriptions_are_nonempty_sentences(self):
        for key, desc in DECISION_TYPE_DESCRIPTIONS.items():
            assert desc and desc.strip(), f"{key} has empty description"
            assert desc.endswith("."), f"{key} description should end with a period"


# ============================================================================
# input_hash
# ============================================================================


class TestComputeInputHash:
    def test_deterministic(self):
        h1 = compute_input_hash(prompt="hello", tool_state={"a": 1, "b": 2})
        h2 = compute_input_hash(prompt="hello", tool_state={"a": 1, "b": 2})
        assert h1 == h2
        assert len(h1) == 64

    def test_key_order_insensitive(self):
        h1 = compute_input_hash(prompt="x", tool_state={"a": 1, "b": 2})
        h2 = compute_input_hash(prompt="x", tool_state={"b": 2, "a": 1})
        assert h1 == h2

    def test_different_prompt_different_hash(self):
        h1 = compute_input_hash(prompt="one")
        h2 = compute_input_hash(prompt="two")
        assert h1 != h2

    def test_none_prompt_is_empty(self):
        # None prompt is treated as empty string; doesn't raise
        h = compute_input_hash(prompt=None, tool_state=None)
        assert len(h) == 64

    def test_extra_affects_hash(self):
        h1 = compute_input_hash(prompt="x", tool_state={}, extra={"version": 1})
        h2 = compute_input_hash(prompt="x", tool_state={}, extra={"version": 2})
        assert h1 != h2


# ============================================================================
# Collector: basic capture
# ============================================================================


class TestCollectorCapture:
    def test_capture_populates_all_fields(self):
        c = RationaleCollector()
        ev = c.capture(
            step_id="s1",
            decision_type="tool_choice",
            chosen="search",
            input_hash="a" * 64,
            alternatives=["read", "edit"],
            rationale_text="need to look something up",
            turn=2,
            model_reported_confidence=0.9,
        )
        assert ev is not None
        assert ev.chosen == "search"
        assert ev.alternatives == ["read", "edit"]
        assert ev.rationale_text == "need to look something up"
        assert ev.turn == 2
        assert ev.model_reported_confidence == 0.9
        assert ev.truncated is False
        assert c.dropped() == 0
        assert len(c) == 1

    def test_capture_tool_choice_filters_chosen_from_alternatives(self):
        c = RationaleCollector()
        ev = c.capture_tool_choice(
            step_id="s1",
            chosen_tool="search",
            available_tools=["search", "read", "edit", ""],
            prompt="query",
            tool_state={"round": 1},
        )
        assert ev is not None
        assert "search" not in ev.alternatives
        assert "" not in ev.alternatives
        assert set(ev.alternatives) == {"read", "edit"}

    def test_capture_branch_populates_decision_type(self):
        c = RationaleCollector()
        ev = c.capture_branch(
            step_id="s1",
            chosen_branch="agent_a",
            available_branches=["agent_a", "agent_b", "agent_c"],
            state_summary={"pending_tasks": 2},
        )
        assert ev is not None
        assert ev.decision_type == "branch"
        assert ev.chosen == "agent_a"
        assert set(ev.alternatives) == {"agent_b", "agent_c"}

    def test_events_returns_copy(self):
        c = RationaleCollector()
        c.capture(
            step_id="s1", decision_type="refusal", chosen="decline",
            input_hash="b" * 64,
        )
        events = c.events()
        events.clear()  # Mutating returned list must not touch the collector.
        assert len(c.events()) == 1


# ============================================================================
# Collector: caps (truncation + drop)
# ============================================================================


class TestCollectorCaps:
    def test_rationale_text_truncated(self):
        c = RationaleCollector()
        oversize = "x" * (RATIONALE_MAX_TEXT_BYTES + 500)
        ev = c.capture(
            step_id="s1",
            decision_type="tool_choice",
            chosen="t",
            input_hash="c" * 64,
            rationale_text=oversize,
        )
        assert ev is not None
        assert ev.truncated is True
        assert len(ev.rationale_text.encode("utf-8")) <= RATIONALE_MAX_TEXT_BYTES

    def test_under_cap_not_truncated(self):
        c = RationaleCollector()
        ev = c.capture(
            step_id="s1",
            decision_type="tool_choice",
            chosen="t",
            input_hash="d" * 64,
            rationale_text="short",
        )
        assert ev is not None
        assert ev.truncated is False

    def test_none_text_stays_none(self):
        c = RationaleCollector()
        ev = c.capture(
            step_id="s1",
            decision_type="retry",
            chosen="retry",
            input_hash="e" * 64,
        )
        assert ev is not None
        assert ev.rationale_text is None
        assert ev.truncated is False

    def test_drops_past_cap(self, caplog):
        c = RationaleCollector()
        # Fill to cap
        for i in range(RATIONALE_MAX_EVENTS_PER_RUN):
            assert c.capture(
                step_id=f"s{i}",
                decision_type="tool_choice",
                chosen="t",
                input_hash="0" * 64,
            ) is not None
        # Overflow is silently dropped
        with caplog.at_level(logging.WARNING):
            dropped = c.capture(
                step_id="overflow",
                decision_type="tool_choice",
                chosen="t",
                input_hash="0" * 64,
            )
        assert dropped is None
        assert c.dropped() == 1
        assert len(c) == RATIONALE_MAX_EVENTS_PER_RUN
        assert any("hit cap" in r.message for r in caplog.records)

    def test_cap_warning_fires_once(self, caplog):
        c = RationaleCollector()
        for i in range(RATIONALE_MAX_EVENTS_PER_RUN):
            c.capture(
                step_id=f"s{i}",
                decision_type="tool_choice",
                chosen="t",
                input_hash="0" * 64,
            )
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                c.capture(
                    step_id="over",
                    decision_type="tool_choice",
                    chosen="t",
                    input_hash="0" * 64,
                )
        warnings = [r for r in caplog.records if "hit cap" in r.message]
        assert len(warnings) == 1
        assert c.dropped() == 5


# ============================================================================
# Adapter integration (smoke)
# ============================================================================


class TestAnthropicAdapterRationaleSmoke:
    """The Anthropic adapter should attach rationale_events to the trace.

    We stub the SDK so no real API call fires; the point is to prove the
    wiring, not to exercise the SDK.
    """

    @pytest.mark.asyncio
    async def test_thinking_block_becomes_rationale_text(self):
        from evalview.adapters.anthropic_adapter import AnthropicAdapter

        # Fake Anthropic response with one thinking + one tool_use block.
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "I should search for the answer."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search"
        tool_block.input = {"q": "capital of france"}
        tool_block.id = "tool-1"

        final_response = MagicMock()
        final_response.content = [thinking_block]
        final_response.model = "claude-sonnet-4-5-20250929"
        final_response.stop_reason = "end_turn"
        final_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        # Configure a response with both blocks for round 1, then a
        # text-only response to end the agent loop.
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Paris."
        end_response = MagicMock()
        end_response.content = [text_block]
        end_response.model = "claude-sonnet-4-5-20250929"
        end_response.stop_reason = "end_turn"
        end_response.usage = MagicMock(input_tokens=5, output_tokens=2)

        round1 = MagicMock()
        round1.content = [thinking_block, tool_block]
        round1.model = "claude-sonnet-4-5-20250929"
        round1.stop_reason = "tool_use"
        round1.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[round1, end_response])

        adapter = AnthropicAdapter(
            model="claude-sonnet-4-5-20250929",
            tools=[{"name": "search", "description": "", "input_schema": {}}],
            tool_executor=lambda name, args: "Paris",
        )

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            trace = await adapter.execute("What is the capital of France?")

        assert len(trace.rationale_events) == 1
        ev = trace.rationale_events[0]
        assert ev.decision_type == "tool_choice"
        assert ev.chosen == "search"
        assert ev.rationale_text == "I should search for the answer."
        assert ev.step_id == "tool-1"

    @pytest.mark.asyncio
    async def test_no_thinking_block_still_captures_tool_choice(self):
        from evalview.adapters.anthropic_adapter import AnthropicAdapter

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search"
        tool_block.input = {"q": "x"}
        tool_block.id = "tool-1"

        round1 = MagicMock()
        round1.content = [tool_block]
        round1.model = "claude-sonnet-4-5-20250929"
        round1.stop_reason = "tool_use"
        round1.usage = MagicMock(input_tokens=10, output_tokens=5)

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "done"
        end_response = MagicMock()
        end_response.content = [text_block]
        end_response.model = "claude-sonnet-4-5-20250929"
        end_response.stop_reason = "end_turn"
        end_response.usage = MagicMock(input_tokens=1, output_tokens=1)

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[round1, end_response])

        adapter = AnthropicAdapter(
            model="claude-sonnet-4-5-20250929",
            tools=[{"name": "search", "description": "", "input_schema": {}}],
            tool_executor=lambda name, args: "ok",
        )

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            trace = await adapter.execute("q")

        assert len(trace.rationale_events) == 1
        ev = trace.rationale_events[0]
        assert ev.chosen == "search"
        assert ev.rationale_text is None
