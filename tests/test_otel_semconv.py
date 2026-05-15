"""Tests for the OTel semantic-convention spec.

The constants in ``evalview.core.otel_semconv`` are the ONLY public
contract — adapters import them by name. These tests pin the spec so a
rename, drop, or version bump is always a deliberate, reviewable change
rather than something that silently breaks every adopter's tests.
"""
from __future__ import annotations

import re

from evalview.core import otel_semconv


# ---------------------------------------------------------------------------
# Stability pins
# ---------------------------------------------------------------------------


class TestSpecVersion:
    def test_version_is_semver_shaped(self) -> None:
        # Spec version is the consumer's only signal that something
        # breaking changed. Keeping it semver-shaped means consumers can
        # branch on major.minor without inventing a parser.
        assert re.match(r"^\d+\.\d+\.\d+$", otel_semconv.OTEL_SEMCONV_VERSION)


class TestSpanNames:
    def test_every_span_constant_is_in_the_index(self) -> None:
        # If you add a new SPAN_* constant, you MUST add it to SPAN_NAMES.
        # Otherwise consumers iterating SPAN_NAMES silently miss it.
        module_constants = {
            value for name, value in vars(otel_semconv).items()
            if name.startswith("SPAN_") and isinstance(value, str)
        }
        assert module_constants == set(otel_semconv.SPAN_NAMES)

    def test_span_names_follow_agent_dot_verb(self) -> None:
        # Pin the naming convention so the spec stays grep-able and
        # alphabetically grouped in any UI.
        for name in otel_semconv.SPAN_NAMES:
            assert name.startswith("agent."), name

    def test_no_duplicate_span_names(self) -> None:
        assert len(otel_semconv.SPAN_NAMES) == len(set(otel_semconv.SPAN_NAMES))


class TestAttributes:
    def test_every_attribute_constant_is_in_the_index(self) -> None:
        # Same contract as SPAN_NAMES — adding ATTR_* without adding to
        # ATTRIBUTES means consumers iterating the index miss the new key.
        module_constants = {
            value for name, value in vars(otel_semconv).items()
            if name.startswith("ATTR_") and isinstance(value, str)
        }
        assert module_constants == set(otel_semconv.ATTRIBUTES)

    def test_attributes_namespaced_under_agent_or_well_known(self) -> None:
        # Every attribute must either live under our ``agent.*`` namespace
        # OR be a deliberate borrow from upstream OTel (none currently).
        # This test enforces the "don't fragment the ecosystem" rule.
        for attr in otel_semconv.ATTRIBUTES:
            assert attr.startswith("agent."), attr

    def test_no_duplicate_attribute_names(self) -> None:
        assert len(otel_semconv.ATTRIBUTES) == len(set(otel_semconv.ATTRIBUTES))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    def test_is_known_span_round_trips(self) -> None:
        for name in otel_semconv.SPAN_NAMES:
            assert otel_semconv.is_known_span(name)

    def test_is_known_span_rejects_unknown(self) -> None:
        # Strict on purpose — adapter authors must add new spans to the
        # spec before emitting them, not invent in-line.
        assert not otel_semconv.is_known_span("agent.something_not_in_spec")
        assert not otel_semconv.is_known_span("")

    def test_is_known_attribute_round_trips(self) -> None:
        for attr in otel_semconv.ATTRIBUTES:
            assert otel_semconv.is_known_attribute(attr)

    def test_is_known_attribute_rejects_unknown(self) -> None:
        assert not otel_semconv.is_known_attribute("agent.invented")


class TestAttributesForSpan:
    def test_every_recommendation_is_a_known_attribute(self) -> None:
        # The recommendations table itself must not drift away from the
        # ATTR_* constants — a typo here would tell adapter authors to
        # emit attributes the spec doesn't define.
        for span in otel_semconv.SPAN_NAMES:
            recommended = otel_semconv.attributes_for_span(span)
            for attr in recommended:
                assert otel_semconv.is_known_attribute(attr), (
                    f"{span} recommends unknown attr {attr!r}"
                )

    def test_unknown_span_returns_only_base_identity(self) -> None:
        # Defensive: if a caller passes a span we don't know, return the
        # identity attributes anyway so they always have something to
        # stamp. Avoids forcing every caller to wrap in try/except.
        result = otel_semconv.attributes_for_span("agent.something_unknown")
        assert otel_semconv.ATTR_AGENT_NAME in result
        assert otel_semconv.ATTR_AGENT_RUN_ID in result

    def test_tool_call_recommends_parameters_fingerprint(self) -> None:
        # Pin the "fingerprint, don't dump" guidance for the most-emitted
        # span: tool calls should hash params, not log them. If this
        # recommendation goes away, downstream adapters will start
        # logging full payloads and bills will spike.
        rec = otel_semconv.attributes_for_span(otel_semconv.SPAN_AGENT_TOOL_CALL)
        assert otel_semconv.ATTR_TOOL_PARAMETERS_FINGERPRINT in rec
