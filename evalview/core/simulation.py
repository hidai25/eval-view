"""Pre-deployment simulation harness.

The :class:`Simulator` wraps an :class:`~evalview.adapters.base.AgentAdapter`
and serves mocks for tool calls before the real ``tool_executor`` is
invoked. That keeps a simulated run hermetic — no network, no LLM
cost beyond the agent's own model calls — so CI can run large
what-if sweeps cheaply and deterministically.

Scope of the OSS v1 engine:

* **Tool mocks** — full support. A ``ToolMock`` matches on ``tool``
  (exact name) plus an optional ``match_params`` subset. Matching
  mocks return ``returns`` directly, simulate ``latency_ms``, and can
  raise via ``error``.
* **Response mocks / HTTP mocks** — recorded as "applied" when the
  adapter is given a chance to use them via ``install_mock_interceptor``,
  but the default path only wires them for adapters that explicitly opt
  in. Adding them is non-breaking for adapters that don't.
* **Variants** — ``run(variants=N)`` re-executes the test N times with
  the configured ``seed`` advanced per-variant so callers can cluster
  outcomes.

Cloud never runs simulations server-side; it only renders the
:class:`~evalview.core.types.SimulationResult` attached to
:class:`~evalview.core.types.EvaluationResult.simulation`.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from evalview.adapters.base import AgentAdapter
from evalview.core.cassette import (
    Cassette,
    RecordingToolExecutor,
    ReplayToolExecutor,
    new_cassette,
)
from evalview.core.types import (
    AppliedMock,
    BranchExploration,
    ExecutionTrace,
    HttpMock,
    MockSpec,
    ResponseMock,
    SimulationResult,
    TestCase,
    VariantOutcome,
)

logger = logging.getLogger(__name__)


ToolExecutor = Callable[[str, Dict[str, Any]], Any]


class UnmatchedMockError(RuntimeError):
    """Raised when ``MockSpec.strict`` is true and a call has no mock."""


class UninterceptableAdapterError(RuntimeError):
    """Raised when a hermetic-required run targets an adapter that has no
    interception seam (no ``tool_executor``, no ``install_mock_interceptor``).

    Without one of those hooks, mocks/cassettes/replay can't actually
    install — the run would silently hit live services. Failing fast
    is the only honest answer when the user has explicitly opted into
    hermetic semantics via ``--record``, ``--replay``, or
    ``MockSpec.strict=True``.
    """


def adapter_simulation_capability(adapter: AgentAdapter) -> Dict[str, bool]:
    """Report which simulation layers an adapter can actually intercept.

    Returns a dict with three booleans:
        tools     — has a ``tool_executor`` attribute the simulator can swap.
        responses — has ``install_mock_interceptor``, so it can pull
                    LLM-response mocks from the simulator.
        http      — declares ``supports_http_mocks=True`` (opt-in flag
                    adapters set when they wire ``http_mock_for`` into
                    their transport).

    The check is intentionally cheap and side-effect-free so callers
    can run it before every ``Simulator.run`` to decide whether to
    warn / raise / proceed.
    """
    return {
        "tools": hasattr(adapter, "tool_executor"),
        "responses": callable(getattr(adapter, "install_mock_interceptor", None)),
        "http": bool(getattr(adapter, "supports_http_mocks", False)),
    }


def _params_match(call_params: Dict[str, Any], match_params: Optional[Dict[str, Any]]) -> bool:
    """Subset-match: every key/value in ``match_params`` must equal the call."""
    if not match_params:
        return True
    for key, expected in match_params.items():
        if key not in call_params:
            return False
        if call_params[key] != expected:
            return False
    return True


@dataclass
class _MockHitCounter:
    """Tracks which mocks fired and how often, keyed on the matcher string."""

    hits: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def record(self, kind: str, matcher: str) -> None:
        key = (kind, matcher)
        self.hits[key] = self.hits.get(key, 0) + 1

    def to_applied(self) -> List[AppliedMock]:
        return [
            AppliedMock(kind=kind, matcher=matcher, count=count)  # type: ignore[arg-type]
            for (kind, matcher), count in self.hits.items()
        ]


class MockedToolExecutor:
    """Wraps a real tool_executor with a mock layer.

    Callable signature matches the adapter contract:
    ``executor(tool_name, params) -> result``. Accepts both sync and
    async wrapped executors — callers ``await`` via ``asyncio.to_thread``
    or ``asyncio.iscoroutine`` at the adapter layer as they do today.
    """

    def __init__(
        self,
        spec: MockSpec,
        real_executor: Optional[ToolExecutor],
        counter: _MockHitCounter,
        rng: random.Random,
    ) -> None:
        self._spec = spec
        self._real = real_executor
        self._counter = counter
        self._rng = rng

    def __call__(self, tool_name: str, params: Dict[str, Any]) -> Any:
        for mock in self._spec.tool_mocks:
            if mock.tool != tool_name:
                continue
            if not _params_match(params or {}, mock.match_params):
                continue
            self._counter.record("tool", mock.tool)
            if mock.latency_ms > 0:
                # Sleep is fine here — adapters already run tool
                # executors on a worker thread via asyncio.to_thread.
                time.sleep(mock.latency_ms / 1000.0)
            if mock.error:
                raise RuntimeError(mock.error)
            return mock.returns

        if self._spec.strict:
            raise UnmatchedMockError(
                f"No tool_mock matches call to '{tool_name}' and strict=True"
            )
        if self._real is None:
            # Non-strict but no real executor — return a placeholder so
            # the run completes instead of raising a TypeError in the
            # adapter. Surfaces clearly in the final output.
            logger.debug("Unmatched tool call '%s' with no real executor; returning None.", tool_name)
            return None
        return self._real(tool_name, params)


def _match_response_mock(prompt: str, mocks: List[ResponseMock]) -> Optional[ResponseMock]:
    for m in mocks:
        if m.regex:
            if re.search(m.match_prompt, prompt):
                return m
        elif m.match_prompt in prompt:
            return m
    return None


def _match_http_mock(url: str, method: Optional[str], mocks: List[HttpMock]) -> Optional[HttpMock]:
    for m in mocks:
        if m.method and method and m.method.upper() != method.upper():
            continue
        if m.regex:
            if re.search(m.url_pattern, url):
                return m
        elif m.url_pattern in url:
            return m
    return None


class Simulator:
    """Runs a test case through an adapter with mocks applied.

    Construct once per run. Use :meth:`run` for a single execution;
    use :meth:`run_variants` to fan out deterministic replays with
    advancing seeds.
    """

    def __init__(self, adapter: AgentAdapter, spec: MockSpec) -> None:
        self._adapter = adapter
        self._spec = spec

    # ------------------------------------------------------------------
    # Internal: capability gate
    # ------------------------------------------------------------------

    def _check_capability(
        self,
        *,
        replay: bool,
        record: bool,
        allow_live: bool,
    ) -> Dict[str, bool]:
        """Detect uninterceptable adapters before the run starts.

        Three escalation tiers:
          - Adapter has any interception seam → silent (capability is
            still recorded on the result for transparency).
          - No seam, but the user did not opt into hermetic semantics
            (no record/replay, ``strict=False``) → log a warning and
            proceed unless ``allow_live`` is set, in which case the
            warning drops to INFO.
          - No seam AND the user opted in (``record``, ``replay``, or
            ``MockSpec.strict``) → raise
            :class:`UninterceptableAdapterError` so the run fails fast
            instead of silently going live.
        """
        cap = adapter_simulation_capability(self._adapter)
        if any(cap.values()):
            return cap

        hermetic_required = record or replay or self._spec.strict
        adapter_name = getattr(self._adapter, "name", type(self._adapter).__name__)
        msg = (
            f"Adapter '{adapter_name}' has no tool interception seam "
            "(no `tool_executor`, no `install_mock_interceptor`, no "
            "`supports_http_mocks`). Mocks/cassette/replay cannot be "
            "installed — the run would hit live services. "
            "See docs/SIMULATE.md#adapter-support-matrix."
        )
        if hermetic_required:
            raise UninterceptableAdapterError(msg)
        if allow_live:
            logger.info(msg)
        else:
            logger.warning(msg)
        return cap

    # ------------------------------------------------------------------
    # Public helpers the CLI / adapters can import
    # ------------------------------------------------------------------

    def response_mock_for(self, prompt: str) -> Optional[ResponseMock]:
        """Public hook for adapters that opt into LLM-response mocking."""
        hit = _match_response_mock(prompt, self._spec.response_mocks)
        return hit

    def http_mock_for(self, url: str, method: Optional[str] = None) -> Optional[HttpMock]:
        """Public hook for adapters that opt into HTTP mocking."""
        hit = _match_http_mock(url, method, self._spec.http_mocks)
        return hit

    # ------------------------------------------------------------------
    # Internal: build the per-run executor stack
    # ------------------------------------------------------------------

    def _build_stack(
        self,
        rng: random.Random,
        counter: _MockHitCounter,
        replay_cassette: Optional[Cassette],
        record_into: Optional[List["RecordingToolExecutor"]],
    ) -> ToolExecutor:
        """Layer the executor stack: mocks → recorder → (replay | real).

        Order rationale:
        - Declarative mocks always win first so users can override a
          single cassette entry without re-recording the whole run.
        - The recorder sits *below* mocks so synthetic mock results
          never leak into a fresh cassette — only what the real layer
          (or replay layer) actually returned.
        - Replay, when present, replaces the real executor entirely;
          it falls through to the real executor on a miss only when
          ``MockSpec.strict`` is false.
        """
        real_executor = getattr(self._adapter, "tool_executor", None)

        underlying: Optional[ToolExecutor]
        if replay_cassette is not None:
            underlying = ReplayToolExecutor(
                replay_cassette,
                real=real_executor,
                strict=self._spec.strict,
            )
        else:
            underlying = real_executor

        if record_into is not None and underlying is not None:
            recorder = RecordingToolExecutor(real=underlying)
            record_into.append(recorder)
            underlying = recorder

        # When a replay cassette is installed, hermetic-strict semantics
        # belong to the cassette layer, not the mock layer — otherwise
        # the mock layer rejects any call that isn't explicitly mocked,
        # never giving the cassette a chance to serve it.
        mock_spec = self._spec
        if replay_cassette is not None and self._spec.strict:
            mock_spec = self._spec.model_copy(update={"strict": False})

        return MockedToolExecutor(mock_spec, underlying, counter, rng)

    # ------------------------------------------------------------------
    # Run entry points
    # ------------------------------------------------------------------

    async def run(
        self,
        test_case: TestCase,
        seed_override: Optional[int] = None,
        *,
        replay_cassette: Optional[Cassette] = None,
        record: bool = False,
        allow_live: bool = False,
    ) -> Tuple[ExecutionTrace, SimulationResult]:
        """Execute one simulated run and return (trace, SimulationResult).

        The simulator installs a :class:`MockedToolExecutor` as the
        adapter's ``tool_executor`` attribute (Python-adapter
        convention). The original executor is restored after
        execution. Unmatched calls fall through unless ``spec.strict``
        is true.

        When ``replay_cassette`` is provided, recorded tool results
        serve calls in their place (per-tool sequential matching).
        When ``record`` is true, the run captures every real tool call
        into ``SimulationResult.recorded_cassette`` so the caller can
        persist it with :func:`evalview.core.cassette.save_cassette`.

        Raises :class:`UninterceptableAdapterError` when the run requires
        hermetic semantics (``record``, ``replay_cassette``, or
        ``MockSpec.strict``) but the adapter exposes no interception
        seam. Otherwise an uninterceptable adapter logs a warning
        (downgraded to INFO when ``allow_live=True``) and proceeds.
        """
        capability = self._check_capability(
            replay=replay_cassette is not None,
            record=record,
            allow_live=allow_live,
        )

        counter = _MockHitCounter()
        seed = self._spec.seed if seed_override is None else seed_override
        rng = random.Random(seed)

        real_executor = getattr(self._adapter, "tool_executor", None)
        recorders: Optional[List[RecordingToolExecutor]] = [] if record else None
        mocked = self._build_stack(rng, counter, replay_cassette, recorders)

        # Install the mock layer via the adapter attribute rather than
        # the context dict — HTTP/streaming adapters JSON-serialize
        # context to send over the wire, which fails on callables.
        # Python-native adapters (Anthropic, OpenAI, Ollama, CrewAI
        # native) all read ``self.tool_executor`` with context as a
        # fallback, so this path covers them without any per-adapter
        # changes.
        context = dict(test_case.input.context or {})
        installer = getattr(self._adapter, "install_mock_interceptor", None)
        if callable(installer):
            try:
                installer(self)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("install_mock_interceptor raised: %s", exc)

        had_attr = hasattr(self._adapter, "tool_executor")
        if had_attr:
            setattr(self._adapter, "tool_executor", mocked)
        try:
            trace = await self._adapter.execute(test_case.input.query, context)
        finally:
            if had_attr:
                setattr(self._adapter, "tool_executor", real_executor)

        # A single (non-fan-out) run gets its path recorded as one
        # branch so the cloud UI always has something to render, even
        # when the agent only took the happy path.
        path = [f"{s.step_id}:{s.tool_name}" for s in trace.steps]
        recorded_cassette: Optional[Cassette] = None
        if recorders:
            recorded_cassette = new_cassette(test_case.name, adapter=self._adapter.name)
            for r in recorders:
                recorded_cassette.interactions.extend(r.interactions)
        result = SimulationResult(
            seed=seed,
            mocks_applied=counter.to_applied(),
            branches_explored=[
                BranchExploration(
                    branch_id="b0",
                    parent_branch_id=None,
                    decision_path=path,
                    final_output=trace.final_output,
                    passed=None,
                )
            ],
            variant_outcomes=[],
            recorded_cassette=recorded_cassette,
            adapter_capability=capability,
        )
        return trace, result

    async def run_variants(
        self,
        test_case: TestCase,
        variants: int,
        *,
        replay_cassette: Optional[Cassette] = None,
        record: bool = False,
        allow_live: bool = False,
    ) -> Tuple[List[ExecutionTrace], SimulationResult]:
        """Fan out ``variants`` deterministic replays and aggregate.

        Seeds advance by 1 per variant starting at ``spec.seed``. Each
        variant's tool path is recorded; ``variant_outcomes`` carries
        the final output and cost/latency so cloud can render a pass/
        fail matrix. Scoring is left to the evaluator downstream —
        the simulator only reports raw outcomes.

        ``replay_cassette`` and ``record`` behave the same as in
        :meth:`run`. When recording, the cassette captures the
        interactions from the *first* variant only — additional
        variants would overwrite each other and reproducibility is
        defined per-(test, seed) pair.
        """
        if variants < 1:
            raise ValueError("variants must be >= 1")

        capability = self._check_capability(
            replay=replay_cassette is not None,
            record=record,
            allow_live=allow_live,
        )

        traces: List[ExecutionTrace] = []
        branches: List[BranchExploration] = []
        outcomes: List[VariantOutcome] = []
        combined_counter = _MockHitCounter()
        recorded_cassette: Optional[Cassette] = None

        # Adapter is the same object across variants — resolve its
        # interception attributes once.
        real_executor = getattr(self._adapter, "tool_executor", None)
        installer = getattr(self._adapter, "install_mock_interceptor", None)
        had_attr = hasattr(self._adapter, "tool_executor")

        for i in range(variants):
            counter = _MockHitCounter()
            seed = (self._spec.seed or 0) + i
            rng = random.Random(seed)
            recorders: Optional[List[RecordingToolExecutor]] = (
                [] if (record and i == 0) else None
            )
            mocked = self._build_stack(rng, counter, replay_cassette, recorders)

            context: Dict[str, Any] = dict(test_case.input.context or {})
            if callable(installer):
                try:
                    installer(self)
                except Exception as exc:  # pragma: no cover
                    logger.warning("install_mock_interceptor raised: %s", exc)

            if had_attr:
                setattr(self._adapter, "tool_executor", mocked)
            try:
                trace = await self._adapter.execute(test_case.input.query, context)
            finally:
                if had_attr:
                    setattr(self._adapter, "tool_executor", real_executor)
            traces.append(trace)

            for (kind, matcher), count in counter.hits.items():
                combined_counter.hits[(kind, matcher)] = (
                    combined_counter.hits.get((kind, matcher), 0) + count
                )

            if recorders:
                recorded_cassette = new_cassette(test_case.name, adapter=self._adapter.name)
                for r in recorders:
                    recorded_cassette.interactions.extend(r.interactions)

            branch_id = f"b{i}"
            path = [f"{s.step_id}:{s.tool_name}" for s in trace.steps]
            branches.append(
                BranchExploration(
                    branch_id=branch_id,
                    parent_branch_id=None,
                    decision_path=path,
                    final_output=trace.final_output,
                    passed=None,
                )
            )
            outcomes.append(
                VariantOutcome(
                    variant_index=i,
                    branch_id=branch_id,
                    passed=True,  # refined by evaluator later
                    score=None,
                    total_cost=trace.metrics.total_cost,
                    total_latency_ms=trace.metrics.total_latency,
                )
            )

        result = SimulationResult(
            seed=self._spec.seed or 0,
            mocks_applied=combined_counter.to_applied(),
            branches_explored=branches,
            variant_outcomes=outcomes,
            recorded_cassette=recorded_cassette,
            adapter_capability=capability,
        )
        return traces, result


__all__ = [
    "MockedToolExecutor",
    "Simulator",
    "UninterceptableAdapterError",
    "UnmatchedMockError",
    "adapter_simulation_capability",
]
