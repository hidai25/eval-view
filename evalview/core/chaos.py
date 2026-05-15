"""Chaos injection — controlled disruption for agent simulation.

The eval-discourse complaint: *"static prompts/tasks don't simulate real
users — goal drift, info drift, interruptions, tool flakiness."* This
module gives ``evalview simulate`` a vocabulary for those disruptions,
deterministic enough that a CI run can repeat them and an LLM judge can
reason about them.

Pure data + a deterministic injector. The actual wiring into
``evalview simulate`` happens via a small orchestrator that takes a
:class:`ChaosScenario` and a callback per disruption — see
``docs/agent-recipes/add-chaos-mode.md`` for how to add a new disruption.

Modes shipped in v0:

- **tool_failure**: a designated tool fails (or returns a designated
  error payload) on the Nth invocation.
- **latency_spike**: the Nth tool call sleeps an extra ``delay_ms``
  before returning.
- **goal_interruption**: between two designated steps, a synthetic
  "user message" is injected that asks for something else.

The seed makes runs reproducible — every chaos decision is derivable
from ``(scenario, seed, step_index)`` so the same suite + seed produces
the same disruptions every time.

Contributor recipe: ``docs/agent-recipes/add-chaos-mode.md``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Mode constants ──────────────────────────────────────────────────────────

MODE_TOOL_FAILURE: str = "tool_failure"
MODE_LATENCY_SPIKE: str = "latency_spike"
MODE_GOAL_INTERRUPTION: str = "goal_interruption"

# Public registry. Append, don't reorder, when adding modes — preserves
# ordering for serialized scenarios.
SHIPPED_MODES: Tuple[str, ...] = (
    MODE_TOOL_FAILURE,
    MODE_LATENCY_SPIKE,
    MODE_GOAL_INTERRUPTION,
)


# ── Data shapes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChaosDisruption:
    """One disruption planned for the trajectory.

    Frozen + hashable so a scenario plan can be cached / compared
    cleanly across runs.
    """

    mode: str
    step_index: int
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "step_index": self.step_index,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class ChaosScenario:
    """A reproducible plan of disruptions for one simulation run."""

    seed: int
    disruptions: Tuple[ChaosDisruption, ...]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "description": self.description,
            "disruptions": [d.to_dict() for d in self.disruptions],
        }

    def disruption_at(self, step_index: int) -> Optional[ChaosDisruption]:
        """Return the disruption (if any) planned for this step.

        At most one disruption per step — keeps the simulation easy to
        reason about. If you need multiple, model them as separate
        steps in the scenario.
        """
        for d in self.disruptions:
            if d.step_index == step_index:
                return d
        return None


# ── Deterministic seeding ───────────────────────────────────────────────────


def _seeded_choice(seed: int, *parts: object) -> int:
    """Stable integer hash from a seed + extra context.

    Used to pick disruption parameters (which step to fail on, how
    long the latency spike is) deterministically. ``hashlib`` over
    Python's built-in ``hash()`` because the latter is randomized
    across interpreter starts and would break reproducibility.
    """
    payload = "|".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


# ── Disruption builders ─────────────────────────────────────────────────────


def tool_failure(
    *,
    tool: str,
    on_call_index: int = 0,
    error_payload: Optional[Dict[str, Any]] = None,
) -> ChaosDisruption:
    """A specific tool fails on its ``on_call_index``-th invocation.

    ``error_payload`` is what the simulator returns to the agent in
    place of the real tool result. Defaults to ``{"error":
    "simulated_failure"}`` — same shape most adapters already pass
    through, so existing agent error-handling exercises naturally.
    """
    return ChaosDisruption(
        mode=MODE_TOOL_FAILURE,
        step_index=on_call_index,
        params={
            "tool": tool,
            "error_payload": error_payload or {"error": "simulated_failure"},
        },
    )


def latency_spike(
    *,
    on_call_index: int,
    delay_ms: int = 5000,
) -> ChaosDisruption:
    """The ``on_call_index``-th tool call sleeps ``delay_ms`` extra.

    Latency, not failure. Useful for testing timeouts, retry logic,
    and user-facing progress indicators. ``delay_ms`` defaults to 5s,
    which is past the default tool timeout of 30s but conservative
    enough that most CI runs survive it.
    """
    return ChaosDisruption(
        mode=MODE_LATENCY_SPIKE,
        step_index=on_call_index,
        params={"delay_ms": int(delay_ms)},
    )


def goal_interruption(
    *,
    after_step: int,
    new_message: str,
) -> ChaosDisruption:
    """Inject a synthetic user message after step ``after_step``.

    Models the "user changed their mind mid-task" disruption — exactly
    the failure mode goal-drift detection (``evalview.core.goal_drift``)
    is designed to catch. Pair the two in tests.
    """
    return ChaosDisruption(
        mode=MODE_GOAL_INTERRUPTION,
        step_index=after_step,
        params={"new_message": new_message},
    )


# ── Scenario synthesis ──────────────────────────────────────────────────────


def build_scenario(
    seed: int,
    *,
    disruptions: List[ChaosDisruption],
    description: str = "",
) -> ChaosScenario:
    """Bundle a list of disruptions into a reproducible scenario.

    Sorts by ``step_index`` and rejects duplicates at the same step.
    The "one disruption per step" rule keeps the simulator
    deterministic and the scenario easy for humans to read.
    """
    seen: set[int] = set()
    unique: List[ChaosDisruption] = []
    for d in disruptions:
        if d.step_index in seen:
            raise ValueError(
                f"two disruptions target step {d.step_index}; "
                "model them as separate steps in the scenario"
            )
        seen.add(d.step_index)
        unique.append(d)
    unique.sort(key=lambda d: d.step_index)
    return ChaosScenario(
        seed=seed,
        disruptions=tuple(unique),
        description=description,
    )


def random_scenario(
    seed: int,
    *,
    available_tools: List[str],
    max_steps: int,
    n_disruptions: int = 1,
    modes: Tuple[str, ...] = SHIPPED_MODES,
) -> ChaosScenario:
    """Construct a chaos scenario deterministically from a seed.

    Same ``(seed, available_tools, max_steps, n_disruptions, modes)``
    always produces the same scenario — that's how runs become
    reproducible across machines and CI re-runs. Useful for property-
    testing-style sweeps where you want N varied chaos plans without
    writing each by hand.
    """
    if not modes:
        raise ValueError("at least one mode must be available")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    disruptions: List[ChaosDisruption] = []
    used_steps: set[int] = set()
    for i in range(n_disruptions):
        mode = modes[_seeded_choice(seed, "mode", i) % len(modes)]
        # Pick a step that isn't taken yet. After many collisions we
        # accept the duplicate by skipping — the scenario builder will
        # catch it. In practice with any reasonable max_steps this loop
        # converges in one or two tries.
        step = _seeded_choice(seed, "step", i) % max_steps
        attempts = 0
        while step in used_steps and attempts < max_steps:
            step = (step + 1) % max_steps
            attempts += 1
        used_steps.add(step)

        if mode == MODE_TOOL_FAILURE:
            tool = available_tools[
                _seeded_choice(seed, "tool", i) % max(1, len(available_tools))
            ] if available_tools else "<unknown>"
            disruptions.append(tool_failure(
                tool=tool, on_call_index=step,
            ))
        elif mode == MODE_LATENCY_SPIKE:
            delay = 1000 + (_seeded_choice(seed, "delay", i) % 5000)
            disruptions.append(latency_spike(
                on_call_index=step, delay_ms=delay,
            ))
        elif mode == MODE_GOAL_INTERRUPTION:
            disruptions.append(goal_interruption(
                after_step=step,
                new_message="Actually, ignore that and tell me about the weather.",
            ))
        # New shipped modes go here; the helper functions encapsulate
        # parameter selection so this loop only knows about mode names.

    return build_scenario(
        seed,
        disruptions=disruptions,
        description=f"random scenario seed={seed} n={n_disruptions}",
    )


# ── Roadmap of additional chaos modes (contributor surface) ─────────────────


CHAOS_MODES_ROADMAP: Tuple[str, ...] = (
    "info_drift: mid-trajectory, swap a known fact in the working context "
    "(e.g. change the user's stated location). Tests robustness to silent "
    "context corruption.",
    "rate_limit: nth tool call returns a 429 with an honored retry-after. "
    "Tests adapter retry / backoff behavior.",
    "partial_handoff: a multi-agent handoff drops half the state. Tests "
    "downstream agent's tolerance to incomplete handoff payloads.",
    "memory_corruption: a memory read returns stale or partially edited "
    "content. Pairs with retrieval_lineage's stale-memory detection.",
    "schema_drift: a tool returns a payload that's almost-but-not-quite "
    "the declared schema. Tests JSON schema enforcement and recovery.",
    "user_typo: the next user message arrives with realistic typos. Tests "
    "robustness vs. clean-input training data.",
)
