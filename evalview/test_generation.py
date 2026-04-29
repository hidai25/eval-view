"""Draft regression-suite generation from live agent probing."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set

import yaml  # type: ignore[import-untyped]

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ConversationTurn,
    ExpectedBehavior,
    ExpectedOutput,
    TestCase,
    TestInput,
    Thresholds,
)
from evalview.importers.log_importer import LogEntry
from evalview.test_generation_constants import (
    REFUSAL_PATTERNS,  # noqa: F401  (re-exported for backward compat — used by core.model_check_scoring)
    _CAPABILITY_PROMPT,
    _DANGEROUS_TOOL_KEYWORDS,
    _DISCOVERY_PROMPTS,
    _GENERATE_REPORT_VERSION,
    _GENERIC_PROMPTS,
    _REFUSAL_PATTERNS,
    _SAFE_FAILURE_PROMPTS,
    _SAFE_FOLLOW_UP,
    _TOOL_PROMPT_LIBRARY,
)
from evalview.test_generation_discovery import (
    _normalize_name,
    _trace_from_log_entry,
    _utc_now_iso,
    discover_tool_schemas,
)
from evalview.test_generation_seeds import GenerationHelpersMixin
from evalview.test_generation_synthesis import PromptSynthesisMixin
from evalview.test_generation_types import (
    GenerationResult,
    ProbeResult,
    PromptCandidate,
    _normalize_text_for_comparison,
)

logger = logging.getLogger(__name__)


class AgentTestGenerator(PromptSynthesisMixin, GenerationHelpersMixin):
    """Generate a draft regression suite by probing an agent endpoint."""

    def __init__(
        self,
        adapter: Optional[AgentAdapter],
        endpoint: str,
        adapter_type: str,
        include_tools: Optional[Sequence[str]] = None,
        exclude_tools: Optional[Sequence[str]] = None,
        allow_live_side_effects: bool = False,
        project_root: Optional[Path] = None,
    ):
        self.adapter = adapter
        self.endpoint = endpoint
        self.adapter_type = adapter_type
        self.include_tools = {_normalize_name(tool) for tool in include_tools or []}
        self.exclude_tools = {_normalize_name(tool) for tool in exclude_tools or []}
        self.allow_live_side_effects = allow_live_side_effects
        self.discovered_tools: List[Dict[str, Any]] = []
        self.project_root = project_root or Path.cwd()
        self.prompt_sources: Dict[str, str] = {}
        self._synthesis_succeeded: bool = False

    async def generate(
        self,
        budget: int = 20,
        seed_prompts: Optional[Sequence[str]] = None,
        synthesize: bool = True,
        on_probe_complete: Optional[Callable[[int, int, str, str, List[str]], None]] = None,
        synth_model: Optional[str] = None,
        max_multi_turn: Optional[int] = None,
        turns_per_multi: int = 2,
    ) -> GenerationResult:
        self._synth_model_override = synth_model
        self.discovered_tools = await discover_tool_schemas(self.adapter, self.adapter_type, self.endpoint)
        queue = self._build_probe_queue(seed_prompts or [], budget=budget)
        self.prompt_sources = {prompt: source for prompt, source in queue}
        queue_text: Deque[str] = deque(prompt for prompt, _ in queue)
        seen_queries = set()
        clustered: Dict[str, ProbeResult] = {}
        signatures_seen: Counter[str] = Counter()
        tools_seen: Counter[str] = Counter()
        failures: List[str] = []
        probes_run = 0
        discovery_done_count = 0
        synthesis_done = False
        synthesis_count = 0
        discovery_responses: List[str] = []
        # Wait for N discovery probes before synthesizing.  More responses =
        # better domain understanding.  Scale down for tiny budgets.
        discovery_target = min(len(_DISCOVERY_PROMPTS), max(1, budget // 4))

        while queue_text and probes_run < budget:
            query = queue_text.popleft().strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)

            try:
                if self.adapter is None:
                    raise RuntimeError("No adapter configured for live probing")
                trace = await self.adapter.execute(query)
            except Exception as exc:
                failures.append(f"{query[:80]}: {exc}")
                # Discovery probe failures don't count against budget
                if self.prompt_sources.get(query) != "discovery":
                    probes_run += 1
                if on_probe_complete:
                    on_probe_complete(probes_run, budget, query[:60], "fail", [])
                continue

            probe = self._build_probe_result(query, trace, self.prompt_sources.get(query, "live_probe"))
            signatures_seen[probe.signature] += 1
            tools_seen.update(probe.tools)

            # Discovery probes gather context for synthesis — they should
            # NOT become test cases themselves.  "Hello, what can you help me
            # with?" is a generator artifact, not a production user task.
            # They don't count against the budget so users get the full
            # number of real test probes they asked for.
            is_discovery = (
                self.prompt_sources.get(query) == "discovery"
                or query.strip().lower() in {p.lower() for p in _DISCOVERY_PROMPTS}
            )
            if is_discovery:
                discovery_responses.append(probe.trace.final_output or "")
                discovery_done_count += 1
                if on_probe_complete:
                    on_probe_complete(probes_run, budget, query[:60], "ok", probe.tools)
            else:
                probes_run += 1
                if on_probe_complete:
                    on_probe_complete(probes_run, budget, query[:60], "ok", probe.tools)

            if not is_discovery and probe.signature not in clustered:
                clustered[probe.signature] = probe

            # Synthesize immediately after discovery completes — don't wait
            # for a non-discovery probe.  We now have capability overview +
            # example requests + domain info + tool schemas — enough for the
            # LLM to derive the exact domain and generate domain-native prompts.
            if not synthesis_done and synthesize and discovery_done_count >= discovery_target:
                synthesis_done = True
                synthesized = await self._synthesize_prompts(
                    discovery_responses=discovery_responses,
                    budget=budget,
                )
                for s_prompt in reversed(synthesized):
                    if s_prompt.text not in seen_queries:
                        self.prompt_sources.setdefault(s_prompt.text, s_prompt.source)
                        queue_text.appendleft(s_prompt.text)
                        synthesis_count += 1
                if synthesis_count > 0:
                    self._synthesis_succeeded = True

            # Multi-turn: generate a natural follow-up and attach it to
            # the SAME probe — it enriches the parent test, not a separate one.
            # Follow-ups don't count against the budget since they're part of
            # the parent probe's test case.
            mt_limit = max_multi_turn if max_multi_turn is not None else max(1, budget // 4)
            existing_mt_tools = {
                frozenset(p.tools) for p in clustered.values()
                if p.behavior_class == "multi_turn"
            }
            skip_mt = mt_limit == 0 or frozenset(probe.tools) in existing_mt_tools or len(existing_mt_tools) >= mt_limit
            if not is_discovery and not skip_mt and probe.behavior_class in {"tool_path", "clarification"}:
                # Chain follow-ups to reach the desired turns_per_multi depth.
                # Turn 1 is the original probe; each follow-up adds one turn.
                extra_turns_needed = turns_per_multi - 1
                current_probe = probe
                follow_up_queries: List[str] = []
                follow_up_tools_list: List[List[str]] = []
                all_succeeded = False

                for turn_i in range(extra_turns_needed):
                    if on_probe_complete:
                        label = f"generating follow-up {turn_i + 1}/{extra_turns_needed}..."
                        on_probe_complete(probes_run, budget, label, "info", [])
                    follow_up_probe = await self._generate_multi_turn_probe(current_probe)
                    if follow_up_probe is None:
                        break  # can't generate more turns
                    follow_up_queries.append(follow_up_probe.query)
                    follow_up_tools_list.append(follow_up_probe.tools)
                    tools_seen.update(follow_up_probe.tools)
                    if on_probe_complete:
                        on_probe_complete(probes_run, budget, f"turn {turn_i + 2}: {follow_up_probe.query[:50]}", "ok", follow_up_probe.tools)
                    # Update current_probe for the next follow-up to chain from
                    current_probe = follow_up_probe
                    all_succeeded = True

                if all_succeeded:
                    # Enrich the original probe with multi-turn data
                    old_sig = probe.signature
                    probe.conversation_history = current_probe.conversation_history
                    probe.behavior_class = "multi_turn"
                    probe.signature = self._build_signature("multi_turn", probe.tools)
                    probe.rationale = current_probe.rationale
                    # Store follow-up queries and tools for test case building
                    probe._follow_up_query = follow_up_queries[-1]  # type: ignore[attr-defined]
                    probe._follow_up_tools = follow_up_tools_list[-1]  # type: ignore[attr-defined]
                    probe._all_follow_up_queries = follow_up_queries  # type: ignore[attr-defined]
                    probe._all_follow_up_tools = follow_up_tools_list  # type: ignore[attr-defined]
                    # Replace old signature with new multi-turn signature
                    clustered.pop(old_sig, None)
                    clustered[probe.signature] = probe
                    signatures_seen[probe.signature] += 1
                    logger.debug(
                        "Multi-turn enriched (%d turns): %s",
                        len(follow_up_queries) + 1, probe.query[:40],
                    )

            prioritized_candidates = list(self._expand_probe_candidates(probe))
            for candidate in reversed(prioritized_candidates):
                if candidate not in seen_queries:
                    self.prompt_sources.setdefault(candidate, "follow_up")
                    # When synthesis succeeded, don't let heuristic follow-ups
                    # jump ahead of synthesized prompts in the queue.
                    if self._synthesis_succeeded:
                        queue_text.append(candidate)
                    else:
                        queue_text.appendleft(candidate)

        tests = [self._build_test_case(probe, clustered) for probe in clustered.values()]

        # Refine test names and output assertions via cheap LLM call.
        if synthesize:
            await self._refine_tests_with_llm(tests, list(clustered.values()))

        # Drop tests where the prompt intent doesn't match observed behavior
        # (e.g., prompt says "run collection" but agent only searched).
        if synthesize:
            tests = await self._filter_incoherent_tests(tests)

        report = self._build_report(
            clustered=list(clustered.values()),
            probes_run=probes_run,
            signatures_seen=signatures_seen,
            tools_seen=tools_seen,
            failures=failures,
        )
        report["prompt_synthesis"] = {"count": synthesis_count}
        return GenerationResult(
            tests=tests,
            probes_run=probes_run,
            signatures_seen=signatures_seen,
            tools_seen=tools_seen,
            failures=failures,
            report=report,
        )

    def write_suite(
        self,
        result: GenerationResult,
        out_dir: Path,
        *,
        replace_existing: bool = True,
    ) -> List[Path]:
        """Write generated tests and report to disk."""
        previous_report = self._load_previous_report(out_dir / "generated.report.json")
        if replace_existing:
            self._clear_generated_suite(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        if previous_report is not None:
            result.report["changes_since_last_generation"] = self._compute_report_delta(
                previous_report,
                result.report,
            )
        used_names: Set[str] = set()
        for test in result.tests:
            file_name = self._slugify(test.name)[:60] or "generated-test"
            # Dedupe filenames — append suffix if collision
            base = file_name
            counter = 2
            while file_name in used_names:
                file_name = f"{base[:55]}-{counter}"
                counter += 1
            used_names.add(file_name)
            path = out_dir / f"{file_name}.yaml"
            self._write_test_yaml(test, path)
            written.append(path)

        report_path = out_dir / "generated.report.json"
        report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
        written.append(report_path)
        return written

    def _clear_generated_suite(self, out_dir: Path) -> None:
        """Remove only prior EvalView-generated artifacts from an output directory."""
        if not out_dir.exists():
            return

        for yaml_path in out_dir.glob("*.yaml"):
            if yaml_path.is_file() and self._is_generated_yaml(yaml_path):
                yaml_path.unlink()

        report_path = out_dir / "generated.report.json"
        if report_path.exists():
            report_path.unlink()

        pycache_dir = out_dir / "__pycache__"
        if pycache_dir.exists() and pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)

    def _replace_all_yaml_suite(self, out_dir: Path) -> None:
        """Remove all YAML drafts in an output directory after explicit confirmation."""
        if not out_dir.exists():
            return

        for yaml_path in out_dir.glob("*.yaml"):
            if yaml_path.is_file():
                yaml_path.unlink()

        report_path = out_dir / "generated.report.json"
        if report_path.exists():
            report_path.unlink()

    def classify_output_dir(self, out_dir: Path) -> tuple[list[Path], list[Path]]:
        """Return (generated_yaml, handwritten_yaml) for an output folder."""
        generated: list[Path] = []
        handwritten: list[Path] = []
        if not out_dir.exists():
            return generated, handwritten

        for yaml_path in sorted(out_dir.glob("*.yaml")):
            if not yaml_path.is_file():
                continue
            if self._is_generated_yaml(yaml_path):
                generated.append(yaml_path)
            else:
                handwritten.append(yaml_path)
        return generated, handwritten

    def _is_generated_yaml(self, path: Path) -> bool:
        """Best-effort check for EvalView-generated draft YAML."""
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            return False

        if raw.startswith("# Auto-generated by: evalview generate"):
            return True

        try:
            data = yaml.safe_load(raw) or {}
        except Exception:
            return False

        if data.get("generated") is True:
            return True

        meta = data.get("meta") or {}
        generated_by = str(meta.get("generated_by") or "")
        return generated_by in {"evalview generate", "evalview init"}

    def generate_from_log_entries(self, entries: Sequence[LogEntry]) -> GenerationResult:
        """Generate a draft suite from imported traffic/log entries."""
        clustered: Dict[str, ProbeResult] = {}
        signatures_seen: Counter[str] = Counter()
        tools_seen: Counter[str] = Counter()

        for entry in entries:
            probe = self._build_probe_result_from_log(entry)
            signatures_seen[probe.signature] += 1
            tools_seen.update(probe.tools)
            if probe.signature not in clustered:
                clustered[probe.signature] = probe

        tests = [self._build_test_case(probe, clustered) for probe in clustered.values()]
        report = self._build_report(
            clustered=list(clustered.values()),
            probes_run=len(entries),
            signatures_seen=signatures_seen,
            tools_seen=tools_seen,
            failures=[],
        )
        report["source"] = "logs"
        return GenerationResult(
            tests=tests,
            probes_run=len(entries),
            signatures_seen=signatures_seen,
            tools_seen=tools_seen,
            failures=[],
            report=report,
        )

    def _build_probe_queue(self, seed_prompts: Sequence[str], budget: int = 20) -> Deque[tuple[str, str]]:
        queue: Deque[tuple[str, str]] = deque()
        seen: Set[str] = set()

        # User-provided seeds (from --seed-file) are trusted and queued directly.
        user_seed_prompts = [PromptCandidate(prompt.strip(), "seed_file") for prompt in seed_prompts if prompt.strip()]

        def enqueue(prompt: str, source: str) -> None:
            normalized = prompt.strip()
            if not normalized or normalized in seen:
                return
            if self._prompt_is_allowed(normalized):
                seen.add(normalized)
                queue.append((normalized, source))

        # Scale discovery probes to budget: always ask capability, add
        # example/domain probes only when budget allows.
        max_discovery = min(len(_DISCOVERY_PROMPTS), max(1, budget // 4))
        for prompt in _DISCOVERY_PROMPTS[:max_discovery]:
            enqueue(prompt, "discovery")

        # Only queue user-provided seeds directly.  Workspace seeds (from
        # existing test files, project docs, schema) are NOT queued as probes
        # — they may come from a different agent/domain.  Instead they are
        # passed as context to the LLM synthesis step, which decides what's
        # relevant to the current agent's discovered capabilities.
        for candidate in user_seed_prompts:
            enqueue(candidate.text, candidate.source)

        fallback_generic_prompts = [] if user_seed_prompts else list(_GENERIC_PROMPTS)
        for prompt in fallback_generic_prompts:
            enqueue(prompt, "generic")
        for prompt in _SAFE_FAILURE_PROMPTS:
            enqueue(prompt, "safety")
        return queue

    def _build_probe_result(self, query: str, trace: Any, prompt_source: str) -> ProbeResult:
        tools = [step.tool_name for step in trace.steps if getattr(step, "tool_name", None)]
        behavior_class = self._classify_behavior(trace, tools)
        signature = self._build_signature(behavior_class, tools)

        if tools:
            rationale = f"Observed tool path: {' -> '.join(tools)}"
        else:
            rationale = f"Observed {behavior_class.replace('_', ' ')} path"

        return ProbeResult(
            query=query,
            trace=trace,
            tools=tools,
            signature=signature,
            behavior_class=behavior_class,
            rationale=rationale,
            safety_probe=query in _SAFE_FAILURE_PROMPTS,
            prompt_source=prompt_source,
        )

    def _build_probe_result_from_log(self, entry: LogEntry) -> ProbeResult:
        tools = [tool for tool in entry.tool_calls if self._tool_is_allowed(tool)]
        behavior_class = self._classify_log_behavior(entry, tools)
        signature = self._build_signature(behavior_class, tools)
        rationale = (
            f"Imported from logs with tool path: {' -> '.join(tools)}"
            if tools
            else f"Imported from logs as {behavior_class.replace('_', ' ')} path"
        )
        trace = _trace_from_log_entry(entry)
        return ProbeResult(
            query=entry.query,
            trace=trace,
            tools=tools,
            signature=signature,
            behavior_class=behavior_class,
            rationale=rationale,
            safety_probe="safe" in str(entry.metadata).lower() or "refus" in (entry.output or "").lower(),
            prompt_source="logs",
        )

    def _expand_probe_candidates(self, probe: ProbeResult) -> List[str]:
        candidates = []

        # Only extract example queries from the agent's response when
        # synthesis didn't fire.  When synthesis succeeded, the LLM already
        # generated better prompts — raw response fragments like
        # "Competitors mentioned: LangSmith, Langfuse" read as mined topic
        # text, not real user requests.
        if not self._synthesis_succeeded:
            output = probe.trace.final_output or ""
            candidates.extend(self._extract_example_queries(output))

        # Only inject generic library prompts when LLM synthesis didn't
        # produce domain-specific alternatives — otherwise the library
        # floods the queue with off-domain noise ("weather in SF", etc.).
        if not self._synthesis_succeeded:
            for tool_name in probe.tools:
                if not self._tool_is_allowed(tool_name):
                    continue
                normalized = tool_name.lower().replace("_", " ").replace("-", " ")
                for keyword, prompts in _TOOL_PROMPT_LIBRARY.items():
                    if keyword in normalized:
                        candidates.extend(prompts)

        if probe.behavior_class == "refusal":
            candidates.append("Tell me what safe alternative you can help with instead.")

        return [candidate for candidate in candidates if self._prompt_is_allowed(candidate)]

    async def _generate_multi_turn_probe(self, probe: ProbeResult) -> Optional[ProbeResult]:
        """Generate a natural follow-up question from the agent's response.

        Instead of a static "use the most sensible default" prompt, we ask an
        LLM to write a realistic follow-up that a real user would type after
        seeing the agent's answer.  This produces reliable, reproducible
        multi-turn conversations grounded in actual agent behavior.

        Falls back to the static follow-up if no LLM is available.
        """
        if self.adapter is None:
            return None

        output = (probe.trace.final_output or "").strip()
        if not output or len(output) < 20:
            return None

        # Generate a natural follow-up via LLM
        follow_up_query = await self._synthesize_follow_up_query(probe.query, output)
        if not follow_up_query:
            # Fallback: static follow-up for clarifications only
            if probe.behavior_class == "clarification":
                follow_up_query = _SAFE_FOLLOW_UP
            else:
                return None

        history = [
            {"role": "user", "content": probe.query},
            {"role": "assistant", "content": output},
        ]
        try:
            trace = await self.adapter.execute(
                follow_up_query, {"conversation_history": history}
            )
        except Exception:
            return None

        follow_up = self._build_probe_result(probe.query, trace, "multi_turn")
        if any(not self._tool_is_allowed(tool) for tool in follow_up.tools):
            return None
        if not self._is_meaningful_follow_up(probe, follow_up):
            return None
        follow_up.behavior_class = "multi_turn"
        follow_up.signature = self._build_signature("multi_turn", follow_up.tools)
        follow_up.rationale = (
            f"Turn 1: {probe.query[:60]} → Turn 2: {follow_up_query[:60]}"
        )
        follow_up.conversation_history = history
        # Store the actual follow-up query so the test case uses it
        follow_up.query = follow_up_query
        return follow_up

    async def _synthesize_follow_up_query(
        self, original_query: str, agent_response: str
    ) -> Optional[str]:
        """Use LLM to generate a natural follow-up question from an agent response."""
        client = self._select_synthesis_client(model_override=getattr(self, "_synth_model_override", None))
        if client is None:
            return None

        try:
            result = await client.chat_completion(
                system_prompt=(
                    "You write a single realistic follow-up question a user would "
                    "ask after seeing an AI agent's response. Write ONLY the follow-up "
                    "question, nothing else. Keep it short and natural — like a real "
                    "person continuing a conversation. "
                    'Return JSON: {"follow_up": "your question here"}'
                ),
                user_prompt=(
                    f"User asked: {original_query[:200]}\n\n"
                    f"Agent responded: {agent_response[:500]}\n\n"
                    "What would the user naturally ask next?"
                ),
                temperature=0.7,
                max_tokens=150,
            )
        except Exception:
            return None

        follow_up = (result.get("follow_up") or "").strip()
        if not follow_up or len(follow_up) < 5 or len(follow_up) > 200:
            return None
        return follow_up

    def _classify_behavior(self, trace: Any, tools: Sequence[str]) -> str:
        output = (trace.final_output or "").lower()
        if any(not step.success for step in trace.steps):
            return "error_path"
        if tools:
            return "tool_path"
        if any(pattern in output for pattern in _REFUSAL_PATTERNS):
            return "refusal"
        if "?" in output:
            return "clarification"
        return "direct_answer"

    def _classify_log_behavior(self, entry: LogEntry, tools: Sequence[str]) -> str:
        output = (entry.output or "").lower()
        if tools:
            return "tool_path"
        if any(pattern in output for pattern in _REFUSAL_PATTERNS):
            return "refusal"
        if "?" in output:
            return "clarification"
        return "direct_answer"

    def _is_meaningful_follow_up(self, first_probe: ProbeResult, follow_up_probe: ProbeResult) -> bool:
        """Only keep follow-ups that materially advance the conversation.

        A follow-up is meaningful if it produces a different response from
        the first turn — regardless of whether it uses tools.  Real multi-turn
        conversations often have a tool-using first turn and a text-only
        follow-up (or vice versa).
        """
        first_output = _normalize_text_for_comparison(first_probe.trace.final_output or "")
        second_output = _normalize_text_for_comparison(follow_up_probe.trace.final_output or "")

        if not second_output:
            return False
        if first_output == second_output:
            return False
        if _normalize_text_for_comparison(_SAFE_FOLLOW_UP) in second_output:
            return False
        return True

    def _build_signature(self, behavior_class: str, tools: Sequence[str]) -> str:
        if tools:
            # Preserve full tool path including repeated calls — different
            # repetition counts represent distinct agent behaviors (e.g.
            # search twice vs three times) and should produce separate tests.
            return f"{behavior_class}:{'->'.join(tools)}"
        return behavior_class

    def _prompt_is_allowed(self, prompt: str) -> bool:
        if not self._prompt_matches_tool_filters(prompt):
            return False
        if self.allow_live_side_effects:
            return True
        lowered = prompt.lower()
        blocked_terms = ("send email", "draft an email", "email ", "delete", "purchase", "pay", "transfer money", "book ")
        return not any(term in lowered for term in blocked_terms)

    def _tool_is_allowed(self, tool_name: str) -> bool:
        if not self._matches_tool_filters(tool_name):
            return False
        if self.allow_live_side_effects:
            return True
        return not self._is_dangerous_tool_name(tool_name)

    def _matches_tool_filters(self, tool_name: str) -> bool:
        normalized = _normalize_name(tool_name)
        if self.include_tools and normalized not in self.include_tools:
            return False
        if normalized in self.exclude_tools:
            return False
        return True

    def _prompt_matches_tool_filters(self, prompt: str) -> bool:
        lowered = prompt.lower()
        prompt_tools = self._infer_prompt_tools(lowered)
        if self.include_tools and prompt_tools and not (prompt_tools & self.include_tools):
            return False
        if self.exclude_tools and prompt_tools and (prompt_tools & self.exclude_tools):
            return False
        return True

    def _infer_prompt_tools(self, lowered_prompt: str) -> Set[str]:
        prompt_tools = {
            _normalize_name(keyword)
            for keyword in _TOOL_PROMPT_LIBRARY
            if keyword in lowered_prompt
        }
        heuristic_map = {
            "calculator": ("calculate", "divided", "multiply", "subtract", "math"),
            "weather": ("weather", "umbrella", "forecast"),
            "email": ("email", "follow-up"),
            "calendar": ("schedule", "appointment", "meeting"),
            "search": ("search", "find recent", "look up"),
        }
        for tool_name, hints in heuristic_map.items():
            if any(hint in lowered_prompt for hint in hints):
                prompt_tools.add(_normalize_name(tool_name))
        return prompt_tools

    def _build_test_case(self, probe: ProbeResult, clustered: Dict[str, ProbeResult]) -> TestCase:
        expected = ExpectedBehavior()
        if probe.tools:
            # Assert the full tool path including repeated calls — repetition
            # counts are meaningful (e.g. two searches vs three).
            expected.tools = list(probe.tools)
            # Assert sequence when there are multiple calls (including repeats)
            if len(probe.tools) > 1:
                expected.sequence = list(probe.tools)

        phrases = self._extract_stable_phrases(
            probe.trace.final_output,
            behavior_class=probe.behavior_class,
            has_tools=bool(probe.tools),
        )
        if phrases:
            expected.output = ExpectedOutput(
                contains=phrases,
                not_contains=["error", "traceback"],
            )
        elif probe.behavior_class in {"refusal", "clarification"}:
            expected.output = ExpectedOutput(not_contains=["error", "traceback"])

        confidence = self._confidence_for_probe(probe)
        dangerous_tools = self._infer_forbidden_tools(probe, list(clustered.values()))
        if dangerous_tools:
            expected.forbidden_tools = dangerous_tools

        description = self._build_description(probe, dangerous_tools)

        test_case = TestCase(
            name=self._generate_test_name(probe.query, probe.tools, probe.behavior_class),
            description=description,
            meta={
                "generated_by": "evalview generate",
                "review_status": "draft",
                "confidence": confidence,
                "rationale": probe.rationale,
                "behavior_class": probe.behavior_class,
                "signature": probe.signature,
                "prompt_source": probe.prompt_source,
            },
            input=TestInput(query=probe.query),
            expected=expected,
            thresholds=self._generate_thresholds(probe.trace),
            adapter=self.adapter_type,
            endpoint=self.endpoint,
            generated=True,
        )
        if probe.behavior_class == "multi_turn" and probe.conversation_history:
            all_follow_up_queries = getattr(probe, "_all_follow_up_queries", None) or []
            all_follow_up_tools = getattr(probe, "_all_follow_up_tools", None) or []

            # Fall back to single follow-up for backward compat
            if not all_follow_up_queries:
                follow_up_query = getattr(probe, "_follow_up_query", None) or _SAFE_FOLLOW_UP
                all_follow_up_queries = [follow_up_query]
                follow_up_tools = getattr(probe, "_follow_up_tools", None) or []
                all_follow_up_tools = [follow_up_tools] if follow_up_tools else []

            # Merge tools from all turns
            all_tools = list(probe.tools)
            for ft in all_follow_up_tools:
                all_tools.extend(ft)
            if all_tools:
                expected.tools = list(all_tools)
                if len(all_tools) > 1:
                    expected.sequence = list(all_tools)

            # Build turns list: original query + each follow-up
            turns = [ConversationTurn(query=probe.query)]
            for fq in all_follow_up_queries:
                turns.append(ConversationTurn(query=fq))
            test_case.turns = turns
        return test_case

    def _build_report(
        self,
        clustered: Sequence[ProbeResult],
        probes_run: int,
        signatures_seen: Counter[str],
        tools_seen: Counter[str],
        failures: Sequence[str],
    ) -> Dict[str, Any]:
        covered_classes = Counter(probe.behavior_class for probe in clustered)
        return {
            "report_version": _GENERATE_REPORT_VERSION,
            "generated_at": _utc_now_iso(),
            "probes_run": probes_run,
            "tests_generated": len(clustered),
            "discovery": {
                "count": len(self.discovered_tools),
                "tools": [
                    {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                    }
                    for tool in self.discovered_tools
                ],
            },
            "prompt_sources": dict(Counter(probe.prompt_source for probe in clustered)),
            "behavior_signatures": dict(signatures_seen),
            "covered": {
                "tool_paths": covered_classes.get("tool_path", 0),
                "direct_answers": covered_classes.get("direct_answer", 0),
                "clarifications": covered_classes.get("clarification", 0),
                "multi_turn": covered_classes.get("multi_turn", 0),
                "refusals": covered_classes.get("refusal", 0),
                "error_paths": covered_classes.get("error_path", 0),
            },
            "tools_seen": dict(tools_seen),
            "failures": list(failures),
            "draft_tests": [
                {
                    "name": self._generate_test_name(probe.query, probe.tools, probe.behavior_class),
                    "query": probe.query,
                    "signature": probe.signature,
                    "rationale": probe.rationale,
                    "prompt_source": probe.prompt_source,
                }
                for probe in clustered
            ],
            "gaps": self._identify_gaps(covered_classes, tools_seen),
        }

    def _identify_gaps(self, covered_classes: Counter[str], tools_seen: Counter[str]) -> List[str]:
        gaps: List[str] = []
        missing_discovered = [
            tool.get("name", "")
            for tool in self.discovered_tools
            if _normalize_name(tool.get("name", "")) not in {_normalize_name(name) for name in tools_seen}
            and self._tool_is_allowed(tool.get("name", ""))
        ]
        if not tools_seen:
            gaps.append("No tool-using behavior observed; add seed prompts for tool flows.")
        elif missing_discovered:
            gaps.append(
                "Discovered but not exercised: " + ", ".join(missing_discovered[:5])
            )
        if covered_classes.get("clarification", 0) == 0:
            gaps.append("No clarification path observed.")
        if covered_classes.get("multi_turn", 0) == 0:
            gaps.append("No multi-turn completion path observed after clarification.")
        if covered_classes.get("refusal", 0) == 0:
            gaps.append("No refusal or safety path observed.")
        if covered_classes.get("error_path", 0) == 0:
            gaps.append("No error-path behavior observed.")
        return gaps

    def _infer_forbidden_tools(
        self,
        probe: ProbeResult,
        clustered: Sequence[ProbeResult],
    ) -> Optional[List[str]]:
        """Infer a narrow forbidden_tools contract from safety refusals."""
        if not probe.safety_probe and probe.behavior_class != "refusal":
            return None
        if probe.tools:
            return None

        observed_dangerous: List[str] = []
        seen: Set[str] = set()
        for candidate in clustered:
            for tool in candidate.tools:
                normalized = _normalize_name(tool)
                if normalized in seen:
                    continue
                if any(keyword in normalized for keyword in _DANGEROUS_TOOL_KEYWORDS):
                    seen.add(normalized)
                    observed_dangerous.append(tool)

        if observed_dangerous:
            return observed_dangerous

        discovered_dangerous = [
            tool.get("name", "")
            for tool in self.discovered_tools
            if tool.get("name")
            and self._matches_tool_filters(tool["name"])
            and self._is_dangerous_tool_schema(tool)
        ]
        return discovered_dangerous or None

    def _build_description(self, probe: ProbeResult, dangerous_tools: Optional[List[str]]) -> str:
        confidence = self._confidence_for_probe(probe)
        parts = [
            "Draft generated by evalview generate.",
            f"Confidence: {confidence}.",
            probe.rationale + ".",
        ]
        if dangerous_tools:
            parts.append(f"Inferred forbidden_tools: {', '.join(dangerous_tools)}.")
        parts.append("Review before snapshotting.")
        return " ".join(parts)

    def _confidence_for_probe(self, probe: ProbeResult) -> str:
        return "high" if probe.tools or probe.behavior_class in {"refusal", "multi_turn"} else "medium"

    def _schema_prompts(self, project_prompts: Sequence[str]) -> List[str]:
        prompts: List[str] = []
        prompt_context = "\n".join(project_prompts[:20])
        for tool in self.discovered_tools:
            tool_name = tool.get("name", "")
            description = (tool.get("description") or "").strip()
            if not tool_name or not self._tool_is_allowed(tool_name):
                continue
            prompt = self._tool_prompt_from_schema(
                tool_name,
                description,
                tool.get("inputSchema") or {},
                prompt_context,
            )
            if prompt:
                prompts.append(prompt)
        return prompts[:10]

    def _tool_prompt_from_schema(
        self,
        tool_name: str,
        description: str,
        input_schema: Dict[str, Any],
        project_prompt_context: str,
    ) -> Optional[str]:
        normalized = _normalize_name(tool_name)
        if not self._synthesis_succeeded:
            for keyword, prompts in _TOOL_PROMPT_LIBRARY.items():
                if keyword in normalized:
                    return prompts[0]

        domain_match = self._find_domain_prompt_for_tool(
            tool_name,
            description,
            project_prompt_context,
        )
        if domain_match:
            return domain_match

        # Generate a natural-sounding prompt from the description
        if description:
            desc_clean = description.strip().rstrip(".")
            desc_lower = desc_clean.lower()

            # Verbs where we replace the action word: "get entries" → "Show me entries"
            _verb_replacements = {
                "get": "Show me", "fetch": "Show me", "retrieve": "Show me",
                "list": "Show me", "show": "Show me", "find": "Find",
                "search": "Search for", "query": "Look up",
            }
            for verb, replacement in _verb_replacements.items():
                if desc_lower.startswith(verb):
                    rest = desc_lower.split(" ", 1)[-1] if " " in desc_lower else desc_lower
                    return f"{replacement} {rest}"

            # Verbs where we prepend a prefix: "log pain" → "I need to log pain"
            _verb_prefixes = {
                "create": "I need to", "add": "I need to", "log": "I need to",
                "record": "I need to", "save": "I need to", "insert": "I need to",
                "update": "Can you", "modify": "Can you", "change": "Can you",
                "analyze": "Please", "calculate": "Please", "compute": "Please",
                "summarize": "Please",
            }
            for verb, prefix in _verb_prefixes.items():
                if desc_lower.startswith(verb):
                    return f"{prefix} {desc_lower}"

            # Passthrough verbs: use the description as-is
            if any(desc_lower.startswith(v) for v in ("check", "verify", "validate")):
                return desc_clean

            return f"I need help with this: {desc_lower}"

        human_name = tool_name.replace("_", " ").replace("-", " ")
        return f"Help me with {human_name}"

    def _find_domain_prompt_for_tool(
        self,
        tool_name: str,
        description: str,
        project_prompt_context: str,
    ) -> Optional[str]:
        normalized_tool = _normalize_name(tool_name)
        normalized_description = _normalize_name(description)
        for candidate in self._workspace_seed_prompts():
            prompt = candidate.text
            normalized_prompt = _normalize_name(prompt)
            if not normalized_prompt:
                continue
            if normalized_tool and normalized_tool[:12] in normalized_prompt:
                return prompt
            if normalized_description and any(
                token and token in normalized_prompt
                for token in normalized_description.split()
                if len(token) > 4
            ):
                return prompt
        if project_prompt_context:
            example_queries = self._extract_example_queries(project_prompt_context)
            if example_queries:
                return example_queries[0]
        return None

    def _generate_thresholds(self, trace: Any) -> Thresholds:
        # Generated drafts should not hard-fail on cost or latency swings.
        # LLM-backed agents have high variance in both — a 2x cost spike is
        # normal when the agent takes a different tool path.  Cost and latency
        # still appear as metrics in the report for visibility.
        # Score 70 = "acceptable quality" — the universal default.
        return Thresholds(min_score=50.0, max_cost=None, max_latency=None)

    def _generate_test_name(self, query: str, tools: Sequence[str], behavior_class: str) -> str:
        normalized_query = " ".join(query.lower().split())
        if normalized_query == _CAPABILITY_PROMPT.lower():
            base = "Capability overview"
        elif normalized_query == _SAFE_FOLLOW_UP.lower():
            base = "Clarification follow-up"
        elif any(normalized_query == p.lower() for p in _DISCOVERY_PROMPTS):
            base = "Discovery probe"
        else:
            _name_stop = {
                "what", "when", "where", "which", "with", "from", "about",
                "have", "help", "could", "would", "should", "this", "that",
                "your", "today", "please", "most", "sensible", "default",
                "continue", "the", "for", "and", "can", "you", "show",
                "give", "tell", "some", "few", "example", "requests",
                "tasks", "handle", "well", "types", "data", "information",
                "work", "need", "realistic", "task", "include", "tool",
                "use", "like", "want", "know", "does", "will", "just",
                "also", "very", "been", "into", "over", "more", "them",
                "then", "there", "make", "here", "those", "these",
            }
            words = re.findall(r"\b\w+\b", query)
            key_words = [w for w in words if len(w) > 2 and w.lower() not in _name_stop][:6]
            base = " ".join(key_words).capitalize() if key_words else "Generated test"

        # Add behavior context — NOT raw tool names.
        # Names must be alphanumeric + spaces + hyphens only (Pydantic validation).
        if behavior_class == "refusal":
            return f"Refusal - {base}"
        if behavior_class == "error_path":
            return f"Error - {base}"
        if behavior_class == "multi_turn":
            return f"{base} - multi-turn"
        if behavior_class == "clarification" and base not in {"Capability overview", "Discovery probe"}:
            return f"{base} - clarification"
        return base

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_seed_prompts(seed_path: Optional[str]) -> List[str]:
    """Load line-based seed prompts from disk."""
    if not seed_path:
        return []
    path = Path(seed_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed prompt file not found: {seed_path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_generation(
    adapter: Optional[AgentAdapter],
    endpoint: str,
    adapter_type: str,
    budget: int,
    seed_prompts: Optional[Sequence[str]] = None,
    include_tools: Optional[Sequence[str]] = None,
    exclude_tools: Optional[Sequence[str]] = None,
    allow_live_side_effects: bool = False,
    project_root: Optional[Path] = None,
    synthesize: bool = True,
    on_probe_complete: Optional[Callable[[int, int, str, str, List[str]], None]] = None,
    synth_model: Optional[str] = None,
    max_multi_turn: Optional[int] = None,
    turns_per_multi: int = 2,
) -> GenerationResult:
    """Sync wrapper for CLI usage."""
    generator = AgentTestGenerator(
        adapter=adapter,
        endpoint=endpoint,
        adapter_type=adapter_type,
        include_tools=include_tools,
        exclude_tools=exclude_tools,
        allow_live_side_effects=allow_live_side_effects,
        project_root=project_root,
    )
    return asyncio.run(generator.generate(
        budget=budget,
        seed_prompts=seed_prompts,
        synthesize=synthesize,
        on_probe_complete=on_probe_complete,
        synth_model=synth_model,
        max_multi_turn=max_multi_turn,
        turns_per_multi=turns_per_multi,
    ))


