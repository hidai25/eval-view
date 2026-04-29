"""PromptSynthesisMixin — LLM-powered prompt synthesis for `AgentTestGenerator`.

Inherits-into AgentTestGenerator so the parent class stays focused on the
probe → cluster → test-case pipeline. All methods here read `self.project_root`,
`self.discovered_tools`, `self._prompt_is_allowed`, and (`self._workspace_seed_prompts`)
which are provided by the parent.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from evalview.test_generation_constants import (
    _DISCOVERY_PROMPTS,
    _PROJECT_CONTEXT_FILES,
    _SYNTHESIS_PROVIDER_PRIORITY,
    _SYNTHESIS_SYSTEM_PROMPT,
)
from evalview.test_generation_types import ProbeResult, PromptCandidate

logger = logging.getLogger(__name__)


if TYPE_CHECKING:

    class _ParentProtocol:
        """Forward declarations of attributes/methods this mixin reads from
        the composed `AgentTestGenerator`. Existing only for mypy — at
        runtime the MRO resolves these against the real implementations on
        the parent class and the sister `GenerationHelpersMixin`."""

        project_root: Path
        discovered_tools: List[Dict[str, Any]]

        def _prompt_is_allowed(self, prompt: str) -> bool: ...
        def _workspace_seed_prompts(self) -> List[PromptCandidate]: ...

    _MixinBase = _ParentProtocol
else:
    _MixinBase = object


class PromptSynthesisMixin(_MixinBase):
    """LLM-driven prompt and test-case synthesis."""

    @staticmethod
    def _select_synthesis_client(model_override: Optional[str] = None) -> Optional[Any]:
        """Pick an LLM for prompt synthesis.

        If *model_override* is provided (e.g. ``--synth-model gpt-4o``), it
        takes precedence.  Otherwise the cheapest available provider is used.
        """
        try:
            from evalview.core.llm_provider import LLMClient, detect_available_providers
            from evalview.core.llm_configs import LLMProvider as LLMProviderEnum
        except ImportError:
            return None

        available = detect_available_providers()
        if not available:
            return None

        # User-specified model override
        if model_override:
            try:
                return LLMClient(model=model_override)
            except Exception:
                pass

        available_map = {p.provider.value: p.api_key for p in available}

        for provider_name, model in _SYNTHESIS_PROVIDER_PRIORITY:
            if provider_name in available_map:
                try:
                    return LLMClient(
                        provider=LLMProviderEnum(provider_name),
                        api_key=available_map[provider_name],
                        model=model,
                    )
                except Exception:
                    continue

        # Fallback: first available provider with its default model
        provider, api_key = available[0]
        try:
            return LLMClient(provider=provider, api_key=api_key)
        except Exception:
            return None

    async def _synthesize_prompts(
        self,
        discovery_responses: List[str],
        budget: int,
    ) -> List[PromptCandidate]:
        """Use LLM to synthesize realistic user prompts from agent context.

        Sends a single LLM call with all discovery probe responses (capability
        overview, example requests, domain info), tool schemas, and project
        docs.  The LLM derives the exact domain from this combined context
        before generating prompts — critical for cold start where no existing
        tests or user queries exist.

        Returns an empty list if no LLM provider is available or synthesis
        fails — the generator falls back to heuristic prompts automatically.
        """
        client = self._select_synthesis_client(model_override=getattr(self, "_synth_model_override", None))
        if client is None:
            return []

        n_prompts = min(max(budget - 1, 3), 15)
        user_prompt = self._build_synthesis_user_prompt(discovery_responses, n_prompts)

        try:
            result = await client.chat_completion(
                system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.8,
                max_tokens=2000,
            )
        except Exception as exc:
            logger.debug("Prompt synthesis LLM call failed: %s", exc)
            return []

        prompts_data = result.get("prompts", [])
        if not isinstance(prompts_data, list):
            return []

        candidates: List[PromptCandidate] = []
        for item in prompts_data:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text or len(text) < 8 or len(text) > 300:
                continue
            if not self._prompt_is_allowed(text):
                continue
            category = item.get("category", "happy_path")
            candidates.append(PromptCandidate(text, f"llm_synthesized:{category}"))

        return candidates

    def _build_synthesis_user_prompt(self, discovery_responses: List[str], n_prompts: int) -> str:
        """Build the user-side prompt for the synthesis LLM call.

        Assembles all available cold-start context so the LLM can derive
        the exact domain before generating prompts:

        1. Discovery responses — the agent's own answers to "what can you
           do?", "give me example requests", "what data do you work with?"
        2. Tool schemas — names, descriptions, parameters
        3. Project docs — README, CONTEXT.md, AGENTS.md (if present)
        4. Existing queries — from hand-written tests or captured traffic

        At true cold start, only (1) and (2) are available, which is why
        multi-probe discovery is so important.
        """
        tools_text = self._format_tools_for_synthesis()
        project_context = self._collect_project_context_text()

        parts: List[str] = [f"Generate {n_prompts} realistic user prompts for this agent.\n"]

        # Discovery responses — the primary cold-start signal.
        # Format as Q&A pairs so the LLM can see what questions were asked
        # and trace domain meaning from the agent's own words.
        if discovery_responses:
            qa_pairs: List[str] = []
            for i, response in enumerate(discovery_responses):
                text = (response or "").strip()
                if not text:
                    continue
                question = _DISCOVERY_PROMPTS[i] if i < len(_DISCOVERY_PROMPTS) else "Follow-up"
                qa_pairs.append(f"Q: {question}\nA: {text[:1500]}")
            if qa_pairs:
                parts.append(
                    "AGENT RESPONSES (read ALL of these to understand the exact domain):\n"
                    + "\n\n".join(qa_pairs)
                    + "\n"
                )

        if tools_text:
            parts.append(
                f"TOOLS AVAILABLE (do NOT mention these names in prompts):\n{tools_text}\n"
            )

        if project_context:
            parts.append(f"PROJECT CONTEXT:\n{project_context}\n")

        # Existing queries from hand-written tests / captured traffic.
        workspace_seeds = self._workspace_seed_prompts()
        if workspace_seeds:
            seed_lines = "\n".join(f"- {s.text}" for s in workspace_seeds[:10])
            parts.append(
                f"REAL USER QUERIES (match this style and domain vocabulary):\n{seed_lines}\n"
            )

        parts.append(
            f"Generate exactly {n_prompts} prompts. Each must be a standalone "
            f"user task — something a real user would type to START a work session.\n"
            f"Think: what does a user open this agent to DO on a Monday morning?\n"
        )

        # Build example prompts dynamically from what the agent actually does.
        # The discovery responses above tell you the agent's exact domain —
        # generate examples that match THAT domain, not a generic one.
        parts.append(
            "CRITICAL: Your prompts MUST match the agent's actual domain and capabilities "
            "as described in the AGENT RESPONSES above. Do NOT invent capabilities "
            "the agent doesn't have. Every prompt should be something this specific "
            "agent can realistically handle based on what it told you it does.\n"
        )
        parts.append(
            "Qualities of GOOD prompts:\n"
            "- Specific business tasks the agent told you it handles\n"
            "- Use the same vocabulary and entities the agent mentioned\n"
            "- Vary difficulty: some simple, some multi-step\n"
            "- Include edge cases (missing info, unusual requests)\n"
        )
        parts.append(
            "Qualities of BAD prompts (avoid these):\n"
            "- Capability probes like \"What can you help me with?\"\n"
            "- System artifacts like \"Use the most sensible default and continue\"\n"
            "- Meta-queries like \"Show me example requests\"\n"
            "- Tasks outside the agent's stated domain\n"
        )

        return "\n".join(parts)

    def _format_tools_for_synthesis(self) -> str:
        """Format discovered tool schemas as concise context for the synthesis LLM."""
        if not self.discovered_tools:
            return ""
        lines: List[str] = []
        for tool in self.discovered_tools[:10]:
            name = tool.get("name", "")
            desc = (tool.get("description") or "").strip()
            params = list((tool.get("inputSchema") or {}).get("properties", {}).keys())
            param_text = f" (params: {', '.join(params[:5])})" if params else ""
            lines.append(f"- {name}: {desc}{param_text}")
        return "\n".join(lines)

    def _collect_project_context_text(self) -> str:
        """Gather text from project documentation files for synthesis context."""
        parts: List[str] = []
        for relative_name in _PROJECT_CONTEXT_FILES:
            path = self.project_root / relative_name
            if not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")[:2000]
                parts.append(f"--- {relative_name} ---\n{text}")
            except Exception:
                continue
        return "\n\n".join(parts)[:3000]

    async def _refine_tests_with_llm(
        self,
        tests: List[Any],
        probes: Sequence[ProbeResult],
    ) -> None:
        """Batch-refine test names and output assertions via a single cheap LLM call.

        For each test, the LLM sees the query, tools used, behavior class, and a
        preview of the agent's response, then returns a concise name and 2-3 stable
        phrases that should appear in any correct response (avoiding volatile content
        like numbers, timestamps, or session-specific values).

        Modifies tests in place. Fails silently if no LLM provider is available.
        """
        client = self._select_synthesis_client(model_override=getattr(self, "_synth_model_override", None))
        if client is None or not probes:
            return

        items: List[Dict[str, Any]] = []
        for probe in probes:
            output_preview = (probe.trace.final_output or "")[:400]
            items.append({
                "query": probe.query[:200],
                "tools_used": probe.tools[:5],
                "behavior": probe.behavior_class,
                "output_preview": output_preview,
            })

        system = (
            "You refine test suites for AI agent regression testing. "
            "For each test case, generate a clear name and robust output checks."
        )
        user = (
            f"For each test case, return:\n"
            f"1. name: A concise, descriptive name (3-8 words, no tool names)\n"
            f"2. contains: 1-2 single KEYWORDS or short entity names (1-3 words max) "
            f"that any correct response must mention regardless of exact wording. "
            f"Think topic anchors, not full phrases. Pick a key entity or domain "
            f"term from the query — use just the keyword, not a full phrase. "
            f"For a capabilities query, use the product or service name. "
            f"NEVER use full sentences, long phrases, or wording that could change "
            f"on acceptable rewrites.\n\n"
            f"Test cases:\n{json.dumps(items, indent=2)}\n\n"
            f'Return JSON: {{"tests": [{{"name": "...", "contains": ["...", "..."]}}]}}'
        )

        try:
            result = await client.chat_completion(
                system_prompt=system,
                user_prompt=user,
                temperature=0.3,
                max_tokens=1500,
            )
        except Exception as exc:
            logger.debug("Test refinement LLM call failed: %s", exc)
            return

        refined = result.get("tests", [])
        if not isinstance(refined, list):
            return

        for i, test in enumerate(tests):
            if i >= len(refined):
                break
            r = refined[i]
            if not isinstance(r, dict):
                continue
            # Update name if the LLM produced a reasonable one.
            # Sanitize to alphanumeric + spaces + hyphens (Pydantic constraint).
            name = (r.get("name") or "").strip()
            name = re.sub(r"[^a-zA-Z0-9 \-]", "", name).strip()
            if name and 5 < len(name) < 60:
                test.name = name
            # Update contains with short keyword anchors (not full phrases)
            contains = r.get("contains", [])
            if isinstance(contains, list) and contains:
                # Reject anything longer than ~25 chars — those are phrases, not keywords
                stable = [p for p in contains if isinstance(p, str) and 2 < len(p) < 25]
                if stable and test.expected.output:
                    test.expected.output.contains = stable[:3]

    async def _filter_incoherent_tests(self, tests: List[Any]) -> List[Any]:
        """Drop tests where the prompt intent doesn't match observed behavior.

        Uses a cheap LLM call to check semantic coherence: if a prompt asks
        for one action but the agent performed a completely different one, the
        test is misleading and should be dropped rather than creating a noisy
        regression baseline.
        """
        client = self._select_synthesis_client(model_override=getattr(self, "_synth_model_override", None))
        if client is None or len(tests) <= 1:
            return tests

        items = []
        for test in tests:
            tools = test.expected.tools or []
            items.append({
                "query": test.input.query[:200],
                "tools_used": tools[:5],
                "test_name": test.name,
            })

        try:
            result = await client.chat_completion(
                system_prompt=(
                    "You validate test coherence. For each test, check if the "
                    "user's query semantically matches the tools that were actually "
                    "called. A mismatch means the agent didn't do what the user asked."
                ),
                user_prompt=(
                    "For each test, respond with 'keep' if the query and tools match "
                    "semantically, or 'drop' if they clearly don't match.\n"
                    "Example: query='Process a refund' tools=['lookup_order', 'process_refund'] → keep "
                    "(user asked for refund, agent looked up order and processed it)\n"
                    "Example: query='Delete my account' tools=['search_faq'] → drop "
                    "(user asked for deletion, agent only searched FAQ)\n\n"
                    f"Tests:\n{json.dumps(items, indent=2)}\n\n"
                    'Return JSON: {"verdicts": ["keep", "drop", ...]}'
                ),
                temperature=0.1,
                max_tokens=200,
            )
        except Exception:
            return tests

        verdicts = result.get("verdicts", [])
        if not isinstance(verdicts, list) or len(verdicts) != len(tests):
            return tests

        filtered = [t for t, v in zip(tests, verdicts) if v != "drop"]
        dropped = len(tests) - len(filtered)
        if dropped:
            logger.debug("Coherence filter dropped %d incoherent test(s)", dropped)
        return filtered if filtered else tests  # never drop everything
