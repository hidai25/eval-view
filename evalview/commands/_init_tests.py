"""Test-suite generation helpers used by `evalview init`.

Three flows live here:
- _autogen_tests: lightweight HTTP probing into hand-written YAML
- _generate_init_draft_suite: full generation engine, isolated draft folder
- _write_init_suite + _print_generated_test_preview: writing/previewing output

Extracted from init_cmd.py so the main init flow stays focused on flag
handling and orchestration.
"""
from __future__ import annotations

import re
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml  # type: ignore[import-untyped]

from evalview.commands.shared import console
from evalview.core.adapter_factory import create_adapter
from evalview.test_generation import AgentTestGenerator, run_generation


def _autogen_tests(endpoint: str, tests_dir: Path) -> int:
    """Probe the agent and generate test YAML files from real responses."""
    _FRAGMENT_ENDINGS = (
        " for", " the", " a", " an", " of", " in", " on", " to",
        " with", " and", " or", " e.g.", "(e.g.",
    )

    def _extract_example_queries(text: str) -> List[str]:
        quoted = re.findall(r'["“”]([^"“”]{20,80})["“”]', text)
        bulleted = re.findall(r'[-•]\s+"?([A-Z][^"\n]{20,80})"?\s*$', text, re.MULTILINE)
        candidates = quoted + bulleted
        valid = []
        for q in candidates:
            q = q.strip().rstrip(",.")
            words = q.split()
            if len(words) < 3:
                continue
            if q.lower().endswith(_FRAGMENT_ENDINGS):
                continue
            if "?" in q or q[0].isupper():
                valid.append(q)
        return valid[:3]

    def _stable_phrases(text: str) -> List[str]:
        lines = [ln.strip().lstrip("#*• ") for ln in text.splitlines() if ln.strip()]
        phrases = []
        for line in lines[:5]:
            clean = re.sub(r"\*+|`", "", line).strip()
            if "http" in clean or len(clean) < 4:
                continue
            fragment = clean[:40].strip()
            if fragment:
                phrases.append(fragment)
                break
        return phrases

    def _probe(query: str) -> Optional[Dict[str, Any]]:
        try:
            r = httpx.post(endpoint, json={"query": query}, timeout=30.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _write_test(name: str, query: str, data: Dict[str, Any]) -> bool:
        query = query.strip()
        if len(query) < 10 or len(query.split()) < 3 or query.lower().endswith(_FRAGMENT_ENDINGS):
            return False

        output = data.get("output", "")
        tool_calls = data.get("tool_calls", [])
        tools = [tc["name"] for tc in tool_calls if isinstance(tc, dict)]
        phrases = _stable_phrases(output)

        tools_yaml = ""
        if tools:
            tools_list = "\n".join(f"    - {t}" for t in tools)
            tools_yaml = f"  tools:\n{tools_list}\n"

        contains_yaml = ""
        if phrases:
            contains_list = "\n".join(f'      - "{p}"' for p in phrases)
            contains_yaml = f"    contains:\n{contains_list}\n"

        content = f"""name: "{name}"
description: "Auto-generated from real agent response"
generated: true

endpoint: {endpoint}
adapter: http

input:
  query: "{query}"

expected:
{tools_yaml}  output:
{contains_yaml}    not_contains:
      - "error"
      - "Error"

thresholds:
  min_score: 60
  max_latency: 30000
"""
        safe_name = re.sub(r"[^a-z0-9-]", "-", name.lower())[:40]
        path = tests_dir / f"{safe_name}.yaml"
        path.write_text(content)
        return True

    tests_dir.mkdir(parents=True, exist_ok=True)

    console.print("[dim]  Sending capability probe...[/dim]")
    cap_data = _probe("Hello, what can you help me with?")
    if not cap_data:
        return 0

    generated = 0

    if _write_test("what-can-you-do", "Hello, what can you help me with?", cap_data):
        generated += 1

    examples = _extract_example_queries(cap_data.get("output", ""))
    for i, query in enumerate(examples[:2]):
        console.print(f"[dim]  Probing: {query[:50]}...[/dim]")
        data = _probe(query)
        if data:
            name = f"test-{i + 2}"
            if _write_test(name, query, data):
                generated += 1

    return generated


def _generate_init_draft_suite(
    endpoint: str,
    out_dir: Path,
    budget: int = 8,
    synth_model: Optional[str] = None,
) -> tuple[int, dict[str, Any], list]:
    """Generate an isolated draft suite for onboarding.

    Uses the same generation engine as `evalview generate`, but writes into a
    dedicated folder so first-run onboarding does not mix with stale tests.
    Timeout is 120s to accommodate LLM-backed agents (e.g. Claude Sonnet).

    Returns (count, report, tests) — tests are NOT written to disk yet so the
    caller can show them for review and ask for approval first.
    """
    adapter = create_adapter(
        adapter_type="http",
        endpoint=endpoint,
        timeout=120.0,
        allow_private_urls=True,
    )
    _gen_start = _time.time()
    _gen_state: dict = {"phase": f"Probing [1/{budget}]...", "completed": 0, "total": budget, "stop": False}

    def _timer_thread() -> None:
        while not _gen_state["stop"]:
            elapsed = _time.time() - _gen_start
            mins, secs = divmod(elapsed, 60)
            ts = f"{int(mins):02d}:{int(secs):02d}"
            n = _gen_state["completed"]
            t = _gen_state["total"] or "?"
            phase = _gen_state["phase"]
            try:
                console.file.write(f"\r\033[K  ⏱  {ts}  [{n}/{t}]  {phase}")
                console.file.flush()
            except Exception:
                pass
            _time.sleep(0.25)
        try:
            console.file.write("\r\033[K")
            console.file.flush()
        except Exception:
            pass

    def _on_probe(num: int, total: int, query: str, status: str, tools: list) -> None:
        _gen_state["total"] = total
        if status == "info":
            _gen_state["phase"] = query
            return
        try:
            console.file.write("\r\033[K")
            console.file.flush()
        except Exception:
            pass
        _gen_state["completed"] = num
        if status == "fail":
            console.print(f"[dim]  [red]✗[/red] [{num}/{total}] {query}[/dim]")
            # Store first failure for later display
            if "first_error" not in _gen_state:
                _gen_state["first_error"] = query
        elif tools:
            console.print(f"[dim]  [green]✓[/green] [{num}/{total}] {query} → {', '.join(tools[:3])}[/dim]")
        else:
            console.print(f"[dim]  [green]✓[/green] [{num}/{total}] {query}[/dim]")
        if num < total:
            _gen_state["phase"] = f"Probing [{num + 1}/{total}]..."
        else:
            _gen_state["phase"] = "Building tests..."

    timer = threading.Thread(target=_timer_thread, daemon=True)
    timer.start()

    result = run_generation(
        adapter=adapter,
        endpoint=endpoint,
        adapter_type="http",
        budget=budget,
        allow_live_side_effects=False,
        on_probe_complete=_on_probe,
        synth_model=synth_model,
    )

    _gen_state["stop"] = True
    if timer.is_alive():
        timer.join(timeout=2)
    if not result.tests:
        return 0, result.report, []

    approved_at = datetime.now(timezone.utc).isoformat()
    for test_case in result.tests:
        meta = dict(test_case.meta or {})
        meta["generated_by"] = "evalview init"
        meta["review_status"] = "approved"
        meta["approved_at"] = approved_at
        test_case.meta = meta
        test_case.thresholds.min_score = min(test_case.thresholds.min_score, 50.0)
        test_case.thresholds.max_latency = None

    return len(result.tests), result.report, result.tests


def _write_init_suite(tests: list, out_dir: Path, endpoint: str) -> None:
    """Write approved init tests to disk."""
    from evalview.test_generation import GenerationResult

    result = GenerationResult(tests=tests)
    generator = AgentTestGenerator(
        adapter=None,
        endpoint=endpoint,
        adapter_type="http",
        allow_live_side_effects=False,
    )
    generator.write_suite(result, out_dir, replace_existing=True)


def _print_generated_test_preview(tests_dir: Path, max_files: int = 1) -> None:
    """Print generated YAML inline so users can see the draft immediately."""
    yaml_files = sorted(
        [path for path in tests_dir.glob("*.yaml") if path.is_file()]
    )
    if not yaml_files:
        return

    console.print()
    console.print("[bold]Generated Test Preview[/bold]")
    for path in yaml_files[:max_files]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        meta = data.get("meta") or {}
        behavior = str(meta.get("behavior_class") or "unknown").replace("_", " ")
        turns = data.get("turns") or []
        turn_label = f"{len(turns)} turns" if turns else "single turn"
        console.print(f"[dim]{path}[/dim]")
        console.print(f"[dim]Behavior: {behavior} | {turn_label}[/dim]")
        console.print(path.read_text(encoding="utf-8").rstrip())
        console.print()
    if len(yaml_files) > max_files:
        console.print(f"[dim]+ {len(yaml_files) - max_files} more generated test file(s)[/dim]\n")
