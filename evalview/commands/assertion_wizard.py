"""Post-capture assertion wizard — suggests smart assertions from captured traffic."""
from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class CaptureAnalysis:
    """Analyzes captured interactions to suggest assertions."""

    def __init__(self, captures: List[Dict[str, Any]]):
        self.captures = captures
        self.tool_sequences: List[List[str]] = []
        self.all_tools: List[str] = []
        self.output_lengths: List[int] = []
        self.latencies: List[float] = []
        self._analyze()

    def _analyze(self) -> None:
        for cap in self.captures:
            tools = cap.get("tools", [])
            self.tool_sequences.append(tools)
            self.all_tools.extend(tools)
            output = cap.get("output", "")
            self.output_lengths.append(len(output))
            if "latency_ms" in cap:
                self.latencies.append(cap["latency_ms"])

    @property
    def agent_type(self) -> str:
        """Detect agent type from captured behavior."""
        avg_tools = statistics.mean([len(s) for s in self.tool_sequences]) if self.tool_sequences else 0
        has_multi_step = any(len(s) > 2 for s in self.tool_sequences)
        unique_tools = len(set(self.all_tools))

        if not self.all_tools:
            return "chat"
        elif has_multi_step and unique_tools > 3:
            return "multi-step"
        elif unique_tools > 1:
            return "tool-use"
        else:
            return "single-tool"

    @property
    def consistent_tool_sequence(self) -> Optional[List[str]]:
        """Return tool sequence if it's consistent across captures."""
        if not self.tool_sequences:
            return None
        # Check if >70% of captures use the same sequence
        seq_strs = [",".join(s) for s in self.tool_sequences if s]
        if not seq_strs:
            return None
        counter = Counter(seq_strs)
        most_common, count = counter.most_common(1)[0]
        if count / len(seq_strs) >= 0.7:
            return most_common.split(",")
        return None

    @property
    def common_tools(self) -> List[str]:
        """Return tools that appear in >50% of captures."""
        if not self.tool_sequences:
            return []
        n = len(self.tool_sequences)
        tool_freq: Counter = Counter()
        for seq in self.tool_sequences:
            for tool in set(seq):  # Count each tool once per capture
                tool_freq[tool] += 1
        return [tool for tool, count in tool_freq.most_common() if count / n > 0.5]

    @property
    def suggested_latency_threshold(self) -> Optional[int]:
        """Suggest latency threshold (p95 * 1.5, rounded up to nearest 1000)."""
        if not self.latencies:
            return None
        sorted_lat = sorted(self.latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
        # Round up to nearest 1000 with 50% headroom
        threshold = int((p95 * 1.5 + 999) // 1000 * 1000)
        return max(threshold, 5000)  # Minimum 5s

    def suggest_output_keywords(self, cap: Dict[str, Any]) -> List[str]:
        """Extract meaningful keywords from a single capture's output."""
        output = cap.get("output", "")
        if not output:
            return []

        keywords: List[str] = []

        # Numbers (likely data the agent fetched)
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", output)
        for n in numbers[:2]:
            if n not in keywords and len(n) > 1:
                keywords.append(n)

        # Proper nouns (entities)
        skip = {"the", "this", "that", "they", "their", "then", "there", "here", "have", "been"}
        proper = re.findall(r"\b[A-Z][a-z]{3,}\b", output)
        for w in proper:
            if len(keywords) >= 4:
                break
            if w.lower() not in skip and w not in keywords:
                keywords.append(w)

        # Domain-specific phrases (quoted, bold, or code)
        quoted = re.findall(r'"([^"]{4,40})"', output)
        for q in quoted[:2]:
            if q not in keywords:
                keywords.append(q)

        return keywords[:5]


class AssertionWizard:
    """Interactive wizard that suggests assertions after capture."""

    def __init__(self, captures: List[Dict[str, Any]]):
        self.captures = captures
        self.analysis = CaptureAnalysis(captures)
        self.suggestions: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Run the interactive wizard. Returns chosen assertion config."""
        import click

        console.print()
        console.print(Panel(
            f"[bold]Assertion Wizard[/bold] — analyzing {len(self.captures)} captured interactions\n\n"
            f"[dim]Agent type detected: [cyan]{self.analysis.agent_type}[/cyan][/dim]",
            border_style="cyan",
            padding=(1, 2),
        ))
        console.print()

        # Show analysis summary
        self._show_analysis()

        # Suggest and confirm assertions
        suggestions = self._build_suggestions()
        accepted = self._confirm_suggestions(suggestions)

        return accepted

    def _show_analysis(self) -> None:
        """Display what we learned from the captured traffic."""
        table = Table(title="Capture Analysis", box=None, show_header=False, padding=(0, 2))
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="cyan")

        table.add_row("Interactions", str(len(self.captures)))
        table.add_row("Agent type", self.analysis.agent_type)

        if self.analysis.all_tools:
            unique = sorted(set(self.analysis.all_tools))
            table.add_row("Tools seen", ", ".join(unique))
            avg_tools = statistics.mean([len(s) for s in self.analysis.tool_sequences])
            table.add_row("Avg tools/call", f"{avg_tools:.1f}")

        if self.analysis.consistent_tool_sequence:
            table.add_row("Consistent sequence", " → ".join(self.analysis.consistent_tool_sequence))

        if self.analysis.latencies:
            avg_lat = statistics.mean(self.analysis.latencies)
            table.add_row("Avg latency", f"{avg_lat:.0f}ms")

        console.print(table)
        console.print()

    def _build_suggestions(self) -> List[Dict[str, Any]]:
        """Build list of suggested assertions based on analysis."""
        suggestions: List[Dict[str, Any]] = []

        # 1. Tool sequence lock (if consistent)
        consistent_seq = self.analysis.consistent_tool_sequence
        if consistent_seq:
            suggestions.append({
                "id": "tool_sequence",
                "label": f"Lock tool sequence: {' → '.join(consistent_seq)}",
                "description": "Your agent consistently uses this tool order. Lock it to catch regressions.",
                "yaml_key": "expected.tool_sequence",
                "value": consistent_seq,
                "recommended": True,
            })

        # 2. Required tools (common tools)
        common = self.analysis.common_tools
        if common:
            suggestions.append({
                "id": "required_tools",
                "label": f"Require tools: {', '.join(common)}",
                "description": "These tools appear in >50% of interactions. Ensure they're always called.",
                "yaml_key": "expected.tools",
                "value": common,
                "recommended": True,
            })

        # 3. Latency threshold
        lat_threshold = self.analysis.suggested_latency_threshold
        if lat_threshold:
            suggestions.append({
                "id": "latency",
                "label": f"Max latency: {lat_threshold}ms",
                "description": f"Based on p95 of your captured traffic with 50% headroom.",
                "yaml_key": "thresholds.max_latency",
                "value": lat_threshold,
                "recommended": True,
            })

        # 4. Output not_contains (always suggest error detection)
        suggestions.append({
            "id": "no_errors",
            "label": "Reject error outputs",
            "description": "Fail if agent returns error messages.",
            "yaml_key": "expected.output.not_contains",
            "value": ["error", "Error", "failed", "exception"],
            "recommended": True,
        })

        # 5. Min score threshold
        suggestions.append({
            "id": "min_score",
            "label": "Minimum quality score: 70",
            "description": "Require at least 70/100 on the evaluation score.",
            "yaml_key": "thresholds.min_score",
            "value": 70,
            "recommended": True,
        })

        return suggestions

    def _confirm_suggestions(self, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Show suggestions and let user accept/reject each."""
        import click

        console.print("[bold]Suggested assertions:[/bold]\n")

        accepted: Dict[str, Any] = {}

        for i, s in enumerate(suggestions, 1):
            rec = " [green](recommended)[/green]" if s["recommended"] else ""
            console.print(f"  [cyan]{i}.[/cyan] {s['label']}{rec}")
            console.print(f"     [dim]{s['description']}[/dim]")

        console.print()

        # Batch accept: recommended by default
        choice = click.prompt(
            "Accept all recommended? [Y]es / [n]o (pick individually) / [s]kip wizard",
            default="y",
            show_default=False,
        ).strip().lower()

        if choice == "s":
            return {}

        if choice in ("y", "yes", ""):
            for s in suggestions:
                if s["recommended"]:
                    accepted[s["id"]] = s
            return accepted

        # Individual selection
        for s in suggestions:
            rec_default = "y" if s["recommended"] else "n"
            answer = click.prompt(
                f"  {s['label']}?",
                default=rec_default,
                show_default=True,
            ).strip().lower()
            if answer in ("y", "yes"):
                accepted[s["id"]] = s

        return accepted


def apply_wizard_to_yaml(yaml_path: str, accepted: Dict[str, Any]) -> None:
    """Rewrite a captured test YAML to include wizard-suggested assertions."""
    from pathlib import Path

    path = Path(yaml_path)
    if not path.exists():
        return

    content = path.read_text()
    lines = content.split("\n")
    new_lines: List[str] = []

    # Track what we need to add
    needs_tool_sequence = "tool_sequence" in accepted
    needs_required_tools = "required_tools" in accepted
    needs_no_errors = "no_errors" in accepted
    needs_latency = "latency" in accepted
    needs_min_score = "min_score" in accepted

    in_expected = False
    in_thresholds = False
    added_tools = False
    added_not_contains = False

    for line in lines:
        stripped = line.strip()

        # Track sections
        if stripped.startswith("expected:"):
            in_expected = True
            in_thresholds = False
            new_lines.append(line)

            # Add tool sequence right after expected:
            if needs_tool_sequence and not added_tools:
                seq = accepted["tool_sequence"]["value"]
                new_lines.append("  tool_sequence:")
                for tool in seq:
                    new_lines.append(f"    - {tool}")
                added_tools = True

            # Add required tools
            if needs_required_tools and not needs_tool_sequence and not added_tools:
                tools = accepted["required_tools"]["value"]
                new_lines.append("  tools:")
                for tool in tools:
                    new_lines.append(f"    - {tool}")
                added_tools = True

            continue

        if stripped.startswith("thresholds:"):
            in_expected = False
            in_thresholds = True
            new_lines.append(line)

            if needs_latency:
                lat = accepted["latency"]["value"]
                new_lines.append(f"  max_latency: {lat}")
            if needs_min_score:
                new_lines.append(f"  min_score: {accepted['min_score']['value']}")
            continue

        # Replace existing not_contains with enriched version
        if stripped.startswith("not_contains:") and needs_no_errors and not added_not_contains:
            new_lines.append(line)
            values = accepted["no_errors"]["value"]
            for v in values:
                new_lines.append(f'      - "{v}"')
            added_not_contains = True
            # Skip the old not_contains entries
            continue

        # Skip old threshold values if we're replacing them
        if in_thresholds and (stripped.startswith("max_latency:") or stripped.startswith("min_score:")):
            continue

        # Skip old not_contains entries (single value '- "error"')
        if added_not_contains and stripped.startswith('- "error"'):
            continue

        new_lines.append(line)

    path.write_text("\n".join(new_lines))


def enhance_captured_tests(
    captures: List[Dict[str, Any]],
    output_dir: str,
    saved_files: List[str],
) -> None:
    """Main entry point: run wizard and apply to saved test files."""
    if not captures or not saved_files:
        return

    wizard = AssertionWizard(captures)
    accepted = wizard.run()

    if not accepted:
        console.print("[dim]Skipped assertion wizard — using default assertions.[/dim]\n")
        return

    # Apply to all saved files
    for filepath in saved_files:
        apply_wizard_to_yaml(filepath, accepted)

    n = len(accepted)
    console.print(f"\n[green]Applied {n} assertion{'s' if n != 1 else ''} to {len(saved_files)} test file{'s' if len(saved_files) != 1 else ''}[/green]")
    console.print("[dim]Review the YAML files to fine-tune — these are smart defaults from your real traffic.[/dim]\n")
