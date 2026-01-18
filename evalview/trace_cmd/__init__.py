"""Trace command module for instrumenting arbitrary Python scripts.

Provides the `evalview trace` command that automatically instruments
OpenAI, Anthropic, and Ollama SDK calls without code changes.

Architecture:
    - collector.py: TraceCollector writes spans to a temp JSONL file
    - patcher.py: Patches SDK clients when imported
    - runner.py: Subprocess launcher with PYTHONPATH injection

Usage:
    evalview trace my_agent.py
    evalview trace -o trace.jsonl my_agent.py arg1 arg2
"""

from evalview.trace_cmd.runner import run_traced_command

__all__ = ["run_traced_command"]
