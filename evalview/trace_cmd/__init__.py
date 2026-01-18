"""Trace command module for instrumenting arbitrary Python scripts.

Provides the `evalview trace -- python script.py` command that automatically
instruments OpenAI and Anthropic SDK calls without code changes.

Architecture:
    - collector.py: TraceCollector writes spans to a temp JSONL file
    - patcher.py: Import hook that patches SDK clients when imported
    - runner.py: Subprocess launcher with PYTHONPATH injection

Usage:
    evalview trace -- python my_agent.py
    evalview trace --output trace.jsonl -- python my_agent.py
"""

from evalview.trace_cmd.runner import run_traced_command

__all__ = ["run_traced_command"]
