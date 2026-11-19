"""OpenAI Codex CLI adapter for skill testing.

Executes skills through the Codex CLI and captures structured traces.
Delegates subprocess management and environment
sanitisation to :class:`CLIAgentAdapter`.

Example::

    config = AgentConfig(type=AgentType.CODEX)
    adapter = CodexAdapter(config)
    trace = await adapter.execute(skill, "Create a React component")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evalview.skills.adapters.base import CLIAgentAdapter, SkillAgentAdapterError
from evalview.skills.agent_types import SkillAgentTrace, TraceEvent, TraceEventType
from evalview.skills.types import Skill


class CodexAdapter(CLIAgentAdapter):
    """Adapter for executing skills through OpenAI Codex CLI.

    Convert completed Codex JSONL items into EvalView skill trace events.
    """

    # -- Required hooks --------------------------------------------------

    @property
    def name(self) -> str:
        return "codex"

    @property
    def binary_name(self) -> str:
        return "codex"

    def _install_hint(self) -> str:
        return (
            "Install Codex CLI: npm install -g @openai/codex\n"
            "Or visit: https://github.com/openai/codex"
        )

    def _candidate_paths(self) -> List[Path]:
        return [
            Path.home() / ".npm-global" / "bin" / "codex",
            Path.home() / ".local" / "bin" / "codex",
            Path("/usr/local/bin") / "codex",
            Path.home() / ".nvm" / "current" / "bin" / "codex",
        ]

    def _build_command(self, skill: Skill, query: str) -> List[str]:
        if self.config.tools is not None or "max_turns" in self.config.model_fields_set:
            raise SkillAgentAdapterError(
                "Codex exec does not support tools or max_turns restrictions. "
                "Omit those settings and use the suite timeout to bound execution.",
                self.name,
            )
        command = [self.cli_path, "exec", "--json", "--ephemeral", "--skip-git-repo-check"]
        if self.config.model:
            command.extend(["--model", self.config.model])
        return [*command, "--", f"{self._format_skill_context(skill)}\nUser request:\n{query}"]

    def _parse_execution_result(
        self, stdout: str, stderr: str, returncode: int, session_id: str, skill_name: str,
        test_name: str, start_time: datetime, end_time: datetime,
    ) -> SkillAgentTrace:
        if returncode:
            raise SkillAgentAdapterError(f"Codex exited with code {returncode}: {stderr[:1000]}", self.name)
        trace = SkillAgentTrace(
            session_id=session_id, skill_name=skill_name, test_name=test_name,
            start_time=start_time, end_time=end_time,
        )
        # Items can complete in a different order than they were invoked.
        items: Dict[str, dict] = {}
        pending = set()
        completed = False
        try:
            for line in stdout.splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                kind = event["type"]
                if kind in ("error", "turn.failed"):
                    detail = event.get("error", event)
                    raise SkillAgentAdapterError(str(detail.get("message", kind)), self.name)
                if kind.startswith("item."):
                    item = event["item"]
                    item_id = item["id"]
                    items[item_id] = item
                    if kind == "item.completed":
                        pending.discard(item_id)
                    else:
                        pending.add(item_id)
                elif kind == "turn.started":
                    completed = False
                elif kind == "turn.completed":
                    completed = True
                    trace.total_input_tokens += event["usage"]["input_tokens"]
                    trace.total_output_tokens += event["usage"]["output_tokens"]
            if not completed or pending:
                raise SkillAgentAdapterError("Codex returned an incomplete execution trace", self.name)
            for item in items.values():
                self._record_item(trace, item)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise SkillAgentAdapterError(f"Invalid Codex JSONL trace: {exc}", self.name) from exc
        return trace

    def _record_item(self, trace: SkillAgentTrace, item: Dict[str, Any]) -> None:
        kind = item["type"]
        if kind == "agent_message":
            trace.final_output = item["text"]
            return
        if kind in ("reasoning", "todo_list", "error"):
            return
        tool = kind
        arguments: Dict[str, Any] = {}
        output = None
        success = item.get("status") == "completed" if "status" in item else None
        if kind == "command_execution":
            command = item["command"]
            arguments = {"command": command}
            trace.commands_ran.append(command)
            output = item.get("aggregated_output")
        elif kind == "file_change":
            arguments = {"changes": item["changes"]}
            if success:
                for change in item["changes"]:
                    path = change["path"]
                    if change["kind"] == "add":
                        trace.files_created.append(path)
                    else:
                        trace.files_modified.append(path)
        elif kind == "mcp_tool_call":
            tool = item["tool"]
            arguments = {"server": item["server"], "arguments": item["arguments"]}
            output = item.get("result")
        elif kind == "web_search":
            arguments = {"query": item.get("query"), "action": item.get("action")}
        elif kind == "collab_tool_call":
            tool = item["tool"]
            arguments = {"receiver_thread_ids": item.get("receiver_thread_ids", [])}
        else:
            raise SkillAgentAdapterError(f"Unsupported Codex trace item: {kind}", self.name)
        trace.tool_calls.append(tool)
        trace.events.append(TraceEvent(
            type=TraceEventType.TOOL_CALL, tool_name=tool, tool_input=arguments,
            tool_output=output, tool_success=success, command_exit_code=item.get("exit_code"),
        ))
