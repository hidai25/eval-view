"""Aider CLI adapter for EvalView.

Aider (https://aider.chat) is an open-source AI pair programmer that edits
files in a git repo from a natural-language prompt. This adapter runs Aider
non-interactively via ``aider --message`` and parses the resulting output
plus file diffs into an ExecutionTrace.

Requirements:
    - Aider installed: pip install aider-chat
    - API key for the chosen provider (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY)

Usage in YAML test cases:
    adapter: aider
    adapter_config:
      model: sonnet          # any aider model string
      timeout: 180

    input:
      query: "Fix the off-by-one bug in find_max()"
      context:
        cwd: demo/fixtures/aider
        files: [buggy.py]
        reset_files: true    # snapshot + restore files around the run
"""

import asyncio
import difflib
import logging
import os
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evalview.adapters.base import AgentAdapter
from evalview.core.tracing import Tracer
from evalview.core.types import (
    ExecutionMetrics,
    ExecutionTrace,
    SpanKind,
    StepMetrics,
    StepTrace,
    TokenUsage,
)

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_APPLIED_EDIT_RE = re.compile(r"^Applied edit to (.+)$", re.MULTILINE)
_TOKENS_RE = re.compile(
    r"Tokens:\s*([\d.,]+)\s*sent,\s*([\d.,]+)\s*received", re.IGNORECASE
)
_COST_RE = re.compile(r"Cost:\s*\$([\d.]+)\s*message", re.IGNORECASE)


class AiderAdapter(AgentAdapter):
    """Adapter for the Aider AI pair programmer.

    Runs ``aider --message <query>`` with a fixed set of flags that make the
    session deterministic (no streaming, no auto-commit, no repo map, no
    network side effects) and parses:

    * ``Applied edit to <file>`` markers + before/after file snapshots for
      ``edit_file`` steps (with the unified diff as the step output)
    * Positional file arguments for implicit ``read_file`` steps
    * ``Tokens: X sent, Y received`` line for token usage
    * ``Cost: $X.XXX message`` line for per-run cost

    If ``reset_files`` is truthy in the context, the adapter snapshots the
    contents of every file in ``cwd`` before the run and restores them
    afterwards. This makes repeated runs idempotent — required for drift
    detection on coding tasks.
    """

    def __init__(
        self,
        endpoint: str = "",  # unused; Aider runs locally
        timeout: float = 300.0,
        model: Optional[str] = None,
        cwd: Optional[str] = None,
        aider_path: Optional[str] = None,
        reset_files: bool = True,
        **kwargs: Any,
    ) -> None:
        self.timeout = timeout
        self.model = model
        self.cwd = cwd
        # Resolve aider binary path: explicit arg > AIDER_PATH env var > "aider" on PATH.
        self.aider_path = aider_path or os.getenv("AIDER_PATH") or "aider"
        self.reset_files = reset_files
        self._last_raw_output: Optional[str] = None

    @property
    def name(self) -> str:
        return "aider"

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        context = context or {}

        model = context.get("model", self.model)
        if not model:
            raise ValueError(
                "AiderAdapter requires a model. Set it in adapter_config.model "
                "or context.model. Examples: sonnet, gpt-4o, deepseek"
            )

        cwd = context.get("cwd", self.cwd)
        if cwd:
            cwd = os.path.abspath(os.path.expanduser(cwd))
        if not cwd:
            raise ValueError("AiderAdapter requires a cwd (working directory)")

        files = list(context.get("files") or [])
        reset = context.get("reset_files", self.reset_files)

        snapshots: Dict[Path, bytes] = {}
        if reset:
            snapshots = self._snapshot_dir(Path(cwd))

        cmd = self._build_command(query, model, files)
        logger.info("Aider command: %s (cwd=%s)", " ".join(cmd), cwd)

        start_time = datetime.now()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=self.timeout,
                    env=self._build_env(context),
                ),
            )
            end_time = datetime.now()
            self._last_raw_output = result.stdout

            if result.stderr:
                logger.debug("Aider stderr: %s", result.stderr)

            post_state = self._read_files(cwd, snapshots) if reset else {}
            trace = self._build_trace(
                stdout=_ANSI_RE.sub("", result.stdout),
                stderr=result.stderr,
                returncode=result.returncode,
                start_time=start_time,
                end_time=end_time,
                files=files,
                cwd=cwd,
                pre_snapshots=snapshots,
                post_snapshots=post_state,
            )
            return trace

        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            return self._error_trace(
                f"Aider timed out after {self.timeout}s", start_time, end_time
            )
        except FileNotFoundError:
            end_time = datetime.now()
            return self._error_trace(
                "Aider CLI not found. Install with: pip install aider-chat",
                start_time,
                end_time,
            )
        except Exception as exc:
            end_time = datetime.now()
            return self._error_trace(str(exc), start_time, end_time)
        finally:
            if reset and snapshots:
                self._restore_dir(snapshots)

    # ------------------------------------------------------------------
    # Command + environment
    # ------------------------------------------------------------------

    def _build_command(
        self, query: str, model: str, files: List[str]
    ) -> List[str]:
        cmd = [
            self.aider_path,
            "--message", query,
            "--model", model,
            "--yes-always",
            "--no-pretty",
            "--no-stream",
            "--no-auto-commits",
            "--no-dirty-commits",
            "--no-detect-urls",
            "--no-show-release-notes",
            "--no-check-update",
            "--no-analytics",
            "--no-git",
            "--no-gitignore",
            "--map-tokens", "0",
        ]
        cmd.extend(files)
        return cmd

    def _build_env(self, context: Dict[str, Any]) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("AIDER_ANALYTICS", "false")
        # Merge .env.local / .env from the caller's CWD (not the subprocess cwd),
        # matching the OpenCode adapter convention so API keys travel to Aider.
        for candidate in [Path(".env.local"), Path(".env")]:
            if candidate.exists():
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
                break
        if "env" in context:
            env.update(context["env"])
        return env

    # ------------------------------------------------------------------
    # File snapshot / restore
    # ------------------------------------------------------------------

    def _snapshot_dir(self, root: Path) -> Dict[Path, bytes]:
        """Read every file under ``root`` into memory for later restore."""
        if not root.exists() or not root.is_dir():
            return {}
        snapshots: Dict[Path, bytes] = {}
        for path in root.rglob("*"):
            if path.is_file() and not self._is_ignored(path):
                try:
                    snapshots[path] = path.read_bytes()
                except OSError as exc:
                    logger.debug("Skipping %s during snapshot: %s", path, exc)
        return snapshots

    def _restore_dir(self, snapshots: Dict[Path, bytes]) -> None:
        """Write snapshotted contents back; delete any files Aider created."""
        seen: set = set()
        for path, content in snapshots.items():
            try:
                path.write_bytes(content)
                seen.add(path)
            except OSError as exc:
                logger.warning("Failed to restore %s: %s", path, exc)
        # Remove files that didn't exist before the run
        if snapshots:
            common_root = Path(os.path.commonpath([str(p) for p in snapshots]))
            if common_root.is_dir():
                for path in common_root.rglob("*"):
                    if path.is_file() and path not in seen and not self._is_ignored(path):
                        try:
                            path.unlink()
                        except OSError:
                            pass

    def _read_files(
        self, cwd: str, pre_snapshots: Dict[Path, bytes]
    ) -> Dict[Path, bytes]:
        post: Dict[Path, bytes] = {}
        for path in pre_snapshots:
            try:
                post[path] = path.read_bytes() if path.exists() else b""
            except OSError:
                post[path] = b""
        return post

    @staticmethod
    def _is_ignored(path: Path) -> bool:
        parts = set(path.parts)
        return bool(
            parts & {".git", "__pycache__", ".evalview", ".mypy_cache", "venv", "node_modules"}
        )

    # ------------------------------------------------------------------
    # Trace building
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        start_time: datetime,
        end_time: datetime,
        files: List[str],
        cwd: str,
        pre_snapshots: Dict[Path, bytes],
        post_snapshots: Dict[Path, bytes],
    ) -> ExecutionTrace:
        session_id = f"aider-{uuid.uuid4().hex[:8]}"
        total_latency_ms = (end_time - start_time).total_seconds() * 1000

        steps: List[StepTrace] = []

        # Implicit read_file for each positional file argument
        for idx, fname in enumerate(files):
            steps.append(
                StepTrace(
                    step_id=f"read-{idx}",
                    step_name=f"read_file ({fname})",
                    tool_name="read_file",
                    parameters={"path": fname},
                    output="",
                    success=True,
                    error=None,
                    metrics=StepMetrics(),
                )
            )

        # edit_file for each modified file (based on byte-level diff)
        edit_idx = 0
        diffs = self._compute_diffs(pre_snapshots, post_snapshots, cwd)
        applied_edits = set(_APPLIED_EDIT_RE.findall(stdout))
        for rel_path, diff_text in diffs:
            steps.append(
                StepTrace(
                    step_id=f"edit-{edit_idx}",
                    step_name=f"edit_file ({rel_path})",
                    tool_name="edit_file",
                    parameters={"path": rel_path},
                    output=diff_text,
                    success=True,
                    error=None,
                    metrics=StepMetrics(),
                )
            )
            edit_idx += 1

        # Fallback: Aider reported edits but we didn't snapshot → record anyway
        if not diffs and applied_edits:
            for idx, fname in enumerate(sorted(applied_edits)):
                steps.append(
                    StepTrace(
                        step_id=f"edit-{idx}",
                        step_name=f"edit_file ({fname})",
                        tool_name="edit_file",
                        parameters={"path": fname},
                        output="",
                        success=True,
                        error=None,
                        metrics=StepMetrics(),
                    )
                )

        final_output = self._extract_final_output(stdout)
        if returncode != 0 and not final_output:
            final_output = stderr.strip() or f"Aider exited with code {returncode}"

        tokens = self._extract_tokens(stdout)
        cost = self._extract_cost(stdout)

        tracer = Tracer()
        with tracer.start_span("Aider Execution", SpanKind.AGENT):
            for step in steps:
                tracer.record_tool_call(
                    tool_name=step.tool_name,
                    parameters=step.parameters,
                    result=step.output,
                    error=step.error,
                    duration_ms=step.metrics.latency if step.metrics else 0.0,
                )

        return ExecutionTrace(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=ExecutionMetrics(
                total_cost=cost,
                total_latency=total_latency_ms,
                total_tokens=tokens,
            ),
            trace_context=tracer.build_trace_context(),
        )

    def _compute_diffs(
        self,
        pre_snapshots: Dict[Path, bytes],
        post_snapshots: Dict[Path, bytes],
        cwd: str,
    ) -> List[Tuple[str, str]]:
        diffs: List[Tuple[str, str]] = []
        cwd_path = Path(cwd)
        # Files that existed before and changed
        for path, pre in pre_snapshots.items():
            post = post_snapshots.get(path, b"")
            if pre == post:
                continue
            rel = str(path.relative_to(cwd_path)) if path.is_relative_to(cwd_path) else str(path)
            diff = self._unified_diff(pre, post, rel)
            if diff:
                diffs.append((rel, diff))
        return diffs

    @staticmethod
    def _unified_diff(pre: bytes, post: bytes, rel: str) -> str:
        try:
            pre_text = pre.decode("utf-8")
            post_text = post.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary file changed: {rel}>"
        diff_lines = difflib.unified_diff(
            pre_text.splitlines(keepends=True),
            post_text.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=3,
        )
        return "".join(diff_lines)

    # ------------------------------------------------------------------
    # Text parsing
    # ------------------------------------------------------------------

    def _extract_final_output(self, stdout: str) -> str:
        """Return the last meaningful assistant block from Aider's stdout.

        Strategy: drop Aider's banner/status lines, drop SEARCH/REPLACE
        fences, keep the last ~20 non-empty paragraphs.
        """
        lines = stdout.splitlines()
        skip_prefixes = (
            "Aider v",
            "Main model:",
            "Weak model:",
            "Git repo:",
            "Repo-map:",
            "Added ",
            "Applied edit",
            "Committing",
            "Commit ",
            "Tokens:",
            "Cost:",
            "Warning:",
            "Model ",
            "Use ",
            "You can",
            "https://",
            "Can't initialize",
            "Initial repo scan",
        )
        in_fence = False
        cleaned: List[str] = []
        for raw in lines:
            line = raw.rstrip()
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if not line.strip():
                cleaned.append("")
                continue
            if any(line.strip().startswith(p) for p in skip_prefixes):
                continue
            if line.strip() in {"<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"}:
                continue
            cleaned.append(line)
        text = "\n".join(cleaned).strip()
        # Trim to last ~3000 chars to keep golden files small
        if len(text) > 3000:
            text = text[-3000:]
        return text

    def _extract_tokens(self, stdout: str) -> Optional[TokenUsage]:
        match = _TOKENS_RE.search(stdout)
        if not match:
            return None
        try:
            sent = int(float(match.group(1).replace(",", "")))
            received = int(float(match.group(2).replace(",", "")))
        except ValueError:
            return None
        return TokenUsage(input_tokens=sent, output_tokens=received)

    def _extract_cost(self, stdout: str) -> float:
        match = _COST_RE.search(stdout)
        if not match:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _error_trace(
        self, message: str, start_time: datetime, end_time: datetime
    ) -> ExecutionTrace:
        return ExecutionTrace(
            session_id=f"aider-error-{uuid.uuid4().hex[:8]}",
            start_time=start_time,
            end_time=end_time,
            steps=[
                StepTrace(
                    step_id="error",
                    step_name="Error",
                    tool_name="error",
                    parameters={},
                    output=message,
                    success=False,
                    error=message,
                    metrics=StepMetrics(),
                )
            ],
            final_output=message,
            metrics=ExecutionMetrics(
                total_cost=0.0,
                total_latency=(end_time - start_time).total_seconds() * 1000,
                total_tokens=None,
            ),
        )

    async def health_check(self) -> bool:
        try:
            result = subprocess.run(
                [self.aider_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception as exc:
            logger.warning("Aider health check failed: %s", exc)
            return False
