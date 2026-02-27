#!/usr/bin/env python3
"""Custom runner for procrastination-buster skill tests.

Uses the claude CLI instead of the Anthropic SDK so it works for
both API key users and Claude Code OAuth users.
"""
import json, os, sys, subprocess, tempfile, traceback, shutil

try:
    claude_path = shutil.which("claude")
    if not claude_path:
        print("RUNNER ERROR: claude CLI not found in PATH", file=sys.stderr)
        sys.exit(1)

    skill_path = os.environ.get("SKILL_PATH", "")
    query = os.environ.get("QUERY", "")

    skill_content = open(skill_path).read() if skill_path and os.path.exists(skill_path) else ""
    system_prompt = (
        f"You have the following skill loaded:\n\n{skill_content}\n\n"
        "Apply this skill when responding."
    )

    # Strip Claude Code session markers + any inherited session-scoped auth token
    # so the inner claude falls back to ~/.claude.json credentials.
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.pop("ANTHROPIC_API_KEY", None)

    stdout_path = tempfile.mktemp(suffix=".stdout")
    stderr_path = tempfile.mktemp(suffix=".stderr")

    try:
        with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
            proc = subprocess.Popen(
                [
                    claude_path, "--print",
                    "-p", query,
                    "--append-system-prompt", system_prompt,
                    "--output-format", "stream-json",
                    "--verbose",
                    "--dangerously-skip-permissions",
                ],
                stdin=subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                env=env,
                start_new_session=True,
            )
            proc.wait(timeout=60)

        stdout = open(stdout_path, "r", errors="replace").read()
    finally:
        for p in (stdout_path, stderr_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    # Parse stream-json to extract final output and token counts
    final_output = ""
    input_tokens = 0
    output_tokens = 0

    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if data.get("type") == "result":
                final_output = data.get("result", "")
                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
        except json.JSONDecodeError:
            pass

    print(json.dumps({
        "output": final_output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }))

except Exception as e:
    print(f"RUNNER ERROR: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
