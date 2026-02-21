#!/usr/bin/env python3
import json, os
import anthropic

skill_path = os.environ.get("SKILL_PATH", "")
query = os.environ.get("QUERY", "")

skill_content = open(skill_path).read() if skill_path and os.path.exists(skill_path) else ""

client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    system=f"You have the following skill loaded:\n\n{skill_content}\n\nApply this skill when responding.",
    messages=[{"role": "user", "content": query}],
)
print(json.dumps({
    "output": msg.content[0].text,
    "input_tokens": msg.usage.input_tokens,
    "output_tokens": msg.usage.output_tokens,
}))
