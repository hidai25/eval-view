# OpenAI Example — Testing OpenAI Responses API Agents with EvalView

> Test OpenAI agents built on the Responses API with EvalView — verify hosted tool usage, output quality, and detect regressions.

> **Migration note:** This example previously targeted the OpenAI Assistants
> API, which OpenAI removed on **August 26, 2026**. The `openai-assistants`
> adapter keeps its name for config compatibility, but it now runs on the
> Responses API (with isolated Conversations and replayed prior messages). There is no
> assistant to create anymore — model, instructions, and tools are configured
> per-request from your EvalView config.

> **Installation:** This example requires the **unreleased development source**.
> PyPI 0.8.1 still uses Assistants. Follow the [migration guide](../../docs/OPENAI_MIGRATION.md)
> to preserve your agent settings. Custom function execution requires your own
> agent loop, connected through the HTTP adapter.

## Setup

### 1. Set Environment Variables

```bash
export OPENAI_API_KEY=your-api-key
```

That's it — no assistant ID needed. Optionally, if you manage a reusable
Prompt object in the OpenAI dashboard (https://platform.openai.com/prompts),
you can point the adapter at it:

```bash
export OPENAI_PROMPT_ID=pmpt_xxxxx  # Optional
```

### 2. Configure the Adapter

Model, instructions, and tools live in `.evalview/config.yaml`:

```yaml
adapter: openai-assistants  # Historical name — runs on the Responses API
timeout: 120

model:
  name: gpt-4o

# Optional: system instructions for the agent under test
# instructions: "You are a helpful technical assistant."

# Optional: built-in tools (defaults to code_interpreter)
# tools:
#   - code_interpreter
```

### 3. Run a Simple Server (Optional)

If you need an HTTP wrapper around your own agent:

```python
# server.py
import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    conversation = client.conversations.create()
    response = client.responses.create(
        model="gpt-4o",
        conversation=conversation.id,
        input=data['query'],
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    )
    return jsonify({"output": response.output_text})

if __name__ == '__main__':
    app.run(port=8000)
```

### 4. Run EvalView Test

```bash
evalview run --pattern examples/openai-assistants/test-case.yaml
```

## Links

- **Responses API**: https://platform.openai.com/docs/api-reference/responses
- **Conversations API**: https://platform.openai.com/docs/api-reference/conversations
- **Assistants migration guide**: https://platform.openai.com/docs/assistants/migration
