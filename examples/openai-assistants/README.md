# OpenAI Assistants Example

Test an OpenAI Assistants API agent with EvalView.

## Setup

### 1. Create an Assistant

Go to https://platform.openai.com/assistants and create an assistant, or use the API:

```python
from openai import OpenAI
client = OpenAI()

assistant = client.beta.assistants.create(
    name="Test Assistant",
    instructions="You are a helpful assistant that can search the web and analyze data.",
    model="gpt-4o",
    tools=[
        {"type": "code_interpreter"},
        {"type": "file_search"}
    ]
)

print(f"Assistant ID: {assistant.id}")
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY=your-api-key
export OPENAI_ASSISTANT_ID=asst_xxxxx  # From step 1
```

### 3. Run a Simple Server (Optional)

If you need an HTTP wrapper:

```python
# server.py
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=data['query']
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=os.environ['OPENAI_ASSISTANT_ID']
    )
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return jsonify({"output": messages.data[0].content[0].text.value})

if __name__ == '__main__':
    app.run(port=8000)
```

### 4. Run EvalView Test

```bash
evalview run --pattern examples/openai-assistants/test-case.yaml
```

## Links

- **Assistants API**: https://platform.openai.com/docs/assistants/overview
- **Playground**: https://platform.openai.com/assistants
- **Quickstart**: https://platform.openai.com/docs/assistants/quickstart
