# AutoGen Example

Test a Microsoft AutoGen multi-agent conversation with EvalView.

## Setup

### 1. Install AutoGen

```bash
pip install autogen-agentchat
```

### 2. Clone AutoGen Examples

```bash
git clone https://github.com/microsoft/autogen.git
cd autogen/samples
```

### 3. Run Example Agent

```python
# simple_server.py
from autogen import AssistantAgent, UserProxyAgent
from flask import Flask, request, jsonify

app = Flask(__name__)

config_list = [{"model": "gpt-4o", "api_key": "your-key"}]

assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    response = user_proxy.initiate_chat(assistant, message=data['query'], max_turns=3)
    return jsonify({"output": response.summary, "steps": []})

if __name__ == '__main__':
    app.run(port=8000)
```

```bash
python simple_server.py
```

### 4. Run EvalView Test

```bash
evalview run --pattern examples/autogen/test-case.yaml
```

## Links

- **Repo**: https://github.com/microsoft/autogen
- **Docs**: https://microsoft.github.io/autogen/
- **Examples**: https://github.com/microsoft/autogen/tree/main/samples
- **Quickstart**: https://microsoft.github.io/autogen/docs/Getting-Started
