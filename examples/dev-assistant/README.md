# Dev Assistant Example

Test a code generation and development assistant agent.

## Setup

### Option A: OpenAI Code Interpreter

Use OpenAI Assistants API with code interpreter enabled.

```bash
# Set your API key
export OPENAI_API_KEY=your-key

# See openai-assistants example for full setup
```

### Option B: Aider

```bash
pip install aider-chat

# Run aider in API mode
aider --api-server
```

### Option C: Continue.dev or Cursor-style Agent

Any coding assistant that exposes an HTTP API.

### Option D: Custom Code Agent

Build with LangChain or LlamaIndex:

```bash
# LangChain code agent
git clone https://github.com/langchain-ai/langchain.git
cd langchain/cookbook/code-analysis
```

### Run EvalView Test

```bash
evalview run --pattern examples/dev-assistant/test-case.yaml
```

## Links

- **Aider**: https://github.com/paul-gauthier/aider
- **OpenAI Assistants**: https://platform.openai.com/docs/assistants
- **LangChain Code**: https://python.langchain.com/docs/tutorials/code_analysis/
