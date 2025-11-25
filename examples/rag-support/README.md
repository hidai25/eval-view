# RAG Support Agent Example

Test a customer support agent with RAG (Retrieval Augmented Generation).

## Setup

### Option A: LangChain RAG Template

```bash
# Install dependencies
pip install langchain langchain-openai chromadb

# Clone LangChain templates
git clone https://github.com/langchain-ai/langchain.git
cd langchain/templates/rag-conversation
```

### Option B: LlamaIndex RAG

```bash
pip install llama-index

# Clone examples
git clone https://github.com/run-llama/llama_index.git
cd llama_index/examples/chat_engine
```

### Option C: Build Your Own

Any RAG agent that exposes an HTTP API works. See `test-case.yaml` for expected format.

### Run the Agent

```bash
# Start your RAG agent server
python server.py  # or your entrypoint

# Default: http://localhost:8000
```

### Run EvalView Test

```bash
evalview run --pattern examples/rag-support/test-case.yaml
```

## Links

- **LangChain RAG**: https://github.com/langchain-ai/langchain/tree/master/templates/rag-conversation
- **LlamaIndex**: https://github.com/run-llama/llama_index
- **RAG Tutorial**: https://python.langchain.com/docs/tutorials/rag/
