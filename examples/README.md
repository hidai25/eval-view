# EvalView Examples

Working examples for the most popular AI agent frameworks.

## Quick Start

```bash
# 1. Pick a framework below and follow its README
# 2. Start your agent
# 3. Run EvalView against it
evalview run --pattern examples/<framework>/test-case.yaml
```

## Examples

| Framework | Folder | What it tests |
|-----------|--------|---------------|
| 🦜 **LangGraph** | [langgraph/](langgraph/) | Multi-step research agent with tool calls |
| 🚢 **CrewAI** | [crewai/](crewai/) | Multi-agent team collaboration |
| 🤖 **AutoGen** | [autogen/](autogen/) | Multi-agent conversation patterns |
| 🎨 **Dify** | [dify/](dify/) | Visual workflow builder |
| 💬 **OpenAI Assistants** | [openai-assistants/](openai-assistants/) | Native OpenAI Assistants API |
| 🤖 **Anthropic** | [anthropic/](anthropic/) | Claude direct API + Claude Agent SDK |

## New to EvalView?

Start here — a complete working agent you can run in 2 minutes:

```bash
# Clone and run the demo agent
curl -O https://raw.githubusercontent.com/hidai25/eval-view/main/demo-agent/agent.py
pip install fastapi uvicorn
python agent.py

# Point EvalView at it
evalview run
```

Or see EvalView catch a real regression without any setup:

```bash
evalview demo
```

## Implementing Your Own Agent

See [`backend-implementations.md`](backend-implementations.md) for copy-paste examples in FastAPI, Flask, Express.js, and streaming JSONL — with the exact request/response format EvalView expects.

## Questions?

- [GitHub Discussions](https://github.com/hidai25/eval-view/discussions)
- [Backend requirements spec](../docs/BACKEND_REQUIREMENTS.md)
