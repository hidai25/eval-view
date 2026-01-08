# EvalView Dogfood

Test EvalView with EvalView itself.

## How it works

1. `agent.py` wraps EvalView's chat mode as an HTTP agent
2. Test cases exercise the chat mode's knowledge and capabilities
3. EvalView runs tests against its own chat mode

## Run it

```bash
# Terminal 1: Start the dogfood agent
cd dogfood
python agent.py

# Terminal 2: Run the tests
evalview run dogfood/
```

## Test Cases

| Test | What it checks |
|------|----------------|
| 01-list-adapters | Does chat know EvalView's adapters? |
| 02-explain-test-case | Can chat explain test case format? |
| 03-run-command | Can chat suggest valid commands? |
| 04-golden-baseline | Does chat understand regression testing? |

## Requirements

- An LLM provider (Ollama, OpenAI, or Anthropic API key)
- OpenAI API key for LLM-as-judge evaluation
