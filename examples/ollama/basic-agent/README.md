# Basic Ollama Agent — Offline Testing with EvalView

Test a fully local AI agent using Ollama and EvalView.
**Zero cloud API keys required.**

---

## Requirements

- [Ollama](https://ollama.com) installed and running
- Python 3.9+
- EvalView installed (`pip install evalview`)

---

## Setup

**1. Pull the model:**
```bash
ollama pull llama3.2:1b
```

**2. Start the agent server:**
```bash
python agent.py serve
```
Leave this terminal open.

**3. In a new terminal, run the eval:**
```bash
cd /path/to/eval-view
python -m evalview run examples/ollama/basic-agent/basic-eval.yaml --no-judge
```

---

## What this does

- `agent.py` runs a local HTTP server on `localhost:8123` that forwards prompts to Ollama
- `basic-eval.yaml` defines the test case and expected output
- `--no-judge` skips LLM-as-judge scoring, uses deterministic scoring only (free, offline)

---

## Expected Output
```
✅ AGENT HEALTHY
✓ Passed: 1    ✗ Failed: 0
Score: 87.5/100   Cost: $0.0000
```

---

## Save a baseline for regression detection
```bash
python -m evalview snapshot
python -m evalview check   # run this on future changes
```

---

## Model

Uses `llama3.2:1b` by default. To change it, edit the `MODEL` variable in `agent.py`.
