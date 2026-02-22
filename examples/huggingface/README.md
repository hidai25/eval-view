# HuggingFace Examples — Testing HuggingFace Spaces Agents with EvalView

> Test HuggingFace Spaces agents (Gradio-based) with EvalView — 100% open source, zero OpenAI costs. Use Llama as a free LLM-as-judge.

## Quick Start

```bash
# 1. Set your HuggingFace token
export HF_TOKEN="hf_your_token_here"

# 2. Use Llama for evaluation (skip OpenAI)
export EVAL_PROVIDER=huggingface

# 3. Run tests against a public Space
cd examples/huggingface
evalview run --verbose
```

## What's Included

```
huggingface/
├── config.yaml           # Example configuration
├── test-cases/
│   └── chatbot-test.yaml # Sample test case
└── README.md             # This file
```

## Configuration

The `config.yaml` connects to a public HuggingFace Space:

```yaml
adapter: huggingface
endpoint: HuggingFaceH4/zephyr-chat  # Public chatbot Space
```

## Test Your Own Space

1. Update `config.yaml` with your Space:
   ```yaml
   endpoint: your-username/your-space
   ```

2. Modify test cases in `test-cases/` for your agent

3. Run:
   ```bash
   evalview run
   ```

## Using Llama as Judge

No OpenAI API key needed! Use Llama models for evaluation:

```bash
export HF_TOKEN="hf_xxx"
export EVAL_PROVIDER=huggingface
export EVAL_MODEL=meta-llama/Llama-3.1-70B-Instruct  # Optional

evalview run
```

## Notes

- Free-tier Spaces sleep after inactivity (30-60s cold start)
- Set generous `max_latency` thresholds for cold starts
- HF Inference API has rate limits on free tier

## More Info

- [Full HuggingFace Guide](../../docs/QUICKSTART_HUGGINGFACE.md)
- [Adapter Documentation](../../docs/ADAPTERS.md)
