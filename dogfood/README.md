# EvalView Dogfood

Test EvalView with EvalView itself.

## How it works

1. `agent.py` wraps EvalView's chat mode as an HTTP agent on localhost.
2. Test cases exercise explanations and explicit CLI execution requests.
3. Approved CLI commands run without a shell in a temporary directory. The
   assistant receives the actual tool results before writing its final answer.
4. EvalView checks that evidence, including the original tool and trust checks.

Commands in illustrative answers never execute unless the test explicitly grants
that exact command through `input.context.allowed_commands`. The harness supports
only `evalview adapters`, `evalview demo`, and `evalview snapshot --help`; it cannot
run arbitrary shell commands, recursive live evals, or overwrite project baselines.
Provider errors and failed commands return HTTP errors instead of plausible-looking
answers that could be mistaken for agent regressions.

## Run it

```bash
# From the repository root, install dependencies
uv sync --all-extras

# Terminal 1: Start the dogfood agent
uv run python dogfood/agent.py

# Terminal 2: Run the tests
uv run evalview run dogfood/
```

## Test Cases

| Test | What it checks |
|------|----------------|
| 01-list-adapters | Does chat run the adapter listing and explain its results? |
| 02-explain-test-case | Can chat explain test case format? |
| 03-run-command | Can chat run the local demo and explain its results? |
| 04-golden-baseline | Can chat use snapshot help to explain golden baselines? |

## Requirements

- An LLM provider (Ollama, OpenAI, or Anthropic API key)
- OpenAI API key for LLM-as-judge evaluation

## Daily workflow and failure handling

The daily workflow separates deterministic tests from live tests using the
`requires_api_key` pytest marker. It checks provider access with small, bounded
requests before running the live suite. Missing credentials, exhausted credits,
rate limits, and provider outages are reported as infrastructure failures. Live
tests blocked by preflight are unavailable, not passing.

Every required check must pass for the workflow to be green or for the rolling
dogfood issue to close. A final failure gate preserves unsuccessful outcomes even
when individual checks continue so later diagnostics can run. Complete logs,
JSON results, HTML reports, goldens, and monitor evidence are retained in the
`dogfood-evidence-<run-id>` artifact for 30 days.

If `provider-output.txt` reports `provider_quota`, a maintainer must restore API
credits or usable project quota for the `OPENAI_API_KEY` GitHub Actions secret.
For `provider_auth` or `provider_not_configured`, correct that secret. Then rerun
**Daily Dogfood** from GitHub Actions. Do not close the issue, lower assertions,
or refresh baselines to make a billing or availability failure appear green.
