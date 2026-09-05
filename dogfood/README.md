# EvalView Dogfood

Test EvalView with EvalView itself, with separate core and live-provider evidence.

## Core checks: automatic, no paid inference

[Core Dogfood](../.github/workflows/dogfood.yml) runs daily at 09:00 UTC, on pull
requests, on main pushes, and on manual dispatch. It runs all non-live tests,
type checks, the local demo, mock-agent end-to-end tests, `snapshot` / `check`,
and a monitor smoke test. No paid API credentials are provided. “Provider-free”
refers to inference; GitHub runner usage may still cost money.

Run the core test suite locally from the repository root:

```bash
uv sync --all-extras
uv run pytest -m 'not requires_api_key'
```

Mock tests verify EvalView's behavior under controlled conditions. They cannot
verify live provider availability, quality, or drift. Core success is not a
replacement for live evidence.

## Live chat harness

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

## Run live checks deliberately

[Live Provider Checks](../.github/workflows/dogfood-live.yml) runs **only by manual
dispatch on main**, with `confirm_paid_api=true`. There is no automatic live
schedule. A maintainer must supply the repository's `OPENAI_API_KEY` secret from
an authorized, funded project and explicitly accept paid API use.

Once this workflow is merged, dispatch it in GitHub Actions or run:

```bash
gh workflow run dogfood-live.yml --repo hidai25/eval-view --ref main -f confirm_paid_api=true
```

The confirmation defaults to false; other branches are rejected. The live run
preflights `gpt-5.4-mini` and `gpt-4o`, then tests the live evaluators and chat
harness with the existing assertions and trust checks. Its status badge describes
the last manual run, not an automatic health monitor.

Serial execution, bounded retries, and concurrency control limit work, not dollars.
The paid stages have 1-, 5-, and 8-minute timeouts; the 25-minute job limit reserves
time for setup and evidence collection. Use a dedicated API project with provider-enforced spend
controls for a financial limit. `--no-judge` does not remove backend API fees, and
semantic-diff embeddings remain opt-in; the core workflow uses neither paid
backend calls nor embeddings.

## Test Cases

| Test | What it checks |
|------|----------------|
| 01-list-adapters | Does chat run the adapter listing and explain its results? |
| 02-explain-test-case | Can chat explain test case format? |
| 03-run-command | Can chat run the local demo and explain its results? |
| 04-golden-baseline | Can chat use snapshot help to explain golden baselines? |

## Failure handling

The `requires_api_key` pytest marker separates live tests from the automatic core
suite. In a manually authorized live run, provider preflight checks access before
the suite. Missing credentials, exhausted credits, rate limits, and outages are
reported as service failures. Live tests blocked by preflight are unavailable,
not passing.

Every required check must pass for its workflow to be green or for its scoped
rolling issue to close. Core uses `dogfood-core`; live uses `dogfood-live`.
Neither can close the other's issue. A final failure gate preserves unsuccessful
outcomes even when individual checks continue to collect diagnostics. Each scope
retains its logs and available reports as artifacts for 30 days; core also retains
its mock goldens and monitor evidence.

If `provider-output.txt` reports `provider_quota`, a maintainer must restore API
credits or usable project quota for the `OPENAI_API_KEY` GitHub Actions secret.
For `provider_auth` or `provider_not_configured`, correct that secret. Then rerun
**Live Provider Checks** from main with paid-API confirmation. Do not close the
issue, lower assertions, or refresh baselines to make a billing or availability
failure appear green.

The historical [incident #264](https://github.com/hidai25/eval-view/issues/264)
is deliberately preserved. Neither scoped workflow automatically closes it; a
maintainer must review fresh complete core and live evidence plus the historical
failure details. This separation does not claim provider recovery. See the
[triage guide](../docs/INTERNAL_DOGFOODING.md#daily-failure-triage).
