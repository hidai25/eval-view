# Internal Dogfooding

This is the smallest operational version of "EvalView evaluates EvalView."

The point is not to add process theater.
The point is to make the internal build loop repeatable:

1. write a short spec
2. implement with agent help
3. review hard
4. run the relevant dogfood slice
5. snapshot intentional changes when behavior changed
6. run `check`
7. ship

## Daily failure triage

Core and live-provider checks produce separate evidence and statuses:

| Workflow | When it runs | What it establishes |
| --- | --- | --- |
| [Core Dogfood](../.github/workflows/dogfood.yml) | Daily at 09:00 UTC, pull requests, main pushes, or manually | Non-live tests, type checks, mock-agent snapshot/check, demo, and monitor behavior without paid inference |
| [Live Provider Checks](../.github/workflows/dogfood-live.yml) | Manually, on main, with explicit paid-API confirmation | Provider access, live evaluator behavior, and chat-agent behavior with the configured models |

Core runs use no paid API credentials. “Provider-free” means zero paid inference
calls, not zero GitHub runner cost. Mock tests can catch library regressions but
cannot establish provider availability, model quality, or provider drift. The live
badge describes its last manual run, not a continuously monitored service.

A missing key, exhausted quota, timeout, or provider error makes the live run
incomplete. Keep it visibly failing and report the service problem. It must not
be converted to an agent answer, a passing skipped test, or a new baseline.

Each workflow maintains only its own rolling issue: `dogfood-core` or
`dogfood-live`. An issue may close only when every required check in that scope
succeeds. A successful core run cannot close a live failure, and a successful
live run cannot close a core failure. Setup errors and incomplete evidence fail
their scope even when subsequent diagnostics continue.

1. Open the workflow summary and download its logs/reports artifact. Record the
   failing step, source revision, model, and provider error category.
2. For a live preflight failure, inspect the funded API project behind the
   `OPENAI_API_KEY` Actions secret. Quota, credential, and provider failures are
   service problems; never include keys in issues.
3. After restoring access, explicitly opt into **Live Provider Checks** on the
   reviewed main revision. Provider recovery does not resolve remaining behavior
   failures automatically. Core checks can continue independently.
4. For an actual behavior failure, reproduce it with a focused test and inspect
   the raw tool outputs and trust findings before changing code or expectations.
5. Review the complete evidence for the affected scope. A failed setup,
   interrupted run, or skipped required stage is not recovery.

[Issue #264](https://github.com/hidai25/eval-view/issues/264) records the history.
Its September 4, 2026 run reports `credit_balance_exhausted` from OpenAI. Earlier
entries also contain execution timeouts and trust warnings; fixing quota does not
establish that those older issues are resolved. Before this follow-up, the workflow
could report overall success despite its recorded failures, so historical green
badges alone are not evidence of a healthy run.

The split deliberately preserves this historical incident: neither workflow
automatically closes #264. A maintainer must review fresh successful core **and**
live evidence, along with the older timeouts and trust findings, before resolving
it. Splitting the workflows is not a claim of funded provider recovery.

The chat harness now distinguishes instructions to execute from explanatory code
examples. It runs only explicitly approved commands, in an isolated directory,
and asks for a final answer after recording the actual tool results. Trust checks
remain enabled. Their warnings warrant investigation; they are not a causal proof
of gaming, provider drift, or a library defect.

## Manually verify the live provider

After the workflow changes are merged, a maintainer can choose **Live Provider
Checks** in GitHub Actions, select `main`, and enable `confirm_paid_api`. The
confirmation defaults to false and the workflow rejects other branches.
The equivalent CLI command is:

```bash
gh workflow run dogfood-live.yml --repo hidai25/eval-view --ref main -f confirm_paid_api=true
```

This authorizes real API calls using the funded `OPENAI_API_KEY` repository
secret. The live workflow keeps `gpt-5.4-mini` and `gpt-4o`; it does not substitute
mock results. There is no automatic weekly or other live schedule.

The workflow serializes test execution, bounds retries, and limits concurrent runs.
Preflight, live evaluator tests, and chat tests have 1-, 5-, and 8-minute limits;
the 25-minute job timeout also leaves room for setup and saving evidence.
These controls do not guarantee a dollar cap.
Use a dedicated API project and provider-enforced spend controls for a financial
limit; budget notifications alone are not an enforced cap.

`--no-judge` disables judge calls only. Running a cloud-backed agent still uses
its API, and opting into semantic comparison may use embedding APIs. Core
dogfood uses local mocks and leaves semantic-diff embeddings disabled.

## Canonical Internal Slices

These are the default internal slices for EvalView development.

### `mcp`

Use when changing:

- `evalview/mcp_server.py`
- MCP contracts
- MCP tool schemas
- MCP command routing

Run:

```bash
make dogfood-mcp
```

### `healing`

Use when changing:

- `evalview/core/healing.py`
- healing policy
- healing audit/reporting behavior
- model-update recovery behavior

Run:

```bash
make dogfood-healing
```

### `snapshot`

Use when changing:

- `snapshot` command behavior
- golden storage behavior
- baseline creation / reset / variants

Run:

```bash
make dogfood-snapshot-core
```

### `check`

Use when changing:

- `check` command behavior
- diffing
- root-cause summaries
- tag filtering
- fail-on / strict semantics

Run:

```bash
make dogfood-check-core
```

### `reporting`

Use when changing:

- HTML report rendering
- CLI regression presentation
- PR/CI comment reporting
- diff explanation rendering

Run:

```bash
make dogfood-reporting
```

### `agent_docs`

Use when changing:

- `README.md`
- `AGENTS.md`
- `docs/agent-recipes/`
- agent-native guidance

This slice is partly manual today.

Run:

```bash
make dogfood-agent-docs
```

Then manually review:

- `README.md`
- `AGENTS.md`
- `docs/agent-recipes/README.md`

## Feature-to-Eval Matrix

Use this as the default ship gate.

| Change type | Must run |
|-------------|----------|
| MCP feature or contract change | `make dogfood-mcp` and `make dogfood-check-core` |
| Healing policy or audit change | `make dogfood-healing`, `make dogfood-check-core`, `make dogfood-reporting` |
| Snapshot / golden behavior change | `make dogfood-snapshot-core` and `make dogfood-check-core` |
| Check / diff / root-cause change | `make dogfood-check-core` and `make dogfood-reporting` |
| HTML / CLI report change | `make dogfood-reporting` |
| Agent docs / recipes / README change | `make dogfood-agent-docs` |
| Cross-cutting core change | `make dogfood-core` |

## Default Ship Loop

For most changes:

1. write a short spec
2. implement
3. review
4. run the slice from the matrix above
5. if behavior intentionally changed, update snapshot/baseline
6. run a broader `check` slice if the change touched core behavior
7. ship

## Bug-to-Test Rule

Every meaningful dogfood failure should become at least one of:

- a new test
- a stronger test
- a new tag
- a documented invariant

If a failure keeps happening and never becomes an eval asset, the loop is broken.

## Current Limits

This is intentionally lightweight.

It does **not** yet provide:

- automatic slice selection
- automatic selection and enforcement of the feature-specific slices above
- automatic bug-to-test conversion

Those can come later if they prove useful.

For now, the goal is simple:

**make the internal EvalView build loop explicit and repeatable.**
