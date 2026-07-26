# EvalView Roadmap

> Where EvalView is going, and where you can help.

EvalView's mission is simple: **treat agent prompt and model changes like schema migrations** — versioned, diffable, and gated by a deterministic CI check. Everything on this roadmap serves that mission.

This doc is intentionally short. We'd rather ship a small batch every month than maintain a 50-item wishlist that rots.

---

## Where we are (July 2026 — v0.8.1)

- 15+ adapters (HTTP, Anthropic, OpenAI, LangGraph, CrewAI, Pydantic AI, **Vercel AI SDK**, Aider, Goose, MCP, …)
- Snapshot → diff → check loop with multi-variant goldens for non-determinism
- Tool-call, sequence, output (LLM-as-judge), cost, latency, safety, hallucination, and PII evaluators
- `evalview capture` — recording proxy that turns real traffic into test cases
- `evalview monitor` with Slack **and Discord** alerts and JSONL history for production
- `evalview watch` / `run --watch` — re-run on file change for a tight inner loop
- Record/replay cassettes for hermetic CI
- GitHub Action (`action.yml`) for drop-in CI, **with PR-comment diffs** (`evalview ci comment`)
- Fully typed: PEP 561 `py.typed` marker, mypy-clean across adapters, evaluators, and reporters

A lot of the list above came from outside contributors — 25+ merged community PRs so far. The Vercel AI SDK adapter ([@gauravxthakur](https://github.com/gauravxthakur), #250), the Mistral and Cohere adapters and the deterministic PII evaluator ([@XJ789](https://github.com/XJ789), #68/#69/#70), Discord alerting ([@zeel2104](https://github.com/zeel2104), #147), `--schedule` cron syntax, TOML test cases, and the `diff` and `validate` commands ([@mvanhorn](https://github.com/mvanhorn), #224/#209/#198/#195), JSONL monitor history ([@clawtom](https://github.com/clawtom), #85) — among others. Thank you. Adapters and CLI flags are the easiest places to start.

## The pains driving the next batch

Refreshed July 2026. Every item links to a real thread — that's the bar for getting on this list.

1. **"Evals green, tools silently stopped being called."** A prompt tone tweak made a support agent stop calling `lookup_order` and start answering order-status questions from memory. Exact-match, semantic similarity, and an LLM judge all passed, because the fabricated answers were fluent and on-tone. *"The bug wasn't in the words. It was in the behavior."* → [r/LLMDevs](https://old.reddit.com/r/LLMDevs/comments/1u5yt47/my_agent_passed_every_eval_then_quietly_stopped/). This is the thing EvalView exists for; the deterministic tool + sequence diff catches it with no judge at all.
2. **"Eval latency added 18 minutes to our CI."** LangGraph agent, p99 build 6min → 24min, judge calls dominating (~200 scenarios × 2 samples). *"Engineers are batching changes to avoid the gate. Defeats CD entirely."* → [r/AI_Agents](https://old.reddit.com/r/AI_Agents/comments/1uhot34/agent_eval_latency_added_18_minutes_to_our_ci_how/). Our tool/sequence diff is deterministic, offline, and costs $0 — this needs to be the headline, not a footnote.
3. **"A spreadsheet of prompts is not a regression suite."** The asker's spec — versioned cases, runs on every change, fails the build, tracks pass-rate over time, grows on new failure modes — is EvalView's feature list verbatim, and the thread's top reply says building it is *"multiple sprints minimum."* → [r/AIQuality](https://old.reddit.com/r/AIQuality/comments/1ukkc7w/anyone_maintaining_a_real_agent_regression_suite/). That belief is the thing to falsify.
4. **"Our eval set drifted away from production."** *"Eval sets have to evolve with production or they slowly become benchmarks for a product you shipped six months ago."* → [r/AIQuality](https://old.reddit.com/r/AIQuality/comments/1ujot04/our_evals_were_green_for_a_month_straight_while/). `evalview capture` answers this directly and is under-documented.
5. **"Multi-turn trajectories, not single answers."** From the same thread: *"Our worst failures aren't one bad answer. They're five or six reasonable answers that collectively take the conversation somewhere dumb."* Most frameworks are still request/response shaped. We do multi-turn, but it's filed under power-user features.
6. **"Agents are stateful, and nobody tests that."** Teams *"test agents like they are stateless functions, when in reality they are long-running stateful processes"* — step 3 of 4 fails, records orphan or duplicate, and almost nobody tests idempotency or mid-task restart. → [Ask HN](https://news.ycombinator.com/item?id=47325105). This one we genuinely don't cover yet.
7. **"Non-determinism makes snapshots flaky."** Multi-variant goldens + best-match severity ranking are the answer; they just need to be easier to reach for.

## Next batch (Q3 2026)

The exact contents of 0.9.0 aren't locked yet. These are the candidates, mapped to the pains above.

### Story / docs — the biggest gap, and it isn't code
- **Publish the cost of the gate** (pain 2). Measured wall-clock and dollar cost for a 200-scenario `check` run with no judge, next to what a judge-based suite costs. The number exists; nobody has been shown it.
- **Promote `capture` into the Quick Start** (pain 4). It's the answer to eval-set rot and the README currently doesn't mention it at all.

### Developer loop
- Conversation-trajectory diffing as a first-class primitive (pain 5)
- Reusable assertion library for common agent invariants (tool ordering, max-retry, no-PII-in-output, …)
- `evalview check --watch` ([#242](https://github.com/hidai25/eval-view/issues/242)) — good first issue, the watcher already exists
- Web UI for trace browsing — today it's CLI + JSON/HTML reports

### Coverage
- Pydantic AI tool-call schema validator ([#240](https://github.com/hidai25/eval-view/issues/240)) — catches wrong-argument regressions earlier
- Stateful / cascade-failure testing (pain 6): mid-run failure injection, resume, idempotency assertions. Nothing in this space covers it — the most interesting unclaimed problem on this list, and a great one to pick up if you want a meaty first contribution.

The linked ones are open issues with acceptance criteria; the rest are not filed yet. If you want to take one, open an issue saying so and I'll write up the criteria with you rather than making you guess. [Help wanted issues →](https://github.com/hidai25/eval-view/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)

## Slightly further out

- More language SDKs for emitting traces (the CLI is Python; trace ingestion shouldn't be)
- Auto-generated test suites from production failure clusters (pull failing prod traces → draft regression cases)

These are *direction*, not commitments — if you want one of them to happen sooner, open an issue describing the use case.

## How decisions get made

- **Pain signal beats feature request.** A linked Reddit / HN / blog post describing the problem moves a feature up the queue faster than a "would be nice" comment.
- **Small > clever.** A 100-line adapter that ships beats a 1000-line abstraction that doesn't.
- **Tests included.** Anything that changes diff behavior needs at least one regression test against the existing goldens.
- **Backwards-compatible by default.** The `check` exit code contract is the most load-bearing surface in the project — changes there get extra scrutiny.

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, coding standards, and PR flow. Three things worth knowing up front:

1. We're friendly. First-time contributors get review, not gatekeeping.
2. We have a pinned discussion: [What's the most painful thing about testing your agent right now?](https://github.com/hidai25/eval-view/discussions) — your answer shapes this roadmap.
3. `make ci` runs the same checks CI runs. If it's green locally, your PR is 90% of the way there.
