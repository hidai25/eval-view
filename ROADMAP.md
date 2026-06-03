# EvalView Roadmap

> Where EvalView is going, and where you can help.

EvalView's mission is simple: **treat agent prompt and model changes like schema migrations** — versioned, diffable, and gated by a deterministic CI check. Everything on this roadmap serves that mission.

This doc is intentionally short. We'd rather ship a small batch every month than maintain a 50-item wishlist that rots.

---

## Where we are (June 2026)

- 14+ adapters (HTTP, Anthropic, OpenAI, LangGraph, CrewAI, Pydantic AI, Aider, Goose, MCP, …)
- Snapshot → diff → check loop with multi-variant goldens for non-determinism
- Tool-call, sequence, output (LLM-as-judge), cost, latency, safety, hallucination, and PII evaluators
- `evalview monitor` with Slack **and Discord** alerts and JSONL history for production
- `evalview watch` / `run --watch` — re-run on file change for a tight inner loop
- Record/replay cassettes for hermetic CI
- GitHub Action (`action.yml`) for drop-in CI, **with PR-comment diffs** (`evalview ci comment`)

## The pains driving the next batch

These are the things real teams are writing about in May 2026 that EvalView is built to absorb:

1. **"One prompt edit silently broke three unrelated paths."** Writeups this month describe one-line prompt changes causing 14-point refusal-rate jumps and downstream field-extraction regressions days later. Versioned, paired evaluation on the same examples is the only thing that catches this.
2. **"Lab benchmarks lie."** A widely-cited 2026 figure puts the gap between bench scores and production at ~37%. Trace-level regression detection on *your* traffic beats any leaderboard.
3. **"Non-determinism makes snapshots flaky."** GitHub's own blog asks "how do you validate behavior when 'correct' isn't deterministic?" — our multi-variant golden + best-match severity ranking is the answer; it just needs to be easier to reach for.
4. **"How do I gate tool-call sequences in CI?"** Teams want a fast, framework-agnostic CLI, not a SaaS platform with a setup cost.
5. **"CrewAI/LangGraph debugging is painful."** Print statements don't escape task callbacks. People want to *see what the agent did and diff it* — not stare at logs.

## Next batch (Q2/Q3 2026)

### Coverage
- Vercel AI SDK adapter — currently the most-requested missing framework
- Pydantic AI tool-call schema validator — catches wrong-argument regressions earlier

### Developer loop
- Reusable assertion library for common agent invariants (tool ordering, max-retry, no-PII-in-output, …)
- Web UI for trace browsing — today it's CLI + JSON/HTML reports

### Story / docs
- More framework quick-starts (Vercel AI SDK, Pydantic AI) once the adapters land

Each of these is open as a GitHub issue with `help wanted` and clear acceptance criteria. Pick one and go: [help wanted issues →](https://github.com/hidai25/eval-view/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)

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
