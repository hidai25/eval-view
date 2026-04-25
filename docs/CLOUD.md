# EvalView Cloud — what it is, what it stores, what it never runs

> Optional dashboard for teams. The CLI stays fully local; cloud is
> additive — runs flow in via HTTP, analytics flow back, nothing is
> ever required.

EvalView is a local-first CLI. `evalview check`, `evalview simulate`,
`evalview monitor`, `evalview autopr` — all run on your machine, write
to `.evalview/`, and never touch the network unless you tell them to.

EvalView Cloud is the optional team layer on top of that:

- **A persistent home for runs.** Trends, ship decisions, drift
  charts, and PR comment archives across machines and CI jobs.
- **Cross-run analytics for rationale capture.** Decision drift,
  novel-decision-type alerts, branch causal graphs.
- **Slack/email digests.** Suppress noise via the same n=2
  confirmation gate the CLI already uses locally.
- **Public share links** for simulations, so you can show a teammate
  a what-if outcome without granting account access.

## What cloud explicitly does not do

This is the load-bearing invariant: **cloud holds no API keys, runs
no agents, and runs no simulations.** Everything that costs money,
makes a model call, or executes a tool happens locally. Cloud only
persists and aggregates the structured payloads the CLI emits.

Concretely:

- Cloud never runs `evalview check`, `evalview simulate`, or any
  adapter. It receives the resulting `EvaluationResult` payload over
  `POST /api/v1/results` and stores it.
- Cloud never replays traces or scores responses. The judge model,
  if any, is configured locally and runs locally.
- Cloud never pulls from your repo or CI. The CLI pushes; cloud
  receives. There is no inverse channel.

This boundary is enforced both in code (the cloud route deliberately
skips the drift/quarantine fan-out for `run_type="simulation"`
because mock-driven outcomes shouldn't poison the trend line) and in
the schema — the simulation payload is an opaque JSON the cloud only
walks for rendering, never for re-execution.

## Connecting the CLI

```bash
evalview login   # opens a browser to evalview.com, drops a token in ~/.evalview/config.toml
evalview check   # subsequent runs auto-push to cloud after local eval finishes
```

The push is best-effort and synchronous-but-bounded: up to 3 retries
with exponential backoff over ~15 seconds. Auth/billing errors fail
immediately without retry. A failed push never blocks the CLI exit
code — local eval is the source of truth.

To opt out at any time: `evalview logout`, or unset
`EVALVIEW_API_TOKEN`. The CLI is fully functional without cloud.

## What's on the wire

`evalview/cloud/push.py::push_result()` is the single push site.
The payload is documented inline; the bounded fields are:

| Field | Owner | Cap |
|---|---|---|
| `run_type` | OSS — `"standard"` or `"simulation"` | enum |
| `summary` + `diffs` | OSS gate result, mirrored verbatim | per-test diff blob ≤ ~1 MB |
| `verdict` | OSS — see `evalview/core/verdict.py` | 4-tier enum |
| `behavioral_anomalies` / `trust_scores` / `coherence_analysis` | OSS observability per-test arrays | aggregated to top-N per push |
| `simulation` | OSS engine output — opaque to cloud | run-level |
| `rationale_events` | OSS adapter capture hook | 500/run, 4 KB/event (caps in `evalview/core/rationale.py`) |

## When OSS and cloud need to move together

A small set of constants and enums must stay in lockstep across the
two repos. When you change them on the OSS side, coordinate a cloud
deploy in the same PR cycle:

| Surface | OSS source of truth | Cloud mirror |
|---|---|---|
| 4-tier verdict enum | `evalview/core/verdict.py::Verdict` | `src/lib/verdict.ts::Verdict` |
| Verdict headline strings | `evalview/core/verdict.py::_HEADLINE` | `src/lib/verdict.ts::HEADLINES` |
| Rationale caps | `RATIONALE_MAX_EVENTS_PER_RUN`, `RATIONALE_MAX_TEXT_BYTES` | `src/lib/result-schemas.ts` |
| `decision_type` enum | `evalview/core/types.py::DecisionType` | Zod enum + DB CHECK constraint |
| `decision_type` descriptions | `evalview/core/rationale.py::DECISION_TYPE_DESCRIPTIONS` | `src/lib/rationale-decision-types.ts` |
| Observability schema version | `evalview/core/observability.py::OBSERVABILITY_SCHEMA_VERSION` | accepted on the wire, no version-gating yet |

The cloud schema is intentionally tolerant where it can be (loose
`mocks_applied` arrays, aliasing `run_type="check"` → `"standard"`)
so an OSS upgrade ahead of a cloud deploy never bricks ingest.

## Privacy posture

- Trace contents (prompts, completions, tool args) live in
  `result_json` and `diffs[*].diff_json`. Both are stored verbatim.
  If you're testing on customer data, scrub before push or skip
  cloud entirely — `evalview check` works fully without it.
- Cloud never stores raw API keys. Tokens minted by `evalview
  login` are scoped per-org and revocable from the dashboard.
- Public simulation share links (`/share/simulations/<slug>`) are
  the one anonymous-readable surface. Slugs are 128-bit random;
  admins create and revoke them.

## Related

- `docs/RATIONALE.md` — what gets sent under `rationale_events`
- `docs/SIMULATE.md` — what gets sent under `simulation`
- `docs/OPERATING_MODEL.md` — how teams run the loop end-to-end
- `evalview/cloud/push.py` — the canonical push site
