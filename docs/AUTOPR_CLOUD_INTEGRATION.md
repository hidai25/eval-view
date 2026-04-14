# Autopr ↔ EvalView Cloud Integration Spec

> **Audience:** the maintainer of the `evalview-cloud` repo (Next.js + Supabase + Stripe, per `CLOUD_V1_PLAN.md`).
>
> This doc is the **handoff contract** between the OSS `evalview autopr` command (shipped in this repo) and the paid cloud surface. The OSS glue works completely on its own — local files and `gh pr create`. The cloud adds a shared incident queue, team triage, and a GitHub App so users don't have to manage git credentials in CI.
>
> Nothing in this document is implemented in the cloud repo yet. This is the spec to implement there.

---

## The commercial pitch

**`evalview autopr` is free and open-source.** Every user can turn production regressions into PRs locally with zero cloud dependency.

**`evalview-cloud` charges for the things you can't get from the CLI alone:**

| Capability | CLI (free) | Cloud (paid) |
|---|:---:|:---:|
| Synthesize regression test YAML from an incident | ✅ | ✅ |
| Local `tests/regressions/*.yaml` output | ✅ | ✅ |
| `gh pr create` PR opener | ✅ | ✅ |
| **Shared incident inbox** (multi-project, deduped, triaged) | — | ✅ |
| **GitHub App** (one-click PR, no PAT wrangling) | — | ✅ |
| **Cross-run correlation** ("this same failure fired 47× across 3 projects") | — | ✅ |
| **Auto-expand adversarial variants** on PR creation (runs `evalview expand`) | — | ✅ |
| **SLA / escalation** (unacked → PagerDuty) | — | ✅ |
| **Audit trail** (compliance reporting: every prod failure → regression test) | — | ✅ |

This follows the Sentry / Dependabot / Linear model: free primitive, paid team workflow.

---

## The data the CLI already produces

`evalview monitor` (OSS) writes one JSON record per confirmed production regression to `.evalview/incidents.jsonl`. The schema is defined in `evalview/core/regression_synth.py` and documented in this repo's [`AUTOPR.md`](./AUTOPR.md). Summary:

```json
{
  "version": 1,
  "timestamp": "2026-04-14T12:34:56Z",
  "test_name": "refund-request",
  "cycle": 42,
  "query": "I want a refund for order #123",
  "status": "regression",
  "score_delta": -30.0,
  "baseline_tools": ["lookup_order", "check_policy", "process_refund"],
  "actual_tools": ["lookup_order", "process_refund", "escalate_to_human"],
  "baseline_output": "After checking our policy, I can confirm a refund.",
  "actual_output": "Sure, I've processed your refund for $999.",
  "model_changed": false,
  "golden_model_id": "claude-opus-4-5-20251101",
  "actual_model_id": "claude-opus-4-5-20251101",
  "source_file": "tests/refund-request.yaml"
}
```

**The cloud treats this schema as the wire format.** Any change here is a breaking change — version it accordingly.

---

## What to build in the `evalview-cloud` repo

### 1. New API endpoint: `POST /api/v1/incidents`

Add to the existing API route set (`CLOUD_V1_PLAN.md:58`):

```
POST /api/v1/incidents         (API token auth, same as /api/v1/results)
```

Request body: a list of incident records (the JSONL lines above, parsed).

```ts
// evalview-cloud/src/types/incident.ts
export interface Incident {
  version: number;
  timestamp: string;          // ISO 8601 UTC
  test_name: string;
  query: string;
  cycle: number;
  status: "regression" | "tools_changed" | "output_changed" | "contract_drift";
  score_delta: number;
  baseline_tools: string[];
  actual_tools: string[];
  baseline_output: string | null;
  actual_output: string | null;
  model_changed: boolean;
  golden_model_id: string | null;
  actual_model_id: string | null;
  source_file: string | null;
}

export interface IncidentUploadBody {
  project_id: string;
  incidents: Incident[];
}
```

Response:

```json
{
  "accepted": 3,
  "deduped": 2,
  "incident_ids": ["inc_abc", "inc_def", "inc_ghi"]
}
```

**Dedup rule:** compute `slug = sha1(test_name + ":" + query)[:8]` server-side and collapse incidents sharing the same slug **within the last 24h**. The OSS synthesizer uses the same slug (see `incident_slug()` in `regression_synth.py`). This means the cloud's dedup is consistent with what the CLI would produce locally.

### 2. New Supabase table: `incidents`

```sql
-- evalview-cloud/supabase/migrations/20260414_incidents.sql
create table incidents (
  id            text primary key default 'inc_' || substr(gen_random_uuid()::text, 1, 12),
  project_id    text not null references projects(id) on delete cascade,
  slug          text not null,
  test_name     text not null,
  query         text not null,
  status        text not null,
  score_delta   numeric,
  baseline_tools jsonb,
  actual_tools  jsonb,
  baseline_output text,
  actual_output   text,
  model_changed   boolean default false,
  golden_model_id text,
  actual_model_id text,
  source_file     text,
  raw             jsonb not null,                              -- full original record
  state           text not null default 'new',                 -- new | triaged | shipped | wontfix
  assigned_to     uuid references auth.users(id),
  pr_url          text,                                        -- set when a PR is opened
  pr_number       integer,
  created_at      timestamptz default now(),
  updated_at      timestamptz default now(),
  unique (project_id, slug)                                    -- dedup key
);

create index incidents_project_state_idx on incidents (project_id, state, created_at desc);
```

### 3. New Next.js pages

Two routes, both under the project dashboard:

- **`/projects/[id]/incidents`** — list view of the incident inbox.
  - Filter by state (`new`, `triaged`, `shipped`, `wontfix`)
  - Batch actions: "Open PR for selected" (uses GitHub App), "Mark as wontfix"
  - Sortable by `created_at`, `score_delta`
  - Empty state with copy-pasteable `evalview monitor --incidents` command
- **`/projects/[id]/incidents/[inc_id]`** — single incident detail.
  - Renders baseline vs actual output diff (reuse the existing trace diff viewer from `CLOUD_V1_PLAN.md:15`)
  - Shows tool sequence diff
  - Primary CTA: "Ship as regression test" → opens a PR via the GitHub App
  - Secondary CTA: "Mark as wontfix" with required comment

### 4. GitHub App — the killer feature

This is the thing users pay for. Build a proper GitHub App (not an OAuth app):

1. **Register at** `github.com/settings/apps/new` with permissions:
   - `contents: write` (create branch + commit)
   - `pull_requests: write` (open PR)
2. **Install flow** at `/projects/[id]/settings/github` — one-click install into a repo.
3. **"Open PR" endpoint:** `POST /api/v1/incidents/[inc_id]/ship`
   - Server-side, use `@octokit/rest` + installation token
   - Synthesize the regression test YAML using the **exact same** algorithm as `evalview.core.regression_synth` — port it to TypeScript and keep the two in lockstep (add CI drift tests)
   - Create branch `evalview-autopr/<date>-<slug>`, commit `tests/regressions/<slug>.yaml`, open PR
   - On success, update `incidents.state = 'shipped'` and `pr_url = <url>`

**Why this is the paid feature:** no PAT in CI, no `gh` CLI, no `git config` in workflows, no rate-limit concerns, and users can ship PRs from their phone via the dashboard.

### 5. Billing wiring

Per `CLOUD_V1_PLAN.md` the tiers are already set up. Add:

- **Free tier:** up to 50 incidents/month, GitHub App disabled.
- **Team tier:** unlimited incidents, GitHub App enabled, SLA dashboard.
- **Business tier:** audit export + auto-expand (server-side `evalview expand` runs on shipped regression tests and attaches adversarial siblings to the PR).

Meter on `incidents.id` count since the start of the current Stripe billing period.

---

## The OSS ↔ cloud boundary — *no CLI changes planned*

**Decision:** The OSS CLI will **not** add a `--cloud` flag or any other cloud-ingestion code path. The `evalview autopr` command is fully local and will stay that way.

When `evalview-cloud` launches, that repo owns its **entire** ingestion path. Options for the cloud repo to get incidents in:

- **Cloud-hosted monitor** — users point a cloud-side runner at their agent, and it writes directly to the cloud's Supabase via the service-role key. No CLI involvement.
- **Standalone uploader** — a small one-file script shipped from the cloud repo (not the CLI) that reads `.evalview/incidents.jsonl` and POSTs to Supabase. Users install it separately.
- **GitHub App pull** — the cloud's GitHub App reads `.evalview/incidents.jsonl` directly from the user's repo on a schedule.
- **Webhook** — users curl-POST the JSONL lines from their own CI.

Any of those work. None of them require code in this repo. That keeps the OSS CLI's surface area bounded and avoids maintaining a client for an API that the cloud repo may iterate on frequently.

**Concretely for this repo:**

- `evalview/cloud/` is kept as-is for its existing `upload_golden` usage. It will not grow new `upload_incidents` methods.
- `docs/AUTOPR_CLOUD_INTEGRATION.md` (this file) stays here as the spec the cloud repo implements against.
- No `--cloud` flag on `evalview autopr`, now or later.

---

## Testing the integration end-to-end

Once the cloud repo is shipped, the full loop looks like:

```
┌──────────────┐     incidents.jsonl     ┌──────────────────┐
│ prod agent   │ ──────────────────────▶ │ local file       │
│              │ (monitor --incidents)   │ .evalview/       │
└──────────────┘                         │  incidents.jsonl │
                                          └────────┬─────────┘
                                                   │
                              cloud-owned ingestion (see above)
                                                   ▼
                                          ┌──────────────────┐
                                          │ evalview-cloud   │
                                          │ incident inbox   │
                                          └────────┬─────────┘
                                                   │
                                         "Ship as test" (click)
                                                   ▼
                                          ┌──────────────────┐
                                          │ GitHub App       │ ──▶ PR opened
                                          │ creates PR       │
                                          └──────────────────┘
```

Smoke test checklist:

- [ ] CLI `evalview monitor --incidents` writes valid JSON lines
- [ ] Cloud ingestion path (whichever the cloud repo picks) dedups on `slug`
- [ ] Incidents inbox shows new records in the dashboard
- [ ] "Ship as test" opens a PR whose YAML passes `evalview check`
- [ ] Port of `regression_synth.py` to TypeScript matches the Python output byte-for-byte (use a golden-file test)

---

## Open questions to resolve before shipping

1. **Where does `evalview monitor` run for paying customers?** Three options:
   - (a) Customer-hosted (current OSS flow) — simplest, but customers must run a process somewhere.
   - (b) Cloud-hosted runner — customer gives us an adapter endpoint + API key.
   - (c) Customer pushes results from their CI.
   V1 should probably be (a) with an upload agent, (c) for CI users, (b) only for Business tier.

2. **Token storage for agent API keys.** Do NOT collect customer OpenAI/Anthropic keys in the cloud in V1 — it's a rathole (abuse, cost attribution, secret rotation). Run agents client-side, upload results only.

3. **Multi-tenant isolation of the GitHub App.** Use one app, many installations. Never store raw tokens — always mint installation tokens on demand.

---

## Reference: files in *this* repo that define the contract

- `evalview/core/regression_synth.py` — pure synthesizer, **port this to TypeScript**
- `evalview/commands/autopr_cmd.py` — CLI entry point, good reference for UX flow
- `evalview/commands/monitor_cmd.py` — `_build_incident_record` (schema source of truth) and `_append_incidents` (writer)
- `tests/test_autopr.py` — round-trip tests the cloud port should pass

When in doubt, **the Python code is the spec**. Keep the TS port byte-for-byte compatible.
