# Recipe: Add a Goal-Drift Judge

## Goal

Plug a smarter goal-drift detector into `evalview.core.goal_drift`. The
shipped baseline uses Jaccard token overlap — fine for the "agent
wandered into a totally different topic" case, but blind to the
common-words-different-intent failure (e.g. "cancel my refund" vs
"check my refund status" share most tokens but differ in intent).

A good judge catches that case while the baseline doesn't.

## Read These Files First

- `evalview/core/goal_drift.py` — the `GoalDriftJudge` callable type and
  the `analyze_goal_drift` entry point.
- `evalview/core/freshness.py` — the existing tokenization pattern,
  shared with goal_drift; mirror it if you build a deterministic judge.
- `evalview/evaluators/` — for a reference of how LLM judges are
  organized in the codebase.
- `tests/test_goal_drift.py` — patterns to mirror.

## Requirements

- **Match the type.** Your judge implements:

  ```
  GoalDriftJudge = Callable[[str, str], Optional[float]]
  ```

  Take `(stated_goal, trajectory_summary)`, return a float in `[0.0,
  1.0]` (higher = more on-goal) OR `None` to fall back to the
  deterministic baseline.

- **Fail soft.** Returning `None` (or raising) must never break the
  caller. The wrapper in `analyze_goal_drift` catches exceptions and
  falls back to Jaccard.

- **Keep it stateless.** A judge is invoked many times during one
  `analyze_per_step` run. Cache externally if you must, but don't
  store conversation state inside the callable.

- **Bound the cost.** Even cheap judges become expensive in per-step
  mode. Document expected cost-per-call in your docstring so callers
  can decide whether to use you in `analyze_per_step` vs only
  `analyze_goal_drift`.

## Steps

1. **Decide deterministic or LLM.** A deterministic judge that beats
   Jaccard (e.g. embedding cosine via a tiny local model, or a smarter
   bag-of-N-grams) is preferable when it works — no per-call cost, no
   network. An LLM judge is the right choice when you genuinely need
   semantic understanding.

2. **Write the function.** New file in `evalview/judges/` (or
   `evalview/core/` if it's pure). Example skeleton:

   ```python
   def my_drift_judge(stated_goal: str, trajectory_summary: str) -> Optional[float]:
       # Compute similarity in [0, 1]; return None on any error.
       ...
   ```

3. **Don't change the default.** The baseline stays deterministic so
   `analyze_goal_drift` works without API keys. Callers opt in via
   `judge=my_drift_judge`.

4. **Add tests.** Mirror `tests/test_goal_drift.py`:
   - Round-trips stated_goal == trajectory_summary → score ≥ 0.95.
   - Disjoint inputs → score ≤ 0.2.
   - Returning `None` doesn't crash `analyze_goal_drift`.
   - For LLM judges: gate the live test with `requires_api_key`
     marker (see `pytest.ini`); a deterministic mock test runs in CI.

5. **Document the call cost.** Adapter authors will want to know the
   per-call $ before they wire your judge into `analyze_per_step`.

## Done Criteria

- New judge function implements the `GoalDriftJudge` signature.
- Existing tests still pass; no change to default behavior.
- New tests cover happy path, fallback path, and (for LLM) cost
  contract.
- Docstring states whether the judge is deterministic, what model it
  uses (if any), and the expected per-call cost.

## Common Pitfalls

- **Returning a number outside [0, 1].** `analyze_goal_drift` clamps,
  but a judge that consistently returns >1 indicates a similarity
  metric that isn't normalized — the comparison threshold gets
  meaningless.
- **Treating "no signal" as drift.** Empty inputs should return None
  (or a high similarity if your judge is positively certain), not 0.
  The drift detector handles missing data; don't double-count it.
- **Adding latency to the per-step path.** A 200ms LLM call × 30
  trajectory steps × 10 monitor cycles = 60 seconds. Either return
  early when the deterministic baseline is decisive, or document that
  your judge is intended only for whole-trajectory analysis.
- **Forgetting prompt-injection defenses on LLM judges.** Trajectory
  summaries are derived from agent output — assume they contain
  adversarial text. Use the same sanitization helpers as the
  LLM-as-judge evaluator (see SECURITY.md).

## Roadmap (Good First Issues)

- **Embedding-cosine judge** using `text-embedding-3-small` (already
  used in `evalview/core/semantic_diff.py`). One API call, deterministic
  per cache hit, ~$0.00002/call.
- **Bag-of-bigrams baseline.** Pure, no dependencies; should beat
  unigram Jaccard on the "common words, different intent" case.
- **Per-step `--drift` flag on `evalview replay`** that renders the
  similarity sparkline so reviewers can see *when* the agent wandered.
- **OTel attribute emission.** Adapters that compute drift should set
  `agent.goal.drift_delta` (defined in `evalview/core/otel_semconv.py`)
  on the `agent.turn` span.
