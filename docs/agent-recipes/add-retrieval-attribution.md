# Recipe: Add a Retrieval / Memory Attribution Judge

## Goal

Plug a smarter chunk-attribution method into
`evalview.core.retrieval_lineage`. The shipped baseline uses token-recall
(fraction of chunk tokens appearing in the output). It catches the obvious
"which chunks did the agent quote?" cases but is blind to the harder
ones: a chunk whose *facts* shaped the output even though the wording
diverges, a chunk whose entities the output cited via paraphrase.

## Read These Files First

- `evalview/core/retrieval_lineage.py` — the `AttributionJudge` callable
  type and the `attribute_chunks` entry point.
- `evalview/core/semantic_diff.py` — existing OpenAI-embedding pattern
  if you want a template for an embedding-based attribution method.
- `tests/test_retrieval_lineage.py` — patterns to mirror.

## Requirements

- **Match the type.** Your judge implements:

  ```
  AttributionJudge = Callable[
      [str, Sequence[RetrievedChunk]], Optional[Sequence[float]]
  ]
  ```

  Take `(output_text, chunks)`, return either a sequence of raw scores
  in `[0, 1]` parallel to `chunks`, OR `None` to fall back.

- **Fail soft.** Returning `None` (or raising) must never break
  attribution. Identical contract to the goal-drift judge.

- **No length asymmetry surprises.** The baseline uses recall on the
  *chunk*, not Jaccard, because a chunk shouldn't be penalized for the
  output containing extra material. If you build a different metric,
  document the asymmetry explicitly.

- **Score per chunk, not per (chunk, token).** The module normalizes
  across the chunk set; if your judge returns absolute "this chunk
  was 0.7 important" numbers, the normalization preserves their
  ordering.

## Steps

1. **Decide deterministic or LLM.** A deterministic upgrade (embedding
   cosine, LCS, BLEU-ish overlap) is preferable for cost. An LLM is
   right when you need true semantic attribution ("this chunk's date
   ended up in the output, paraphrased").

2. **Write the function.** Skeleton:

   ```python
   def my_attribution_judge(output_text, chunks):
       try:
           # Score each chunk against the output.
           return [score(c, output_text) for c in chunks]
       except Exception:
           return None  # fall back to the deterministic baseline
   ```

3. **Don't change the default.** The baseline stays deterministic so
   `attribute_chunks` works with no API keys.

4. **Add tests.** Mirror `tests/test_retrieval_lineage.py`:
   - Identical chunk + output → highest-influence chunk.
   - Disjoint chunk + output → ~0 influence.
   - Returning `None` does not crash `attribute_chunks`.
   - Raising does not crash `attribute_chunks`.
   - Normalization invariant (returned scores sum to 1.0 when any are nonzero).

5. **Document the cost.** Adapter authors will weigh judge cost against
   per-trace value. State expected dollars-per-call and per-chunk
   latency in the docstring.

## Done Criteria

- New judge function implements `AttributionJudge`.
- Default behavior unchanged; new tests cover happy / fallback paths.
- Docstring states determinism + cost contract.
- (Optional) Wire into an existing RAG adapter as a `judge=...` kwarg.

## Common Pitfalls

- **Returning the wrong list length.** The judge MUST return a sequence
  parallel to `chunks` (same length, same order). Anything else means
  the normalization step assigns scores to the wrong chunks.
- **Returning probabilities instead of influence.** "There's a 70%
  chance this chunk was used" and "this chunk contributed 70% of the
  influence" are different metrics. The latter is what callers want.
- **Forgetting prompt-injection defenses on LLM judges.** Outputs come
  from your agent, which could be relaying adversarial chunk content.
  Apply the same sanitization the LLM-as-judge evaluator uses (see
  SECURITY.md).
- **Sorting chunks before scoring.** The module relies on order — an
  attribution at position 2 in your return list is for `chunks[2]`.
  Re-sorting breaks the mapping.

## Roadmap (Good First Issues)

- **Embedding-cosine judge** using `text-embedding-3-small`. One
  embedding per chunk + one per output, then per-chunk cosine.
  Deterministic with cache, ~$0.00002/call.
- **Mechanistic baseline** that detects entity overlap (named
  entities, dates, IDs, numbers) — catches paraphrased citations the
  token-recall baseline misses.
- **`evalview retrieval-stats` subcommand** that aggregates lineage
  across many runs and surfaces dead-weight chunks (always-low score)
  for index pruning.
- **OTel attribute emission.** Adapters should set
  `agent.retrieval.influence_scores` (defined in
  `evalview/core/otel_semconv.py`) on the `agent.retrieval` span using
  the normalized scores from `attribute_chunks`.

If you're unsure whether your method beats the baseline, run both on
a fixed set of (output, chunks) fixtures and post the per-chunk score
comparison. That's almost always the right way in.
