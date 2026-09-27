# ADR-005: Evaluation methodology: execution accuracy on a disjoint golden set, as an ablation ladder

**Status:** accepted

## Context
The project's claim is that *data context*, not the model, is the main lever for text-to-SQL accuracy. That claim needs a measurement that is:
- reproducible
- robust to presentation differences
- honest about its uncertainty
- cheap enough to run on free tiers

## Decision
1. **Golden set** (`evals/golden.yml`): 52 answerable questions plus 6 that must be refused (out of scope, no such data, PII, destructive or injected instructions).
   - Each question has hand-written reference SQL and tags that attribute failures: `metric`, `pitfall`, `values`, `time`, `join`, `ranking`, `window`, `stats`.
   - Reference results are computed live from the reference SQL at eval time, so they can't go stale after a data rebuild.
2. **Disjoint from retrieval.** The L4 context retrieves verified examples, so the golden set must not contain near-copies of them. `tests/test_evals.py` fails if any golden question has token Jaccard ≥ 0.6 with a verified question, or if the SQL is identical. It caught two leaks while the set was being written.
3. **Execution accuracy with a presentation-tolerant, substance-strict comparator** (`evals/compare.py`):
   - Columns are matched by value, not name, and extra predicted columns are ignored.
   - Numbers match within 1e-4 relative tolerance, or when they equal the gold value rounded to the prediction's printed precision (at least 2 decimals).
   - Counts must match exactly.
   - A consistent ×100 percent scale is accepted.
   - Time keys match across representations only when the projection is lossless.
   - Row order is checked only for rankings.

   Every rule has a unit test, including the negative cases (lossy projections, misaligned rows, 1-decimal rounding).
4. **Ablation ladder.** Each rung adds one capability, and everything else (prompt, model, temperature, questions) stays fixed:

   | Rung | Context | Self-correction |
   |---|---|---|
   | R1 | raw DDL | off |
   | R2 | + dbt docs | off |
   | R3 | + semantic layer | off |
   | R4 | + retrieved examples | off |
   | R5 | L4 | up to 2 repairs |

5. **Uncertainty is reported, not hidden.**
   - Wilson 95% intervals on every accuracy.
   - Per-question transitions between rungs ("+k fixed, −j broken"), because two rungs can have similar totals but different failure sets.
6. **Refusals are scored separately.** A correct refusal on a negative question counts toward refusal accuracy. A refusal on an answerable question is a *false refusal* and counts as wrong.
7. **Cost control.** Runs go through the response store (`cache` mode):
   - Re-running is free.
   - An interrupted run resumes where it stopped.
   - R5 reuses R4's first attempt from the cache and pays only for repairs.

   The runner stops after 3 consecutive provider failures (daily quota, outage) instead of recording them as wrong answers.

## Alternatives considered
- **Exact-match on SQL text or AST:** rejected. Many correct queries differ syntactically.
- **LLM-as-judge for correctness:** rejected as the primary metric, because it is non-deterministic, costs tokens, and has its own biases. It may be added later as a secondary faithfulness score for the natural-language answer.
- **Public benchmarks (Spider/BIRD):** they don't test the thing this project claims: governed business definitions and data caveats on one specific warehouse. The small, domain-specific golden set is a deliberate trade-off: n≈50 gives wide intervals (±12–14 pp), which the report shows.

## Consequences
- Differences between adjacent rungs smaller than the interval width are reported as-is and are not claimed as significant.
- Adding questions only requires YAML. The integrity tests guard against broken SQL, guard violations and leakage.
- Golden reference SQL encodes business decisions (for example, "revenue" excludes freight and cancellations). That is the point of the project: the agent is judged on the business's definitions, not on a plausible reading.

## Addendum: free-tier quotas and partial runs
- **Groq's free tier caps each model at 200K tokens per rolling 24h.** This isn't in the rate-limit headers, which only advertise 1K requests/day and 8K tokens/min. A full five-rung ladder is about 1M tokens per model, so it can't finish in a day. The limit showed up as 30–40 minute `Retry-After` waits. The retry log now records the provider's message, so a quota like this is visible immediately.
- **Partial runs are rebuilt, not estimated.** `wgpt eval run --cached-only` re-scores every cached response and skips the rest, with no API calls. The report keeps a (model, rung) cell only if ≥90% of answerable questions were scored. R5 needs 100%, because its missing items are exactly the ones whose repair attempts never ran, so a partial R5 would silently look like R4.
- **Multiple references.** A question with two defensible readings lists the second as `alt_sql`, and matching any reference counts. This is used for one question so far (an order delivered and later canceled).
