# ADR-004: Agent as an explicit state machine with layered SQL safety and free-tier-aware LLM access

**Status:** accepted

## Context
The agent turns a question into one SQL query, runs it and explains the result. It must:
- be measurable: each capability can be switched off for the ablation evals
- be safe against a model that writes destructive or exfiltrating SQL, whether by mistake or through prompt injection
- run at $0 on free tiers with hard limits (Groq: 8,000 tokens/min and 1,000 requests/day per model), and test offline

## Decision
1. **LangGraph state machine, not a free-form tool-calling loop.** The nodes are `context → generate → validate → execute → answer`, and `repair` loops back to `generate` on:
   - a guard violation
   - an engine error
   - an empty result (retried once per distinct query)
   - a reply with no SQL

   The loop is bounded by `max_repairs`, and `max_repairs=0` is the "no self-correction" rung of the ablation ladder. The state machine makes every run traceable per node and keeps it deterministic enough to evaluate.
2. **Two independent safety layers.**
   - *sqlglot guard*: one statement; a SELECT/set-operation root; no DDL, DML, COPY, ATTACH, PRAGMA or SET anywhere in the tree; tables must be on the allowlist (the `marts` and `meta` schemas, never `staging` or `information_schema`); no file, network or engine functions (`read_csv`, `delta_scan`, `getenv`, ...). Violations go back to the model as fixable feedback.
   - *DuckDB sandbox*: the warehouse is `ATTACH`ed `READ_ONLY` into an in-memory database, then `enable_external_access=false` and `lock_configuration=true`. A guard bypass still can't write data, read files, install extensions or undo the settings. Queries run with a timeout (`interrupt()`) and a row cap.
3. **Refusal is a first-class outcome.** The model answers `CANNOT_ANSWER: <reason>` for out-of-scope or unsafe requests, and the run ends as `refused`, not as a failure.
4. **One LLM client for every provider** (LiteLLM), with:
   - short model aliases that carry free-tier limits
   - a sliding-window request and token rate limiter, corrected with actual usage
   - exponential backoff that honors `Retry-After`
   - per-model token accounting
5. **Content-addressed response store** keyed by (model, messages, params). It has three modes:
   - `cache` is the dev default: repeated eval runs cost nothing.
   - `replay` is for tests and CI: no network, no keys, and a miss is an error.
   - `live` always calls the API.

   Committed cassettes make end-to-end agent tests reproducible.
6. **Identical system prompt at every context level**, so ablation differences come only from context.

## Alternatives considered
- **ReAct/tool-calling agent with free exploration** (e.g. `list_tables`, `sample_rows` tools): more flexible, but runs vary in length, each step spends tokens against an 8k tokens/min budget, and it's hard to ablate. It can be revisited as an extra rung.
- **Guard only, or sandbox only:** each alone has failure modes. The guard alone relies on a parser matching the engine exactly, and the sandbox alone gives the model unhelpful errors.
- **Rewriting SQL to inject `LIMIT`:** rejected, because it changes the query being evaluated. The executor fetches `max_rows + 1` rows to detect truncation instead.

## Consequences
- Adding a provider means adding one `ModelSpec` line.
- Prompt or context changes invalidate cassettes by design. They are re-recorded with `WGPT_RECORD_CASSETTES=1`.
- The in-process limiter doesn't coordinate separate processes. Parallel CLI runs fall back to provider 429s and backoff.
