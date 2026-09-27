# WarehouseGPT

**A production-grade analytics agent that answers business questions in plain English over a Spark + dbt lakehouse, with measured accuracy.**

> Thesis: text-to-SQL agents fail because of bad *data context*, not bad models.
> A governed semantic layer, dbt tests and data-quality metadata fix that. The ablation eval
> suite measures how much each layer adds.

Everything runs at **$0**: local Spark, DuckDB, free-tier LLM APIs or local Ollama models.

## Status

| Milestone | State |
|---|---|
| 1. Data foundation: Spark bronze/silver (Delta Lake), DQ framework, dbt gold layer | ✅ done |
| 2. Semantic layer + metadata index | ✅ done |
| 3. Agent MVP: LangGraph pipeline, LiteLLM router, sqlglot guard, sandboxed DuckDB, CLI | ✅ done |
| 4. Eval harness: golden set, comparator, ablation ladder, leaderboard | ✅ done (partial results, see below) |
| 5. Hardening: self-correction, guard, cassettes, tracing, logging, answer cache, red-team suite | ✅ built (red-team run pending) |
| 6. FastAPI (JSON + SSE) + Streamlit UI | ✅ done |
| 7. CI, Docker, Hugging Face Spaces deploy | ⏳ next |

## Architecture (data layer)

```
Olist CSVs (Kaggle, 9 tables, ~1.5M rows)
   │  wgpt data download
   ▼
Bronze  (Spark → Delta)   raw strings + lineage columns; nothing dropped at ingestion
   │
Silver  (Spark → Delta)   typed, deduplicated, normalized; 31 declarative DQ checks
   │                      results persisted to `dq_results` (the agent reads these as caveats)
   ▼
Gold    (dbt-duckdb)      star schema read in place via delta_scan; 49 dbt tests
        marts.fct_orders · fct_order_items · fct_order_payments
        marts.dim_customers · dim_sellers · dim_products · dim_date
        meta.dq_check_results
```

Data issues found and handled (not hidden):

| Issue in source | Handling |
|---|---|
| `customer_id` is issued per order, not per person | `dim_customers` keyed on `customer_unique_id`; documented as the correct way to count customers |
| Duplicate review ids, multiple reviews per order | Silver keeps the latest answered review per order |
| ~1M geolocation rows, some outside Brazil | Collapsed to 19k zip-prefix centroids inside Brazil's bounding box |
| Misspelled columns (`product_name_lenght`) | Renamed once in silver |
| 2 categories missing from translation file | Filled in silver |
| 8 delivered orders without delivery date, 9 zero-value payments, 610 products without category | Warn-level DQ checks, surfaced to the agent as caveats |

## Context engineering (the ablation ladder)

The agent's context is rendered at four levels. The evals measure how much each one adds:

| Level | Adds | ~Tokens |
|---|---|---|
| L1 `RAW_DDL` | `CREATE TABLE` statements only (what most demos use) | 0.7k |
| L2 `DBT_DOCS` | table/column docs, primary and foreign keys, accepted values | 2.4k |
| L3 `SEMANTIC` | 17 governed metric recipes (MetricFlow), join paths, exact column values, synonyms, pitfalls, DQ caveats | 4.6k |
| L4 `EXAMPLES` | 3 verified question→SQL examples retrieved by local embeddings (LanceDB) | 4.8k |

Metrics are defined once in dbt's MetricFlow spec ([`_semantic.yml`](warehouse/dbt/models/marts/_semantic.yml)) and compiled to SQL recipes. Tests execute every recipe and pin the known values. See [ADR-003](docs/adr/003-context-engineering.md).

```bash
uv run wgpt context build                                   # profile + embed + index
uv run wgpt context show "revenue by month in 2018" --level 3   # see exactly what the LLM sees
```

## The agent

```
question ─► context (L1–L4) ─► generate SQL ─► sqlglot guard ─► DuckDB sandbox ─► answer + caveats
                                    ▲                │                  │
                                    └──── repair ◄───┴──────────────────┘
                     (guard violation · engine error · empty result · no SQL; bounded by max_repairs)
```

```bash
uv run wgpt models                                           # aliases, free-tier limits, key status
uv run wgpt ask "Top 5 categories by revenue in 2017 and their review scores?"
uv run wgpt ask "How many customers do we have?" --level 1 --max-repairs 0 --trace
```

- **Models:** one LiteLLM client across Gemini (free tier), Groq (free tier: `gpt-oss-120b`, `qwen3-27b`) and local Ollama.
  - A sliding-window limiter keeps within Groq's 8k tokens/min.
  - Retries honor `Retry-After`.
  - Tokens are counted per model.
- **Safety, in two independent layers:**
  - A sqlglot guard allows one read-only query over allowlisted `marts`/`meta` tables, with no file, network or engine functions.
  - DuckDB attaches the warehouse `READ_ONLY` with external access disabled and the configuration locked.
  - Out-of-scope or unsafe requests end as `CANNOT_ANSWER` refusals.
- **Reproducible:** LLM responses are content-addressed on disk.
  - `WGPT_LLM_MODE=cache` (default) makes re-runs free.
  - `replay` runs the end-to-end tests from committed cassettes, with no network and no keys.

See [ADR-004](docs/adr/004-agent-architecture.md).

## Results

Execution accuracy on 52 answerable golden questions. The set is disjoint from the examples the agent retrieves, and a test enforces that. Full report: [`evals/REPORT.md`](evals/REPORT.md).

| Model (Groq free tier) | R1 raw DDL | R2 + dbt docs | R4 full context (docs + semantic layer + examples) |
|---|---|---|---|
| `gpt-oss-120b` | 63.5% | **82.7%** | *pending* |
| `qwen3-27b` | 73.1% | **82.7%** | *pending* |
| `gpt-oss-20b` | 63.5% | – | **98.0%** (50/51) |

- **Documentation alone is worth about 10–19 points.** Adding dbt descriptions, keys and accepted values fixed 11 questions and broke 1 on `gpt-oss-120b`.
- **Full context takes a small model from 63.5% to 98%.** `gpt-oss-20b` with the semantic layer and retrieved examples beats both larger models running on docs alone.
- **The remaining R2 errors are mostly definitional.** 7 of `gpt-oss-120b`'s 9 R2 failures used `order_value` (freight included) as "revenue". Only the governed metric definitions at R3 encode that rule.
- **Honest intervals.** With n=52 the 95% CIs are ±10–14 points, and the report shows them for every cell. Refusal questions (6) are scored separately.

> **Why some cells are pending:** Groq's free tier has an undocumented cap of **200K tokens per model per rolling 24h**. It isn't in the rate-limit headers; I found it by forcing a refusal. The ~5K-token R3/R4 prompts exhausted it mid-run. Runs are resumable from the response cache at zero cost, and the report excludes partial rungs rather than estimating them (`MIN_COVERAGE`).

**Auditing the eval, not just the model.** Every failure was inspected before the numbers were published. This found two comparator false negatives: `1.60` arriving as the float `1.6`, and an extra `NULL` group row. It also found one genuinely ambiguous question, now scored with an accepted alternative reference (`alt_sql`). Each fix has a regression test.

## API, UI and tracing

```bash
uv run wgpt serve                    # FastAPI on :8000  (OpenAPI docs at /docs)
uv run wgpt ui                       # Streamlit chat on :8501, talks to the API over SSE
uv run --extra phoenix phoenix serve # optional: Arize Phoenix trace UI on :6006
WGPT_TRACING=true uv run wgpt serve  # ...and send traces to it
```

| Endpoint | |
|---|---|
| `POST /ask` | Full answer as JSON: rows, answer, SQL, Vega-Lite chart spec, token usage, per-node trace |
| `POST /ask/stream` | Server-Sent Events: one `step` event per agent node (SQL shows up as soon as it is written), then `final` |
| `GET /models`, `GET /metrics`, `GET /health` | Model aliases and free-tier limits, governed metrics, cache stats |

- **Protections:**
  - Optional bearer token (`WGPT_API_TOKEN`).
  - A per-client sliding-window rate limit that returns 429 with `Retry-After`.
  - Input validation.
  - A model allowlist, so clients can't point the server at arbitrary providers.
- **Answer cache:** exact normalized-question match with LRU + TTL. It is deliberately *not* a semantic cache: "revenue in 2017" and "revenue in 2018" embed almost identically, and reusing one for the other would be a silent wrong answer.
- **Observability:**
  - OpenTelemetry spans follow OpenInference conventions (AGENT → RETRIEVER / CHAIN / GUARDRAIL / TOOL / LLM with messages and token counts), so Phoenix renders the full tree.
  - JSON logs carry a request id from the HTTP middleware into the agent's worker thread.
  - Tracing costs nothing when off.
- **Streaming design:** the graph runs on a worker thread and hands events over a queue, so a run and its trace stay on one thread even though the ASGI server iterates the response from a thread pool.

## Quickstart

```bash
uv sync
cp .env.example .env        # add free GEMINI_API_KEY / GROQ_API_KEY; on Windows also JDK 21 + winutils
uv run wgpt data build      # download → spark → dbt build + docs → context index (~3 min)
uv run pytest               # unit + warehouse contract tests
```

Individual steps: `wgpt data download`, `wgpt data spark`, `wgpt data dbt`.

### Windows notes
Spark on Windows needs:
- **JDK 17 or 21.** Hadoop's use of `Subject.getSubject` breaks on Java 23+.
- **`winutils.exe` and `hadoop.dll`** in `%HADOOP_HOME%\bin`.

Set `WGPT_JAVA_HOME` and `WGPT_HADOOP_HOME` in `.env`. The Spark session factory wires them in, so you don't need to change system settings. See [ADR-001](docs/adr/001-spark-delta-versions.md).

## Repo layout

```
src/warehouse_gpt/
  config.py                 settings (env / .env, prefix WGPT_)
  cli.py                    `wgpt` CLI
  pipelines/                ingest, bronze, silver, dq framework, spark session
  context/                  catalog, semantic compiler, profiler, vector index, renderer
warehouse/dbt/              dbt project (staging → marts, meta), semantic layer, tests, docs
warehouse/semantic/         verified question→SQL examples
tests/                      spark unit tests + warehouse contract tests
docs/adr/                   architecture decision records
```
