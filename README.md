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
| 4. Eval harness: ablation ladder + model leaderboard | ⏳ next |
| 5. Hardening: self-correction, guardrails, cassettes, tracing | |
| 6. FastAPI + Streamlit UI | |
| 7. CI, Docker, Hugging Face Spaces deploy | |

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
