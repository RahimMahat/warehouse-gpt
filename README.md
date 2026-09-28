# WarehouseGPT

**An analytics agent that answers business questions in plain English over a Spark + dbt lakehouse, and a measured answer to the question of *why* text-to-SQL agents get things wrong.**

> **Thesis:** text-to-SQL agents fail because of bad *data context*, not bad models.
> A governed semantic layer, dbt documentation and data-quality metadata fix that.
> An ablation eval suite measures how much each layer adds.

Everything runs at **$0**: local Spark, DuckDB, free-tier LLM APIs or local Ollama models.

## Highlights

- **Full data platform:** PySpark medallion pipeline on Delta Lake with a declarative data-quality framework, a dbt star schema with 49 tests, and a MetricFlow semantic layer with 17 governed metrics.
- **Agent:** a LangGraph state machine: retrieve context, generate SQL, validate, execute, and self-correct.
  - It runs behind two independent safety layers: a sqlglot guard, and a read-only DuckDB sandbox with its configuration locked.
  - It works across providers through one LiteLLM client (Groq, Gemini, local Ollama), with free-tier-aware rate limiting.
- **Evaluation:** a 58-question golden set scored by execution accuracy.
  - The comparator is presentation-tolerant but substance-strict.
  - A five-rung context ablation ladder, 95% confidence intervals, and a model leaderboard.
  - A 12-attack red-team suite.
- **Product surface:** a FastAPI service with Server-Sent-Events streaming, and a Streamlit chat UI with automatic charts and a per-step trace.
  - OpenTelemetry tracing (Arize Phoenix-ready) and structured JSON logs.
- **Engineering hygiene:** 120+ tests, ruff and mypy clean, and recorded LLM responses so the end-to-end tests run offline with no API keys.
  - Five architecture decision records.

## Results

Execution accuracy on 52 answerable golden questions, with 6 must-refuse questions scored separately. The golden set is disjoint from the examples the agent can retrieve, and a test enforces that. Full report: [`evals/REPORT.md`](evals/REPORT.md).

**Documentation alone lifts accuracy on two model families:**

| Model (Groq) | Raw DDL only | + dbt docs, keys, accepted values |
|---|---|---|
| `gpt-oss-120b` | 63.5% | **82.7%** (+11 fixed, −1 broken) |
| `qwen3-27b` | 73.1% | **82.7%** |

**Full context makes a small model beat both large ones:**

| Model (Groq) | Raw DDL only | Docs + semantic layer + retrieved examples |
|---|---|---|
| `gpt-oss-20b` | 63.5% | **98.0%** (50/51) |

- **What goes wrong without a semantic layer is definitional, not syntactic.** 7 of `gpt-oss-120b`'s 9 remaining errors on docs-only context used `order_value` (freight included) as "revenue". The business rule, "revenue excludes freight and cancellations", lives only in the governed metric definitions.
- **Intervals are reported, not hidden.** With n=52 the 95% CIs are ±10–14 points. Every cell in the report carries one, plus the questions fixed and broken between rungs.
- **The eval was audited too, not just the model.** Every failure was inspected before the numbers were published.
  - That found two comparator false negatives: `1.60` arriving as the float `1.6`, and an extra `NULL` group row.
  - It also found one genuinely ambiguous question, now scored with an accepted alternative reference.
  - Each fix has a regression test.

## Architecture

```
Olist CSVs (Kaggle, 9 tables, ~1.5M rows)
   │
Bronze  (Spark → Delta)   raw strings + lineage columns; nothing dropped at ingestion
   │
Silver  (Spark → Delta)   typed, deduplicated, normalized; 31 declarative DQ checks
   │                      results persisted to `dq_results`, which the agent reads as caveats
   ▼
Gold    (dbt-duckdb)      star schema read in place via delta_scan; 49 dbt tests
        marts.fct_orders · fct_order_items · fct_order_payments
        marts.dim_customers · dim_sellers · dim_products · dim_date · meta.dq_check_results
   │
Semantic layer            MetricFlow spec → compiled SQL recipes, join paths, synonyms, pitfalls
Metadata index            local embeddings (fastembed) in LanceDB: verified question→SQL examples
   │
Agent (LangGraph)
  question ─► context ─► generate SQL ─► sqlglot guard ─► DuckDB sandbox ─► answer + caveats
                             ▲                │                  │
                             └──── repair ◄───┴──────────────────┘
   │
FastAPI (JSON + SSE)  ─►  Streamlit UI          OpenTelemetry spans ─► Arize Phoenix
```

### Data issues found and handled (not hidden)

| Issue in source | Handling |
|---|---|
| `customer_id` is issued per order, not per person | `dim_customers` keyed on `customer_unique_id`; documented as the correct way to count customers |
| Duplicate review ids, multiple reviews per order | Silver keeps the latest answered review per order |
| ~1M geolocation rows, some outside Brazil | Collapsed to 19k zip-prefix centroids inside Brazil's bounding box |
| Misspelled columns (`product_name_lenght`) | Renamed once in silver |
| 2 categories missing from the translation file | Filled in silver |
| 8 delivered orders without a delivery date, 9 zero-value payments, 610 products without a category | Warn-level DQ checks, surfaced to the agent as caveats |

## Context engineering

The agent's context can be rendered at four levels, and the evals measure what each one adds:

| Level | Adds | ~Tokens |
|---|---|---|
| L1 `RAW_DDL` | `CREATE TABLE` statements only (what most demos use) | 0.7k |
| L2 `DBT_DOCS` | table/column docs, primary and foreign keys, accepted values | 2.4k |
| L3 `SEMANTIC` | 17 governed metric recipes, join paths, exact column values, synonyms, pitfalls, DQ caveats | 4.6k |
| L4 `EXAMPLES` | 3 verified question→SQL examples retrieved by local embeddings | 4.8k |

Metrics are defined once in dbt's MetricFlow spec ([`_semantic.yml`](warehouse/dbt/models/marts/_semantic.yml)) and compiled to SQL recipes. Tests execute every recipe and pin the known values: 96,096 customers, 99,441 orders, R$13.49M revenue.

```bash
uv run wgpt context show "revenue by month in 2018" --level 3   # see exactly what the LLM sees
```

## The agent

```bash
uv run wgpt models        # aliases, free-tier limits, key status
uv run wgpt ask "Top 5 categories by revenue in 2017 and their review scores?"
uv run wgpt ask "How many customers do we have?" --level 1 --max-repairs 0 --trace
```

- **Self-correction:** guard violations, engine errors, empty results and replies without SQL are fed back to the model, bounded by `max_repairs`.
- **Two independent safety layers:**
  - The sqlglot guard allows a single read-only query over allowlisted `marts`/`meta` tables, with no file, network or engine functions.
  - DuckDB attaches the warehouse `READ_ONLY` with external access disabled and the configuration locked, so a guard bypass still can't write or read files.
  - Out-of-scope or unsafe requests end as `CANNOT_ANSWER` refusals.
- **Provider-agnostic LLM client:** model aliases carry their free-tier limits. It uses a sliding-window request and token limiter, `Retry-After`-aware backoff, and per-model token accounting.
- **Reproducible:** LLM responses are content-addressed on disk.
  - `cache` mode makes re-runs free.
  - `replay` mode runs the end-to-end tests from committed cassettes.

## Evaluation

```bash
uv run wgpt eval run -m gpt-oss-120b            # golden set across the ablation ladder
uv run wgpt eval run -m gpt-oss-120b --suite redteam
uv run wgpt eval report                         # rebuild evals/REPORT.md
```

- **Golden set** ([`evals/golden.yml`](evals/golden.yml)): questions tagged by the skill they test (metric definitions, pitfalls, exact values, time handling, joins, ranking, window functions).
- **Comparator** ([`compare.py`](src/warehouse_gpt/evals/compare.py)):
  - Columns are matched by value, not name, and extra columns are ignored.
  - Numbers match within a tolerance or a rounding rule, and counts must match exactly.
  - Percent scale is accepted, and time keys match across representations.
  - Row order is checked only for rankings.
- **Ablation ladder:** R1 raw DDL → R2 docs → R3 semantic layer → R4 examples → R5 self-correction. The prompt is identical across rungs, so only the context changes.
- **Red-team suite** ([`evals/redteam.yml`](evals/redteam.yml)): destructive SQL, stacked-query injection, file and secret exfiltration, staging bypass, prompt leaks, a jailbreak, and an injected instruction. It reports which layer stopped each attack.

See [ADR-005](docs/adr/005-eval-methodology.md) for the methodology and its trade-offs.

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
| `POST /ask/stream` | Server-Sent Events: one `step` event per agent node (the SQL appears as soon as it is written), then `final` |
| `GET /models`, `GET /metrics`, `GET /health` | Model aliases and limits, governed metrics, cache stats |

- **Protections:** optional bearer token, per-client rate limit (429 with `Retry-After`), input validation, and a model allowlist.
- **Answer cache:** exact normalized-question match with LRU + TTL. It is deliberately *not* a semantic cache: "revenue in 2017" and "revenue in 2018" embed almost identically, and reusing one for the other would be a silent wrong answer.
- **Observability:**
  - OpenTelemetry spans use OpenInference conventions (AGENT → RETRIEVER / CHAIN / GUARDRAIL / TOOL / LLM), so Phoenix renders the full tree with prompts and token counts.
  - JSON logs carry a request id from the HTTP layer into the agent's worker thread.

## Quickstart

```bash
uv sync
cp .env.example .env        # add free GROQ_API_KEY / GEMINI_API_KEY; on Windows also JDK 21 + winutils
uv run wgpt data build      # download → spark → dbt build + docs → context index (~3 min)
uv run pytest               # 120+ tests; LLM tests replay recorded responses, no keys needed
uv run wgpt ask "Which 3 states have the most customers?"
```

### Windows notes
Spark on Windows needs:
- **JDK 17 or 21.** Hadoop's use of `Subject.getSubject` breaks on Java 23+.
- **`winutils.exe` and `hadoop.dll`** in `%HADOOP_HOME%\bin`.

Set `WGPT_JAVA_HOME` and `WGPT_HADOOP_HOME` in `.env`. See [ADR-001](docs/adr/001-spark-delta-versions.md).

## Tech stack

PySpark 4.0 · Delta Lake · dbt-duckdb · MetricFlow · DuckDB · LangGraph · LiteLLM · sqlglot · fastembed · LanceDB · FastAPI · Streamlit · OpenTelemetry · Arize Phoenix · Pydantic · Typer · uv · pytest · ruff · mypy

## Repo layout

```
src/warehouse_gpt/
  pipelines/     ingest, bronze, silver, data-quality framework, Spark session
  context/       catalog, semantic-layer compiler, profiler, vector index, context renderer
  agent/         LangGraph graph, LLM client, SQL guard, sandboxed executor, prompts, charts
  evals/         golden-set runner, comparator, report
  api/           FastAPI app, rate limiter, answer cache
  observability.py   tracing + structured logging
  cli.py         `wgpt` CLI
warehouse/dbt/   dbt project: staging → marts + meta, semantic layer, tests
warehouse/semantic/   verified question→SQL examples
evals/           golden set, red-team suite, results, REPORT.md
ui/              Streamlit app
tests/           unit, contract, agent, API, eval tests + recorded LLM cassettes
docs/adr/        architecture decision records
```

The full build story, covering the decisions, what broke and what changed, is in [STORY.md](STORY.md).
