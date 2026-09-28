# The story of WarehouseGPT

## Overview

WarehouseGPT is an analytics agent. You ask it a business question in plain English, such as "Which 5 states generated the most revenue in 2018?", and it writes SQL, runs it safely against a real warehouse, and explains the answer, with the SQL, a chart and any relevant data-quality caveats.

The agent is only half the project. The other half is an evaluation suite that measures *why* agents like this succeed or fail. I came to this as a data engineer, and my hypothesis was that text-to-SQL agents mostly fail because nobody gives them the context a new analyst would get on day one, not because the model is too weak. What does "revenue" mean here? Which customer id counts people? Which months are incomplete? I built the data platform, the context layers and the evals to test that hypothesis, and the numbers support it.

## The problem

"Chat with your database" demos usually work like this: paste the `CREATE TABLE` statements into a prompt and let the model write SQL. On a real warehouse that produces answers that *run* but are *wrong*:

- **Revenue** summed over `order_value` includes freight and canceled orders. The business defines revenue as item prices on non-canceled orders.
- **Customers** counted with `customer_id`: in this dataset that id is issued per order, so it overcounts people by several thousand.
- **Filters** use `'cancelled'` or `'Sao Paulo'` when the data says `'canceled'` and `'SP'`.
- **Year-over-year** comparisons treat 2016 and late 2018 as full years, when they are partial periods.

A data engineer would recognize every one of these. They are exactly what dbt docs, a semantic layer and data-quality checks are for. The question I wanted to answer was *how much* each of those layers actually helps an LLM, measured rather than asserted.

## Timeline (September 2026)

1. **Data foundation.** Ingest the Olist Brazilian e-commerce dataset (9 tables, about 1.5M rows) with PySpark into bronze and silver Delta tables. Build a small declarative data-quality framework, then a dbt star schema on DuckDB with 49 tests.
2. **Semantic layer and metadata.** Define 17 business metrics in dbt's MetricFlow spec. Compile them into concrete SQL recipes. Profile the warehouse for exact values and time coverage. Index 22 verified example queries with local embeddings.
3. **The agent.** A LangGraph pipeline with a provider-agnostic LLM client, a SQL guard, a sandboxed executor, a self-correction loop and a CLI.
4. **Evaluation.** A 58-question golden set, an execution-accuracy comparator, the ablation runner and the report. Then audit the eval itself.
5. **Hardening.** OpenTelemetry tracing, structured logging, an answer cache and a red-team suite.
6. **Product surface.** A FastAPI service with streaming, and a Streamlit chat UI.

## Stack

- **Processing:** PySpark 4.0 and Delta Lake, in local mode.
- **Warehouse:** DuckDB with dbt-duckdb, reading the Delta tables in place.
- **Semantic layer:** dbt MetricFlow spec, plus my own compiler to SQL recipes.
- **Retrieval:** fastembed (`bge-small-en-v1.5`, ONNX on CPU) and LanceDB, both embedded, with no services to run.
- **Agent:** LangGraph, LiteLLM (Groq, Gemini, Ollama) and sqlglot.
- **Serving:** FastAPI (JSON and Server-Sent Events) and Streamlit.
- **Observability:** OpenTelemetry with OpenInference conventions, Arize Phoenix, and structlog.
- **Tooling:** uv, pytest, ruff, mypy and Typer.
- **Budget:** $0. Every component is local or on a free tier.

## Scale

- **Data:**
  - 99,441 orders, 112,650 order items, 103,886 payments, 96,096 unique customers, 32,951 products and 3,095 sellers.
  - About 1M raw geolocation rows, collapsed to 19k centroids.
- **Warehouse:** 7 mart tables and a meta table, 31 Spark data-quality checks, and 49 dbt tests.
- **Semantic layer:** 17 public metrics, 6 semantic models and 22 verified question→SQL examples.
- **Evals:** 52 answerable golden questions, 6 must-refuse questions, and 12 red-team attacks.
- **Code:** about 4,600 lines of Python in the package, 1,200 lines of tests (120+ test cases), 5 ADRs.

## Architecture

```
Raw CSVs ─► Spark bronze ─► Spark silver (+ DQ checks) ─► dbt gold (DuckDB) ─► semantic layer + index
                                                                                     │
      question ─► context (L1–L4) ─► generate SQL ─► guard ─► sandbox ─► answer ◄────┘
                                          ▲             │         │
                                          └── repair ◄──┴─────────┘
```

The agent's context can be rendered at four levels, and those levels are the rungs of the ablation ladder:

1. Raw DDL.
2. Adds dbt docs, keys and accepted values.
3. Adds governed metric recipes, join paths, exact column values, synonyms, pitfalls and DQ caveats.
4. Adds retrieved verified examples.

A fifth rung switches on self-correction. The system prompt is identical at every rung, so any difference comes from the context alone.

## Result

| Comparison | Before | After |
|---|---|---|
| `gpt-oss-120b`: raw DDL → + dbt docs | 63.5% | **82.7%** |
| `qwen3-27b`: raw DDL → + dbt docs | 73.1% | **82.7%** |
| `gpt-oss-20b`: raw DDL → full context | 63.5% | **98.0%** |

- Documentation alone added 10–19 points.
- With the full context, the *smallest* model beat both larger models running on docs alone.
- Most errors that remained on docs-only context came down to one business rule: revenue excludes freight and cancellations. Nothing in a schema says that. It lives only in the governed metric definitions.

## The interesting decisions

**Spark and dbt, not just one of them.** Bronze and silver are Spark jobs writing Delta, which is where a real team would do heavy, typed, deduplicated processing. Gold is dbt on DuckDB, reading those Delta tables in place through `delta_scan`. Delta is the contract between the two, so the dbt project could move to Databricks by swapping the adapter.

**Metrics defined once, in a standard spec.** Rather than inventing a YAML format, I wrote the metrics in dbt's MetricFlow spec, so `dbt parse` validates them. A small compiler turns each one into a concrete SQL recipe. Agent-only hints (synonyms, pitfalls) go in `config.meta`, so the spec stays standard. Tests run every recipe and pin known values such as 96,096 customers and R$13.49M revenue.

**Data-quality results become answer caveats.** The Spark DQ framework writes its results to a Delta table, dbt exposes it as `meta.dq_check_results`, and the profiler turns failing warn-level checks into caveats. When the agent answered a category-revenue question, it pointed out on its own that 610 products have no category.

**A state machine, not a free-form tool loop.** A ReAct-style agent is more flexible, but its runs vary in length, it burns tokens exploring, and it's hard to ablate. An explicit LangGraph graph makes each capability something the evals can switch off, and makes every run traceable step by step.

**Two safety layers that don't trust each other.** The sqlglot guard gives the model precise, fixable feedback ("table staging.stg_orders is not allowed"). The DuckDB sandbox is the backstop: the warehouse is attached `READ_ONLY` into an in-memory database, external access is disabled, and the configuration is locked, so even a guard bug can't write data or read files.

**A cache that deliberately isn't semantic.** A semantic cache would match "revenue in 2017" to "revenue in 2018", since they embed almost identically, and silently return the wrong year. So the answer cache matches exact normalized questions only, and fuzzy reuse happens safely one layer down, where LLM responses are keyed on the full prompt.

**Recorded LLM responses as test fixtures.** Every LLM call goes through a content-addressed store. In development it makes re-runs free. In tests, `replay` mode runs the end-to-end agent from committed responses, with no network and no API keys, and a changed prompt is a loud failure rather than a silent API call.

**A comparator that is tolerant of presentation and strict on substance.**
- Columns are matched by value, extra columns are ignored, and percent scale and time-key formats are normalized.
- Integer counts must match exactly, rounding coarser than 2 decimals is rejected, and rows must align, not just columns.
- Every rule has a unit test, including the negative cases.

## What broke, and what I changed

**Java 24 and Hadoop.** Spark died with `getSubject is not supported`: Hadoop relies on a security API removed in Java 23. I moved to a portable JDK 21 and wired `JAVA_HOME` and `HADOOP_HOME` in the Spark session factory, so nothing needs installing system-wide. Windows also needed `winutils.exe` and `hadoop.dll`.

**Delta and Spark version drift.** Delta 4.4 against Spark 4.2 failed with a `NoSuchMethodError` deep in the SQL parser. I pinned Spark 4.0.x with Delta 4.0.x and wrote down why in ADR-001.

**Python workers crashing.** `createDataFrame` from a Python list crashed the Python worker on Windows. I rewrote it to build literal rows entirely in the JVM (`F.inline(F.array(F.struct(...)))`), which also avoids a round trip.

**Timestamps that shifted days.** Timestamps were landing as timezone-aware, which could move an order into the previous year near midnight. I switched to `to_timestamp_ntz`, enabled Delta's `timestampNtz` table feature, and rebuilt silver.

**MetricFlow ratio metrics.** Ratio metrics must reference metrics, not measures. I added hidden helper metrics and filtered them out of what the agent sees.

**DuckDB refusing a second connection.** The sandboxed executor opened the warehouse with custom settings, and DuckDB refused because another connection to the same file already existed with different settings. That would also have broken the API server. Attaching the file `READ_ONLY` into an in-memory database fixed it, and it turned out to be a stronger sandbox, because the settings could then be locked.

**Free tiers that aren't what they say.** Gemini 2.5 Flash was closed to new accounts, and Gemini's free tier turned out to allow 20 requests per day per model. Groq advertises 1,000 requests per day and 8,000 tokens per minute, but it also has a cap of 200,000 tokens per model per day that never appears in its headers. I found it only by deliberately sending an oversized request and reading the refusal. With ~5,000-token context prompts, that cap shapes how much of the ablation ladder can run per day. I changed three things:
- The retry log now records the provider's error message.
- Eval runs resume from the response cache, so nothing is lost when a quota hits.
- A `--cached-only` mode rebuilds scored results without any API calls.

**The eval lying to me in both directions.** Before publishing anything, I inspected every failure:
- **Two comparator false negatives:** a percentage printed as `1.60` arrives as the float `1.6`, which broke my rounding rule, and an extra `NULL` group row sank an otherwise correct answer. Both are fixed with regression tests.
- **One genuinely ambiguous question:** an order that was delivered and later canceled. It now accepts either reading through an alternative reference.
- **Two leaks into the test set:** two golden questions were near-copies of examples the agent can retrieve. A disjointness test caught them, and I replaced them.
- **A partial result that looked like a real one:** a partially rebuilt self-correction rung looked identical to the rung below it, because the missing items were exactly the ones where repair would have happened. The report now drops any rung with incomplete coverage instead of estimating it.

**Smaller things that were still real bugs:**
- A log field named `level` was silently overwritten by the log level.
- Streaming needed the agent to run on its own worker thread, so a single run and its trace span stay on one thread while the web server reads the stream from a thread pool.
- Request ids had to be copied explicitly into that worker thread for the logs to line up.

## What I'd do next

- **A token-efficient context rung:** include only the metrics and value lists relevant to each question. It would likely keep accuracy while halving tokens, which on free tiers directly doubles how much can be evaluated per day.
- **Metric queries instead of free-form SQL** for questions the semantic layer can fully express, with free-form SQL kept as the fallback.
- **A larger golden set** to narrow the ±10–14 point confidence intervals.
