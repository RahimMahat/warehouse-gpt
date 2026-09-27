# ADR-002: Spark for bronze/silver, dbt-duckdb for gold

**Status:** accepted

## Context
The agent needs a fast, embedded, read-only SQL engine for serving queries. The data platform should also reflect how a real team works: a distributed engine for ingestion and cleaning, and SQL-first modeling for business logic.

## Decision
- **Bronze (Spark → Delta):** every column is kept as a string, with `_ingested_at` and `_source_file` lineage columns. Nothing is dropped at ingestion, so type problems show up in silver instead of disappearing silently.
- **Silver (Spark → Delta):** typing, dedup, normalization, then declarative DQ checks. `error` checks block the pipeline. `warn` checks are known caveats. All results are appended to the `dq_results` Delta table.
- **Gold (dbt-duckdb):** star schema built by reading silver Delta tables in place (`delta_scan`), with no copy step. Business definitions live in dbt docs, and the tests guard the facts the agent relies on.
- **Serving:** the agent opens `warehouse.duckdb` read-only.

## Consequences
- The same dbt project can target Databricks/Spark SQL by swapping the adapter. Delta is the shared format.
- DQ results flow downstream into `meta.dq_check_results`. This lets the agent add caveats to its answers, which is a key differentiator.
