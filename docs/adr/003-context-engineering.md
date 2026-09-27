# ADR-003: Context engineering with a MetricFlow semantic layer and a leveled renderer

**Status:** accepted

## Context
Most text-to-SQL failures on real warehouses are *semantic*, not syntactic:
- counting `customer_id` (per order) instead of `customer_unique_id` (per person)
- including canceled orders in revenue, or adding freight when the business means GMV
- filtering on `'cancelled'` or `'Sao Paulo'` when the data says `'canceled'` or `'SP'`
- comparing full years when 2016 and late 2018 are partial

The model can't infer these from DDL. They are data-team knowledge.

## Decision
1. **Metrics are defined once, in dbt's MetricFlow spec** (`models/marts/_semantic.yml`), and validated by `dbt parse`. Agent-facing hints (`synonyms`, `pitfalls`, `hidden`) go in `config.meta`, so the spec stays standard.
2. **A small in-house compiler** (`context/semantic.py`) reads `semantic_manifest.json` and turns each metric into a concrete SQL recipe. Ratio metrics become `num / NULLIF(den, 0)` over the same model. Join paths come from the entities. Tests execute every recipe and pin the known values: 96,096 customers, 99,441 orders, R$13.49M revenue.
3. **The physical schema always comes from the live database**, then gets enriched with dbt docs, keys and accepted values. The context can't mention a column that doesn't exist.
4. **Profiling** stores the exact values of low-cardinality text columns, time coverage, and failing DQ checks as caveats.
5. **Retrieval:** local embeddings (fastembed `bge-small-en-v1.5`, ONNX on CPU) in an embedded LanceDB index. It finds verified examples by comparing questions to questions, and prunes the schema once the warehouse passes 25 tables. It isn't used for 8 tables, because a retrieval miss costs more than about 2k extra tokens.
6. **Leveled rendering** (`ContextLevel` 1–4) makes each piece of knowledge separable, so the eval suite can measure how much each one adds.

## Alternatives considered
- **Using MetricFlow as the query engine (the agent emits metric queries):** strongest correctness, but it limits the agent to questions the semantic layer can express. Deferred. Recipes give most of the benefit while still allowing free-form SQL.
- **Hosted embeddings / vector DB:** rejected because of the $0 constraint. The local approach also has no network dependency.

## Consequences
- Adding a metric means editing YAML only. The tests confirm it executes, and the synonym-conflict test catches ambiguous vocabulary.
- Verified examples must stay disjoint from the eval golden set. A test enforces this once the golden set exists.
