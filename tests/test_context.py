"""Tests for the context layer: catalog, semantic layer, verified queries, rendering, index."""

from __future__ import annotations

import pytest

from warehouse_gpt.config import get_settings
from warehouse_gpt.context.catalog import load_catalog
from warehouse_gpt.context.examples import load_verified_queries
from warehouse_gpt.context.index import HashingEmbedder, MetadataIndex
from warehouse_gpt.context.profile import build_profile
from warehouse_gpt.context.render import ContextLevel, ContextRenderer
from warehouse_gpt.context.semantic import load_semantic_layer
from warehouse_gpt.context.store import build_documents

settings = get_settings()


@pytest.fixture(scope="module")
def catalog(warehouse):
    return load_catalog(settings.warehouse_path, settings.manifest_path)


@pytest.fixture(scope="module")
def semantic():
    if not settings.semantic_manifest_path.exists():
        pytest.skip("dbt artifacts missing; run `wgpt data dbt`")
    return load_semantic_layer(settings.semantic_manifest_path)


@pytest.fixture(scope="module")
def examples():
    return load_verified_queries(settings.verified_queries_path)


@pytest.fixture(scope="module")
def profile(catalog):
    return build_profile(settings.warehouse_path, catalog)


@pytest.fixture(scope="module")
def renderer(tmp_path_factory, catalog, semantic, profile, examples):
    docs = build_documents(catalog, semantic, profile, examples)
    index = MetadataIndex.build(tmp_path_factory.mktemp("idx"), docs, HashingEmbedder())
    return ContextRenderer(catalog, semantic, profile, examples, index)


# -- catalog: governance ----------------------------------------------------------------
def test_every_mart_column_is_documented(catalog):
    missing = [
        f"{t.fqn}.{c.name}"
        for t in catalog.tables.values()
        if t.schema == "marts"
        for c in t.columns
        if not c.description
    ]
    assert not missing, f"undocumented columns: {missing}"


def test_foreign_keys_and_primary_keys_detected(catalog):
    fks = {(f.table, f.column, f.ref_table) for f in catalog.foreign_keys}
    assert ("fct_order_items", "order_id", "fct_orders") in fks
    assert ("fct_orders", "customer_unique_id", "dim_customers") in fks
    assert catalog.table("fct_orders").column("order_id").is_primary_key  # type: ignore[union-attr]


# -- semantic layer ---------------------------------------------------------------------
def test_every_public_metric_recipe_executes(warehouse, semantic):
    for m in semantic.public_metrics:
        r = semantic.recipe(m.name)
        value = warehouse.sql(f"SELECT {r.select_sql} FROM {r.table}").fetchone()[0]
        assert value is not None and value > 0, m.name
        if m.name.endswith("_rate"):
            assert 0 < value < 1, m.name


def test_metric_recipes_match_known_values(warehouse, semantic):
    def value(name: str) -> float:
        r = semantic.recipe(name)
        return float(warehouse.sql(f"SELECT {r.select_sql} FROM {r.table}").fetchone()[0])

    assert value("customers") == 96_096  # distinct customer_unique_id, not customer_id
    assert value("orders") == 99_441
    assert value("revenue") == pytest.approx(13_494_400, rel=1e-4)
    assert value("average_review_score") == pytest.approx(4.09, abs=0.01)


def test_synonyms_are_unambiguous(semantic):
    seen: dict[str, str] = {}
    for m in semantic.public_metrics:
        for term in (m.name, m.label, *m.synonyms):
            key = term.lower()
            assert seen.get(key, m.name) == m.name, f"'{term}' maps to {seen[key]} and {m.name}"
            seen[key] = m.name
    idx = semantic.synonym_index()
    assert idx["gmv"] == "revenue"
    assert idx["aov"] == "average_order_value"


def test_join_paths_cover_fact_to_dimension(semantic):
    paths = {(p.left_table, p.right_table) for p in semantic.join_paths()}
    assert ("marts.fct_order_items", "marts.fct_orders") in paths
    assert ("marts.fct_order_items", "marts.dim_sellers") in paths
    assert ("marts.fct_orders", "marts.dim_customers") in paths


# -- verified examples ------------------------------------------------------------------
def test_verified_queries_execute(warehouse, examples):
    for q in examples:
        rows = warehouse.sql(q.sql).fetchall()
        assert rows, f"{q.id} returned no rows"


# -- rendering --------------------------------------------------------------------------
def test_context_grows_monotonically(renderer):
    q = "What was revenue by month in 2018?"
    sizes = [renderer.render(q, lvl).approx_tokens for lvl in ContextLevel]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_raw_ddl_has_no_business_knowledge(renderer):
    text = renderer.render("q", ContextLevel.RAW_DDL).text
    assert "--" not in text and "REFERENCES" not in text and "Governed" not in text


def test_semantic_level_carries_pitfalls_and_caveats(renderer):
    text = renderer.render("how many customers", ContextLevel.SEMANTIC).text
    assert "customer_id is issued per order" in text
    assert "Known data caveats" in text
    assert "boleto, credit_card" in text  # value hints


def test_examples_level_retrieves_relevant_examples(renderer):
    ctx = renderer.render("What was the revenue each month in 2018?", ContextLevel.EXAMPLES)
    assert "vq_revenue_by_month" in ctx.example_ids
