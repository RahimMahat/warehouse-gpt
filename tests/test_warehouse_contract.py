"""Contract tests on the built warehouse: the facts the agent and evals rely on."""

from __future__ import annotations

EXPECTED_TABLES = {
    ("marts", "fct_orders"),
    ("marts", "fct_order_items"),
    ("marts", "fct_order_payments"),
    ("marts", "dim_customers"),
    ("marts", "dim_sellers"),
    ("marts", "dim_products"),
    ("marts", "dim_date"),
    ("meta", "dq_check_results"),
}


def test_expected_tables_exist(warehouse):
    rows = warehouse.sql("select schema_name, table_name from duckdb_tables()").fetchall()
    assert set(rows) >= EXPECTED_TABLES


def test_known_dataset_facts(warehouse):
    orders, customers = warehouse.sql(
        "select count(*), count(distinct customer_unique_id) from marts.fct_orders"
    ).fetchone()
    assert orders == 99_441
    assert customers == 96_096


def test_customer_id_is_order_level(warehouse):
    # The trap the semantic layer must guard against: customer_id != customer.
    n_ids, n_people = warehouse.sql(
        "select count(distinct customer_id), count(distinct customer_unique_id) from marts.fct_orders"
    ).fetchone()
    assert n_ids > n_people
