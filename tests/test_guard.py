"""SQL guard: read-only, single statement, allowlisted tables, no file/network functions."""

from __future__ import annotations

import pytest

from warehouse_gpt.agent.guard import SQLGuard

ALLOWED = ["marts.fct_orders", "marts.dim_customers", "marts.fct_order_items", "meta.dq_check_results"]


@pytest.fixture(scope="module")
def guard() -> SQLGuard:
    return SQLGuard(ALLOWED)


@pytest.mark.parametrize(
    ("sql", "tables"),
    [
        ("select count(*) from marts.fct_orders", ["marts.fct_orders"]),
        ("select count(*) from fct_orders;", ["marts.fct_orders"]),  # bare names resolve to the allowlist
        ("with x as (select * from marts.fct_orders) select count(*) from x", ["marts.fct_orders"]),
        (
            "select c.customer_state, count(*) from fct_orders o "
            "join dim_customers c using (customer_unique_id) group by 1",
            ["marts.dim_customers", "marts.fct_orders"],
        ),
        ("select 1 union all select count(*) from marts.fct_orders", ["marts.fct_orders"]),
        ("select * from range(10)", []),
        ("select * from meta.dq_check_results", ["meta.dq_check_results"]),
        (
            "select * from fct_orders qualify row_number() over "
            "(partition by customer_unique_id order by purchase_ts) = 1",
            ["marts.fct_orders"],
        ),
    ],
)
def test_allows_read_only_queries(guard, sql, tables):
    r = guard.check(sql)
    assert r.ok, r.errors
    assert r.tables == tables


@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("select * from fct_orders; drop table fct_orders", "exactly one statement"),
        ("delete from marts.fct_orders", "only SELECT"),
        ("update marts.fct_orders set order_status = 'x'", "only SELECT"),
        ("create table t as select 1", "only SELECT"),
        ("select * into t from fct_orders", "INTO is not allowed"),
        ("copy fct_orders to 'x.csv'", "only SELECT"),
        ("attach 'x.db' as y", "only SELECT"),
        ("pragma database_list", "only SELECT"),
        ("set search_path = 'main'", "only SELECT"),
        ("select * from read_csv('x.csv')", "read_csv() is not allowed"),
        ("select * from read_parquet('x')", "read_parquet() is not allowed"),
        ("select * from delta_scan('x')", "delta_scan() is not allowed"),
        ("select getenv('HOME')", "getenv() is not allowed"),
        ("select * from duckdb_settings()", "duckdb_settings() is not allowed"),
        ("select * from 'data/raw/orders.csv'", "is not allowed"),
        ("select * from staging.stg_orders", "staging.stg_orders is not allowed"),
        ("select * from information_schema.tables", "information_schema.tables is not allowed"),
        ("select * from other.marts.fct_orders", "cross-database"),
        ("selec 1", "does not parse"),
        ("", "empty query"),
    ],
)
def test_blocks_unsafe_queries(guard, sql, error):
    r = guard.check(sql)
    assert not r.ok
    assert any(error in e for e in r.errors), r.errors


def test_error_lists_available_tables_for_repair(guard):
    r = guard.check("select * from orders")
    assert "available: marts.dim_customers" in r.errors[0]
