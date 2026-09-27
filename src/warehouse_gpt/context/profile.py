"""Data profile computed from the live warehouse: value hints, coverage and DQ caveats.

Value hints (the actual distinct values of low-cardinality text columns) are one of the
highest-leverage pieces of context for text-to-SQL: they stop the model from filtering on
'Sao Paulo' when the data says 'SP', or 'cancelled' when the data says 'canceled'.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb

from warehouse_gpt.context.catalog import Catalog

MAX_HINT_CARDINALITY = 80
TEXT_TYPES = ("VARCHAR",)


@dataclass
class DataProfile:
    # "marts.fct_orders.order_status" -> ["delivered", ...]
    value_hints: dict[str, list[str]] = field(default_factory=dict)
    # "marts.fct_orders.purchased_at" -> ["2016-09-04 ...", "2018-10-17 ..."]
    time_coverage: dict[str, list[str]] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DataProfile:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _skip_hint(table: str, column: str) -> bool:
    # Keys are high-cardinality; calendar names are common knowledge; the Portuguese
    # category name duplicates the English one.
    return column.endswith(("_id", "_key")) or table == "dim_date" or column.endswith("_pt")


def _caveat_text(table: str, check: str, failing: int) -> str:
    return f"{table}: {failing:,} row(s) fail data-quality check '{check}'."


def build_profile(warehouse_path: Path, catalog: Catalog) -> DataProfile:
    profile = DataProfile()
    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        for t in catalog.tables.values():
            if t.schema != "marts":
                continue
            rel = f'"{t.schema}"."{t.name}"'
            for c in t.columns:
                key = f"{t.fqn}.{c.name}"
                if c.data_type in TEXT_TYPES and not _skip_hint(t.name, c.name):
                    n = con.execute(f'select count(distinct "{c.name}") from {rel}').fetchone()[0]  # type: ignore[index]
                    if 0 < n <= MAX_HINT_CARDINALITY:
                        rows = con.execute(
                            f'select distinct "{c.name}" from {rel} where "{c.name}" is not null order by 1'
                        ).fetchall()
                        profile.value_hints[key] = [r[0] for r in rows]
                elif c.data_type.startswith(("TIMESTAMP", "DATE")) and t.name.startswith("fct_"):
                    lo, hi = con.execute(f'select min("{c.name}"), max("{c.name}") from {rel}').fetchone()  # type: ignore[misc]
                    if lo is not None:
                        profile.time_coverage[key] = [str(lo), str(hi)]
        if "meta.dq_check_results" in catalog.tables:
            rows = con.execute(
                "select table_name, check_name, failing_rows from meta.dq_check_results "
                "where not passed order by table_name, check_name"
            ).fetchall()
            profile.caveats = [_caveat_text(*r) for r in rows]
    finally:
        con.close()
    # Coverage facts that matter for correct answers but live in no schema.
    purchased = profile.time_coverage.get("marts.fct_orders.purchased_at")
    if purchased:
        profile.caveats.append(
            f"Orders span {purchased[0][:10]} to {purchased[1][:10]}; 2016 and Sep-Oct 2018 are "
            "sparse partial periods, so full-year comparisons should use 2017 or Jan-Aug."
        )
    return profile
