"""Warehouse catalog: physical schema from DuckDB enriched with dbt docs and tests.

The physical schema (tables, columns, types) always comes from the live database so
context can never reference a column that does not exist. dbt's manifest adds the
human knowledge: descriptions, foreign keys (relationships tests) and allowed
values (accepted_values tests).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

SERVED_SCHEMAS = ("marts", "meta")
_REF_RE = re.compile(r"ref\(['\"](\w+)['\"]\)")


@dataclass
class Column:
    name: str
    data_type: str
    description: str = ""
    accepted_values: list[str] = field(default_factory=list)
    is_primary_key: bool = False


@dataclass(frozen=True)
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Table:
    schema: str
    name: str
    columns: list[Column]
    description: str = ""
    row_count: int = 0

    @property
    def fqn(self) -> str:
        return f"{self.schema}.{self.name}"

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)


@dataclass
class Catalog:
    tables: dict[str, Table]  # keyed by fqn
    foreign_keys: list[ForeignKey]

    def table(self, name: str) -> Table:
        """Look up by fqn ('marts.fct_orders') or bare name ('fct_orders')."""
        if name in self.tables:
            return self.tables[name]
        matches = [t for t in self.tables.values() if t.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def by_bare_name(self) -> dict[str, Table]:
        return {t.name: t for t in self.tables.values()}


def _physical_schema(con: duckdb.DuckDBPyConnection) -> dict[str, Table]:
    placeholders = ",".join("?" * len(SERVED_SCHEMAS))
    rows = con.execute(
        f"""
        select table_schema, table_name, column_name, data_type
        from information_schema.columns
        where table_schema in ({placeholders})
        order by table_schema, table_name, ordinal_position
        """,
        list(SERVED_SCHEMAS),
    ).fetchall()
    tables: dict[str, Table] = {}
    for schema, name, col, dtype in rows:
        t = tables.setdefault(f"{schema}.{name}", Table(schema, name, []))
        t.columns.append(Column(col, dtype))
    for t in tables.values():
        t.row_count = con.execute(f'select count(*) from "{t.schema}"."{t.name}"').fetchone()[0]  # type: ignore[index]
    return tables


def _apply_manifest(tables: dict[str, Table], manifest: dict) -> list[ForeignKey]:
    by_name = {t.name: t for t in tables.values()}
    nodes = manifest.get("nodes", {})
    for node in nodes.values():
        if node.get("resource_type") != "model" or node["name"] not in by_name:
            continue
        t = by_name[node["name"]]
        t.description = " ".join(node.get("description", "").split())
        for col_name, col_doc in node.get("columns", {}).items():
            if col := t.column(col_name):
                col.description = " ".join(col_doc.get("description", "").split())

    fks: list[ForeignKey] = []
    uniques: dict[str, set[str]] = {}
    not_nulls: dict[str, set[str]] = {}
    for node in nodes.values():
        meta = node.get("test_metadata") or {}
        kwargs = meta.get("kwargs", {})
        attached = (node.get("attached_node") or "").split(".")[-1]
        column = node.get("column_name") or kwargs.get("column_name")
        if attached not in by_name or not column:
            continue
        test = meta.get("name")
        if test == "relationships":
            m = _REF_RE.search(kwargs.get("to", ""))
            if m and m.group(1) in by_name:
                fks.append(ForeignKey(attached, column, m.group(1), kwargs["field"]))
        elif test == "accepted_values":
            if col := by_name[attached].column(column):
                col.accepted_values = [str(v) for v in kwargs.get("values", [])]
        elif test == "unique":
            uniques.setdefault(attached, set()).add(column)
        elif test == "not_null":
            not_nulls.setdefault(attached, set()).add(column)

    # A unique + not_null single column is treated as the primary key.
    for name, cols in uniques.items():
        for c in cols & not_nulls.get(name, set()):
            if col := by_name[name].column(c):
                col.is_primary_key = True
    return sorted(fks, key=lambda f: (f.table, f.column))


def load_catalog(warehouse_path: Path, manifest_path: Path | None) -> Catalog:
    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        tables = _physical_schema(con)
    finally:
        con.close()
    fks: list[ForeignKey] = []
    if manifest_path and manifest_path.exists():
        fks = _apply_manifest(tables, json.loads(manifest_path.read_text(encoding="utf-8")))
    return Catalog(tables=tables, foreign_keys=fks)
