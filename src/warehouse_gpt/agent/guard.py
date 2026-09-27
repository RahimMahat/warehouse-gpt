"""Static SQL guard: only a single read-only query over allowlisted tables reaches the database.

This is the first of two layers. The executor also opens DuckDB read-only with external access
disabled, so a guard bug still can't write data or read files. The guard gives the model precise,
fixable feedback ("table staging.stg_orders is not allowed") instead of a raw engine error.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

DIALECT = "duckdb"

# Statement / clause types that must never appear anywhere in the tree.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    t
    for name in (
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "TruncateTable",
        "Command",
        "Copy",
        "Set",
        "Use",
        "Pragma",
        "Transaction",
        "Commit",
        "Rollback",
        "LoadData",
        "Attach",
        "Detach",
        "Install",
        "Grant",
        "Revoke",
        "Describe",
        "Into",
        "Export",
    )
    if isinstance(t := getattr(exp, name, None), type)
)

# Table functions and scalar functions that touch files, the network or engine internals.
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "parquet_scan",
        "read_json",
        "read_json_auto",
        "read_json_objects",
        "read_ndjson",
        "read_ndjson_auto",
        "read_text",
        "read_blob",
        "read_xlsx",
        "delta_scan",
        "iceberg_scan",
        "glob",
        "sniff_csv",
        "query",
        "query_table",
        "getenv",
        "sqlite_scan",
        "postgres_scan",
        "mysql_scan",
        "duckdb_secrets",
        "duckdb_settings",
        "load_extension",
        "install_extension",
        "system",
    }
)

_QUERY_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect)


@dataclass
class GuardResult:
    ok: bool
    sql: str
    tables: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SQLGuard:
    def __init__(self, allowed_tables: Iterable[str]) -> None:
        """``allowed_tables`` are fully qualified names such as ``marts.fct_orders``."""
        self.allowed = {t.lower() for t in allowed_tables}
        self._by_bare = {t.split(".", 1)[1]: t for t in self.allowed}

    def check(self, sql: str) -> GuardResult:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            return GuardResult(False, sql, errors=["empty query"])
        try:
            statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
        except ParseError as e:
            return GuardResult(False, sql, errors=[f"SQL does not parse: {_first_line(e)}"])
        if len(statements) != 1:
            return GuardResult(False, sql, errors=[f"exactly one statement allowed, got {len(statements)}"])

        tree = statements[0]
        errors: list[str] = []
        if not isinstance(tree, _QUERY_ROOTS):
            errors.append(f"only SELECT queries are allowed, got {tree.key.upper()}")

        for node in tree.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                errors.append(f"{node.key.upper()} is not allowed")
            elif isinstance(node, exp.Func):
                name = _func_name(node)
                if name in FORBIDDEN_FUNCTIONS:
                    errors.append(f"function {name}() is not allowed")

        tables = self._check_tables(tree, errors)
        return GuardResult(not errors, sql, sorted(tables), list(dict.fromkeys(errors)))

    def _check_tables(self, tree: exp.Expr, errors: list[str]) -> set[str]:
        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        used: set[str] = set()
        for table in tree.find_all(exp.Table):
            if not table.name:  # table-valued function, e.g. FROM range(10) / unnest(...)
                continue
            if table.args.get("catalog"):
                errors.append(f"cross-database reference {table.sql(DIALECT)} is not allowed")
                continue
            name, schema = table.name.lower(), (table.db or "").lower()
            if not schema and name in cte_names:
                continue
            fqn = f"{schema}.{name}" if schema else self._by_bare.get(name, name)
            if fqn in self.allowed:
                used.add(fqn)
            else:
                errors.append(
                    f"table {table.sql(DIALECT)} is not allowed; available: {', '.join(sorted(self.allowed))}"
                )
        return used


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def _first_line(e: Exception) -> str:
    return str(e).splitlines()[0][:300]
