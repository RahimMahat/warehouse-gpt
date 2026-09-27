"""Sandboxed query execution: read-only DuckDB, no external access, timeout and row cap."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool = False
    elapsed_ms: float = 0.0
    error: str | None = None
    types: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_markdown(self, max_rows: int = 20) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        if not self.rows:
            return "(0 rows)"
        head = "| " + " | ".join(self.columns) + " |"
        sep = "|" + "---|" * len(self.columns)
        body = ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in self.rows[:max_rows]]
        more = self.row_count - max_rows
        tail = [f"... {more} more rows{' (result truncated)' if self.truncated else ''}"] if more > 0 else []
        return "\n".join([head, sep, *body, *tail])

    def to_records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, r, strict=True)) for r in self.rows]


def _fmt(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float | Decimal):
        return f"{float(v):,.4f}".rstrip("0").rstrip(".")
    if isinstance(v, datetime | date):
        return v.isoformat()
    return str(v).replace("|", "\\|")


class QueryExecutor:
    """Sandboxed DuckDB: the warehouse is attached READ_ONLY to an in-memory database, then file and
    network access is disabled and the configuration locked, so no SQL can turn it back on.

    Attaching (rather than opening the file with a custom config) lets the executor coexist with
    other read-only connections to the same file in one process. Each query runs on its own
    cursor, so the executor is thread-safe.
    """

    def __init__(self, warehouse_path: Path, timeout_s: float = 30.0, max_rows: int = 1000) -> None:
        self.timeout_s, self.max_rows = timeout_s, max_rows
        self._con = duckdb.connect(":memory:")
        path = str(warehouse_path).replace("'", "''")
        self._con.execute(f"ATTACH '{path}' AS warehouse (READ_ONLY)")
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET lock_configuration = true")

    def run(self, sql: str) -> QueryResult:
        cur = self._con.cursor()
        # Unqualified names resolve to the governed schemas, never to staging.
        cur.execute("USE warehouse")
        cur.execute("SET search_path = 'marts,meta'")
        timer = threading.Timer(self.timeout_s, cur.interrupt)
        t0 = time.perf_counter()
        timer.start()
        try:
            cur.execute(sql)
            columns = [d[0] for d in cur.description or []]
            types = [str(d[1]) for d in cur.description or []]
            rows = cur.fetchmany(self.max_rows + 1)
        except duckdb.InterruptException:
            return QueryResult(
                [], [], error=f"query timed out after {self.timeout_s:.0f}s", elapsed_ms=_ms(t0)
            )
        except duckdb.Error as e:
            return QueryResult([], [], error=_clean_error(e), elapsed_ms=_ms(t0))
        finally:
            timer.cancel()
            cur.close()
        truncated = len(rows) > self.max_rows
        return QueryResult(columns, rows[: self.max_rows], truncated, _ms(t0), types=types)

    def close(self) -> None:
        self._con.close()


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _clean_error(e: Exception) -> str:
    # DuckDB errors often end with a long "LINE 1: ..." echo of the query; keep the useful part.
    text = str(e).strip()
    return text if len(text) < 800 else text[:800] + "..."
