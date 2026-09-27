"""Tiny declarative data-quality framework.

Each check yields a result row (table, check, severity, passed, failing_rows).
Results are appended to the ``dq_results`` Delta table so downstream consumers,
including the agent, can surface caveats like "reviews failed a uniqueness check".
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import structlog
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from warehouse_gpt.pipelines.spark_utils import literal_df

log = structlog.get_logger(__name__)

Severity = Literal["error", "warn"]


@dataclass(frozen=True)
class Check:
    name: str
    severity: Severity
    # Returns the number of offending rows (0 == pass).
    count_failures: Callable[[DataFrame], int]


@dataclass(frozen=True)
class CheckResult:
    table: str
    check: str
    severity: Severity
    passed: bool
    failing_rows: int


def not_null(*cols: str, severity: Severity = "error") -> Check:
    cond: Column = F.lit(False)
    for c in cols:
        cond = cond | F.col(c).isNull()
    return Check(f"not_null({', '.join(cols)})", severity, lambda df: df.filter(cond).count())


def unique(*cols: str, severity: Severity = "error") -> Check:
    def _count(df: DataFrame) -> int:
        return df.groupBy(*cols).count().filter("count > 1").count()

    return Check(f"unique({', '.join(cols)})", severity, _count)


def in_range(
    col: str, lo: float | None = None, hi: float | None = None, severity: Severity = "error"
) -> Check:
    cond: Column = F.lit(False)
    if lo is not None:
        cond = cond | (F.col(col) < lo)
    if hi is not None:
        cond = cond | (F.col(col) > hi)
    return Check(f"in_range({col}, {lo}, {hi})", severity, lambda df: df.filter(cond).count())


def accepted_values(col: str, values: Sequence[str], severity: Severity = "error") -> Check:
    cond = F.col(col).isNotNull() & ~F.col(col).isin(*values)
    return Check(f"accepted_values({col})", severity, lambda df: df.filter(cond).count())


def expression(name: str, failing_condition: str, severity: Severity = "warn") -> Check:
    return Check(name, severity, lambda df: df.filter(F.expr(failing_condition)).count())


def min_rows(n: int, severity: Severity = "error") -> Check:
    return Check(f"min_rows({n})", severity, lambda df: max(0, n - df.count()))


class DataQualityError(RuntimeError):
    pass


def run_checks(table: str, df: DataFrame, checks: Iterable[Check]) -> list[CheckResult]:
    results = []
    for check in checks:
        failures = check.count_failures(df)
        result = CheckResult(table, check.name, check.severity, failures == 0, failures)
        results.append(result)
        log_fn = log.info if result.passed else (log.error if check.severity == "error" else log.warning)
        log_fn("dq.check", table=table, check=check.name, failing_rows=failures)
    return results


def persist_results(spark: SparkSession, results: list[CheckResult], path: str) -> str:
    run_id = uuid.uuid4().hex[:12]
    run_at = datetime.now(UTC)
    rows = [(run_id, run_at, r.table, r.check, r.severity, r.passed, r.failing_rows) for r in results]
    schema = (
        "run_id string, run_at timestamp, table_name string, check_name string, "
        "severity string, passed boolean, failing_rows long"
    )
    literal_df(spark, rows, schema).write.format("delta").mode("append").save(path)
    return run_id


def raise_on_errors(results: list[CheckResult]) -> None:
    failed = [r for r in results if not r.passed and r.severity == "error"]
    if failed:
        detail = "; ".join(f"{r.table}.{r.check} ({r.failing_rows} rows)" for r in failed)
        raise DataQualityError(f"{len(failed)} blocking data-quality check(s) failed: {detail}")
