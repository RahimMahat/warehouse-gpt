"""Spark helpers that stay on the JVM (no Python worker processes)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType


def literal_df(spark: SparkSession, rows: Sequence[Sequence[Any]], schema: str) -> DataFrame:
    """Build a small DataFrame from Python literals without ``createDataFrame``.

    ``createDataFrame(list)`` ships rows through a Python worker, which is slow
    and unreliable on Windows. Here rows become literal structs that the JVM
    explodes, so no Python worker is started.
    """
    struct = StructType.fromDDL(schema)
    assert isinstance(struct, StructType)
    names = [f.name for f in struct.fields]
    if not rows:
        return spark.range(0).select(*[F.lit(None).cast(f.dataType).alias(f.name) for f in struct.fields])
    structs = [
        F.struct(*[F.lit(v).cast(f.dataType).alias(f.name) for v, f in zip(row, struct.fields, strict=True)])
        for row in rows
    ]
    return spark.range(1).select(F.inline(F.array(*structs))).toDF(*names)
