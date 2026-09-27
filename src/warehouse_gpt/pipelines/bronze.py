"""Bronze layer: raw CSV -> Delta, all columns kept as strings, plus lineage metadata.

Keeping bronze untyped means a malformed value never drops a row at ingestion;
typing and cleaning happen in silver, where failures are visible as DQ results.
"""

from __future__ import annotations

import structlog
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from warehouse_gpt.config import Settings, get_settings
from warehouse_gpt.pipelines.tables import SOURCES

log = structlog.get_logger(__name__)


def read_raw_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("multiLine", True)  # review messages contain embedded newlines
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .csv(path)
    )


def build_bronze(spark: SparkSession, settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    counts: dict[str, int] = {}
    for table in SOURCES.values():
        df = read_raw_csv(spark, str(settings.raw_dir / table.raw_file))
        # Strip a UTF-8 BOM that some exports put on the first header.
        df = df.toDF(*[c.lstrip("﻿") for c in df.columns])
        df = df.withColumn("_ingested_at", F.current_timestamp()).withColumn(
            "_source_file", F.lit(table.raw_file)
        )
        target = str(settings.bronze_dir / table.name)
        df.write.format("delta").mode("overwrite").option("overwriteSchema", True).save(target)
        counts[table.name] = spark.read.format("delta").load(target).count()
        log.info("bronze.written", table=table.name, rows=counts[table.name])
    return counts
