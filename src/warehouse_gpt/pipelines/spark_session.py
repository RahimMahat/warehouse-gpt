"""Spark session factory with Delta Lake and Windows-friendly JVM setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from warehouse_gpt.config import Settings, get_settings


def _configure_jvm_env(settings: Settings) -> None:
    """Point PySpark at a compatible JDK and, on Windows, the Hadoop native libs."""
    if settings.java_home:
        os.environ["JAVA_HOME"] = str(settings.java_home)
    if settings.hadoop_home:
        hadoop_home = Path(settings.hadoop_home)
        os.environ["HADOOP_HOME"] = str(hadoop_home)
        bin_dir = str(hadoop_home / "bin")
        if sys.platform == "win32" and bin_dir not in os.environ.get("PATH", ""):
            # hadoop.dll must be loadable by the JVM for local file writes.
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    # Make sure executors use the same interpreter as the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def get_spark(app_name: str = "warehouse-gpt", settings: Settings | None = None) -> SparkSession:
    from delta import configure_spark_with_delta_pip

    settings = settings or get_settings()
    _configure_jvm_env(settings)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(settings.spark_master)
        .config("spark.driver.memory", settings.spark_driver_memory)
        .config("spark.sql.shuffle.partitions", settings.spark_shuffle_partitions)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "false")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
