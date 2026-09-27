"""Central configuration. Values come from env vars (prefix ``WGPT_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WGPT_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    kaggle_dataset: str = "olistbr/brazilian-ecommerce"

    # Spark runtime. On Windows, Spark needs JDK 17/21 (Hadoop breaks on 23+)
    # and winutils.exe/hadoop.dll under HADOOP_HOME/bin.
    java_home: Path | None = None
    hadoop_home: Path | None = None
    spark_master: str = "local[*]"
    spark_driver_memory: str = "4g"
    spark_shuffle_partitions: int = Field(default=8, ge=1)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def lake_dir(self) -> Path:
        return self.data_dir / "lake"

    @property
    def bronze_dir(self) -> Path:
        return self.lake_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.lake_dir / "silver"

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / "warehouse.duckdb"


@lru_cache
def get_settings() -> Settings:
    return Settings()
