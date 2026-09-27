"""Central configuration. Values come from env vars (prefix ``WGPT_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
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

    # Context / retrieval
    dbt_dir: Path = PROJECT_ROOT / "warehouse" / "dbt"
    verified_queries_path: Path = PROJECT_ROOT / "warehouse" / "semantic" / "verified_queries.yml"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # LLM providers. Keys use the providers' conventional names (no WGPT_ prefix).
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "WGPT_GEMINI_API_KEY")
    )
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY", "WGPT_GROQ_API_KEY")
    )
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "gpt-oss-120b"
    # live: always call the API. cache: reuse stored responses, store new ones (dev default).
    # replay: stored responses only, a miss is an error (tests/CI, no network, no keys).
    llm_mode: Literal["live", "cache", "replay"] = "cache"
    llm_store_dir: Path | None = None  # defaults to data/llm_cache

    # Agent / execution
    max_repairs: int = Field(default=2, ge=0)
    query_timeout_s: float = 30.0
    max_result_rows: int = 1000

    @property
    def manifest_path(self) -> Path:
        return self.dbt_dir / "target" / "manifest.json"

    @property
    def semantic_manifest_path(self) -> Path:
        return self.dbt_dir / "target" / "semantic_manifest.json"

    @property
    def context_dir(self) -> Path:
        return self.data_dir / "context"

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

    @property
    def llm_store_path(self) -> Path:
        return self.llm_store_dir or self.data_dir / "llm_cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
