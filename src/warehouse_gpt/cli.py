"""``wgpt`` command-line entry point."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import typer
from rich.console import Console
from rich.table import Table

from warehouse_gpt.config import PROJECT_ROOT, get_settings

app = typer.Typer(no_args_is_help=True, help="WarehouseGPT: analytics agent over a lakehouse.")
data_app = typer.Typer(no_args_is_help=True, help="Build the lakehouse: raw -> bronze -> silver -> dbt.")
app.add_typer(data_app, name="data")
console = Console()

DBT_DIR = PROJECT_ROOT / "warehouse" / "dbt"


def _print_counts(title: str, counts: dict[str, int]) -> None:
    table = Table(title=title)
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name, n in counts.items():
        table.add_row(name, f"{n:,}")
    console.print(table)


@data_app.command("download")
def download(force: bool = False) -> None:
    """Download raw Olist CSVs."""
    from warehouse_gpt.pipelines.ingest import download_raw

    path = download_raw(force=force)
    console.print(f"[green]raw data at[/] {path}")


@data_app.command("spark")
def spark_layers(fail_on_dq_error: bool = True) -> None:
    """Run the Spark bronze and silver jobs (Delta Lake)."""
    from warehouse_gpt.pipelines.bronze import build_bronze
    from warehouse_gpt.pipelines.silver import build_silver
    from warehouse_gpt.pipelines.spark_session import get_spark

    spark = get_spark("wgpt-lakehouse")
    try:
        t0 = time.perf_counter()
        _print_counts("bronze", build_bronze(spark))
        _print_counts("silver", build_silver(spark, fail_on_error=fail_on_dq_error))
        console.print(f"[green]spark layers built in {time.perf_counter() - t0:.0f}s[/]")
    finally:
        spark.stop()


def _dbt(*args: str) -> None:
    settings = get_settings()
    env = {
        **os.environ,
        "WGPT_SILVER_DIR": settings.silver_dir.as_posix(),
        "WGPT_WAREHOUSE_PATH": settings.warehouse_path.as_posix(),
    }
    dbt = [sys.executable, "-m", "dbt.cli.main"]
    cmd = [*dbt, *args, "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@data_app.command("dbt")
def dbt_build(docs: bool = typer.Option(True, help="Also generate dbt docs/catalog.")) -> None:
    """Build and test the dbt gold layer into DuckDB."""
    _dbt("build")
    if docs:
        _dbt("docs", "generate")


@data_app.command("build")
def build_all() -> None:
    """End-to-end: download -> spark (bronze/silver) -> dbt (gold)."""
    download()
    spark_layers()
    dbt_build()


if __name__ == "__main__":
    app()
