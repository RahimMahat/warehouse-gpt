"""``wgpt`` command-line entry point."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import typer
from rich.console import Console
from rich.table import Table

from warehouse_gpt.config import get_settings

app = typer.Typer(no_args_is_help=True, help="WarehouseGPT: analytics agent over a lakehouse.")
data_app = typer.Typer(no_args_is_help=True, help="Build the lakehouse: raw -> bronze -> silver -> dbt.")
context_app = typer.Typer(no_args_is_help=True, help="Semantic layer, profiling and metadata index.")
app.add_typer(data_app, name="data")
app.add_typer(context_app, name="context")
console = Console()


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
    cmd = [*dbt, *args, "--project-dir", str(settings.dbt_dir), "--profiles-dir", str(settings.dbt_dir)]
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
    context_build()


@context_app.command("build")
def context_build() -> None:
    """Profile the warehouse and (re)build the metadata vector index."""
    from warehouse_gpt.context.store import ContextStore

    store = ContextStore.build()
    n = store.index.count() if store.index else 0
    console.print(
        f"[green]context built:[/] {len(store.catalog.tables)} tables, "
        f"{len(store.semantic.public_metrics)} metrics, {len(store.examples)} verified queries, "
        f"{len(store.profile.value_hints)} value hints, {n} indexed documents"
    )


@context_app.command("show")
def context_show(
    question: str,
    level: int = typer.Option(4, min=1, max=4, help="1=raw DDL, 2=+dbt docs, 3=+semantic, 4=+examples"),
) -> None:
    """Print the exact context the agent would see for QUESTION at a given level."""
    from warehouse_gpt.context.render import ContextLevel
    from warehouse_gpt.context.store import ContextStore

    rendered = ContextStore.load().renderer.render(question, ContextLevel(level))
    console.print(rendered.text, markup=False, highlight=False)
    console.print(
        f"\n[dim]level={rendered.level.name} tables={len(rendered.tables)} "
        f"examples={rendered.example_ids} ~{rendered.approx_tokens:,} tokens[/]"
    )


if __name__ == "__main__":
    app()
