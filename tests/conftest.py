from __future__ import annotations

from collections.abc import Iterator

import pytest

from warehouse_gpt.config import get_settings


@pytest.fixture(scope="session")
def spark() -> Iterator[object]:
    from warehouse_gpt.pipelines.spark_session import get_spark

    session = get_spark("wgpt-tests")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def warehouse():
    import duckdb

    path = get_settings().warehouse_path
    if not path.exists():
        pytest.skip("warehouse not built; run `wgpt data build`")
    con = duckdb.connect(str(path), read_only=True)
    yield con
    con.close()
