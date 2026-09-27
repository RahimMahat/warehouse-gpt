"""Unit tests for Spark transforms and the DQ framework (need a local JVM)."""

from __future__ import annotations

import pytest

from warehouse_gpt.pipelines import dq
from warehouse_gpt.pipelines.silver import _order_reviews, normalize_text, zip_prefix
from warehouse_gpt.pipelines.spark_utils import literal_df

pytestmark = pytest.mark.spark


def test_literal_df_roundtrip(spark):
    df = literal_df(spark, [("a", 1), ("b", None)], "k string, v int")
    assert df.columns == ["k", "v"]
    assert sorted(df.collect(), key=lambda r: r.k) == [("a", 1), ("b", None)]


def test_literal_df_empty(spark):
    df = literal_df(spark, [], "k string")
    assert df.count() == 0 and df.columns == ["k"]


def test_normalize_text_and_zip(spark):
    df = literal_df(spark, [("  São   Paulo ", "1037")], "city string, zip string")
    row = df.select(normalize_text("city").alias("c"), zip_prefix("zip").alias("z")).first()
    assert row.c == "sao paulo"
    assert row.z == "01037"


def test_reviews_keep_latest_per_order(spark):
    raw = literal_df(
        spark,
        [
            ("r1", "o1", "3", "", "old", "2018-01-01 00:00:00", "2018-01-02 00:00:00"),
            ("r2", "o1", "5", "t", "new", "2018-01-03 00:00:00", "2018-01-04 00:00:00"),
            ("r3", "o2", "1", None, None, "2018-01-01 00:00:00", None),
        ],
        "review_id string, order_id string, review_score string, review_comment_title string, "
        "review_comment_message string, review_creation_date string, review_answer_timestamp string",
    )
    out = {r.order_id: r for r in _order_reviews(lambda _: raw).collect()}
    assert set(out) == {"o1", "o2"}
    assert out["o1"].review_id == "r2" and out["o1"].review_score == 5
    assert out["o2"].review_title is None


def test_dq_checks_detect_failures(spark):
    df = literal_df(spark, [(1, "a"), (1, None), (3, "zz")], "id int, s string")
    results = {
        r.check: r
        for r in dq.run_checks(
            "t",
            df,
            [
                dq.unique("id"),
                dq.not_null("s"),
                dq.accepted_values("s", ["a"]),
                dq.in_range("id", lo=0, hi=2),
                dq.min_rows(2),
            ],
        )
    }
    assert results["unique(id)"].failing_rows == 1
    assert results["not_null(s)"].failing_rows == 1
    assert results["accepted_values(s)"].failing_rows == 1
    assert results["in_range(id, 0, 2)"].failing_rows == 1
    assert results["min_rows(2)"].passed


def test_raise_on_errors_ignores_warnings():
    warn = dq.CheckResult("t", "c", "warn", False, 5)
    dq.raise_on_errors([warn])
    with pytest.raises(dq.DataQualityError):
        dq.raise_on_errors([warn, dq.CheckResult("t", "c2", "error", False, 1)])
