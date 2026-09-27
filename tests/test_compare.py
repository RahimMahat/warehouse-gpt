"""Comparator: tolerant to presentation, strict on substance."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from warehouse_gpt.agent.executor import QueryResult
from warehouse_gpt.evals.compare import compare, num_eq


def res(columns, rows, error=None) -> QueryResult:
    return QueryResult(list(columns), [tuple(r) for r in rows], error=error)


@pytest.mark.parametrize(
    ("gold", "pred"),
    [
        (6108492.27, 6108492.27),
        (6108492.27, 6108492),  # rounded to integer: within 1e-4
        (0.2265, 0.23),  # rounded to 2 decimals
        (4.0864, 4.09),
        (22.6537, 22.65),
        (99441, 99441.0),
        (1.6008514, 1.6),  # ROUND(x, 2) printed as 1.60 arrives as the float 1.6
        (10.004, 10),
    ],
)
def test_num_eq_accepts(gold, pred):
    assert num_eq(gold, pred)


@pytest.mark.parametrize(
    ("gold", "pred"), [(0.2265, 0.2), (6108492.27, 6200000), (99441, 99440), (4.086, 4.1)]
)
def test_num_eq_rejects(gold, pred):
    assert not num_eq(gold, pred)


def test_scalar_with_extra_columns_and_decimal():
    gold = res(["revenue"], [[Decimal("6108492.27")]])
    assert compare(gold, res(["year", "total_revenue"], [[2017, 6108492.27]])).correct


def test_wrong_scalar():
    gold = res(["customers"], [[96096]])
    r = compare(gold, res(["n"], [[99441]]))
    assert not r.correct and "96096" in r.reason


def test_percent_scale_is_accepted():
    gold = res(["rate"], [[0.06774]])
    assert compare(gold, res(["late_pct"], [[6.774]])).correct
    assert compare(gold, res(["late_pct"], [[6.77]])).correct


def test_unordered_rows_with_renamed_columns():
    gold = res(["y", "rate"], [[2016, 0.1003], [2017, 0.0160], [2018, 0.0089]])
    pred = res(["cancel_rate", "year"], [[0.0089, 2018], [0.1003, 2016], [0.0160, 2017]])
    assert compare(gold, pred).correct


def test_rows_must_align_not_just_columns():
    gold = res(["state", "n"], [["SP", 10], ["RJ", 5]])
    pred = res(["state", "n"], [["SP", 5], ["RJ", 10]])
    assert not compare(gold, pred).correct


def test_time_keys_across_representations():
    gold = res(["quarter", "revenue"], [[1, 100.0], [2, 200.0], [3, 300.0], [4, 400.0]])
    pred = res(
        ["q_start", "revenue"],
        [
            [date(2017, 1, 1), 100.0],
            [date(2017, 4, 1), 200.0],
            [date(2017, 7, 1), 300.0],
            [date(2017, 10, 1), 400.0],
        ],
    )
    assert compare(gold, pred).correct
    months = res(["month", "v"], [[1, 1.0], [2, 2.0]])
    assert compare(months, res(["ym", "v"], [["2018-01", 1.0], ["2018-02", 2.0]])).correct
    assert compare(
        months, res(["m", "v"], [[datetime(2018, 1, 1), 1.0], [datetime(2018, 2, 1), 2.0]])
    ).correct


def test_lossy_time_projection_is_rejected():
    # Gold has months across two years; month numbers alone would collapse them.
    gold = res(["ym", "v"], [["2017-01", 1.0], ["2018-01", 2.0]])
    assert not compare(gold, res(["month", "v"], [[1, 1.0], [1, 2.0]])).correct


def test_strings_case_insensitive():
    gold = res(["state"], [["SP"]])
    assert compare(gold, res(["s"], [["sp "]]), ordered=True).correct


def test_ordered_ranking():
    gold = res(["state"], [["SP"], ["RJ"], ["MG"]])
    assert compare(gold, res(["state", "rev"], [["SP", 3], ["RJ", 2], ["MG", 1]]), ordered=True).correct
    assert not compare(gold, res(["state"], [["RJ"], ["SP"], ["MG"]]), ordered=True).correct
    # returning more rows than asked is fine if the top N match
    assert compare(gold, res(["state"], [["SP"], ["RJ"], ["MG"], ["RS"]]), ordered=True).correct
    assert not compare(gold, res(["state"], [["SP"], ["RJ"]]), ordered=True).correct


def test_unordered_requires_same_row_count():
    gold = res(["state"], [["SP"], ["RJ"]])
    assert not compare(gold, res(["state"], [["SP"], ["RJ"], ["MG"]])).correct


def test_rounded_percent_keeps_its_precision():
    # 100 * -0.11376 printed as -11.38: dividing by 100 would give -0.11380000000000001 and break rounding
    gold = res(["m", "pct"], [[1, None], [2, -0.11376608431046553], [3, 0.17086]])
    assert compare(
        gold, res(["m", "pct"], [["2018-01", None], ["2018-02", -11.38], ["2018-03", 17.09]])
    ).correct


def test_nulls_match_nulls_only():
    gold = res(["m", "pct"], [[1, None], [2, -0.11]])
    assert compare(gold, res(["m", "pct"], [[1, None], [2, -11.0]])).correct
    assert not compare(gold, res(["m", "pct"], [[1, 0.0], [2, -0.11]])).correct


def test_prediction_error_is_wrong():
    gold = res(["n"], [[1]])
    assert not compare(gold, res([], [], error="Binder Error")).correct
    assert not compare(gold, None).correct


def test_extra_null_group_row_is_ignored_but_real_extra_rows_are_not():
    gold = res(["method", "score"], [["boleto", 4.08], ["voucher", 4.0]])
    assert compare(gold, res(["m", "s"], [["boleto", 4.08], ["voucher", 4.0], [None, 4.3]])).correct
    assert not compare(
        gold, res(["m", "s"], [["boleto", 4.08], ["voucher", 4.0], ["debit_card", 4.1]])
    ).correct


def test_percent_rounded_to_two_decimals_with_trailing_zero():
    gold = res(["year", "rate"], [[2016, 0.10030395], [2017, 0.01600851], [2018, 0.00886856]])
    assert compare(gold, res(["year", "pct"], [[2016, 10.03], [2017, 1.6], [2018, 0.89]])).correct
