"""Execution-accuracy comparator: does the agent's result answer the question like the gold result?

A prediction is correct when every gold column is matched by a distinct predicted column, and the
rows agree under that mapping. What counts as agreeing:

- **Extra predicted columns are ignored.** Gold SQL selects only what the question requires.
- **Column names are ignored.** Columns are matched by their values.
- **Numbers** match within 1e-4 relative tolerance, or when the prediction equals the gold value
  rounded to the prediction's precision but never coarser than 2 decimals (0.23 and 1.6 match
  0.2265 and 1.6008; 0.2 does not match 0.2265). Integer counts must match exactly. A consistent
  x100 scale is accepted for the whole column (22.65% vs 0.2265).
- **Time keys** match across representations when the projection is lossless on the gold column:
  2017-03-01 / '2017-03' / month 3 (within one year) / quarter 1 / year 2017.
- **Strings** match case-insensitively after trimming.
- **An extra NULL-key group row** (e.g. "orders with no payment type") is ignored when the
  question did not ask for it.
- **Row order** matters only for `ordered` questions (rankings). There the prediction may return
  more rows than asked if its first N rows match.

Stricter than set equality on raw tuples (no partial credit, no "close enough" on keys) but
tolerant to presentation, which is what a business user would accept.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from itertools import product
from typing import Any

from warehouse_gpt.agent.executor import QueryResult

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?(?:[ T].*)?$")
_SCALES = (1.0, 100.0)


@dataclass
class Comparison:
    correct: bool
    reason: str


# -- value views ---------------------------------------------------------------------------
def _temporal(v: Any) -> tuple[int, int, int] | None:
    if isinstance(v, datetime | date):
        return v.year, v.month, v.day
    if isinstance(v, str) and (m := _DATE_RE.match(v.strip())):
        return int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
    return None


def views(v: Any) -> dict[str, Any]:
    """All the ways a single value can be read. Keys are projection names."""
    if v is None:
        return {}
    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, int | float | Decimal):
        f = float(v)
        out: dict[str, Any] = {"num": f}
        if f.is_integer():
            i = int(f)
            if 1 <= i <= 12:
                out["month"] = i
            if 1 <= i <= 4:
                out["quarter"] = i
            if 1900 <= i <= 2100:
                out["year"] = i
        return out
    if (t := _temporal(v)) is not None:
        y, m, d = t
        return {"day": (y, m, d), "year_month": (y, m), "year": y, "month": m, "quarter": (m - 1) // 3 + 1}
    return {"str": str(v).strip().lower()}


def _decimals(x: float) -> int:
    s = repr(x)
    if "e" in s or "E" in s:
        return 12
    if "." not in s:
        return 0
    frac = s.split(".", 1)[1]
    return 0 if frac == "0" else len(frac)


def num_eq(gold: float, pred: float) -> bool:
    if gold.is_integer() and pred.is_integer():
        return gold == pred  # counts are exact
    if math.isclose(gold, pred, rel_tol=1e-4, abs_tol=1e-9):
        return True
    # Rounded predictions: equal to gold rounded to the prediction's precision, but never coarser than
    # 2 decimals. 1.6 (a printed 1.60) matches 1.6008; 0.2 does not match 0.2265.
    d = max(_decimals(pred), 2)
    return math.isclose(round(gold, d), pred, rel_tol=1e-9, abs_tol=1e-12)


@dataclass(frozen=True)
class Projection:
    name: str
    scale: float = 1.0


def _project(v: Any, p: Projection, pred: bool = False) -> Any:
    """The value under a projection, None for SQL NULL, or a sentinel if it has no such view.

    The scale is applied to the *gold* side (gold 0.2265 x 100 vs predicted 22.65), so the
    prediction keeps its own printed precision for the rounding rule in :func:`num_eq`.
    """
    if v is None:
        return None
    view = views(v).get(p.name, _MISSING)
    if not pred and p.name == "num" and view is not _MISSING and p.scale != 1.0:
        return view * p.scale
    return view


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def _eq(a: Any, b: Any, p: Projection) -> bool:
    if a is _MISSING or b is _MISSING:
        return False
    if a is None or b is None:
        return a is None and b is None
    if p.name == "num":
        return num_eq(a, b)
    return bool(a == b)


def _sort_key(v: Any) -> tuple[int, Any]:
    if v is None or v is _MISSING:
        return (0, 0)
    if isinstance(v, float):
        return (1, v)
    if isinstance(v, tuple):
        return (2, v)
    return (3, str(v))


# -- column candidates ---------------------------------------------------------------------
def _projections(gold_col: Sequence[Any], pred_col: Sequence[Any]) -> list[Projection]:
    """Projections under which the two columns could be compared (lossless on the gold column)."""
    g_names = set.intersection(*[set(views(v)) for v in gold_col if v is not None] or [set()])
    p_names = set.intersection(*[set(views(v)) for v in pred_col if v is not None] or [set()])
    distinct_gold = len({repr(v) for v in gold_col})
    out: list[Projection] = []
    for name in sorted(g_names & p_names):
        if name != "num" and len({repr(views(v).get(name)) for v in gold_col}) != distinct_gold:
            continue  # e.g. 'year' would collapse different months into one key
        out.extend(Projection(name, s) for s in _SCALES) if name == "num" else out.append(Projection(name))
    return out


def _column_matches(gold_col: Sequence[Any], pred_col: Sequence[Any], p: Projection, ordered: bool) -> bool:
    g = [_project(v, p) for v in gold_col]
    q = [_project(v, p, pred=True) for v in pred_col]
    if not ordered:
        g, q = sorted(g, key=_sort_key), sorted(q, key=_sort_key)
    return all(_eq(a, b, p) for a, b in zip(g, q, strict=True))


def _rows_match(
    gold_rows: Sequence[tuple[Any, ...]],
    pred_rows: Sequence[tuple[Any, ...]],
    mapping: Sequence[tuple[int, Projection]],
    ordered: bool,
) -> bool:
    g = [tuple(_project(r[j], p) for j, (_, p) in enumerate(mapping)) for r in gold_rows]
    q = [tuple(_project(r[k], p, pred=True) for k, p in mapping) for r in pred_rows]
    if not ordered:
        key = lambda row: tuple(_sort_key(v) for v in row)  # noqa: E731
        g, q = sorted(g, key=key), sorted(q, key=key)
    return all(
        _eq(a, b, p)
        for gr, pr in zip(g, q, strict=True)
        for a, b, (_, p) in zip(gr, pr, mapping, strict=True)
    )


# -- public --------------------------------------------------------------------------------
def compare(gold: QueryResult, pred: QueryResult | None, ordered: bool = False) -> Comparison:
    if pred is None:
        return Comparison(False, "no result")
    if not pred.ok:
        return Comparison(False, f"query error: {(pred.error or '')[:120]}")
    if not gold.ok:
        raise ValueError(f"gold query failed: {gold.error}")

    n, m = gold.row_count, pred.row_count
    if ordered:
        if m < n:
            return Comparison(False, f"expected {n} rows, got {m}")
        pred_rows = pred.rows[:n]
    else:
        pred_rows = pred.rows
        if m > n:
            # An extra "NULL group" (e.g. orders with no payment type) that the question never asked
            # about is presentation, not substance: drop it before comparing.
            pred_rows = [r for r in pred_rows if not _is_null_group(r)]
        if len(pred_rows) != n:
            return Comparison(False, f"expected {n} rows, got {m}")

    gold_cols = [[r[j] for r in gold.rows] for j in range(len(gold.columns))]
    pred_cols = [[r[k] for r in pred_rows] for k in range(len(pred.columns))]

    candidates: list[list[tuple[int, Projection]]] = []
    for j, gc in enumerate(gold_cols):
        cands = [
            (k, p)
            for k, pc in enumerate(pred_cols)
            for p in _projections(gc, pc)
            if _column_matches(gc, pc, p, ordered)
        ]
        if not cands:
            return Comparison(False, f"no column matches gold '{gold.columns[j]}' ({_preview(gc)})")
        candidates.append(cands)

    for mapping in product(*candidates):
        if len({k for k, _ in mapping}) == len(mapping) and _rows_match(
            gold.rows, pred_rows, mapping, ordered
        ):
            return Comparison(True, "match")
    return Comparison(False, "columns match individually but rows do not align")


def _is_null_group(row: tuple[Any, ...]) -> bool:
    """True when the row's grouping keys (its non-numeric cells) are all NULL."""
    keys = [v for v in row if not isinstance(v, int | float | Decimal) or isinstance(v, bool)]
    return any(v is None for v in row) and all(v is None for v in keys)


def _preview(col: Sequence[Any]) -> str:
    vals = ", ".join(str(v) for v in col[:3])
    return vals + (", ..." if len(col) > 3 else "")
