"""Pick a sensible chart for a query result and describe it as a Vega-Lite spec.

Deterministic heuristics, no LLM call: a time key with a measure is a line, a category with a
measure is a sorted bar, a single row of numbers is a KPI (no chart). The API returns the spec and
any Vega-Lite renderer (Streamlit, Altair, a browser) can draw it.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from warehouse_gpt.agent.executor import QueryResult

_YM = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")
_ORDINAL_NAMES = re.compile(r"(^|_)(month|quarter|year|hour|week|day|weekday|dow)(_|$)", re.I)
MAX_CATEGORIES = 30


def jsonable(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, datetime | date):
        return v.isoformat()
    return v


def _kind(name: str, values: list[Any]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "empty"
    if all(isinstance(v, datetime | date) for v in vals) or all(
        isinstance(v, str) and _YM.match(v) for v in vals
    ):
        return "temporal"
    if all(isinstance(v, bool) for v in vals):
        return "nominal"
    if all(isinstance(v, int | float | Decimal) for v in vals):
        if _ORDINAL_NAMES.search(name) and all(float(v).is_integer() for v in vals):
            return "ordinal"
        return "quantitative"
    return "nominal"


def chart_spec(result: QueryResult | None, title: str | None = None) -> dict[str, Any] | None:
    if result is None or not result.ok or result.row_count < 2 or len(result.columns) < 2:
        return None
    cols = result.columns
    kinds = {c: _kind(c, [r[i] for r in result.rows]) for i, c in enumerate(cols)}
    measures = [c for c in cols if kinds[c] == "quantitative"]
    keys = [c for c in cols if kinds[c] in ("temporal", "ordinal", "nominal")]
    if not measures or not keys:
        return None
    x, y = keys[0], measures[0]
    records = [{c: jsonable(v) for c, v in zip(cols, r, strict=True)} for r in result.rows]

    base: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": records},
        "width": "container",
    }
    if title:
        base["title"] = title
    tooltip = [{"field": c, "type": kinds[c] if kinds[c] != "empty" else "nominal"} for c in cols]

    if kinds[x] == "temporal":
        return base | {
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {"field": x, "type": "temporal"},
                "y": {"field": y, "type": "quantitative"},
                "tooltip": tooltip,
            },
        }
    if kinds[x] == "ordinal":
        return base | {
            "mark": "bar",
            "encoding": {
                "x": {"field": x, "type": "ordinal"},
                "y": {"field": y, "type": "quantitative"},
                "tooltip": tooltip,
            },
        }
    if result.row_count > MAX_CATEGORIES:
        return None
    return base | {
        "mark": "bar",
        "encoding": {
            "y": {"field": x, "type": "nominal", "sort": "-x"},
            "x": {"field": y, "type": "quantitative"},
            "tooltip": tooltip,
        },
    }
