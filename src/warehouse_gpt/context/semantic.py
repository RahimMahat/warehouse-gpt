"""Compile dbt's semantic_manifest.json (MetricFlow spec) into agent-usable form.

For every metric we derive a concrete SQL recipe (aggregate expression + source
table), so the model never has to guess how "revenue" or "customers" is defined.
Entities give the join graph between semantic models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AGG_SQL = {
    "sum": "SUM({expr})",
    "count": "COUNT({expr})",
    "count_distinct": "COUNT(DISTINCT {expr})",
    "average": "AVG({expr})",
    "min": "MIN({expr})",
    "max": "MAX({expr})",
    "sum_boolean": "SUM(CASE WHEN {expr} THEN 1 ELSE 0 END)",
    "median": "MEDIAN({expr})",
}


@dataclass(frozen=True)
class Measure:
    name: str
    agg: str
    expr: str
    model: str
    description: str = ""

    @property
    def sql(self) -> str:
        template = _AGG_SQL.get(self.agg)
        if template is None:
            raise ValueError(f"unsupported aggregation {self.agg!r} on measure {self.name}")
        return template.format(expr=self.expr)


@dataclass(frozen=True)
class Entity:
    name: str
    type: str  # primary | foreign | unique | natural
    expr: str


@dataclass(frozen=True)
class Dimension:
    name: str
    type: str  # categorical | time
    expr: str
    synonyms: tuple[str, ...] = ()


@dataclass
class SemanticModel:
    name: str
    table: str  # schema.table
    description: str
    entities: list[Entity]
    dimensions: list[Dimension]
    measures: dict[str, Measure]
    agg_time_dimension: str | None = None


@dataclass
class Metric:
    name: str
    label: str
    description: str
    type: str
    synonyms: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    hidden: bool = False
    measure: str | None = None
    numerator: str | None = None
    denominator: str | None = None


@dataclass(frozen=True)
class MetricRecipe:
    """A metric resolved to SQL: SELECT <select_sql> FROM <table> [WHERE ...] [GROUP BY ...]."""

    metric: str
    select_sql: str
    table: str
    time_column: str | None


@dataclass(frozen=True)
class JoinPath:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    entity: str

    def sql(self) -> str:
        return f"{self.left_table}.{self.left_column} = {self.right_table}.{self.right_column}"


@dataclass
class SemanticLayer:
    models: dict[str, SemanticModel]
    metrics: dict[str, Metric]
    measures: dict[str, Measure] = field(init=False)

    def __post_init__(self) -> None:
        self.measures = {m.name: m for sm in self.models.values() for m in sm.measures.values()}

    @property
    def public_metrics(self) -> list[Metric]:
        return [m for m in self.metrics.values() if not m.hidden]

    def _measure_of(self, metric_name: str) -> Measure:
        metric = self.metrics[metric_name]
        if metric.type != "simple" or not metric.measure:
            raise ValueError(f"{metric_name} is not a simple metric")
        return self.measures[metric.measure]

    def recipe(self, metric_name: str) -> MetricRecipe:
        metric = self.metrics[metric_name]
        if metric.type == "simple":
            measure = self._measure_of(metric_name)
            model = self.models[measure.model]
            return MetricRecipe(metric_name, measure.sql, model.table, self._time_col(model))
        if metric.type == "ratio":
            assert metric.numerator and metric.denominator
            num, den = self._measure_of(metric.numerator), self._measure_of(metric.denominator)
            if num.model != den.model:
                raise ValueError(f"ratio {metric_name} spans models; not supported")
            model = self.models[num.model]
            select = f"CAST({num.sql} AS DOUBLE) / NULLIF({den.sql}, 0)"
            return MetricRecipe(metric_name, select, model.table, self._time_col(model))
        raise ValueError(f"unsupported metric type {metric.type!r}")

    def _time_col(self, model: SemanticModel) -> str | None:
        dim = next((d for d in model.dimensions if d.name == model.agg_time_dimension), None)
        return dim.expr if dim else None

    def join_paths(self) -> list[JoinPath]:
        """Foreign entity in one model -> primary entity of another."""
        primaries = {
            e.name: (sm.table, e.expr)
            for sm in self.models.values()
            for e in sm.entities
            if e.type == "primary"
        }
        paths = []
        for sm in self.models.values():
            for e in sm.entities:
                if e.type == "foreign" and e.name in primaries:
                    table, col = primaries[e.name]
                    if table != sm.table:
                        paths.append(JoinPath(sm.table, e.expr, table, col, e.name))
        return sorted(paths, key=lambda p: (p.left_table, p.right_table))

    def synonym_index(self) -> dict[str, str]:
        """Lower-cased synonym/label/name -> metric name, for lookup and tests."""
        index: dict[str, str] = {}
        for m in self.public_metrics:
            for term in (m.name, m.label, *m.synonyms):
                index[term.lower()] = m.name
        return index


def _meta(node: dict[str, Any]) -> dict[str, Any]:
    return (node.get("config") or {}).get("meta") or {}


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def load_semantic_layer(path: Path) -> SemanticLayer:
    raw = json.loads(path.read_text(encoding="utf-8"))
    models: dict[str, SemanticModel] = {}
    for sm in raw["semantic_models"]:
        rel = sm["node_relation"]
        table = f"{rel['schema_name']}.{rel['alias']}"
        models[sm["name"]] = SemanticModel(
            name=sm["name"],
            table=table,
            description=_clean(sm.get("description")),
            entities=[Entity(e["name"], e["type"], e.get("expr") or e["name"]) for e in sm["entities"]],
            dimensions=[
                Dimension(
                    d["name"],
                    d["type"],
                    d.get("expr") or d["name"],
                    tuple(_meta(d).get("synonyms", [])),
                )
                for d in sm["dimensions"]
            ],
            measures={
                m["name"]: Measure(
                    m["name"],
                    m["agg"],
                    _clean(m.get("expr") or m["name"]),
                    sm["name"],
                    _clean(m.get("description")),
                )
                for m in sm["measures"]
            },
            agg_time_dimension=(sm.get("defaults") or {}).get("agg_time_dimension"),
        )

    metrics: dict[str, Metric] = {}
    for m in raw["metrics"]:
        tp = m["type_params"]
        meta = _meta(m)
        metrics[m["name"]] = Metric(
            name=m["name"],
            label=m.get("label") or m["name"],
            description=_clean(m.get("description")),
            type=m["type"],
            synonyms=tuple(meta.get("synonyms", [])),
            pitfalls=tuple(meta.get("pitfalls", [])),
            hidden=bool(meta.get("hidden", False)),
            measure=(tp.get("measure") or {}).get("name"),
            numerator=(tp.get("numerator") or {}).get("name"),
            denominator=(tp.get("denominator") or {}).get("name"),
        )
    return SemanticLayer(models=models, metrics=metrics)
