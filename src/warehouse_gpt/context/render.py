"""Render warehouse context for the LLM at increasing levels of richness.

The levels are the rungs of the ablation ladder the evals measure:

    L1 RAW_DDL   CREATE TABLE statements only (what most text-to-SQL demos give the model)
    L2 DBT_DOCS  + table/column descriptions, primary/foreign keys, accepted values
    L3 SEMANTIC  + governed metric recipes, join paths, value hints, synonyms, DQ caveats
    L4 EXAMPLES  + verified question->SQL examples retrieved by similarity

(Rung 5, self-correction, is agent behavior rather than context.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from warehouse_gpt.context.catalog import Catalog, ForeignKey, Table
from warehouse_gpt.context.examples import VerifiedQuery
from warehouse_gpt.context.index import MetadataIndex
from warehouse_gpt.context.profile import DataProfile
from warehouse_gpt.context.semantic import SemanticLayer

SCHEMA_PRUNING_THRESHOLD = 25  # tables; below this the full schema is cheaper than a retrieval miss


class ContextLevel(IntEnum):
    RAW_DDL = 1
    DBT_DOCS = 2
    SEMANTIC = 3
    EXAMPLES = 4


@dataclass
class RenderedContext:
    level: ContextLevel
    text: str
    tables: list[str]
    example_ids: list[str] = field(default_factory=list)

    @property
    def approx_tokens(self) -> int:
        return len(self.text) // 4


def _fk_map(fks: list[ForeignKey], catalog: Catalog) -> dict[tuple[str, str], str]:
    names = catalog.by_bare_name()
    return {
        (fk.table, fk.column): f"{names[fk.ref_table].fqn}({fk.ref_column})"
        for fk in fks
        if fk.ref_table in names
    }


def _ddl(table: Table, documented: bool, fks: dict[tuple[str, str], str]) -> str:
    lines = []
    if documented and table.description:
        lines.append(f"-- {table.description} ({table.row_count:,} rows)")
    lines.append(f"CREATE TABLE {table.fqn} (")
    cols = []
    for c in table.columns:
        col = f"  {c.name} {c.data_type}"
        if documented:
            if c.is_primary_key:
                col += " PRIMARY KEY"
            if ref := fks.get((table.name, c.name)):
                col += f" REFERENCES {ref}"
        comment = ""
        if documented and c.description:
            comment = c.description
            if c.accepted_values:
                comment += f" Allowed values: {', '.join(c.accepted_values)}."
        cols.append((col, comment))
    for i, (col, comment) in enumerate(cols):
        sep = "," if i < len(cols) - 1 else ""
        lines.append(f"{col}{sep}" + (f"  -- {comment}" if comment else ""))
    lines.append(");")
    return "\n".join(lines)


class ContextRenderer:
    def __init__(
        self,
        catalog: Catalog,
        semantic: SemanticLayer | None = None,
        profile: DataProfile | None = None,
        examples: list[VerifiedQuery] | None = None,
        index: MetadataIndex | None = None,
    ) -> None:
        self.catalog = catalog
        self.semantic = semantic
        self.profile = profile
        self.examples = {e.id: e for e in examples or []}
        self.index = index

    # -- table selection -------------------------------------------------------------
    def select_tables(self, question: str) -> list[Table]:
        tables = sorted(self.catalog.tables.values(), key=lambda t: t.fqn)
        if len(tables) <= SCHEMA_PRUNING_THRESHOLD or self.index is None:
            return tables
        hits = self.index.search(question, kind="table", k=10)
        chosen = {h.payload["fqn"] for h in hits}
        # Expand one hop along foreign keys so join targets are never missing.
        names = self.catalog.by_bare_name()
        for fk in self.catalog.foreign_keys:
            if names[fk.table].fqn in chosen and fk.ref_table in names:
                chosen.add(names[fk.ref_table].fqn)
        return [t for t in tables if t.fqn in chosen]

    # -- sections --------------------------------------------------------------------
    def _schema_section(self, tables: list[Table], documented: bool) -> str:
        fks = _fk_map(self.catalog.foreign_keys, self.catalog) if documented else {}
        return "\n\n".join(_ddl(t, documented, fks) for t in tables)

    def _metrics_section(self) -> str:
        assert self.semantic
        out = ["## Governed business metrics (use these exact definitions)"]
        for m in sorted(self.semantic.public_metrics, key=lambda m: m.name):
            r = self.semantic.recipe(m.name)
            out.append(f"- {m.name}: {m.label}. {m.description}")
            if m.synonyms:
                out.append(f"  Also called: {', '.join(m.synonyms)}")
            time_hint = f"  (time column: {r.time_column})" if r.time_column else ""
            out.append(f"  SQL: SELECT {r.select_sql} FROM {r.table}{time_hint}")
            out.extend(f"  Pitfall: {p}" for p in m.pitfalls)
        return "\n".join(out)

    def _joins_section(self) -> str:
        assert self.semantic
        out = ["## Join paths"]
        out += [f"- {p.sql()}  (entity: {p.entity})" for p in self.semantic.join_paths()]
        out.append("- marts.fct_orders.purchase_date = marts.dim_date.date_day  (calendar)")
        return "\n".join(out)

    def _dimension_synonyms_section(self) -> str:
        assert self.semantic
        lines = [
            f"- {d.name} ({sm.table}.{d.expr}): {', '.join(d.synonyms)}"
            for sm in self.semantic.models.values()
            for d in sm.dimensions
            if d.synonyms
        ]
        return "## Dimension synonyms\n" + "\n".join(lines) if lines else ""

    def _values_section(self, tables: list[Table]) -> str:
        assert self.profile
        fqns = {t.fqn for t in tables}
        # Group columns that share the same value set (e.g. state codes appear in 5 tables)
        # so each list is printed once.
        groups: dict[tuple[str, ...], list[str]] = {}
        for key, values in sorted(self.profile.value_hints.items()):
            if key.rsplit(".", 1)[0] in fqns:
                groups.setdefault(tuple(values), []).append(key.removeprefix("marts."))
        lines = [f"- {', '.join(cols)}: {', '.join(values)}" for values, cols in groups.items()]
        return "## Exact column values (filter with these spellings)\n" + "\n".join(lines)

    def _caveats_section(self) -> str:
        assert self.profile
        if not self.profile.caveats:
            return ""
        return "## Known data caveats (mention when relevant)\n" + "\n".join(
            f"- {c}" for c in self.profile.caveats
        )

    def _examples_section(self, question: str, k: int) -> tuple[str, list[str]]:
        if self.index is None:
            return "", []
        hits = self.index.search(question, kind="example", k=k)
        chosen = [self.examples[h.id] for h in hits if h.id in self.examples]
        if not chosen:
            return "", []
        blocks = ["## Similar verified queries (patterns to follow, not answers)"]
        for e in chosen:
            blocks.append(f"Question: {e.question}\n```sql\n{e.sql}\n```")
        return "\n\n".join(blocks), [e.id for e in chosen]

    # -- public ----------------------------------------------------------------------
    def render(self, question: str, level: ContextLevel, k_examples: int = 3) -> RenderedContext:
        tables = self.select_tables(question)
        documented = level >= ContextLevel.DBT_DOCS
        sections = ["## Database schema (DuckDB SQL dialect)", self._schema_section(tables, documented)]
        example_ids: list[str] = []

        if level >= ContextLevel.SEMANTIC:
            if not (self.semantic and self.profile):
                raise ValueError("SEMANTIC level requires a semantic layer and data profile")
            sections += [
                self._metrics_section(),
                self._joins_section(),
                self._dimension_synonyms_section(),
                self._values_section(tables),
                self._caveats_section(),
            ]
        if level >= ContextLevel.EXAMPLES:
            text, example_ids = self._examples_section(question, k_examples)
            sections.append(text)

        body = "\n\n".join(s for s in sections if s)
        return RenderedContext(level, body, [t.fqn for t in tables], example_ids)
