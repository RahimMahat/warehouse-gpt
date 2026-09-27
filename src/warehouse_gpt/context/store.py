"""Build and load all context artifacts (profile + vector index) in one place."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from warehouse_gpt.config import Settings, get_settings
from warehouse_gpt.context.catalog import Catalog, load_catalog
from warehouse_gpt.context.examples import VerifiedQuery, load_verified_queries
from warehouse_gpt.context.index import Document, Embedder, FastEmbedEmbedder, MetadataIndex
from warehouse_gpt.context.profile import DataProfile, build_profile
from warehouse_gpt.context.render import ContextRenderer
from warehouse_gpt.context.semantic import SemanticLayer, load_semantic_layer

log = structlog.get_logger(__name__)


def build_documents(
    catalog: Catalog,
    semantic: SemanticLayer,
    profile: DataProfile,
    examples: list[VerifiedQuery],
) -> list[Document]:
    docs: list[Document] = []
    for t in catalog.tables.values():
        cols = ", ".join(c.name for c in t.columns)
        docs.append(
            Document(f"table:{t.fqn}", "table", f"{t.fqn}: {t.description} Columns: {cols}", {"fqn": t.fqn})
        )
        for c in t.columns:
            hint = profile.value_hints.get(f"{t.fqn}.{c.name}", [])[:15]
            text = f"{t.fqn}.{c.name} ({c.data_type}): {c.description}"
            if hint:
                text += f" Values: {', '.join(hint)}"
            docs.append(Document(f"column:{t.fqn}.{c.name}", "column", text, {"fqn": t.fqn}))
    for m in semantic.public_metrics:
        text = f"Metric {m.name} ({m.label}): {m.description} Synonyms: {', '.join(m.synonyms)}"
        docs.append(Document(f"metric:{m.name}", "metric", text, {"metric": m.name}))
    for e in examples:
        # Embed only the question: retrieval matches new questions to old questions.
        docs.append(Document(e.id, "example", e.question, {"sql": e.sql}))
    return docs


@dataclass
class ContextStore:
    catalog: Catalog
    semantic: SemanticLayer
    profile: DataProfile
    examples: list[VerifiedQuery]
    index: MetadataIndex | None

    @property
    def renderer(self) -> ContextRenderer:
        return ContextRenderer(self.catalog, self.semantic, self.profile, self.examples, self.index)

    @classmethod
    def build(cls, settings: Settings | None = None, embedder: Embedder | None = None) -> ContextStore:
        settings = settings or get_settings()
        catalog = load_catalog(settings.warehouse_path, settings.manifest_path)
        semantic = load_semantic_layer(settings.semantic_manifest_path)
        examples = load_verified_queries(settings.verified_queries_path)
        profile = build_profile(settings.warehouse_path, catalog)
        profile.save(settings.context_dir / "profile.json")

        embedder = embedder or FastEmbedEmbedder(settings.embedding_model)
        docs = build_documents(catalog, semantic, profile, examples)
        index = MetadataIndex.build(settings.context_dir / "index", docs, embedder)
        log.info("context.built", documents=len(docs), tables=len(catalog.tables))
        return cls(catalog, semantic, profile, examples, index)

    @classmethod
    def load(cls, settings: Settings | None = None, embedder: Embedder | None = None) -> ContextStore:
        settings = settings or get_settings()
        profile_path = settings.context_dir / "profile.json"
        if not profile_path.exists():
            raise FileNotFoundError("context not built; run `wgpt context build`")
        index = None
        if (settings.context_dir / "index").exists():
            index = MetadataIndex(
                settings.context_dir / "index", embedder or FastEmbedEmbedder(settings.embedding_model)
            )
        return cls(
            catalog=load_catalog(settings.warehouse_path, settings.manifest_path),
            semantic=load_semantic_layer(settings.semantic_manifest_path),
            profile=DataProfile.load(profile_path),
            examples=load_verified_queries(settings.verified_queries_path),
            index=index,
        )
