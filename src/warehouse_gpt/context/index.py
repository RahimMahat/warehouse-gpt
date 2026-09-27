"""Metadata index: embeddings of tables, columns, metrics and verified examples in LanceDB.

Used for (a) retrieving few-shot examples similar to the user's question and
(b) schema pruning: selecting the relevant tables when a warehouse is too large to
fit in the prompt. Embeddings run locally (fastembed / ONNX, no GPU, no API cost).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import lancedb

DocKind = Literal["table", "column", "metric", "example"]
TABLE_NAME = "metadata"


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings (default: BAAI/bge-small-en-v1.5, 384 dims)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests and offline CI only."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@dataclass(frozen=True)
class Document:
    id: str
    kind: DocKind
    text: str  # what gets embedded
    payload: dict[str, Any]


@dataclass(frozen=True)
class Hit:
    id: str
    kind: DocKind
    text: str
    payload: dict[str, Any]
    distance: float


class MetadataIndex:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        self.path = path
        self.embedder = embedder
        self._table = lancedb.connect(str(path)).open_table(TABLE_NAME)

    @classmethod
    def build(cls, path: Path, docs: Sequence[Document], embedder: Embedder) -> MetadataIndex:
        if not docs:
            raise ValueError("no documents to index")
        vectors = embedder.embed_documents([d.text for d in docs])
        rows = [
            {
                "id": d.id,
                "kind": d.kind,
                "text": d.text,
                "payload": json.dumps(d.payload),
                "vector": v,
            }
            for d, v in zip(docs, vectors, strict=True)
        ]
        path.mkdir(parents=True, exist_ok=True)
        lancedb.connect(str(path)).create_table(TABLE_NAME, data=rows, mode="overwrite")
        return cls(path, embedder)

    def search(self, query: str, kind: DocKind | None = None, k: int = 5) -> list[Hit]:
        q = self._table.search(self.embedder.embed_query(query))
        if kind:
            q = q.where(f"kind = '{kind}'", prefilter=True)
        return [
            Hit(r["id"], r["kind"], r["text"], json.loads(r["payload"]), float(r["_distance"]))
            for r in q.limit(k).to_list()
        ]

    def count(self) -> int:
        return self._table.count_rows()
