"""Verified question -> SQL examples used for few-shot retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class VerifiedQuery:
    id: str
    question: str
    sql: str
    tags: tuple[str, ...] = ()


def load_verified_queries(path: Path) -> list[VerifiedQuery]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = [
        VerifiedQuery(q["id"], q["question"].strip(), q["sql"].strip(), tuple(q.get("tags", [])))
        for q in raw["queries"]
    ]
    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate verified query ids")
    return queries
