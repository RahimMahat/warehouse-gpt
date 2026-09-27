"""Run the golden set through the agent across the ablation ladder and score every answer.

Rungs (each adds one capability to the previous):

    R1  L1 raw DDL            no self-correction
    R2  L2 + dbt docs         no self-correction
    R3  L3 + semantic layer   no self-correction
    R4  L4 + examples         no self-correction
    R5  L4 + self-correction  up to 2 repair attempts

Because R5's first attempt sends exactly the same prompt as R4, the response store serves it for
free: R5 only pays for repairs.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from warehouse_gpt.agent.executor import QueryExecutor, QueryResult
from warehouse_gpt.agent.graph import Agent
from warehouse_gpt.agent.llm import CassetteMiss, resolve_model
from warehouse_gpt.context.render import ContextLevel
from warehouse_gpt.evals.compare import compare


@dataclass(frozen=True)
class Rung:
    id: int
    label: str
    level: ContextLevel
    max_repairs: int


RUNGS: dict[int, Rung] = {
    r.id: r
    for r in [
        Rung(1, "L1 raw DDL", ContextLevel.RAW_DDL, 0),
        Rung(2, "L2 + dbt docs", ContextLevel.DBT_DOCS, 0),
        Rung(3, "L3 + semantic layer", ContextLevel.SEMANTIC, 0),
        Rung(4, "L4 + examples", ContextLevel.EXAMPLES, 0),
        Rung(5, "L4 + self-correction", ContextLevel.EXAMPLES, 2),
    ]
}


@dataclass
class GoldenQuestion:
    id: str
    question: str
    difficulty: str
    tags: list[str]
    sql: str | None = None
    ordered: bool = False
    expect: str = "answer"  # or "refuse"
    alt_sql: list[str] = field(default_factory=list)  # other readings that also count as correct


def load_golden(path: Path) -> list[GoldenQuestion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [GoldenQuestion(**q) for q in data["questions"]]


def golden_fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()[:12]


@dataclass
class ItemResult:
    id: str
    rung: int
    model: str
    difficulty: str
    tags: list[str]
    expect: str
    status: str
    correct: bool
    reason: str
    sql: str | None
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    llm_latency_s: float
    context_tokens: int
    refusal: str | None = None
    error: str | None = None
    guard_blocks: int = 0  # SQL attempts rejected by the guard during this run

    @property
    def repaired(self) -> bool:
        return self.correct and self.attempts > 1


@dataclass
class RunMeta:
    model: str
    litellm_model: str
    rungs: list[int]
    golden_fingerprint: str
    n_questions: int
    started_at: str
    finished_at: str = ""
    git_sha: str = ""
    complete: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def gold_results(
    questions: Iterable[GoldenQuestion], executor: QueryExecutor
) -> dict[str, list[QueryResult]]:
    """Reference results per question: the primary SQL first, then any accepted alternatives."""
    out: dict[str, list[QueryResult]] = {}
    for q in questions:
        refs = [q.sql, *q.alt_sql] if q.sql else []
        for sql in refs:
            r = executor.run(sql)
            if not r.ok or r.row_count == 0:
                raise ValueError(f"gold SQL for {q.id} is broken: {r.error or 'no rows'}")
            out.setdefault(q.id, []).append(r)
    return out


def score(q: GoldenQuestion, response: Any, gold: list[QueryResult] | None) -> tuple[bool, str]:
    if q.expect == "refuse":
        if response.status == "refused":
            return True, "refused as expected"
        return False, f"expected a refusal, got {response.status}"
    if response.status == "refused":
        return False, f"false refusal: {response.refusal}"
    assert gold
    first = compare(gold[0], response.result, ordered=q.ordered)
    if first.correct:
        return True, first.reason
    for alt in gold[1:]:
        if compare(alt, response.result, ordered=q.ordered).correct:
            return True, "matches an accepted alternative reading"
    return False, first.reason


class ProviderError(RuntimeError):
    """Too many consecutive provider failures (daily quota, outage); the run stops early."""


def run_eval(
    agent: Agent,
    questions: list[GoldenQuestion],
    model: str,
    rungs: Iterable[int],
    gold: dict[str, list[QueryResult]],
    on_item: Callable[[ItemResult], None] | None = None,
    max_consecutive_errors: int = 3,
    skip_uncached: bool = False,
) -> list[ItemResult]:
    """Score every question at every rung. With ``skip_uncached`` (replay mode), questions without a
    stored response are skipped instead of aborting, which rebuilds results for a partial run."""
    results: list[ItemResult] = []
    consecutive = 0
    for rung_id in rungs:
        rung = RUNGS[rung_id]
        for q in questions:
            try:
                r = agent.ask(
                    q.question, level=rung.level, model=model, max_repairs=rung.max_repairs, answer=False
                )
            except CassetteMiss:
                if skip_uncached:
                    continue
                raise
            except Exception as exc:
                consecutive += 1
                if consecutive >= max_consecutive_errors:
                    raise ProviderError(f"{consecutive} consecutive failures, last: {exc!r}") from exc
                time.sleep(5)
                continue
            consecutive = 0
            correct, reason = score(q, r, gold.get(q.id))
            item = ItemResult(
                id=q.id,
                rung=rung_id,
                model=model,
                difficulty=q.difficulty,
                tags=q.tags,
                expect=q.expect,
                status=r.status,
                correct=correct,
                reason=reason,
                sql=r.sql,
                attempts=r.attempts,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                llm_latency_s=r.llm_latency_s,
                context_tokens=r.context_tokens,
                refusal=r.refusal,
                error=r.error,
                guard_blocks=sum(1 for t in r.trace if t["node"] == "validate" and not t["ok"]),
            )
            results.append(item)
            if on_item:
                on_item(item)
    return results


# -- persistence ---------------------------------------------------------------------------
def results_path(results_dir: Path, model: str) -> Path:
    return results_dir / f"{model.replace('/', '__')}.jsonl"


def save_results(results_dir: Path, meta: RunMeta, items: list[ItemResult]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_path(results_dir, meta.model)
    lines = [json.dumps({"meta": asdict(meta)})] + [json.dumps(asdict(i)) for i in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_results(path: Path) -> tuple[RunMeta, list[ItemResult]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = RunMeta(**json.loads(lines[0])["meta"])
    return meta, [ItemResult(**json.loads(line)) for line in lines[1:] if line.strip()]


def new_meta(model: str, rungs: list[int], golden_path: Path, n: int) -> RunMeta:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        sha = ""
    return RunMeta(
        model=model,
        litellm_model=resolve_model(model).litellm_model,
        rungs=rungs,
        golden_fingerprint=golden_fingerprint(golden_path),
        n_questions=n,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=sha,
    )
