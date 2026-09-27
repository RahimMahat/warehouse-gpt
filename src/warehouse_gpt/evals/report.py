"""Aggregate eval results into the markdown report: ablation ladder, leaderboard, breakdowns."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from warehouse_gpt.evals.runner import RUNGS, ItemResult, RunMeta, load_results


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class RungStats:
    rung: int
    n: int
    correct: int
    valid_sql: int
    refusals_ok: int
    refusals_n: int
    false_refusals: int
    repaired: int
    avg_attempts: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    p50_latency: float
    p95_latency: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.correct, self.n)


def rung_stats(items: list[ItemResult], rung: int) -> RungStats:
    rows = [i for i in items if i.rung == rung]
    ans = [i for i in rows if i.expect == "answer"]
    neg = [i for i in rows if i.expect == "refuse"]
    lat = sorted(i.llm_latency_s for i in rows) or [0.0]
    return RungStats(
        rung=rung,
        n=len(ans),
        correct=sum(i.correct for i in ans),
        valid_sql=sum(i.status == "answered" for i in ans),
        refusals_ok=sum(i.correct for i in neg),
        refusals_n=len(neg),
        false_refusals=sum(i.status == "refused" for i in ans),
        repaired=sum(i.repaired for i in ans),
        avg_attempts=statistics.fmean(i.attempts for i in rows) if rows else 0.0,
        avg_prompt_tokens=statistics.fmean(i.prompt_tokens for i in rows) if rows else 0.0,
        avg_completion_tokens=statistics.fmean(i.completion_tokens for i in rows) if rows else 0.0,
        p50_latency=lat[len(lat) // 2],
        p95_latency=lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)],
    )


def _cell(text: str, limit: int = 160) -> str:
    """One markdown table cell: single line, no pipes, bounded length."""
    flat = " ".join(text.replace("|", "/").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ladder_table(items: list[ItemResult], rungs: Iterable[int]) -> list[str]:
    out = [
        "| Rung | Execution accuracy | 95% CI | Valid SQL | Refusals correct | Repaired "
        "| Tokens in/out per q | p50 / p95 latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rungs:
        s = rung_stats(items, r)
        lo, hi = s.ci
        out.append(
            f"| R{r} {RUNGS[r].label} | **{pct(s.accuracy)}** ({s.correct}/{s.n}) | {pct(lo)}–{pct(hi)} "
            f"| {pct(s.valid_sql / s.n if s.n else 0)} | {s.refusals_ok}/{s.refusals_n} | {s.repaired} "
            f"| {s.avg_prompt_tokens:,.0f} / {s.avg_completion_tokens:,.0f} "
            f"| {s.p50_latency:.1f}s / {s.p95_latency:.1f}s |"
        )
    return out


def _chart(model: str, items: list[ItemResult], rungs: list[int]) -> list[str]:
    acc = [round(100 * rung_stats(items, r).accuracy, 1) for r in rungs]
    labels = ", ".join(f'"R{r}"' for r in rungs)
    return [
        "```mermaid",
        "xychart-beta",
        f'    title "Execution accuracy by rung ({model})"',
        f"    x-axis [{labels}]",
        '    y-axis "accuracy %" 0 --> 100',
        f"    bar [{', '.join(str(a) for a in acc)}]",
        "```",
    ]


def _by_tag(items: list[ItemResult], rungs: list[int]) -> list[str]:
    tags = sorted({t for i in items if i.expect == "answer" for t in i.tags})
    out = ["| Tag | n | " + " | ".join(f"R{r}" for r in rungs) + " |", "|---|---|" + "---|" * len(rungs)]
    for tag in tags:
        cells = []
        n = 0
        for r in rungs:
            rows = [i for i in items if i.rung == r and i.expect == "answer" and tag in i.tags]
            n = len(rows)
            cells.append(pct(sum(i.correct for i in rows) / n) if n else "–")
        out.append(f"| {tag} | {n} | " + " | ".join(cells) + " |")
    return out


def _by_difficulty(items: list[ItemResult], rungs: list[int]) -> list[str]:
    out = [
        "| Difficulty | n | " + " | ".join(f"R{r}" for r in rungs) + " |",
        "|---|---|" + "---|" * len(rungs),
    ]
    for d in ("easy", "medium", "hard"):
        cells, n = [], 0
        for r in rungs:
            rows = [i for i in items if i.rung == r and i.expect == "answer" and i.difficulty == d]
            n = len(rows)
            cells.append(pct(sum(i.correct for i in rows) / n) if n else "–")
        out.append(f"| {d} | {n} | " + " | ".join(cells) + " |")
    return out


def _transitions(items: list[ItemResult], a: int, b: int) -> tuple[int, int]:
    """Questions fixed (wrong at a, right at b) and broken (right at a, wrong at b)."""
    ra = {i.id: i.correct for i in items if i.rung == a and i.expect == "answer"}
    rb = {i.id: i.correct for i in items if i.rung == b and i.expect == "answer"}
    common = ra.keys() & rb.keys()
    return sum(not ra[q] and rb[q] for q in common), sum(ra[q] and not rb[q] for q in common)


MIN_COVERAGE = 0.9  # share of answerable questions a rung must have scored to be reported


def apply_coverage_rule(
    runs: dict[str, tuple[RunMeta, list[ItemResult]]],
) -> tuple[dict[str, tuple[RunMeta, list[ItemResult]]], list[str]]:
    """Drop (model, rung) cells that were only partly run.

    Partial runs happen when a provider quota stops a run midway; results are rebuilt from the
    response cache. A rung needs >= 90% of the answerable questions. R5 needs 100%: its missing
    items are exactly the ones whose repair attempts were never made, so a partial R5 would silently
    look like R4.
    """
    total = max((len({i.id for i in its if i.expect == "answer"}) for _, its in runs.values()), default=0)
    kept: dict[str, tuple[RunMeta, list[ItemResult]]] = {}
    notes: list[str] = []
    for model, (meta, items) in runs.items():
        keep: list[ItemResult] = []
        for r in sorted({i.rung for i in items}):
            rows = [i for i in items if i.rung == r]
            n = sum(i.expect == "answer" for i in rows)
            need = 1.0 if r == 5 else MIN_COVERAGE
            if total and n / total >= need:
                keep.extend(rows)
            else:
                notes.append(f"{model} R{r}: {n}/{total} answerable questions scored (excluded)")
        if keep:
            kept[model] = (meta, keep)
    return kept, notes


def build_report(results_dir: Path, primary: str | None = None) -> str:
    runs: dict[str, tuple[RunMeta, list[ItemResult]]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        meta, items = load_results(path)
        runs[meta.model] = (meta, items)
    runs, excluded = apply_coverage_rule(runs)
    if not runs:
        return "# Evaluation report\n\nNo results yet. Run `wgpt eval run`.\n"

    def ladder_len(model: str) -> int:
        return len({i.rung for i in runs[model][1]})

    primary = primary if primary in runs else max(runs, key=ladder_len)
    meta, items = runs[primary]
    rungs = sorted({i.rung for i in items})
    n_answer = len({i.id for i in items if i.expect == "answer"})
    n_refuse = len({i.id for i in items if i.expect == "refuse"})

    lines = [
        "# Evaluation report",
        "",
        f"Golden set: {n_answer} answerable questions + {n_refuse} that must be refused "
        f"(`evals/golden.yml`, fingerprint `{meta.golden_fingerprint}`). Execution accuracy compares "
        "the agent's result set with the reference query's (see `evals/compare.py` for the rules). "
        "Refusal questions are scored separately and excluded from accuracy.",
        "",
        *(
            [
                "> **Partial results.** Runs were stopped by the Groq free tier's undocumented daily cap "
                "(200K tokens per model per rolling 24h). Rungs with incomplete coverage are excluded, not "
                "estimated:",
                ">",
                *[f"> - {n}" for n in excluded],
                "",
            ]
            if excluded
            else []
        ),
        f"## Ablation ladder: {primary}",
        "",
        f"`{meta.litellm_model}` · git `{meta.git_sha}` · {meta.finished_at or meta.started_at}"
        + ("" if meta.complete else " · **incomplete run**"),
        "",
        *_ladder_table(items, rungs),
        "",
        *_chart(primary, items, rungs),
        "",
    ]
    steps = list(pairwise(rungs))
    if steps:
        lines += ["What each rung changed (questions fixed / broken versus the previous rung):", ""]
        for a, b in steps:
            fixed, broken = _transitions(items, a, b)
            lines.append(f"- R{a} → R{b} ({RUNGS[b].label}): +{fixed} fixed, −{broken} broken")
        lines.append("")

    lines += [
        "### By question tag",
        "",
        *_by_tag(items, rungs),
        "",
        "### By difficulty",
        "",
        *_by_difficulty(items, rungs),
        "",
    ]

    if len(runs) > 1:
        lines += [
            "## Model leaderboard",
            "",
            "| Model | Rung | Execution accuracy | 95% CI | Refusals correct "
            "| Tokens in/out per q | p50 latency |",
            "|---|---|---|---|---|---|---|",
        ]
        board = []
        for model, (m, its) in runs.items():
            for r in sorted({i.rung for i in its}):
                board.append((model, m, r, rung_stats(its, r)))
        board.sort(key=lambda t: (-t[2], -t[3].accuracy))
        for model, m, r, s in board:
            lo, hi = s.ci
            lines.append(
                f"| {model} (`{m.litellm_model}`) | R{r} | **{pct(s.accuracy)}** ({s.correct}/{s.n}) "
                f"| {pct(lo)}–{pct(hi)} | {s.refusals_ok}/{s.refusals_n} "
                f"| {s.avg_prompt_tokens:,.0f} / {s.avg_completion_tokens:,.0f} | {s.p50_latency:.1f}s |"
            )
        lines.append("")

    lines += redteam_section(results_dir / "redteam")

    top = max(rungs)
    wrong = [i for i in items if i.rung == top and not i.correct]
    lines += [f"## Remaining failures ({primary}, R{top})", ""]
    if wrong:
        lines += ["| Question | Tags | Reason |", "|---|---|---|"]
        lines += [f"| `{i.id}` | {', '.join(i.tags)} | {_cell(i.reason)} |" for i in wrong]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def redteam_section(results_dir: Path) -> list[str]:
    """Which layer stopped each attack, per model (see evals/redteam.yml)."""
    runs = [load_results(p) for p in sorted(results_dir.glob("*.jsonl"))] if results_dir.exists() else []
    if not runs:
        return []
    lines = [
        "## Red-team suite",
        "",
        "Attacks the agent must not fulfil (destructive SQL, file/secret exfiltration, staging bypass, "
        "prompt leaks, jailbreaks, injected instructions). *Refused* = the model declined; *guard* = "
        "forbidden SQL written and rejected by the sqlglot guard; the DuckDB sandbox is the last layer.",
        "",
        "| Model | Handled correctly | Refused by model | Guard rejections | Answered | Failed |",
        "|---|---|---|---|---|---|",
    ]
    for meta, items in runs:
        n = len(items)
        lines.append(
            f"| {meta.model} | **{sum(i.correct for i in items)}/{n}** "
            f"| {sum(i.status == 'refused' for i in items)} | {sum(i.guard_blocks for i in items)} "
            f"| {sum(i.status == 'answered' for i in items)} | {sum(i.status == 'failed' for i in items)} |"
        )
    lines += ["", "| Attack | " + " | ".join(m.model for m, _ in runs) + " |", "|---|" + "---|" * len(runs)]
    ids = [i.id for i in runs[0][1]]
    for qid in ids:
        cells = []
        for _, items in runs:
            it = next((i for i in items if i.id == qid), None)
            if it is None:
                cells.append("–")
                continue
            mark = "✓" if it.correct else "✗"
            guard = f", guard ×{it.guard_blocks}" if it.guard_blocks else ""
            cells.append(f"{mark} {it.status}{guard}")
        lines.append(f"| `{qid}` | " + " | ".join(cells) + " |")
    return [*lines, ""]
