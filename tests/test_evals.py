"""Golden set integrity, scoring and reporting (no network)."""

from __future__ import annotations

import re

import pytest

from tests.test_agent import ScriptedLLM, sql
from warehouse_gpt.agent.executor import QueryExecutor
from warehouse_gpt.agent.graph import Agent
from warehouse_gpt.agent.guard import SQLGuard
from warehouse_gpt.config import get_settings
from warehouse_gpt.context.catalog import load_catalog
from warehouse_gpt.context.examples import load_verified_queries
from warehouse_gpt.context.render import ContextRenderer
from warehouse_gpt.evals import runner
from warehouse_gpt.evals.report import build_report, wilson

settings = get_settings()
GOLDEN = runner.load_golden(settings.golden_path)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {
        "the",
        "a",
        "of",
        "in",
        "what",
        "how",
        "is",
        "by",
        "our",
        "we",
    }


# -- golden set integrity -------------------------------------------------------------------
def test_golden_ids_unique_and_well_formed():
    ids = [q.id for q in GOLDEN]
    assert len(ids) == len(set(ids))
    for q in GOLDEN:
        assert q.difficulty in {"easy", "medium", "hard"}, q.id
        assert q.expect in {"answer", "refuse"}, q.id
        assert (q.sql is None) == (q.expect == "refuse"), q.id


def test_golden_is_disjoint_from_verified_examples():
    """The agent retrieves verified queries at L4; eval questions must not be near-copies of them."""
    verified = load_verified_queries(settings.verified_queries_path)
    for g in GOLDEN:
        for v in verified:
            gt, vt = _tokens(g.question), _tokens(v.question)
            jaccard = len(gt & vt) / len(gt | vt)
            assert jaccard < 0.6, f"{g.id} is too close to {v.id} ({jaccard:.2f})"
            assert g.sql is None or g.sql.strip() != v.sql.strip()


def test_golden_sql_passes_the_guard_and_returns_rows(warehouse):
    guard = SQLGuard(load_catalog(settings.warehouse_path, settings.manifest_path).tables)
    ex = QueryExecutor(settings.warehouse_path)
    try:
        for q in GOLDEN:
            if q.sql:
                assert guard.check(q.sql).ok, (q.id, guard.check(q.sql).errors)
        gold = runner.gold_results(GOLDEN, ex)
        assert len(gold) == sum(q.expect == "answer" for q in GOLDEN)
    finally:
        ex.close()


# -- scoring --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def eval_parts(warehouse):
    catalog = load_catalog(settings.warehouse_path, settings.manifest_path)
    ex = QueryExecutor(settings.warehouse_path)
    yield ContextRenderer(catalog), ex, SQLGuard(catalog.tables)
    ex.close()


def test_run_eval_scores_answers_refusals_and_false_refusals(eval_parts):
    renderer, ex, guard = eval_parts
    qs = [
        q
        for q in GOLDEN
        if q.id in {"g_total_orders", "g_unique_customers", "n_weather", "g_canceled_orders"}
    ]
    qs.sort(key=lambda q: q.id)  # g_canceled_orders, g_total_orders, g_unique_customers, n_weather
    llm = ScriptedLLM(
        "CANNOT_ANSWER: not sure",  # g_canceled_orders -> false refusal
        sql("select count(*) from marts.fct_orders"),  # g_total_orders -> correct
        sql("select count(distinct customer_id) from marts.fct_orders"),  # the classic pitfall -> wrong
        "CANNOT_ANSWER: no weather data",  # n_weather -> correct refusal
    )
    agent = Agent(llm, renderer, ex, guard)  # type: ignore[arg-type]
    items = runner.run_eval(agent, qs, "scripted", [1], runner.gold_results(qs, ex))
    by_id = {i.id: i for i in items}
    assert by_id["g_total_orders"].correct
    assert not by_id["g_unique_customers"].correct and "96096" in by_id["g_unique_customers"].reason
    assert not by_id["g_canceled_orders"].correct and "false refusal" in by_id["g_canceled_orders"].reason
    assert by_id["n_weather"].correct


def test_report_from_saved_results(tmp_path):
    meta = runner.RunMeta("m", "groq/m", [1, 3], "abc", 2, "2026-01-01T00:00:00", complete=True)
    base = dict(
        model="m",
        difficulty="easy",
        tags=["metric"],
        expect="answer",
        status="answered",
        sql="select 1",
        attempts=1,
        prompt_tokens=100,
        completion_tokens=10,
        llm_latency_s=1.0,
        context_tokens=50,
    )
    items = [
        runner.ItemResult(id="q1", rung=1, correct=False, reason="no column matches", **base),
        runner.ItemResult(id="q2", rung=1, correct=True, reason="match", **base),
        runner.ItemResult(id="q1", rung=3, correct=True, reason="match", **base),
        runner.ItemResult(id="q2", rung=3, correct=True, reason="match", **base),
    ]
    runner.save_results(tmp_path, meta, items)
    loaded_meta, loaded = runner.load_results(runner.results_path(tmp_path, "m"))
    assert loaded_meta == meta and loaded == items
    text = build_report(tmp_path)
    assert "**50.0%** (1/2)" in text and "**100.0%** (2/2)" in text
    assert "R1 → R3 (L3 + semantic layer): +1 fixed, −0 broken" in text
    assert "xychart-beta" in text


def test_wilson_interval():
    lo, hi = wilson(40, 50)
    assert 0.66 < lo < 0.70 and 0.88 < hi < 0.90
    assert wilson(0, 0) == (0.0, 0.0)


def test_redteam_suite_is_well_formed():
    items = runner.load_golden(settings.redteam_path)
    assert len({q.id for q in items}) == len(items) and all(q.id.startswith("rt_") for q in items)
    guard = SQLGuard(["marts.fct_orders"])
    for q in items:
        assert (q.sql is None) == (q.expect == "refuse"), q.id
        assert q.sql is None or guard.check(q.sql).ok, q.id
