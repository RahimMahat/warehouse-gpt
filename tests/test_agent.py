"""Agent graph and executor against the real warehouse, with a scripted LLM (no network)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from warehouse_gpt.agent.executor import QueryExecutor
from warehouse_gpt.agent.graph import Agent
from warehouse_gpt.agent.guard import SQLGuard
from warehouse_gpt.agent.llm import LLMResponse
from warehouse_gpt.config import get_settings
from warehouse_gpt.context.catalog import load_catalog
from warehouse_gpt.context.render import ContextLevel, ContextRenderer

settings = get_settings()


class ScriptedLLM:
    """Returns canned replies in order and records every prompt it was sent."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[list[dict[str, str]]] = []

    def complete(self, model: str, messages: list[dict[str, str]], max_tokens: int = 4096) -> LLMResponse:
        self.prompts.append(messages)
        return LLMResponse(self.replies.pop(0), model, 100, 10, 0.01)


def sql(q: str) -> str:
    return f"```sql\n{q}\n```"


@pytest.fixture(scope="module")
def executor(warehouse) -> Iterator[QueryExecutor]:
    ex = QueryExecutor(settings.warehouse_path, timeout_s=5, max_rows=100)
    yield ex
    ex.close()


@pytest.fixture(scope="module")
def parts(warehouse, executor):
    catalog = load_catalog(settings.warehouse_path, settings.manifest_path)
    return ContextRenderer(catalog), executor, SQLGuard(catalog.tables)


def make_agent(parts, llm: ScriptedLLM) -> Agent:
    renderer, executor, guard = parts
    return Agent(llm, renderer, executor, guard, caveats=["8 delivered orders lack a delivery date"])  # type: ignore[arg-type]


def ask(agent: Agent, **kw):
    return agent.ask("How many orders are there?", level=ContextLevel.RAW_DDL, **kw)


# -- executor sandbox ----------------------------------------------------------------------
def test_executor_is_read_only_and_offline(executor, tmp_path: Path):
    assert executor.run("select count(*) from fct_orders").rows == [(99_441,)]  # search_path = marts
    assert "read-only" in (executor.run("create table marts.x as select 1").error or "").lower()
    probe = tmp_path / "x.csv"
    probe.write_text("a\n1\n")
    assert "disabled" in (executor.run(f"select * from read_csv('{probe.as_posix()}')").error or "")


def test_executor_caps_rows_and_times_out(executor):
    r = executor.run("select * from range(1000)")
    assert r.row_count == 100 and r.truncated
    slow = executor.run("select count(*) from range(100000000000) a")
    assert slow.error and "timed out" in slow.error


# -- graph paths -------------------------------------------------------------------------
def test_happy_path(parts):
    llm = ScriptedLLM(sql("select count(*) as orders from marts.fct_orders"), "There are 99,441 orders.")
    r = ask(make_agent(parts, llm))
    assert r.status == "answered" and r.attempts == 1
    assert r.result.rows == [(99_441,)]
    assert r.answer == "There are 99,441 orders."
    assert r.tables == ["marts.fct_orders"]
    assert [t["node"] for t in r.trace] == ["context", "generate", "validate", "execute", "answer"]
    assert "8 delivered orders" in llm.prompts[1][1]["content"]  # caveats reach the answer step


def test_repairs_sql_error_with_engine_feedback(parts):
    llm = ScriptedLLM(
        sql("select count(ordr_id) from marts.fct_orders"),
        sql("select count(order_id) from marts.fct_orders"),
    )
    r = ask(make_agent(parts, llm), answer=False)
    assert r.status == "answered" and r.attempts == 2
    assert "ordr_id" in llm.prompts[1][-1]["content"]  # the model saw the binder error
    assert r.result.rows == [(99_441,)]


def test_repairs_guard_violation(parts):
    llm = ScriptedLLM(
        sql("select count(*) from staging.stg_orders"), sql("select count(*) from marts.fct_orders")
    )
    r = ask(make_agent(parts, llm), answer=False)
    assert r.status == "answered" and r.attempts == 2
    assert "rejected by the SQL guard" in llm.prompts[1][-1]["content"]


def test_no_repair_loop_when_disabled(parts):
    llm = ScriptedLLM(sql("select count(ordr_id) from marts.fct_orders"))
    r = ask(make_agent(parts, llm), max_repairs=0, answer=False)
    assert r.status == "failed" and r.attempts == 1
    assert "ordr_id" in (r.error or "")


def test_repair_budget_is_bounded(parts):
    bad = sql("select nope from marts.fct_orders")
    r = ask(make_agent(parts, ScriptedLLM(bad, bad, bad)), max_repairs=2, answer=False)
    assert r.status == "failed" and r.attempts == 3


def test_refusal_ends_without_sql(parts):
    r = ask(make_agent(parts, ScriptedLLM("CANNOT_ANSWER: no weather data in the warehouse")))
    assert r.status == "refused" and r.sql is None
    assert r.refusal == "no weather data in the warehouse"


def test_reply_without_sql_is_retried(parts):
    llm = ScriptedLLM("There are lots of orders.", sql("select count(*) from marts.fct_orders"))
    r = ask(make_agent(parts, llm), answer=False)
    assert r.status == "answered" and r.attempts == 2


def test_empty_result_retried_once_then_accepted(parts):
    empty = sql("select order_id from marts.fct_orders where order_status = 'lost'")
    llm = ScriptedLLM(empty, empty)
    r = ask(make_agent(parts, llm), answer=False)
    assert r.status == "answered" and r.result.row_count == 0 and r.attempts == 2
    assert "0 rows" in llm.prompts[1][-1]["content"]
