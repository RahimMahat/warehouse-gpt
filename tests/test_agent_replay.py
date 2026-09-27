"""End-to-end agent runs replayed from recorded LLM responses (tests/cassettes).

No network or API keys needed. After changing prompts or context, re-record with:

    WGPT_RECORD_CASSETTES=1 uv run pytest tests/test_agent_replay.py

A replay miss means the exact prompt changed; the test fails rather than silently calling an API.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from warehouse_gpt.agent.graph import Agent
from warehouse_gpt.agent.llm import LLMClient
from warehouse_gpt.config import get_settings
from warehouse_gpt.context.render import ContextLevel

CASSETTES = Path(__file__).parent / "cassettes"
MODEL = "gpt-oss-120b"


@pytest.fixture(scope="module")
def agent(warehouse) -> Agent:
    settings = get_settings()
    if not (settings.context_dir / "profile.json").exists():
        pytest.skip("context not built; run `wgpt context build`")
    mode = "cache" if os.environ.get("WGPT_RECORD_CASSETTES") else "replay"
    return Agent.from_settings(llm=LLMClient(settings, mode=mode, store_dir=CASSETTES))


def test_answers_metric_question(agent, warehouse):
    r = agent.ask("What was total revenue in 2018?", level=ContextLevel.EXAMPLES, model=MODEL)
    expected = warehouse.sql(
        "select sum(price) from marts.fct_order_items where not is_canceled and year(purchase_date) = 2018"
    ).fetchone()[0]
    assert r.status == "answered", r.error
    assert float(r.result.rows[0][0]) == pytest.approx(float(expected), rel=1e-6)
    assert r.answer


def test_answers_ranked_dimension_question(agent, warehouse):
    r = agent.ask("Which 3 states have the most customers?", level=ContextLevel.EXAMPLES, model=MODEL)
    expected = warehouse.sql(
        "select state from marts.dim_customers group by 1 order by count(*) desc limit 3"
    ).fetchall()
    assert r.status == "answered", r.error
    assert [row[0] for row in r.result.rows] == [row[0] for row in expected]


def test_refuses_out_of_scope_question(agent):
    r = agent.ask("What will the weather be in Sao Paulo tomorrow?", level=ContextLevel.EXAMPLES, model=MODEL)
    assert r.status == "refused"
    assert r.sql is None
