"""API, streaming, tracing, charts and request guards, driven by a scripted LLM (no network)."""

from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from tests.test_agent import ScriptedLLM, sql
from warehouse_gpt import observability as obs
from warehouse_gpt.agent.chart import chart_spec
from warehouse_gpt.agent.executor import QueryExecutor, QueryResult
from warehouse_gpt.agent.graph import Agent
from warehouse_gpt.agent.guard import SQLGuard
from warehouse_gpt.api.app import create_app
from warehouse_gpt.api.guards import AnswerCache, ClientRateLimiter, normalize_question
from warehouse_gpt.config import Settings, get_settings
from warehouse_gpt.context.catalog import load_catalog
from warehouse_gpt.context.render import ContextRenderer

settings = get_settings()
COUNT_SQL = sql("select count(*) as orders from marts.fct_orders")


@pytest.fixture(scope="module")
def parts(warehouse):
    catalog = load_catalog(settings.warehouse_path, settings.manifest_path)
    ex = QueryExecutor(settings.warehouse_path)
    yield ContextRenderer(catalog), ex, SQLGuard(catalog.tables)
    ex.close()


def make_agent(parts, *replies: str) -> tuple[Agent, ScriptedLLM]:
    llm = ScriptedLLM(*replies)
    renderer, ex, guard = parts
    return Agent(llm, renderer, ex, guard), llm  # type: ignore[arg-type]


def client_for(agent: Agent, **overrides) -> TestClient:
    s = Settings(**{"api_rate_limit_per_min": 100, **overrides})
    return TestClient(create_app(agent=agent, settings=s))


# -- API ------------------------------------------------------------------------------------
def test_ask_returns_rows_answer_chartless_kpi_and_usage(parts):
    agent, _ = make_agent(parts, COUNT_SQL, "There are 99,441 orders.")
    with client_for(agent) as c:
        r = c.post("/ask", json={"question": "How many orders?", "level": 1})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "answered" and body["rows"] == [[99441]]
        assert body["answer"] == "There are 99,441 orders."
        assert body["chart"] is None  # a single number is a KPI, not a chart
        assert body["usage"]["prompt_tokens"] == 200 and body["level"] == "RAW_DDL"
        assert r.headers["x-request-id"] == body["request_id"]


def test_answer_cache_serves_repeat_questions_without_llm_calls(parts):
    agent, llm = make_agent(parts, COUNT_SQL, "99,441 orders.")
    with client_for(agent) as c:
        first = c.post("/ask", json={"question": "How many orders?", "level": 1}).json()
        again = c.post("/ask", json={"question": "  how many ORDERS  ", "level": 1}).json()
        assert not first["cached"] and again["cached"]
        assert again["rows"] == first["rows"] and len(llm.prompts) == 2  # no third LLM call
        assert c.get("/health").json()["cache"]["hits"] == 1


def test_stream_emits_steps_then_final(parts):
    agent, _ = make_agent(parts, COUNT_SQL)
    with client_for(agent) as c:
        r = c.post("/ask/stream", json={"question": "How many orders?", "level": 1, "answer": False})
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [
            (
                block.split("\n")[0].removeprefix("event: "),
                json.loads(block.split("\n")[1].removeprefix("data: ")),
            )
            for block in r.text.strip().split("\n\n")
        ]
        nodes = [d["node"] for e, d in events if e == "step"]
        assert nodes == ["context", "generate", "validate", "execute"]
        assert events[1][1]["sql"].startswith("select count(*)")  # SQL is visible as soon as it's generated
        assert events[-1][0] == "final" and events[-1][1]["rows"] == [[99441]]


def test_stream_reports_agent_errors_as_events(parts):
    agent, _ = make_agent(parts)  # no replies: the LLM raises IndexError
    with client_for(agent) as c:
        r = c.post("/ask/stream", json={"question": "How many orders?", "level": 1})
        assert "event: error" in r.text and "IndexError" in r.text


def test_validation_and_model_allowlist(parts):
    agent, _ = make_agent(parts)
    with client_for(agent) as c:
        assert c.post("/ask", json={"question": "hi"}).status_code == 422  # too short
        assert c.post("/ask", json={"question": "How many orders?", "level": 9}).status_code == 422
        bad = c.post("/ask", json={"question": "How many orders?", "model": "openai/gpt-4o"})
        assert bad.status_code == 422 and "unknown model" in bad.text


def test_bearer_token_and_rate_limit(parts):
    agent, _ = make_agent(parts, COUNT_SQL, "ok", COUNT_SQL, "ok")
    with client_for(agent, api_token=SecretStr("s3cret"), api_rate_limit_per_min=1) as c:
        q = {"question": "How many orders?", "level": 1}
        assert c.post("/ask", json=q).status_code == 401
        assert c.post("/ask", json=q, headers={"Authorization": "Bearer nope"}).status_code == 401
        ok = c.post("/ask", json=q, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        limited = c.post("/ask", json=q, headers={"Authorization": "Bearer s3cret"})
        assert limited.status_code == 429 and int(limited.headers["retry-after"]) > 0


def test_models_and_metrics_endpoints(parts):
    agent, _ = make_agent(parts)
    with client_for(agent) as c:
        models = c.get("/models").json()
        assert any(m["alias"] == "gpt-oss-120b" and m["tokens_per_min"] == 8000 for m in models)
        assert c.get("/metrics").json() == []  # the L1-only test renderer has no semantic layer


# -- tracing --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spans():
    exporter = InMemorySpanExporter()
    if not obs.init_tracing(Settings(), exporter=exporter):
        pytest.skip("a tracer provider was already installed in this process")
    return exporter


def test_trace_tree_follows_openinference_conventions(parts, spans):
    spans.clear()
    agent, _ = make_agent(parts, COUNT_SQL, "99,441 orders.")
    agent.ask("How many orders?", level=1)
    finished = {s.name: s for s in spans.get_finished_spans()}
    root = finished["agent"]
    assert root.attributes[obs.KIND] == "AGENT" and root.attributes["wgpt.status"] == "answered"
    for node, kind in [
        ("context", "RETRIEVER"),
        ("generate", "CHAIN"),
        ("validate", "GUARDRAIL"),
        ("execute", "TOOL"),
    ]:
        assert finished[node].attributes[obs.KIND] == kind
        assert finished[node].parent.span_id == root.context.span_id, node
    assert "99441" in finished["execute"].attributes[obs.OUTPUT]


def test_llm_span_carries_messages_and_token_counts(spans, tmp_path):
    from warehouse_gpt.agent.llm import LLMClient, LLMResponse, ResponseStore, resolve_model

    client = LLMClient(mode="replay", store_dir=tmp_path)
    msgs = [{"role": "system", "content": "ctx"}, {"role": "user", "content": "How many orders?"}]
    spec = resolve_model("gpt-oss-120b")
    key = ResponseStore.key(spec.litellm_model, msgs, {"max_tokens": 4096, **spec.params, "temperature": 0})
    client.store.put(key, {}, LLMResponse(sql("select 1"), "m", 120, 30, 1.5))
    spans.clear()
    client.complete("gpt-oss-120b", msgs)
    (llm,) = [s for s in spans.get_finished_spans() if s.name == "llm"]
    a = llm.attributes
    assert a[obs.KIND] == "LLM" and a["llm.model_name"] == "groq/openai/gpt-oss-120b"
    assert a["llm.input_messages.1.message.content"] == "How many orders?"
    assert a["llm.token_count.total"] == 150 and a["wgpt.cached"] is True


# -- charts & guards ------------------------------------------------------------------------
def test_chart_heuristics():
    line = chart_spec(QueryResult(["month", "revenue"], [(date(2018, 1, 1), 1.0), (date(2018, 2, 1), 2.0)]))
    assert line and line["mark"]["type"] == "line" and line["encoding"]["x"]["type"] == "temporal"
    bars = chart_spec(QueryResult(["state", "orders"], [("SP", 10), ("RJ", 5)]))
    assert bars and bars["mark"] == "bar" and bars["encoding"]["y"]["sort"] == "-x"
    ordinal = chart_spec(QueryResult(["hour", "orders"], [(10, 3), (11, 4)]))
    assert ordinal and ordinal["encoding"]["x"]["type"] == "ordinal"
    assert chart_spec(QueryResult(["orders"], [(1,), (2,)])) is None  # nothing to plot against
    assert chart_spec(QueryResult(["a", "b"], [("x", "y"), ("z", "w")])) is None  # no measure


def test_rate_limiter_window():
    now = [0.0]
    rl = ClientRateLimiter(2, clock=lambda: now[0])
    assert rl.check("a") == 0 and rl.check("a") == 0
    assert rl.check("a") == 60.0 and rl.check("b") == 0  # per client
    now[0] = 61
    assert rl.check("a") == 0


def test_answer_cache_ttl_and_lru():
    now = [0.0]
    cache = AnswerCache(maxsize=2, ttl_s=10, clock=lambda: now[0])
    k1, k2, k3 = (AnswerCache.key(q, level=4) for q in ("a q", "b q", "c q"))
    cache.put(k1, 1)
    cache.put(k2, 2)
    cache.put(k3, 3)  # evicts k1
    assert cache.get(k1) is None and cache.get(k3) == 3
    now[0] = 11
    assert cache.get(k3) is None  # expired
    assert normalize_question(" Revenue  in 2017? ") == "revenue in 2017"
    assert AnswerCache.key("revenue in 2017") != AnswerCache.key("revenue in 2018")


def test_ui_renders_without_api(monkeypatch):
    """The Streamlit app must degrade gracefully (an error message, no exception) when the API is down."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("WGPT_API_URL", "http://127.0.0.1:9")  # nothing listens on the discard port
    at = AppTest.from_file("../ui/app.py", default_timeout=30).run()
    assert not at.exception
    assert any("API not reachable" in e.value for e in at.sidebar.error)
