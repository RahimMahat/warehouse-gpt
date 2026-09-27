"""WarehouseGPT chat UI (Streamlit). Talks to the API from `wgpt serve` over Server-Sent Events.

wgpt serve      # terminal 1
wgpt ui         # terminal 2 -> http://localhost:8501
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("WGPT_API_URL", "http://localhost:8000")
LEVELS = {
    1: "L1 · raw DDL",
    2: "L2 · + dbt docs",
    3: "L3 · + semantic layer",
    4: "L4 · + verified examples",
}
EXAMPLES = [
    "What were the top 5 product categories by revenue in 2017?",
    "How did the late delivery rate change by month in 2018?",
    "Which states have the most unique customers?",
    "What share of orders were paid with boleto?",
]

st.set_page_config(page_title="WarehouseGPT", page_icon="📊", layout="wide")


# -- API helpers ---------------------------------------------------------------------------
def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


@st.cache_data(ttl=60, show_spinner=False)
def fetch(path: str, api_url: str, token: str) -> Any:
    r = httpx.get(f"{api_url}{path}", headers=_headers(token), timeout=10)
    r.raise_for_status()
    return r.json()


def stream_ask(api_url: str, token: str, payload: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event, data) pairs from the SSE endpoint."""
    with httpx.stream(
        "POST", f"{api_url}/ask/stream", json=payload, headers=_headers(token), timeout=300
    ) as r:
        if r.status_code != 200:
            r.read()
            detail = r.json().get("detail") if "json" in r.headers.get("content-type", "") else r.text
            yield "error", {"detail": f"HTTP {r.status_code}: {detail}"}
            return
        event, data = "message", ""
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
            elif not line and data:
                yield event, json.loads(data)
                event, data = "message", ""


def describe_step(step: dict[str, Any]) -> str:
    node = step["node"]
    if node == "context":
        ex = f", examples: {', '.join(step['examples'])}" if step.get("examples") else ""
        return f"**Context** · {step['level']} · ~{step['tokens']:,} tokens{ex}"
    if node == "generate":
        if step.get("refused"):
            return "**Model declined** (out of scope)"
        return f"**Generated SQL** · attempt {step['attempt']} · {step['ms'] / 1000:.1f}s"
    if node == "validate":
        if step["ok"]:
            return f"**SQL guard passed** · tables: {', '.join(step['tables']) or '-'}"
        return f"**SQL guard rejected** · {'; '.join(step['errors'])[:160]}"
    if node == "execute":
        if step["ok"]:
            return f"**Executed** · {step['rows']:,} rows in {step['db_ms']:.0f} ms"
        return f"**Query error** · {(step.get('error') or '')[:160]}"
    if node == "repair":
        return f"**Self-correcting** · {step.get('reason', '')[:120]}"
    if node == "answer":
        return "**Wrote the answer**"
    return node


# -- rendering -----------------------------------------------------------------------------
def render_result(res: dict[str, Any]) -> None:
    status = res["status"]
    if status == "refused":
        st.warning(f"I can't answer that from the warehouse: {res.get('refusal') or 'out of scope'}")
    elif status == "failed":
        st.error(
            f"I couldn't produce a working query after {res['attempts']} attempt(s). {res.get('error') or ''}"
        )
    if res.get("answer"):
        st.markdown(res["answer"])

    if status != "refused" and (res.get("sql") or res.get("rows")):
        tabs = st.tabs(["Chart", "Table", "SQL", "Trace"] if res.get("chart") else ["Table", "SQL", "Trace"])
        i = 0
        if res.get("chart"):
            with tabs[0]:
                st.vega_lite_chart(res["chart"], width="stretch")
            i = 1
        with tabs[i]:
            if res["columns"]:
                st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]), width="stretch")
                note = " (truncated)" if res["truncated"] else ""
                st.caption(f"{res['row_count']:,} rows{note}")
            else:
                st.caption("No rows.")
        with tabs[i + 1]:
            st.code(res.get("sql") or "", language="sql")
            if res.get("tables"):
                st.caption("Tables: " + ", ".join(res["tables"]))
        with tabs[i + 2]:
            u = res["usage"]
            cols = st.columns(4)
            cols[0].metric("Attempts", res["attempts"])
            cols[1].metric("Tokens in / out", f"{u['prompt_tokens']:,} / {u['completion_tokens']:,}")
            cols[2].metric("LLM time", f"{u['llm_latency_s']:.1f}s")
            cols[3].metric("Total", f"{u['total_s']:.1f}s")
            st.dataframe(pd.DataFrame(res["trace"]), width="stretch", hide_index=True)
    meta = f"{res['model']} · {res['level']} · request {res['request_id']}"
    st.caption(meta + (" · served from cache" if res.get("cached") else ""))


# -- sidebar -------------------------------------------------------------------------------
with st.sidebar:
    st.title("WarehouseGPT")
    st.caption("Ask business questions about the Olist marketplace (2016-2018).")
    api_url = st.text_input("API URL", API_URL)
    token = st.text_input("API token", os.environ.get("WGPT_API_TOKEN", ""), type="password")
    try:
        models = [m for m in fetch("/models", api_url, token) if m["configured"]]
        default = next((i for i, m in enumerate(models) if m["default"]), 0)
        model = st.selectbox("Model", [m["alias"] for m in models], index=default)
        api_ok = True
    except httpx.HTTPError as exc:
        st.error(f"API not reachable at {api_url}: {exc}")
        model, api_ok = None, False
    level = st.select_slider("Context", options=list(LEVELS), value=4, format_func=LEVELS.get)
    self_correct = st.toggle("Self-correction", value=True, help="Retry on SQL errors and empty results.")
    if api_ok:
        with st.expander("Metrics I know"):
            for m in fetch("/metrics", api_url, token):
                st.markdown(f"**{m['label']}** · {m['description']}")
    st.divider()
    st.caption("Try:")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state.pending = ex

# -- chat ----------------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["content"])

question = st.chat_input(
    "Ask about orders, revenue, customers, sellers, delivery..."
) or st.session_state.pop("pending", None)
if question and api_ok:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        payload = {
            "question": question,
            "level": level,
            "model": model,
            "max_repairs": 2 if self_correct else 0,
        }
        final: dict[str, Any] | None = None
        with st.status("Thinking...", expanded=True) as status:
            for event, data in stream_ask(api_url, token, payload):
                if event == "step":
                    st.markdown(describe_step(data))
                elif event == "final":
                    final = data
                    label = {"answered": "Done", "refused": "Declined", "failed": "Failed"}[data["status"]]
                    status.update(
                        label=label,
                        state="error" if data["status"] == "failed" else "complete",
                        expanded=False,
                    )
                else:
                    status.update(label="Error", state="error")
                    st.error(data.get("detail", "unknown error"))
        if final is not None:
            render_result(final)
            st.session_state.messages.append({"role": "assistant", "content": final})
