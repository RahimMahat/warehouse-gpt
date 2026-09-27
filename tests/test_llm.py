"""LLM layer: model resolution, response store / replay, rate limiting, reply parsing. No network."""

from __future__ import annotations

import pytest

from warehouse_gpt.agent.llm import (
    CassetteMiss,
    LLMClient,
    LLMResponse,
    RateLimiter,
    ResponseStore,
    _retry_after,
    resolve_model,
)
from warehouse_gpt.agent.prompts import parse_reply


def test_resolve_alias_and_raw_model_strings():
    assert resolve_model("gpt-oss-120b").litellm_model == "groq/openai/gpt-oss-120b"
    assert resolve_model("gpt-oss-120b").tpm == 8_000
    raw = resolve_model("groq/llama-3.3-70b-versatile")
    assert raw.provider == "groq" and raw.litellm_model == "groq/llama-3.3-70b-versatile"
    assert resolve_model("ollama_chat/llama3").provider == "ollama"
    assert resolve_model("gemini/gemini-3.8-flash").temperature is False


def test_store_key_is_stable_and_request_sensitive():
    msgs = [{"role": "user", "content": "hi"}]
    k = ResponseStore.key("m", msgs, {"max_tokens": 10})
    assert k == ResponseStore.key("m", [dict(m) for m in msgs], {"max_tokens": 10})
    assert k != ResponseStore.key("m", msgs, {"max_tokens": 11})
    assert k != ResponseStore.key("other", msgs, {"max_tokens": 10})


def test_replay_serves_stored_response_and_fails_on_miss(tmp_path):
    msgs = [{"role": "user", "content": "q"}]
    client = LLMClient(mode="replay", store_dir=tmp_path)
    with pytest.raises(CassetteMiss):
        client.complete("gpt-oss-120b", msgs)

    # Record through the public path's key derivation, then replay.
    spec = resolve_model("gpt-oss-120b")
    params = {"max_tokens": 4096, **spec.params, "temperature": 0}
    key = ResponseStore.key(spec.litellm_model, msgs, params)
    client.store.put(key, {}, LLMResponse("```sql\nselect 1\n```", "m", 10, 5, 1.2))

    r = client.complete("gpt-oss-120b", msgs)
    assert r.cached and r.text.startswith("```sql")
    assert client.usage["gpt-oss-120b"].cached_calls == 1
    assert client.usage["gpt-oss-120b"].latency_s == 0  # cached calls cost no time


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_rate_limiter_enforces_token_budget():
    clock = FakeClock()
    rl = RateLimiter(rpm=30, tpm=8_000, clock=clock.now, sleep=clock.sleep)
    assert rl.acquire(5_000) == 0
    waited = rl.acquire(5_000)  # would exceed 8k tokens in the window
    assert waited == pytest.approx(60.05)
    assert rl.acquire(20_000) > 0  # oversized requests are clamped to the budget, not rejected forever


def test_rate_limiter_enforces_request_budget():
    clock = FakeClock()
    rl = RateLimiter(rpm=2, tpm=10**9, clock=clock.now, sleep=clock.sleep)
    rl.acquire(1)
    clock.t = 10
    rl.acquire(1)
    assert rl.acquire(1) == pytest.approx(50.05)  # the first request leaves the window at t=60


def test_rate_limiter_correction_frees_budget():
    clock = FakeClock()
    rl = RateLimiter(rpm=30, tpm=8_000, clock=clock.now, sleep=clock.sleep)
    rl.acquire(6_000)
    rl.correct(6_000, 1_000)  # the call actually used far fewer tokens than estimated
    assert rl.acquire(6_000) == 0


def test_retry_after_parses_provider_messages():
    assert _retry_after(Exception("Rate limit reached. Please try again in 7.5s.")) == pytest.approx(8.0)
    assert _retry_after(Exception("Please try again in 1m2.5s")) == pytest.approx(63.0)
    assert _retry_after(Exception("boom")) is None


@pytest.mark.parametrize(
    ("text", "sql", "refusal"),
    [
        ("```sql\nSELECT 1\n```", "SELECT 1", None),
        ("Here you go:\n```\nselect 2;\n```", "select 2;", None),
        ("```sql\nselect 1\n```\nActually:\n```sql\nselect 3\n```", "select 3", None),
        ("SELECT count(*) FROM marts.fct_orders", "SELECT count(*) FROM marts.fct_orders", None),
        ("CANNOT_ANSWER: no weather data", None, "no weather data"),
        ("I think the answer is 42.", None, None),
    ],
)
def test_parse_reply(text, sql, refusal):
    parsed = parse_reply(text)
    assert parsed.sql == sql
    assert parsed.refusal == refusal
