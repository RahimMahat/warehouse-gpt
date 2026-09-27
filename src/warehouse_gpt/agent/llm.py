"""Provider-agnostic LLM client: model registry, free-tier rate limiting, retries, record/replay.

Every call goes through :class:`LLMClient.complete`, which

1. resolves a short alias ("gpt-oss-120b") to a LiteLLM model string and its free-tier limits,
2. serves the response from the on-disk store when the mode allows it (``cache``/``replay``),
3. waits for request and token budget in a sliding 60s window (Groq's free tier is 8k tokens/min),
4. retries rate-limit and overload errors with exponential backoff, honoring ``Retry-After``,
5. records token usage, so every eval run reports what it consumed.

The store doubles as the cassette layer: tests run with ``mode="replay"`` against committed
responses, with no network and no API keys.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog

from warehouse_gpt.config import Settings, get_settings

log = structlog.get_logger(__name__)

Mode = Literal["live", "cache", "replay"]
Message = dict[str, str]


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    litellm_model: str
    provider: Literal["gemini", "groq", "ollama", "other"]
    rpm: int  # requests per minute
    tpm: int  # tokens per minute
    temperature: bool = True  # Gemini 3+ deprecates sampling params
    params: dict[str, Any] = field(default_factory=dict)


# Limits are the free-tier values observed from provider headers (Groq) or docs (Gemini).
MODELS: dict[str, ModelSpec] = {
    s.alias: s
    for s in [
        ModelSpec(
            "gemini-flash",
            "gemini/gemini-flash-latest",
            "gemini",
            rpm=10,
            tpm=250_000,
            temperature=False,
            params={"reasoning_effort": "low"},
        ),
        ModelSpec(
            "gemini-flash-lite",
            "gemini/gemini-flash-lite-latest",
            "gemini",
            rpm=15,
            tpm=250_000,
            temperature=False,
        ),
        ModelSpec(
            "gpt-oss-120b",
            "groq/openai/gpt-oss-120b",
            "groq",
            rpm=30,
            tpm=8_000,
            params={"reasoning_effort": "medium"},
        ),
        ModelSpec(
            "gpt-oss-20b",
            "groq/openai/gpt-oss-20b",
            "groq",
            rpm=30,
            tpm=8_000,
            params={"reasoning_effort": "medium"},
        ),
        ModelSpec("qwen3-27b", "groq/qwen/qwen3.8-27b", "groq", rpm=30, tpm=8_000),
        ModelSpec("ollama-qwen-coder", "ollama_chat/qwen2.5-coder:7b", "ollama", rpm=1_000, tpm=10**9),
    ]
}


def resolve_model(name: str) -> ModelSpec:
    """Alias from :data:`MODELS`, or any raw LiteLLM string such as ``groq/llama-3.3-70b-versatile``."""
    if name in MODELS:
        return MODELS[name]
    provider = name.split("/", 1)[0]
    if provider.startswith("ollama"):
        return ModelSpec(name, name, "ollama", rpm=1_000, tpm=10**9)
    if provider in ("gemini", "groq"):
        return ModelSpec(name, name, provider, rpm=10, tpm=8_000, temperature=provider != "gemini")  # type: ignore[arg-type]
    return ModelSpec(name, name, "other", rpm=10, tpm=8_000)


def estimate_tokens(messages: list[Message]) -> int:
    return sum(len(m["content"]) for m in messages) // 4 + 8 * len(messages)


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool = False


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.cached_calls += r.cached
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.latency_s += 0.0 if r.cached else r.latency_s


class CassetteMiss(LookupError):
    """Replay mode found no stored response for this exact request."""


class ResponseStore:
    """Content-addressed JSON files: one per (model, messages, params) request."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(model: str, messages: list[Message], params: dict[str, Any]) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))["response"]
        return LLMResponse(**{**data, "cached": True})

    def put(self, key: str, request: dict[str, Any], response: LLMResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"request": request, "response": {**asdict(response), "cached": False}}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class RateLimiter:
    """Sliding 60s window over requests and tokens, per model. Thread-safe."""

    def __init__(self, rpm: int, tpm: int, clock: Any = time.monotonic, sleep: Any = time.sleep) -> None:
        self.rpm, self.tpm = rpm, tpm
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    def acquire(self, tokens: int) -> float:
        """Block until a request of ``tokens`` fits the budget. Returns seconds waited."""
        tokens = min(tokens, self.tpm)  # a single oversized request still goes through alone
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._events and now - self._events[0][0] >= 60:
                    self._events.popleft()
                used = sum(t for _, t in self._events)
                if len(self._events) < self.rpm and used + tokens <= self.tpm:
                    self._events.append((now, tokens))
                    return waited
                delay = 60 - (now - self._events[0][0]) + 0.05
            self._sleep(delay)
            waited += delay

    def correct(self, estimated: int, actual: int) -> None:
        """Replace the most recent estimate with the actual token count."""
        with self._lock:
            for i in range(len(self._events) - 1, -1, -1):
                t, tok = self._events[i]
                if tok == min(estimated, self.tpm):
                    self._events[i] = (t, actual)
                    return


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMClient:
    MAX_ATTEMPTS = 6

    def __init__(
        self,
        settings: Settings | None = None,
        mode: Mode | None = None,
        store_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.mode: Mode = mode or self.settings.llm_mode
        self.store = ResponseStore(store_dir or self.settings.llm_store_path)
        self.usage: dict[str, Usage] = {}
        self._limiters: dict[str, RateLimiter] = {}

    def _limiter(self, spec: ModelSpec) -> RateLimiter:
        if spec.litellm_model not in self._limiters:
            self._limiters[spec.litellm_model] = RateLimiter(spec.rpm, spec.tpm)
        return self._limiters[spec.litellm_model]

    def _credentials(self, spec: ModelSpec) -> dict[str, Any]:
        s = self.settings
        if spec.provider == "gemini":
            key = s.gemini_api_key
        elif spec.provider == "groq":
            key = s.groq_api_key
        elif spec.provider == "ollama":
            return {"api_base": s.ollama_base_url}
        else:
            return {}
        if key is None:
            raise RuntimeError(f"{spec.provider.upper()}_API_KEY is not set (add it to .env)")
        return {"api_key": key.get_secret_value()}

    def complete(self, model: str, messages: list[Message], max_tokens: int = 4096) -> LLMResponse:
        spec = resolve_model(model)
        params: dict[str, Any] = {"max_tokens": max_tokens, **spec.params}
        if spec.temperature:
            params["temperature"] = 0
        key = self.store.key(spec.litellm_model, messages, params)

        if self.mode in ("cache", "replay") and (hit := self.store.get(key)):
            self._record(spec, hit)
            return hit
        if self.mode == "replay":
            raise CassetteMiss(
                f"no stored response for {spec.alias} (key {key}); re-record with WGPT_LLM_MODE=cache"
            )

        response = self._call(spec, messages, params)
        self._record(spec, response)
        if self.mode == "cache":
            self.store.put(
                key, {"model": spec.litellm_model, "messages": messages, "params": params}, response
            )
        return response

    def _call(self, spec: ModelSpec, messages: list[Message], params: dict[str, Any]) -> LLMResponse:
        import litellm

        litellm.suppress_debug_info = True
        retryable = (
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            litellm.APIConnectionError,
            litellm.Timeout,
        )
        estimate = estimate_tokens(messages) + min(params["max_tokens"], 1024)
        limiter = self._limiter(spec)
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            waited = limiter.acquire(estimate)
            if waited:
                log.info("llm.rate_limited_locally", model=spec.alias, waited_s=round(waited, 1))
            t0 = time.perf_counter()
            try:
                raw = litellm.completion(
                    model=spec.litellm_model,
                    messages=messages,
                    timeout=180,
                    **params,
                    **self._credentials(spec),
                )
            except retryable as exc:
                if attempt == self.MAX_ATTEMPTS:
                    raise
                delay = _retry_after(exc) or min(60.0, 2.0**attempt) + random.uniform(0, 1)
                log.warning(
                    "llm.retry",
                    model=spec.alias,
                    attempt=attempt,
                    error=type(exc).__name__,
                    sleep_s=round(delay, 1),
                )
                time.sleep(delay)
                continue
            latency = time.perf_counter() - t0
            usage = raw.usage
            text = _THINK.sub("", raw.choices[0].message.content or "").strip()
            response = LLMResponse(
                text=text,
                model=str(raw.model or spec.litellm_model),
                prompt_tokens=int(usage.prompt_tokens or 0),
                completion_tokens=int(usage.completion_tokens or 0),
                latency_s=round(latency, 3),
            )
            limiter.correct(estimate, response.prompt_tokens + response.completion_tokens)
            return response
        raise AssertionError("unreachable")

    def _record(self, spec: ModelSpec, response: LLMResponse) -> None:
        self.usage.setdefault(spec.alias, Usage()).add(response)
        log.debug(
            "llm.call",
            model=spec.alias,
            cached=response.cached,
            latency_s=response.latency_s,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )


def _retry_after(exc: Exception) -> float | None:
    """Seconds from a Retry-After header or a 'try again in 7.5s' message, if present."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if value:
        try:
            return float(value) + 0.5
        except ValueError:
            pass
    if m := re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc)):
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 0.5
    return None
