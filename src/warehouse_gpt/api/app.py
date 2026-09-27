"""HTTP API: JSON and Server-Sent-Events endpoints over the agent.

    POST /ask          -> full answer as JSON
    POST /ask/stream   -> SSE: one `step` event per agent node, then `final` (or `error`)
    GET  /models       -> model aliases, free-tier limits, credential status
    GET  /metrics      -> governed business metrics (what the agent knows how to compute)
    GET  /health

Run with ``wgpt serve``. Set ``WGPT_API_TOKEN`` to require ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import json
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from warehouse_gpt.agent.chart import chart_spec, jsonable
from warehouse_gpt.agent.graph import Agent, AgentResponse
from warehouse_gpt.agent.llm import MODELS
from warehouse_gpt.api.guards import AnswerCache, ClientRateLimiter
from warehouse_gpt.config import Settings, get_settings
from warehouse_gpt.context.render import ContextLevel
from warehouse_gpt.observability import configure_logging, init_tracing

log = structlog.get_logger(__name__)
MAX_ROWS_RETURNED = 500


# -- schemas -------------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    level: int = Field(default=4, ge=1, le=4, description="1=DDL 2=+docs 3=+semantic 4=+examples")
    model: str | None = Field(default=None, description="Model alias from GET /models")
    max_repairs: int | None = Field(default=None, ge=0, le=3)
    answer: bool = True

    @field_validator("model")
    @classmethod
    def _known_model(cls, v: str | None) -> str | None:
        # Only registered aliases: clients must not be able to point the server at arbitrary providers.
        if v is not None and v not in MODELS:
            raise ValueError(f"unknown model '{v}'; choose one of {sorted(MODELS)}")
        return v


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cached_llm_calls: int
    llm_latency_s: float
    total_s: float


class AskResponse(BaseModel):
    request_id: str
    question: str
    status: str
    answer: str | None
    sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    refusal: str | None
    error: str | None
    chart: dict[str, Any] | None
    model: str
    level: str
    attempts: int
    context_tokens: int
    example_ids: list[str]
    tables: list[str]
    usage: Usage
    trace: list[dict[str, Any]]
    cached: bool = False


def to_response(r: AgentResponse, request_id: str) -> AskResponse:
    result = r.result if r.result is not None and r.result.ok else None
    rows = [[jsonable(v) for v in row] for row in (result.rows[:MAX_ROWS_RETURNED] if result else [])]
    return AskResponse(
        request_id=request_id,
        question=r.question,
        status=r.status,
        answer=r.answer,
        sql=r.sql,
        columns=result.columns if result else [],
        rows=rows,
        row_count=result.row_count if result else 0,
        truncated=bool(result and (result.truncated or result.row_count > MAX_ROWS_RETURNED)),
        refusal=r.refusal,
        error=r.error,
        chart=chart_spec(result),
        model=r.model,
        level=r.level.name,
        attempts=r.attempts,
        context_tokens=r.context_tokens,
        example_ids=r.example_ids,
        tables=r.tables,
        usage=Usage(
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            cached_llm_calls=r.cached_calls,
            llm_latency_s=r.llm_latency_s,
            total_s=r.total_s,
        ),
        trace=[{k: jsonable(v) for k, v in step.items()} for step in r.trace],
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# -- app -----------------------------------------------------------------------------------
def create_app(agent: Agent | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging("INFO" if settings.log_level == "WARNING" else settings.log_level, json_logs=True)
    init_tracing(settings)
    limiter = ClientRateLimiter(settings.api_rate_limit_per_min)
    cache = AnswerCache(settings.answer_cache_size, settings.answer_cache_ttl_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.agent = agent or Agent.from_settings()
        log.info("api.started", default_model=app.state.agent.default_model)
        yield

    app = FastAPI(
        title="WarehouseGPT",
        description="Analytics agent over a Spark + dbt lakehouse.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return response

    def authorize(request: Request) -> str:
        """Returns the client key used for rate limiting."""
        if settings.api_token is not None:
            header = request.headers.get("authorization", "")
            token = header.removeprefix("Bearer ").strip()
            if not secrets.compare_digest(token, settings.api_token.get_secret_value()):
                raise HTTPException(401, "missing or invalid bearer token")
            return f"token:{token[-6:]}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def rate_limited(client: str = Depends(authorize)) -> str:
        wait = limiter.check(client)
        if wait:
            raise HTTPException(
                429, f"rate limit exceeded; retry in {wait:.0f}s", {"Retry-After": str(int(wait) + 1)}
            )
        return client

    def cache_key(req: AskRequest, default_model: str) -> tuple[Any, ...]:
        return AnswerCache.key(
            req.question,
            level=req.level,
            model=req.model or default_model,
            max_repairs=req.max_repairs,
            answer=req.answer,
        )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        a: Agent = request.app.state.agent
        return {
            "status": "ok",
            "default_model": a.default_model,
            "cache": {"hits": cache.hits, "misses": cache.misses},
        }

    @app.get("/models")
    def models() -> list[dict[str, Any]]:
        keys = {"gemini": settings.gemini_api_key, "groq": settings.groq_api_key}
        return [
            {
                "alias": m.alias,
                "model": m.litellm_model,
                "provider": m.provider,
                "requests_per_min": m.rpm,
                "tokens_per_min": m.tpm,
                "configured": m.provider == "ollama" or keys.get(m.provider) is not None,
                "default": m.alias == settings.default_model,
            }
            for m in MODELS.values()
        ]

    @app.get("/metrics")
    def metrics(request: Request) -> list[dict[str, Any]]:
        semantic = request.app.state.agent.renderer.semantic
        if semantic is None:
            return []
        return [
            {"name": m.name, "label": m.label, "description": m.description, "synonyms": m.synonyms}
            for m in sorted(semantic.public_metrics, key=lambda m: m.name)
        ]

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request, _: str = Depends(rate_limited)) -> AskResponse:
        a: Agent = request.app.state.agent
        key = cache_key(req, a.default_model)
        if (hit := cache.get(key)) is not None:
            return hit.model_copy(update={"request_id": request.state.request_id, "cached": True})
        try:
            r = a.ask(
                req.question,
                level=ContextLevel(req.level),
                model=req.model,
                max_repairs=req.max_repairs,
                answer=req.answer,
            )
        except Exception as exc:
            log.exception("agent.error")
            raise HTTPException(503, f"agent unavailable: {type(exc).__name__}") from exc
        out = to_response(r, request.state.request_id)
        if r.status in ("answered", "refused"):
            cache.put(key, out)
        return out

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest, request: Request, _: str = Depends(rate_limited)) -> StreamingResponse:
        a: Agent = request.app.state.agent
        request_id = request.state.request_id
        key = cache_key(req, a.default_model)

        def events() -> Iterator[str]:
            if (hit := cache.get(key)) is not None:
                cached = hit.model_copy(update={"request_id": request_id, "cached": True})
                yield _sse("final", cached.model_dump())
                return
            try:
                for kind, payload in a.stream(
                    req.question,
                    level=ContextLevel(req.level),
                    model=req.model,
                    max_repairs=req.max_repairs,
                    answer=req.answer,
                ):
                    if kind == "step":
                        yield _sse("step", {k: jsonable(v) for k, v in payload.items()})
                    else:
                        out = to_response(payload, request_id)
                        if payload.status in ("answered", "refused"):
                            cache.put(key, out)
                        yield _sse("final", out.model_dump())
            except Exception as exc:
                log.exception("agent.error")
                yield _sse("error", {"detail": f"agent unavailable: {type(exc).__name__}"})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "x-request-id": request_id},
        )

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return app
