"""The agent as a LangGraph state machine.

    context -> generate -> validate -> execute -> answer
                  ^           |           |
                  +-- repair -+-----------+   (guard violation, SQL error, empty result, no SQL)

``max_repairs=0`` turns the self-correction loop off, which is the rung below it on the ablation
ladder. A CANNOT_ANSWER reply ends the run as a refusal.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from warehouse_gpt.agent import prompts
from warehouse_gpt.agent.executor import QueryExecutor, QueryResult
from warehouse_gpt.agent.guard import SQLGuard
from warehouse_gpt.agent.llm import LLMClient, LLMResponse, Message
from warehouse_gpt.context.render import ContextLevel, ContextRenderer, RenderedContext

Status = Literal["answered", "refused", "failed"]


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    level: ContextLevel
    model: str
    max_repairs: int
    want_answer: bool
    # working state
    context: RenderedContext
    messages: list[Message]
    sql: str | None
    refusal: str | None
    feedback: str | None
    empty_retried_sql: str | None
    attempts: int
    result: QueryResult | None
    answer: str | None
    # accumulated
    llm_calls: Annotated[list[LLMResponse], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class AgentResponse:
    question: str
    status: Status
    sql: str | None
    result: QueryResult | None
    answer: str | None
    refusal: str | None
    error: str | None
    model: str
    level: ContextLevel
    attempts: int
    context_tokens: int
    example_ids: list[str]
    trace: list[dict[str, Any]]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_s: float = 0.0
    total_s: float = 0.0
    tables: list[str] = field(default_factory=list)


def _step(node: str, t0: float, **info: Any) -> list[dict[str, Any]]:
    return [{"node": node, "ms": round((time.perf_counter() - t0) * 1000, 1), **info}]


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        renderer: ContextRenderer,
        executor: QueryExecutor,
        guard: SQLGuard,
        caveats: list[str] | None = None,
        default_model: str = "gpt-oss-120b",
        default_max_repairs: int = 2,
    ) -> None:
        self.llm, self.renderer, self.executor, self.guard = llm, renderer, executor, guard
        self.caveats = caveats or []
        self.default_model, self.default_max_repairs = default_model, default_max_repairs
        self.graph = self._build().compile()

    @classmethod
    def from_settings(cls, llm: LLMClient | None = None) -> Agent:
        from warehouse_gpt.config import get_settings
        from warehouse_gpt.context.store import ContextStore

        settings = get_settings()
        store = ContextStore.load(settings)
        return cls(
            llm=llm or LLMClient(settings),
            renderer=store.renderer,
            executor=QueryExecutor(
                settings.warehouse_path, settings.query_timeout_s, settings.max_result_rows
            ),
            guard=SQLGuard(store.catalog.tables),
            caveats=store.profile.caveats,
            default_model=settings.default_model,
            default_max_repairs=settings.max_repairs,
        )

    # -- nodes ----------------------------------------------------------------------------
    def _context(self, s: AgentState) -> AgentState:
        t0 = time.perf_counter()
        ctx = self.renderer.render(s["question"], s["level"])
        messages: list[Message] = [
            {"role": "system", "content": prompts.SQL_SYSTEM.format(context=ctx.text)},
            {"role": "user", "content": s["question"]},
        ]
        return {
            "context": ctx,
            "messages": messages,
            "attempts": 0,
            "trace": _step(
                "context", t0, level=ctx.level.name, tokens=ctx.approx_tokens, examples=ctx.example_ids
            ),
        }

    def _generate(self, s: AgentState) -> AgentState:
        t0 = time.perf_counter()
        reply = self.llm.complete(s["model"], s["messages"])
        parsed = prompts.parse_reply(reply.text)
        messages = [*s["messages"], {"role": "assistant", "content": reply.text}]
        feedback = None if parsed.sql or parsed.refusal is not None else prompts.REPAIR_NO_SQL
        return {
            "messages": messages,
            "sql": parsed.sql,
            "refusal": parsed.refusal,
            "feedback": feedback,
            "result": None,
            "attempts": s["attempts"] + 1,
            "llm_calls": [reply],
            "trace": _step(
                "generate",
                t0,
                attempt=s["attempts"] + 1,
                cached=reply.cached,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                refused=parsed.refusal is not None,
            ),
        }

    def _validate(self, s: AgentState) -> AgentState:
        t0 = time.perf_counter()
        check = self.guard.check(s["sql"] or "")
        feedback = (
            None
            if check.ok
            else prompts.REPAIR_ERROR.format(
                error="The query was rejected by the SQL guard:\n- " + "\n- ".join(check.errors)
            )
        )
        return {
            "sql": check.sql,
            "feedback": feedback,
            "trace": _step("validate", t0, ok=check.ok, tables=check.tables, errors=check.errors),
        }

    def _execute(self, s: AgentState) -> AgentState:
        t0 = time.perf_counter()
        result = self.executor.run(s["sql"] or "")
        feedback: str | None = None
        empty_retried = s.get("empty_retried_sql")
        if not result.ok:
            feedback = prompts.REPAIR_ERROR.format(error=result.error)
        elif result.row_count == 0 and empty_retried != s["sql"]:
            # Ask once per distinct query; the same query again means "empty is correct".
            feedback, empty_retried = prompts.REPAIR_EMPTY, s["sql"]
        return {
            "result": result,
            "feedback": feedback,
            "empty_retried_sql": empty_retried,
            "trace": _step(
                "execute",
                t0,
                ok=result.ok,
                rows=result.row_count,
                truncated=result.truncated,
                db_ms=result.elapsed_ms,
                error=result.error,
            ),
        }

    def _repair(self, s: AgentState) -> AgentState:
        messages = [*s["messages"], {"role": "user", "content": s["feedback"] or ""}]
        return {
            "messages": messages,
            "trace": [{"node": "repair", "ms": 0.0, "reason": (s["feedback"] or "").splitlines()[0]}],
        }

    def _answer(self, s: AgentState) -> AgentState:
        result = s.get("result")
        if not s.get("want_answer", True) or result is None or not result.ok:
            return {}
        t0 = time.perf_counter()
        user = prompts.ANSWER_USER.format(
            question=s["question"],
            sql=s["sql"],
            row_count=result.row_count,
            truncated=", truncated" if result.truncated else "",
            table=result.to_markdown(30),
            caveats="\n".join(f"- {c}" for c in self.caveats) or "(none)",
        )
        reply = self.llm.complete(
            s["model"],
            [{"role": "system", "content": prompts.ANSWER_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=2048,
        )
        return {
            "answer": reply.text,
            "llm_calls": [reply],
            "trace": _step(
                "answer",
                t0,
                cached=reply.cached,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
            ),
        }

    # -- routing --------------------------------------------------------------------------
    def _can_repair(self, s: AgentState) -> bool:
        return s["attempts"] <= s["max_repairs"]

    def _after_generate(self, s: AgentState) -> str:
        if s.get("refusal") is not None:
            return END
        if s.get("feedback"):
            return "repair" if self._can_repair(s) else END
        return "validate"

    def _after_validate(self, s: AgentState) -> str:
        if s.get("feedback"):
            return "repair" if self._can_repair(s) else END
        return "execute"

    def _after_execute(self, s: AgentState) -> str:
        if s.get("feedback") and self._can_repair(s):
            return "repair"
        return "answer"

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        nodes = {
            "context": self._context,
            "generate": self._generate,
            "validate": self._validate,
            "execute": self._execute,
            "repair": self._repair,
            "answer": self._answer,
        }
        for name, fn in nodes.items():
            g.add_node(name, fn)  # type: ignore[call-overload]  # langgraph's overloads reject bound methods
        g.add_edge(START, "context")
        g.add_edge("context", "generate")
        g.add_conditional_edges("generate", self._after_generate, ["validate", "repair", END])
        g.add_conditional_edges("validate", self._after_validate, ["execute", "repair", END])
        g.add_conditional_edges("execute", self._after_execute, ["repair", "answer"])
        g.add_edge("repair", "generate")
        g.add_edge("answer", END)
        return g

    # -- public ---------------------------------------------------------------------------
    def ask(
        self,
        question: str,
        level: ContextLevel = ContextLevel.EXAMPLES,
        model: str | None = None,
        max_repairs: int | None = None,
        answer: bool = True,
    ) -> AgentResponse:
        t0 = time.perf_counter()
        model = model or self.default_model
        s = cast(
            AgentState,
            self.graph.invoke(
                {
                    "question": question,
                    "level": ContextLevel(level),
                    "model": model,
                    "max_repairs": self.default_max_repairs if max_repairs is None else max_repairs,
                    "want_answer": answer,
                    "llm_calls": [],
                    "trace": [],
                },
                {"recursion_limit": 50},
            ),
        )
        return self._response(s, model, time.perf_counter() - t0)

    def _response(self, s: AgentState, model: str, elapsed: float) -> AgentResponse:
        result = s.get("result")
        refusal = s.get("refusal")
        status: Status
        error: str | None = None
        if refusal is not None:
            status = "refused"
        elif result is not None and result.ok and not s.get("feedback"):
            status = "answered"
        elif result is not None and result.ok and result.row_count == 0:
            status = "answered"  # empty after the retry budget: a legitimate (if suspicious) answer
        else:
            status = "failed"
            error = (result.error if result is not None and not result.ok else None) or s.get("feedback")
        calls = s.get("llm_calls", [])
        ctx = s["context"]
        guard_tables: list[str] = next(
            (t["tables"] for t in reversed(s["trace"]) if t["node"] == "validate"), []
        )
        return AgentResponse(
            question=s["question"],
            status=status,
            sql=s.get("sql"),
            result=result,
            answer=s.get("answer"),
            refusal=refusal,
            error=error,
            model=model,
            level=s["level"],
            attempts=s.get("attempts", 0),
            context_tokens=ctx.approx_tokens,
            example_ids=ctx.example_ids,
            trace=s["trace"],
            prompt_tokens=sum(c.prompt_tokens for c in calls),
            completion_tokens=sum(c.completion_tokens for c in calls),
            llm_latency_s=round(sum(c.latency_s for c in calls if not c.cached), 2),
            total_s=round(elapsed, 2),
            tables=guard_tables,
        )
