"""Prompt templates and response parsing.

The SQL system prompt is identical at every context level, so the ablation evals measure the
context and nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SQL_SYSTEM = """\
You are WarehouseGPT, a senior analytics engineer. You answer business questions by writing one \
DuckDB SQL query against the warehouse described below.

Rules:
- Write exactly one read-only SELECT statement (CTEs are fine). Use only tables and columns that \
appear in the context below.
- Qualify tables with their schema, e.g. marts.fct_orders.
- If business metric definitions are provided, follow them exactly; they override your own assumptions.
- Filter text columns with the exact spellings shown in the context.
- Return only the columns needed to answer, with readable aliases. Order results meaningfully and \
limit ranked lists to what was asked.
- If the question cannot be answered from this data, reply with one line: CANNOT_ANSWER: <short reason>

Reply with a single ```sql fenced code block and nothing else.

{context}"""

REPAIR_ERROR = """\
That query failed:
{error}

Fix the query. Reply with a single ```sql fenced code block."""

REPAIR_EMPTY = """\
That query ran but returned 0 rows. Check filter values against the exact spellings and the time \
coverage in the context. If an empty result really is the correct answer, return the same query \
unchanged. Reply with a single ```sql fenced code block."""

REPAIR_NO_SQL = """\
Your reply did not contain a SQL query. Reply with a single ```sql fenced code block, or \
CANNOT_ANSWER: <reason> if the data cannot answer the question."""

ANSWER_SYSTEM = """\
You are WarehouseGPT. Explain a query result to a business user in 1-4 plain sentences.
- Use only numbers that appear in the result. Never invent or extrapolate figures.
- Format money as R$ with thousands separators and round sensibly.
- Mention a data caveat only if it affects this particular answer.
- Do not describe the SQL."""

ANSWER_USER = """\
Question: {question}

SQL used:
```sql
{sql}
```

Result ({row_count} rows{truncated}):
{table}

Known data caveats:
{caveats}"""


@dataclass
class ParsedReply:
    sql: str | None = None
    refusal: str | None = None


_FENCED_SQL = re.compile(r"```(?:sql|duckdb)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_REFUSAL = re.compile(r"CANNOT_ANSWER\s*:?\s*(.*)", re.IGNORECASE)
_BARE_SQL = re.compile(r"^\s*(WITH|SELECT)\b", re.IGNORECASE)


def parse_reply(text: str) -> ParsedReply:
    blocks = [b.strip() for b in _FENCED_SQL.findall(text) if b.strip()]
    if blocks:
        return ParsedReply(sql=blocks[-1])  # the last block is the final answer if the model iterated
    if m := _REFUSAL.search(text):
        return ParsedReply(refusal=m.group(1).strip().splitlines()[0] if m.group(1).strip() else "")
    if _BARE_SQL.match(text):
        return ParsedReply(sql=text.strip())
    return ParsedReply()
