"""Request-level protections for the API: per-client rate limit and an answer cache."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict, deque
from typing import Any


class ClientRateLimiter:
    """Non-blocking sliding-window limit per client key. ``check`` returns seconds to wait (0 = ok)."""

    def __init__(self, per_minute: int, clock: Any = time.monotonic) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def check(self, key: str) -> float:
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return round(60 - (now - hits[0]), 1)
            hits.append(now)
            return 0.0


_WS = re.compile(r"\s+")


def normalize_question(q: str) -> str:
    return _WS.sub(" ", q).strip().rstrip("?.! ").lower()


class AnswerCache:
    """LRU + TTL cache of finished answers, keyed on the *exact* normalized question.

    Deliberately not a semantic (embedding-similarity) cache: "revenue in 2017" and "revenue in
    2018" embed almost identically, and serving one for the other is a silent wrong answer. Fuzzy
    reuse already happens safely one layer down, where the LLM response store keys on the full
    prompt.
    """

    def __init__(self, maxsize: int = 256, ttl_s: float = 600, clock: Any = time.monotonic) -> None:
        self.maxsize, self.ttl_s = maxsize, ttl_s
        self._data: OrderedDict[tuple[Any, ...], tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._clock = clock
        self.hits = self.misses = 0

    @staticmethod
    def key(question: str, **params: Any) -> tuple[Any, ...]:
        return (normalize_question(question), *sorted(params.items()))

    def get(self, key: tuple[Any, ...]) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or self._clock() - item[0] > self.ttl_s:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def put(self, key: tuple[Any, ...], value: Any) -> None:
        if self.maxsize <= 0 or self.ttl_s <= 0:
            return
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
