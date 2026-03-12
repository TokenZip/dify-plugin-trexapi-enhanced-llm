"""Multi-level in-memory cache — adapted from rag_interceptor/cache.py."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

from .models import TZPPayload


class InterceptorCache:
    def __init__(self, max_entries: int = 128) -> None:
        self._max = max_entries
        self._store: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._lock = threading.Lock()

    def _get(self, level: str, key: str) -> Any | None:
        with self._lock:
            compound = (level, key)
            val = self._store.get(compound)
            if val is not None:
                self._store.move_to_end(compound)
                return val
            return None

    def _set(self, level: str, key: str, value: Any) -> None:
        with self._lock:
            compound = (level, key)
            if compound in self._store:
                self._store.move_to_end(compound)
            self._store[compound] = value
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def get_payload(self, trex_id: str) -> TZPPayload | None:
        return self._get("payload", trex_id)

    def set_payload(self, trex_id: str, payload: TZPPayload) -> None:
        self._set("payload", trex_id, payload)

    def get_text(self, trex_id: str) -> str | None:
        return self._get("text", trex_id)

    def set_text(self, trex_id: str, text: str) -> None:
        self._set("text", trex_id, text)

    def get_chunks(self, key: str) -> list[str] | None:
        return self._get("chunks", key)

    def set_chunks(self, key: str, chunks: list[str]) -> None:
        self._set("chunks", key, chunks)

    def get_index(self, key: str) -> tuple[Any, list[str]] | None:
        return self._get("index", key)

    def set_index(self, key: str, index: Any, chunks: list[str]) -> None:
        self._set("index", key, (index, chunks))

    @staticmethod
    def build_cache_key(
        trex_id: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> str:
        raw = f"{trex_id}:{embedding_model}:{chunk_size}:{chunk_overlap}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
