"""Type definitions for the TZP Dify plugin shared library.

Adapted from rag_interceptor/models.py — LangChain-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class RetrievalTier(str, Enum):
    FALLBACK_STRONG = "fallback_strong"
    VECTOR_ONLY = "vector_only"


class ErrorMode(str, Enum):
    PLACEHOLDER = "placeholder"
    SILENT_SKIP = "silent_skip"
    SUMMARY = "summary"


@dataclass
class TrexConfig:
    api_base_url: str = "http://localhost:3000"
    api_key: str = ""
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("TREX_API_KEY", "")
        self.api_base_url = self.api_base_url.rstrip("/")


@dataclass
class RAGConfig:
    embedding_model: Literal["bge-m3", "voyage-3", "minilm"] = "minilm"
    top_k: int = 5
    voyage_api_key: str = ""
    minilm_model_name: str = "all-MiniLM-L6-v2"
    bge_m3_model_name: str = "BAAI/bge-m3"

    def __post_init__(self) -> None:
        if not self.voyage_api_key:
            self.voyage_api_key = os.environ.get("VOYAGE_API_KEY", "")


@dataclass
class ChunkConfig:
    chunk_size: int = 256
    chunk_overlap: int = 32
    strategy: Literal["payload_first", "recut"] = "payload_first"


@dataclass
class BudgetConfig:
    max_tzp_tags: int = 5
    max_total_chunks: int = 50
    max_total_fallback_chars: int = 50_000
    max_total_embedding_texts: int = 100
    max_inject_tokens: int = 500


@dataclass
class InterceptorConfig:
    trex: TrexConfig = field(default_factory=TrexConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    error_mode: ErrorMode = ErrorMode.PLACEHOLDER
    inject_with_source_label: bool = True


@dataclass
class QuantParams:
    min: float
    max: float
    method: str = "percentile_99_9_int8"


@dataclass
class TZPPayload:
    trex_id: str
    tzp_version: str
    vector_seq_b64: list[str]
    quant_params: list[QuantParams]
    dimensions: int
    chunk_count: int
    fallback_text_zstd_b64: str | None = None
    summary: str | None = None
    source_lang: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum_sha256: str = ""

    @property
    def has_fallback_text(self) -> bool:
        return bool(self.fallback_text_zstd_b64)

    @property
    def is_per_chunk_quant(self) -> bool:
        return len(self.quant_params) == self.chunk_count


class TrexPullError(Exception):
    def __init__(self, trex_id: str, error_code: str, detail: str = ""):
        self.trex_id = trex_id
        self.error_code = error_code
        self.detail = detail
        super().__init__(f"[TZP_ERROR: {trex_id}: {error_code}] {detail}")


@dataclass
class TagMatch:
    trex_id: str
    start: int
    end: int


@dataclass
class RetrievalResult:
    trex_id: str
    chunks: list[str] = field(default_factory=list)
    tier: RetrievalTier = RetrievalTier.VECTOR_ONLY
    embedding_model: str = ""
    chunk_scores: list[float] = field(default_factory=list)
    error: str | None = None
