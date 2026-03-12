"""Lightweight TZP retriever — no LangChain dependency.

Adapted from rag_interceptor/retriever.py with the LangChain BaseRetriever
replaced by a plain class using numpy + faiss directly.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cache import InterceptorCache
from .chunking import get_chunks
from .dequantize import decompress_fallback_text, dequantize_payload
from .embeddings import EmbeddingModel, get_embedding_model, get_minilm_model
from .models import ChunkConfig, RAGConfig, RetrievalTier, TZPPayload

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    content: str
    score: float
    chunk_index: int
    retrieval_tier: str
    embedding_model: str
    trex_id: str


class TZPRetrieverLite:
    """Retriever that builds an on-demand FAISS index from a TZP payload."""

    def __init__(
        self,
        payload: TZPPayload,
        rag_config: RAGConfig,
        chunk_config: ChunkConfig,
        cache: InterceptorCache | None = None,
    ) -> None:
        self._payload = payload
        self._rag_config = rag_config
        self._chunk_config = chunk_config
        self._cache = cache
        self._chunks: list[str] | None = None
        self._faiss_index: Any = None
        self._embedding_model: EmbeddingModel | None = None
        self._embedding_model_name: str = ""
        self._tier: str = ""
        self._prepare()

    def _cache_key(self) -> str:
        em = self._rag_config.embedding_model if self._tier == RetrievalTier.FALLBACK_STRONG else "minilm"
        return InterceptorCache.build_cache_key(
            self._payload.trex_id, em,
            self._chunk_config.chunk_size, self._chunk_config.chunk_overlap,
        )

    def _prepare(self) -> None:
        payload = self._payload
        rag = self._rag_config

        if payload.has_fallback_text:
            self._tier = RetrievalTier.FALLBACK_STRONG
        else:
            self._tier = RetrievalTier.VECTOR_ONLY

        if self._cache is not None:
            cached = self._cache.get_index(self._cache_key())
            if cached is not None:
                self._faiss_index, self._chunks = cached
                self._embedding_model = (
                    get_embedding_model(rag)
                    if self._tier == RetrievalTier.FALLBACK_STRONG
                    else get_minilm_model(rag)
                )
                self._embedding_model_name = (
                    rag.embedding_model
                    if self._tier == RetrievalTier.FALLBACK_STRONG
                    else "minilm"
                )
                logger.info("RetrieverLite: cache hit for %s", payload.trex_id)
                return

        if self._tier == RetrievalTier.FALLBACK_STRONG:
            self._prepare_path_a(payload, rag)
        else:
            self._prepare_path_b(payload, rag)

        if self._cache is not None and self._faiss_index is not None:
            self._cache.set_index(self._cache_key(), self._faiss_index, self._chunks or [])

    def _prepare_path_a(self, payload: TZPPayload, rag: RAGConfig) -> None:
        logger.info("RetrieverLite: Path A (fallback_strong) — re-embedding with %s", rag.embedding_model)
        self._embedding_model_name = rag.embedding_model

        text: str | None = None
        if self._cache is not None:
            text = self._cache.get_text(payload.trex_id)
        if text is None:
            text = decompress_fallback_text(payload.fallback_text_zstd_b64)  # type: ignore[arg-type]
            if self._cache is not None:
                self._cache.set_text(payload.trex_id, text)

        self._chunks = get_chunks(payload, text, self._chunk_config)
        if not self._chunks:
            self._chunks = [text] if text.strip() else []

        self._embedding_model = get_embedding_model(rag)
        doc_embeddings = self._embedding_model.embed_documents(self._chunks)
        self._build_faiss_index(np.array(doc_embeddings, dtype=np.float32))

    def _prepare_path_b(self, payload: TZPPayload, rag: RAGConfig) -> None:
        logger.info("RetrieverLite: Path B (vector_only) — dequantized MiniLM vectors")
        self._embedding_model_name = "minilm"
        vectors = dequantize_payload(payload)
        self._chunks = [
            f"[Chunk {i}] (vector-only, no text available)"
            for i in range(len(vectors))
        ]
        self._embedding_model = get_minilm_model(rag)
        self._build_faiss_index(np.stack(vectors).astype(np.float32))

    def _build_faiss_index(self, matrix: np.ndarray) -> None:
        import faiss
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._faiss_index = index

    def search(self, query: str, top_k: int | None = None, score_threshold: float = 0.0) -> list[RetrievedChunk]:
        """Search the payload for chunks relevant to the query."""
        effective_top_k = min(top_k or self._rag_config.top_k, len(self._chunks or []))
        if effective_top_k == 0 or self._embedding_model is None:
            return []

        query_vec = np.array(
            self._embedding_model.embed_query(query), dtype=np.float32
        ).reshape(1, -1)

        import faiss
        faiss.normalize_L2(query_vec)
        scores, indices = self._faiss_index.search(query_vec, effective_top_k)

        results: list[RetrievedChunk] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            if float(score) < score_threshold:
                continue
            results.append(RetrievedChunk(
                content=self._chunks[idx],  # type: ignore[index]
                score=float(score),
                chunk_index=int(idx),
                retrieval_tier=self._tier,
                embedding_model=self._embedding_model_name,
                trex_id=self._payload.trex_id,
            ))
        return results
