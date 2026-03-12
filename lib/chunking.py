"""Payload-first chunking strategy.

Adapted from rag_interceptor/chunking.py — no changes needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import ChunkConfig

if TYPE_CHECKING:
    from .models import TZPPayload

logger = logging.getLogger(__name__)


def get_chunks(
    payload: TZPPayload,
    text: str | None,
    config: ChunkConfig,
) -> list[str]:
    if text is None:
        return []
    if config.strategy == "payload_first":
        chunks = chunk_by_payload_boundary(text, payload.chunk_count)
        if chunks:
            return chunks
        logger.info("Payload-first chunking produced no results, falling back to recut")
    return chunk_by_tokens(text, config.chunk_size, config.chunk_overlap)


def chunk_by_payload_boundary(text: str, chunk_count: int) -> list[str]:
    if chunk_count <= 0 or not text.strip():
        return []
    words = text.split()
    if not words:
        return []
    if chunk_count == 1:
        return [text.strip()]
    words_per_chunk = max(1, len(words) // chunk_count)
    chunks: list[str] = []
    for i in range(chunk_count):
        start = i * words_per_chunk
        if i == chunk_count - 1:
            segment = " ".join(words[start:])
        else:
            segment = " ".join(words[start : start + words_per_chunk])
        if segment.strip():
            chunks.append(segment)
    return chunks


def chunk_by_tokens(
    text: str,
    chunk_size: int = 256,
    overlap: int = 32,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    words_per_chunk = max(1, int(chunk_size * 1.3))
    overlap_words = max(0, int(overlap * 1.3))
    step = max(1, words_per_chunk - overlap_words)
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + words_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks
