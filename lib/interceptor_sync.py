"""Synchronous TZP Interceptor — 6-phase pipeline.

Adapted from rag_interceptor/interceptor.py with all async
code converted to synchronous for Dify plugin compatibility.
"""

from __future__ import annotations

import logging
from typing import Any

from .cache import InterceptorCache
from .models import (
    ErrorMode,
    InterceptorConfig,
    RetrievalResult,
    RetrievalTier,
    TagMatch,
    TrexPullError,
    TZPPayload,
    BudgetConfig,
)
from .parser import extract_tzp_tags, replace_tzp_tags, strip_escaped_markers
from .retriever_lite import TZPRetrieverLite
from .trex_client_sync import TrexClientSync

logger = logging.getLogger(__name__)


class BudgetController:
    """Stateless guardrail that trims inputs to stay within budget."""

    def __init__(self, config: BudgetConfig) -> None:
        self._cfg = config

    def check_tag_limit(self, tags: list[TagMatch]) -> list[TagMatch]:
        limit = self._cfg.max_tzp_tags
        if len(tags) <= limit:
            return tags
        logger.warning("Tag budget exceeded: %d tags, keeping first %d", len(tags), limit)
        return tags[:limit]

    def truncate_injection(self, chunks: list[str]) -> list[str]:
        budget_words = int(self._cfg.max_inject_tokens * 0.75)
        result: list[str] = []
        total = 0
        for chunk in chunks:
            words = chunk.split()
            if total + len(words) > budget_words:
                remaining = budget_words - total
                if remaining > 0:
                    result.append(" ".join(words[:remaining]))
                break
            result.append(chunk)
            total += len(words)
        return result


class TZPInterceptorSync:
    """Synchronous interceptor that processes TZP markers in prompts."""

    def __init__(self, config: InterceptorConfig) -> None:
        self._config = config
        self._cache = InterceptorCache()
        self._client = TrexClientSync(config.trex, cache=self._cache)
        self._budget = BudgetController(config.budget)

    def process(self, prompt: str, user_query: str) -> str:
        """Full 6-phase pipeline: detect -> pull -> materialize -> retrieve -> inject -> failsafe."""
        prompt = strip_escaped_markers(prompt)
        tags = extract_tzp_tags(prompt)
        if not tags:
            return prompt

        tags = self._budget.check_tag_limit(tags)
        trex_ids = list(dict.fromkeys(t.trex_id for t in tags))
        logger.info("Phase 1 (Parse): %d tag(s) -> %s", len(tags), trex_ids)

        payloads: dict[str, TZPPayload] = {}
        errors: dict[str, str] = {}
        for trex_id in trex_ids:
            try:
                payloads[trex_id] = self._client.pull(trex_id)
            except TrexPullError as exc:
                errors[trex_id] = exc.error_code
        logger.info("Phase 2 (Pull): %d ok, %d errors", len(payloads), len(errors))

        retrieval_map: dict[str, RetrievalResult] = {}
        for trex_id, payload in payloads.items():
            try:
                rr = self._materialize_and_retrieve(payload, user_query)
                retrieval_map[trex_id] = rr
            except Exception as exc:
                logger.exception("Retrieval failed for %s", trex_id)
                errors[trex_id] = f"RETRIEVAL_FAILED: {exc}"

        replacements = self._build_replacements(retrieval_map, errors, tags)
        return replace_tzp_tags(prompt, tags, replacements)

    def _materialize_and_retrieve(
        self, payload: TZPPayload, user_query: str
    ) -> RetrievalResult:
        retriever = TZPRetrieverLite(
            payload=payload,
            rag_config=self._config.rag,
            chunk_config=self._config.chunk,
            cache=self._cache,
        )
        results = retriever.search(user_query)
        chunks = [r.content for r in results]
        scores = [r.score for r in results]
        tier_val = results[0].retrieval_tier if results else RetrievalTier.VECTOR_ONLY
        em = results[0].embedding_model if results else ""

        chunks = self._budget.truncate_injection(chunks)

        return RetrievalResult(
            trex_id=payload.trex_id,
            chunks=chunks,
            tier=tier_val if isinstance(tier_val, RetrievalTier) else RetrievalTier(tier_val),
            embedding_model=em,
            chunk_scores=scores[: len(chunks)],
        )

    def _build_replacements(
        self,
        retrieval_map: dict[str, RetrievalResult],
        errors: dict[str, str],
        tags: list[TagMatch],
    ) -> dict[str, str]:
        replacements: dict[str, str] = {}
        error_mode = self._config.error_mode
        with_label = self._config.inject_with_source_label
        seen_ids = {t.trex_id for t in tags}

        for trex_id in seen_ids:
            if trex_id in errors:
                replacements[trex_id] = self._format_error(trex_id, errors[trex_id], error_mode)
                continue
            rr = retrieval_map.get(trex_id)
            if rr is None or not rr.chunks:
                replacements[trex_id] = self._format_error(trex_id, "NO_RELEVANT_CONTENT", error_mode)
                continue
            replacements[trex_id] = self._format_context(trex_id, rr, with_label)
        return replacements

    @staticmethod
    def _format_context(trex_id: str, rr: RetrievalResult, with_label: bool) -> str:
        lines: list[str] = []
        if with_label:
            lines.append(f"[Context from {trex_id}]")
        for chunk in rr.chunks:
            lines.append(f"- {chunk}")
        return "\n".join(lines)

    @staticmethod
    def _format_error(trex_id: str, code: str, mode: ErrorMode) -> str:
        if mode == ErrorMode.PLACEHOLDER:
            return f"[TZP_ERROR: {trex_id}: {code}]"
        if mode == ErrorMode.SILENT_SKIP:
            return ""
        return f"(Context {trex_id} is temporarily unavailable)"

    @property
    def cache(self) -> InterceptorCache:
        return self._cache

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TZPInterceptorSync:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
