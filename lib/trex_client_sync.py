"""Synchronous HTTP client for TrexAPI — adapted from rag_interceptor/trex_client.py.

Replaces httpx.AsyncClient with httpx.Client for Dify plugin compatibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .cache import InterceptorCache
from .models import QuantParams, TrexConfig, TrexPullError, TZPPayload

logger = logging.getLogger(__name__)

TREX_NOT_FOUND = "TREX_NOT_FOUND"
TREX_EXPIRED = "TREX_EXPIRED"
TREX_FORBIDDEN = "TREX_FORBIDDEN"
TREX_TIMEOUT = "TREX_TIMEOUT"
TREX_SERVER_ERROR = "TREX_SERVER_ERROR"

_STATUS_TO_ERROR = {
    404: TREX_NOT_FOUND,
    410: TREX_EXPIRED,
    403: TREX_FORBIDDEN,
    401: TREX_FORBIDDEN,
}


def _parse_quant_params(raw: Any) -> list[QuantParams]:
    if isinstance(raw, dict):
        return [QuantParams(min=raw["min"], max=raw["max"],
                            method=raw.get("method", "percentile_99_9_int8"))]
    if isinstance(raw, list):
        return [
            QuantParams(min=p["min"], max=p["max"],
                        method=p.get("method", "percentile_99_9_int8"))
            for p in raw
        ]
    raise ValueError(f"Invalid quant_params format: {type(raw)}")


def _parse_payload(data: dict[str, Any]) -> TZPPayload:
    payload = data["payload"]
    metadata = data.get("metadata", {})
    qp = _parse_quant_params(payload["quant_params"])
    return TZPPayload(
        trex_id=data["trex_id"],
        tzp_version=data.get("tzp_version", "1.0"),
        vector_seq_b64=payload["vector_seq_b64"],
        quant_params=qp,
        dimensions=payload.get("dimensions", 384),
        chunk_count=payload.get("chunk_count", len(payload["vector_seq_b64"])),
        fallback_text_zstd_b64=payload.get("fallback_text_zstd_b64"),
        summary=payload.get("summary"),
        source_lang=payload.get("source_lang"),
        metadata=metadata,
        checksum_sha256=data.get("checksum_sha256", ""),
    )


class TrexClientSync:
    """Synchronous client for TrexAPI operations with optional caching."""

    def __init__(
        self,
        config: TrexConfig,
        cache: InterceptorCache | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"
            self._client = httpx.Client(
                base_url=self._config.api_base_url,
                headers=headers,
                timeout=self._config.timeout,
            )
        return self._client

    def pull(self, trex_id: str) -> TZPPayload:
        if self._cache:
            cached = self._cache.get_payload(trex_id)
            if cached is not None:
                return cached

        client = self._get_client()
        try:
            resp = client.get(f"/v1/payloads/{trex_id}")
        except httpx.TimeoutException:
            raise TrexPullError(trex_id, TREX_TIMEOUT, "Request timed out")
        except httpx.HTTPError as exc:
            raise TrexPullError(trex_id, TREX_SERVER_ERROR, str(exc))

        if resp.status_code != 200:
            error_code = _STATUS_TO_ERROR.get(resp.status_code, TREX_SERVER_ERROR)
            raise TrexPullError(trex_id, error_code, f"HTTP {resp.status_code}")

        payload = _parse_payload(resp.json())
        if self._cache:
            self._cache.set_payload(trex_id, payload)
        return payload

    def push(self, payload_json: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Push a TZP payload to TrexAPI. Returns the response body including trex_id."""
        client = self._get_client()
        body: dict[str, Any] = {"payload": payload_json}
        if metadata:
            body["metadata"] = metadata
        try:
            resp = client.post("/v1/payloads", json=body)
        except httpx.TimeoutException:
            raise TrexPullError("unknown", TREX_TIMEOUT, "Push request timed out")
        except httpx.HTTPError as exc:
            raise TrexPullError("unknown", TREX_SERVER_ERROR, str(exc))

        if resp.status_code not in (200, 201):
            raise TrexPullError(
                "unknown", TREX_SERVER_ERROR,
                f"Push failed: HTTP {resp.status_code} — {resp.text[:200]}"
            )
        return resp.json()

    def head(self, trex_id: str) -> dict[str, str]:
        """Probe payload status via HEAD request. Returns response headers."""
        client = self._get_client()
        try:
            resp = client.head(f"/v1/payloads/{trex_id}")
        except httpx.TimeoutException:
            raise TrexPullError(trex_id, TREX_TIMEOUT, "HEAD request timed out")
        except httpx.HTTPError as exc:
            raise TrexPullError(trex_id, TREX_SERVER_ERROR, str(exc))

        if resp.status_code != 200:
            error_code = _STATUS_TO_ERROR.get(resp.status_code, TREX_SERVER_ERROR)
            raise TrexPullError(trex_id, error_code, f"HTTP {resp.status_code}")
        return dict(resp.headers)

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()
            self._client = None

    def __enter__(self) -> TrexClientSync:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
