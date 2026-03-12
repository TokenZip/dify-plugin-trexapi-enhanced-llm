"""Embedding model wrappers — LangChain-free.

Supports three backends:
  - minilm:  local sentence-transformers, 384d (default, lightweight)
  - bge-m3:  local sentence-transformers, 1024d (strong)
  - voyage-3: Voyage AI API, 1024d (strong, API-based)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .models import RAGConfig


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        embedding = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return embedding[0].tolist()


class VoyageEmbeddings:
    def __init__(self, api_key: str, model: str = "voyage-3") -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _load(self):
        if self._client is None:
            import voyageai
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._load()
        result = client.embed(texts, model=self._model, input_type="document")
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        client = self._load()
        result = client.embed([text], model=self._model, input_type="query")
        return result.embeddings[0]


_MODEL_CACHE: dict[str, EmbeddingModel] = {}


def get_embedding_model(config: RAGConfig) -> EmbeddingModel:
    key = config.embedding_model
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if key == "bge-m3":
        model: EmbeddingModel = SentenceTransformerEmbeddings(config.bge_m3_model_name)
    elif key == "voyage-3":
        if not config.voyage_api_key:
            raise ValueError("voyage_api_key is required for voyage-3 embedding model")
        model = VoyageEmbeddings(api_key=config.voyage_api_key)
    elif key == "minilm":
        model = SentenceTransformerEmbeddings(config.minilm_model_name)
    else:
        raise ValueError(f"Unknown embedding model: {key}")

    _MODEL_CACHE[key] = model
    return model


def get_minilm_model(config: RAGConfig) -> EmbeddingModel:
    cache_key = "minilm"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    model: EmbeddingModel = SentenceTransformerEmbeddings(config.minilm_model_name)
    _MODEL_CACHE[cache_key] = model
    return model
